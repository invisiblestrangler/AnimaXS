import Foundation
import Metal

/// GPU-side numerical-health monitoring (optimization runbook Phase 3).
///
/// A tiny shared stats buffer (one uint32 quad per probe slot) is written by
/// probe kernels via relaxed atomics. Probes are placed at every FP16
/// boundary that can turn a finite value unsafe, plus a handful of always-on
/// FP32 boundaries (velocity / denoised / Euler). The monitor accumulates
/// flags + max-abs magnitude for the whole run and records the FIRST
/// (step, block, probe) that ever went unsafe via cheap per-block reads of
/// the 512-byte stats buffer (the block's command buffer has already
/// completed when the read happens, so there is no extra GPU sync).
///
/// Slot order in `Probe` mirrors execution order so "first unsafe boundary"
/// attribution is deterministic.
final class NumericalMonitor {

    /// Probe slots, in execution order. Raw values are the stats-buffer slot
    /// indices and MUST stay in sync with the `*.metal` probe kernels' ABI.
    enum Probe: Int, CaseIterable {
        case crossContextToHalf = 0
        case patchesToHalf
        case selfProjectionInput
        case selfQToken
        case selfKToken
        case selfVToken
        case selfScores
        case selfAttended
        case selfBranch
        case selfGateAdd
        case selfResidual
        case crossProjectionInput
        case crossQToken
        case crossKToken
        case crossVToken
        case crossScores
        case crossAttended
        case crossBranch
        case crossGateAdd
        case crossResidual
        case mlpProjectionInput
        case mlpHiddenToHalf
        case mlpBranch
        case mlpGateAdd
        case mlpResidual
        case finalResidualToHalf
        case finalNormalizedToHalf
        case finalProjectionInput
        case finalProjected
        case velocity
        case denoised
        case eulerOutput

        /// Probes that require detailed (developer-flag) mode: they are
        /// standalone extra passes over MPS outputs or fp32 residual copies.
        /// Everything else is an in-kernel probe with ~zero added traffic.
        var requiresDetailedMode: Bool {
            switch self {
            case .selfQToken, .selfKToken, .selfVToken,
                 .selfAttended, .selfBranch, .selfResidual,
                 .crossQToken, .crossKToken, .crossVToken,
                 .crossAttended, .crossBranch, .crossResidual,
                 .mlpBranch, .mlpResidual,
                 .finalProjected:
                return true
            default:
                return false
            }
        }

        /// Human-readable stage label for failure attribution.
        var stageLabel: String {
            switch self {
            case .crossContextToHalf: return "cross-context conversion"
            case .patchesToHalf: return "patch embedding conversion"
            case .selfProjectionInput: return "self-attention projection input"
            case .selfQToken: return "self-attention Q projection"
            case .selfKToken: return "self-attention K projection"
            case .selfVToken: return "self-attention V projection"
            case .selfScores: return "self-attention scores"
            case .selfAttended: return "self-attention output"
            case .selfBranch: return "self-attention output projection"
            case .selfGateAdd: return "self-attention output"
            case .selfResidual: return "self-attention residual"
            case .crossProjectionInput: return "cross-attention projection input"
            case .crossQToken: return "cross-attention Q projection"
            case .crossKToken: return "cross-attention K projection"
            case .crossVToken: return "cross-attention V projection"
            case .crossScores: return "cross-attention scores"
            case .crossAttended: return "cross-attention output"
            case .crossBranch: return "cross-attention output projection"
            case .crossGateAdd: return "cross-attention output"
            case .crossResidual: return "cross-attention residual"
            case .mlpProjectionInput: return "MLP projection input"
            case .mlpHiddenToHalf: return "MLP hidden conversion"
            case .mlpBranch: return "MLP output projection"
            case .mlpGateAdd: return "MLP output"
            case .mlpResidual: return "MLP residual"
            case .finalResidualToHalf: return "final-layer residual conversion"
            case .finalNormalizedToHalf: return "final-layer norm conversion"
            case .finalProjectionInput: return "final-layer projection input"
            case .finalProjected: return "final-layer projection"
            case .velocity: return "velocity conversion"
            case .denoised: return "FLOW denoised conversion"
            case .eulerOutput: return "Euler update"
            }
        }
    }

    struct Stats: Equatable {
        static let noIssue = Stats()
        var flags: UInt32 = 0
        var maxAbs: Float = 0
        var firstIndex: UInt32 = .max

        var hasIssue: Bool { flags != 0 }
        var hasNaN: Bool { flags & Flag.nan.rawValue != 0 }
        var hasPosInf: Bool { flags & Flag.posInf.rawValue != 0 }
        var hasNegInf: Bool { flags & Flag.negInf.rawValue != 0 }
        var hasHalfOverflow: Bool { flags & Flag.halfOverflow.rawValue != 0 }
        var resultNaN: Bool { flags & Flag.resultNaN.rawValue != 0 }
        var resultInf: Bool { flags & Flag.resultInf.rawValue != 0 }

        /// Short human condition, e.g. "Inf detected" / "NaN detected".
        var condition: String {
            if hasNaN { return "NaN detected" }
            if hasPosInf { return "positive Inf detected" }
            if hasNegInf { return "negative Inf detected" }
            if resultNaN { return "residual became NaN" }
            if resultInf { return "residual became non-finite" }
            if hasHalfOverflow { return "value exceeded FP16 finite range" }
            return "non-finite value detected"
        }
    }

    /// Flag bit meanings (must match `*.metal` constants).
    enum Flag: UInt32 {
        case nan = 1
        case posInf = 2
        case negInf = 4
        case halfOverflow = 8
        case resultNaN = 16
        case resultInf = 32
    }

    /// The first unsafe event attributed to (step, block, probe).
    struct FirstIssue: Equatable {
        let step: Int          // 1-based
        let block: Int?        // 1-based
        let probe: Probe
        let stats: Stats
    }

    /// Detailed standalone probe passes cost extra kernel launches and full
    /// buffer reads; they default OFF in production (the always-on in-kernel
    /// probes already attribute the failure origin). Enable for stress runs.
    static var detailedProbesEnabled = false

    private let context: MetalContext
    private let statsBuffer: MTLBuffer
    private let slotCount: Int
    private var previousFlags: [UInt32]
    private var firstIssue: FirstIssue?
    private var nextBlockAttribution: (step: Int, block: Int) = (0, 0)

    init(context: MetalContext) throws {
        self.context = context
        self.slotCount = Probe.allCases.count
        guard let buffer = context.device.makeBuffer(
            length: slotCount * 16, options: .storageModeShared) else {
            throw AnimapkError.validation("failed to allocate numerical monitor stats")
        }
        self.statsBuffer = buffer
        self.previousFlags = [UInt32](repeating: 0, count: slotCount)
    }

    /// Zero the stats buffer and clear attribution state for a new run.
    func beginRun() {
        memset(statsBuffer.contents(), 0, statsBuffer.length)
        previousFlags = [UInt32](repeating: 0, count: slotCount)
        firstIssue = nil
    }

    /// The first unsafe (step, block, probe) observed so far, if any.
    var earliestIssue: FirstIssue? { firstIssue }

    /// Full report of every slot's accumulated stats (magnitude evidence).
    func report() -> [Probe: Stats] {
        let raw = readRaw()
        var result: [Probe: Stats] = [:]
        for probe in Probe.allCases {
            result[probe] = stats(at: probe.rawValue, raw: raw)
        }
        return result
    }

    // MARK: - Per-block / per-step attribution checkpoints (cheap CPU reads)

    /// Must be called after each DiT block's command buffer completes.
    func noteBlockCompleted(step: Int, block: Int) {
        nextBlockAttribution = (step, block)
        recordNewFlags(step: step, block: block)
    }

    /// Must be called after the step's velocity/denoised/Euler probes
    /// complete, before the CPU finite check.
    func noteStepCompleted(step: Int) {
        recordNewFlags(step: step, block: nil)
    }

    private func recordNewFlags(step: Int, block: Int?) {
        guard firstIssue == nil else { return }
        let raw = readRaw()
        var index = 0
        for probe in Probe.allCases {
            let flags = raw[index * 4]
            let delta = flags & ~previousFlags[index]
            if delta != 0 {
                firstIssue = FirstIssue(
                    step: step + 1, block: block.map { $0 + 1 },
                    probe: probe, stats: stats(at: index, raw: raw))
                break
            }
            index += 1
        }
        // Advance previousFlags only where no issue was found so a later read
        // can still delta against the true baseline.
        if firstIssue == nil {
            index = 0
            for probe in Probe.allCases {
                previousFlags[index] = raw[index * 4]
                index += 1
            }
        }
    }

    // MARK: - Probe encoding helpers

    /// Bind the stats buffer + probe slot for an in-kernel probe.
    /// `statsIndex`/`slotIndex` are the kernel's buffer indices for the stats
    /// array and probe slot constant (they differ per kernel, see the .metal
    /// probe ABI): float_to_half_probe = (3, 4); gate_add_half_f32_probe and
    /// the softmax probes = (5, 6).
    func bindProbe(
        _ encoder: MTLComputeCommandEncoder, probe: Probe,
        statsIndex: Int, slotIndex: Int
    ) {
        var slot = UInt32(probe.rawValue)
        encoder.setBuffer(statsBuffer, offset: 0, index: statsIndex)
        encoder.setBytes(&slot, length: 4, index: slotIndex)
    }

    /// Encode a standalone stats pass over an fp16 buffer (MPS outputs).
    func encodeProbe(
        _ command: MTLCommandBuffer, values: MTLBuffer, count: Int, probe: Probe
    ) throws {
        guard count > 0 else { return }
        let pipeline = try context.pipeline(named: "probe_f16_stats")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create fp16 probe encoder")
        }
        var slot = UInt32(probe.rawValue)
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(statsBuffer, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        encoder.setBytes(&slot, length: 4, index: 3)
        dispatchProbe(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    /// Encode a standalone stats pass over an fp32 buffer.
    func encodeProbeF32(
        _ command: MTLCommandBuffer, values: MTLBuffer, count: Int, probe: Probe
    ) throws {
        guard count > 0 else { return }
        let pipeline = try context.pipeline(named: "probe_f32_stats")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create fp32 probe encoder")
        }
        var slot = UInt32(probe.rawValue)
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(statsBuffer, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        encoder.setBytes(&slot, length: 4, index: 3)
        dispatchProbe(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func dispatchProbe(
        _ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int
    ) {
        let groups = (count + 255) / 256
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Readback

    /// Test-only access to the shared stats buffer bytes (crafting/verifying
    /// probe results without going through kernels).
    var statsBufferForTesting: UnsafeMutableRawPointer { statsBuffer.contents() }

    private func readRaw() -> [UInt32] {
        let pointer = statsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        return Array(UnsafeBufferPointer(start: pointer, count: slotCount * 4))
    }

    private func stats(at slot: Int, raw: [UInt32]) -> Stats {
        let base = slot * 4
        var stats = Stats()
        stats.flags = raw[base]
        stats.maxAbs = Float(bitPattern: raw[base + 1])
        stats.firstIndex = raw[base + 2]
        return stats
    }
}
