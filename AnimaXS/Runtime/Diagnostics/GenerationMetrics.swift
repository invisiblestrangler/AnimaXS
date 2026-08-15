import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// A start/end environment fact recorded around a generation (unplugged is
/// authoritative for benchmarking). Values are observational telemetry — they
/// never gate or throttle generation.
struct EnvironmentSnapshot: Equatable {
    let powerState: String
    let batteryLevel: Int // 0...100 percent, -1 when unknown
    let thermalState: String
    let lowPowerMode: Bool

    static let unknown = EnvironmentSnapshot(
        powerState: "n/a", batteryLevel: -1, thermalState: "n/a", lowPowerMode: false)

    var batteryText: String {
        batteryLevel < 0 ? "n/a" : "\(batteryLevel)%"
    }
}

/// One diffusion step's telemetry (optimization runbook Phase 7 / P2).
///
/// All values are accumulated by `MetricsCollector` from the SAME record calls
/// that feed the global counters, so per-step sums reconcile with the global
/// totals (P2 gate). `completed == false` marks a step that threw mid-way
/// (e.g. the W8 failure case): its partial durations/counters are still
/// published so a device log can attribute the slowdown to the failing step.
struct DiffusionStepMetrics: Equatable {
    let step: Int
    var completed: Bool = false
    var wallSeconds: Double = 0
    var blockWallSeconds: Double = 0
    var gpuCommandSeconds: Double = 0
    var weightCopySeconds: Double = 0
    var weightCopyBytes: UInt64 = 0
    var metalEncodeSeconds: Double = 0
    var hostWaitSeconds: Double = 0
    var minimumAvailableProcessMemory: UInt64 = .max
    var peakMetalAllocated: UInt64 = 0
    var linearGEMMTiles: Int = 0
    var attentionQueryTiles: Int = 0
    var dequantizedWeightBytesWritten: UInt64 = 0
    var transposeBytes: UInt64 = 0
    var conversionBytes: UInt64 = 0
    /// P3: activation traffic eliminated by the fused paths (norm/modulated
    /// fp32 intermediates, hidden fp32 GELU intermediate). Recorded on the
    /// fused path only; legacy path records 0.
    var fusedTrafficSavedBytes: UInt64 = 0
    var crossKVHits: Int = 0
    var crossKVMisses: Int = 0
    var mmapNoCopyBytes: UInt64 = 0
    var qgemmCalls: Int = 0
    // Observational per-step environment facts (cheap OS queries only —
    // never gate or throttle generation).
    var thermalState: String = "n/a"
    var availableProcessMemory: UInt64 = 0
    var metalAllocated: UInt64 = 0
}

/// Timing + memory telemetry for one generation run (optimization runbook
/// Phase 7). The unplugged device is the authoritative benchmark; this type
/// exists so the app itself can produce it without a cable.
///
/// All wall times are `ProcessInfo.systemUptime` seconds (monotonic). GPU
/// command time comes from `MTLCommandBuffer.gpuStartTime/gpuEndTime` where
/// populated; "host wait" is the residual of block wall time after copy +
/// encode + GPU command time, so the counters reconcile with wall time.
struct GenerationMetrics: Equatable {
    // End-to-end / stages (seconds)
    var totalWall: Double = 0
    var textEncode: Double = 0
    var adapter: Double = 0
    var diffusion: Double = 0
    var vae: Double = 0
    var other: Double = 0

    // Diffusion (seconds)
    var stepTimes: [Double] = []
    var blockTimes: [Double] = []
    var blockCount: Int = 0

    /// Per-step telemetry (P2). Append-only: one entry per executed step, in
    /// step order, INCLUDING partial (uncompleted) steps recorded when a step
    /// throws mid-way. Per-step sums reconcile with the global counters.
    var stepMetrics: [DiffusionStepMetrics] = []

    // Logical traffic counters (arithmetic byte counters, not GPU readbacks).
    // Counted once ("bytes materialized") — consistent across all sites.
    var dequantizedWeightBytesWritten: UInt64 = 0
    var transposeBytes: UInt64 = 0
    var conversionBytes: UInt64 = 0
    /// P3: activation traffic eliminated by the fused paths (fused
    /// LayerNorm+AdaLN+to-half, in-place half GELU). See
    /// `DiffusionStepMetrics.fusedTrafficSavedBytes`.
    var fusedTrafficSavedBytes: UInt64 = 0
    /// P5: cross-attention K/V cache hit/miss counts across the whole run.
    /// First executed step = 28 misses (one per DiT block); each later step
    /// = 28 hits. See `DiffusionStepMetrics.crossKVHits/crossKVMisses`.
    var crossKVHits: Int = 0
    var crossKVMisses: Int = 0

    // Weight streaming + Metal accounting (seconds)
    var weightCopyTime: Double = 0
    var weightCopyBytes: Int64 = 0
    var encodeTime: Double = 0
    var gpuCommandTime: Double = 0
    var hostWaitTime: Double = 0

    // Memory / thermal
    var peakMetalAllocation: UInt64 = 0
    var currentMetalAllocation: UInt64 = 0
    var minAvailableMemory: UInt64 = UInt64.max
    var thermalState: String = "n/a"

    // Numerical health
    var numericalWarnings: Int = 0
    /// Per-probe detail, e.g. "self-attention scores: Inf detected; MLP output: NaN detected".
    var numericalDetails: String = ""
    /// True when the monitor was DISABLED for this run (warnings were not
    /// collected). The final finite guard stays on regardless.
    var numericalMonitoringDisabled: Bool = false

    // Immutable per-run configuration snapshot (captured at Generate time).
    var optimizationConfig: InferenceOptimizationConfig?

    /// Filename of the DiT pack actually used by this run (e.g.
    /// `anima-turbo-v1.0-xsmax-w4.animapk` or the W8-v2 pack), so the summary
    /// reports which imported variant ran even though it is not a config
    /// choice.
    var ditPackFilename: String?
    /// Variant id of the DiT pack ("w4" or "w8-v2"), so telemetry visibly
    /// reports which variant ran even though the app-owned local file is
    /// always named after the W4 slot.
    var ditPackVariantID: String?
    /// SHA-256 of the DiT pack actually used by this run.
    var ditPackSHA256: String?
    /// Size in bytes of the DiT pack actually used by this run.
    var ditPackBytes: UInt64 = 0

    /// Why the run was cancelled, when it was cancelled. Telemetry only.
    var cancellationReason: GenerationCancellationReason?

    // Cheap executor counters (simple integer increments; no GPU readbacks).
    var linearGEMMTiles: Int = 0
    var linearDirectInputTiles: Int = 0
    var linearCopiedInputTiles: Int = 0
    var linearDirectOutputTiles: Int = 0
    var linearCopiedOutputTiles: Int = 0
    var attentionQueryTiles: Int = 0

    // Environment start/end (observational telemetry).
    var environmentStart: EnvironmentSnapshot = .unknown
    var environmentEnd: EnvironmentSnapshot = .unknown

    var averageBlockWall: Double {
        guard !blockTimes.isEmpty else { return 0 }
        return blockTimes.reduce(0, +) / Double(blockTimes.count)
    }

    /// Whether checkpointing was enabled for this run (always on: the DiT
    /// slot holds one verified pack, so there is no separate experimental
    /// state a checkpoint could collide with).
    var checkpointingEnabled: Bool {
        optimizationConfig?.checkpointingEnabled ?? true
    }

    /// Compact text report (also the in-app post-generation summary shape).
    var summaryText: String {
        var lines: [String] = []
        lines.append(String(format: "Generation: %.1f s", totalWall))
        if let reason = cancellationReason {
            lines.append("Cancellation: \(reason.rawValue)")
        }
        lines.append(String(format: "Text encode: %.1f s", textEncode))
        lines.append(String(format: "Adapter: %.1f s", adapter))
        lines.append(String(format: "DiT: %.1f s", diffusion))
        lines.append(String(format: "VAE: %.1f s", vae))
        lines.append(String(format: "Other: %.1f s", other))
        lines.append("")
        lines.append(String(format: "Average block wall time: %.2f s", averageBlockWall))
        lines.append(String(format: "Measured GPU command time: %.1f s", gpuCommandTime))
        lines.append(String(format: "Metal encode time: %.1f s", encodeTime))
        lines.append(String(format: "Weight copy/load CPU work: %.1f s, %.0f MB",
                            weightCopyTime, Double(weightCopyBytes) / 1_048_576))
        lines.append("  (may overlap GPU time when ping-pong is on)")
        if mmapNoCopyBytes > 0 {
            lines.append(String(format: "Weight bytes served mmap no-copy: %.0f MB (memcpy eliminated)",
                                Double(mmapNoCopyBytes) / 1_048_576))
        }
        lines.append(String(format: "Host/other measured time: %.1f s", hostWaitTime))
        lines.append("")
        if linearGEMMTiles > 0 || attentionQueryTiles > 0 {
            lines.append("Linear GEMM tiles: \(linearGEMMTiles)")
            lines.append("Linear input tiles: \(linearDirectInputTiles) direct / \(linearCopiedInputTiles) copied")
            lines.append("Linear output tiles: \(linearDirectOutputTiles) direct / \(linearCopiedOutputTiles) copied")
            lines.append("Attention query tiles: \(attentionQueryTiles)")
            lines.append("")
        }
        if !stepMetrics.isEmpty {
            lines.append("Per-step")
            lines.append("step  done  wall    block   gpu     copy    encode  host    avail-min  thermal")
            for entry in stepMetrics {
                let availableText = entry.minimumAvailableProcessMemory == .max
                    ? "n/a" : String(format: "%.0f MB", Double(entry.minimumAvailableProcessMemory) / 1_048_576)
                lines.append(String(
                    format: "%-5d %-5@ %-7.1f %-7.1f %-7.1f %-7.1f %-7.1f %-7.1f %-10@ %@",
                    entry.step,
                    entry.completed ? "yes" : "no",
                    entry.wallSeconds,
                    entry.blockWallSeconds,
                    entry.gpuCommandSeconds,
                    entry.weightCopySeconds,
                    entry.metalEncodeSeconds,
                    entry.hostWaitSeconds,
                    availableText,
                    entry.thermalState))
            }
            lines.append("")
            lines.append("Traffic/backend")
            lines.append(String(format: "dequantized weight bytes written: %.0f MB",
                                Double(dequantizedWeightBytesWritten) / 1_048_576))
            lines.append(String(format: "transpose bytes: %.0f MB",
                                Double(transposeBytes) / 1_048_576))
            lines.append(String(format: "conversion bytes: %.0f MB",
                                Double(conversionBytes) / 1_048_576))
            if fusedTrafficSavedBytes > 0 {
                lines.append(String(format: "fused activation traffic saved: %.0f MB",
                                    Double(fusedTrafficSavedBytes) / 1_048_576))
            }
            if crossKVHits > 0 || crossKVMisses > 0 {
                lines.append("cross-KV cache hits/misses: \(crossKVHits)/\(crossKVMisses)")
            }
            let noCopySum = stepMetrics.reduce(UInt64(0)) { $0 + $1.mmapNoCopyBytes }
            if noCopySum > 0 {
                lines.append(String(format: "weight bytes mmap no-copy: %.0f MB (memcpy eliminated)",
                                    Double(noCopySum) / 1_048_576))
            }
            lines.append("")
        }
        lines.append(String(format: "Peak Metal allocation: %.2f GB",
                            Double(peakMetalAllocation) / 1_073_741_824))
        if minAvailableMemory != UInt64.max {
            lines.append(String(format: "Minimum available process memory: %.0f MB (telemetry, not a budget)",
                                Double(minAvailableMemory) / 1_048_576))
        }
        if numericalMonitoringDisabled {
            lines.append("Numerical monitor: off (Euler finite guard on)")
            lines.append("Numerical warnings: not collected")
        } else {
            lines.append("Numerical monitor: on")
            lines.append("Numerical warnings: \(numericalWarnings)"
                + (numericalDetails.isEmpty ? "" : " (\(numericalDetails))"))
        }
        lines.append("")
        lines.append("Inference configuration")
        lines.append("DiT pack: \(ditPackFilename ?? "unknown")"
            + (ditPackVariantID.map { " (\($0))" } ?? "")
            + (ditPackSHA256.map { " \($0.prefix(12))…" } ?? "")
            + (ditPackBytes > 0 ? " \(ditPackBytes) bytes" : ""))
        if let config = optimizationConfig {
            lines.append("Linear tile rows: \(config.linearTileRows)")
            lines.append("Attention tile rows: \(config.attentionTileRows)")
            lines.append("Direct MPS linear I/O: \(config.directLinearMPSIO ? "on" : "off")")
            lines.append("Ping-pong weight streaming: \(config.pingPongWeightStreaming ? "on" : "off")")
            lines.append("Numerical monitor: \(config.numericalMonitoring ? "on" : "off")")
            lines.append("Mmap no-copy weight source: \(config.noCopyWeightSource ? "on" : "off")")
        }
        lines.append("Checkpointing: \(checkpointingEnabled ? "on" : "off")")
        lines.append("")
        lines.append("Environment")
        lines.append("Power: \(environmentStart.powerState) -> \(environmentEnd.powerState)")
        lines.append("Battery: \(environmentStart.batteryText) -> \(environmentEnd.batteryText)")
        lines.append("Thermal: \(environmentStart.thermalState) -> \(environmentEnd.thermalState)")
        lines.append("Low Power Mode: \(environmentStart.lowPowerMode ? "on" : "off") -> \(environmentEnd.lowPowerMode ? "on" : "off")")
        return lines.joined(separator: "\n")
    }
}

/// Reference-type accumulator for one run's metrics. Thread-confined to the
/// engine's serial executor (created per generation in `generate`).
final class MetricsCollector {
    enum Stage {
        case textEncode, adapter, diffusion, vae, tokenizing
    }

    private(set) var metrics = GenerationMetrics()
    private var stageStart: [Stage: Double] = [:]
    private var stepStartTime: Double?
    private var blockStartTime: Double?
    private(set) var currentStep = -1
    private(set) var currentBlock = -1
    /// Index into `metrics.stepMetrics` of the ACTIVE step, or nil when no
    /// diffusion step is in progress. Every record method also accumulates
    /// into this step so per-step totals reconcile with the globals (P2).
    private var activeStepIndex: Int?

    private func now() -> Double { ProcessInfo.processInfo.systemUptime }

    // MARK: - Stages

    func beginStage(_ stage: Stage) {
        stageStart[stage] = now()
    }

    func endStage(_ stage: Stage) {
        guard let start = stageStart.removeValue(forKey: stage) else { return }
        let elapsed = now() - start
        switch stage {
        case .textEncode: metrics.textEncode += elapsed
        case .adapter: metrics.adapter += elapsed
        case .diffusion: metrics.diffusion += elapsed
        case .vae: metrics.vae += elapsed
        case .tokenizing: break // folded into "other"
        }
    }

    // MARK: - Diffusion steps / blocks

    func beginStep(_ step: Int) {
        currentStep = step
        stepStartTime = now()
        metrics.stepMetrics.append(DiffusionStepMetrics(step: step))
        activeStepIndex = metrics.stepMetrics.count - 1
        // Observational per-step environment snapshot at step START (cheap OS
        // queries; never gates inference). Memory facts refresh again at
        // endStep, where the step's blocks have completed.
        recordEnvironmentIntoActiveStep()
    }

    /// Finalizes the active step. `completed` is false when the step threw
    /// mid-way — its partial durations/counters are still published so a
    /// device log can attribute a slowdown to the failing step (P2-B).
    func endStep(completed: Bool = true) {
        guard let start = stepStartTime else { return }
        let wall = now() - start
        metrics.stepTimes.append(wall)
        stepStartTime = nil
        guard let index = activeStepIndex, metrics.stepMetrics.indices.contains(index) else {
            activeStepIndex = nil
            return
        }
        metrics.stepMetrics[index].completed = completed
        metrics.stepMetrics[index].wallSeconds = wall
        if wall > 0 {
            metrics.stepMetrics[index].blockWallSeconds = max(0, wall - metrics.stepMetrics[index].weightCopySeconds)
        }
        recordEnvironmentIntoActiveStep()
        activeStepIndex = nil
    }

    private func recordEnvironmentIntoActiveStep() {
        guard let index = activeStepIndex, metrics.stepMetrics.indices.contains(index) else { return }
        metrics.stepMetrics[index].thermalState = String(describing: ProcessInfo.processInfo.thermalState)
        let available = UInt64(os_proc_available_memory())
        metrics.stepMetrics[index].availableProcessMemory = available
        metrics.stepMetrics[index].metalAllocated = metrics.currentMetalAllocation
    }

    func beginBlock(_ block: Int) {
        currentBlock = block
        blockStartTime = now()
    }

    func endBlock() {
        guard let start = blockStartTime else { return }
        metrics.blockTimes.append(now() - start)
        metrics.blockCount += 1
        blockStartTime = nil
    }

    // MARK: - Metal accounting

    func recordWeightCopy(bytes: Int, seconds: Double) {
        metrics.weightCopyTime += seconds
        metrics.weightCopyBytes += Int64(bytes)
        if let index = activeStepIndex {
            metrics.stepMetrics[index].weightCopySeconds += seconds
            metrics.stepMetrics[index].weightCopyBytes += UInt64(max(0, bytes))
        }
    }

    func recordEncode(seconds: Double) {
        metrics.encodeTime += seconds
        if let index = activeStepIndex {
            metrics.stepMetrics[index].metalEncodeSeconds += seconds
        }
    }

    func recordGPUCommand(seconds: Double) {
        metrics.gpuCommandTime += seconds
        if let index = activeStepIndex {
            metrics.stepMetrics[index].gpuCommandSeconds += seconds
        }
    }

    func recordHostWait(seconds: Double) {
        metrics.hostWaitTime += max(0, seconds)
        if let index = activeStepIndex {
            metrics.stepMetrics[index].hostWaitSeconds += max(0, seconds)
        }
    }

    // MARK: - Logical traffic counters (P2-C)

    /// Full-weight dequantization materialized bytes (counted once).
    func recordDequantizedWeightBytesWritten(_ bytes: UInt64) {
        metrics.dequantizedWeightBytesWritten += bytes
        if let index = activeStepIndex {
            metrics.stepMetrics[index].dequantizedWeightBytesWritten += bytes
        }
    }

    /// Transpose traffic (counted once — "bytes materialized", consistent
    /// with conversions).
    func recordTransposeBytes(_ bytes: UInt64) {
        metrics.transposeBytes += bytes
        if let index = activeStepIndex {
            metrics.stepMetrics[index].transposeBytes += bytes
        }
    }

    /// FP16/FP32 conversion traffic (counted once — "bytes materialized").
    func recordConversionBytes(_ bytes: UInt64) {
        metrics.conversionBytes += bytes
        if let index = activeStepIndex {
            metrics.stepMetrics[index].conversionBytes += bytes
        }
    }

    /// P3: activation traffic ELIMINATED by a fused path (e.g. the fp32
    /// norm/modulated intermediates or the fp32 MLP GELU intermediate that the
    /// fused kernel no longer materializes). Recorded on the fused path only;
    /// the legacy path records 0. Proves an optimization removed traffic even
    /// when thermal conditions make raw wall time noisy.
    func recordFusedTrafficSaved(_ bytes: UInt64) {
        metrics.fusedTrafficSavedBytes += bytes
        if let index = activeStepIndex {
            metrics.stepMetrics[index].fusedTrafficSavedBytes += bytes
        }
    }

    /// P5: a cross-attention K/V cache HIT (the invariant K/V for this block
    /// was reused instead of being re-projected). Recorded on the cache path.
    func recordCrossKVHit() {
        metrics.crossKVHits += 1
        if let index = activeStepIndex {
            metrics.stepMetrics[index].crossKVHits += 1
        }
    }

    /// P6: weight bytes served directly from the mmap'd pack via a
    /// `bytesNoCopy` MTLBuffer instead of a CPU memcpy into the slot ring.
    /// Recorded on the no-copy path only (the copied path records 0 and keeps
    /// charging `recordWeightCopy`), so device logs can prove the memcpy was
    /// eliminated for page-aligned ranges.
    func recordMmapNoCopyBytes(_ bytes: UInt64) {
        metrics.mmapNoCopyBytes += bytes
        if let index = activeStepIndex {
            metrics.stepMetrics[index].mmapNoCopyBytes += bytes
        }
    }

    /// P5: a cross-attention K/V cache MISS (the invariant K/V for this block
    /// was projected and stored for reuse on later steps). Recorded on the
    /// cache path.
    func recordCrossKVMiss() {
        metrics.crossKVMisses += 1
        if let index = activeStepIndex {
            metrics.stepMetrics[index].crossKVMisses += 1
        }
    }

    // MARK: - Memory / thermal

    func recordMemory(allocated: UInt64, available: UInt64) {
        metrics.peakMetalAllocation = max(metrics.peakMetalAllocation, allocated)
        metrics.currentMetalAllocation = allocated
        if available < metrics.minAvailableMemory { metrics.minAvailableMemory = available }
        if let index = activeStepIndex {
            metrics.stepMetrics[index].peakMetalAllocated = max(metrics.stepMetrics[index].peakMetalAllocated, allocated)
            if available < metrics.stepMetrics[index].minimumAvailableProcessMemory {
                metrics.stepMetrics[index].minimumAvailableProcessMemory = available
            }
        }
    }

    func recordThermal(_ state: ProcessInfo.ThermalState) {
        metrics.thermalState = String(describing: state)
    }

    func setNumericalWarnings(_ count: Int) {
        metrics.numericalWarnings = count
    }

    func setNumericalDetails(_ text: String) {
        metrics.numericalDetails = text
    }

    func setNumericalMonitoringDisabled(_ disabled: Bool) {
        metrics.numericalMonitoringDisabled = disabled
    }

    /// Records the immutable config snapshot actually used by the run.
    func recordOptimizationConfig(_ config: InferenceOptimizationConfig) {
        metrics.optimizationConfig = config
        metrics.numericalMonitoringDisabled = !config.numericalMonitoring
    }

    /// Records which DiT pack variant this run actually used, including the
    /// variant id ("w4" or "w8-v2"), display filename, SHA-256, and byte size.
    /// This visibly reports W8-v2 when W8 is installed even though the
    /// app-owned local file is named like the W4 slot.
    func recordDiTPackIdentity(id: String, filename: String, sha256: String, bytes: UInt64) {
        metrics.ditPackFilename = filename
        metrics.ditPackVariantID = id
        metrics.ditPackSHA256 = sha256
        metrics.ditPackBytes = bytes
    }

    /// Records why a generation run was cancelled (user / background /
    /// memory-warning / …). Telemetry only.
    func recordCancellationReason(_ reason: GenerationCancellationReason) {
        metrics.cancellationReason = reason
    }

    func recordEnvironmentStart(_ snapshot: EnvironmentSnapshot) {
        metrics.environmentStart = snapshot
    }

    func recordEnvironmentEnd(_ snapshot: EnvironmentSnapshot) {
        metrics.environmentEnd = snapshot
    }

    // MARK: - Executor tile counters

    func recordLinearGEMMTile(directInput: Bool, directOutput: Bool) {
        metrics.linearGEMMTiles += 1
        if directInput { metrics.linearDirectInputTiles += 1 } else { metrics.linearCopiedInputTiles += 1 }
        if directOutput { metrics.linearDirectOutputTiles += 1 } else { metrics.linearCopiedOutputTiles += 1 }
    }

    func recordAttentionQueryTile() {
        metrics.attentionQueryTiles += 1
    }

    func finalize(totalWall: Double) {
        metrics.totalWall = totalWall
        let accounted = metrics.textEncode + metrics.adapter + metrics.diffusion + metrics.vae
        metrics.other = max(0, totalWall - accounted)
    }

    func snapshot() -> GenerationMetrics { metrics }
}
