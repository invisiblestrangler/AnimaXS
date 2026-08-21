import Foundation
import Metal

/// One iteration's weight-slot decision in the DiT block loop. Pure and
/// testable so the ping-pong ON/OFF seam is verified without a real pack.
struct DitLoopStep: Equatable {
    let slot: Int
    let prefetchIndex: Int?
    let prefetchSlot: Int
}

/// Streamed 28-block MiniTrainDIT transformer loop. Each block is awaited before
/// the next locator-derived range replaces the one-slot weight ring.
final class DitForward {
    private let block: DiTBlockExecutor
    private let finalLayer: DiTFinalLayerExecutor

    /// Pure loop-shape helper: with ping-pong ON (2 slots) block i executes
    /// from slot i%2 and prefetches block i+1 into the other slot. With
    /// ping-pong OFF (1 slot) every block uses slot 0 and NO prefetch is ever
    /// requested — the synchronous one-slot pattern.
    static func loopStep(logicalIndex: Int, blockCount: Int, pingPong: Bool) -> DitLoopStep {
        if pingPong {
            let slot = logicalIndex % 2
            let prefetch: (index: Int, slot: Int)?
            if logicalIndex + 1 < blockCount {
                prefetch = (logicalIndex + 1, (logicalIndex + 1) % 2)
            } else {
                prefetch = nil
            }
            return DitLoopStep(
                slot: slot, prefetchIndex: prefetch?.index, prefetchSlot: prefetch?.slot ?? 0)
        }
        return DitLoopStep(slot: 0, prefetchIndex: nil, prefetchSlot: 0)
    }

    /// Run telemetry collector; forwarded to the block and final-layer executors.
    var metrics: MetricsCollector? {
        didSet {
            block.metrics = metrics
            finalLayer.metrics = metrics
        }
    }

    init(context: MetalContext, file: AnimapkFile,
         attentionNumerics: AttentionNumerics = .legacy,
         activationNumerics: ActivationNumerics = .legacy,
         finalResidualBoundary: FinalResidualBoundary = .fp16Legacy,
         monitor: NumericalMonitor? = nil,
         optimization: InferenceOptimizationConfig = .currentBaseline,
         crossKVCache: CrossKVCache? = nil) throws {
        block = try DiTBlockExecutor(
            context: context, file: file, attentionNumerics: attentionNumerics,
            activationNumerics: activationNumerics, monitor: monitor,
            optimization: optimization, crossKVCache: crossKVCache)
        finalLayer = try DiTFinalLayerExecutor(
            context: context, file: file, activationNumerics: activationNumerics,
            finalResidualBoundary: finalResidualBoundary,
            monitor: monitor, optimization: optimization)
    }

    /// Configured once before a generation starts. The block owns the concrete
    /// generation-local GPU adapter buffers; the final layer is deliberately
    /// outside the v1 external-LoRA target set.
    func configureLoRA(_ selection: ResolvedLoRA?) throws {
        try block.configureLoRA(selection)
    }

    /// Mutates the tightly packed fp32 `[1024,2048]` residual in place.
    /// The optional callback runs after GPU completion and exists for diagnostics/tests;
    /// normal inference leaves it nil and performs no per-block CPU readback.
    ///
    /// Execution uses the bounded two-slot ping-pong weight streamer (Phase 12)
    /// when `optimization.pingPongWeightStreaming` is ON: block N's weights are
    /// copied into the other slot while block N-1 executes on the GPU, so CPU
    /// memcpy hides behind GPU work. Outputs are byte-identical to the one-slot
    /// pattern (same bytes at the same offsets in the slot).
    ///
    /// When ping-pong is OFF (the measurement experiment), a single weight slot
    /// is used with no look-ahead prefetch: each block synchronously loads into
    /// slot 0, executes, then the next block loads.
    func execute(
        residual: MTLBuffer,
        emb: MTLBuffer,
        adalnLora: MTLBuffer,
        crossContext: MTLBuffer,
        rope: MTLBuffer,
        blockCompleted: ((Int, MTLBuffer) throws -> Void)? = nil,
        diagnosticBranchFilter: ((Int) -> Bool)? = nil,
        diagnosticBranchCompleted: ((Int, String, MTLBuffer) throws -> Void)? = nil
    ) async throws {
        let blockCount = DiTBlockLocator.blockCount
        let pingPong = block.slotCount > 1
        try block.prefetch(blockIndex: 0, slot: 0)
        for logicalIndex in 0..<blockCount {
            let step = Self.loopStep(logicalIndex: logicalIndex, blockCount: blockCount, pingPong: pingPong)
            let branchCallback: DiTBlockExecutor.DiagnosticBranchCompleted?
            if diagnosticBranchFilter?(logicalIndex) ?? true,
               let diagnosticBranchCompleted {
                branchCallback = { branch, current in
                    try diagnosticBranchCompleted(logicalIndex, branch, current)
                }
            } else {
                branchCallback = nil
            }
            try await block.execute(
                blockIndex: logicalIndex, residual: residual, emb: emb,
                adalnLora: adalnLora, crossContext: crossContext, rope: rope,
                slot: step.slot, prefetchIndex: step.prefetchIndex,
                prefetchSlot: step.prefetchSlot,
                diagnosticBranchCompleted: branchCallback)
            try blockCompleted?(logicalIndex, residual)
        }
    }

    func executeVelocity(
        residual: MTLBuffer, emb: MTLBuffer, adalnLora: MTLBuffer,
        crossContext: MTLBuffer, rope: MTLBuffer, velocity: MTLBuffer
    ) async throws {
        try await execute(residual: residual, emb: emb, adalnLora: adalnLora,
                          crossContext: crossContext, rope: rope)
        try await finalLayer.execute(
            residual: residual, emb: emb, adalnLora: adalnLora, velocity: velocity)
    }

    /// Final projection for callers that have already executed the block loop.
    func executeVelocityFinalLayer(
        residual: MTLBuffer, emb: MTLBuffer, adalnLora: MTLBuffer,
        velocity: MTLBuffer
    ) async throws {
        try await finalLayer.execute(
            residual: residual, emb: emb, adalnLora: adalnLora, velocity: velocity)
    }
}
