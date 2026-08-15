import Foundation
import Metal

/// Streamed 28-block MiniTrainDIT transformer loop. Each block is awaited before
/// the next locator-derived range replaces the one-slot weight ring.
final class DitForward {
    private let block: DiTBlockExecutor
    private let finalLayer: DiTFinalLayerExecutor

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
         monitor: NumericalMonitor? = nil) throws {
        block = try DiTBlockExecutor(
            context: context, file: file, attentionNumerics: attentionNumerics,
            activationNumerics: activationNumerics, monitor: monitor)
        finalLayer = try DiTFinalLayerExecutor(
            context: context, file: file, activationNumerics: activationNumerics,
            monitor: monitor)
    }

    /// Mutates the tightly packed fp32 `[1024,2048]` residual in place.
    /// The optional callback runs after GPU completion and exists for diagnostics/tests;
    /// normal inference leaves it nil and performs no per-block CPU readback.
    ///
    /// Execution uses the bounded two-slot ping-pong weight streamer (Phase 12):
    /// block N's weights are copied into the other slot while block N-1 executes
    /// on the GPU, so CPU memcpy hides behind GPU work. Outputs are byte-identical
    /// to the one-slot pattern (same bytes at the same offsets in the slot).
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
        // Ping-pong prologue: block 0 into slot 0.
        try block.prefetch(blockIndex: 0, slot: 0)
        for logicalIndex in 0..<blockCount {
            let slot = logicalIndex % 2
            let prefetch: (index: Int, slot: Int)?
            if logicalIndex + 1 < blockCount {
                prefetch = (logicalIndex + 1, (logicalIndex + 1) % 2)
            } else {
                prefetch = nil
            }
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
                slot: slot, prefetchIndex: prefetch?.index,
                prefetchSlot: prefetch?.slot ?? 0,
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
