import Foundation
import Metal

/// Streamed 28-block MiniTrainDIT transformer loop. Each block is awaited before
/// the next locator-derived range replaces the one-slot weight ring.
final class DitForward {
    private let block: DiTBlockExecutor
    private let finalLayer: DiTFinalLayerExecutor

    init(context: MetalContext, file: AnimapkFile,
         attentionNumerics: AttentionNumerics = .legacy) throws {
        block = try DiTBlockExecutor(
            context: context, file: file, attentionNumerics: attentionNumerics)
        finalLayer = try DiTFinalLayerExecutor(context: context, file: file)
    }

    /// Mutates the tightly packed fp32 `[1024,2048]` residual in place.
    /// The optional callback runs after GPU completion and exists for diagnostics/tests;
    /// normal inference leaves it nil and performs no per-block CPU readback.
    func execute(
        residual: MTLBuffer,
        emb: MTLBuffer,
        adalnLora: MTLBuffer,
        crossContext: MTLBuffer,
        rope: MTLBuffer,
        blockCompleted: ((Int, MTLBuffer) throws -> Void)? = nil
    ) async throws {
        for logicalIndex in 0..<DiTBlockLocator.blockCount {
            try await block.execute(
                blockIndex: logicalIndex, residual: residual, emb: emb,
                adalnLora: adalnLora, crossContext: crossContext, rope: rope)
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
