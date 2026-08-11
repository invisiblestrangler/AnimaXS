import Foundation
import Metal

/// Canonical Anima-Turbo 8-step sgm_uniform schedule and fp32 Euler update.
/// The denoised input is the model-sampling output, not the raw DiT velocity.
final class EulerSampler {
    static let sigmas: [Float] = [
        1.0,
        0.9546938,
        0.90035903,
        0.8339981,
        0.7511211,
        0.64468634,
        0.50298506,
        0.30500895,
        0.0
    ]

    private let context: MetalContext

    init(context: MetalContext) {
        self.context = context
    }

    static func cpuStep(
        latent: [Float], denoised: [Float], sigma: Float, nextSigma: Float
    ) throws -> [Float] {
        guard !latent.isEmpty, latent.count == denoised.count,
              sigma.isFinite, sigma > 0, nextSigma.isFinite else {
            throw AnimapkError.validation("invalid Euler sampler input")
        }
        let delta = nextSigma - sigma
        return zip(latent, denoised).map { x, d in
            x + (x - d) / sigma * delta
        }
    }

    func encodeStep(
        commandBuffer: MTLCommandBuffer, latent: MTLBuffer,
        denoised: MTLBuffer, output: MTLBuffer,
        sigma: Float, nextSigma: Float, count: Int
    ) throws {
        guard count > 0, sigma.isFinite, sigma > 0, nextSigma.isFinite,
              latent.length >= count * 4, denoised.length >= count * 4,
              output.length >= count * 4 else {
            throw AnimapkError.validation("invalid Euler sampler buffer or sigma")
        }
        let pipeline = try context.pipeline(named: "euler_step_f32")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Euler sampler encoder")
        }
        var sigma = sigma, delta = nextSigma - sigma, count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(latent, offset: 0, index: 0)
        encoder.setBuffer(denoised, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&sigma, length: 4, index: 3)
        encoder.setBytes(&delta, length: 4, index: 4)
        encoder.setBytes(&count, length: 4, index: 5)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func executeStep(
        latent: MTLBuffer, denoised: MTLBuffer, output: MTLBuffer,
        sigma: Float, nextSigma: Float, count: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create Euler command buffer")
        }
        try encodeStep(commandBuffer: command, latent: latent, denoised: denoised,
                       output: output, sigma: sigma, nextSigma: nextSigma, count: count)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }
}
