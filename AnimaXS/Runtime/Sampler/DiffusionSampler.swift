import Foundation
import Metal

/// Production single-pass (CFG=1) Anima-Turbo diffusion loop.
///
/// Model output is FLOW velocity. It is materialized as fp32 denoised output
/// before the fp32 Euler update to preserve the pinned ComfyUI operation order.
final class DiffusionSampler {
    static let latentElements = 16 * 64 * 64

    typealias BlockProgress = (_ step: Int, _ block: Int) throws -> Void
    typealias StepCompleted = (
        _ step: Int, _ sigma: Float, _ nextSigma: Float,
        _ denoised: MTLBuffer, _ latent: MTLBuffer
    ) throws -> Void
    typealias DiagnosticStepPrepared = (
        _ step: Int, _ residual: MTLBuffer, _ embedding: MTLBuffer,
        _ adalnLora: MTLBuffer, _ crossContextHalf: MTLBuffer, _ rope: MTLBuffer
    ) throws -> Void
    typealias DiagnosticBlockCompleted = (
        _ step: Int, _ block: Int, _ residual: MTLBuffer
    ) throws -> Void
    typealias DiagnosticBranchCompleted = (
        _ step: Int, _ block: Int, _ branch: String, _ residual: MTLBuffer
    ) throws -> Void

    private let context: MetalContext
    private let preparation: DiTPreparationExecutor
    private let forward: DitForward
    private let euler: EulerSampler
    private let buffers: BufferPool
    private let rope: MTLBuffer
    private let stateLock = NSLock()
    private var running = false

    init(context: MetalContext, file: AnimapkFile,
         attentionNumerics: AttentionNumerics = .legacy,
         activationNumerics: ActivationNumerics = .legacy) throws {
        self.context = context
        preparation = try DiTPreparationExecutor(
            context: context, file: file, activationNumerics: activationNumerics)
        forward = try DitForward(
            context: context, file: file, attentionNumerics: attentionNumerics,
            activationNumerics: activationNumerics)
        euler = EulerSampler(context: context)
        buffers = BufferPool(device: context.device)

        let values = DitRoPE.generate()
        guard let rope = context.device.makeBuffer(
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw AnimapkError.validation("failed to allocate DiT RoPE")
        }
        values.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress { memcpy(rope.contents(), base, bytes.count) }
        }
        self.rope = rope
    }

    /// Runs the eight model evaluations (or the remaining ones when resuming
    /// from a checkpoint) and writes the final fp32 latent.
    /// `stepCompleted` is called after each finite post-Euler latent and is the
    /// checkpoint boundary. Its buffers remain valid only for the callback.
    ///
    /// - Parameter startStep: Number of Euler steps already completed
    ///   (I004 resume semantics). `0` = begin at step 0; `8` = all steps
    ///   complete — the checkpoint latent is copied straight to output.
    /// - Precondition: `0 <= startStep <= 8`.
    func execute(
        initialLatent: MTLBuffer,
        crossContext: MTLBuffer,
        outputLatent: MTLBuffer,
        startStep: Int = 0,
        blockProgress: BlockProgress? = nil,
        stepCompleted: StepCompleted? = nil
    ) async throws {
        try await executeDiagnostic(
            initialLatent: initialLatent, crossContext: crossContext,
            outputLatent: outputLatent, startStep: startStep,
            blockProgress: blockProgress, stepCompleted: stepCompleted)
    }

    func executeDiagnostic(
        initialLatent: MTLBuffer,
        crossContext: MTLBuffer,
        outputLatent: MTLBuffer,
        startStep: Int = 0,
        blockProgress: BlockProgress? = nil,
        stepCompleted: StepCompleted? = nil,
        diagnosticStepPrepared: DiagnosticStepPrepared? = nil,
        diagnosticBlockCompleted: DiagnosticBlockCompleted? = nil,
        diagnosticBranchFilter: ((_ step: Int, _ block: Int) -> Bool)? = nil,
        diagnosticBranchCompleted: DiagnosticBranchCompleted? = nil
    ) async throws {
        try beginRun()
        defer { endRun() }
        guard (0...ModelConstants.samplerSteps).contains(startStep) else {
            throw AnimapkError.validation(
                "startStep \\(startStep) out of range 0...\\(ModelConstants.samplerSteps)")
        }
        let bytes = Self.latentElements * 4
        guard initialLatent.length >= bytes, outputLatent.length >= bytes,
              crossContext.length >= 512 * 1_024 * 4 else {
            throw AnimapkError.validation("invalid diffusion sampler buffer")
        }

        let latentA = buffers.buffer(key: "diffusion.latent.a.f32", bytes: bytes)
        let latentB = buffers.buffer(key: "diffusion.latent.b.f32", bytes: bytes)
        try await copy(initialLatent, to: latentA, bytes: bytes)
        var latent = latentA
        var next = latentB

        let residual = buffers.buffer(
            key: "diffusion.residual.f32",
            bytes: DiTPreparationExecutor.tokens * DiTPreparationExecutor.hidden * 4)
        let embedding = buffers.buffer(
            key: "diffusion.embedding.f32", bytes: DiTPreparationExecutor.hidden * 4)
        let adaln = buffers.buffer(
            key: "diffusion.adaln.f32", bytes: DiTPreparationExecutor.adaln * 4)
        let velocity = buffers.buffer(key: "diffusion.velocity.f32", bytes: bytes)
        let denoised = buffers.buffer(key: "diffusion.denoised.f32", bytes: bytes)
        let crossHalf = buffers.buffer(
            key: "diffusion.cross-context.f16", bytes: 512 * 1_024 * 2)
        try await convertToHalf(crossContext, output: crossHalf, count: 512 * 1_024)

        // Resume: the checkpoint latent already contains the result of steps
        // [0, startStep); only execute the remaining sigma transitions.
        for step in startStep..<EulerSampler.sigmas.count - 1 {
            let sigma = EulerSampler.sigmas[step]
            let nextSigma = EulerSampler.sigmas[step + 1]
            try await preparation.execute(
                latent: latent, sigma: sigma, residual: residual,
                embedding: embedding, adalnLora: adaln)
            try diagnosticStepPrepared?(
                step, residual, embedding, adaln, crossHalf, rope)
            try await forward.execute(
                residual: residual, emb: embedding, adalnLora: adaln,
                crossContext: crossHalf, rope: rope,
                blockCompleted: { block, _ in
                try blockProgress?(step, block)
                try diagnosticBlockCompleted?(step, block, residual)
            }, diagnosticBranchFilter: { block in
                diagnosticBranchFilter?(step, block) ?? true
            }, diagnosticBranchCompleted: { block, branch, current in
                try diagnosticBranchCompleted?(step, block, branch, current)
            })
            try await forward.executeVelocityFinalLayer(
                residual: residual, emb: embedding, adalnLora: adaln,
                velocity: velocity)
            try await convertVelocity(
                latent: latent, velocity: velocity, denoised: denoised, sigma: sigma)
            try await euler.executeStep(
                latent: latent, denoised: denoised, output: next,
                sigma: sigma, nextSigma: nextSigma, count: Self.latentElements)
            guard isFinite(next, count: Self.latentElements) else {
                throw AnimapkError.validation("non-finite diffusion latent after step \\(step)")
            }
            try stepCompleted?(step, sigma, nextSigma, denoised, next)
            swap(&latent, &next)
        }
        try await copy(latent, to: outputLatent, bytes: bytes)
    }

    static func cpuDenoised(latent: [Float], velocity: [Float], sigma: Float) throws -> [Float] {
        guard !latent.isEmpty, latent.count == velocity.count, sigma.isFinite else {
            throw AnimapkError.validation("invalid FLOW conversion input")
        }
        return zip(latent, velocity).map { $0 - sigma * $1 }
    }

    private func beginRun() throws {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard !running else { throw AnimapkError.validation("diffusion sampler is already running") }
        running = true
    }

    private func endRun() {
        stateLock.lock()
        running = false
        stateLock.unlock()
    }

    private func convertVelocity(
        latent: MTLBuffer, velocity: MTLBuffer, denoised: MTLBuffer, sigma: Float
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create FLOW conversion command")
        }
        let pipeline = try context.pipeline(named: "flow_velocity_to_denoised_f32")
        var sigma = sigma, count = UInt32(Self.latentElements)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(latent, offset: 0, index: 0)
        encoder.setBuffer(velocity, offset: 0, index: 1)
        encoder.setBuffer(denoised, offset: 0, index: 2)
        encoder.setBytes(&sigma, length: 4, index: 3)
        encoder.setBytes(&count, length: 4, index: 4)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: Self.latentElements, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        try await commit(command)
    }

    private func convertToHalf(
        _ input: MTLBuffer, output: MTLBuffer, count: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create context conversion command")
        }
        let pipeline = try context.pipeline(named: "float_to_half")
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        try await commit(command)
    }

    private func copy(_ source: MTLBuffer, to destination: MTLBuffer, bytes: Int) async throws {
        guard let command = context.commandQueue.makeCommandBuffer(),
              let blit = command.makeBlitCommandEncoder() else {
            throw AnimapkError.validation("failed to create diffusion blit command")
        }
        blit.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        try await commit(command)
    }

    private func commit(_ command: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }

    private func isFinite(_ buffer: MTLBuffer, count: Int) -> Bool {
        let values = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for index in 0..<count where !values[index].isFinite { return false }
        return true
    }
}
