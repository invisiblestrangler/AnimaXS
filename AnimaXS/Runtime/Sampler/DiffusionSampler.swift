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
    /// nil when `optimization.numericalMonitoring` is OFF (the experiment that
    /// measures monitoring overhead). The final CPU finite guard stays on
    /// regardless — a non-finite latent still fails safely.
    private let monitor: NumericalMonitor?
    private let stateLock = NSLock()
    private var running = false

    /// Run telemetry collector (injected by GenerationEngine after stage
    /// construction). Forwarded to the stage executors.
    var metrics: MetricsCollector? {
        didSet {
            preparation.metrics = metrics
            forward.metrics = metrics
            euler.metrics = metrics
        }
    }

    init(context: MetalContext, file: AnimapkFile,
         attentionNumerics: AttentionNumerics = .legacy,
         activationNumerics: ActivationNumerics = .legacy,
         optimization: InferenceOptimizationConfig = .currentBaseline) throws {
        self.context = context
        // Numerical monitoring OFF removes monitor/probe work from the
        // production path (the final CPU finite guard is retained). ON keeps
        // the current production monitor exactly.
        let monitor = optimization.numericalMonitoring
            ? try NumericalMonitor(context: context) : nil
        self.monitor = monitor
        preparation = try DiTPreparationExecutor(
            context: context, file: file, activationNumerics: activationNumerics,
            monitor: monitor)
        forward = try DitForward(
            context: context, file: file, attentionNumerics: attentionNumerics,
            activationNumerics: activationNumerics, monitor: monitor,
            optimization: optimization)
        euler = EulerSampler(context: context, monitor: monitor)
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
        monitor?.beginRun()
        guard (0...ModelConstants.samplerSteps).contains(startStep) else {
            throw AnimapkError.validation(
                "startStep \(startStep) out of range 0...\(ModelConstants.samplerSteps)")
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
            metrics?.beginStep(step)
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
                blockCompleted: { [self] block, _ in
                try blockProgress?(step, block)
                try diagnosticBlockCompleted?(step, block, residual)
                monitor?.noteBlockCompleted(step: step, block: block)
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
            monitor?.noteStepCompleted(step: step)
            guard isFinite(next, count: Self.latentElements) else {
                // 1-based step for the human-visible message; attribution
                // (block/stage/condition) is added by the numerical monitor.
                throw numericalFailure(step: step)
            }
            metrics?.endStep()
            try stepCompleted?(step, sigma, nextSigma, denoised, next)
            swap(&latent, &next)
        }
        if let monitor {
            metrics?.setNumericalWarnings(monitor.warningCount())
            metrics?.setNumericalDetails(monitor.warningDetails())
        } else {
            // Monitor OFF: warnings were not collected — never report "0".
            metrics?.setNumericalMonitoringDisabled(true)
        }
        try await copy(latent, to: outputLatent, bytes: bytes)
    }

    // MARK: - Diagnostic accessors (stress harness / tests)

    /// Full probe report after a completed (or failed) run. Empty when the
    /// numerical monitor was disabled for this run.
    var numericalReport: [NumericalMonitor.Probe: NumericalMonitor.Stats] {
        monitor?.report() ?? [:]
    }

    /// First unsafe (step, block, probe) attribution, if any. nil when the
    /// monitor was disabled or no issue was recorded.
    var earliestNumericalIssue: NumericalMonitor.FirstIssue? {
        monitor?.earliestIssue
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

    private func numericalFailure(step: Int) -> NumericalFailure {
        if let monitor, let issue = monitor.earliestIssue {
            return NumericalFailure.attributed(
                step: issue.step, totalSteps: ModelConstants.samplerSteps,
                block: issue.block ?? 1, totalBlocks: DiTBlockLocator.blockCount,
                stage: issue.probe.stageLabel, condition: issue.stats.condition)
        }
        return NumericalFailure.eulerOutput(
            step: step + 1, totalSteps: ModelConstants.samplerSteps)
    }

    /// Test seam: the failure produced when the Euler finite guard fires with
    /// no monitor attached (monitoring OFF). Proves the guard is independent
    /// of monitoring and still fails safely with a 1-based step.
    static func numericalFailureForTesting(
        step: Int, monitor: NumericalMonitor?
    ) -> NumericalFailure {
        if let monitor, let issue = monitor.earliestIssue {
            return NumericalFailure.attributed(
                step: issue.step, totalSteps: ModelConstants.samplerSteps,
                block: issue.block ?? 1, totalBlocks: DiTBlockLocator.blockCount,
                stage: issue.probe.stageLabel, condition: issue.stats.condition)
        }
        return NumericalFailure.eulerOutput(
            step: step + 1, totalSteps: ModelConstants.samplerSteps)
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
        if let monitor {
            try monitor.encodeProbeF32(
                command, values: denoised, count: Self.latentElements, probe: .denoised)
        }
        try await commit(command)
    }

    private func convertToHalf(
        _ input: MTLBuffer, output: MTLBuffer, count: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create context conversion command")
        }
        // Use the non-probe conversion when the numerical monitor is OFF so
        // the monitoring-overhead experiment measures a clean path.
        let pipeline = try context.pipeline(
            named: monitor != nil ? "float_to_half_probe" : "float_to_half")
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        if let monitor {
            monitor.bindProbe(encoder, probe: .crossContextToHalf, statsIndex: 3, slotIndex: 4)
        }
        let groups = (Int(count) + 255) / 256
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
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
