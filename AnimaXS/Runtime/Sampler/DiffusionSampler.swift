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
    /// P5: per-generation cross-attention K/V cache (nil when the toggle is off
    /// or allocation failed). Owned here so its lifetime is exactly one
    /// generation; never persisted across prompts.
    private let crossKVCache: CrossKVCache?
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
         optimization: InferenceOptimizationConfig = .currentBaseline,
         numerics: DiTNumericsPolicy? = nil) throws {
        self.context = context
        // When the pack-derived policy is supplied it selects the numerical
        // fidelity. Production W8-v2 resolves to `w8LegacyStabilized` (legacy
        // numerics — the known-good path); the BF16 experimental policy is
        // never selected by variant-id resolution, only by explicit request.
        // Policy is derived from the resolved variant id, never from the
        // app-owned filename.
        let resolvedActivation: ActivationNumerics
        let resolvedAttention: AttentionNumerics
        let resolvedFinalResidualBoundary: FinalResidualBoundary
        if let numerics {
            (resolvedActivation, resolvedAttention) = Self.resolvedNumerics(for: numerics)
            // The policy also selects the final-residual ENTRY boundary
            // (decoupled from activation numerics): production W8-v2 keeps
            // legacy block numerics but its large residual enters the final
            // layer via BF16-RNE-in-FP32, never FP16.
            resolvedFinalResidualBoundary = Self.resolvedFinalResidualBoundary(for: numerics)
        } else {
            resolvedActivation = activationNumerics
            resolvedAttention = attentionNumerics
            // Explicit construction (no pack policy): preserve the existing
            // experimental behavior — .bf16Compute activation numerics implies
            // the BF16 residual boundary, everything else keeps the FP16
            // legacy boundary.
            resolvedFinalResidualBoundary = activationNumerics == .bf16Compute
                ? .bf16RNEInFP32 : .fp16Legacy
        }
        // Numerical monitoring OFF removes monitor/probe work from the
        // production path (the final CPU finite guard is retained). ON keeps
        // the current production monitor exactly.
        let monitor = optimization.numericalMonitoring
            ? try NumericalMonitor(context: context) : nil
        self.monitor = monitor
        preparation = try DiTPreparationExecutor(
            context: context, file: file, activationNumerics: resolvedActivation,
            monitor: monitor, optimization: optimization)
        // P5: per-generation exact cross-attention K/V cache. The ANE W8
        // backend always requests it because device measurements show that the
        // post-hit six-program working set materially reduces private-runtime
        // load/residency/unload cost. Allocation failure still falls back
        // safely to bounded full8 execution for every traversal.
        let cache = Self.shouldUseCrossKVCache(optimization: optimization)
            ? CrossKVCache(device: context.device) : nil
        self.crossKVCache = cache
        forward = try DitForward(
            context: context, file: file, attentionNumerics: resolvedAttention,
            activationNumerics: resolvedActivation,
            finalResidualBoundary: resolvedFinalResidualBoundary,
            monitor: monitor,
            optimization: optimization, crossKVCache: cache)
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

    /// Pure policy seam: P5 remains opt-in for Metal backends, while ANE
    /// makes exact K/V reuse part of its measured production recipe.
    static func shouldUseCrossKVCache(optimization: InferenceOptimizationConfig) -> Bool {
        optimization.crossKVCache || optimization.linearBackend == .aneHybridW8
    }

    /// Runs the eight model evaluations and writes the final fp32 latent.
    /// `stepCompleted` is called after each finite post-Euler latent. Its
    /// buffers remain valid only for the callback.
    ///
    /// - Parameter startStep: Number of Euler steps to skip. Production always
    ///   passes `0` (start from step 0); diagnostic/trajectory tests may pass
    ///   a nonzero value to resume from a previously captured latent.
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
        // Publish numerical-monitor bookkeeping in a defer so it is recorded on
        // FAILURE as well as success (a thrown step must still surface its
        // collected warnings). We only publish state the monitor has already
        // completed — no GPU readback of an unfinished command buffer here.
        defer {
            if let monitor {
                metrics?.setNumericalWarnings(monitor.warningCount())
                metrics?.setNumericalDetails(monitor.warningDetails())
            } else {
                // Monitor OFF: warnings were not collected — never report "0".
                metrics?.setNumericalMonitoringDisabled(true)
            }
        }
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

        // Only execute the remaining sigma transitions (startStep == 8 copies
        // the latent straight to output).
        for step in startStep..<EulerSampler.sigmas.count - 1 {
            metrics?.beginStep(step)
            // P2: a step that throws is still recorded as a PARTIAL step
            // (completed == false) with its partial durations/counters, so a
            // device log can attribute a slowdown to the failing step (e.g.
            // the W8 failure case). On success the explicit endStep below
            // keeps the historical timing point (before the step-completed
            // callback); the defer only fires on failure.
            var completedForMetrics = false
            defer {
                if !completedForMetrics { metrics?.endStep(completed: false) }
            }
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
            metrics?.endStep(completed: true)
            completedForMetrics = true
            try stepCompleted?(step, sigma, nextSigma, denoised, next)
            swap(&latent, &next)
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
            return Self.attributedFailure(from: issue)
        }
        return NumericalFailure.eulerOutput(
            step: step + 1, totalSteps: ModelConstants.samplerSteps)
    }

    /// Maps a numerical-monitor first-issue to a failure with the correct
    /// location. Final-layer probes are attributed as "final layer" (never a
    /// fabricated block); block probes use the attributed block; otherwise the
    /// post-Euler guard is the location.
    private static func attributedFailure(
        from issue: NumericalMonitor.FirstIssue
    ) -> NumericalFailure {
        if issue.probe.isFinalLayer {
            return NumericalFailure.finalLayer(
                step: issue.step, totalSteps: ModelConstants.samplerSteps,
                totalBlocks: DiTBlockLocator.blockCount,
                stage: issue.probe.stageLabel, condition: issue.stats.condition)
        }
        if let block = issue.block {
            return NumericalFailure.attributed(
                step: issue.step, totalSteps: ModelConstants.samplerSteps,
                block: block, totalBlocks: DiTBlockLocator.blockCount,
                stage: issue.probe.stageLabel, condition: issue.stats.condition)
        }
        return NumericalFailure.eulerOutput(
            step: issue.step, totalSteps: ModelConstants.samplerSteps)
    }

    /// Test seam: the failure produced when the Euler finite guard fires with
    /// no monitor attached (monitoring OFF). Proves the guard is independent
    /// of monitoring and still fails safely with a 1-based step.
    static func numericalFailureForTesting(
        step: Int, monitor: NumericalMonitor?
    ) -> NumericalFailure {
        if let monitor, let issue = monitor.earliestIssue {
            return attributedFailure(from: issue)
        }
        return NumericalFailure.eulerOutput(
            step: step + 1, totalSteps: ModelConstants.samplerSteps)
    }

    /// Resolves the attention/activation numerics selected by a DiT numerics
    /// policy. The single source of truth for the policy -> numerics mapping,
    /// used by `init` (test seam: no Metal context required to assert it).
    static func resolvedNumerics(
        for policy: DiTNumericsPolicy
    ) -> (activation: ActivationNumerics, attention: AttentionNumerics) {
        switch policy {
        case .w4Legacy:
            return (.legacy, .legacy)
        case .w8LegacyStabilized:
            return (.legacy, .legacy)
        case .w8BF16Experimental:
            return (.bf16Compute, .bf16Compute)
        }
    }

    /// Resolves the final-residual ENTRY boundary selected by a DiT numerics
    /// policy. DECOUPLED from `resolvedNumerics`: production W8-v2
    /// (w8LegacyStabilized) keeps legacy activation numerics yet its large
    /// final residual must never be converted to FP16 (overflow above 65,504),
    /// so the policy carries a source-faithful BF16-RNE-in-FP32 boundary. W4
    /// keeps its byte-for-byte FP16 legacy boundary. Test seam: no Metal
    /// context required.
    static func resolvedFinalResidualBoundary(
        for policy: DiTNumericsPolicy
    ) -> FinalResidualBoundary {
        switch policy {
        case .w4Legacy:
            return .fp16Legacy
        case .w8LegacyStabilized:
            return .bf16RNEInFP32
        case .w8BF16Experimental:
            return .bf16RNEInFP32
        }
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
