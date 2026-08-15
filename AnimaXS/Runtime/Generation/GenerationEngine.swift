import Foundation
import Metal

/// One resolved model pack: its URL, component, and the variant descriptor
/// identifying which accepted variant (W4 or W8-v2 for DiT) was actually
/// installed and verified.
struct ResolvedModelPack: Equatable {
    let url: URL
    let component: ModelComponent
    let variant: ModelVariantDescriptor
}

/// Production model resolution: exactly three packs (K002 §5.1).
///
/// The DiT pack serves both the LLM adapter and the diffusion sampler —
/// there is deliberately no fourth "adapter" pack in production app state.
/// Isolated adapter tests may construct `LLMAdapterMetal` directly with a
/// dedicated fixture; that fixture architecture never leaks into here.
///
/// Each pack carries its resolved variant descriptor so consumers can report
/// which variant (W4 or W8-v2) actually ran, and checkpoint identity can use
/// the actual resolved hashes instead of a hardcoded W4 baseline.
struct ResolvedModels: Equatable {
    let textEncoder: ResolvedModelPack
    let dit: ResolvedModelPack
    let vae: ResolvedModelPack

    /// The actual resolved model hashes for checkpoint identity (not the
    /// hardcoded W4 production baseline).
    var hashes: ModelHashes {
        ModelHashes(
            dit: dit.variant.sha256,
            textEncoder: textEncoder.variant.sha256,
            vae: vae.variant.sha256)
    }

    init(textEncoder: ResolvedModelPack, dit: ResolvedModelPack, vae: ResolvedModelPack) {
        self.textEncoder = textEncoder
        self.dit = dit
        self.vae = vae
    }

    /// Convenience init from raw URLs and variant descriptors.
    init(textEncoderURL: URL, textEncoderVariant: ModelVariantDescriptor,
         ditURL: URL, ditVariant: ModelVariantDescriptor,
         vaeURL: URL, vaeVariant: ModelVariantDescriptor) {
        self.textEncoder = ResolvedModelPack(url: textEncoderURL, component: .textEncoder, variant: textEncoderVariant)
        self.dit = ResolvedModelPack(url: ditURL, component: .dit, variant: ditVariant)
        self.vae = ResolvedModelPack(url: vaeURL, component: .vae, variant: vaeVariant)
    }
}

/// Cooperative cancellation policy for a generation run (K003 core).
enum GenerationCancellation {
    case none
    case requested
}

/// Why a generation run was cancelled. Telemetry only — it lets final metrics
/// and the cancelled UI state distinguish user-initiated cancellation from
/// automatic app-lifecycle / resource-driven cancellation. It is not a
/// resource policy and introduces no thermal cancellation.
enum GenerationCancellationReason: String, Codable {
    case user
    case background
    case memoryWarning
    case taskCancellation
    case unknown
}

// MARK: - Stage seams (narrow dependency injection for orchestration tests)

/// Protocol for the Qwen text-encoder stage. Production: `QwenEncoderMetal`.
protocol PromptEncoderStage: AnyObject {
    func execute(
        tokenIDs: [Int], output: MTLBuffer,
        layerCompleted: ((Int, MTLBuffer) throws -> Void)?
    ) async throws
}

/// Protocol for the LLM adapter stage. Production: `LLMAdapterMetal`.
protocol ContextAdapterStage: AnyObject {
    func execute(
        qwenContext: MTLBuffer, contextTokens: Int,
        t5IDs: [Int], t5Weights: [Float], output: MTLBuffer,
        layerCompleted: ((Int, MTLBuffer) throws -> Void)?
    ) async throws
}

/// Protocol for the diffusion sampler stage. Production: `DiffusionSampler`.
protocol DiffusionStage: AnyObject {
    func execute(
        initialLatent: MTLBuffer,
        crossContext: MTLBuffer,
        outputLatent: MTLBuffer,
        startStep: Int,
        blockProgress: ((Int, Int) throws -> Void)?,
        stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?
    ) async throws
}

/// Protocol for the VAE decode stage. Production: `VAEDecoder`.
protocol VAEDecodeStage: AnyObject {
    func decode(latent: MTLBuffer) async throws -> DecodedRGBA8
}

/// Builds stage objects for the engine. The production factory constructs the
/// real Metal executors; tests inject probe factories with `deinit` tracking.
protocol GenerationStageFactory {
    func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage
    func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage
    func makeDiffusion(
        context: MetalContext, fileURL: URL,
        optimization: InferenceOptimizationConfig,
        numerics: DiTNumericsPolicy
    ) throws -> DiffusionStage
    func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage
}

/// Production factory: real executors over real `AnimapkFile` mmaps.
struct ProductionStageFactory: GenerationStageFactory {
    func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
        try QwenEncoderMetal(context: context, file: try AnimapkFile(url: fileURL))
    }

    func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
        try LLMAdapterMetal(context: context, file: try AnimapkFile(url: fileURL))
    }

    func makeDiffusion(
        context: MetalContext, fileURL: URL,
        optimization: InferenceOptimizationConfig,
        numerics: DiTNumericsPolicy
    ) throws -> DiffusionStage {
        try DiffusionSampler(
            context: context, file: try AnimapkFile(url: fileURL),
            optimization: optimization, numerics: numerics)
    }

    func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
        try VAEDecoder(context: context, file: try AnimapkFile(url: fileURL))
    }
}

// MARK: - Engine

/// Non-UI inference engine (K002 §5.4). NOT MainActor-isolated: model
/// construction, tokenization, mmap setup and heavy Metal work run off the
/// main actor. The UI-facing coordinator publishes progress to the view.
///
/// Stage-scoped lifetime (K002 §5.3): each pipeline stage runs inside its own
/// helper function. The heavy model object AND its `AnimapkFile` mmap are
/// local to that function and cannot escape it — the only value that crosses
/// a stage boundary is the fp32 conditioning buffer (512×1024) between the
/// text encoder and the DiT.
struct GenerationEngine {
    let context: MetalContext
    private let factory: any GenerationStageFactory

    /// Progress callback invoked synchronously on the engine's executor.
    typealias ProgressCallback = (GenerationStage) -> Void

    enum GenerationStage: Equatable {
        case tokenizing
        case encodingPrompt
        case adapting
        case diffusing(step: Int, block: Int, totalSteps: Int, totalBlocks: Int)
        case decoding
    }

    init(context: MetalContext, factory: any GenerationStageFactory = ProductionStageFactory()) {
        self.context = context
        self.factory = factory
    }

    /// Runs the full production pipeline: prompt → tokens → Qwen → adapter →
    /// diffusion → VAE → RGBA8.
    ///
    /// - Parameters:
    ///   - prompt: User prompt. Not altered between UI and tokenization.
    ///   - seed: User seed. `SeededRNG(seed:)` creates the initial latent unless
    ///           `noise` is injected (test-only golden path).
    ///   - models: Exactly three resolved pack URLs.
    ///   - noise: Optional pre-generated initial noise (test-only injection).
    ///   - startStep: Resume support: number of Euler steps already completed.
    ///   - progress: Stage progress, including diffusion step/block counts.
    ///   - checkpoint: Called after each completed diffusion step with the
    ///                 completed step index and the fp32 latent snapshot.
    ///   - metrics: Optional collector for this run's timing/memory telemetry;
    ///              a private one is created when nil (test path).
    ///   - optimization: Immutable per-run inference configuration snapshot.
    ///              Captured at Generate time; never mutated mid-run.
    func generate(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil,
        startStep: Int = 0,
        progress: ProgressCallback? = nil,
        checkpoint: ((Int, [Float]) throws -> Void)? = nil,
        metrics metricsIn: MetricsCollector? = nil,
        optimization: InferenceOptimizationConfig = .currentBaseline
    ) async throws -> DecodedRGBA8 {
        let metrics = metricsIn ?? MetricsCollector()
        metrics.recordOptimizationConfig(optimization)
        let generationStart = ProcessInfo.processInfo.systemUptime
        defer {
            metrics.finalize(totalWall: ProcessInfo.processInfo.systemUptime - generationStart)
        }
        // ---- 1. Tokenization (production TokenizerLoader semantics) ----
        progress?(.tokenizing)
        try Task.checkCancellation()
        let tokenized = try measuredSync(.tokenizing, metrics: metrics) {
            try tokenize(prompt: prompt)
        }
        let qwenTokenIDs = tokenized.qwen
        let t5IDs = tokenized.t5
        let t5Weights = tokenized.t5Weights

        // ---- 2. Qwen text encoding (stage-scoped; Qwen + its mmap cannot
        // escape this helper) ----
        progress?(.encodingPrompt)
        try Task.checkCancellation()
        let qwenOutput = try await measured(.textEncode, metrics: metrics) {
            try await encodePrompt(models: models, tokenIDs: qwenTokenIDs)
        }

        // ---- 3. Adapter → crossContext [512, 1024] fp32 (adapter + its mmap
        // cannot escape this helper) ----
        progress?(.adapting)
        try Task.checkCancellation()
        let cross = try await measured(.adapter, metrics: metrics) {
            try await adaptPrompt(
                models: models, qwenContext: qwenOutput,
                contextTokens: qwenTokenIDs.count, t5IDs: t5IDs, t5Weights: t5Weights)
        }

        // ---- 4. Diffusion: seeded noise → final latent (sampler-space) ----
        let totalSteps = ModelConstants.samplerSteps
        let initialLatent = try makeInitialLatent(seed: seed, noise: noise)
        let finalLatent = try await measured(.diffusion, metrics: metrics) {
            try await diffuse(
                models: models, initialLatent: initialLatent, cross: cross,
                startStep: startStep, totalSteps: totalSteps,
                progress: progress, checkpoint: checkpoint, metrics: metrics,
                optimization: optimization)
        }

        // ---- 4b. Wan21 latent-format boundary (sampler-space → VAE-space) ----
        // Root-cause fix (2026-08-14): ComfyUI applies latent_format.process_out
        // to the sampler return before the workflow's VAE decode
        // (samplers.py CFGGuider.inner_sample). The custom runtime omitted it,
        // decoding raw sampler latents as VAE latents — the 8px grid. Apply
        // EXACTLY ONCE here; VAEDecoder still consumes its canonical latent
        // unchanged (D060). Checkpoints remain sampler-space (diffusion math).
        Wan21LatentFormat.applyProcessOutInPlace(finalLatent)

        // ---- 5. VAE decode (VAE + its mmap cannot escape this helper) ----
        progress?(.decoding)
        try Task.checkCancellation()
        let decoded = try await measured(.vae, metrics: metrics) {
            try await decodeVAE(models: models, latent: finalLatent)
        }
        return decoded
    }

    // MARK: - Stage helpers (lexical lifetime boundaries)

    /// Measures a synchronous stage, ALWAYS closing its timer even when `body`
    /// throws (so a failed stage is not silently folded into "Other").
    private func measuredSync<T>(
        _ stage: MetricsCollector.Stage, metrics: MetricsCollector,
        _ body: () throws -> T
    ) rethrows -> T {
        metrics.beginStage(stage)
        defer { metrics.endStage(stage) }
        return try body()
    }

    /// Measures an async stage, ALWAYS closing its timer even when `body`
    /// throws. This is the failure-safe replacement for scattered begin/end
    /// pairs: e.g. a thrown `diffuse()` must still record nonzero diffusion time.
    private func measured<T>(
        _ stage: MetricsCollector.Stage, metrics: MetricsCollector,
        _ body: () async throws -> T
    ) async rethrows -> T {
        metrics.beginStage(stage)
        defer { metrics.endStage(stage) }
        return try await body()
    }

    private func tokenize(prompt: String) throws -> (qwen: [Int], t5: [Int], t5Weights: [Float]) {
        // Qwen: encode(prompt, no specials) — no start/end token.
        // T5:   encode(prompt, no specials) + [1] (trailing </s> EOS).
        // t5Weights: all 1.0 (verified from case1 fixture JSON).
        let qwenTokenizer = try TokenizerLoader.qwen()
        let qwen = qwenTokenizer.encode(text: prompt, addSpecialTokens: false)
        guard !qwen.isEmpty else {
            throw GenerationError.tokenizer("Qwen tokenizer produced no tokens")
        }
        let t5Tokenizer = try TokenizerLoader.t5()
        let t5 = t5Tokenizer.encode(text: prompt, addSpecialTokens: false) + [1]
        let t5Weights = [Float](repeating: 1.0, count: t5.count)
        return (qwen, t5, t5Weights)
    }

    private func encodePrompt(
        models: ResolvedModels, tokenIDs: [Int]
    ) async throws -> MTLBuffer {
        let encoder = try factory.makePromptEncoder(context: context, fileURL: models.textEncoder.url)
        // Structural lifetime boundary: `encoder` (and its AnimapkFile mmap) is
        // strongly held by this defer until the helper returns, then released.
        defer { withExtendedLifetime(encoder) {} }
        let output = try makeBuffer(
            length: tokenIDs.count * QwenEncoderMetal.hidden * 4,
            "Qwen output buffer")
        try await encoder.execute(tokenIDs: tokenIDs, output: output, layerCompleted: nil)
        return output
    }

    private func adaptPrompt(
        models: ResolvedModels, qwenContext: MTLBuffer, contextTokens: Int,
        t5IDs: [Int], t5Weights: [Float]
    ) async throws -> MTLBuffer {
        // Production topology: adapter reads the DiT pack (same URL as sampler).
        let adapter = try factory.makeContextAdapter(context: context, fileURL: models.dit.url)
        defer { withExtendedLifetime(adapter) {} }
        let output = try makeBuffer(
            length: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden * 4,
            "adapter output buffer")
        try await adapter.execute(
            qwenContext: qwenContext, contextTokens: contextTokens,
            t5IDs: t5IDs, t5Weights: t5Weights, output: output, layerCompleted: nil)
        return output
    }

    private func diffuse(
        models: ResolvedModels, initialLatent: MTLBuffer, cross: MTLBuffer,
        startStep: Int, totalSteps: Int,
        progress: ProgressCallback?, checkpoint: ((Int, [Float]) throws -> Void)?,
        metrics: MetricsCollector, optimization: InferenceOptimizationConfig
    ) async throws -> MTLBuffer {
        guard (0...ModelConstants.samplerSteps).contains(startStep) else {
            throw GenerationError.sampler(
                "startStep \(startStep) out of range 0...\(ModelConstants.samplerSteps)")
        }
        let sampler = try factory.makeDiffusion(
            context: context, fileURL: models.dit.url,
            optimization: optimization,
            numerics: DiTNumericsPolicy.fromVariantID(models.dit.variant.id))
        // Production path: inject the run's metrics collector into the sampler
        // (and through it the preparation/forward/final-layer/euler executors).
        if let sampler = sampler as? DiffusionSampler {
            sampler.metrics = metrics
        }
        defer { withExtendedLifetime(sampler) {} }
        let output = try makeBuffer(
            length: DiffusionSampler.latentElements * 4, "diffusion output buffer")
        let blocks = ModelConstants.ditBlocks
        try await sampler.execute(
            initialLatent: initialLatent, crossContext: cross, outputLatent: output,
            startStep: startStep,
            blockProgress: { step, block in
                try Task.checkCancellation()
                progress?(.diffusing(
                    step: step + 1, block: block + 1,
                    totalSteps: totalSteps, totalBlocks: blocks))
            },
            stepCompleted: { step, _, _, denoised, latent in
                // Copy the fp32 latent into owned storage INSIDE the callback:
                // the Metal buffer is only valid for the callback duration.
                let values = readFloats(latent, count: DiffusionSampler.latentElements)
                try checkpoint?(step, values)
                progress?(.diffusing(
                    step: step + 1, block: blocks,
                    totalSteps: totalSteps, totalBlocks: blocks))
            })
        return output
    }

    private func decodeVAE(models: ResolvedModels, latent: MTLBuffer) async throws -> DecodedRGBA8 {
        let decoder = try factory.makeVAE(context: context, fileURL: models.vae.url)
        defer { withExtendedLifetime(decoder) {} }
        return try await decoder.decode(latent: latent)
    }

    // MARK: - Helpers

    /// Seeded production RNG (K002 §5.2). The user's seed always feeds
    /// `SeededRNG`; only an explicit test-injected noise buffer bypasses it.
    private func makeInitialLatent(seed: UInt64, noise: MTLBuffer?) throws -> MTLBuffer {
        if let noise { return noise }
        var rng = SeededRNG(seed: seed)
        let count = DiffusionSampler.latentElements
        let buffer = try makeBuffer(length: count * 4, "initial latent")
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { pointer[i] = rng.nextNormal() }
        return buffer
    }

    private func makeBuffer(length: Int, _ label: String) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(
            length: length, options: .storageModeShared) else {
            throw GenerationError.metal("failed to allocate \(label)")
        }
        return buffer
    }

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
