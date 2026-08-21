import Foundation
import Metal

/// One resolved model pack: its URL, component, and the variant descriptor
/// identifying which accepted variant (W4 or W8-v2 for DiT) was actually
/// installed and verified.
struct ResolvedModelPack: Hashable {
    let url: URL
    let component: ModelComponent
    let variant: ModelVariantDescriptor
}

/// Immutable user-LoRA selection captured at Generate time. The adapter is not
/// a model pack and never replaces/mutates the base DiT. Strength and identity
/// are frozen for the whole generation, which also makes generation-local
/// cross-K/V caching safe when cross-attention K/V are adapted.
struct ResolvedLoRA: Equatable {
    let url: URL
    let displayName: String
    let strength: Float
}

/// Production model resolution: exactly three base packs (K002 §5.1), plus an
/// optional external DiT LoRA snapshot.
///
/// The DiT pack serves both the LLM adapter and the diffusion sampler. External
/// LoRA is an overlay over diffusion projections; it is deliberately not a
/// fourth model pack and does not participate in base model resolution.
///
/// Each pack carries its resolved variant descriptor so consumers can report
/// which variant (W4 or W8-v2) actually ran.
struct ResolvedModels: Equatable {
    let textEncoder: ResolvedModelPack
    let dit: ResolvedModelPack
    let vae: ResolvedModelPack
    let lora: ResolvedLoRA?

    init(
        textEncoder: ResolvedModelPack,
        dit: ResolvedModelPack,
        vae: ResolvedModelPack,
        lora: ResolvedLoRA? = nil
    ) {
        self.textEncoder = textEncoder
        self.dit = dit
        self.vae = vae
        self.lora = lora
    }

    /// Convenience init from raw URLs and variant descriptors.
    init(textEncoderURL: URL, textEncoderVariant: ModelVariantDescriptor,
         ditURL: URL, ditVariant: ModelVariantDescriptor,
         vaeURL: URL, vaeVariant: ModelVariantDescriptor,
         lora: ResolvedLoRA? = nil) {
        self.textEncoder = ResolvedModelPack(url: textEncoderURL, component: .textEncoder, variant: textEncoderVariant)
        self.dit = ResolvedModelPack(url: ditURL, component: .dit, variant: ditVariant)
        self.vae = ResolvedModelPack(url: vaeURL, component: .vae, variant: vaeVariant)
        self.lora = lora
    }

    func withLoRA(_ lora: ResolvedLoRA?) -> ResolvedModels {
        ResolvedModels(textEncoder: textEncoder, dit: dit, vae: vae, lora: lora)
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
    func prepareDiffusion(
        fileURL: URL, optimization: InferenceOptimizationConfig, metrics: MetricsCollector
    ) throws
    func makeDiffusion(
        context: MetalContext, fileURL: URL,
        optimization: InferenceOptimizationConfig,
        numerics: DiTNumericsPolicy
    ) throws -> DiffusionStage
    func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage
}

extension GenerationStageFactory {
    /// Test factories do not own real model packs, so preparation is a no-op
    /// unless the production factory overrides it.
    func prepareDiffusion(
        fileURL: URL, optimization: InferenceOptimizationConfig, metrics: MetricsCollector
    ) throws {}
}

/// Production factory: real executors over real `AnimapkFile` mmaps.
struct ProductionStageFactory: GenerationStageFactory {
    func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
        try QwenEncoderMetal(context: context, file: try AnimapkFile(url: fileURL))
    }

    func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
        try LLMAdapterMetal(context: context, file: try AnimapkFile(url: fileURL))
    }

    func prepareDiffusion(
        fileURL: URL, optimization: InferenceOptimizationConfig, metrics: MetricsCollector
    ) throws {
        guard optimization.linearBackend == .aneHybridW8 else { return }
        let file = try AnimapkFile(url: fileURL)
        let result = try ANEW8ModelPreparer.ensurePrepared(file: file)
        metrics.recordANECachePreparation(result)
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
    /// diffusion → VAE → RGBA8. Always starts diffusion from step 0.
    ///
    /// - Parameters:
    ///   - prompt: User prompt. Not altered between UI and tokenization.
    ///   - seed: User seed. `SeededRNG(seed:)` creates the initial latent unless
    ///           `noise` is injected (test-only golden path).
    ///   - models: Exactly three resolved base packs plus optional LoRA snapshot.
    ///   - noise: Optional pre-generated initial noise (test-only injection).
    ///   - progress: Stage progress, including diffusion step/block counts.
    ///   - metrics: Optional collector for this run's timing/memory telemetry;
    ///              a private one is created when nil (test path).
    ///   - optimization: Immutable per-run inference configuration snapshot.
    ///              Captured at Generate time; never mutated mid-run.
    func generate(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil,
        progress: ProgressCallback? = nil,
        metrics metricsIn: MetricsCollector? = nil,
        optimization: InferenceOptimizationConfig = .currentBaseline
    ) async throws -> DecodedRGBA8 {
        let metrics = metricsIn ?? MetricsCollector()
        metrics.recordOptimizationConfig(optimization)
        let numerics = DiTNumericsPolicy.fromVariantID(models.dit.variant.id)
        if let reason = InferenceOptimizationConfig.blockingReason(
            for: optimization, numerics: numerics, ditVariantID: models.dit.variant.id) {
            throw AnimapkError.validation(reason)
        }
        let generationStart = ProcessInfo.processInfo.systemUptime
        defer {
            metrics.finalize(totalWall: ProcessInfo.processInfo.systemUptime - generationStart)
        }
        progress?(.tokenizing)
        try Task.checkCancellation()
        let tokenized = try measuredSync(.tokenizing, metrics: metrics) {
            try tokenize(prompt: prompt)
        }
        let qwenTokenIDs = tokenized.qwen
        let t5IDs = tokenized.t5
        let t5Weights = tokenized.t5Weights

        progress?(.encodingPrompt)
        try Task.checkCancellation()
        let qwenOutput = try await measured(.textEncode, metrics: metrics) {
            try await encodePrompt(models: models, tokenIDs: qwenTokenIDs)
        }

        progress?(.adapting)
        try Task.checkCancellation()
        let cross = try await measured(.adapter, metrics: metrics) {
            try await adaptPrompt(
                models: models, qwenContext: qwenOutput,
                contextTokens: qwenTokenIDs.count, t5IDs: t5IDs, t5Weights: t5Weights)
        }

        let totalSteps = ModelConstants.samplerSteps
        let initialLatent = try makeInitialLatent(seed: seed, noise: noise)
        let finalLatent = try await measured(.diffusion, metrics: metrics) {
            try await diffuse(
                models: models, initialLatent: initialLatent, cross: cross,
                totalSteps: totalSteps,
                progress: progress, metrics: metrics,
                optimization: optimization)
        }

        Wan21LatentFormat.applyProcessOutInPlace(finalLatent)

        progress?(.decoding)
        try Task.checkCancellation()
        let decoded = try await measured(.vae, metrics: metrics) {
            try await decodeVAE(models: models, latent: finalLatent)
        }
        return decoded
    }

    // MARK: - Stage helpers (lexical lifetime boundaries)

    private func measuredSync<T>(
        _ stage: MetricsCollector.Stage, metrics: MetricsCollector,
        _ body: () throws -> T
    ) rethrows -> T {
        metrics.beginStage(stage)
        defer { metrics.endStage(stage) }
        return try body()
    }

    private func measured<T>(
        _ stage: MetricsCollector.Stage, metrics: MetricsCollector,
        _ body: () async throws -> T
    ) async rethrows -> T {
        metrics.beginStage(stage)
        defer { metrics.endStage(stage) }
        return try await body()
    }

    private func tokenize(prompt: String) throws -> (qwen: [Int], t5: [Int], t5Weights: [Float]) {
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
        totalSteps: Int,
        progress: ProgressCallback?,
        metrics: MetricsCollector, optimization: InferenceOptimizationConfig
    ) async throws -> MTLBuffer {
        try factory.prepareDiffusion(
            fileURL: models.dit.url, optimization: optimization, metrics: metrics)
        let sampler = try factory.makeDiffusion(
            context: context, fileURL: models.dit.url,
            optimization: optimization,
            numerics: DiTNumericsPolicy.fromVariantID(models.dit.variant.id))
        if let productionSampler = sampler as? DiffusionSampler {
            productionSampler.metrics = metrics
            try productionSampler.configureLoRA(models.lora)
        }
        defer { withExtendedLifetime(sampler) {} }
        let output = try makeBuffer(
            length: DiffusionSampler.latentElements * 4, "diffusion output buffer")
        let blocks = ModelConstants.ditBlocks
        try await sampler.execute(
            initialLatent: initialLatent, crossContext: cross, outputLatent: output,
            startStep: 0,
            blockProgress: { step, block in
                try Task.checkCancellation()
                progress?(.diffusing(
                    step: step + 1, block: block + 1,
                    totalSteps: totalSteps, totalBlocks: blocks))
            },
            stepCompleted: { step, _, _, _, _ in
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
}
