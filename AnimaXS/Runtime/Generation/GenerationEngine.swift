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

/// Production model resolution: exactly three packs (K002 §5.1).
///
/// The DiT pack serves both the LLM adapter and the diffusion sampler —
/// there is deliberately no fourth "adapter" pack in production app state.
/// Isolated adapter tests may construct `LLMAdapterMetal` directly with a
/// dedicated fixture; that fixture architecture never leaks into here.
///
/// Each pack carries its resolved variant descriptor so consumers can report
/// which variant (W4 or W8-v2) actually ran.
struct ResolvedModels: Equatable {
    let textEncoder: ResolvedModelPack
    let dit: ResolvedModelPack
    let vae: ResolvedModelPack

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

enum GenerationCancellation {
    case none
    case requested
}

enum GenerationCancellationReason: String, Codable {
    case user
    case background
    case memoryWarning
    case taskCancellation
    case unknown
}

protocol PromptEncoderStage: AnyObject {
    func execute(
        tokenIDs: [Int], output: MTLBuffer,
        layerCompleted: ((Int, MTLBuffer) throws -> Void)?
    ) async throws
}

protocol ContextAdapterStage: AnyObject {
    func execute(
        qwenContext: MTLBuffer, contextTokens: Int,
        t5IDs: [Int], t5Weights: [Float], output: MTLBuffer,
        layerCompleted: ((Int, MTLBuffer) throws -> Void)?
    ) async throws
}

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

protocol VAEDecodeStage: AnyObject {
    func decode(latent: MTLBuffer) async throws -> DecodedRGBA8
}

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
    func prepareDiffusion(
        fileURL: URL, optimization: InferenceOptimizationConfig, metrics: MetricsCollector
    ) throws {}
}

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
        // Both private-ANE backends consume the same already-prepared native
        // projection donors. The multiprocedure backend combines those donors
        // into one 10-procedure model per block at first use; keeping the
        // existing preparer here makes the old and new paths directly A/B-able.
        guard optimization.linearBackend.isANEW8 else { return }
        let file = try AnimapkFile(url: fileURL)
        if optimization.linearBackend == .aneMultiProcW8 {
            ANEW8MultiProcModelCache.selectNamespace(try ANEW8NativePack.namespace(file: file))
        }
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

struct GenerationEngine {
    let context: MetalContext
    private let factory: any GenerationStageFactory

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

    func generate(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil,
        progress: ProgressCallback? = nil,
        metrics metricsIn: MetricsCollector? = nil,
        optimization: InferenceOptimizationConfig = .currentBaseline,
        resolution: GenerationResolution = .square512
    ) async throws -> DecodedRGBA8 {
        try await GenerationGeometryRuntime.$current.withValue(resolution) {
            try await generateConfigured(
                prompt: prompt, seed: seed, models: models, noise: noise,
                progress: progress, metrics: metricsIn, optimization: optimization)
        }
    }

    private func generateConfigured(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer?,
        progress: ProgressCallback?,
        metrics metricsIn: MetricsCollector?,
        optimization: InferenceOptimizationConfig
    ) async throws -> DecodedRGBA8 {
        let metrics = metricsIn ?? MetricsCollector()
        metrics.recordOptimizationConfig(optimization)
        metrics.recordResolution(GenerationGeometryRuntime.current)
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
        if let sampler = sampler as? DiffusionSampler {
            sampler.metrics = metrics
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
