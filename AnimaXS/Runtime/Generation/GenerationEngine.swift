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

/// Optional negative-prompt input captured into the same immutable generation
/// snapshot as the model/LoRA choices. This is NOT CFG: GenerationEngine folds
/// it into the one positive conditioning stream and emits a V-only NegPiP sign
/// mask, keeping Anima-Turbo at one DiT evaluation per sigma step.
struct ResolvedNegPiP: Equatable {
    let prompt: String
}

/// The dual-tokenized conditioning contract consumed by the production text
/// stages. `negPiPValueSigns == nil` is the exact historical baseline path.
/// When present it is always 512 rows: +1 for ordinary/padding/EOS rows and -1
/// for negative-weight rows from inline syntax or the user-facing negative box.
struct CompiledPromptConditioning: Equatable {
    let qwenIDs: [Int]
    let t5IDs: [Int]
    let t5Weights: [Float]
    let negPiPValueSigns: [Float]?
}

/// One syntax-stripped prompt segment and its accumulated ComfyUI-compatible
/// weight. Negative weight sign is split from magnitude only after T5 tokenization.
struct PromptWeightSegment: Equatable {
    let text: String
    var weight: Float
}

/// Production model resolution: exactly three base packs (K002 §5.1), plus
/// optional generation-local overlays/options.
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
    let negPiP: ResolvedNegPiP?

    init(
        textEncoder: ResolvedModelPack,
        dit: ResolvedModelPack,
        vae: ResolvedModelPack,
        lora: ResolvedLoRA? = nil,
        negPiP: ResolvedNegPiP? = nil
    ) {
        self.textEncoder = textEncoder
        self.dit = dit
        self.vae = vae
        self.lora = lora
        self.negPiP = negPiP
    }

    /// Convenience init from raw URLs and variant descriptors.
    init(textEncoderURL: URL, textEncoderVariant: ModelVariantDescriptor,
         ditURL: URL, ditVariant: ModelVariantDescriptor,
         vaeURL: URL, vaeVariant: ModelVariantDescriptor,
         lora: ResolvedLoRA? = nil,
         negPiP: ResolvedNegPiP? = nil) {
        self.textEncoder = ResolvedModelPack(url: textEncoderURL, component: .textEncoder, variant: textEncoderVariant)
        self.dit = ResolvedModelPack(url: ditURL, component: .dit, variant: ditVariant)
        self.vae = ResolvedModelPack(url: vaeURL, component: .vae, variant: vaeVariant)
        self.lora = lora
        self.negPiP = negPiP
    }

    func withLoRA(_ lora: ResolvedLoRA?) -> ResolvedModels {
        ResolvedModels(
            textEncoder: textEncoder, dit: dit, vae: vae,
            lora: lora, negPiP: negPiP)
    }

    func withNegPiP(_ negPiP: ResolvedNegPiP?) -> ResolvedModels {
        ResolvedModels(
            textEncoder: textEncoder, dit: dit, vae: vae,
            lora: lora, negPiP: negPiP)
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
        let conditioning = try measuredSync(.tokenizing, metrics: metrics) {
            try Self.compileConditioning(
                positive: prompt,
                negative: models.negPiP?.prompt ?? "")
        }

        progress?(.encodingPrompt)
        try Task.checkCancellation()
        let qwenOutput = try await measured(.textEncode, metrics: metrics) {
            try await encodePrompt(models: models, tokenIDs: conditioning.qwenIDs)
        }

        progress?(.adapting)
        try Task.checkCancellation()
        let cross = try await measured(.adapter, metrics: metrics) {
            try await adaptPrompt(
                models: models, qwenContext: qwenOutput,
                contextTokens: conditioning.qwenIDs.count,
                t5IDs: conditioning.t5IDs,
                t5Weights: conditioning.t5Weights)
        }

        let totalSteps = ModelConstants.samplerSteps
        let initialLatent = try makeInitialLatent(seed: seed, noise: noise)
        let finalLatent = try await measured(.diffusion, metrics: metrics) {
            try await diffuse(
                models: models, initialLatent: initialLatent, cross: cross,
                negPiPValueSigns: conditioning.negPiPValueSigns,
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

    // MARK: - Conditioning

    /// ComfyUI-compatible parser mirrored from Anima WebGPU:
    /// `(x)` = ×1.1, `(x:w)` = ×w, nesting multiplies, and `\(`/`\)` escape
    /// literal parentheses. Syntax is removed before Qwen sees the caption.
    static func parsePromptWeights(_ text: String) -> [PromptWeightSegment] {
        var output: [PromptWeightSegment] = []
        var starts: [Int] = []
        var buffer = ""
        let characters = Array(text)

        func flush() {
            guard !buffer.isEmpty else { return }
            output.append(PromptWeightSegment(text: buffer, weight: 1))
            buffer = ""
        }

        var index = 0
        while index < characters.count {
            let character = characters[index]
            if character == "\\", index + 1 < characters.count,
               characters[index + 1] == "(" || characters[index + 1] == ")" {
                buffer.append(characters[index + 1])
                index += 2
                continue
            }
            if character == "(" {
                flush()
                starts.append(output.count)
                index += 1
                continue
            }
            if character == ")" {
                var weight: Float = 1.1
                if !starts.isEmpty,
                   let explicit = explicitPromptWeight(in: buffer) {
                    buffer = explicit.text
                    weight = explicit.weight
                }
                flush()
                if let start = starts.popLast(), start < output.count {
                    for segmentIndex in start..<output.count {
                        output[segmentIndex].weight *= weight
                    }
                }
                index += 1
                continue
            }
            buffer.append(character)
            index += 1
        }
        flush()
        return output
    }

    /// Compiles the two-box UI and inline weighting syntax into Anima's one
    /// CFG=1 conditioning stream. A plain positive prompt with an empty
    /// negative box deliberately executes the exact historical tokenizer path.
    /// Otherwise Qwen gets syntax-stripped text, T5 is tokenized per weighted
    /// segment, magnitude is applied at the adapter output, and only the sign
    /// is carried to the cross-attention V-only NegPiP boundary.
    static func compileConditioning(
        positive: String,
        negative: String
    ) throws -> CompiledPromptConditioning {
        let qwenTokenizer = try TokenizerLoader.qwen()
        let t5Tokenizer = try TokenizerLoader.t5()
        let hasNegative = !negative.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let positiveSegments = parsePromptWeights(positive)
        let positiveClean = positiveSegments.map(\.text).joined()
        let positiveUsesWeightSyntax = positiveClean != positive ||
            positiveSegments.contains { $0.weight != 1 }

        // Feature-off path: preserve the exact pre-NegPiP recipe and golden IDs.
        if !hasNegative && !positiveUsesWeightSyntax {
            let qwen = qwenTokenizer.encode(text: positive, addSpecialTokens: false)
            guard !qwen.isEmpty else {
                throw GenerationError.tokenizer("Qwen tokenizer produced no tokens")
            }
            let t5 = t5Tokenizer.encode(text: positive, addSpecialTokens: false) + [1]
            return CompiledPromptConditioning(
                qwenIDs: qwen,
                t5IDs: t5,
                t5Weights: [Float](repeating: 1.0, count: t5.count),
                negPiPValueSigns: nil)
        }

        let weightedSource: String
        if hasNegative {
            let separator: String
            if positive.last?.isWhitespace == true || positive.hasSuffix(",") {
                separator = " "
            } else {
                separator = ", "
            }
            // Match the public Anima NegPiP recipe: the whole negative box is
            // one outer -1 group, so inner `(tag:1.2)` naturally becomes -1.2.
            weightedSource = positive + separator + "(" + negative + ":-1)"
        } else {
            weightedSource = positive
        }

        let segments = parsePromptWeights(weightedSource)
        guard segments.allSatisfy({ $0.weight.isFinite }) else {
            throw GenerationError.tokenizer("Prompt weight must be a finite number")
        }
        let cleanCaption = segments.map(\.text).joined()
        let qwen = qwenTokenizer.encode(text: cleanCaption, addSpecialTokens: false)
        guard !qwen.isEmpty else {
            throw GenerationError.tokenizer("Qwen tokenizer produced no tokens")
        }
        guard qwen.count <= QwenEncoderMetal.maximumTokens else {
            throw GenerationError.tokenizer(
                "Weighted positive + negative prompt is too long for Qwen (\(qwen.count)/\(QwenEncoderMetal.maximumTokens) tokens).")
        }

        // Reference behavior tokenizes T5 independently per weighted segment,
        // removes each segment's implicit special handling, then appends EOS once.
        var t5IDs: [Int] = []
        var t5Weights: [Float] = []
        var rowSigns: [Float] = []
        for segment in segments where !segment.text.isEmpty {
            let ids = t5Tokenizer.encode(text: segment.text, addSpecialTokens: false)
            let magnitude = abs(segment.weight)
            let sign: Float = segment.weight < 0 ? -1 : 1
            t5IDs.append(contentsOf: ids)
            t5Weights.append(contentsOf: repeatElement(magnitude, count: ids.count))
            rowSigns.append(contentsOf: repeatElement(sign, count: ids.count))
        }
        t5IDs.append(1)
        t5Weights.append(1)
        rowSigns.append(1)

        guard t5IDs.count <= LLMAdapterMetal.maximumTokens else {
            throw GenerationError.tokenizer(
                "Weighted positive + negative prompt is too long for the Anima adapter (\(t5IDs.count)/\(LLMAdapterMetal.maximumTokens) tokens).")
        }

        let negPiPValueSigns: [Float]?
        if rowSigns.contains(-1) {
            negPiPValueSigns = rowSigns + [Float](
                repeating: 1, count: LLMAdapterMetal.maximumTokens - rowSigns.count)
        } else {
            negPiPValueSigns = nil
        }

        return CompiledPromptConditioning(
            qwenIDs: qwen,
            t5IDs: t5IDs,
            t5Weights: t5Weights,
            negPiPValueSigns: negPiPValueSigns)
    }

    private static func explicitPromptWeight(in buffer: String) -> (text: String, weight: Float)? {
        let pattern = #"^([\s\S]*?):([+-]?\d+(?:\.\d+)?)\s*$"#
        guard let expression = try? NSRegularExpression(pattern: pattern),
              let match = expression.firstMatch(
                in: buffer, range: NSRange(buffer.startIndex..., in: buffer)),
              match.numberOfRanges == 3,
              let textRange = Range(match.range(at: 1), in: buffer),
              let weightRange = Range(match.range(at: 2), in: buffer),
              let weight = Float(buffer[weightRange]) else {
            return nil
        }
        return (String(buffer[textRange]), weight)
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
        negPiPValueSigns: [Float]?,
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

        func executeSampler() async throws {
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
        }

        // Task-local binding means only this diffusion call observes the mask.
        // Empty-negative/all-positive paths do not create a scope at all.
        if let negPiPScope = try NegPiPGenerationContext.make(signs: negPiPValueSigns) {
            try await NegPiPGenerationContext.$active.withValue(negPiPScope) {
                try await executeSampler()
            }
        } else {
            try await executeSampler()
        }
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
