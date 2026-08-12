import Foundation
import Metal
import UIKit

/// Progress state for image generation (K002).
enum GenerationState: Equatable {
    case idle
    case tokenizing
    case encodingPrompt
    case adapting
    case diffusing(step: Int, total: Int)
    case decoding
    case completed
    case failed(String)
}

/// Production image generation orchestrator (K002).
///
/// Owns the full inference pipeline:
///   prompt → tokenizers → Qwen → adapter → diffusion → VAE → DecodedRGBA8 → UIImage
///
/// Stage-scoped object lifetime: heavy model objects (Qwen, adapter, sampler,
/// VAE + their AnimapkFile mmaps) are created inside `generate` and go out of
/// scope when the method returns, making them eligible for release. The
/// coordinator retains only the final `UIImage`.
///
/// This coordinator does NOT duplicate model mathematics — it wires together
/// the existing Metal executors.
@MainActor
final class GenerationCoordinator: ObservableObject {
    @Published private(set) var state: GenerationState = .idle
    @Published private(set) var image: UIImage?

    private let context: MetalContext

    init(context: MetalContext? = nil) {
        if let context {
            self.context = context
        } else if let fallback = MetalContext() {
            self.context = fallback
        } else {
            fatalError("Metal device unavailable")
        }
    }

    /// Generate an image from a text prompt.
    /// - Parameters:
    ///   - prompt: Text prompt for image generation.
    ///   - modelURLs: URLs to the four model packs.
    ///   - noise: Optional pre-generated initial noise (for reproducible tests).
    ///            If nil, generates fresh noise from a random seed.
    func generate(
        prompt: String,
        modelURLs: ModelURLs,
        noise: MTLBuffer? = nil
    ) async {
        // Don't start if already generating.
        switch state {
        case .tokenizing, .encodingPrompt, .adapting,
             .diffusing, .decoding:
            return
        default:
            break
        }
        do {
            try Task.checkCancellation()
            state = .tokenizing

            // ---- 1. Tokenization (production TokenizerLoader semantics) ----
            // Qwen: encode(prompt, no specials) — no start/end token.
            // T5:   encode(prompt, no specials) + [1] (trailing </s> EOS).
            // t5Weights: all 1.0 (verified from case1 fixture JSON).
            let qwenTokenizer = try TokenizerLoader.qwen()
            let qwenTokenIDs = qwenTokenizer.encode(text: prompt, addSpecialTokens: false)
            guard !qwenTokenIDs.isEmpty else {
                throw GenerationError.tokenizer("Qwen tokenizer produced no tokens")
            }
            let t5Tokenizer = try TokenizerLoader.t5()
            let t5IDs = t5Tokenizer.encode(text: prompt, addSpecialTokens: false) + [1]
            let t5Weights = [Float](repeating: 1.0, count: t5IDs.count)

            try Task.checkCancellation()
            state = .encodingPrompt

            // ---- 2. Qwen text encoding ----
            // QwenEncoderMetal is stage-scoped: created, used, released.
            let qwenFile = try AnimapkFile(url: modelURLs.qwen)
            let qwen = try QwenEncoderMetal(context: context, file: qwenFile)
            let qwenOutput = context.device.makeBuffer(
                length: qwenTokenIDs.count * QwenEncoderMetal.hidden * 4,
                options: .storageModeShared)!
            try await qwen.execute(
                tokenIDs: qwenTokenIDs, output: qwenOutput, layerCompleted: nil)

            try Task.checkCancellation()
            state = .adapting

            // ---- 3. Adapter → crossContext ----
            let adapterFile = try AnimapkFile(url: modelURLs.adapter)
            let adapter = try LLMAdapterMetal(context: context, file: adapterFile)
            let cross = context.device.makeBuffer(
                length: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden * 4,
                options: .storageModeShared)!
            try await adapter.execute(
                qwenContext: qwenOutput, contextTokens: qwenTokenIDs.count,
                t5IDs: t5IDs, t5Weights: t5Weights, output: cross, layerCompleted: nil)
            // qwen, adapter, qwenFile, adapterFile go out of scope below.

            try Task.checkCancellation()
            let totalSteps = 8
            state = .diffusing(step: 0, total: totalSteps)

            // ---- 4. Diffusion: noise → final latent ----
            let ditFile = try AnimapkFile(url: modelURLs.dit)
            let sampler = try DiffusionSampler(context: context, file: ditFile)
            let initialLatent = noise ?? makeRandomNoise(on: context.device)
            let finalLatent = context.device.makeBuffer(
                length: DiffusionSampler.latentElements * 4,
                options: .storageModeShared)!
            try await sampler.execute(
                initialLatent: initialLatent, crossContext: cross,
                outputLatent: finalLatent,
                blockProgress: nil,
                stepCompleted: { step, _, _, _, _ in
                    Task { @MainActor in
                        self.state = .diffusing(step: step + 1, total: totalSteps)
                    }
                })
            // sampler, ditFile go out of scope below.

            try Task.checkCancellation()
            state = .decoding

            // ---- 5. VAE decode → DecodedRGBA8 → UIImage ----
            let vaeFile = try AnimapkFile(url: modelURLs.vae)
            let vae = try VAEDecoder(context: context, file: vaeFile)
            let decoded = try await vae.decode(latent: finalLatent)
            // vae, vaeFile go out of scope below.

            // ---- 6. UIImage from decoded RGBA8 ----
            let uiImage = makeUIImage(from: decoded)
            self.image = uiImage
            state = .completed
        } catch is CancellationError {
            state = .failed("cancelled")
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    // MARK: - Private helpers

    private func makeRandomNoise(on device: MTLDevice) -> MTLBuffer {
        var rng = SeededRNG(seed: UInt64.random(in: 0..<UInt64.max))
        let count = DiffusionSampler.latentElements
        let buffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { pointer[i] = rng.nextNormal() }
        return buffer
    }

    private func makeUIImage(from decoded: DecodedRGBA8) -> UIImage? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(decoded.bytes) as CFData),
              let cgImage = CGImage(
                width: decoded.width, height: decoded.height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: decoded.width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

/// URLs for the four model packs required for inference.
struct ModelURLs {
    let qwen: URL
    let adapter: URL
    let dit: URL
    let vae: URL
}

/// Stage-specific errors for generation.
enum GenerationError: Error, LocalizedError {
    case tokenizer(String)
    case qwenPack(String)
    case adapterPack(String)
    case diffusionPack(String)
    case vaePack(String)
    case metal(String)

    var errorDescription: String? {
        switch self {
        case .tokenizer(let m): return "Tokenizer: \(m)"
        case .qwenPack(let m): return "Qwen pack: \(m)"
        case .adapterPack(let m): return "Adapter pack: \(m)"
        case .diffusionPack(let m): return "Diffusion pack: \(m)"
        case .vaePack(let m): return "VAE pack: \(m)"
        case .metal(let m): return "Metal: \(m)"
        }
    }
}
