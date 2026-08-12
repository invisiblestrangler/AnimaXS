import Foundation
import Metal
import UIKit

/// Progress state for image generation (K002/K003).
enum GenerationState: Equatable {
    case idle
    case tokenizing
    case encodingPrompt
    case adapting
    case diffusing(step: Int, block: Int, totalSteps: Int, totalBlocks: Int)
    case decoding
    case completed
    case cancelled
    case failed(String)
}

/// Errors surfaced by the production generation pipeline (K002 §5.5: no
/// `fatalError` for normal runtime environment failures).
enum GenerationError: Error, LocalizedError {
    case tokenizer(String)
    case qwenPack(String)
    case adapterPack(String)
    case diffusionPack(String)
    case vaePack(String)
    case metal(String)
    case sampler(String)
    case models(String)

    var errorDescription: String? {
        switch self {
        case .tokenizer(let m): return "Tokenizer: \(m)"
        case .qwenPack(let m): return "Qwen pack: \(m)"
        case .adapterPack(let m): return "Adapter pack: \(m)"
        case .diffusionPack(let m): return "Diffusion pack: \(m)"
        case .vaePack(let m): return "VAE pack: \(m)"
        case .metal(let m): return "Metal: \(m)"
        case .sampler(let m): return "Sampler: \(m)"
        case .models(let m): return "Models: \(m)"
        }
    }
}

/// MainActor-facing generation view model (K002 §5.4).
///
/// Holds observable UI state; all heavy pipeline work happens in
/// `GenerationEngine` off the main actor. The coordinator:
/// - guarantees one generation at a time;
/// - forwards the visible seed into production `SeededRNG`;
/// - reports diffusion step AND block progress to the UI;
/// - exposes cooperative cancellation (K003 core);
/// - keeps the last successful image across failed runs.
@MainActor
final class GenerationCoordinator: ObservableObject {
    @Published private(set) var state: GenerationState = .idle
    @Published private(set) var image: UIImage?

    private let context: MetalContext?
    private let factory: any GenerationStageFactory
    private var generationTask: Task<Void, Never>?
    private var latestCheckpoint: (step: Int, latent: [Float])?

    init(
        context: MetalContext? = nil,
        factory: any GenerationStageFactory = ProductionStageFactory(),
        attemptMetalFallback: Bool = true
    ) {
        self.factory = factory
        if let context {
            self.context = context
        } else if attemptMetalFallback {
            // Recoverable: no Metal is a user-visible state, not a crash.
            self.context = MetalContext()
        } else {
            // Test seam: simulate an environment with no Metal device.
            self.context = nil
        }
    }

    deinit {
        generationTask?.cancel()
    }

    var isGenerating: Bool {
        switch state {
        case .tokenizing, .encodingPrompt, .adapting, .diffusing, .decoding:
            return true
        default:
            return false
        }
    }

    var canResume: Bool {
        if case .idle = state, latestCheckpoint != nil { return true }
        if case .cancelled = state, latestCheckpoint != nil { return true }
        return false
    }

    var completedSteps: Int? {
        latestCheckpoint?.step
    }

    /// Starts one generation. Does nothing when a generation is already
    /// running (K002: only one generation may execute at once).
    func generate(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil
    ) {
        guard !isGenerating else { return }
        guard let context else {
            state = .failed(GenerationError.metal("Metal device unavailable").localizedDescription)
            return
        }
        // A fresh generation supersedes any previous checkpoint.
        latestCheckpoint = nil
        // Enter the generating state synchronously so a second `generate`
        // call (even on the same runloop tick) is rejected.
        state = .tokenizing
        run(engine: GenerationEngine(context: context, factory: factory),
            prompt: prompt, seed: seed, models: models,
            noise: noise, startStep: 0)
    }

    /// Resumes from the latest completed diffusion step checkpoint.
    /// Reconstructs conditioning (tokenization + Qwen + adapter), then starts
    /// diffusion at `checkpoint.step`.
    func resume(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil
    ) {
        guard !isGenerating, let checkpoint = latestCheckpoint else { return }
        guard let context else {
            state = .failed(GenerationError.metal("Metal device unavailable").localizedDescription)
            return
        }
        let buffer = context.device.makeBuffer(
            length: checkpoint.latent.count * 4, options: .storageModeShared)!
        buffer.contents().copyMemory(
            from: checkpoint.latent, byteCount: checkpoint.latent.count * 4)
        state = .tokenizing
        run(engine: GenerationEngine(context: context, factory: factory),
            prompt: prompt, seed: seed, models: models,
            noise: noise ?? buffer, startStep: checkpoint.step)
    }

    /// Cooperative cancellation (K003 core): the engine stops at the next safe
    /// boundary; the last completed diffusion step checkpoint is retained.
    func cancel() {
        guard isGenerating else { return }
        generationTask?.cancel()
    }

    /// Drops the retained checkpoint (e.g. incompatible models after resume).
    func discardCheckpoint() {
        latestCheckpoint = nil
        if case .cancelled = state { state = .idle }
    }

    // MARK: - Private

    private func run(
        engine: GenerationEngine,
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer?,
        startStep: Int
    ) {
        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                let decoded = try await engine.generate(
                    prompt: prompt,
                    seed: seed,
                    models: models,
                    noise: noise,
                    startStep: startStep,
                    progress: { stage in
                        let snapshot = GenerationState.from(stage)
                        Task { @MainActor [weak self] in
                            self?.state = snapshot
                        }
                    },
                    checkpoint: { step, latent in
                        let values = latent
                        Task { @MainActor [weak self] in
                            self?.latestCheckpoint = (step + 1, values)
                        }
                    })
                guard !Task.isCancelled else {
                    self.state = .cancelled
                    return
                }
                self.image = GenerationCoordinator.makeUIImage(from: decoded)
                self.state = .completed
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    private static func makeUIImage(from decoded: DecodedRGBA8) -> UIImage? {
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

extension GenerationState {
    /// Maps engine stage progress onto the observable UI state.
    static func from(_ stage: GenerationEngine.GenerationStage) -> GenerationState {
        switch stage {
        case .tokenizing: return .tokenizing
        case .encodingPrompt: return .encodingPrompt
        case .adapting: return .adapting
        case .diffusing(let step, let block, let totalSteps, let totalBlocks):
            return .diffusing(
                step: step, block: block, totalSteps: totalSteps, totalBlocks: totalBlocks)
        case .decoding: return .decoding
        }
    }
}
