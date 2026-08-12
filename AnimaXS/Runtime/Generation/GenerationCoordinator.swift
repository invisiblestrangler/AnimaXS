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
/// - persists a full-metadata checkpoint after each completed diffusion step
///   (I004/K003) so cancel/background/memory-warning can resume;
/// - validates checkpoint compatibility (prompt/seed/hashes) before resume;
/// - exposes cooperative cancellation (K003 core);
/// - keeps the last successful image across failed runs.
@MainActor
final class GenerationCoordinator: ObservableObject {
    @Published private(set) var state: GenerationState = .idle
    @Published private(set) var image: UIImage?

    private let context: MetalContext?
    private let factory: any GenerationStageFactory
    private let checkpointStore: CheckpointStore?
    private var generationTask: Task<Void, Never>?
    private var latestCheckpoint: GenerationCheckpoint?

    init(
        context: MetalContext? = nil,
        factory: any GenerationStageFactory = ProductionStageFactory(),
        attemptMetalFallback: Bool = true,
        checkpointStore: CheckpointStore? = nil
    ) {
        self.factory = factory
        self.checkpointStore = checkpointStore
        if let context {
            self.context = context
        } else if attemptMetalFallback {
            // Recoverable: no Metal is a user-visible state, not a crash.
            self.context = MetalContext()
        } else {
            // Test seam: simulate an environment with no Metal device.
            self.context = nil
        }
        // Cold launch: a valid persisted checkpoint is offered for resume.
        if let checkpointStore {
            latestCheckpoint = checkpointStore.load()
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
        guard let checkpoint = latestCheckpoint else { return false }
        switch state {
        case .idle, .cancelled:
            return checkpoint.step >= 1 && checkpoint.step <= ModelConstants.samplerSteps
        default:
            return false
        }
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
        checkpointStore?.remove()
        // Enter the generating state synchronously so a second `generate`
        // call (even on the same runloop tick) is rejected.
        state = .tokenizing
        run(engine: GenerationEngine(context: context, factory: factory),
            prompt: prompt, seed: seed, models: models,
            noise: noise, startStep: 0)
    }

    /// Resumes from the latest completed diffusion step checkpoint.
    /// Reconstructs conditioning (tokenization + Qwen + adapter), then starts
    /// diffusion at `checkpoint.step`. Rejects incompatible checkpoints
    /// (prompt/seed/resolution/model-hash mismatch) as a recoverable error.
    func resume(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil
    ) {
        guard !isGenerating else { return }
        guard let checkpoint = latestCheckpoint else { return }
        guard let context else {
            state = .failed(GenerationError.metal("Metal device unavailable").localizedDescription)
            return
        }
        do {
            let hashes = try ModelManifest.productionHashes()
            let store = try checkpointStore ?? CheckpointStore()
            _ = try store.validate(
                checkpoint, prompt: prompt, seed: seed,
                resolution: (512, 512), modelHashes: hashes)
        } catch {
            // Incompatible or corrupt: drop it and surface a recoverable error.
            discardCheckpoint()
            state = .failed(error.localizedDescription)
            return
        }
        guard let latent = try? checkpoint.latentValues() else {
            discardCheckpoint()
            state = .failed("checkpoint latent is corrupt")
            return
        }
        let buffer = context.device.makeBuffer(
            length: latent.count * 4, options: .storageModeShared)!
        buffer.contents().copyMemory(
            from: latent, byteCount: latent.count * 4)
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
        checkpointStore?.remove()
        if case .cancelled = state { state = .idle }
    }

    // MARK: - App lifecycle (K003)

    /// App moved to background while generating: request safe cancellation,
    /// retain the latest completed-step checkpoint, and release the heavy
    /// generation stage. Do not promise unrestricted background GPU inference.
    func appDidEnterBackground() {
        guard isGenerating else { return }
        // Cooperative cancellation: the engine stops at the next safe block
        // boundary; when the cancel lands, state becomes .cancelled and the
        // checkpoint remains available for Resume.
        cancel()
    }

    /// App returned to foreground: nothing to do — a compatible checkpoint is
    /// already surfaced through `canResume`/`completedSteps`.
    func appWillEnterForeground() {
        // Resume availability is derived from `latestCheckpoint` on demand.
    }

    // MARK: - Resource policy (K004)

    /// A memory warning arrived during generation: cancel at the nearest safe
    /// boundary, preserve the last completed-step checkpoint, and surface a
    /// recoverable message. Never try to "free random buffers" that an active
    /// Metal command still owns.
    func handleMemoryWarning() {
        guard isGenerating else { return }
        cancel()
        // The engine stops cooperatively; the checkpoint is already persisted.
        state = .cancelled
    }

    /// Thermal policy (documented in DECISIONS.md D0xx): nominal/fair →
    /// continue; serious/critical → stop safely and preserve resume state.
    func handleThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal, .fair:
            return // continue generation
        case .serious, .critical:
            if isGenerating {
                cancel()
                self.state = .cancelled
            }
        @unknown default:
            return
        }
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
                    checkpoint: { completedStep, latent in
                        // Persist the full-metadata checkpoint immediately so
                        // cancel/background/memory-warning can always resume
                        // from the last fully completed step (I004 §6.3).
                        let nextStep = completedStep + 1
                        guard let checkpoint = try? GenerationCheckpoint(
                            latent: latent, step: nextStep,
                            prompt: prompt, seed: seed,
                            width: ModelConstants.imageSize,
                            height: ModelConstants.imageSize,
                            modelHashes: try ModelManifest.productionHashes()) else {
                            return
                        }
                        Task { @MainActor [weak self] in
                            self?.latestCheckpoint = checkpoint
                            if let store = self?.checkpointStore {
                                try? store.save(checkpoint)
                            }
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
