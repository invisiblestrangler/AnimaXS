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
    /// Compact text summary of the most recent generation's telemetry
    /// (Phase 8: readable without a cable).
    @Published private(set) var lastMetricsText: String?

    private let context: MetalContext?
    private let factory: any GenerationStageFactory
    private let checkpointStore: CheckpointStore?
    private var generationTask: Task<Void, Never>?
    private var latestCheckpoint: GenerationCheckpoint?
    /// Monotonic run identifier. Checkpoint-save tasks capture the epoch at
    /// callback time and only apply if the run is still current — a save
    /// queued before completion (or before a new Generate) can never
    /// resurrect a stale checkpoint afterwards.
    private var generationEpoch = 0

    /// Test seam (mirrors `ModelStore.secureInstalls`): when set, the *default*
    /// persistent store (used when a coordinator is wired with no explicit
    /// `checkpointStore`, exactly as production `ContentView` does) is created
    /// under this directory. Production leaves this `nil` so the default store
    /// uses the real persistent Application Support location. Tests that need
    /// to prove the default wiring provides cold-launch recovery set this to an
    /// isolated temp directory and reset it afterward.
    static nonisolated(unsafe) var defaultCheckpointStoreDirectoryOverride: URL?

    @MainActor
    private static func makeDefaultCheckpointStore() -> CheckpointStore? {
        if let override = defaultCheckpointStoreDirectoryOverride {
            return try? CheckpointStore(directory: override)
        }
        return try? CheckpointStore()
    }

    init(
        context: MetalContext? = nil,
        factory: any GenerationStageFactory = ProductionStageFactory(),
        attemptMetalFallback: Bool = true,
        checkpointStore: CheckpointStore? = nil
    ) {
        self.factory = factory
        // Production (default) wiring must persist checkpoints across cold
        // launches so Resume works after the app is killed. `nil` here means
        // "use the normal persistent store"; tests inject an isolated store
        // (or leave nil and rely on `defaultCheckpointStoreDirectoryOverride`
        // for isolation).
        self.checkpointStore = checkpointStore ?? Self.makeDefaultCheckpointStore()
        if let context {
            self.context = context
        } else if attemptMetalFallback {
            // Recoverable: no Metal is a user-visible state, not a crash.
            self.context = MetalContext()
        } else {
            // Test seam: simulate an environment with no Metal device.
            self.context = nil
        }
        // Cold launch: a valid, non-terminal persisted checkpoint is offered
        // for resume. A terminal checkpoint (step == samplerSteps) means the
        // previous diffusion run fully completed — it must be cleared, not
        // retained, so it can never replace the Generate button with a
        // meaningless "8/8 Resume" (and cannot linger as dead cache).
        if let checkpointStore = self.checkpointStore,
           let loaded = checkpointStore.load() {
            if loaded.step >= ModelConstants.samplerSteps {
                checkpointStore.remove()
            } else {
                latestCheckpoint = loaded
            }
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

    /// Whether Metal is available for generation (recoverable, not a crash).
    var isMetalAvailable: Bool {
        context != nil
    }

    var canResume: Bool {
        guard let checkpoint = latestCheckpoint else { return false }
        switch state {
        case .idle, .cancelled:
            // Only a PARTIALLY completed diffusion run is resumable. A
            // checkpoint at step == samplerSteps is terminal (all diffusion
            // steps done) — there is nothing to resume, and offering it would
            // suppress the Generate button with a no-op "8/8 Resume".
            return checkpoint.step >= 1 && checkpoint.step < ModelConstants.samplerSteps
        default:
            return false
        }
    }

    var completedSteps: Int? {
        latestCheckpoint?.step
    }

    /// Starts one generation. Does nothing when a generation is already
    /// running (K002: only one generation may execute at once).
    ///
    /// - Parameter optimization: Immutable per-run inference configuration,
    ///   captured at Generate time. When it selects an experimental W8 pack,
    ///   checkpoint persistence/resume is disabled for the run.
    func generate(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer? = nil,
        optimization: InferenceOptimizationConfig = .currentBaseline
    ) {
        guard !isGenerating else { return }
        guard let context else {
            state = .failed(GenerationError.metal("Metal device unavailable").localizedDescription)
            return
        }
        // A fresh generation supersedes any previous checkpoint.
        latestCheckpoint = nil
        checkpointStore?.remove()
        generationEpoch += 1
        // Enter the generating state synchronously so a second `generate`
        // call (even on the same runloop tick) is rejected.
        state = .tokenizing
        run(engine: GenerationEngine(context: context, factory: factory),
            prompt: prompt, seed: seed, models: models,
            noise: noise, startStep: 0, optimization: optimization)
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
        generationEpoch += 1
        state = .tokenizing
        // Resume is production-W4 only: it reconstructs the same checkpoint
        // state and is not a performance-comparison vehicle (per the runtime
        // experiment runbook, use fresh Generate for benchmarks).
        run(engine: GenerationEngine(context: context, factory: factory),
            prompt: prompt, seed: seed, models: models,
            noise: noise ?? buffer, startStep: checkpoint.step,
            optimization: .currentBaseline)
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
    ///
    /// The engine's natural cancellation path persists the checkpoint (the
    /// step-completed callback enqueues the write on the main actor) and then
    /// transitions to `.cancelled`; we only request the cooperative cancel.
    func handleMemoryWarning() {
        guard isGenerating else { return }
        cancel()
    }

    // MARK: - Private

    private func run(
        engine: GenerationEngine,
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer?,
        startStep: Int,
        optimization: InferenceOptimizationConfig
    ) {
        let metrics = MetricsCollector()
        metrics.recordOptimizationConfig(optimization)
        metrics.recordDiTPackFilename(models.dit.lastPathComponent)
        // Observational environment telemetry (never a generation gate):
        // recorded from the coordinator/UI-safe layer, not inside Metal.
        UIDevice.current.isBatteryMonitoringEnabled = true
        metrics.recordEnvironmentStart(Self.environmentSnapshot())
        generationTask = Task { [weak self] in
            guard let self else { return }
            defer {
                metrics.recordEnvironmentEnd(Self.environmentSnapshot())
                self.publishMetrics(metrics)
            }
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
                    checkpoint: { [weak self] completedStep, latent in
                        // Experimental W8 runs never persist checkpoints: the
                        // production W4 hash set does not describe the W8 pack,
                        // and a W8 checkpoint must not resurrect unrelated
                        // production Resume state. Production W4 keeps the
                        // current-HEAD checkpoint behavior exactly.
                        guard optimization.checkpointingEnabled,
                              let self else { return }
                        // Persist the full-metadata checkpoint immediately so
                        // cancel/background/memory-warning can always resume
                        // from the last fully completed step (I004 §6.3).
                        let nextStep = completedStep + 1
                        // A terminal checkpoint (all steps complete) has no
                        // diffusion left to resume: never retain or persist it,
                        // or it would replace Generate with "8/8 Resume" and
                        // linger as dead cache.
                        guard nextStep < ModelConstants.samplerSteps else { return }
                        guard let checkpoint = try? GenerationCheckpoint(
                            latent: latent, step: nextStep,
                            prompt: prompt, seed: seed,
                            width: ModelConstants.imageSize,
                            height: ModelConstants.imageSize,
                            modelHashes: try ModelManifest.productionHashes()) else {
                            return
                        }
                        // Apply on the main actor, but only while this run is
                        // still current (epoch) and still generating: a save
                        // queued just before completion or a fresh Generate
                        // must never resurrect a stale checkpoint afterwards.
                        let epoch = self.generationEpoch
                        Task { @MainActor [weak self] in
                            guard let self,
                                  self.generationEpoch == epoch,
                                  self.isGenerating else { return }
                            self.latestCheckpoint = checkpoint
                            if let store = self.checkpointStore {
                                try? store.save(checkpoint)
                            }
                        }
                    },
                    metrics: metrics,
                    optimization: optimization)
                guard !Task.isCancelled else {
                    self.state = .cancelled
                    return
                }
                self.image = GenerationCoordinator.makeUIImage(from: decoded)
                self.state = .completed
                // A finished generation has nothing to resume: clear the
                // retained/persisted checkpoint so the next action is a fresh
                // Generate, never a stale "N/8 Resume".
                self.clearCheckpoint()
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    /// Captures power/battery/thermal/low-power facts for the run summary.
    /// Observational only — never gates or throttles generation.
    @MainActor
    private static func environmentSnapshot() -> EnvironmentSnapshot {
        let battery = UIDevice.current
        let batteryLevel: Int
        if battery.batteryState == .unknown || battery.batteryLevel < 0 {
            batteryLevel = -1
        } else {
            batteryLevel = Int((battery.batteryLevel * 100).rounded())
        }
        let powerState: String
        switch battery.batteryState {
        case .charging: powerState = "charging"
        case .full: powerState = "full"
        case .unplugged: powerState = "unplugged"
        default: powerState = "unknown"
        }
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        return EnvironmentSnapshot(
            powerState: powerState,
            batteryLevel: batteryLevel,
            thermalState: thermal,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }

    /// Drops the retained and persisted checkpoint without touching state
    /// (used when a generation completes: nothing is left to resume).
    /// Bumps the epoch so any in-flight checkpoint-save from the just-ended
    /// run is invalidated and can never resurrect the terminal checkpoint.
    private func clearCheckpoint() {
        generationEpoch += 1
        latestCheckpoint = nil
        checkpointStore?.remove()
    }

    /// Publish the run's telemetry summary (works for completed, failed, and
    /// cancelled runs — partial metrics are still useful evidence).
    private func publishMetrics(_ metrics: MetricsCollector) {
        let summary = metrics.snapshot().summaryText
        Task { @MainActor [weak self] in
            self?.lastMetricsText = summary
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
