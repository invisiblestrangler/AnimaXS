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
/// - exposes cooperative cancellation (K003 core) — there is no checkpoint or
///   resume state, so backgrounding / memory warnings just cancel the run;
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
    private var generationTask: Task<Void, Never>?
    /// The reason the most recent cooperative cancel was requested. Telemetry
    /// only — published into final metrics / the cancelled state to distinguish
    /// user-initiated from automatic (background / memory-warning) cancellation.
    private var pendingCancellationReason: GenerationCancellationReason?

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

    /// Whether Metal is available for generation (recoverable, not a crash).
    var isMetalAvailable: Bool {
        context != nil
    }

    /// Starts one generation. Does nothing when a generation is already
    /// running (K002: only one generation may execute at once).
    ///
    /// - Parameter optimization: Immutable per-run inference configuration,
    ///   captured at Generate time.
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
        // Enter the generating state synchronously so a second `generate`
        // call (even on the same runloop tick) is rejected.
        state = .tokenizing
        run(engine: GenerationEngine(context: context, factory: factory),
            prompt: prompt, seed: seed, models: models,
            noise: noise, optimization: optimization)
    }

    /// Cooperative cancellation (K003 core): the engine stops at the next safe
    /// boundary. There is no resume state to preserve. The reason is telemetry
    /// only (user/background/memory-warning/…).
    func cancel(reason: GenerationCancellationReason = .user) {
        guard isGenerating else { return }
        pendingCancellationReason = reason
        generationTask?.cancel()
    }

    // MARK: - App lifecycle (K003)

    /// App moved to background while generating: request cooperative
    /// cancellation at the next safe boundary. The generation run ends in the
    /// `.cancelled` state; a fresh Generate is available afterward. There is
    /// no checkpoint to retain and no resume path.
    func appDidEnterBackground() {
        guard isGenerating else { return }
        cancel(reason: .background)
    }

    /// App returned to foreground: nothing to do — cancellation already
    /// completed; the user starts a fresh Generate if desired.
    func appWillEnterForeground() {}

    // MARK: - Resource policy (K004)

    /// A memory warning arrived during generation: cancel at the nearest safe
    /// boundary and surface a recoverable message. Never try to "free random
    /// buffers" that an active Metal command still owns. There is no
    /// checkpoint state to preserve.
    func handleMemoryWarning() {
        guard isGenerating else { return }
        cancel(reason: .memoryWarning)
    }

    // MARK: - Private

    private func run(
        engine: GenerationEngine,
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        noise: MTLBuffer?,
        optimization: InferenceOptimizationConfig
    ) {
        let metrics = MetricsCollector()
        metrics.recordOptimizationConfig(optimization)
        metrics.recordDiTPackIdentity(
            id: models.dit.variant.id,
            filename: models.dit.variant.displayFilename,
            sha256: models.dit.variant.sha256,
            bytes: models.dit.variant.size)
        // Observational environment telemetry (never a generation gate):
        // recorded from the coordinator/UI-safe layer, not inside Metal.
        UIDevice.current.isBatteryMonitoringEnabled = true
        metrics.recordEnvironmentStart(Self.environmentSnapshot())
        generationTask = Task { [weak self] in
            guard let self else { return }
            defer {
                metrics.recordEnvironmentEnd(Self.environmentSnapshot())
                if let reason = self.pendingCancellationReason {
                    metrics.recordCancellationReason(reason)
                }
                self.publishMetrics(metrics)
            }
            do {
                let decoded = try await engine.generate(
                    prompt: prompt,
                    seed: seed,
                    models: models,
                    noise: noise,
                    progress: { stage in
                        let snapshot = GenerationState.from(stage)
                        Task { @MainActor [weak self] in
                            self?.state = snapshot
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
