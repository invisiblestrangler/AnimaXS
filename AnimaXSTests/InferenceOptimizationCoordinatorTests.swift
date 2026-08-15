import XCTest
import Metal
@testable import AnimaXS

/// Coordinator integration for the runtime optimization config (§18.7):
/// the immutable snapshot must be forwarded unchanged, and the checkpoint
/// path must be independent of the optimization settings.
final class InferenceOptimizationCoordinatorTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-optcoord-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        GenerationCoordinator.defaultCheckpointStoreDirectoryOverride = dir
    }

    override func tearDownWithError() throws {
        GenerationCoordinator.defaultCheckpointStoreDirectoryOverride = nil
        try super.tearDownWithError()
    }

    private func makeContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        return context
    }

    private func testModels() -> ResolvedModels {
        ResolvedModels(
            textEncoderURL: URL(fileURLWithPath: "/tmp/qwen.animapk"),
            textEncoderVariant: ModelVariantDescriptor(
                id: "textEncoder", displayFilename: "qwen3-0.6b-xsmax-w8.animapk",
                size: 635_305_984,
                sha256: "ba59e4d1797de5f6512aeafcecf3f38e1f62a47313a2a400b949c9018d84ceab"),
            ditURL: URL(fileURLWithPath: "/tmp/dit.animapk"),
            ditVariant: ModelManifest.ditW4,
            vaeURL: URL(fileURLWithPath: "/tmp/vae.animapk"),
            vaeVariant: ModelVariantDescriptor(
                id: "vae", displayFilename: "qwen-image-vae-xsmax-fp16.animapk",
                size: 256_163_840,
                sha256: "10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447"))
    }

    /// Captures the optimization config handed to the diffusion stage.
    private final class CapturingFactory: GenerationStageFactory {
        private(set) var capturedConfig: InferenceOptimizationConfig?
        func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
            ProbeEncoder()
        }
        func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
            ProbeAdapter()
        }
        func makeDiffusion(
            context: MetalContext, fileURL: URL,
            optimization: InferenceOptimizationConfig
        ) throws -> DiffusionStage {
            capturedConfig = optimization
            return ProbeSampler()
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }

    private final class ProbeEncoder: PromptEncoderStage {
        func execute(tokenIDs: [Int], output: MTLBuffer,
                     layerCompleted: ((Int, MTLBuffer) throws -> Void)?) async throws {}
    }
    private final class ProbeAdapter: ContextAdapterStage {
        func execute(qwenContext: MTLBuffer, contextTokens: Int, t5IDs: [Int],
                     t5Weights: [Float], output: MTLBuffer,
                     layerCompleted: ((Int, MTLBuffer) throws -> Void)?) async throws {}
    }
    private final class ProbeSampler: DiffusionStage {
        func execute(initialLatent: MTLBuffer, crossContext: MTLBuffer,
                     outputLatent: MTLBuffer, startStep: Int,
                     blockProgress: ((Int, Int) throws -> Void)?,
                     stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?) async throws {}
    }
    private final class ProbeVAE: VAEDecodeStage {
        func decode(latent: MTLBuffer) async throws -> DecodedRGBA8 {
            DecodedRGBA8(width: 512, height: 512, bytes: [UInt8](repeating: 255, count: 512 * 512 * 4))
        }
    }

    /// The immutable config snapshot passed to `generate` must reach the
    /// diffusion stage factory unchanged.
    @MainActor
    func testConfigSnapshotForwardedUnchanged() async throws {
        let context = try makeContext()
        let factory = CapturingFactory()
        let coordinator = GenerationCoordinator(
            context: context, factory: factory,
            attemptMetalFallback: false, checkpointStore: nil)
        var config = InferenceOptimizationConfig.currentBaseline
        config.linearTileRows = 1024
        config.attentionTileRows = 512
        config.directLinearMPSIO = true
        config.pingPongWeightStreaming = false
        config.numericalMonitoring = false

        coordinator.generate(prompt: "p", seed: 1, models: testModels(), optimization: config)
        // Wait for the engine to construct the diffusion stage.
        for _ in 0..<250 {
            if factory.capturedConfig != nil { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        let captured = try XCTUnwrap(factory.capturedConfig)
        XCTAssertEqual(captured, config, "the engine must forward the exact immutable snapshot")
    }

    /// Production W4 still persists checkpoints exactly as current HEAD: a
    /// PARTIAL run (cancelled after step 0) retains its checkpoint.
    @MainActor
    func testProductionW4KeepsCheckpointPath() async throws {
        let context = try makeContext()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-w4cp-\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let factory = SuspendSamplerFactory()
        let coordinator = GenerationCoordinator(
            context: context, factory: factory,
            attemptMetalFallback: false, checkpointStore: store)

        coordinator.generate(prompt: "p", seed: 1, models: testModels())
        // Wait for the sampler to complete step 0 and fire the checkpoint.
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        coordinator.cancel()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        // A partial W4 run retains its checkpoint (exactly the D206 behavior).
        XCTAssertTrue(store.hasCheckpoint,
                      "production W4 partial run must retain the checkpoint path")
    }

    /// Checkpointing is not config-dependent: any run (including one with
    /// non-baseline optimization settings) persists a partial-run checkpoint.
    @MainActor
    func testNonBaselineConfigStillPersistsCheckpoint() async throws {
        let context = try makeContext()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-optcp-\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let factory = SuspendSamplerFactory()
        let coordinator = GenerationCoordinator(
            context: context, factory: factory,
            attemptMetalFallback: false, checkpointStore: store)
        var config = InferenceOptimizationConfig.currentBaseline
        config.linearTileRows = 1024
        config.numericalMonitoring = false

        coordinator.generate(prompt: "p", seed: 1, models: testModels(), optimization: config)
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        coordinator.cancel()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        // The DiT slot holds one verified pack; checkpointing is always on.
        XCTAssertTrue(store.hasCheckpoint,
                      "non-baseline optimization config must still persist the checkpoint")
    }

    /// Fires one step-completed callback then suspends until cancelled, so the
    /// checkpoint path runs but the generation stays partial (mirrors the
    /// existing LifecycleSampler pattern in GenerationCoordinatorTests).
    private final class SuspendSampler: DiffusionStage {
        private(set) var completedSteps = 0
        func execute(initialLatent: MTLBuffer, crossContext: MTLBuffer,
                     outputLatent: MTLBuffer, startStep: Int,
                     blockProgress: ((Int, Int) throws -> Void)?,
                     stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?) async throws {
            let count = initialLatent.length / 4
            let out = outputLatent.contents().bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { out[i] = 1 }
            completedSteps += 1
            try stepCompleted?(0, 1.0, 0.5, outputLatent, outputLatent)
            // Deterministic pause: suspend until the coordinator cancels.
            try await Task.sleep(nanoseconds: 60_000_000_000)
        }
    }

    private final class SuspendSamplerFactory: GenerationStageFactory {
        private(set) var sampler: SuspendSampler?
        func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
            ProbeEncoder()
        }
        func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
            ProbeAdapter()
        }
        func makeDiffusion(
            context: MetalContext, fileURL: URL,
            optimization: InferenceOptimizationConfig
        ) throws -> DiffusionStage {
            let sampler = SuspendSampler()
            self.sampler = sampler
            return sampler
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }
}
