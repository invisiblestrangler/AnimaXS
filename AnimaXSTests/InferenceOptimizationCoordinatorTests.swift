import XCTest
import Metal
@testable import AnimaXS

/// Coordinator integration for the runtime optimization config (§18.7):
/// the immutable snapshot must be forwarded unchanged, and experimental W8
/// must disable checkpoint persistence while production W4 keeps the current
/// checkpoint path.
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
            textEncoder: URL(fileURLWithPath: "/tmp/qwen.animapk"),
            dit: URL(fileURLWithPath: "/tmp/dit.animapk"),
            vae: URL(fileURLWithPath: "/tmp/vae.animapk"))
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

    /// A W8 generation never writes a checkpoint (resume must stay off even
    /// though the sampler would normally save at each step). Use a sampler
    /// that fires the step-completed callback so the checkpoint path runs.
    @MainActor
    func testExperimentalW8DisablesCheckpointPersistence() async throws {
        let context = try makeContext()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-w8cp-\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let factory = StepSamplerFactory()
        let coordinator = GenerationCoordinator(
            context: context, factory: factory,
            attemptMetalFallback: false, checkpointStore: store)
        var config = InferenceOptimizationConfig.currentBaseline
        config.ditPackVariant = .experimentalW8V2

        coordinator.generate(prompt: "p", seed: 1, models: testModels(), optimization: config)
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        // No checkpoint written for the experimental W8 run.
        XCTAssertFalse(store.hasCheckpoint,
                       "experimental W8 must never persist a checkpoint")
        XCTAssertNil(coordinator.completedSteps)
    }

    /// Production W4 still persists checkpoints exactly as current HEAD.
    @MainActor
    func testProductionW4KeepsCheckpointPath() async throws {
        let context = try makeContext()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-w4cp-\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let factory = StepSamplerFactory()
        let coordinator = GenerationCoordinator(
            context: context, factory: factory,
            attemptMetalFallback: false, checkpointStore: store)

        coordinator.generate(prompt: "p", seed: 1, models: testModels())
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        // Cancelled after step 0 → a partial checkpoint was saved (as in the
        // existing D206 behavior for a partial run).
        XCTAssertTrue(store.hasCheckpoint,
                      "production W4 must retain the current checkpoint path")
    }

    /// Fires one step-completed callback then completes.
    private final class StepSampler: DiffusionStage {
        func execute(initialLatent: MTLBuffer, crossContext: MTLBuffer,
                     outputLatent: MTLBuffer, startStep: Int,
                     blockProgress: ((Int, Int) throws -> Void)?,
                     stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?) async throws {
            let count = initialLatent.length / 4
            let out = outputLatent.contents().bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { out[i] = 1 }
            try stepCompleted?(0, 1.0, 0.5, outputLatent, outputLatent)
        }
    }

    private final class StepSamplerFactory: GenerationStageFactory {
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
            StepSampler()
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }
}
