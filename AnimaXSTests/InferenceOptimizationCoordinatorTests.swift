import XCTest
import Metal
@testable import AnimaXS

/// Coordinator integration for the runtime optimization config (§18.7):
/// the immutable snapshot must be forwarded unchanged through the
/// coordinator into the diffusion stage factory.
final class InferenceOptimizationCoordinatorTests: XCTestCase {

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
            optimization: InferenceOptimizationConfig,
            numerics: DiTNumericsPolicy
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
            attemptMetalFallback: false)
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
}
