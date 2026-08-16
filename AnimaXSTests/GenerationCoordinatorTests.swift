import XCTest
import Metal
@testable import AnimaXS

/// K002 orchestration tests: stage order, seed forwarding, three-URL model
/// topology, block progress, failure recovery, no-Metal behavior, heavy
/// stage lifetime, and cooperative cancellation (background / memory
/// warning) reaching a terminal cancelled state with a fresh Generate
/// available afterward. No 2 GB model packs are required — probe stages
/// substitute for the real executors.
final class GenerationCoordinatorTests: XCTestCase {

    // MARK: - Probe infrastructure

    private protocol DeinitReporting: AnyObject {
        var onDeinit: (() -> Void)? { get set }
    }

    private final class ProbeEncoder: PromptEncoderStage, DeinitReporting {
        var onDeinit: (() -> Void)?
        deinit { onDeinit?() }
        let onExecute: ((Int, MTLBuffer) throws -> Void)?
        var executed = false
        init(onExecute: ((Int, MTLBuffer) throws -> Void)? = nil) {
            self.onExecute = onExecute
        }
        func execute(
            tokenIDs: [Int], output: MTLBuffer,
            layerCompleted: ((Int, MTLBuffer) throws -> Void)?
        ) async throws {
            executed = true
            try onExecute?(tokenIDs.count, output)
        }
    }

    private final class ProbeAdapter: ContextAdapterStage, DeinitReporting {
        var onDeinit: (() -> Void)?
        deinit { onDeinit?() }
        let onExecute: (() throws -> Void)?
        var executed = false
        init(onExecute: (() throws -> Void)? = nil) { self.onExecute = onExecute }
        func execute(
            qwenContext: MTLBuffer, contextTokens: Int,
            t5IDs: [Int], t5Weights: [Float], output: MTLBuffer,
            layerCompleted: ((Int, MTLBuffer) throws -> Void)?
        ) async throws {
            executed = true
            try onExecute?()
        }
    }

    private final class ProbeSampler: DiffusionStage, DeinitReporting {
        var onDeinit: (() -> Void)?
        deinit { onDeinit?() }
        let steps: Int
        let blocks: Int
        var capturedInitialLatent: [Float]?
        var startStepSeen: Int?
        let onExecute: (() throws -> Void)?
        init(
            steps: Int = 8, blocks: Int = 28,
            onExecute: (() throws -> Void)? = nil
        ) {
            self.steps = steps
            self.blocks = blocks
            self.onExecute = onExecute
        }
        func execute(
            initialLatent: MTLBuffer,
            crossContext: MTLBuffer,
            outputLatent: MTLBuffer,
            startStep: Int,
            blockProgress: ((Int, Int) throws -> Void)?,
            stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?
        ) async throws {
            startStepSeen = startStep
            let count = initialLatent.length / 4
            let pointer = initialLatent.contents().bindMemory(to: Float.self, capacity: count)
            capturedInitialLatent = Array(UnsafeBufferPointer(start: pointer, count: count))
            try onExecute?()
            for step in startStep..<steps {
                for block in 0..<blocks {
                    try blockProgress?(step, block)
                }
                // Write a deterministic post-step latent so the VAE stage has
                // finite data.
                let out = outputLatent.contents().bindMemory(to: Float.self, capacity: count)
                for i in 0..<count { out[i] = Float(step + 1) }
                let denoised = outputLatent
                let latent = outputLatent
                try stepCompleted?(step, 1.0, 0.5, denoised, latent)
            }
        }
    }

    private final class ProbeVAE: VAEDecodeStage, DeinitReporting {
        var onDeinit: (() -> Void)?
        deinit { onDeinit?() }
        var executed = false
        func decode(latent: MTLBuffer) async throws -> DecodedRGBA8 {
            executed = true
            let width = 512, height = 512
            var bytes = [UInt8](repeating: 0, count: width * height * 4)
            for i in 0..<(width * height) { bytes[i * 4 + 3] = 255 }
            return DecodedRGBA8(width: width, height: height, bytes: bytes)
        }
    }

    /// Records every factory call (URL per stage) and creates probes.
    private final class ProbeFactory: GenerationStageFactory {
        struct Call { let stage: String; let url: URL }
        var calls: [Call] = []
        let lock = NSLock()
        var encoderOnExecute: ((Int, MTLBuffer) throws -> Void)?
        var adapterOnExecute: (() throws -> Void)?
        var samplerOnExecute: (() throws -> Void)?

        func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
            lock.lock(); calls.append(Call(stage: "qwen", url: fileURL)); lock.unlock()
            return ProbeEncoder(onExecute: encoderOnExecute)
        }
        func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
            lock.lock(); calls.append(Call(stage: "adapter", url: fileURL)); lock.unlock()
            return ProbeAdapter(onExecute: adapterOnExecute)
        }
        func makeDiffusion(
            context: MetalContext, fileURL: URL,
            optimization: InferenceOptimizationConfig,
            numerics: DiTNumericsPolicy
        ) throws -> DiffusionStage {
            lock.lock(); calls.append(Call(stage: "sampler", url: fileURL)); lock.unlock()
            return ProbeSampler(onExecute: samplerOnExecute)
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            lock.lock(); calls.append(Call(stage: "vae", url: fileURL)); lock.unlock()
            return ProbeVAE()
        }
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

    // MARK: - Three-URL production topology

    func testProductionTopologyUsesExactlyThreeURLsAndSharesDiT() async throws {
        let context = try makeContext()
        let factory = ProbeFactory()
        let engine = GenerationEngine(context: context, factory: factory)
        let models = testModels()

        _ = try await engine.generate(prompt: "test", seed: 1, models: models)

        let stages = factory.calls.map(\.stage)
        XCTAssertEqual(stages, ["qwen", "adapter", "sampler", "vae"],
                       "exactly four stage constructions in pipeline order")
        XCTAssertEqual(Set(factory.calls.map(\.url)).count, 3,
                       "production model mapping uses exactly three URLs")
        // Adapter and sampler must both receive the DiT URL.
        let adapter = factory.calls.first { $0.stage == "adapter" }!
        let sampler = factory.calls.first { $0.stage == "sampler" }!
        XCTAssertEqual(adapter.url, models.dit.url, "adapter reads the DiT pack")
        XCTAssertEqual(sampler.url, models.dit.url, "sampler reads the DiT pack")
        XCTAssertEqual(factory.calls.first { $0.stage == "qwen" }!.url, models.textEncoder.url)
        XCTAssertEqual(factory.calls.first { $0.stage == "vae" }!.url, models.vae.url)
    }

    // MARK: - Seed forwarding

    func testSameSeedProducesSameInitialLatent() async throws {
        let latentA = try await captureNoise(seed: 42)
        let latentB = try await captureNoise(seed: 42)
        XCTAssertEqual(latentA, latentB, "same seed must produce identical initialized noise")
    }

    func testDifferentSeedProducesDifferentInitialLatent() async throws {
        let latentA = try await captureNoise(seed: 1)
        let latentB = try await captureNoise(seed: 2)
        XCTAssertNotEqual(latentA, latentB, "different seeds must produce different noise")
    }

    /// Runs the engine with a capturing factory and returns the initial latent
    /// the sampler received (proving the engine forwarded `SeededRNG(seed:)`).
    private func captureNoise(seed: UInt64) async throws -> [Float] {
        let context = try makeContext()
        let capture = CaptureFactory()
        _ = try await GenerationEngine(context: context, factory: capture)
            .generate(prompt: "same prompt", seed: seed, models: testModels())
        return try XCTUnwrap(capture.sampler?.capturedInitialLatent)
    }

    private final class CaptureFactory: GenerationStageFactory {
        var sampler: ProbeSampler?
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
            let sampler = ProbeSampler()
            self.sampler = sampler
            return sampler
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }

    // MARK: - Stage + progress order

    func testStageOrderIsCorrect() async throws {
        let context = try makeContext()
        let factory = ProbeFactory()
        let engine = GenerationEngine(context: context, factory: factory)
        var stages: [String] = []
        _ = try await engine.generate(
            prompt: "test", seed: 1, models: testModels(),
            progress: { stage in
                let name: String
                switch stage {
                case .tokenizing: name = "tokenizing"
                case .encodingPrompt: name = "encodingPrompt"
                case .adapting: name = "adapting"
                case .diffusing: name = "diffusing"
                case .decoding: name = "decoding"
                }
                // Record only stage transitions (progress fires per block).
                if stages.last != name { stages.append(name) }
            })
        XCTAssertEqual(stages, ["tokenizing", "encodingPrompt", "adapting",
                                "diffusing", "decoding"])
    }

    func testBlockProgressIsPropagated() async throws {
        let context = try makeContext()
        let engine = GenerationEngine(context: context, factory: ProbeFactory())
        var maxBlocks = 0
        var diffusingCount = 0
        _ = try await engine.generate(
            prompt: "test", seed: 1, models: testModels(),
            progress: { stage in
                if case .diffusing(_, let block, _, let totalBlocks) = stage {
                    diffusingCount += 1
                    maxBlocks = max(maxBlocks, block)
                    XCTAssertEqual(totalBlocks, 28)
                }
            })
        XCTAssertEqual(maxBlocks, 28, "block progress reaches 28/28")
        XCTAssertEqual(diffusingCount, 8 * 28 + 8,
                       "one progress per block plus one per completed step")
    }

    // MARK: - Recoverable failure at each stage

    func testFailureAtEachStageIsRecoverable() async throws {
        let context = try makeContext()
        let models = testModels()

        // Qwen stage failure.
        let f1 = ProbeFactory()
        f1.encoderOnExecute = { _, _ in throw GenerationError.qwenPack("boom") }
        let e1 = GenerationEngine(context: context, factory: f1)
        do {
            _ = try await e1.generate(prompt: "t", seed: 1, models: models)
            XCTFail("expected qwen failure")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Qwen"))
        }

        // Adapter stage failure.
        let f2 = ProbeFactory()
        f2.adapterOnExecute = { throw GenerationError.adapterPack("boom") }
        let e2 = GenerationEngine(context: context, factory: f2)
        do {
            _ = try await e2.generate(prompt: "t", seed: 1, models: models)
            XCTFail("expected adapter failure")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Adapter"))
        }

        // Sampler stage failure.
        let f3 = ProbeFactory()
        f3.samplerOnExecute = { throw GenerationError.diffusionPack("boom") }
        let e3 = GenerationEngine(context: context, factory: f3)
        do {
            _ = try await e3.generate(prompt: "t", seed: 1, models: models)
            XCTFail("expected sampler failure")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Diffusion"))
        }

        // VAE stage failure.
        let throwing = ThrowingVAEFactory()
        let e4 = GenerationEngine(context: context, factory: throwing)
        do {
            _ = try await e4.generate(prompt: "t", seed: 1, models: models)
            XCTFail("expected vae failure")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("VAE"))
        }

        // After each failure the engine can run again (recoverable).
        let ok = ProbeFactory()
        let engine = GenerationEngine(context: context, factory: ok)
        _ = try await engine.generate(prompt: "t", seed: 1, models: models)
    }

    private final class ThrowingVAEFactory: GenerationStageFactory {
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
            ProbeSampler()
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            throw GenerationError.vaePack("boom")
        }
    }

    // MARK: - No-Metal path

    @MainActor
    func testNoMetalPathDoesNotCrash() {
        let coordinator = GenerationCoordinator(
            context: nil, factory: ProbeFactory(), attemptMetalFallback: false)
        XCTAssertFalse(coordinator.isGenerating)
        coordinator.generate(prompt: "t", seed: 1, models: testModels())
        if case .failed(let message) = coordinator.state {
            XCTAssertTrue(message.contains("Metal"), "expected Metal-unavailable failure, got \(message)")
        } else {
            XCTFail("expected failed state, got \(coordinator.state)")
        }
        XCTAssertFalse(coordinator.isGenerating, "recoverable: not stuck generating")
    }

    // MARK: - Coordinator: one generation at a time

    @MainActor
    func testOnlyOneGenerationAtATime() async throws {
        let context = try makeContext()
        let factory = BlockingFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "t", seed: 1, models: testModels())
        // State entered synchronously; the sampler is now blocked in-flight.
        XCTAssertTrue(coordinator.isGenerating, "generation starts synchronously")

        // A second call while running must be ignored entirely.
        let callsBefore = factory.callCount
        coordinator.generate(prompt: "t", seed: 2, models: testModels())
        XCTAssertEqual(factory.callCount, callsBefore,
                       "second generation ignored while one is running")

        // Wait until the first generation's sampler is actually blocked.
        // Tokenizer loading is reloaded per generate() call and can take ~2s
        // in the simulator, so budget generously.
        for _ in 0..<250 {
            if factory.isAnySamplerBlocked { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(factory.isAnySamplerBlocked, "first sampler reached the blocked stage")
        factory.releaseAll()
        for _ in 0..<200 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating, "generation finished after release")
        if case .completed = coordinator.state {
            // expected
        } else {
            XCTFail("expected completed, got \(coordinator.state)")
        }
    }

    /// Sampler that blocks until released, for determinism in concurrency tests.
    private final class BlockingFactory: GenerationStageFactory {
        struct Call { let stage: String }
        private var calls: [Call] = []
        private let lock = NSLock()
        private var samplers: [BlockingSampler] = []

        var callCount: Int {
            lock.lock(); defer { lock.unlock() }
            return calls.count
        }

        func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
            lock.lock(); calls.append(Call(stage: "qwen")); lock.unlock()
            return ProbeEncoder()
        }
        func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
            lock.lock(); calls.append(Call(stage: "adapter")); lock.unlock()
            return ProbeAdapter()
        }
        func makeDiffusion(
            context: MetalContext, fileURL: URL,
            optimization: InferenceOptimizationConfig,
            numerics: DiTNumericsPolicy
        ) throws -> DiffusionStage {
            lock.lock(); calls.append(Call(stage: "sampler")); lock.unlock()
            let sampler = BlockingSampler()
            lock.lock(); samplers.append(sampler); lock.unlock()
            return sampler
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            lock.lock(); calls.append(Call(stage: "vae")); lock.unlock()
            return ProbeVAE()
        }
        var isAnySamplerBlocked: Bool {
            lock.lock(); defer { lock.unlock() }
            return samplers.contains { $0.isBlocked }
        }
        func releaseAll() {
            lock.lock()
            let samplers = self.samplers
            lock.unlock()
            for sampler in samplers { sampler.release() }
        }
    }

    /// Blocks its `execute` until `release()` is called. Uses a continuation so
    /// the cooperative thread pool is never blocked.
    private final class BlockingSampler: DiffusionStage {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Void, Never>?
        private(set) var isBlocked = false

        func execute(
            initialLatent: MTLBuffer,
            crossContext: MTLBuffer,
            outputLatent: MTLBuffer,
            startStep: Int,
            blockProgress: ((Int, Int) throws -> Void)?,
            stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?
        ) async throws {
            await withCheckedContinuation { continuation in
                lock.lock()
                self.continuation = continuation
                isBlocked = true
                lock.unlock()
            }
        }

        func release() {
            lock.lock()
            let continuation = self.continuation
            self.continuation = nil
            lock.unlock()
            continuation?.resume()
        }
    }

    // MARK: - Fatal Metal command-buffer fault poisoning (Task 5)

    /// Diffusion stage that throws a synthetic fatal Metal command-buffer
    /// error from `execute`, exercising the coordinator's `run()` catch path
    /// exactly as a real GPU fault would (the raw NSError propagates
    /// un-wrapped through the engine).
    private final class FatalMetalSampler: DiffusionStage {
        let error: Error
        init(error: Error) { self.error = error }
        func execute(
            initialLatent: MTLBuffer,
            crossContext: MTLBuffer,
            outputLatent: MTLBuffer,
            startStep: Int,
            blockProgress: ((Int, Int) throws -> Void)?,
            stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?
        ) async throws {
            throw error
        }
    }

    /// Factory whose diffusion stage throws the injected synthetic error,
    /// so the coordinator's catch path sees exactly that error.
    private final class FatalMetalFactory: GenerationStageFactory {
        let error: Error
        init(error: Error) { self.error = error }
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
            FatalMetalSampler(error: error)
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }

    /// Synthetic MTLCommandBufferErrorDomain fault with the given code.
    private func makeFatalMetalError(
        _ code: MTLCommandBufferError,
        description: String
    ) -> NSError {
        NSError(
            domain: MTLCommandBufferErrorDomain,
            code: Int(code.rawValue),
            userInfo: [NSLocalizedDescriptionKey: description])
    }

    /// Runs one generation against a factory that throws `error` from the
    /// diffusion stage, waits for the terminal state, and returns the
    /// coordinator.
    @MainActor
    private func runFatalFaultGeneration(error: Error) async throws -> GenerationCoordinator {
        let context = try makeContext()
        let coordinator = GenerationCoordinator(
            context: context, factory: FatalMetalFactory(error: error))
        coordinator.generate(prompt: "fatal", seed: 1, models: testModels())
        // Wait for the terminal state (the thrown fault propagates quickly).
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating, "generation ended after the fatal fault")
        return coordinator
    }

    @MainActor
    private func assertPoisoned(_ coordinator: GenerationCoordinator,
                                file: StaticString = #filePath,
                                line: UInt = #line) {
        XCTAssertTrue(coordinator.metalContextPoisoned,
                      "metalContextPoisoned must be true after a fatal Metal fault",
                      file: file, line: line)
        XCTAssertFalse(coordinator.isMetalAvailable,
                       "isMetalAvailable must be false after poisoning",
                       file: file, line: line)
        if case .failed(let message) = coordinator.state {
            XCTAssertTrue(message.contains("Fatal GPU fault"),
                          "expected the fatal fault message, got \(message)",
                          file: file, line: line)
            XCTAssertTrue(message.contains("Restart AnimaXS"),
                          "message must ask for a restart, got \(message)",
                          file: file, line: line)
        } else {
            XCTFail("expected failed state, got \(coordinator.state)",
                    file: file, line: line)
        }
        // Generation eligibility is blocked after poisoning (the same
        // coordinator.isMetalAvailable drives ContentView's eligibility).
        let eligibility = GenerationEligibility.evaluate(
            modelsResolved: true, isGenerating: false,
            prompt: "t", seedText: "1", metalAvailable: coordinator.isMetalAvailable)
        XCTAssertFalse(eligibility.isReady,
                       "eligibility must be blocked after poisoning",
                       file: file, line: line)
    }

    // MARK: - Fatal Metal fault tests

    @MainActor
    func testFatalMetalPageFaultPoisonsContext() async throws {
        let coordinator = try await runFatalFaultGeneration(error: makeFatalMetalError(
            .pageFault,
            description: "kIOGPUCommandBufferCallbackErrorPageFault while serving no-copy bytes"))
        assertPoisoned(coordinator)

        // A subsequent generate() is blocked: no new run, state stays failed.
        coordinator.generate(prompt: "again", seed: 2, models: testModels())
        XCTAssertFalse(coordinator.isGenerating,
                       "a second generate must be blocked after poisoning")
        if case .failed(let message) = coordinator.state {
            XCTAssertTrue(message.contains("Fatal GPU fault"), "got \(message)")
        } else {
            XCTFail("expected failed state after blocked generate, got \(coordinator.state)")
        }
    }

    @MainActor
    func testFatalMetalInvalidResourcePoisonsContext() async throws {
        let coordinator = try await runFatalFaultGeneration(error: makeFatalMetalError(
            .invalidResource,
            description: "resource became invalid while the command buffer executed"))
        assertPoisoned(coordinator)
    }

    @MainActor
    func testFatalMetalInternalErrorPoisonsContext() async throws {
        let coordinator = try await runFatalFaultGeneration(error: makeFatalMetalError(
            .internal,
            description: "internal Metal error"))
        assertPoisoned(coordinator)
    }

    @MainActor
    func testIOGPUPageFaultTextFallbackPoisonsContext() async throws {
        // Device-bridge fallback: no Metal domain/code exposed — the native
        // IOGPU page-fault text alone must still classify as fatal.
        let coordinator = try await runFatalFaultGeneration(error: NSError(
            domain: "IOGPU", code: 0,
            userInfo: [NSLocalizedDescriptionKey:
                "IOGPUCommandBufferCallbackErrorPageFault (kIOGPUCommandBufferCallbackErrorPageFault): GPU page fault"]))
        assertPoisoned(coordinator)
    }

    @MainActor
    func testGenerationErrorMetalStringifyingPageFaultPoisonsContext() async throws {
        // A path that stringifies the underlying fault into GenerationError.metal
        // must still be classified as fatal via the message fallback.
        let coordinator = try await runFatalFaultGeneration(
            error: GenerationError.metal(
                "command buffer failed: kIOGPUCommandBufferCallbackErrorPageFault (page fault)"))
        assertPoisoned(coordinator)
    }

    @MainActor
    func testOrdinaryFailureDoesNotPoisonContext() async throws {
        // A recoverable (non-Metal) failure must NOT poison the context:
        // state becomes failed with the normal message, but a fresh
        // Generate is available afterward.
        let context = try makeContext()
        let factory = ProbeFactory()
        factory.samplerOnExecute = { throw GenerationError.diffusionPack("boom") }
        let coordinator = GenerationCoordinator(context: context, factory: factory)
        coordinator.generate(prompt: "recoverable", seed: 1, models: testModels())
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        XCTAssertFalse(coordinator.metalContextPoisoned,
                       "an ordinary failure must not poison the context")
        XCTAssertTrue(coordinator.isMetalAvailable)
        if case .failed(let message) = coordinator.state {
            XCTAssertTrue(message.contains("Diffusion"), "got \(message)")
        } else {
            XCTFail("expected failed state, got \(coordinator.state)")
        }
        // A fresh Generate still starts after an ordinary failure.
        let okFactory = LifecycleFactory()
        let coordinator2 = GenerationCoordinator(context: context, factory: okFactory)
        coordinator2.generate(prompt: "fresh", seed: 2, models: testModels())
        XCTAssertTrue(coordinator2.isGenerating,
                      "a fresh Generate starts after an ordinary (non-fatal) failure")
        coordinator2.cancel()
    }

    @MainActor
    func testCooperativeCancellationDoesNotPoisonContext() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "lifecycle", seed: 11, models: testModels())
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertGreaterThanOrEqual(factory.sampler?.completedSteps ?? 0, 1)
        coordinator.cancel()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled, got \(coordinator.state)")
        }
        XCTAssertFalse(coordinator.metalContextPoisoned,
                       "cooperative cancellation must NOT poison the context")
        XCTAssertTrue(coordinator.isMetalAvailable,
                      "Metal stays available after cooperative cancellation")
    }

    // MARK: - K003 lifecycle: background/memory-warning cancellation

    /// Sampler that completes one step then suspends until cancellation, so a
    /// background/cancel request lands deterministically: `Task.sleep` is a
    /// cancellation point and throws `CancellationError` when the task is
    /// cancelled.
    private final class LifecycleSampler: DiffusionStage {
        private(set) var completedSteps = 0
        func execute(
            initialLatent: MTLBuffer,
            crossContext: MTLBuffer,
            outputLatent: MTLBuffer,
            startStep: Int,
            blockProgress: ((Int, Int) throws -> Void)?,
            stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?
        ) async throws {
            let count = initialLatent.length / 4
            let out = outputLatent.contents().bindMemory(to: Float.self, capacity: count)
            for step in startStep..<8 {
                for block in 0..<28 {
                    try Task.checkCancellation()
                    try blockProgress?(step, block)
                }
                for i in 0..<count { out[i] = Float(step + 1) }
                completedSteps += 1
                try stepCompleted?(step, 1.0, 0.5, outputLatent, outputLatent)
                if step == 0 && startStep == 0 {
                    // Deterministic pause: suspend until cancelled. Throws
                    // CancellationError when the coordinator cancels the task.
                    try await Task.sleep(nanoseconds: 60_000_000_000)
                }
            }
        }
    }

    private final class LifecycleFactory: GenerationStageFactory {
        var sampler: LifecycleSampler?
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
            let sampler = LifecycleSampler()
            self.sampler = sampler
            return sampler
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }

    @MainActor
    func testBackgroundTransitionCancelsToTerminalCancelledState() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "lifecycle", seed: 11, models: testModels())
        XCTAssertTrue(coordinator.isGenerating)

        // Wait until at least one step completed and the sampler is
        // mid-step-1 where it will observe cancellation.
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertGreaterThanOrEqual(factory.sampler?.completedSteps ?? 0, 1,
                                    "sampler reached step 1")

        // Simulate the app moving to background: cooperative cancellation.
        coordinator.appDidEnterBackground()

        // The engine must stop at the next block boundary and state must
        // become terminal `.cancelled`.
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating, "generation stopped after background")
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled after background, got \(coordinator.state)")
        }

        // A fresh Generate must be available immediately after the cancelled
        // run (no checkpoint/resume state blocks it).
        let factory2 = LifecycleFactory()
        let coordinator2 = GenerationCoordinator(context: context, factory: factory2)
        coordinator2.generate(prompt: "fresh", seed: 12, models: testModels())
        XCTAssertTrue(coordinator2.isGenerating,
                      "a fresh Generate starts after a cancelled run")
        coordinator2.cancel()
    }

    @MainActor
    func testForegroundAfterBackgroundDoesNotOfferResume() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "lifecycle", seed: 11, models: testModels())
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        coordinator.appDidEnterBackground()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled after background, got \(coordinator.state)")
        }

        // Foreground: nothing changes — there is no resume state; the user
        // starts a fresh Generate if desired.
        coordinator.appWillEnterForeground()
        XCTAssertEqual(coordinator.state, .cancelled,
                       "foreground must not resurrect any resume state")
    }

    // MARK: - K004 resource policy: memory warning

    @MainActor
    func testMemoryWarningCancelsToTerminalCancelledState() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "k004", seed: 5, models: testModels())
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        // Memory warning during generation → cooperative cancellation reaches
        // the terminal cancelled state.
        coordinator.handleMemoryWarning()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled after memory warning, got \(coordinator.state)")
        }

        // A fresh Generate is available immediately afterward.
        let factory2 = LifecycleFactory()
        let coordinator2 = GenerationCoordinator(context: context, factory: factory2)
        coordinator2.generate(prompt: "fresh", seed: 6, models: testModels())
        XCTAssertTrue(coordinator2.isGenerating,
                      "a fresh Generate starts after a memory-warning cancellation")
        coordinator2.cancel()
    }

    @MainActor
    func testMemoryWarningWhenIdleDoesNothing() {
        let coordinator = GenerationCoordinator(
            context: nil, factory: ProbeFactory(), attemptMetalFallback: false)
        coordinator.handleMemoryWarning()
        XCTAssertEqual(coordinator.state, .idle, "no generation → no state change")
    }

    // MARK: - User-initiated cancel

    @MainActor
    func testCancelReachesTerminalCancelledStateAndFreshGenerateAvailable() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "full", seed: 3, models: testModels())
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        coordinator.cancel()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled after cancel, got \(coordinator.state)")
        }

        // After a cancelled run the next action is a fresh Generate.
        let factory2 = LifecycleFactory()
        let coordinator2 = GenerationCoordinator(context: context, factory: factory2)
        coordinator2.generate(prompt: "fresh", seed: 4, models: testModels())
        XCTAssertTrue(coordinator2.isGenerating,
                      "a fresh Generate starts after a cancelled run")
        coordinator2.cancel()
    }

    // MARK: - Stage lifetime release

    // P1-F: a thrown diffusion stage must still record a nonzero diffusion time
    // (the measured() helper closes the stage timer on throw), instead of being
    // silently folded into "Other".
    func testThrownDiffusionStageRecordsNonzeroDiffusionTime() async throws {
        let context = try makeContext()
        final class ThrowingSampler: DiffusionStage {
            func execute(
                initialLatent: MTLBuffer, crossContext: MTLBuffer, outputLatent: MTLBuffer,
                startStep: Int,
                blockProgress: ((Int, Int) throws -> Void)?,
                stepCompleted: ((Int, Float, Float, MTLBuffer, MTLBuffer) throws -> Void)?
            ) async throws {
                throw AnimapkError.validation("synthetic diffusion failure")
            }
        }
        final class ThrowingFactory: GenerationStageFactory {
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
                ThrowingSampler()
            }
            func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
                ProbeVAE()
            }
        }
        let engine = GenerationEngine(context: context, factory: ThrowingFactory())
        let metrics = MetricsCollector()
        do {
            _ = try await engine.generate(
                prompt: "test", seed: 1, models: testModels(),
                metrics: metrics, optimization: .currentBaseline)
            XCTFail("expected the throwing sampler to fail the generate call")
        } catch {
            // Expected: the thrown diffusion failure propagates.
        }
        let snapshot = metrics.snapshot()
        // The diffusion stage timer must have been closed by the measured()
        // defer, so diffusion time is nonzero even though diffusion threw.
        XCTAssertGreaterThan(snapshot.diffusion, 0,
                             "a thrown diffusion stage must record nonzero diffusion time")
    }

    func testHeavyStageObjectsAreReleasedAfterTheirStage() async throws {
        let context = try makeContext()
        let counting = CountingFactory()
        let engine = GenerationEngine(context: context, factory: counting)
        _ = try await engine.generate(prompt: "t", seed: 1, models: testModels())
        // All probes must be deallocated after generate() returns: each stage's
        // heavy object (and its mmap in production) is lexically confined to
        // its helper function.
        XCTAssertEqual(counting.aliveCount(), 0,
                       "all stage objects released after generation completes")
    }

    /// Tracks probe deinit so the test can prove stage-scoped release.
    private final class CountingFactory: GenerationStageFactory {
        private let lock = NSLock()
        private var alive = 0

        func aliveCount() -> Int {
            lock.lock(); defer { lock.unlock() }
            return alive
        }

        private func register(_ probe: AnyObject) {
            lock.lock(); alive += 1; lock.unlock()
            guard let hook = probe as? DeinitReporting else { return }
            hook.onDeinit = { [weak self] in
                self?.lock.lock(); self?.alive -= 1; self?.lock.unlock()
            }
        }

        func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
            let probe = ProbeEncoder()
            register(probe)
            return probe
        }
        func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
            let probe = ProbeAdapter()
            register(probe)
            return probe
        }
        func makeDiffusion(
            context: MetalContext, fileURL: URL,
            optimization: InferenceOptimizationConfig,
            numerics: DiTNumericsPolicy
        ) throws -> DiffusionStage {
            let probe = ProbeSampler()
            register(probe)
            return probe
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            let probe = ProbeVAE()
            register(probe)
            return probe
        }
    }

    // MARK: - Task 8: a fresh Generate owns the image/metrics surface

    /// Run 1 succeeds (image + metrics published); run 2 is accepted and then
    /// fails in diffusion (before VAE/decode). The fresh-run clears mean run 2
    /// never displays run 1's image, and metrics are per-run: cleared at the
    /// accepted Generate, re-published by run 2's own defer.
    @MainActor
    func testFreshGenerateClearsPriorImageAndMetricsBeforeFailing() async throws {
        let context = try makeContext()
        let factory = ProbeFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)
        let models = testModels()

        // Run 1: success → image + metrics published.
        coordinator.generate(prompt: "run 1", seed: 1, models: models)
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        XCTAssertNotNil(coordinator.image, "run 1 must publish an image")
        if case .completed = coordinator.state {
            // expected
        } else {
            XCTFail("expected completed after run 1, got \(coordinator.state)")
        }
        // Metrics are published from the run's defer on a MainActor hop; poll.
        for _ in 0..<250 {
            if coordinator.lastMetricsText != nil { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertNotNil(coordinator.lastMetricsText, "run 1 must publish metrics")

        // Run 2: the next diffusion stage throws (fails before decode).
        factory.samplerOnExecute = { throw GenerationError.diffusionPack("boom") }
        coordinator.generate(prompt: "run 2", seed: 2, models: models)
        // Synchronously after the ACCEPTED Generate: run 1's image/metrics are
        // gone, so a failure cannot show run 1's image.
        XCTAssertNil(coordinator.image, "fresh Generate must clear the prior image")
        XCTAssertNil(coordinator.lastMetricsText, "fresh Generate must clear prior metrics")
        XCTAssertTrue(coordinator.isGenerating, "run 2 was accepted")

        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        XCTAssertNil(coordinator.image, "a failed run must not display an image")
        if case .failed(let message) = coordinator.state {
            XCTAssertTrue(message.contains("Diffusion"), "got \(message)")
        } else {
            XCTFail("expected failed after run 2, got \(coordinator.state)")
        }
        // The failed run still publishes its own metrics (defer runs on
        // failure too), so the summary shown belongs to run 2, not run 1.
        for _ in 0..<250 {
            if coordinator.lastMetricsText != nil { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertNotNil(coordinator.lastMetricsText,
                        "the failed run still publishes its own metrics")
    }

    /// A blocked Generate (poisoned context) must not delete the prior run's
    /// result: state, image, and metrics stay exactly as the last run left
    /// them, and no new engine run starts.
    @MainActor
    func testBlockedGenerateAfterPoisoningPreservesLastRunResults() async throws {
        let context = try makeContext()
        let factory = ProbeFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)
        let models = testModels()

        // Run 1: success → image + metrics.
        coordinator.generate(prompt: "run 1", seed: 1, models: models)
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        XCTAssertNotNil(coordinator.image)
        for _ in 0..<250 {
            if coordinator.lastMetricsText != nil { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertNotNil(coordinator.lastMetricsText)

        // Run 2: fatal Metal fault — accepted (clears image/metrics at start),
        // then poisons the context.
        let fatalError = makeFatalMetalError(
            .pageFault,
            description: "kIOGPUCommandBufferCallbackErrorPageFault while serving no-copy bytes")
        factory.samplerOnExecute = { throw fatalError }
        coordinator.generate(prompt: "run 2", seed: 2, models: models)
        XCTAssertNil(coordinator.image, "accepted run 2 cleared run 1's image")
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        XCTAssertTrue(coordinator.metalContextPoisoned, "run 2's fault poisons the context")
        XCTAssertNil(coordinator.image, "the fatal run never produced an image")
        for _ in 0..<250 {
            if coordinator.lastMetricsText != nil { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        let metricsAfterRun2 = try XCTUnwrap(coordinator.lastMetricsText)

        // Run 3: blocked by the poisoned context — nothing may change.
        let callsBefore = factory.calls.count
        let stateBefore = coordinator.state
        coordinator.generate(prompt: "run 3", seed: 3, models: models)
        XCTAssertEqual(factory.calls.count, callsBefore,
                       "a blocked Generate must not start an engine run")
        XCTAssertEqual(coordinator.state, stateBefore,
                       "a blocked Generate must not change state")
        XCTAssertNil(coordinator.image, "a blocked Generate must not clear/alter the image")
        XCTAssertEqual(coordinator.lastMetricsText, metricsAfterRun2,
                       "a blocked Generate must not delete the last run's metrics")
    }

    /// A Generate tapped while a run is in-flight is rejected at the
    /// `!isGenerating` guard and must leave the in-flight run's state, image,
    /// and metrics untouched (no engine run, no clears).
    @MainActor
    func testBlockedGenerateWhileGeneratingDoesNotClearInFlightState() async throws {
        let context = try makeContext()
        let factory = BlockingFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)

        coordinator.generate(prompt: "first", seed: 1, models: testModels())
        XCTAssertTrue(coordinator.isGenerating, "first generation starts synchronously")
        XCTAssertNil(coordinator.image, "in-flight run owns an empty image surface")
        XCTAssertNil(coordinator.lastMetricsText, "no metrics before the run publishes")

        let callsBefore = factory.callCount
        let stateBefore = coordinator.state
        coordinator.generate(prompt: "second", seed: 2, models: testModels())
        XCTAssertEqual(factory.callCount, callsBefore,
                       "second Generate rejected while one is running")
        XCTAssertEqual(coordinator.state, stateBefore,
                       "blocked Generate must not change the in-flight state")
        XCTAssertNil(coordinator.image, "blocked Generate must not touch the image surface")
        XCTAssertNil(coordinator.lastMetricsText,
                     "blocked Generate must not publish or clear metrics")

        for _ in 0..<250 {
            if factory.isAnySamplerBlocked { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(factory.isAnySamplerBlocked, "first sampler reached the blocked stage")
        factory.releaseAll()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        XCTAssertNotNil(coordinator.image,
                        "the in-flight run completes and publishes its image")
    }
}
