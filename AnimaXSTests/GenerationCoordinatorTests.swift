import XCTest
import Metal
@testable import AnimaXS

/// K002 orchestration tests: stage order, seed forwarding, three-URL model
/// topology, block progress, failure recovery, no-Metal behavior, and heavy
/// stage lifetime. No 2 GB model packs are required — probe stages substitute
/// for the real executors.
final class GenerationCoordinatorTests: XCTestCase {

    // MARK: - Default-store isolation

    /// Redirect the coordinator's *default* persistent store to a per-test temp
    /// directory so tests that wire a coordinator with no explicit store never
    /// read or write the real Application Support checkpoint (keeps the suite
    /// hermetic and deterministic, like `ModelStore.secureInstalls`).
    private var defaultStoreDir: URL?

    override func setUpWithError() throws {
        try super.setUpWithError()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-defaultstore-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        GenerationCoordinator.defaultCheckpointStoreDirectoryOverride = dir
        defaultStoreDir = dir
    }

    override func tearDownWithError() throws {
        GenerationCoordinator.defaultCheckpointStoreDirectoryOverride = nil
        if let dir = defaultStoreDir {
            try? FileManager.default.removeItem(at: dir)
        }
        defaultStoreDir = nil
        try super.tearDownWithError()
    }

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
                // Write a deterministic post-step latent so the engine's
                // checkpoint capture and VAE stage have finite data.
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
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
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
            textEncoder: URL(fileURLWithPath: "/tmp/qwen.animapk"),
            dit: URL(fileURLWithPath: "/tmp/dit.animapk"),
            vae: URL(fileURLWithPath: "/tmp/vae.animapk"))
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
        XCTAssertEqual(adapter.url, models.dit, "adapter reads the DiT pack")
        XCTAssertEqual(sampler.url, models.dit, "sampler reads the DiT pack")
        XCTAssertEqual(factory.calls.first { $0.stage == "qwen" }!.url, models.textEncoder)
        XCTAssertEqual(factory.calls.first { $0.stage == "vae" }!.url, models.vae)
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
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
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
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
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
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
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

    // MARK: - K003 lifecycle: background cancel retains checkpoint

    @MainActor
    func testBackgroundTransitionCancelsAndRetainsCheckpoint() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-lifecycle-\\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let coordinator = GenerationCoordinator(
            context: context, factory: factory, checkpointStore: store)

        coordinator.generate(prompt: "lifecycle", seed: 11, models: testModels())
        XCTAssertTrue(coordinator.isGenerating)

        // Wait until at least one step completed (checkpoint exists) and the
        // sampler is mid-step-1 where it will observe cancellation.
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertGreaterThanOrEqual(factory.sampler?.completedSteps ?? 0, 1,
                                    "sampler reached step 1")

        // Simulate the app moving to background.
        coordinator.appDidEnterBackground()

        // The engine must stop at the next block boundary (cooperative), the
        // checkpoint must be retained, and state must become cancelled.
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating, "generation stopped after background")
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled after background, got \\(coordinator.state)")
        }
        // The checkpoint Task persists asynchronously on the main actor; wait
        // for the coordinator to observe it.
        for _ in 0..<100 {
            if coordinator.canResume { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(coordinator.canResume, "checkpoint retained for resume")
        XCTAssertEqual(coordinator.completedSteps, 1,
                       "exactly one completed step retained")
        XCTAssertTrue(store.hasCheckpoint, "checkpoint persisted on disk")
    }

    @MainActor
    func testProductionDefaultWiringPersistsAcrossColdLaunch() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()

        // Production wiring: construct the coordinator with NO explicit store,
        // exactly as ContentView does. The default must resolve to a persistent
        // CheckpointStore (not nil), so a completed step is written to disk.
        let coordinator = GenerationCoordinator(context: context, factory: factory)
        coordinator.generate(prompt: "cold-launch", seed: 77, models: testModels())
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        coordinator.cancel()
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        // The persistence save is enqueued on the main actor; wait for it.
        for _ in 0..<100 {
            if coordinator.canResume { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(coordinator.canResume, "default-wired coordinator retained a checkpoint")
        XCTAssertEqual(coordinator.completedSteps, 1)

        // A file must actually exist on disk at the default store's location.
        let defaultURL = try XCTUnwrap(GenerationCoordinator.defaultCheckpointStoreDirectoryOverride)
            .appendingPathComponent("generation-checkpoint.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: defaultURL.path),
                      "default wiring must persist the checkpoint file to disk")

        // "Cold launch": a FRESH coordinator (same default wiring, no explicit
        // store) must reload the persisted checkpoint on init — proving
        // recovery from disk, not from an in-memory object that no longer exists.
        let fresh = GenerationCoordinator(context: context, factory: LifecycleFactory())
        XCTAssertTrue(fresh.canResume, "cold launch must reload the persisted checkpoint")
        XCTAssertEqual(fresh.completedSteps, 1, "cold launch reloads the completed step")
    }

    @MainActor
    func testForegroundOffersResumeWhenCheckpointCompatible() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-lifecycle-\\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let coordinator = GenerationCoordinator(
            context: context, factory: factory, checkpointStore: store)

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

        // Foreground: the checkpoint is still offered (no auto-discard).
        coordinator.appWillEnterForeground()
        for _ in 0..<100 {
            if coordinator.canResume { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(coordinator.canResume)
        XCTAssertEqual(coordinator.completedSteps, 1)

        // Resuming continues from step 1 and completes.
        coordinator.resume(prompt: "lifecycle", seed: 11, models: testModels())
        for _ in 0..<250 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating)
        if case .completed = coordinator.state {
            // expected
        } else {
            XCTFail("expected completed after resume, got \\(coordinator.state)")
        }
    }

    @MainActor
    func testResumeRejectsIncompatibleCheckpoint() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-lifecycle-\\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let coordinator = GenerationCoordinator(
            context: context, factory: factory, checkpointStore: store)

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
        XCTAssertTrue(coordinator.canResume)

        // Different seed → checkpoint must be rejected and discarded.
        coordinator.resume(prompt: "lifecycle", seed: 99, models: testModels())
        if case .failed(let message) = coordinator.state {
            XCTAssertTrue(message.contains("seed"), "expected seed mismatch, got \\(message)")
        } else {
            XCTFail("expected failed state, got \\(coordinator.state)")
        }
        XCTAssertFalse(coordinator.canResume, "incompatible checkpoint discarded")
        XCTAssertFalse(store.hasCheckpoint, "incompatible checkpoint removed from disk")
    }

    // MARK: - K004 resource policy: memory warning + thermal

    @MainActor
    func testMemoryWarningCancelsAndPreservesCheckpoint() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-k004-\\(UUID().uuidString)", isDirectory: true))
        defer { store.remove() }
        let coordinator = GenerationCoordinator(
            context: context, factory: factory, checkpointStore: store)

        coordinator.generate(prompt: "k004", seed: 5, models: testModels())
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        // Memory warning during generation → recoverable cancelled state with
        // a retained checkpoint (can Resume).
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
        for _ in 0..<250 {
            if coordinator.canResume && store.hasCheckpoint { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(coordinator.canResume, "memory warning preserves resume state")
        XCTAssertTrue(store.hasCheckpoint, "checkpoint persisted before memory warning")
    }

    @MainActor
    func testMemoryWarningWhenIdleDoesNothing() {
        let coordinator = GenerationCoordinator(
            context: nil, factory: ProbeFactory(), attemptMetalFallback: false)
        coordinator.handleMemoryWarning()
        XCTAssertEqual(coordinator.state, .idle, "no generation → no state change")
    }

    @MainActor
    func testThermalPolicyContinuesOnNominalAndFair() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)
        coordinator.generate(prompt: "k004", seed: 5, models: testModels())
        // nominal/fair must not cancel a running generation.
        coordinator.handleThermalState(.nominal)
        coordinator.handleThermalState(.fair)
        XCTAssertTrue(coordinator.isGenerating, "nominal/fair continues generation")
        // Cleanup: cancel and wait for the sampler's cancellation point.
        coordinator.cancel()
        for _ in 0..<100 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    @MainActor
    func testThermalPolicyStopsOnSeriousAndCritical() async throws {
        let context = try makeContext()
        let factory = LifecycleFactory()
        let coordinator = GenerationCoordinator(context: context, factory: factory)
        coordinator.generate(prompt: "k004", seed: 5, models: testModels())
        XCTAssertTrue(coordinator.isGenerating)
        // Wait for a checkpoint to exist before the thermal stop so resume
        // availability can be asserted.
        for _ in 0..<250 {
            if factory.sampler?.completedSteps ?? 0 >= 1 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        coordinator.handleThermalState(.serious)
        for _ in 0..<100 {
            if !coordinator.isGenerating { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(coordinator.isGenerating, "serious thermal state stops generation")
        if case .cancelled = coordinator.state {
            // expected
        } else {
            XCTFail("expected cancelled on serious thermal, got \\(coordinator.state)")
        }
        for _ in 0..<100 {
            if coordinator.canResume { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(coordinator.canResume, "thermal stop preserves resume state")
    }

    /// Sampler that completes one step then suspends until cancellation, so a
    /// background/cancel request lands deterministically: `Task.sleep` is a
    /// cancellation point and throws `CancellationError` when the task is
    /// cancelled. On resume (startStep >= 1) it runs all remaining steps.
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
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
            let sampler = LifecycleSampler()
            self.sampler = sampler
            return sampler
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }

    // MARK: - Stage lifetime release

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
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
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
}
