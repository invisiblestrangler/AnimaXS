import XCTest
import Metal
@testable import AnimaXS

/// I004 resume-execution tests: the sampler's `startStep` continuation must
/// reproduce a full run bit-for-bit, range validation must reject bad steps,
/// and CheckpointStore must reject incompatible/corrupt checkpoints.
final class ResumeEquivalenceTests: XCTestCase {

    // MARK: - Deterministic Euler-like probe sampler

    /// Models the production sampler's recurrence: each step transforms the
    /// current latent deterministically, and only steps `startStep..<8` run.
    /// `latent' = latent * 1.5 + step * 0.25` — depends on the initial latent
    /// AND the step index, so an equivalence failure cannot hide a wrong
    /// initial latent or a wrong startStep.
    private final class RecurrenceSampler: DiffusionStage {
        let onExecute: (() throws -> Void)?
        var maxSteps: Int
        var observedStartStep: Int?
        var blockCalls: [Int] = []
        var lastWritten: [Float]?
        var callbackSteps: [Int] = []

        init(maxSteps: Int = 8, onExecute: (() throws -> Void)? = nil) {
            self.maxSteps = maxSteps
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
            observedStartStep = startStep
            try onExecute?()
            let count = initialLatent.length / 4
            var values = read(initialLatent, count: count)
            for step in startStep..<min(maxSteps, 8) {
                blockCalls.append(step)
                for i in 0..<count {
                    values[i] = values[i] * 1.5 + Float(step) * 0.25
                }
                write(values, to: outputLatent)
                callbackSteps.append(step)
                try stepCompleted?(step, 1.0, 0.5, outputLatent, outputLatent)
            }
            lastWritten = values
        }

        private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
        private func write(_ values: [Float], to buffer: MTLBuffer) {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
            for i in 0..<values.count { pointer[i] = values[i] }
        }
    }

    private final class RecurrenceFactory: GenerationStageFactory {
        var sampler: RecurrenceSampler?
        var maxSteps = 8
        var samplerOnExecute: (() throws -> Void)?
        func makePromptEncoder(context: MetalContext, fileURL: URL) throws -> PromptEncoderStage {
            ProbeEncoder()
        }
        func makeContextAdapter(context: MetalContext, fileURL: URL) throws -> ContextAdapterStage {
            ProbeAdapter()
        }
        func makeDiffusion(context: MetalContext, fileURL: URL) throws -> DiffusionStage {
            let sampler = RecurrenceSampler(maxSteps: maxSteps, onExecute: samplerOnExecute)
            self.sampler = sampler
            return sampler
        }
        func makeVAE(context: MetalContext, fileURL: URL) throws -> VAEDecodeStage {
            ProbeVAE()
        }
    }

    private final class ProbeEncoder: PromptEncoderStage {
        func execute(
            tokenIDs: [Int], output: MTLBuffer,
            layerCompleted: ((Int, MTLBuffer) throws -> Void)?
        ) async throws {}
    }
    private final class ProbeAdapter: ContextAdapterStage {
        func execute(
            qwenContext: MTLBuffer, contextTokens: Int,
            t5IDs: [Int], t5Weights: [Float], output: MTLBuffer,
            layerCompleted: ((Int, MTLBuffer) throws -> Void)?
        ) async throws {}
    }
    private final class ProbeVAE: VAEDecodeStage {
        func decode(latent: MTLBuffer) async throws -> DecodedRGBA8 {
            DecodedRGBA8(width: 512, height: 512, bytes: [UInt8](repeating: 255, count: 512 * 512 * 4))
        }
    }

    private func makeContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        return context
    }

    private func models() -> ResolvedModels {
        ResolvedModels(
            textEncoder: URL(fileURLWithPath: "/tmp/qwen.animapk"),
            dit: URL(fileURLWithPath: "/tmp/dit.animapk"),
            vae: URL(fileURLWithPath: "/tmp/vae.animapk"))
    }

    // MARK: - Equivalence: split run == full run

    /// full run from step 0 → A; run steps 0..N-1 → checkpoint latent C;
    /// resume with C at startStep N → B. A must equal B exactly.
    func testResumedRunMatchesFullRunExactly() async throws {
        let context = try makeContext()

        // Full run: capture the final latent via the engine's checkpoint
        // callback (step 7 completed → nextStep 8).
        let fullFactory = RecurrenceFactory()
        var finalA: [Float]?
        _ = try await GenerationEngine(context: context, factory: fullFactory)
            .generate(prompt: "p", seed: 7, models: models(), checkpoint: { step, latent in
                if step == 7 { finalA = latent }
            })
        let fullLatent = try XCTUnwrap(finalA, "full run must report step 7 checkpoint")

        // Split run at N=3: stop the sampler after step 2 (maxSteps=3), then
        // capture the checkpoint latent the engine persisted at step 2.
        let splitFactory = RecurrenceFactory()
        splitFactory.maxSteps = 3
        var checkpointC: [Float]?
        _ = try await GenerationEngine(context: context, factory: splitFactory)
            .generate(prompt: "p", seed: 7, models: models(), checkpoint: { step, latent in
                if step == 2 { checkpointC = latent }
            })
        let c = try XCTUnwrap(checkpointC, "split run must report step 2 checkpoint")

        // Resume: initial latent = checkpoint latent, startStep = 3.
        let resumeFactory = RecurrenceFactory()
        let resumeBuffer = makeBuffer(c, on: context.device)
        var finalB: [Float]?
        _ = try await GenerationEngine(context: context, factory: resumeFactory)
            .generate(
                prompt: "p", seed: 7, models: models(),
                noise: resumeBuffer, startStep: 3,
                checkpoint: { step, latent in
                    if step == 7 { finalB = latent }
                })

        let b = try XCTUnwrap(finalB, "resumed run must report step 7 checkpoint")
        XCTAssertEqual(resumeFactory.sampler?.observedStartStep, 3,
                       "sampler resumed at step 3")
        XCTAssertEqual(fullLatent.map(\.bitPattern), b.map(\.bitPattern),
                       "split-run final latent must equal full-run final latent bit-for-bit")
    }

    // MARK: - startStep range validation

    func testStartStepBelowRangeRejected() async throws {
        let context = try makeContext()
        let engine = GenerationEngine(context: context, factory: RecurrenceFactory())
        do {
            _ = try await engine.generate(
                prompt: "p", seed: 1, models: models(), startStep: -1)
            XCTFail("expected startStep -1 to be rejected")
        } catch let error as GenerationError {
            if case .sampler = error { /* expected */ } else { XCTFail("unexpected error \(error)") }
        }
    }

    func testStartStepAboveRangeRejected() async throws {
        let context = try makeContext()
        let engine = GenerationEngine(context: context, factory: RecurrenceFactory())
        do {
            _ = try await engine.generate(
                prompt: "p", seed: 1, models: models(), startStep: 9)
            XCTFail("expected startStep 9 to be rejected")
        } catch let error as GenerationError {
            if case .sampler = error { /* expected */ } else { XCTFail("unexpected error \(error)") }
        }
    }

    /// startStep == 8 means all steps done: the sampler must be invoked with
    /// startStep 8 and execute NO blocks — the checkpoint latent is final.
    func testCompletedStepEightBypassesDiffusion() async throws {
        let context = try makeContext()
        let factory = RecurrenceFactory()
        let engine = GenerationEngine(context: context, factory: factory)
        let initial = makeBuffer([Float](repeating: 2.5, count: DiffusionSampler.latentElements),
                                 on: context.device)
        _ = try await engine.generate(
            prompt: "p", seed: 1, models: models(),
            noise: initial, startStep: 8)
        XCTAssertEqual(factory.sampler?.observedStartStep, 8)
        XCTAssertEqual(factory.sampler?.blockCalls, [], "no DiT blocks when resuming at step 8")
        // The output must be the checkpoint latent unchanged.
        let written = try XCTUnwrap(factory.sampler?.lastWritten)
        XCTAssertTrue(written.allSatisfy { $0 == 2.5 },
                      "step-8 resume copies the checkpoint latent without evaluation")
    }

    // MARK: - CheckpointStore validation

    private func makeCheckpoint(
        step: Int = 3, prompt: String = "p", seed: UInt64 = 7,
        width: Int = 512, height: Int = 512,
        hashes: ModelHashes? = nil
    ) throws -> GenerationCheckpoint {
        let count = ModelConstants.ditLatentChannels * (width / 8) * (height / 8)
        let latent = (0..<count).map { Float($0 % 5) - 2 }
        return try GenerationCheckpoint(
            latent: latent, step: step, prompt: prompt, seed: seed,
            width: width, height: height,
            modelHashes: hashes ?? ModelHashes(dit: "d", textEncoder: "t", vae: "v"))
    }

    func testCheckpointValidationAcceptsMatchingInputs() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true))
        let checkpoint = try makeCheckpoint()
        let step = try store.validate(
            checkpoint, prompt: "p", seed: 7, resolution: (512, 512),
            modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v"))
        XCTAssertEqual(step, 3)
    }

    func testCheckpointValidationRejectsTerminalStep() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true))
        // A fully-completed run (step == samplerSteps) has no diffusion left
        // to resume and must be rejected as non-resumable.
        let terminal = try makeCheckpoint(step: ModelConstants.samplerSteps)
        XCTAssertThrowsError(
            try store.validate(terminal, prompt: "p", seed: 7, resolution: (512, 512),
                               modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v"))) { error in
            guard case GenerationError.sampler = error else {
                return XCTFail("expected sampler validation error, got \(error)")
            }
        }
        // The final PARTIAL step is still resumable.
        let partial = try makeCheckpoint(step: ModelConstants.samplerSteps - 1)
        XCTAssertEqual(
            try store.validate(partial, prompt: "p", seed: 7, resolution: (512, 512),
                               modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v")),
            ModelConstants.samplerSteps - 1)
    }

    func testCheckpointValidationRejectsPromptMismatch() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true))
        let checkpoint = try makeCheckpoint()
        XCTAssertThrowsError(
            try store.validate(checkpoint, prompt: "different", seed: 7,
                               resolution: (512, 512),
                               modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v")))
    }

    func testCheckpointValidationRejectsSeedMismatch() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true))
        let checkpoint = try makeCheckpoint()
        XCTAssertThrowsError(
            try store.validate(checkpoint, prompt: "p", seed: 99,
                               resolution: (512, 512),
                               modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v")))
    }

    func testCheckpointValidationRejectsResolutionMismatch() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true))
        let checkpoint = try makeCheckpoint(width: 1024, height: 1024)
        XCTAssertThrowsError(
            try store.validate(checkpoint, prompt: "p", seed: 7,
                               resolution: (512, 512),
                               modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v")))
    }

    func testCheckpointValidationRejectsModelHashMismatch() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true))
        let checkpoint = try makeCheckpoint()
        XCTAssertThrowsError(
            try store.validate(checkpoint, prompt: "p", seed: 7,
                               resolution: (512, 512),
                               modelHashes: ModelHashes(dit: "DIFFERENT", textEncoder: "t", vae: "v")))
    }

    func testCheckpointValidationRejectsNonfiniteLatent() throws {
        let count = 16 * 64 * 64
        var latent = [Float](repeating: 0, count: count)
        latent[123] = .nan
        XCTAssertThrowsError(try GenerationCheckpoint(
            latent: latent, step: 3, prompt: "p", seed: 7,
            width: 512, height: 512,
            modelHashes: ModelHashes(dit: "d", textEncoder: "t", vae: "v")))
    }

    func testCorruptCheckpointFileIsRemovedOnLoad() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try CheckpointStore(directory: directory)
        try Data("not a checkpoint".utf8).write(
            to: directory.appendingPathComponent("generation-checkpoint.json"))
        XCTAssertNil(store.load(), "corrupt checkpoint must be treated as absent")
        XCTAssertFalse(store.hasCheckpoint, "corrupt checkpoint removed on load")
    }

    func testCheckpointStoreRoundTrip() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-resume-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try CheckpointStore(directory: directory)
        let checkpoint = try makeCheckpoint()
        try store.save(checkpoint)
        XCTAssertTrue(store.hasCheckpoint)
        let loaded = try XCTUnwrap(store.load())
        XCTAssertEqual(loaded, checkpoint)
        store.remove()
        XCTAssertFalse(store.hasCheckpoint)
    }

    // MARK: - Helpers

    private func makeBuffer(_ values: [Float], on device: MTLDevice) -> MTLBuffer {
        let buffer = device.makeBuffer(
            length: values.count * 4, options: .storageModeShared)!
        buffer.contents().copyMemory(from: values, byteCount: values.count * 4)
        return buffer
    }
}
