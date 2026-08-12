import XCTest
import Metal
@testable import AnimaXS

final class QwenEncoderMetalTests: XCTestCase {
    func testRealW8EncoderAgainstStructuralOracle() async throws {
        let bundledPack = bundledFixture(named: "qwen3-0.6b-xsmax-w8.animapk")
        let bundledTokens = bundledFixture(named: "qwen_token_ids.i32")
        guard let packURL = ProcessInfo.processInfo.environment["ANIMAXS_QWEN_PACK"]
                .map(URL.init(fileURLWithPath:)) ?? bundledPack,
              let fixtureDirectory = ProcessInfo.processInfo.environment["ANIMAXS_QWEN_ORACLE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledTokens?.deletingLastPathComponent() else {
            throw XCTSkip("real Qwen W8 pack/oracle fixture not available")
        }
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let tokenData = try Data(contentsOf: fixtureDirectory.appendingPathComponent(
            "qwen_token_ids.i32"))
        let tokenIDs = tokenData.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self)).map(Int.init)
        }
        XCTAssertEqual(tokenIDs.count, 46)
        let file = try AnimapkFile(url: packURL)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: tokenIDs.count * 1_024 * 4, options: .storageModeShared))
        let checkpoints: [Int: String] = [0: "qwen_layer_00.f32", 15: "qwen_layer_15.f32",
                                           27: "qwen_layer_27.f32"]
        let start = Date()
        try await QwenEncoderMetal(context: context, file: file).execute(
            tokenIDs: tokenIDs, output: output
        ) { layer, residual in
            guard let name = checkpoints[layer] else { return }
            let expected = try self.floats(name, in: fixtureDirectory)
            let metric = self.metrics(residual, expected)
            print("F007_QWEN_LAYER_\(layer) maxAbs=\(metric.maxAbs) "
                + "rmse=\(metric.rmse) cosine=\(metric.cosine)")
            XCTAssertGreaterThanOrEqual(metric.cosine, 0.999)
            XCTAssertTrue(metric.finite)
        }
        let expected = try floats("qwen_final.f32", in: fixtureDirectory)
        let metric = metrics(output, expected)
        print("F007_QWEN_FINAL maxAbs=\(metric.maxAbs) rmse=\(metric.rmse) "
            + "cosine=\(metric.cosine) seconds=\(Date().timeIntervalSince(start))")
        XCTAssertGreaterThanOrEqual(metric.cosine, 0.999)
        XCTAssertTrue(metric.finite)
    }

    func testRealW4AdapterAgainstStructuralOracle() async throws {
        let bundledPack = bundledFixture(named: "g003-adapter-w4.animapk")
        let bundledIDs = bundledFixture(named: "adapter_t5_ids.i32")
        guard let packURL = ProcessInfo.processInfo.environment["ANIMAXS_ADAPTER_PACK"]
                .map(URL.init(fileURLWithPath:)) ?? bundledPack,
              let fixtureDirectory = ProcessInfo.processInfo.environment["ANIMAXS_ADAPTER_ORACLE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledIDs?.deletingLastPathComponent() else {
            throw XCTSkip("real adapter W4 pack/oracle fixture not available")
        }
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let ids = try int32s("adapter_t5_ids.i32", in: fixtureDirectory)
        let weights = try floats("adapter_t5_weights.f32", in: fixtureDirectory)
        let source = try floats("adapter_context.f32", in: fixtureDirectory)
        XCTAssertEqual(ids.count, 47)
        XCTAssertEqual(weights.count, ids.count)
        XCTAssertEqual(source.count, 46 * 1_024)
        let contextBuffer = makeBuffer(source, context: context)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: 512 * 1_024 * 4, options: .storageModeShared))
        let checkpoints: [Int: String] = [0: "adapter_block_00.f32", 5: "adapter_block_05.f32"]
        let start = Date()
        try await LLMAdapterMetal(context: context, file: AnimapkFile(url: packURL)).execute(
            qwenContext: contextBuffer, contextTokens: 46,
            t5IDs: ids, t5Weights: weights, output: output
        ) { layer, residual in
            guard let name = checkpoints[layer] else { return }
            let expected = try self.floats(name, in: fixtureDirectory)
            let metric = self.metrics(residual, expected)
            print("G003_ADAPTER_LAYER_\(layer) maxAbs=\(metric.maxAbs) "
                + "rmse=\(metric.rmse) cosine=\(metric.cosine)")
            XCTAssertGreaterThanOrEqual(metric.cosine, 0.999)
            XCTAssertTrue(metric.finite)
        }
        let expected = try floats("adapter_final_padded.f32", in: fixtureDirectory)
        let metric = metrics(output, expected)
        print("G003_ADAPTER_FINAL maxAbs=\(metric.maxAbs) rmse=\(metric.rmse) "
            + "cosine=\(metric.cosine) seconds=\(Date().timeIntervalSince(start))")
        XCTAssertGreaterThanOrEqual(metric.cosine, 0.999)
        XCTAssertTrue(metric.finite)
        let pointer = output.contents().bindMemory(to: Float.self, capacity: 512 * 1_024)
        XCTAssertTrue((ids.count * 1_024..<(512 * 1_024)).allSatisfy { pointer[$0] == 0 })
    }

    func testRealW4DiTPreparationAgainstStructuralOracle() async throws {
        let bundledPack = bundledFixture(named: "i002-preparation-w4.animapk")
        let bundledLatent = bundledFixture(named: "dit_prepare_latent.f32")
        guard let packURL = ProcessInfo.processInfo.environment["ANIMAXS_PREPARATION_PACK"]
                .map(URL.init(fileURLWithPath:)) ?? bundledPack,
              let fixtureDirectory = ProcessInfo.processInfo.environment["ANIMAXS_PREPARATION_ORACLE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledLatent?.deletingLastPathComponent() else {
            throw XCTSkip("real DiT preparation W4 pack/oracle fixture not available")
        }
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let latentValues = try floats("dit_prepare_latent.f32", in: fixtureDirectory)
        XCTAssertEqual(latentValues.count, DiTPreparationExecutor.latentElements)
        let latent = makeBuffer(latentValues, context: context)
        let residual = try XCTUnwrap(context.device.makeBuffer(
            length: DiTPreparationExecutor.tokens * DiTPreparationExecutor.hidden * 4,
            options: .storageModeShared))
        let embedding = try XCTUnwrap(context.device.makeBuffer(
            length: DiTPreparationExecutor.hidden * 4, options: .storageModeShared))
        let adaln = try XCTUnwrap(context.device.makeBuffer(
            length: DiTPreparationExecutor.adaln * 4, options: .storageModeShared))
        let start = Date()
        try await DiTPreparationExecutor(
            context: context, file: AnimapkFile(url: packURL)).execute(
                latent: latent, sigma: 1, residual: residual,
                embedding: embedding, adalnLora: adaln)
        for (label, buffer, fixture) in [
            ("RESIDUAL", residual, "dit_prepare_residual.f32"),
            ("EMBEDDING", embedding, "dit_prepare_embedding.f32"),
            ("ADALN", adaln, "dit_prepare_adaln.f32")
        ] {
            let metric = metrics(buffer, try floats(fixture, in: fixtureDirectory))
            print("I002_PREPARE_\(label) maxAbs=\(metric.maxAbs) rmse=\(metric.rmse) "
                + "cosine=\(metric.cosine)")
            XCTAssertGreaterThanOrEqual(metric.cosine, 0.999)
            XCTAssertTrue(metric.finite)
        }
        print("I002_PREPARE_SECONDS=\(Date().timeIntervalSince(start))")
    }

    func testRealW4EightStepDiffusionLoop() async throws {
        let bundledPack = bundledFixture(named: "anima-turbo-v1.0-xsmax-w4.animapk")
        let bundledInitial = bundledFixture(named: "diffusion_initial.f32")
        guard let packURL = ProcessInfo.processInfo.environment["ANIMAXS_DIFFUSION_PACK"]
                .map(URL.init(fileURLWithPath:)) ?? bundledPack,
              let directory = ProcessInfo.processInfo.environment["ANIMAXS_DIFFUSION_ORACLE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledInitial?.deletingLastPathComponent() else {
            throw XCTSkip("full diffusion pack/oracle fixture not available")
        }
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let initialValues = try floats("diffusion_initial.f32", in: directory)
        let contextValues = try floats("diffusion_context.f32", in: directory)
        XCTAssertEqual(initialValues.count, DiffusionSampler.latentElements)
        XCTAssertEqual(contextValues.count, 512 * 1_024)
        let initial = makeBuffer(initialValues, context: context)
        let crossContext = makeBuffer(contextValues, context: context)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: DiffusionSampler.latentElements * 4, options: .storageModeShared))
        let start = Date()
        var completedSteps = 0
        try await DiffusionSampler(
            context: context,
            file: AnimapkFile(url: packURL)).execute(
                initialLatent: initial, crossContext: crossContext, outputLatent: output,
                blockProgress: { step, block in
                    if block == 27 { print("I002_DIFFUSION_BLOCKS step=\(step) completed=28") }
                },
                stepCompleted: { step, _, _, denoised, latent in
                    let denoisedMetric = self.metrics(
                        denoised, try self.floats("diffusion_callback_\(step).f32", in: directory))
                    print("I002_DIFFUSION_CALLBACK step=\(step) maxAbs=\(denoisedMetric.maxAbs) "
                        + "rmse=\(denoisedMetric.rmse) cosine=\(denoisedMetric.cosine)")
                    XCTAssertTrue(denoisedMetric.finite)
                    let pointer = latent.contents().bindMemory(
                        to: Float.self, capacity: DiffusionSampler.latentElements)
                    XCTAssertTrue((0..<DiffusionSampler.latentElements).allSatisfy {
                        pointer[$0].isFinite
                    })
                    completedSteps += 1
                })
        XCTAssertEqual(completedSteps, 8)
        let finalMetric = metrics(output, try floats("diffusion_final.f32", in: directory))
        print("I002_DIFFUSION_FINAL maxAbs=\(finalMetric.maxAbs) rmse=\(finalMetric.rmse) "
            + "cosine=\(finalMetric.cosine) seconds=\(Date().timeIntervalSince(start))")
        XCTAssertTrue(finalMetric.finite)
        // Source-BF16 is an observational quality baseline; every graph component
        // has a much tighter same-W4 gate. Regressions below the recorded cumulative
        // W4 baseline are still caught without pretending this is same-weight parity.
        XCTAssertGreaterThanOrEqual(finalMetric.cosine, 0.65)
    }

    /// J002: full-frame T=1 Wan VAE decode of the canonical final latent.
    /// Fixture-gated: requires the real fp16 VAE pack and the Lane A same-pack
    /// decoded RGB via ANIMAXS_VAE_ORACLE_DIR (see scripts/vae_decoder_oracle.py
    /// --emit-lane-a). Compares the Metal decoder against the Python same-pack
    /// decoder — both execute the identical fp16 pack, so this is the tight
    /// Lane A gate (D060). The canonical source RGB comparison is observational.
    func testRealW4VAEDecodeParity() async throws {
        // Resolve the real pack + Lane A fixtures from env vars (manual
        // workflow) or the bundled test fixtures, mirroring the I002 pattern.
        let bundledPack = bundledFixture(named: "qwen-image-vae-xsmax-fp16.animapk")
        let bundledLatent = bundledFixture(named: "vae_latent.f32")
        guard let packURL = ProcessInfo.processInfo.environment["ANIMAXS_VAE_PACK"]
                .map(URL.init(fileURLWithPath:)) ?? bundledPack else {
            throw XCTSkip("real VAE pack not available")
        }
        guard let directory = ProcessInfo.processInfo.environment["ANIMAXS_VAE_ORACLE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledLatent?.deletingLastPathComponent() else {
            throw XCTSkip("VAE Lane A oracle fixture not available")
        }
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let latentValues = try floats("vae_latent.f32", in: directory)
        let expectedRGB = try floats("vae_lane_a_rgb.f32", in: directory)
        XCTAssertEqual(latentValues.count, 16 * 64 * 64)
        XCTAssertEqual(expectedRGB.count, 3 * 512 * 512)
        let latent = makeBuffer(latentValues, context: context)
        let rgb = try XCTUnwrap(context.device.makeBuffer(
            length: 3 * 512 * 512 * 4, options: .storageModeShared))
        let start = Date()
        try await VAEDecoder(
            context: context,
            file: AnimapkFile(url: packURL)).execute(latent: latent, rgb: rgb)
        let seconds = Date().timeIntervalSince(start)
        let metric = metrics(rgb, expectedRGB)
        print("VAE_DECODE_FINAL maxAbs=\(metric.maxAbs) rmse=\(metric.rmse) "
            + "cosine=\(metric.cosine) seconds=\(seconds)")
        XCTAssertTrue(metric.finite)
        // Lane A: Swift/Metal on the same fp16 pack vs the Python same-pack
        // decoder. fp16 MPS accumulation tolerances; any graph/layout error
        // shows up far beyond this.
        XCTAssertGreaterThanOrEqual(metric.cosine, 0.99)
        XCTAssertLessThanOrEqual(metric.rmse, 0.05)
    }

    private func metrics(
        _ buffer: MTLBuffer, _ expected: [Float]
    ) -> (maxAbs: Double, rmse: Double, cosine: Double, finite: Bool) {
        let actual = buffer.contents().bindMemory(to: Float.self, capacity: expected.count)
        var maxAbs = 0.0, squareError = 0.0, dot = 0.0
        var actualNorm = 0.0, expectedNorm = 0.0, finite = true
        for index in expected.indices {
            let a = Double(actual[index]), e = Double(expected[index])
            finite = finite && a.isFinite
            let error = abs(a - e)
            maxAbs = max(maxAbs, error)
            squareError += error * error
            dot += a * e
            actualNorm += a * a
            expectedNorm += e * e
        }
        return (maxAbs, sqrt(squareError / Double(expected.count)),
                dot / sqrt(actualNorm * expectedNorm), finite)
    }

    private func floats(_ name: String, in directory: URL) throws -> [Float] {
        let data = try Data(contentsOf: directory.appendingPathComponent(name))
        guard data.count.isMultiple(of: 4) else {
            throw AnimapkError.validation("invalid Qwen fixture \(name)")
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func int32s(_ name: String, in directory: URL) throws -> [Int] {
        let data = try Data(contentsOf: directory.appendingPathComponent(name))
        guard data.count.isMultiple(of: 4) else {
            throw AnimapkError.validation("invalid integer fixture \(name)")
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)).map(Int.init) }
    }

    private func makeBuffer(_ values: [Float], context: MetalContext) -> MTLBuffer {
        values.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                      options: .storageModeShared)!
        }
    }

    private func bundledFixture(named name: String) -> URL? {
        guard let root = Bundle(for: Self.self).resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: nil) else { return nil }
        for case let url as URL in enumerator where url.lastPathComponent == name { return url }
        return nil
    }
}
