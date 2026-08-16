import XCTest
import Metal
@testable import AnimaXS

final class DiTFinalLayerExecutorTests: XCTestCase {
    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dit-final-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    func testFinalLayerStreamsExactRangeAndProducesVelocityLayout() async throws {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let url = try TestPackFactory.writePack(
            named: "dit-final", componentCode: 1, quantScheme: "w4", quantGroup: 64,
            blobs: [
                zeroW4("model.diffusion_model.final_layer.adaln_modulation.1.weight",
                       rows: 256, columns: 2_048),
                zeroW4("model.diffusion_model.final_layer.adaln_modulation.2.weight",
                       rows: 4_096, columns: 256),
                zeroW4("model.diffusion_model.final_layer.linear.weight",
                       rows: 64, columns: 2_048),
            ], to: tmpDir)
        let file = try AnimapkFile(url: url)
        let located = try DiTFinalLayerLocator(file: file).range
        XCTAssertEqual(located.logicalIndex, -1)
        XCTAssertEqual(located.tensors.count, 3)
        XCTAssertEqual(located.fileOffset % 16_384, 0)

        let residual = try sharedBuffer(bytes: 1_024 * 2_048 * 4, context: context)
        let emb = try sharedBuffer(bytes: 2_048 * 4, context: context)
        let adaln = try sharedBuffer(bytes: 4_096 * 4, context: context)
        let velocity = try sharedBuffer(bytes: 16 * 64 * 64 * 4, context: context)
        let residualValues = residual.contents().bindMemory(
            to: Float.self, capacity: 1_024 * 2_048)
        for index in 0..<(1_024 * 2_048) {
            residualValues[index] = Float(index % 31 - 15) / 16
        }
        let output = velocity.contents().bindMemory(to: Float.self, capacity: 16 * 64 * 64)
        for index in 0..<(16 * 64 * 64) { output[index] = 1_234_567 }

        try await DiTFinalLayerExecutor(context: context, file: file).execute(
            residual: residual, emb: emb, adalnLora: adaln, velocity: velocity)

        XCTAssertTrue((0..<(16 * 64 * 64)).allSatisfy { output[$0] == 0 })
        print("H007_FINAL_LAYER_SYNTHETIC=PASS rangeBytes=\(located.length)")
    }

    // P0: production W8-v2 resolves to w8LegacyStabilized (legacy
    // activation/attention numerics), but its final-residual ENTRY boundary is
    // decoupled to .bf16RNEInFP32: the large residual is rounded to BF16 RNE
    // while retained in fp32 storage, so residual magnitudes above the FP16 max
    // (65,504) must stay finite through the final layer. The legacy W4 boundary
    // (.fp16Legacy) converts the residual through fp16 and would overflow.
    func testW8ProductionFinalBoundaryKeepsLargeResidualFinite() async throws {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let url = try TestPackFactory.writePack(
            named: "dit-final-w8", componentCode: 1, quantScheme: "w4", quantGroup: 64,
            blobs: [
                zeroW4("model.diffusion_model.final_layer.adaln_modulation.1.weight",
                       rows: 256, columns: 2_048),
                zeroW4("model.diffusion_model.final_layer.adaln_modulation.2.weight",
                       rows: 4_096, columns: 256),
                zeroW4("model.diffusion_model.final_layer.linear.weight",
                       rows: 64, columns: 2_048),
            ], to: tmpDir)
        let file = try AnimapkFile(url: url)

        let magnitudes: [Float] = [60_000, 70_000, 100_000, 280_000]
        for magnitude in magnitudes {
            let residual = try sharedBuffer(bytes: 1_024 * 2_048 * 4, context: context)
            let emb = try sharedBuffer(bytes: 2_048 * 4, context: context)
            let adaln = try sharedBuffer(bytes: 4_096 * 4, context: context)
            let velocity = try sharedBuffer(bytes: 16 * 64 * 64 * 4, context: context)
            let residualValues = residual.contents().bindMemory(
                to: Float.self, capacity: 1_024 * 2_048)
            // Alternate positive/negative so LayerNorm sees real variance.
            for index in 0..<(1_024 * 2_048) {
                residualValues[index] = magnitude * (index.isMultiple(of: 2) ? 1 : -1)
            }
            let output = velocity.contents().bindMemory(
                to: Float.self, capacity: 16 * 64 * 64)
            for index in 0..<(16 * 64 * 64) { output[index] = 1_234_567 }

            // Production W8 contract: LEGACY activation numerics (w8LegacyStabilized)
            // with the decoupled BF16-RNE-in-FP32 final-residual boundary.
            try await DiTFinalLayerExecutor(
                context: context, file: file,
                activationNumerics: .legacy,
                finalResidualBoundary: .bf16RNEInFP32).execute(
                    residual: residual, emb: emb, adalnLora: adaln, velocity: velocity)

            XCTAssertTrue((0..<(16 * 64 * 64)).allSatisfy { output[$0].isFinite },
                          "production W8 final boundary produced non-finite output at residual magnitude \(magnitude)")
        }
        print("H007_W8_PRODUCTION_FINAL_BOUNDARY=PASS magnitudes=60000,70000,100000,280000")
    }

    // W4 contract: the policy resolves to .fp16Legacy, and the executor's
    // default boundary keeps the byte-for-byte FP16 residual path. This test
    // does NOT feed >65,504 through the W4 path and require finite — that
    // would redefine W4. It only asserts the small-residual W4 path is intact.
    func testW4FinalBoundaryResolvesToFP16LegacyAndExecutes() async throws {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        XCTAssertEqual(DiffusionSampler.resolvedFinalResidualBoundary(for: .w4Legacy),
                       .fp16Legacy)
        let url = try TestPackFactory.writePack(
            named: "dit-final-w4", componentCode: 1, quantScheme: "w4", quantGroup: 64,
            blobs: [
                zeroW4("model.diffusion_model.final_layer.adaln_modulation.1.weight",
                       rows: 256, columns: 2_048),
                zeroW4("model.diffusion_model.final_layer.adaln_modulation.2.weight",
                       rows: 4_096, columns: 256),
                zeroW4("model.diffusion_model.final_layer.linear.weight",
                       rows: 64, columns: 2_048),
            ], to: tmpDir)
        let file = try AnimapkFile(url: url)

        let residual = try sharedBuffer(bytes: 1_024 * 2_048 * 4, context: context)
        let emb = try sharedBuffer(bytes: 2_048 * 4, context: context)
        let adaln = try sharedBuffer(bytes: 4_096 * 4, context: context)
        let velocity = try sharedBuffer(bytes: 16 * 64 * 64 * 4, context: context)
        let residualValues = residual.contents().bindMemory(
            to: Float.self, capacity: 1_024 * 2_048)
        for index in 0..<(1_024 * 2_048) {
            residualValues[index] = Float(index % 31 - 15) / 16
        }
        let output = velocity.contents().bindMemory(to: Float.self, capacity: 16 * 64 * 64)
        for index in 0..<(16 * 64 * 64) { output[index] = 1_234_567 }

        // Default boundary is .fp16Legacy (W4 path, byte-for-byte unchanged).
        try await DiTFinalLayerExecutor(context: context, file: file).execute(
            residual: residual, emb: emb, adalnLora: adaln, velocity: velocity)

        XCTAssertTrue((0..<(16 * 64 * 64)).allSatisfy { output[$0] == 0 },
                      "W4 FP16-legacy final boundary must execute unchanged")
        print("H007_W4_FINAL_BOUNDARY=PASS boundary=fp16Legacy")
    }

    func testRealFinalLayerAgainstSameW4Oracle() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundled = bundledFixture(named: "h007_final.animapk")?.deletingLastPathComponent()
        guard let directory = environment["ANIMAXS_H007_FIXTURE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundled else {
            throw XCTSkip("H007 real-weight fixture not available")
        }
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let file = try AnimapkFile(url: directory.appendingPathComponent("h007_final.animapk"))
        let residualValues = try floats("h007_residual.f32", in: directory)
        let expected = try floats("h007_expected.f32", in: directory)
        let residual = buffer(residualValues, context: context)
        let emb = buffer(try floats("h007_emb.f32", in: directory), context: context)
        let adaln = buffer(try floats("h007_adaln.f32", in: directory), context: context)
        let velocity = try sharedBuffer(bytes: expected.count * 4, context: context)

        try await DiTFinalLayerExecutor(context: context, file: file).execute(
            residual: residual, emb: emb, adalnLora: adaln, velocity: velocity)

        let actual = velocity.contents().bindMemory(to: Float.self, capacity: expected.count)
        var maxAbsolute = 0.0, squareError = 0.0
        var dot = 0.0, actualNorm = 0.0, expectedNorm = 0.0
        for index in expected.indices {
            let a = Double(actual[index]), e = Double(expected[index])
            XCTAssertTrue(a.isFinite)
            let error = abs(a - e)
            maxAbsolute = max(maxAbsolute, error)
            squareError += error * error
            dot += a * e
            actualNorm += a * a
            expectedNorm += e * e
        }
        let rmse = sqrt(squareError / Double(expected.count))
        let cosine = dot / sqrt(actualNorm * expectedNorm)
        print("H007_FINAL_REAL maxAbs=\(maxAbsolute) rmse=\(rmse) cosine=\(cosine)")
        XCTAssertGreaterThanOrEqual(cosine, 0.999)
        XCTAssertLessThan(rmse, 0.1)
    }

    private func zeroW4(_ name: String, rows: Int, columns: Int) -> TestPackFactory.BlobSpec {
        let groups = (columns + 63) / 64
        return TestPackFactory.BlobSpec(
            name: name, shape: [rows, columns], logicalDtype: "w4",
            storageDtype: "w4", storageCode: 0,
            data: Data(repeating: 0, count: rows * ((columns + 1) / 2)),
            scale: Data(repeating: 0, count: rows * groups * 2),
            zero: Data(repeating: 0, count: rows * groups * 2))
    }

    private func sharedBuffer(bytes: Int, context: MetalContext) throws -> MTLBuffer {
        let buffer = try XCTUnwrap(context.device.makeBuffer(
            length: bytes, options: .storageModeShared))
        memset(buffer.contents(), 0, bytes)
        return buffer
    }

    private func buffer<T>(_ values: [T], context: MetalContext) -> MTLBuffer {
        values.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                      options: .storageModeShared)!
        }
    }

    private func floats(_ name: String, in directory: URL) throws -> [Float] {
        let data = try Data(contentsOf: directory.appendingPathComponent(name))
        guard data.count.isMultiple(of: 4) else {
            throw AnimapkError.validation("invalid H007 fixture \(name)")
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func bundledFixture(named name: String) -> URL? {
        guard let root = Bundle(for: Self.self).resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: nil) else { return nil }
        for case let url as URL in enumerator where url.lastPathComponent == name { return url }
        return nil
    }
}
