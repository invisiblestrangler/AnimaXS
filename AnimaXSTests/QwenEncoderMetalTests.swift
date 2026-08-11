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
            XCTAssertGreaterThanOrEqual(metric.cosine, 0.995)
            XCTAssertTrue(metric.finite)
        }
        let expected = try floats("qwen_final.f32", in: fixtureDirectory)
        let metric = metrics(output, expected)
        print("F007_QWEN_FINAL maxAbs=\(metric.maxAbs) rmse=\(metric.rmse) "
            + "cosine=\(metric.cosine) seconds=\(Date().timeIntervalSince(start))")
        XCTAssertGreaterThanOrEqual(metric.cosine, 0.995)
        XCTAssertTrue(metric.finite)
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

    private func bundledFixture(named name: String) -> URL? {
        guard let root = Bundle(for: Self.self).resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: nil) else { return nil }
        for case let url as URL in enumerator where url.lastPathComponent == name { return url }
        return nil
    }
}
