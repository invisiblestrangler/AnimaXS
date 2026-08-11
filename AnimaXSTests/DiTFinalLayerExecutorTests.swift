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
}
