import XCTest
import Metal
@testable import AnimaXS

/// Phase 12 — two-slot ping-pong weight streamer correctness.
final class WeightStreamerTests: XCTestCase {

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("WeightStreamerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Build a minimal valid W4 pack (16KB-aligned blobs) via TestPackFactory.
    /// The pack file is intentionally NOT removed here — the returned URL must
    /// stay valid for the duration of the test.
    private func makePack() throws -> URL {
        let dir = try makeTempDir()
        let k = 64
        let data = Data(repeating: 0xAB, count: k / 2)
        let scale = Data(repeating: 0, count: (k / 64) * 2)
        let zero = Data(repeating: 0, count: (k / 64) * 2)
        let spec = TestPackFactory.BlobSpec(
            name: "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            shape: [8, k], logicalDtype: "fp16", storageDtype: "w4",
            storageCode: 0, data: data, scale: scale, zero: zero)
        return try TestPackFactory.writePack(
            named: "t", componentCode: 1, quantScheme: "w4g64", quantGroup: 64,
            blobs: [spec], to: dir)
    }

    /// Build an AnimapkExecutionRange pointing at the single blob.
    private func range(file: AnimapkFile, index: Int) throws -> AnimapkExecutionRange {
        let tensor = try XCTUnwrap(file.tensors.first)
        return AnimapkExecutionRange(
            logicalIndex: index,
            fileOffset: tensor.blobOffset,
            length: tensor.blobSize,
            tensors: [AnimapkTensorSpans(
                tensor: tensor,
                blob: .init(offset: 0, length: tensor.blobSize),
                data: .init(offset: 0, length: tensor.dataSize),
                scale: tensor.scaleOffset.map { .init(offset: $0, length: tensor.scaleSize ?? 0) },
                zero: tensor.zeroOffset.map { .init(offset: $0, length: tensor.zeroSize ?? 0) }
            )])
    }

    func testTwoSlotsAllocateDistinctBuffers() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 1024, slotCount: 2)
        XCTAssertEqual(streamer.slotCount, 2)
        XCTAssertFalse(streamer.buffer(for: 0) === streamer.buffer(for: 1))
        XCTAssertNotEqual(
            streamer.buffer(for: 0).contents(),
            streamer.buffer(for: 1).contents())
    }

    func testLoadIntoDistinctSlotsIsIndependent() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 2)
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let r0 = try range(file: file, index: 0)
        try streamer.load(r0, from: file, slot: 0)
        XCTAssertEqual(streamer.loadedLogicalIndexes[0], 0)
        XCTAssertNil(streamer.loadedLogicalIndexes[1])
    }

    func testInFlightSlotCannotBeOverwritten() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 2)
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let r0 = try range(file: file, index: 0)
        try streamer.load(r0, from: file, slot: 0)
        streamer.markInFlight(0)
        // Loading into the in-flight slot must throw.
        XCTAssertThrowsError(try streamer.load(r0, from: file, slot: 0)) { error in
            guard case AnimapkError.validation = error else {
                return XCTFail("expected validation error, got \(error)")
            }
        }
        // The OTHER slot is still usable while slot 0 is in flight.
        XCTAssertNoThrow(try streamer.load(r0, from: file, slot: 1))
        // Releasing slot 0 allows overwrite again.
        streamer.complete(0)
        XCTAssertNoThrow(try streamer.load(r0, from: file, slot: 0))
    }

    func testInFlightSlotSetTracksLifecycle() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 2)
        streamer.markInFlight(0)
        XCTAssertEqual(streamer.inFlightSlotSet, [0])
        streamer.markInFlight(1)
        XCTAssertEqual(streamer.inFlightSlotSet, [0, 1])
        streamer.complete(0)
        XCTAssertEqual(streamer.inFlightSlotSet, [1])
        streamer.complete(1)
        XCTAssertTrue(streamer.inFlightSlotSet.isEmpty)
    }

    func testSlotCountValidation() {
        let device = try? XCTUnwrap(MTLCreateSystemDefaultDevice())
        guard let device else { return }
        XCTAssertThrowsError(try WeightStreamer(device: device, capacity: 64, slotCount: 0))
        XCTAssertThrowsError(try WeightStreamer(device: device, capacity: 64, slotCount: 5))
        XCTAssertNoThrow(try WeightStreamer(device: device, capacity: 64, slotCount: 1))
        XCTAssertNoThrow(try WeightStreamer(device: device, capacity: 64, slotCount: 4))
    }
}
