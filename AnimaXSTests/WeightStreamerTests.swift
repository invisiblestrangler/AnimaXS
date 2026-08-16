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
    /// The blob payload is sized to a 4096 multiple so the single-blob range
    /// satisfies the P6 page-aligned START AND END eligibility (blob offsets
    /// are 16 KB-aligned by the factory; a 4096-byte blob length makes the
    /// end 4096-aligned too). The pack file is intentionally NOT removed here
    /// — the returned URL must stay valid for the duration of the test.
    private func makePack() throws -> URL {
        let dir = try makeTempDir()
        let k = 64
        // 4092 bytes + 2 scale + 2 zero = 4096-byte blob (page-aligned length).
        let data = Data(repeating: 0xAB, count: 4_092)
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

    // MARK: - P6 mmap no-copy weight source

    /// P6 eligibility is pure logic: TestPackFactory blobs are 16 KB-aligned,
    /// so the single-blob range's absolute file offset is 4096-page-aligned
    /// and inside the file.
    func testP6EligibleRangeIsPageAligned() throws {
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let range = try self.range(file: file, index: 0)
        XCTAssertTrue(WeightNoCopyPolicy.isEligible(range: range, file: file))
        XCTAssertEqual(Int(range.fileRange.lowerBound) % 4_096, 0)
    }

    /// P6: an execution range whose absolute file offset is NOT page-aligned
    /// is ineligible — the no-copy path must never alias a misaligned pointer.
    func testP6NonPageAlignedRangeIsIneligible() throws {
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        // Offset 256 (the JSON section) is in-file but not 4096-aligned.
        let misaligned = AnimapkExecutionRange(
            logicalIndex: 99, fileOffset: 256, length: 64, tensors: [])
        XCTAssertFalse(WeightNoCopyPolicy.isEligible(range: misaligned, file: file))
    }

    /// P6 HARDENED (Task 5): a range with a page-aligned START but a
    /// NON-page-aligned length/end is ineligible — BOTH the start and the end
    /// of the aliased region must sit on page boundaries. Correctness
    /// hardening only: it does NOT claim to prove the historical A12 GPU page
    /// fault (kIOGPUCommandBufferCallbackErrorPageFault) was caused by the
    /// no-copy path.
    func testP6PageAlignedStartButNonPageAlignedEndIsIneligible() throws {
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        // Start is 4096-aligned but the length (and therefore the end) is not.
        let partial = AnimapkExecutionRange(
            logicalIndex: 99, fileOffset: 16_384, length: 4_096 + 64, tensors: [])
        XCTAssertEqual(Int(partial.fileRange.lowerBound) % 4_096, 0,
                       "sanity: start must be page-aligned")
        XCTAssertNotEqual(Int(partial.fileRange.upperBound) % 4_096, 0,
                          "sanity: end must NOT be page-aligned")
        XCTAssertFalse(WeightNoCopyPolicy.isEligible(range: partial, file: file),
                       "page-aligned start with non-page-aligned end must be ineligible")
    }

    /// P6 HARDENED (Task 5): a range whose start AND end are both
    /// page-aligned remains eligible (the existing eligibility contract,
    /// now with the end requirement explicit).
    func testP6PageAlignedStartAndEndIsEligible() throws {
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let aligned = AnimapkExecutionRange(
            logicalIndex: 99, fileOffset: 16_384, length: 8_192, tensors: [])
        XCTAssertEqual(Int(aligned.fileRange.lowerBound) % 4_096, 0)
        XCTAssertEqual(Int(aligned.fileRange.upperBound) % 4_096, 0)
        XCTAssertTrue(WeightNoCopyPolicy.isEligible(range: aligned, file: file))
    }

    /// P6 HARDENED (Task 5): a fully page-aligned range that runs past EOF is
    /// still ineligible (bounds check unchanged).
    func testP6PageAlignedRangePastEOFIsIneligible() throws {
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        // Start + end aligned, but the range extends past the file's size.
        let pastEOF = AnimapkExecutionRange(
            logicalIndex: 99, fileOffset: 16_384,
            length: UInt64(file.header.fileSize) + 4_096, tensors: [])
        XCTAssertFalse(WeightNoCopyPolicy.isEligible(range: pastEOF, file: file))
    }

    /// P6-A/B: loading an eligible range with `.noCopy` yields an MTLBuffer
    /// that ALIASES the file mapping — its bytes equal the file bytes with NO
    /// memcpy in the test setup (the pack is read zero-copy via
    /// `file.bytes(in:)`), and `buffer(for:)` returns the alias.
    func testP6NoCopyLoadAliasesFileBytes() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 1)
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let range = try self.range(file: file, index: 0)
        let result = try streamer.load(range, from: file, slot: 0, mode: .noCopy)
        guard result.mode == .noCopy else {
            throw XCTSkip("SKIPPED_NO_BYTES_NO_COPY: device refused the mmap alias")
        }
        XCTAssertEqual(result.noCopyBytes, range.length)
        XCTAssertTrue(streamer.buffer(for: 0) === result.buffer)
        let fileBytes = try file.bytes(in: range.fileRange)
        let alias = result.buffer.contents().bindMemory(to: UInt8.self, capacity: Int(range.length))
        let reference = fileBytes.bindMemory(to: UInt8.self)
        // Compare a few offsets across the range (no full copy anywhere).
        for offset in [0, 1, 7, Int(range.length) / 2, Int(range.length) - 1] {
            XCTAssertEqual(alias[offset], reference[offset], "byte at offset \(offset)")
        }
        XCTAssertEqual(streamer.loadedLogicalIndexes[0], 0)
    }

    /// P6-B: a non-page-aligned range requested with `.noCopy` falls back to
    /// the copied path — never an alias, bytes identical to the file.
    func testP6NonPageAlignedRangeFallsBackToCopied() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 1)
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let misaligned = AnimapkExecutionRange(
            logicalIndex: 99, fileOffset: 256, length: 64, tensors: [])
        let result = try streamer.load(misaligned, from: file, slot: 0, mode: .noCopy)
        XCTAssertEqual(result.mode, .copied)
        XCTAssertEqual(result.noCopyBytes, 0)
        XCTAssertTrue(streamer.buffer(for: 0) === result.buffer)
        let fileBytes = try file.bytes(in: misaligned.fileRange)
        let copied = result.buffer.contents().bindMemory(to: UInt8.self, capacity: 64)
        let reference = fileBytes.bindMemory(to: UInt8.self)
        for offset in [0, 32, 63] {
            XCTAssertEqual(copied[offset], reference[offset], "byte at offset \(offset)")
        }
    }

    /// P6 regression: the copied backend is byte-identical to the historical
    /// behavior — load with `.copied` returns the slot ring buffer, memcpys
    /// the range, and `buffer(for:)` reports the ring.
    func testP6CopiedBackendIsByteIdenticalToCurrentBehavior() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 1)
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let range = try self.range(file: file, index: 0)
        let result = try streamer.load(range, from: file, slot: 0, mode: .copied)
        XCTAssertEqual(result.mode, .copied)
        XCTAssertEqual(result.noCopyBytes, 0)
        XCTAssertTrue(result.buffer === streamer.buffer(for: 0))
        XCTAssertEqual(streamer.loadedLogicalIndexes[0], 0)
        let fileBytes = try file.bytes(in: range.fileRange)
        let copied = result.buffer.contents().bindMemory(to: UInt8.self, capacity: Int(range.length))
        let reference = fileBytes.bindMemory(to: UInt8.self)
        for offset in [0, 16, Int(range.length) / 2, Int(range.length) - 1] {
            XCTAssertEqual(copied[offset], reference[offset], "byte at offset \(offset)")
        }
    }

    /// P6-C: the no-copy buffer aliases the file mmap, so its contents stay
    /// valid while the file lives; loading a SECOND range into another slot
    /// must not disturb the first alias (the deallocator never frees the
    /// mmap'd pointer).
    func testP6NoCopyBufferLifetimeTracksFile() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let streamer = try WeightStreamer(device: device, capacity: 4096, slotCount: 2)
        let url = try makePack()
        let file = try AnimapkFile(url: url)
        let range = try self.range(file: file, index: 0)
        let first = try streamer.load(range, from: file, slot: 0, mode: .noCopy)
        guard first.mode == .noCopy else {
            throw XCTSkip("SKIPPED_NO_BYTES_NO_COPY: device refused the mmap alias")
        }
        // Load the same range again into the OTHER slot (copied path). The
        // first slot's alias must still reflect the file bytes.
        let second = try streamer.load(range, from: file, slot: 1, mode: .copied)
        XCTAssertEqual(second.mode, .copied)
        let alias = first.buffer.contents().bindMemory(to: UInt8.self, capacity: Int(range.length))
        let fileBytes = try file.bytes(in: range.fileRange)
        let reference = fileBytes.bindMemory(to: UInt8.self)
        for offset in [0, Int(range.length) / 2, Int(range.length) - 1] {
            XCTAssertEqual(alias[offset], reference[offset], "byte at offset \(offset)")
        }
    }
}
