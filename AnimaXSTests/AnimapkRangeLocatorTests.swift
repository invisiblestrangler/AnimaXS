import XCTest
@testable import AnimaXS

final class AnimapkRangeLocatorTests: XCTestCase {
    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("animapk-range-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    func testDiTLogicalOrderAndRelativeSpans() throws {
        let physicalOrder = (0..<DiTBlockLocator.blockCount).sorted {
            String($0) < String($1)
        }
        let blobs = physicalOrder.map { index in
            TestPackFactory.BlobSpec(
                name: "model.diffusion_model.blocks.\(index).weight",
                shape: [4], logicalDtype: "w4", storageDtype: "w4", storageCode: 0,
                data: Data([UInt8(index), 0]), scale: Data([0, 0]), zero: Data([0, 0]))
        }
        let url = try TestPackFactory.writePack(
            named: "dit-ranges", componentCode: 1, quantScheme: "w4", quantGroup: 64,
            blobs: blobs, to: tmpDir)
        let file = try AnimapkFile(url: url)
        let locator = try DiTBlockLocator(file: file)

        XCTAssertEqual(locator.blocks.map(\.logicalIndex), Array(0..<28))
        XCTAssertEqual(Set(locator.blocks.map(\.fileOffset)).count, 28)
        for block in locator.blocks {
            XCTAssertEqual(block.tensors.count, 1)
            XCTAssertEqual(block.fileOffset % 16_384, 0)
            XCTAssertEqual(block.tensors[0].blob.offset, 0)
            XCTAssertEqual(block.tensors[0].data, AnimapkRelativeSpan(offset: 0, length: 2))
            XCTAssertEqual(block.tensors[0].scale, AnimapkRelativeSpan(offset: 2, length: 2))
            XCTAssertEqual(block.tensors[0].zero, AnimapkRelativeSpan(offset: 4, length: 2))
            let mapped = try file.bytes(in: block.fileRange)
            XCTAssertEqual(mapped[0], UInt8(block.logicalIndex))
        }

        // Lexicographic physical order differs from numerical execution order.
        XCTAssertLessThan(try locator.block(10).fileOffset, try locator.block(2).fileOffset)
        let physical = locator.blocks.sorted { $0.fileOffset < $1.fileOffset }
        for pair in zip(physical, physical.dropFirst()) {
            XCTAssertLessThanOrEqual(pair.0.fileRange.upperBound, pair.1.fileRange.lowerBound)
        }
    }

    func testQwenLayersAndEmbeddingRows() throws {
        let rows = 3
        let columns = 65
        var specs: [TestPackFactory.BlobSpec] = [
            TestPackFactory.BlobSpec(
                name: "model.embed_tokens.weight", shape: [rows, columns],
                logicalDtype: "w8", storageDtype: "w8", storageCode: 1,
                data: Data((0..<(rows * columns)).map { UInt8($0 & 0xff) }),
                scale: Data(repeating: 1, count: rows * 2 * 2),
                zero: Data(repeating: 2, count: rows * 2 * 2))
        ]
        specs += (0..<QwenLayerLocator.layerCount).map { index in
            TestPackFactory.BlobSpec(
                name: "model.layers.\(index).weight", shape: [1],
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([UInt8(index), 0]))
        }
        specs.sort { $0.name < $1.name }
        let url = try TestPackFactory.writePack(
            named: "qwen-ranges", componentCode: 2, quantScheme: "w8", quantGroup: 64,
            blobs: specs, to: tmpDir)
        let file = try AnimapkFile(url: url)
        let locator = try QwenLayerLocator(file: file)

        XCTAssertEqual(locator.layers.map(\.logicalIndex), Array(0..<28))
        XCTAssertLessThan(try locator.layer(10).fileOffset, try locator.layer(2).fileOffset)
        let row = try locator.embeddingRow(1)
        XCTAssertEqual(row.data, AnimapkRelativeSpan(offset: 65, length: 65))
        XCTAssertEqual(row.scale, AnimapkRelativeSpan(offset: 195 + 4, length: 4))
        XCTAssertEqual(row.zero, AnimapkRelativeSpan(offset: 195 + 12 + 4, length: 4))
        XCTAssertThrowsError(try locator.embeddingRow(rows))

        let rowFileStart = locator.embeddingFileOffset + row.data.offset
        let mapped = try file.bytes(in: rowFileStart..<(rowFileStart + row.data.length))
        XCTAssertEqual(mapped[0], UInt8(65))
        XCTAssertEqual(mapped.count, columns)
    }

    func testAdapterBlocksEmbeddingRowsAndFinalRange() throws {
        let prefix = "model.diffusion_model.llm_adapter."
        var specs = (0..<LLMAdapterLocator.blockCount).map { index in
            TestPackFactory.BlobSpec(
                name: prefix + "blocks.\(index).weight", shape: [1],
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([UInt8(index), 0]))
        }
        specs += [
            TestPackFactory.BlobSpec(
                name: prefix + "embed.weight", shape: [3, 64],
                logicalDtype: "w4", storageDtype: "w4", storageCode: 0,
                data: Data((0..<96).map(UInt8.init)),
                scale: Data(repeating: 1, count: 6), zero: Data(repeating: 2, count: 6)),
            TestPackFactory.BlobSpec(
                name: prefix + "norm.weight", shape: [1],
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([0, 0])),
            TestPackFactory.BlobSpec(
                name: prefix + "out_proj.bias", shape: [1],
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([0, 0])),
            TestPackFactory.BlobSpec(
                name: prefix + "out_proj.weight", shape: [2, 2],
                logicalDtype: "w4", storageDtype: "w4", storageCode: 0,
                data: Data([0, 0]), scale: Data([0, 0, 0, 0]),
                zero: Data([0, 0, 0, 0]))
        ]
        specs.sort { $0.name < $1.name }
        let url = try TestPackFactory.writePack(
            named: "adapter-ranges", componentCode: 1, quantScheme: "w4", quantGroup: 64,
            blobs: specs, to: tmpDir)
        let locator = try LLMAdapterLocator(file: AnimapkFile(url: url))

        XCTAssertEqual(locator.blocks.map(\.logicalIndex), Array(0..<6))
        XCTAssertLessThan(try locator.block(0).fileOffset, try locator.block(5).fileOffset)
        XCTAssertEqual(locator.final.tensors.count, 3)
        XCTAssertFalse(locator.final.tensors.contains { $0.tensor.name.contains("embed.weight") })
        let row = try locator.embeddingRow(2)
        XCTAssertEqual(row.data.length, 32)
        XCTAssertEqual(row.data.offset, 64)
        XCTAssertEqual(row.scale.length, 2)
        XCTAssertEqual(row.zero.length, 2)
        XCTAssertThrowsError(try locator.embeddingRow(3))
    }
}
