import XCTest
@testable import AnimaXS

final class VAEDecoderLocatorTests: XCTestCase {
    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vae-locator-tests-\\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    /// Build a synthetic fp16 pack carrying the full fixed decoder architecture.
    /// Data payloads are minimal (metadata is what the locator validates).
    private func writeArchitecturePack(
        named name: String,
        mutating: (inout [TestPackFactory.BlobSpec]) -> Void = { _ in }
    ) throws -> URL {
        var specs: [TestPackFactory.BlobSpec] = []
        for (tensorName, shape) in VAEDecoderLocator.expectedArchitecture.sorted(by: { $0.key < $1.key }) {
            specs.append(TestPackFactory.BlobSpec(
                name: tensorName, shape: shape,
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([0, 0])))
        }
        mutating(&specs)
        return try TestPackFactory.writePack(
            named: name, componentCode: 3, quantScheme: "fp16", quantGroup: 64,
            blobs: specs, to: tmpDir)
    }

    func testValidatesCompleteArchitectureAndBuildsDisjointGroups() throws {
        let url = try writeArchitecturePack(named: "vae-arch")
        let locator = try VAEDecoderLocator(file: AnimapkFile(url: url))

        // 21 logical groups: conv2, decoder.conv1, middle x3, 15 upsample
        // modules, head.
        XCTAssertEqual(locator.groups.count, 21)
        XCTAssertEqual(locator.groups.map(\\.logicalIndex), Array(0..<21))

        // Every group is non-empty, aligned, and physically disjoint.
        let physical = locator.groups.sorted { $0.range.fileOffset < $1.range.fileOffset }
        for pair in zip(physical, physical.dropFirst()) {
            XCTAssertLessThanOrEqual(pair.0.range.fileRange.upperBound, pair.1.range.fileRange.lowerBound)
        }
        for group in locator.groups {
            XCTAssertFalse(group.range.tensors.isEmpty)
            XCTAssertEqual(group.range.fileOffset % 16_384, 0)
            XCTAssertLessThanOrEqual(group.range.length, try locator.maximumGroupLength())
        }

        // Unexecuted time_conv tensors exist in the pack but never appear in a
        // streaming group.
        let groupNames = Set(locator.groups.flatMap { $0.range.tensors.map(\\.tensor.name) })
        XCTAssertFalse(groupNames.contains("decoder.upsamples.3.time_conv.weight"))
        XCTAssertFalse(groupNames.contains("decoder.upsamples.7.time_conv.weight"))

        // Group 3 is the middle attention (to_qkv/proj), group 4 the upsample
        // stage-0 first residual.
        let attention = try locator.group(3)
        XCTAssertEqual(
            Set(attention.range.tensors.map(\\.tensor.name)),
            Set(["decoder.middle.1.norm.gamma", "decoder.middle.1.to_qkv.weight",
                 "decoder.middle.1.to_qkv.bias", "decoder.middle.1.proj.weight",
                 "decoder.middle.1.proj.bias"]))
    }

    func testMissingTensorFails() throws {
        let url = try writeArchitecturePack(named: "vae-missing") { specs in
            specs.removeAll { $0.name == "decoder.conv1.weight" }
        }
        XCTAssertThrowsError(try VAEDecoderLocator(file: AnimapkFile(url: url))) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("decoder.conv1.weight"))
        }
    }

    func testWrongShapeFails() throws {
        let url = try writeArchitecturePack(named: "vae-shape") { specs in
            specs.removeAll { $0.name == "decoder.head.2.weight" }
            specs.append(TestPackFactory.BlobSpec(
                name: "decoder.head.2.weight", shape: [3, 96, 1, 1, 1],
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([0, 0])))
        }
        XCTAssertThrowsError(try VAEDecoderLocator(file: AnimapkFile(url: url))) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("decoder.head.2.weight"))
        }
    }

    func testUnexpectedDecoderTensorFails() throws {
        let url = try writeArchitecturePack(named: "vae-extra") { specs in
            specs.append(TestPackFactory.BlobSpec(
                name: "decoder.upsamples.0.surprise.weight", shape: [1],
                logicalDtype: "fp16", storageDtype: "fp16", storageCode: 2,
                data: Data([0, 0])))
        }
        XCTAssertThrowsError(try VAEDecoderLocator(file: AnimapkFile(url: url))) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("surprise"))
        }
    }

    func testNonFP16StorageFails() throws {
        let url = try writeArchitecturePack(named: "vae-w8") { specs in
            specs.removeAll { $0.name == "conv2.weight" }
            specs.append(TestPackFactory.BlobSpec(
                name: "conv2.weight", shape: [16, 16, 1, 1, 1],
                logicalDtype: "w8", storageDtype: "w8", storageCode: 1,
                data: Data([0, 0]), scale: Data([0, 0]), zero: Data([0, 0])))
        }
        XCTAssertThrowsError(try VAEDecoderLocator(file: AnimapkFile(url: url)))
    }
}
