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

    /// Build a synthetic fp16 pack from explicit tensor specs. Small enough for
    /// TestPackFactory's fixed 16 KB JSON region.
    private func writePack(named name: String, specs: [TestPackFactory.BlobSpec]) throws -> URL {
        try TestPackFactory.writePack(
            named: name, componentCode: 3, quantScheme: "fp16", quantGroup: 64,
            blobs: specs, to: tmpDir)
    }

    private func blob(_ name: String, _ shape: [Int],
                      storage: String = "fp16", code: UInt8 = 2) -> TestPackFactory.BlobSpec {
        TestPackFactory.BlobSpec(
            name: name, shape: shape, logicalDtype: storage, storageDtype: storage,
            storageCode: code, data: Data([0, 0]))
    }

    func testValidatesCompleteArchitectureAndBuildsDisjointGroups() throws {
        // A realistic subset of groups (well under the JSON limit): conv2,
        // decoder.conv1, middle attention, two residual groups, one resample
        // group, one shortcut group, and the head. Every tensor must carry the
        // EXACT real shape so the group-building validation passes.
        let specs: [TestPackFactory.BlobSpec] = [
            blob("conv2.weight", [16, 16, 1, 1, 1]),
            blob("conv2.bias", [16]),
            blob("decoder.conv1.weight", [384, 16, 3, 3, 3]),
            blob("decoder.conv1.bias", [384]),
            blob("decoder.middle.0.residual.0.gamma", [384, 1, 1, 1]),
            blob("decoder.middle.0.residual.2.weight", [384, 384, 3, 3, 3]),
            blob("decoder.middle.0.residual.2.bias", [384]),
            blob("decoder.middle.0.residual.3.gamma", [384, 1, 1, 1]),
            blob("decoder.middle.0.residual.6.weight", [384, 384, 3, 3, 3]),
            blob("decoder.middle.0.residual.6.bias", [384]),
            blob("decoder.middle.1.norm.gamma", [384, 1, 1]),
            blob("decoder.middle.1.to_qkv.weight", [1152, 384, 1, 1]),
            blob("decoder.middle.1.to_qkv.bias", [1152]),
            blob("decoder.middle.1.proj.weight", [384, 384, 1, 1]),
            blob("decoder.middle.1.proj.bias", [384]),
            blob("decoder.middle.2.residual.0.gamma", [384, 1, 1, 1]),
            blob("decoder.middle.2.residual.2.weight", [384, 384, 3, 3, 3]),
            blob("decoder.middle.2.residual.2.bias", [384]),
            blob("decoder.middle.2.residual.3.gamma", [384, 1, 1, 1]),
            blob("decoder.middle.2.residual.6.weight", [384, 384, 3, 3, 3]),
            blob("decoder.middle.2.residual.6.bias", [384]),
            blob("decoder.upsamples.3.resample.1.weight", [192, 384, 3, 3]),
            blob("decoder.upsamples.3.resample.1.bias", [192]),
            blob("decoder.upsamples.3.time_conv.weight", [768, 384, 3, 1, 1]),
            blob("decoder.upsamples.3.time_conv.bias", [768]),
            blob("decoder.upsamples.4.residual.0.gamma", [192, 1, 1, 1]),
            blob("decoder.upsamples.4.residual.2.weight", [384, 192, 3, 3, 3]),
            blob("decoder.upsamples.4.residual.2.bias", [384]),
            blob("decoder.upsamples.4.residual.3.gamma", [384, 1, 1, 1]),
            blob("decoder.upsamples.4.residual.6.weight", [384, 384, 3, 3, 3]),
            blob("decoder.upsamples.4.residual.6.bias", [384]),
            blob("decoder.upsamples.4.shortcut.weight", [384, 192, 1, 1, 1]),
            blob("decoder.upsamples.4.shortcut.bias", [384]),
            blob("decoder.head.0.gamma", [96, 1, 1, 1]),
            blob("decoder.head.2.weight", [3, 96, 3, 3, 3]),
            blob("decoder.head.2.bias", [3]),
        ].sorted { $0.name < $1.name }

        let url = try writePack(named: "vae-arch", specs: specs)
        // requiresCompleteArchitecture:false because this is a subset pack; the
        // complete-architecture check is covered against the real pack separately.
        let locator = try VAEDecoderLocator(
            file: AnimapkFile(url: url), requiresCompleteArchitecture: false)

        XCTAssertGreaterThanOrEqual(locator.groups.count, 5)
        let groupIndices = locator.groups.map(\.logicalIndex)
        XCTAssertEqual(groupIndices, Array(0..<locator.groups.count))

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
        let groupNames = Set(locator.groups.flatMap { $0.range.tensors.map(\.tensor.name) })
        XCTAssertFalse(groupNames.contains("decoder.upsamples.3.time_conv.weight"))

        // The attention group (group 3) carries exactly the middle attention set.
        let attention = locator.groups.first { $0.logicalIndex == 3 }
        XCTAssertNotNil(attention)
        XCTAssertEqual(
            Set(attention!.range.tensors.map(\.tensor.name)),
            Set(["decoder.middle.1.norm.gamma", "decoder.middle.1.to_qkv.weight",
                 "decoder.middle.1.to_qkv.bias", "decoder.middle.1.proj.weight",
                 "decoder.middle.1.proj.bias"]))
    }

    func testUnexpectedDecoderTensorFails() throws {
        let specs: [TestPackFactory.BlobSpec] = [
            blob("decoder.head.0.gamma", [96, 1, 1, 1]),
            blob("decoder.head.2.weight", [3, 96, 3, 3, 3]),
            blob("decoder.head.2.bias", [3]),
            blob("decoder.upsamples.0.surprise.weight", [1]),
        ]
        let url = try writePack(named: "vae-extra", specs: specs)
        XCTAssertThrowsError(try VAEDecoderLocator(
            file: AnimapkFile(url: url), requiresCompleteArchitecture: false)) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("surprise"))
        }
    }

    func testNonFP16StorageFails() throws {
        let specs: [TestPackFactory.BlobSpec] = [
            blob("decoder.head.0.gamma", [96, 1, 1, 1]),
            blob("decoder.head.2.weight", [3, 96, 3, 3, 3], storage: "w8", code: 1),
            blob("decoder.head.2.bias", [3]),
        ]
        let url = try writePack(named: "vae-w8", specs: specs)
        XCTAssertThrowsError(try VAEDecoderLocator(
            file: AnimapkFile(url: url), requiresCompleteArchitecture: false)) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("fp16"))
        }
    }

    func testWrongShapeFails() throws {
        let specs: [TestPackFactory.BlobSpec] = [
            blob("decoder.head.0.gamma", [96, 1, 1, 1]),
            blob("decoder.head.2.weight", [3, 96, 1, 1, 1]),  // wrong kernel size
            blob("decoder.head.2.bias", [3]),
        ]
        let url = try writePack(named: "vae-shape", specs: specs)
        XCTAssertThrowsError(try VAEDecoderLocator(
            file: AnimapkFile(url: url), requiresCompleteArchitecture: true)) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("decoder.head.2.weight"))
        }
    }

    func testMissingTensorFails() throws {
        // Complete-architecture mode requires ALL expected tensors, so a small
        // pack with just the head must fail with a missing-tensor error.
        let specs: [TestPackFactory.BlobSpec] = [
            blob("decoder.head.0.gamma", [96, 1, 1, 1]),
            blob("decoder.head.2.weight", [3, 96, 3, 3, 3]),
            blob("decoder.head.2.bias", [3]),
        ]
        let url = try writePack(named: "vae-missing", specs: specs)
        XCTAssertThrowsError(try VAEDecoderLocator(
            file: AnimapkFile(url: url), requiresCompleteArchitecture: true)) { error in
            guard case AnimapkError.validation(let message) = error else {
                return XCTFail("expected validation error, got \\(error)")
            }
            XCTAssertTrue(message.contains("missing"))
        }
    }
}
