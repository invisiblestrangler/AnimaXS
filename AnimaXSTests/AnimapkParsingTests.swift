import XCTest
@testable import AnimaXS

/// Parser tests using small synthetic ANMA v1 packs (no multi-GB production assets needed).
final class AnimapkParsingTests: XCTestCase {

    var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("animapk-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    private func makeW4Pack() throws -> URL {
        // Replicate the real pack's known vector shape [8192, 2048], group 64.
        // Data: first byte 0x6E = 110 (low nibble 14 = 0x0E, high nibble 6).
        // Build a full data payload with the documented first-8 bytes then zeros.
        let k = 2048
        let first8: [UInt8] = [110, 135, 69, 55, 138, 137, 37, 98]
        var data = Data(first8)
        data.append(Data(repeating: 0, count: k / 2 - first8.count))
        // scale: ceil(2048/64)=32 groups, fp16; documented first four: 0.0004537, 0.0004232, 0.0004251, 0.0004313
        var scale = Data()
        let scaleVals: [Float16] = [0.0004537, 0.0004232, 0.0004251, 0.0004313, 0, 0, 0, 0]
        for _ in 0..<32 {
            let v = scaleVals[scale.count / 2 < scaleVals.count ? scale.count / 2 : 0]
            _ = v // placeholder; write explicit below
        }
        var scaleData = Data()
        let refScales: [Float] = [0.0004537, 0.0004232, 0.0004251, 0.0004313]
        for i in 0..<32 {
            let f = i < 4 ? refScales[i] : 0.0
            var h = Float16(f)
            var bits = h.bitPattern.littleEndian
            scaleData.append(contentsOf: withUnsafeBytes(of: &bits) { [UInt8]($0) })
        }
        let zero = Data(repeating: 0, count: 32 * 2)

        let blob = TestPackFactory.BlobSpec(
            name: "model.diffusion_model.blocks.0.mlp.layer1.weight",
            shape: [8192, 2048],
            logicalDtype: "w4", storageDtype: "w4", storageCode: 0,
            data: data, scale: scaleData, zero: zero,
            crc32: Crc32.compute(data)
        )
        return try TestPackFactory.writePack(
            named: "dit", componentCode: 1, quantScheme: "w4", quantGroup: 64,
            blobs: [blob], to: tmpDir)
    }

    func testHeaderParseAndValidation() throws {
        let url = try makeW4Pack()
        let file = try AnimapkFile(url: url)
        XCTAssertEqual(file.header.version, 1)
        XCTAssertEqual(file.header.component, 1)
        XCTAssertEqual(file.header.alignment, 16_384)
        XCTAssertEqual(file.header.tensorCount, 1)
        XCTAssertEqual(file.header.recordSize, 128)
        XCTAssertEqual(file.quantScheme, "w4")
        XCTAssertEqual(file.quantGroup, 64)
    }

    func testJSONIsAuthoritativeForNameAndShape() throws {
        let url = try makeW4Pack()
        let file = try AnimapkFile(url: url)
        let t = try XCTUnwrap(file.tensor(named: "model.diffusion_model.blocks.0.mlp.layer1.weight"))
        XCTAssertEqual(t.shape, [8192, 2048])
        XCTAssertEqual(t.storageDtype, "w4")
        XCTAssertEqual(t.dataSize, 1024)
        XCTAssertEqual(t.blobOffset % 16_384, 0)
    }

    func testCRC32Validation() throws {
        let url = try makeW4Pack()
        let file = try AnimapkFile(url: url)
        XCTAssertEqual(file.crc32MismatchCount(), 0)
    }

    func testBadMagicRejected() throws {
        let url = try makeW4Pack()
        var data = try Data(contentsOf: url)
        data[0] = 0x58
        try data.write(to: url)
        XCTAssertThrowsError(try AnimapkFile(url: url)) { err in
            guard case AnimapkError.header = err else {
                return XCTFail("expected header error, got \(err)")
            }
        }
    }

    func testAlignmentEnforced() throws {
        let k = 64
        let data = Data(repeating: 0, count: k / 2)
        let scale = Data(repeating: 0, count: (k / 64) * 2)
        let zero = Data(repeating: 0, count: (k / 64) * 2)
        let blob = TestPackFactory.BlobSpec(
            name: "x", shape: [64], logicalDtype: "w4", storageDtype: "w4", storageCode: 0,
            data: data, scale: scale, zero: zero)
        let url = try TestPackFactory.writePack(
            named: "t", componentCode: 1, quantScheme: "w4", quantGroup: 64, blobs: [blob], to: tmpDir)
        let file = try AnimapkFile(url: url)
        XCTAssertEqual(file.tensors[0].blobOffset % 16_384, 0)
        XCTAssertEqual(file.crc32MismatchCount(), 0)
    }

    func testFileSizeMismatchRejected() throws {
        let url = try makeW4Pack()
        var data = try Data(contentsOf: url)
        data.append(Data(repeating: 0, count: 100))
        try data.write(to: url)
        XCTAssertThrowsError(try AnimapkFile(url: url))
    }

    func testOptionalANEQuantizationMetadataDecodesWithoutBreakingLegacy() throws {
        let data = Data([0, 127, 255, 64])
        var scale = Data()
        var bias = Data()
        for value: Float in [0.01, 0.02] {
            var bits = value.bitPattern.littleEndian
            scale.append(contentsOf: withUnsafeBytes(of: &bits) { [UInt8]($0) })
        }
        for value: Float in [-1.0, 2.0] {
            var bits = value.bitPattern.littleEndian
            bias.append(contentsOf: withUnsafeBytes(of: &bits) { [UInt8]($0) })
        }
        let digest = String(repeating: "a", count: 64)
        let blob = TestPackFactory.BlobSpec(
            name: "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            shape: [2, 2], logicalDtype: "w8", storageDtype: "w8", storageCode: 1,
            data: data, scale: scale, zero: bias,
            quantizationFormat: "ane_u8_per_row_fp32_v1", blobSHA256: digest)
        let url = try TestPackFactory.writePack(
            named: "ane-meta", componentCode: 1, quantScheme: "w8-ane-hybrid-v1",
            quantGroup: 64, blobs: [blob], to: tmpDir)
        let file = try AnimapkFile(url: url)
        let tensor = try XCTUnwrap(file.tensors.first)
        XCTAssertEqual(tensor.quantizationFormat, "ane_u8_per_row_fp32_v1")
        XCTAssertEqual(tensor.blobSHA256, digest)

        let legacyURL = try makeW4Pack()
        let legacy = try AnimapkFile(url: legacyURL)
        XCTAssertNil(legacy.tensors.first?.quantizationFormat)
        XCTAssertNil(legacy.tensors.first?.blobSHA256)
    }
}

final class ANEW8NativePackTests: XCTestCase {
    private func tensor(
        suffix: String, offset: UInt64, native: Bool,
        rows: Int = 2, columns: Int = 2
    ) -> AnimapkTensor {
        let format = native ? ANEW8NativePack.tensorFormat : "group64_affine_fp16_v2"
        let dataSize = UInt64(rows * columns)
        let paramSize = native ? UInt64(rows * 4) : UInt64(rows * 2)
        return AnimapkTensor(
            name: "model.diffusion_model.blocks.0.\(suffix)",
            shape: [rows, columns], logicalDtype: "w8", storageDtype: "w8",
            crc32: nil, blockIndex: 0, executionIndex: nil,
            blobOffset: offset, blobSize: 16_384,
            dataOffsetField: 0, dataSize: dataSize,
            scaleOffset: dataSize, scaleSize: paramSize,
            zeroOffset: dataSize + paramSize, zeroSize: paramSize,
            quantizationFormat: format,
            blobSHA256: native ? String(repeating: "a", count: 64) : nil)
    }

    func testNativeTensorMetadataValidation() throws {
        let t = tensor(
            suffix: "self_attn.q_proj.weight", offset: 16_384,
            native: true, rows: 2, columns: 2)
        XCTAssertNoThrow(try ANEW8NativePack.validateNativeTensor(
            t, expectedRows: 2, expectedColumns: 2))

        let bad = AnimapkTensor(
            name: t.name, shape: t.shape, logicalDtype: t.logicalDtype,
            storageDtype: t.storageDtype, crc32: nil, blockIndex: 0,
            executionIndex: nil, blobOffset: t.blobOffset, blobSize: t.blobSize,
            dataOffsetField: 0, dataSize: t.dataSize,
            scaleOffset: t.scaleOffset, scaleSize: 2,
            zeroOffset: t.zeroOffset, zeroSize: t.zeroSize,
            quantizationFormat: t.quantizationFormat, blobSHA256: t.blobSHA256)
        XCTAssertThrowsError(try ANEW8NativePack.validateNativeTensor(
            bad, expectedRows: 2, expectedColumns: 2))
    }

    func testCompactMetalRangeExcludesNativeProjectionBlobs() throws {
        let metalSuffixes = [
            "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight",
            "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight",
            "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
        ]
        let nativeSuffixes = ANEW8NativePack.projectionSpecs.map(\.suffix)
        var tensors: [AnimapkTensor] = []
        for (index, suffix) in metalSuffixes.enumerated() {
            tensors.append(tensor(
                suffix: suffix, offset: UInt64(index + 1) * 16_384, native: false))
        }
        for (index, suffix) in nativeSuffixes.enumerated() {
            tensors.append(tensor(
                suffix: suffix, offset: UInt64(index + 11) * 16_384, native: true))
        }
        let range = try DiTANEHybridMetalLocator.metalRange(tensors: tensors, logicalIndex: 0)
        XCTAssertEqual(range.tensors.count, 10)
        XCTAssertTrue(range.tensors.allSatisfy {
            $0.tensor.quantizationFormat != ANEW8NativePack.tensorFormat
        })
        let fullEnd = tensors.map { $0.blobOffset + $0.blobSize }.max()!
        XCTAssertLessThan(range.fileRange.upperBound, fullEnd)
    }

    func testCompactMetalRangeRejectsInterleavedNativeBlob() throws {
        var tensors: [AnimapkTensor] = []
        let metalSuffixes = [
            "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight",
            "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight",
            "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
        ]
        for (index, suffix) in metalSuffixes.enumerated() {
            tensors.append(tensor(
                suffix: suffix, offset: UInt64(index + 1) * 16_384, native: false))
        }
        for (index, spec) in ANEW8NativePack.projectionSpecs.enumerated() {
            let offset: UInt64 = index == 0 ? 5 * 16_384 : UInt64(index + 11) * 16_384
            tensors.append(tensor(suffix: spec.suffix, offset: offset, native: true))
        }
        XCTAssertThrowsError(
            try DiTANEHybridMetalLocator.metalRange(tensors: tensors, logicalIndex: 0))
    }
}

/// Native Anima LoRA parser tests use real production dimensions but tiny rank,
/// keeping each synthetic safetensors fixture only a few kilobytes.
final class DiTLoRAFileTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-parser-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testNativeSelfQKeysRankAndAlphaScaling() throws {
        let url = try makeLoRA(target: .selfQ, rank: 2, alpha: 4)
        let file = try DiTLoRAFile(url: url)
        let descriptor = try XCTUnwrap(file.module(block: 0, target: .selfQ))
        XCTAssertEqual(file.modules.count, 1)
        XCTAssertEqual(descriptor.rank, 2)
        XCTAssertEqual(descriptor.alpha, 4, accuracy: 0.0001)
        XCTAssertEqual(descriptor.scale, 2, accuracy: 0.0001)
        XCTAssertEqual(descriptor.down.shape, [2, 2_048])
        XCTAssertEqual(descriptor.up.shape, [2_048, 2])
    }

    func testCrossKVAndMLPShapesMapToTheirRealFeatureCounts() throws {
        let cross = try DiTLoRAFile(url: makeLoRA(target: .crossK, rank: 1, alpha: 1))
        let crossDescriptor = try XCTUnwrap(cross.module(block: 0, target: .crossK))
        XCTAssertEqual(crossDescriptor.down.shape, [1, 1_024])
        XCTAssertEqual(crossDescriptor.up.shape, [2_048, 1])

        let mlp = try DiTLoRAFile(url: makeLoRA(target: .mlpDown, rank: 1, alpha: 1))
        let mlpDescriptor = try XCTUnwrap(mlp.module(block: 0, target: .mlpDown))
        XCTAssertEqual(mlpDescriptor.down.shape, [1, 8_192])
        XCTAssertEqual(mlpDescriptor.up.shape, [2_048, 1])
    }

    func testMissingAlphaIsRejectedInsteadOfGuessingScaling() throws {
        let url = try makeLoRA(target: .selfQ, rank: 1, alpha: 1, includeAlpha: false)
        XCTAssertThrowsError(try DiTLoRAFile(url: url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("down/up/alpha"))
        }
    }

    func testWrongProjectionShapeIsRejected() throws {
        let url = try makeLoRA(
            target: .selfQ, rank: 1, alpha: 1,
            overrideInputFeatures: 2_047)
        XCTAssertThrowsError(try DiTLoRAFile(url: url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("shape mismatch"))
        }
    }

    func testUnsupportedAnimaModuleIsRejectedRatherThanPartiallyApplied() throws {
        let name = "lora_unet_blocks_0_adaln_modulation_self_attn_1.lora_down.weight"
        let url = try writeSafetensors(entries: [
            name: (dtype: "F16", shape: [1, 1], bytes: halfData([1]))
        ])
        XCTAssertThrowsError(try DiTLoRAFile(url: url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("unsupported Anima LoRA module"))
        }
    }

    private func makeLoRA(
        target: DiTLoRATarget,
        rank: Int,
        alpha: Float,
        includeAlpha: Bool = true,
        overrideInputFeatures: Int? = nil
    ) throws -> URL {
        let stem = "lora_unet_blocks_0_\(target.rawValue)"
        let input = overrideInputFeatures ?? target.inputFeatures
        let output = target.outputFeatures
        var entries: [String: (dtype: String, shape: [Int], bytes: Data)] = [
            "\(stem).lora_down.weight": (
                "F16", [rank, input],
                halfData((0..<(rank * input)).map { Float(($0 % 7) - 3) / 16 })),
            "\(stem).lora_up.weight": (
                "F16", [output, rank],
                halfData((0..<(output * rank)).map { Float(($0 % 5) - 2) / 16 })),
        ]
        if includeAlpha {
            var bits = alpha.bitPattern.littleEndian
            entries["\(stem).alpha"] = (
                "F32", [], Data(bytes: &bits, count: MemoryLayout<UInt32>.size))
        }
        return try writeSafetensors(entries: entries)
    }

    private func halfData(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * 2)
        for value in values {
            var bits = Float16(value).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func writeSafetensors(
        entries: [String: (dtype: String, shape: [Int], bytes: Data)]
    ) throws -> URL {
        var header: [String: Any] = [:]
        var payload = Data()
        for name in entries.keys.sorted() {
            let entry = entries[name]!
            let start = payload.count
            payload.append(entry.bytes)
            header[name] = [
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [start, payload.count],
            ]
        }
        header["__metadata__"] = ["ss_network_module": "networks.lora_anima"]
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var headerLength = UInt64(headerData.count).littleEndian
        var file = Data(bytes: &headerLength, count: MemoryLayout<UInt64>.size)
        file.append(headerData)
        file.append(payload)
        let url = directory.appendingPathComponent("\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }
}
