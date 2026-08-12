import Foundation

/// Full ANMA v1 pack reader.
///
/// Validation performed at open time (runbook §17):
///   - magic == "ANMA", version == 1, alignment == 16384, record size == 128
///   - declared file size == actual size
///   - JSON/table/payload sections inside file
///   - every blob range inside file
///   - every blobOffset % 16384 == 0
///   - binary table blob_offsets cross-match JSON tensor_meta (by blobOffset)
///
/// JSON `tensor_meta` is the authoritative name/shape source. The binary table is
/// parsed only for cross-validation.
final class AnimapkFile {
    let url: URL
    let header: AnimapkHeader
    /// Tensors in JSON order (alphabetical by full name).
    let tensors: [AnimapkTensor]
    let byName: [String: AnimapkTensor]
    let byBlobOffset: [UInt64: AnimapkTensor]
    let quantScheme: String?
    let quantGroup: Int?
    /// "dit" / "te" / "vae" component from the architecture JSON.
    let component: String

    private let map: MappedFile

    init(url: URL) throws {
        self.url = url
        let map = try MappedFile(url: url)
        self.map = map
        let bytes = map.bytes()
        self.header = try AnimapkHeader.parse(bytes)
        try Self.validateHeader(header, fileBytes: bytes.count)

        // --- Architecture JSON (authoritative) ---
        let jsonStart = Int(header.jsonOffset)
        let jsonLen = Int(header.jsonSize)
        let jsonData = Data(bytes: map.pointer(offset: jsonStart), count: jsonLen)
        let envelope: AnimapkEnvelope
        do {
            envelope = try JSONDecoder().decode(AnimapkEnvelope.self, from: jsonData)
        } catch {
            throw AnimapkError.json("failed to decode architecture JSON: \(error)")
        }
        self.tensors = envelope.tensorMeta
        self.component = envelope.component
        self.quantScheme = envelope.quant?.scheme
        self.quantGroup = envelope.quant?.group

        var byName: [String: AnimapkTensor] = [:]
        var byBlob: [UInt64: AnimapkTensor] = [:]
        for t in tensors {
            byName[t.name] = t
            byBlob[t.blobOffset] = t
        }
        self.byName = byName
        self.byBlobOffset = byBlob

        guard tensors.count == Int(header.tensorCount) else {
            throw AnimapkError.json("JSON tensor_meta count \(tensors.count) != header tensor_count \(header.tensorCount)")
        }
        for t in tensors {
            try Self.validateTensor(t, fileBytes: bytes.count)
        }
        try Self.validateTableCrossMatch(header: header, bytes: bytes, byBlobOffset: byBlob)
    }

    // MARK: - Validation

    private static func validateHeader(_ h: AnimapkHeader, fileBytes: Int) throws {
        guard h.fileSize == UInt64(fileBytes) else {
            throw AnimapkError.validation("declared file size \(h.fileSize) != actual \(fileBytes)")
        }
        let end = { (off: UInt64, len: UInt64) in off &+ len }
        guard end(h.jsonOffset, h.jsonSize) <= h.fileSize else {
            throw AnimapkError.validation("JSON section out of file bounds")
        }
        guard end(h.tableOffset, h.tableBytes) <= h.fileSize else {
            throw AnimapkError.validation("table section out of file bounds")
        }
        guard h.payloadOffset <= h.fileSize else {
            throw AnimapkError.validation("payload offset out of file bounds")
        }
        // NOTE: real packs have json_offset=256 and table_offset=327936 — NOT 16 KB aligned
        // (only payload_offset and every blob_offset are). ANIMAPK_SPEC.md's "sections 16 KB
        // aligned" claim is inaccurate; blob alignment is the enforced invariant (see DECISIONS.md).
    }

    private static func validateTensor(_ t: AnimapkTensor, fileBytes: Int) throws {
        guard t.blobOffset + t.blobSize <= UInt64(fileBytes) else {
            throw AnimapkError.validation("blob \(t.name) out of file bounds")
        }
        guard t.blobOffset % 16_384 == 0 else {
            throw AnimapkError.validation("blob \(t.name) offset \(t.blobOffset) not 16 KB aligned")
        }
        guard t.dataOffset + t.dataSize <= t.blobSize else {
            throw AnimapkError.validation("blob \(t.name) data region out of blob bounds")
        }
        if let so = t.scaleOffset, let ss = t.scaleSize, so + ss > t.blobSize {
            throw AnimapkError.validation("blob \(t.name) scale region out of blob bounds")
        }
        if let zo = t.zeroOffset, let zs = t.zeroSize, zo + zs > t.blobSize {
            throw AnimapkError.validation("blob \(t.name) zero region out of blob bounds")
        }
    }

    /// Cross-match the binary table's blob_offsets against the JSON index.
    private static func validateTableCrossMatch(header: AnimapkHeader, bytes: UnsafeRawBufferPointer, byBlobOffset: [UInt64: AnimapkTensor]) throws {
        guard header.tableBytes % UInt64(header.recordSize) == 0 else {
            throw AnimapkError.validation("table bytes not a multiple of record size")
        }
        let recordCount = Int(header.tableBytes) / Int(header.recordSize)
        guard recordCount == Int(header.tensorCount) else {
            throw AnimapkError.validation("table record count \(recordCount) != tensor_count \(header.tensorCount)")
        }
        let base = bytes.baseAddress!.advanced(by: Int(header.tableOffset))
        let recSize = Int(header.recordSize)
        var tableBlobOffsets = Set<UInt64>()
        for r in 0..<recordCount {
            // blob_offset is the UInt64 at record offset 96.
            var v: UInt64 = 0
            let p = base.advanced(by: r * recSize + 96)
            let b = p.assumingMemoryBound(to: UInt8.self)
            for i in 0..<8 { v |= UInt64(b[i]) << (8 * i) }
            tableBlobOffsets.insert(v)
        }
        guard tableBlobOffsets.count == byBlobOffset.count else {
            throw AnimapkError.validation("table blob_offsets (\(tableBlobOffsets.count)) != JSON blob_offsets (\(byBlobOffset.count))")
        }
        for off in tableBlobOffsets where byBlobOffset[off] == nil {
            throw AnimapkError.validation("table blob_offset \(off) missing from JSON tensor_meta")
        }
    }

    // MARK: - Accessors

    func tensor(named name: String) -> AnimapkTensor? { byName[name] }

    /// Zero-copy view of a validated absolute file range. The returned pointer is
    /// valid only while this `AnimapkFile` remains alive.
    func bytes(in range: Range<UInt64>) throws -> UnsafeRawBufferPointer {
        let length = range.upperBound - range.lowerBound
        guard range.lowerBound <= range.upperBound,
              range.upperBound <= header.fileSize,
              range.lowerBound <= UInt64(Int.max),
              length <= UInt64(Int.max) else {
            throw AnimapkError.validation("mmap byte range \(range) is out of bounds")
        }
        return UnsafeRawBufferPointer(
            start: map.pointer(offset: Int(range.lowerBound)),
            count: Int(length)
        )
    }

    /// Raw packed data bytes for a tensor.
    func dataBytes(_ t: AnimapkTensor) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: map.pointer(offset: Int(t.blobOffset + t.dataOffset)), count: Int(t.dataSize))
    }

    func scaleBytes(_ t: AnimapkTensor) -> UnsafeRawBufferPointer? {
        guard let so = t.scaleOffset, let ss = t.scaleSize else { return nil }
        return UnsafeRawBufferPointer(start: map.pointer(offset: Int(t.blobOffset + so)), count: Int(ss))
    }

    func zeroBytes(_ t: AnimapkTensor) -> UnsafeRawBufferPointer? {
        guard let zo = t.zeroOffset, let zs = t.zeroSize else { return nil }
        return UnsafeRawBufferPointer(start: map.pointer(offset: Int(t.blobOffset + zo)), count: Int(zs))
    }

    /// Recompute CRC-32 of the tensor's packed data region and compare with the
    /// JSON metadata value. Returns true when matching or when no reference CRC exists.
    func verifyCRC32(_ t: AnimapkTensor) -> Bool {
        guard let expected = t.crc32 else { return true }
        return Crc32.compute(dataBytes(t)) == expected
    }

    /// Count of tensors whose CRC-32 fails (0 = perfect).
    func crc32MismatchCount() -> Int {
        var bad = 0
        for t in tensors where !verifyCRC32(t) { bad += 1 }
        return bad
    }
}

extension UnsafeRawBufferPointer {
    /// Copy the buffer into a `Data` (used to hand mmap slices to the Data-based decoders).
    var data: Data {
        Data(bytes: baseAddress!, count: count)
    }
}
