import Foundation
@testable import AnimaXS

/// Builds small synthetic ANMA v1 packs in memory for parser/decoder unit tests,
/// so the test target never needs the multi-GB production packs (which CI must not download).
enum TestPackFactory {

    /// Write a minimal ANMA v1 pack with a fixed binary header, one JSON tensor_meta,
    /// one 128-byte table record, and 16 KB-aligned blob(s). Returns the file URL.
    /// `blobs`: [(name, shape, storageCode, data, scale, zero)] — data/scale/zero already
    /// packed little-endian raw bytes by the caller.
    static func writePack(named name: String,
                          componentCode: UInt16,
                          quantScheme: String,
                          quantGroup: Int,
                          blobs: [BlobSpec],
                          to dir: URL) throws -> URL {
        let fileURL = dir.appendingPathComponent("\(name).animapk")


        // Layout plan: header(256) + JSON(js) + pad + table(tb) + pad-to-16K + blobs (16K aligned).
    // We first reserve the JSON/table sizes, compute the payload start, then assign each
    // blob the next 16 KB-aligned absolute offset.
    let jsonOffset = 256
    // JSON built first with placeholder blob offsets, then rewritten after layout.
    func leData<T: FixedWidthInteger>(_ v: T) -> Data {
        var le = v.littleEndian
        return withUnsafeBytes(of: &le) { Data($0) }
    }

    // 1) Build the JSON body (authoritative) with final blob offsets filled in later.
    var jsonBody: Data
    // helper to serialize tensorMeta given blob offsets
    func makeJSON(offsets: [UInt64]) throws -> Data {
        var tensorMeta: [[String: Any]] = []
        for (i, spec) in blobs.enumerated() {
            var entry: [String: Any] = [
                "name": spec.name,
                "shape": spec.shape,
                "logical_dtype": spec.logicalDtype,
                "storage_dtype": spec.storageDtype,
                "blob_offset": offsets[i],
                "blob_size": UInt64(spec.data.count + spec.scale.count + spec.zero.count),
                "data_offset": 0,
                "data_size": UInt64(spec.data.count),
                "scale_offset": UInt64(spec.data.count),
                "scale_size": UInt64(spec.scale.count),
                "zero_offset": UInt64(spec.data.count + spec.scale.count),
                "zero_size": UInt64(spec.zero.count),
            ]
            if spec.crc32 != nil { entry["crc32"] = spec.crc32! }
            tensorMeta.append(entry)
        }
        let json: [String: Any] = [
            "component": componentCode == 1 ? "dit" : (componentCode == 2 ? "te" : "vae"),
            "quant": ["scheme": quantScheme, "group": quantGroup],
            "tensor_meta": tensorMeta,
        ]
        return try JSONSerialization.data(withJSONObject: json)
    }

    // 2) Determine table size (depends only on blob count).
    let recordSize = 128
    let tableBytes = blobs.count * recordSize

    // 3) Fixed region layout (mirrors real packs): JSON at 256, table at the next 16 KB
    //    boundary, payload after that aligned. JSON content size does not affect offsets
    //    (padding absorbs any variance), so no size-stability trick is needed.

    let tableOffset = 16_384
    let payloadStart = (tableOffset + tableBytes + 16_383) / 16_384 * 16_384


    // 4) Assign blob absolute offsets: each at the next 16 KB boundary from payloadStart.
    var blobOffsets: [UInt64] = []
    var cursor = payloadStart
    for spec in blobs {
        let blobLen = spec.data.count + spec.scale.count + spec.zero.count
        blobOffsets.append(UInt64(cursor))
        cursor += blobLen
    }

    // 5) Final JSON (offsets now known).
    let jsonData = try makeJSON(offsets: blobOffsets)
    let jsonSize = jsonData.count
    // Ensure the JSON fits between header(256) and tableOffset(16384).
    precondition(jsonSize <= tableOffset - jsonOffset, "JSON too large for fixed 16 KB region in test factory")

    // 6) Build table records using absolute blob offsets.
    var table = Data()
    for (i, spec) in blobs.enumerated() {
        var rec = Data(repeating: 0, count: recordSize)
        let nameBytes = Data(spec.name.utf8)
        rec.replaceSubrange(0..<min(64, nameBytes.count), with: nameBytes.prefix(64))
        rec.replaceSubrange(64..<68, with: leData(UInt32(spec.shape.count)))
        for d in 0..<4 {
            let v = d < spec.shape.count ? UInt32(spec.shape[d]) : 0
            rec.replaceSubrange((68 + 4 * d)..<(72 + 4 * d), with: leData(v))
        }
        rec[84] = spec.storageCode
        rec[85] = spec.storageCode
        rec.replaceSubrange(88..<96, with: leData(UInt64(spec.shape.reduce(1, *))))
        rec.replaceSubrange(96..<104, with: leData(blobOffsets[i]))
        rec.replaceSubrange(104..<112, with: leData(UInt64(spec.data.count + spec.scale.count + spec.zero.count)))
        rec.replaceSubrange(112..<116, with: leData(UInt32(0)))
        rec.replaceSubrange(116..<120, with: leData(UInt32(spec.data.count)))
        rec.replaceSubrange(120..<124, with: leData(UInt32(spec.data.count)))
        rec.replaceSubrange(124..<128, with: leData(UInt32(spec.data.count + spec.scale.count)))
        table.append(rec)
    }

    // 7) Assemble file: header + json + pad + table + pad + blobs.
    var file = Data()
    // header first (placeholder fields; patched at end)
    var header = Data(repeating: 0, count: 256)
    header[0...3] = Data("ANMA".utf8)
    func putU16(_ v: UInt16, at o: Int) { header.replaceSubrange(o..<(o + 2), with: leData(v)) }
    func putU32(_ v: UInt32, at o: Int) { header.replaceSubrange(o..<(o + 4), with: leData(v)) }
    func putU64(_ v: UInt64, at o: Int) { header.replaceSubrange(o..<(o + 8), with: leData(v)) }
    putU16(1, at: 4)
    putU16(componentCode, at: 6)
    putU32(16_384, at: 8)
    putU64(UInt64(blobs.count), at: 0x0C)
    putU64(UInt64(jsonOffset), at: 0x14)
    putU64(UInt64(jsonSize), at: 0x1C)
    putU64(UInt64(tableOffset), at: 0x24)
    putU64(UInt64(tableBytes), at: 0x2C)
    putU64(UInt64(payloadStart), at: 0x34)
    putU32(UInt32(recordSize), at: 0x44)
    file.append(header)

    file.append(jsonData)
    file.append(Data(repeating: 0, count: tableOffset - (jsonOffset + jsonSize)))
    file.append(table)
    file.append(Data(repeating: 0, count: payloadStart - (tableOffset + tableBytes)))
    for (i, spec) in blobs.enumerated() {
        file.append(spec.data)
        file.append(spec.scale)
        file.append(spec.zero)
        if i < blobs.count - 1 {
            let next = blobOffsets[i + 1]
            let cur = blobOffsets[i] + UInt64(spec.data.count + spec.scale.count + spec.zero.count)
            file.append(Data(repeating: 0, count: Int(next - cur)))
        }
    }
    let fileSize = file.count
    // file_size lives at 0x3C in the already-appended header; patch in place.
    file.replaceSubrange(0x3C..<0x44, with: leData(UInt64(fileSize)))
    try file.write(to: fileURL)
    return fileURL
}

    struct BlobSpec {
        let name: String
        let shape: [Int]
        let logicalDtype: String
        let storageDtype: String
        let storageCode: UInt8
        let data: Data
        let scale: Data
        let zero: Data
        let crc32: UInt32?

        init(name: String, shape: [Int], logicalDtype: String, storageDtype: String,
             storageCode: UInt8, data: Data, scale: Data = Data(), zero: Data = Data(),
             crc32: UInt32? = nil) {
            self.name = name
            self.shape = shape
            self.logicalDtype = logicalDtype
            self.storageDtype = storageDtype
            self.storageCode = storageCode
            self.data = data
            self.scale = scale
            self.zero = zero
            self.crc32 = crc32
        }
    }
}
