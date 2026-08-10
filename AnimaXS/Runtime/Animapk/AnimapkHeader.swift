import Foundation

/// Fixed 256-byte little-endian ANMA v1 header.
/// Field layout verified against PHASE0_2_HANDOFF/ANIMAPK_SPEC.md and the real packs.
struct AnimapkHeader {
    static let size = 256
    static let magic = "ANMA"
    static let alignment16K: UInt32 = 16_384
    static let recordSize128: UInt32 = 128

    let version: UInt16
    let component: UInt16       // 1 = dit, 2 = te, 3 = vae
    let alignment: UInt32
    let tensorCount: UInt64
    let jsonOffset: UInt64
    let jsonSize: UInt64
    let tableOffset: UInt64
    let tableBytes: UInt64
    let payloadOffset: UInt64
    let fileSize: UInt64
    let recordSize: UInt32

    /// Parse from the mapped file bytes. All reads are byte-wise little-endian —
    /// no unsafe unaligned loads.
    static func parse(_ bytes: UnsafeRawBufferPointer) throws -> AnimapkHeader {
        guard bytes.count >= size else {
            throw AnimapkError.header("file smaller than 256-byte header (\(bytes.count) bytes)")
        }
        guard bytes[0] == 0x41, bytes[1] == 0x4E, bytes[2] == 0x4D, bytes[3] == 0x41 else {
            throw AnimapkError.header("bad magic (expected 'ANMA')")
        }
        func u16(_ o: Int) -> UInt16 {
            UInt16(bytes[o]) | (UInt16(bytes[o + 1]) << 8)
        }
        func u32(_ o: Int) -> UInt32 {
            UInt32(bytes[o]) | (UInt32(bytes[o + 1]) << 8) | (UInt32(bytes[o + 2]) << 16) | (UInt32(bytes[o + 3]) << 24)
        }
        func u64(_ o: Int) -> UInt64 {
            var v: UInt64 = 0
            for i in 0..<8 { v |= UInt64(bytes[o + i]) << (8 * i) }
            return v
        }

        let header = AnimapkHeader(
            version: u16(0x04),
            component: u16(0x06),
            alignment: u32(0x08),
            tensorCount: u64(0x0C),
            jsonOffset: u64(0x14),
            jsonSize: u64(0x1C),
            tableOffset: u64(0x24),
            tableBytes: u64(0x2C),
            payloadOffset: u64(0x34),
            fileSize: u64(0x3C),
            recordSize: u32(0x44)
        )

        guard header.version == 1 else {
            throw AnimapkError.header("unsupported ANMA version \(header.version) (expected 1)")
        }
        guard header.alignment == alignment16K else {
            throw AnimapkError.header("unexpected alignment \(header.alignment) (expected 16384)")
        }
        guard header.recordSize == recordSize128 else {
            throw AnimapkError.header("unexpected record size \(header.recordSize) (expected 128)")
        }
        return header
    }
}
