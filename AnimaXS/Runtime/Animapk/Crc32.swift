import Foundation

/// Pure-Swift CRC-32 (zlib/ISO-HDLC, polynomial 0xEDB88320, init 0xFFFFFFFF, final XOR).
/// Matches the zlib crc32() used by pack_anima.py — no external dependency, portable
/// across iOS and Linux.
enum Crc32 {
    private static let table: [UInt32] = {
        var t = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1)
            }
            t[i] = c
        }
        return t
    }()

    static func compute(_ bytes: UnsafeRawBufferPointer) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for b in bytes {
            crc = table[Int((crc ^ UInt32(b)) & 0xFF)] ^ (crc >> 8)
        }
        return crc ^ 0xFFFF_FFFF
    }

    static func compute(_ data: Data) -> UInt32 {
        data.withUnsafeBytes { compute($0) }
    }
}
