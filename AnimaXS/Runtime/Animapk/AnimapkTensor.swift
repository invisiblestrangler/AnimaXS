import Foundation

/// Storage dtype codes from ANIMAPK_SPEC.md §4.
enum StorageDtype: UInt8, Equatable {
    case w4 = 0      // uint4-packed
    case w8 = 1      // uint8
    case fp16 = 2
    case fp32 = 3

    init(code: UInt8) throws {
        guard let d = StorageDtype(rawValue: code) else {
            throw AnimapkError.validation("unknown storage dtype code \(code)")
        }
        self = d
    }
}

/// One tensor entry from the architecture JSON `tensor_meta` array.
/// JSON is the AUTHORITATIVE source for full names and shapes (binary table truncates
/// names to 64 chars and shapes to 4 dims).
struct AnimapkTensor: Codable, Equatable {
    let name: String
    let shape: [Int]
    let logicalDtype: String
    let storageDtype: String
    let crc32: UInt32?
    let blockIndex: Int?
    let executionIndex: Int?
    let blobOffset: UInt64
    let blobSize: UInt64
    /// Absent in real packs (data always starts at blob start); defaults to 0.
    let dataOffsetField: UInt64?
    let dataSize: UInt64
    let scaleOffset: UInt64?
    let scaleSize: UInt64?
    let zeroOffset: UInt64?
    let zeroSize: UInt64?

    enum CodingKeys: String, CodingKey {
        case name, shape
        case logicalDtype = "logical_dtype"
        case storageDtype = "storage_dtype"
        case crc32
        case blockIndex = "block_index"
        case executionIndex = "execution_index"
        case blobOffset = "blob_offset"
        case blobSize = "blob_size"
        case dataOffsetField = "data_offset"
        case dataSize = "data_size"
        case scaleOffset = "scale_offset"
        case scaleSize = "scale_size"
        case zeroOffset = "zero_offset"
        case zeroSize = "zero_size"
    }

    /// Byte offset of the packed data region relative to the file start.
    var dataOffset: UInt64 { dataOffsetField ?? 0 }

    var elementCount: Int { shape.reduce(1, *) }

    var storage: StorageDtype {
        (try? StorageDtype(code: storageDtypeCode)) ?? .fp32
    }

    /// Raw code as stored (0..3). Falls back to fp32 on unknown values.
    private var storageDtypeCode: UInt8 {
        switch storageDtype {
        case "w4": return 0
        case "w8": return 1
        case "fp16": return 2
        case "fp32": return 3
        default: return 3
        }
    }
}

/// Top-level architecture JSON envelope.
struct AnimapkEnvelope: Codable {
    /// "dit" / "te" / "vae" (string in real packs).
    let component: String
    let quant: AnimapkQuant?
    let tensorMeta: [AnimapkTensor]
    let sourceHashes: [String: String]?

    enum CodingKeys: String, CodingKey {
        case component, quant
        case tensorMeta = "tensor_meta"
        case sourceHashes = "source_hashes"
    }
}

struct AnimapkQuant: Codable {
    let scheme: String?
    let group: Int?
}
