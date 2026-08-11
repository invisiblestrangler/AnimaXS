import CryptoKit
import Foundation

enum ModelComponent: String, Codable {
    case dit
    case textEncoder
    case vae
}

struct ModelManifestEntry: Codable, Equatable {
    let filename: String
    let size: UInt64
    let sha256: String
    let url: URL
    let component: ModelComponent
}

enum ModelManifest {
    static let releaseTag = "model-assets-v1"
    private static let releaseBase = URL(
        string: "https://github.com/invisiblestrangler/AnimaXS/releases/download/\(releaseTag)/")!

    static let entries: [ModelManifestEntry] = [
        entry(
            "anima-turbo-v1.0-xsmax-w4.animapk", size: 1_179_435_008,
            sha256: "ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25",
            component: .dit),
        entry(
            "qwen3-0.6b-xsmax-w8.animapk", size: 635_305_984,
            sha256: "ba59e4d1797de5f6512aeafcecf3f38e1f62a47313a2a400b949c9018d84ceab",
            component: .textEncoder),
        entry(
            "qwen-image-vae-xsmax-fp16.animapk", size: 256_163_840,
            sha256: "10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447",
            component: .vae),
    ]

    static func sha256(of url: URL, chunkBytes: Int = 1 << 20) throws -> String {
        guard chunkBytes > 0 else {
            throw AnimapkError.validation("SHA-256 chunk size must be positive")
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while true {
            let data = try handle.read(upToCount: chunkBytes) ?? Data()
            if data.isEmpty { break }
            digest.update(data: data)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func verify(_ url: URL, against entry: ModelManifestEntry) throws {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attributes[.size] as? NSNumber,
              number.uint64Value == entry.size else {
            throw AnimapkError.validation("model size does not match manifest")
        }
        guard try sha256(of: url) == entry.sha256.lowercased() else {
            throw AnimapkError.validation("model SHA-256 does not match manifest")
        }
    }

    private static func entry(
        _ filename: String, size: UInt64, sha256: String, component: ModelComponent
    ) -> ModelManifestEntry {
        ModelManifestEntry(
            filename: filename, size: size, sha256: sha256,
            url: releaseBase.appendingPathComponent(filename), component: component)
    }
}
