import CryptoKit
import Foundation

enum ModelComponent: String, Codable {
    case dit
    case textEncoder
    case vae
}

/// One accepted (size, SHA-256) variant for a model pack.
///
/// The DiT slot is the only entry with alternates: the user imports either the
/// W4 pack (default) or the W8-v2 pack into the same `.dit` slot, and whichever
/// file was imported is the one a generation uses. Text encoder and VAE have a
/// single pinned variant.
struct ModelVariant: Codable, Equatable {
    let size: UInt64
    let sha256: String
}

struct ModelManifestEntry: Codable, Equatable {
    let filename: String
    let size: UInt64
    let sha256: String
    let url: URL
    let component: ModelComponent
    /// Additional accepted (size, SHA-256) variants for this slot (empty for
    /// the single-variant text encoder and VAE). Verification accepts the
    /// primary or any alternate.
    var alternates: [ModelVariant] = []

    /// All accepted variants, primary first.
    var allVariants: [ModelVariant] {
        [ModelVariant(size: size, sha256: sha256)] + alternates
    }
}

/// Runtime descriptor for one accepted model pack variant.
///
/// Unlike `ModelVariant` (which is the persisted receipt shape), this carries
/// the human-visible variant id ("w4" or "w8-v2") and the display filename so
/// telemetry can report which pack actually ran even though the app-owned
/// local file is always named after the W4 slot.
struct ModelVariantDescriptor: Hashable {
    let id: String
    let displayFilename: String
    let size: UInt64
    let sha256: String
}

/// The numerical-fidelity policy for a DiT pack, derived from the resolved
/// pack variant id — never from the app-owned local filename.
///
/// - `w4Legacy`: the known-good current W4 path (FP16 boundaries as today).
/// - `w8LegacyStabilized`: production W8-v2 resolution. Uses the same legacy
///   attention/activation numerics that already produced coherent
///   full-inference CI output for W8 — the BF16 range-emulation path is not
///   yet complete, so production W8 temporarily runs legacy numerics rather
///   than an unfinished experimental path.
/// - `w8BF16Experimental`: BF16-range emulation for W8 (BF16 RNE rounding in
///   FP32 storage, mapping onto the existing `ActivationNumerics` BF16
///   machinery). Retained ONLY for explicitly-requested experiments/diagnostics;
///   it is NOT selected by variant-id resolution.
enum DiTNumericsPolicy: String, Equatable {
    case w4Legacy
    case w8LegacyStabilized
    case w8BF16Experimental

    static func fromVariantID(_ id: String) -> DiTNumericsPolicy {
        id == "w8-v2" ? .w8LegacyStabilized : .w4Legacy
    }
}

enum ModelManifest {
    static let releaseTag = "model-assets-v1"
    private static let releaseBase = URL(
        string: "https://github.com/invisiblestrangler/AnimaXS/releases/download/\(releaseTag)/")!

    /// Descriptor for the primary (W4) DiT variant.
    static let ditW4 = ModelVariantDescriptor(
        id: "w4",
        displayFilename: "anima-turbo-v1.0-xsmax-w4.animapk",
        size: 1_179_435_008,
        sha256: "ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25")

    /// Descriptor for the alternate (W8-v2) DiT variant.
    static let ditW8V2 = ModelVariantDescriptor(
        id: "w8-v2",
        displayFilename: "anima-turbo-v1.0-xsmax-w8-v2.animapk",
        size: 2_232_975_360,
        sha256: "8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130")

    /// Matches `matchedSize` + `matchedSHA256` against the entry's primary
    /// variant then accepted alternates, returning the corresponding
    /// `ModelVariantDescriptor`. Throws when neither matches.
    static func descriptor(
        for entry: ModelManifestEntry, matchedSize: UInt64, matchedSHA256: String
    ) throws -> ModelVariantDescriptor {
        let normalizedSHA = matchedSHA256.lowercased()
        // Primary match
        if entry.size == matchedSize, entry.sha256.lowercased() == normalizedSHA {
            return ModelVariantDescriptor(
                id: primaryVariantID(for: entry),
                displayFilename: entry.filename,
                size: entry.size,
                sha256: entry.sha256.lowercased())
        }
        // Alternate match
        for (index, alt) in entry.alternates.enumerated() {
            if alt.size == matchedSize, alt.sha256.lowercased() == normalizedSHA {
                return alternateDescriptor(for: entry, alternateIndex: index, variant: alt)
            }
        }
        throw AnimapkError.validation(
            "no manifest variant matches size \(matchedSize) / SHA-256 for \(entry.component.rawValue)")
    }

    /// The variant id for the primary variant of an entry. The DiT entry's
    /// primary is "w4"; all other entries have a single variant so their
    /// primary id is the component name.
    private static func primaryVariantID(for entry: ModelManifestEntry) -> String {
        if entry.component == .dit { return ditW4.id }
        return entry.component.rawValue
    }

    /// The descriptor for an alternate variant. Only the DiT entry has
    /// alternates; the W8-v2 alternate is identified by its known descriptor.
    private static func alternateDescriptor(
        for entry: ModelManifestEntry, alternateIndex: Int, variant: ModelVariant
    ) -> ModelVariantDescriptor {
        if entry.component == .dit, alternateIndex == 0 {
            return ditW8V2
        }
        // Fallback for any future alternate: synthesize a descriptor.
        return ModelVariantDescriptor(
            id: "\(entry.component.rawValue)-alt\(alternateIndex + 1)",
            displayFilename: entry.filename,
            size: variant.size,
            sha256: variant.sha256.lowercased())
    }

    static let entries: [ModelManifestEntry] = [
        // The DiT slot accepts either the W4 or the W8-v2 pack. Whichever is
        // imported is the DiT used by generation. W4 remains the primary so
        // existing W4 installs/receipts stay valid.
        entry(
            "anima-turbo-v1.0-xsmax-w4.animapk", size: 1_179_435_008,
            sha256: "ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25",
            component: .dit,
            alternates: [
                ModelVariant(
                    size: 2_232_975_360,
                    sha256: "8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130"),
            ]),
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
            // Bound the lifetime of each chunk's temporary `Data` so Foundation
            // autoreleased temporaries cannot accumulate across thousands of
            // reads on large (multi-hundred-MB / multi-GB) model files. This is
            // pure lifetime hardening: behavior, chunk size, and hashes are
            // unchanged, and the file is never mmap'd nor read via
            // `Data(contentsOf:)`.
            let reachedEOF = try autoreleasepool { () throws -> Bool in
                let data = try handle.read(upToCount: chunkBytes) ?? Data()
                guard !data.isEmpty else { return true }
                digest.update(data: data)
                return false
            }
            if reachedEOF { break }
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Matches `url` against any accepted variant of `entry`. Returns the
    /// matched variant. Throws when size matches no accepted variant or the
    /// digest matches no accepted variant. Performs one streaming hash pass
    /// against the first size that matches (and the primary first), which is
    /// the normal-case single pass for W4/W8.
    static func matchedVariant(of url: URL, against entry: ModelManifestEntry) throws -> ModelVariant {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attributes[.size] as? NSNumber else {
            throw AnimapkError.validation("unable to read model file size")
        }
        let actualSize = number.uint64Value
        let sizeMatches = entry.allVariants.filter { $0.size == actualSize }
        guard !sizeMatches.isEmpty else {
            throw AnimapkError.validation(
                "model size does not match any manifest variant for \(entry.component.rawValue)")
        }
        let digest = try sha256(of: url)
        guard let variant = sizeMatches.first(where: { $0.sha256.lowercased() == digest }) else {
            throw AnimapkError.validation(
                "model SHA-256 does not match manifest for \(entry.component.rawValue)")
        }
        return variant
    }

    static func verify(_ url: URL, against entry: ModelManifestEntry) throws {
        _ = try matchedVariant(of: url, against: entry)
    }

    private static func entry(
        _ filename: String, size: UInt64, sha256: String, component: ModelComponent
    ) -> ModelManifestEntry {
        ModelManifestEntry(
            filename: filename, size: size, sha256: sha256,
            url: releaseBase.appendingPathComponent(filename), component: component,
            alternates: [])
    }

    private static func entry(
        _ filename: String, size: UInt64, sha256: String, component: ModelComponent,
        alternates: [ModelVariant]
    ) -> ModelManifestEntry {
        ModelManifestEntry(
            filename: filename, size: size, sha256: sha256,
            url: releaseBase.appendingPathComponent(filename), component: component,
            alternates: alternates)
    }
}
