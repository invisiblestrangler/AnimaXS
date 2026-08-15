import Foundation

/// Pinned experimental W8 v2 DiT pack metadata (Diagnostics-only).
///
/// Deliberately NOT part of `ModelManifest.entries`: the normal three-pack
/// production topology expects exactly one production entry per model role.
/// Adding a second `.dit` to the production manifest would risk role
/// collisions, incorrect installed-state resolution, and broken three-pack
/// assumptions. W8 is selected per-run from Diagnostics only.
enum ExperimentalDiTManifest {
    /// Repository: ScalingBiz/AnimaXS-DiT-W8
    static let repo = "ScalingBiz/AnimaXS-DiT-W8"
    static let revision = "589d028122f872e66ee20cdd12cb55eb3b816add"
    static let filename = "anima-turbo-v1.0-xsmax-w8-v2.animapk"
    /// Verified against the HuggingFace LFS metadata (tree API, `lfs.size`):
    /// 2,232,975,360 bytes. Do NOT transcribe this from memory — the digit
    /// order is easy to flip (a previous 2_232_973_560 value rejected every
    /// import with a size mismatch).
    static let byteCount: UInt64 = 2_232_975_360
    static let sha256 = "8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130"

    /// The reference release URL (used only to display where the file comes
    /// from; the app never auto-downloads it).
    static var referenceURL: URL {
        URL(string: "https://huggingface.co/\(repo)/tree/\(revision)")!
    }
}
