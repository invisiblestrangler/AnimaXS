import Foundation
import CryptoKit

extension Data {
    /// Hex-encoded SHA-256 digest (CryptoKit streaming/one-shot).
    var sha256Hex: String {
        SHA256.hash(data: self).map { String(format: "%02x", $0) }.joined()
    }
}
