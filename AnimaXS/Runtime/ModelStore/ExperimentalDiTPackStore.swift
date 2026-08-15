import Foundation

/// The verified metadata an experimental DiT pack must satisfy. The production
/// `ExperimentalDiTManifest` provides the pinned W8 v2 values; tests inject a
/// tiny synthetic spec so the store's import/verify/remove logic is exercised
/// without a 2.23 GB file.
protocol ExperimentalDiTPackSpec {
    var filename: String { get }
    var byteCount: UInt64 { get }
    var sha256: String { get }
}

/// Concrete pinned W8 v2 spec (see `ExperimentalDiTManifest`).
struct PinnedExperimentalW8Spec: ExperimentalDiTPackSpec {
    var filename: String { ExperimentalDiTManifest.filename }
    var byteCount: UInt64 { ExperimentalDiTManifest.byteCount }
    var sha256: String { ExperimentalDiTManifest.sha256 }
}

/// Verification receipt for an imported experimental DiT pack. Lets normal
/// discovery trust the installed file with cheap metadata checks instead of
/// re-hashing the multi-GB pack on every launch. If the file's identity no
/// longer agrees with the receipt, the pack is marked unverified and must be
/// re-imported — never silently trusted.
struct ExperimentalDiTReceipt: Codable, Equatable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let filename: String
    let expectedSHA256: String
    let expectedByteCount: UInt64
    let installedByteCount: UInt64
    let installedModificationDate: Date?
}

/// Isolated store for the experimental W8 DiT pack. Lives OUTSIDE the
/// production model directory (`Application Support/AnimaXS/ExperimentalModels/`)
/// and is never part of `ModelManifest.entries`.
///
/// Import is user-triggered only (Diagnostics): verify exact size + pinned
/// SHA-256, stage to a temporary file, then atomically move into place and
/// write a cheap verification receipt. No networking and no auto-download.
actor ExperimentalDiTPackStore {
    enum State: Equatable {
        case missing
        case verifying
        case ready(URL)
        case unverified
        case failed(String)
    }

    typealias Verifier = (URL) throws -> String
    typealias CapacityProvider = (URL) throws -> Int64

    private let directory: URL
    private let spec: ExperimentalDiTPackSpec
    private let verifier: Verifier
    private let availableCapacity: CapacityProvider
    private let secureInstalls: Bool
    private var state: State = .missing

    private static let diskReserve: Int64 = 256 * 1_024 * 1_024

    /// Creates the store for a given spec. Tests inject a tiny spec and a
    /// custom verifier; production uses the pinned W8 v2 spec.
    init(
        directory: URL? = nil,
        spec: ExperimentalDiTPackSpec = PinnedExperimentalW8Spec(),
        verifier: Verifier? = nil,
        availableCapacity: CapacityProvider? = nil,
        secureInstalls: Bool = true
    ) throws {
        if let directory {
            self.directory = directory
        } else {
            let base = try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true)
            self.directory = base.appendingPathComponent(
                "AnimaXS/ExperimentalModels", isDirectory: true)
        }
        self.spec = spec
        self.verifier = verifier ?? Self.sha256Verifier
        self.availableCapacity = availableCapacity ?? Self.defaultCapacityProvider()
        self.secureInstalls = secureInstalls
        try FileManager.default.createDirectory(
            at: self.directory, withIntermediateDirectories: true)
    }

    // MARK: - State / discovery

    var currentState: State { state }

    private var packURL: URL {
        directory.appendingPathComponent(spec.filename, isDirectory: false)
    }

    private var receiptURL: URL {
        directory.appendingPathComponent("\(spec.filename).receipt.json", isDirectory: false)
    }

    /// Local-only discovery. Uses the cheap verification receipt when valid; a
    /// changed/missing file or stale receipt marks the pack unverified (must be
    /// re-imported). Never re-hashes the multi-GB pack here.
    func discover() -> State {
        let url = packURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            state = .missing
            return .missing
        }
        if receiptIsValid(at: url) {
            state = .ready(url)
            return .ready(url)
        }
        state = .unverified
        return .unverified
    }

    /// User-triggered import of a verified pack from a security-scoped source
    /// URL (the caller holds security-scoped access for the whole operation).
    /// Rejects wrong byte size, verifies the pinned SHA-256, stages to a
    /// temporary file, then atomically moves the verified pack into place and
    /// writes a cheap verification receipt.
    func importPack(from source: URL) async throws -> URL {
        state = .verifying
        do {
            // Capacity gate before any copy.
            let capacity: Int64
            do {
                capacity = try availableCapacity(directory)
            } catch {
                throw AnimapkError.validation(
                    "unable to determine available disk space: \(error.localizedDescription)")
            }
            guard capacity >= Int64(spec.byteCount) + Self.diskReserve else {
                throw AnimapkError.validation(
                    "insufficient disk space for experimental W8 import (need \(spec.byteCount) bytes + reserve)")
            }
            // Verify the source exactly (size + pinned SHA-256) BEFORE staging.
            let sourceSize = try fileSize(at: source)
            guard sourceSize == spec.byteCount else {
                throw AnimapkError.validation(
                    "experimental W8 size mismatch: got \(sourceSize), expected \(spec.byteCount)")
            }
            let digest = try verifier(source)
            guard digest == spec.sha256.lowercased() else {
                throw AnimapkError.validation("experimental W8 SHA-256 mismatch")
            }
            // Stage to a temporary file, then atomically move into place.
            // Replaces an existing (possibly corrupt) destination: moveItem
            // cannot overwrite, so a stale W8 file must never block re-import.
            let staging = directory.appendingPathComponent(
                ".\(spec.filename).staging-\(UUID().uuidString)")
            do {
                try FileManager.default.copyItem(at: source, to: staging)
                if secureInstalls {
                    try secure(staging)
                }
                if FileManager.default.fileExists(atPath: packURL.path) {
                    guard let replaced = try FileManager.default.replaceItemAt(
                        packURL, withItemAt: staging) else {
                        throw AnimapkError.validation(
                            "failed to replace existing experimental W8 pack")
                    }
                    writeReceipt(at: replaced)
                    state = .ready(replaced)
                    return replaced
                }
                try FileManager.default.moveItem(at: staging, to: packURL)
            } catch {
                try? FileManager.default.removeItem(at: staging)
                throw AnimapkError.validation("failed to install experimental W8 pack: \(error.localizedDescription)")
            }
            writeReceipt(at: packURL)
            state = .ready(packURL)
            return packURL
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }
    }

    /// Removes the pack and its receipt (Diagnostics "Remove W8 v2").
    func remove() {
        try? FileManager.default.removeItem(at: packURL)
        try? FileManager.default.removeItem(at: receiptURL)
        state = .missing
    }

    // MARK: - Verifiers / capacity

    /// SHA-256 via streaming chunked reads (never loads the whole file).
    static func sha256Verifier(url: URL) throws -> String {
        try ModelManifest.sha256(of: url)
    }

    static func defaultCapacityProvider() -> CapacityProvider {
        { url in
            let values = try url.resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey,
            ])
            if let important = values.volumeAvailableCapacityForImportantUsage {
                return important
            }
            if let capacity = values.volumeAvailableCapacity {
                return Int64(capacity)
            }
            throw AnimapkError.validation(
                "unable to determine available disk space on this volume")
        }
    }

    // MARK: - Private

    private func fileSize(at url: URL) throws -> UInt64 {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let number = attributes[.size] as? NSNumber else {
            throw AnimapkError.validation("unable to read file size")
        }
        return number.uint64Value
    }

    private func receiptIsValid(at url: URL) -> Bool {
        guard let data = try? Data(contentsOf: receiptURL),
              let receipt = try? JSONDecoder().decode(ExperimentalDiTReceipt.self, from: data),
              receipt.schemaVersion == ExperimentalDiTReceipt.currentSchemaVersion,
              receipt.filename == spec.filename,
              receipt.expectedSHA256 == spec.sha256.lowercased(),
              receipt.expectedByteCount == spec.byteCount,
              receipt.installedByteCount == spec.byteCount,
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value,
              size == spec.byteCount else {
            return false
        }
        if let modificationDate = attributes[.modificationDate] as? Date,
           let receiptDate = receipt.installedModificationDate {
            return abs(modificationDate.timeIntervalSince(receiptDate)) < 0.5
        }
        return receipt.installedModificationDate == nil
    }

    private func writeReceipt(at url: URL) {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let receipt = ExperimentalDiTReceipt(
            schemaVersion: ExperimentalDiTReceipt.currentSchemaVersion,
            filename: spec.filename,
            expectedSHA256: spec.sha256.lowercased(),
            expectedByteCount: spec.byteCount,
            installedByteCount: (attributes?[.size] as? NSNumber)?.uint64Value ?? spec.byteCount,
            installedModificationDate: attributes?[.modificationDate] as? Date)
        guard let data = try? JSONEncoder().encode(receipt) else { return }
        try? data.write(to: receiptURL, options: .atomic)
    }

    private func secure(_ url: URL) throws {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try mutableURL.setResourceValues(values)
        try FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
            ofItemAtPath: url.path)
    }
}
