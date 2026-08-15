import CryptoKit
import Foundation

/// Downloads, verifies, and installs the three production model packs, and
/// discovers valid already-installed packs after a cold launch.
///
/// State machine per component: `missing → downloading → verifying → ready`
/// or `failed`. A corrupt existing file is left `failed` until a repair or
/// re-import is requested; both re-validate via `ModelManifest.verify`.
///
/// ## Local discovery vs explicit acquisition
///
/// Discovery (`discover` / `discoverInstalled` / `resolveInstalledModels`) is
/// local-only and **never initiates a network transfer**:
/// - a file covered by a valid verification receipt is trusted with cheap
///   stat/metadata checks only (no SHA-256);
/// - a file without a valid receipt gets one full size + SHA-256 verification
///   (off the main actor, on this actor) and then a receipt is persisted;
/// - a missing file is simply reported missing.
///
/// Acquisition (`download` / `repair` / `importPack` / `verifyExisting`) is
/// always an explicit user action and is the only path that can touch the
/// network or write a new installed file.
actor ModelStore {
    enum State: Equatable {
        case missing
        case downloading
        case verifying
        case ready(URL)
        case failed(String)
    }

    typealias Downloader = (URL) async throws -> URL
    typealias CapacityProvider = (URL) throws -> Int64
    /// Full-file verification returning the matched (size, SHA-256) variant.
    /// Default: `ModelManifest.matchedVariant` (one streaming hash pass).
    typealias ModelVerifier = (URL, ModelManifestEntry) throws -> ModelVariant

    private let directory: URL
    private let downloader: Downloader
    private let availableCapacity: CapacityProvider
    private let secureInstalls: Bool
    /// Read/write chunk size for the single-pass copy+hash stream. Production
    /// defaults to 1 MiB (bounded memory); tests inject a tiny value to force
    /// many stream iterations without a multi-GB fixture.
    private let chunkBytes: Int
    /// Verification hook for the explicit full-file re-verify paths
    /// (`discover` on a missing/stale receipt, `verifyExisting`). Import and
    /// download verify DURING a single streaming copy pass (`verifyAndStage`)
    /// and do not route through this hook.
    private let verifier: ModelVerifier
    private let receiptStore: VerificationReceiptStore
    private var states: [ModelComponent: State] = [:]
    private static let diskReserve: Int64 = 256 * 1_024 * 1_024

    init(
        directory: URL? = nil,
        downloader: Downloader? = nil,
        availableCapacity: CapacityProvider? = nil,
        secureInstalls: Bool = true,
        verifier: ModelVerifier? = nil,
        receiptsDirectory: URL? = nil,
        chunkBytes: Int = 1 << 20
    ) throws {
        self.secureInstalls = secureInstalls
        if let directory {
            self.directory = directory
        } else {
            let base = try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true)
            self.directory = base.appendingPathComponent("AnimaXS/Models", isDirectory: true)
        }
        self.downloader = downloader ?? Self.productionDownloader()
        self.availableCapacity = availableCapacity ?? Self.defaultCapacityProvider()
        self.verifier = verifier ?? ModelManifest.matchedVariant
        self.chunkBytes = chunkBytes
        if let receiptsDirectory {
            self.receiptStore = VerificationReceiptStore(directory: receiptsDirectory)
        } else {
            // Receipts live next to the models directory (not inside it, so
            // they are never mistaken for model packs).
            self.receiptStore = VerificationReceiptStore(
                directory: self.directory.deletingLastPathComponent()
                    .appendingPathComponent("Receipts", isDirectory: true))
        }
        try FileManager.default.createDirectory(
            at: self.directory, withIntermediateDirectories: true)
    }

    func state(for component: ModelComponent) -> State {
        states[component] ?? .missing
    }

    func localURL(for entry: ModelManifestEntry) -> URL {
        directory.appendingPathComponent(entry.filename, isDirectory: false)
    }

    // MARK: - Local discovery (never downloads)

    /// Local-only state for one entry. A valid verification receipt makes this
    /// a pure stat/metadata check; otherwise one full verification runs on this
    /// actor (off the main actor) and writes a receipt. Never touches the
    /// network.
    func discover(_ entry: ModelManifestEntry) async -> State {
        switch states[entry.component] {
        case .downloading, .verifying:
            return states[entry.component] ?? .missing
        default:
            break
        }
        let destination = localURL(for: entry)
        guard FileManager.default.fileExists(atPath: destination.path) else {
            states[entry.component] = .missing
            return .missing
        }
        if receiptIsValid(for: entry, at: destination) {
            states[entry.component] = .ready(destination)
            return .ready(destination)
        }
        // No (valid) receipt: one full local verification, then persist a
        // receipt so the next launch is cheap. This is a local read — never a
        // download — and it runs on this actor, off the main actor.
        states[entry.component] = .verifying
        do {
            let variant = try verifier(destination, entry)
            writeReceipt(for: entry, at: destination, variant: variant)
            states[entry.component] = .ready(destination)
            return .ready(destination)
        } catch {
            let message = error.localizedDescription
            states[entry.component] = .failed(message)
            return .failed(message)
        }
    }

    /// Local-only discovery of every entry. No network, ever.
    func discoverInstalled(
        entries: [ModelManifestEntry] = ModelManifest.entries
    ) async -> [ModelComponent: State] {
        var result: [ModelComponent: State] = [:]
        for entry in entries {
            result[entry.component] = await discover(entry)
        }
        return result
    }

    /// The single validated result for the three production packs (K002 §5.1):
    /// textEncoder, dit, vae. The adapter reads the DiT pack, so only these
    /// three URLs are exposed.
    ///
    /// Local-only: missing packs are NOT downloaded. Throws when the set is
    /// incomplete so callers can present a clear blocked state.
    ///
    /// - Parameter entries: Manifest entries to resolve. Defaults to the
    ///   production `ModelManifest.entries`; a test seam may inject tiny
    ///   synthetic entries to avoid the multi-GB real packs.
    func resolveInstalledModels(
        entries: [ModelManifestEntry] = ModelManifest.entries
    ) async throws -> ResolvedModels {
        var byComponent: [ModelComponent: ResolvedModelPack] = [:]
        for entry in entries {
            if case .ready(let url) = await discover(entry) {
                byComponent[entry.component] = try resolvedPack(entry: entry, url: url)
            }
        }
        guard let textEncoder = byComponent[.textEncoder],
              let dit = byComponent[.dit],
              let vae = byComponent[.vae] else {
            throw AnimapkError.validation(
                "incomplete production model set (local discovery only)")
        }
        return ResolvedModels(textEncoder: textEncoder, dit: dit, vae: vae)
    }

    /// Builds a `ResolvedModelPack` for a ready slot from the slot's valid
    /// verification receipt. The receipt already records the ACTUAL matched
    /// variant (W4 or W8-v2 for the DiT slot), so this is a metadata read —
    /// it never re-hashes the pack.
    private func resolvedPack(
        entry: ModelManifestEntry, url: URL
    ) throws -> ResolvedModelPack {
        guard let receipt = receiptStore.receipt(for: entry.filename),
              let variant = try? ModelManifest.descriptor(
                  for: entry,
                  matchedSize: receipt.expectedSize,
                  matchedSHA256: receipt.expectedSHA256) else {
            throw AnimapkError.validation(
                "valid receipt for \\(entry.component.rawValue) but no manifest variant matches")
        }
        return ResolvedModelPack(url: url, component: entry.component, variant: variant)
    }

    /// Cheap, hashing-free file facts for diagnostics snapshots: existence,
    /// size, and whether a valid verification receipt covers the file.
    func fileStatus(for entry: ModelManifestEntry) async -> ModelFileStatus {
        let url = localURL(for: entry)
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value else {
            return ModelFileStatus(exists: false, sizeBytes: 0, receiptValid: false)
        }
        return ModelFileStatus(
            exists: true, sizeBytes: size,
            receiptValid: receiptIsValid(for: entry, at: url))
    }

    // MARK: - Explicit acquisition (user-triggered only)

    /// Explicit download + verify + install. Refuses to start a second
    /// transfer for the same component while one is active. Returns the
    /// already-installed URL when the component is already ready.
    func download(_ entry: ModelManifestEntry) async throws -> URL {
        if case .ready(let url) = state(for: entry.component) { return url }
        switch state(for: entry.component) {
        case .downloading, .verifying:
            throw AnimapkError.validation("model transfer is already active")
        default:
            break
        }
        return try await downloadAndInstall(entry, destination: localURL(for: entry))
    }

    /// Re-verifies an existing local file (full size + SHA-256) and, on
    /// success, refreshes its verification receipt. Used by the explicit
    /// "Retry" action on a failed row. Never downloads.
    func verifyExisting(_ entry: ModelManifestEntry) async throws -> URL {
        let destination = localURL(for: entry)
        guard FileManager.default.fileExists(atPath: destination.path) else {
            states[entry.component] = .missing
            throw AnimapkError.validation(
                "no local model file to verify for \(entry.component.rawValue)")
        }
        states[entry.component] = .verifying
        do {
            let variant = try verifier(destination, entry)
            writeReceipt(for: entry, at: destination, variant: variant)
            states[entry.component] = .ready(destination)
            return destination
        } catch {
            let message = error.localizedDescription
            states[entry.component] = .failed(message)
            throw error
        }
    }

    /// Repair path (K003/K004): remove/quarantine a mismatched existing file,
    /// then redownload and install it. Throws if the existing file is absent
    /// or there is nothing to repair.
    func repair(_ entry: ModelManifestEntry) async throws -> URL {
        let destination = localURL(for: entry)
        if FileManager.default.fileExists(atPath: destination.path) {
            // Remove the invalid file before reinstalling so the state machine
            // is not left pointing at a stale corrupt path.
            try FileManager.default.removeItem(at: destination)
        }
        return try await downloadAndInstall(entry, destination: destination)
    }

    /// Legal-block fallback: import a user-provided `.animapk` file. The file
    /// is verified (size + SHA-256 against the manifest's accepted variants)
    /// while being streamed into staging exactly once, then installed
    /// atomically, replacing any existing (possibly corrupt) destination. This
    /// single-pass copy+hash bounds memory on multi-GB packs (W8-v2 is 2.23 GB)
    /// instead of the previous full SHA pass followed by a full `copyItem`.
    /// Rejects arbitrary mismatched files. The caller is responsible for
    /// holding security-scoped access to `source` for the whole operation.
    func importPack(_ entry: ModelManifestEntry, from source: URL) async throws -> URL {
        // Verify + copy in a single bounded-memory pass (size gate first).
        let variant = try Self.verifyAndStage(
            from: source, entry: entry, secureInstalls: secureInstalls,
            installDirectory: directory, chunkBytes: chunkBytes)
        let staging = variant.staging
        defer { try? FileManager.default.removeItem(at: staging) }
        let destination = localURL(for: entry)
        let finalURL = try installVerified(staging, to: destination, entry: entry)
        writeReceipt(for: entry, at: finalURL, variant: variant.matched)
        states[entry.component] = .ready(finalURL)
        return finalURL
    }

    // MARK: - Production downloader

    /// Default production downloader: validates the final HTTP response,
    /// requires 2xx, and moves the completed temporary download into an
    /// app-owned staging file before returning so later verification/install
    /// steps never depend on a URLSession-owned temporary file's lifetime.
    /// (GitHub Releases redirects are followed by URLSession automatically;
    /// the final response is the one checked.)
    static func productionDownloader(session: URLSession = .shared) -> Downloader {
        { url in
            let (temporary, response) = try await session.download(from: url)
            guard let http = response as? HTTPURLResponse else {
                throw AnimapkError.validation(
                    "download of \(url.absoluteString) returned a non-HTTP response")
            }
            guard (200..<300).contains(http.statusCode) else {
                throw AnimapkError.validation(
                    "download of \(url.absoluteString) failed with HTTP \(http.statusCode)")
            }
            let staging = FileManager.default.temporaryDirectory
                .appendingPathComponent("anima-xs-download-\(UUID().uuidString).bin")
            try FileManager.default.moveItem(at: temporary, to: staging)
            return staging
        }
    }

    /// Default capacity provider: surfaces an explicit error when the volume
    /// capacity cannot be determined instead of silently treating it as zero.
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

    /// Single-pass bounded-memory verify + stage. Streams `source` into an
    /// app-owned staging file in the install directory EXACTLY ONCE while
    /// incrementally computing SHA-256, then matches the result against the
    /// entry's accepted variants (W4 or W8-v2 for the DiT slot). This replaces
    /// the old full SHA pass + full `copyItem` pass, removing ~2.23 GB of
    /// redundant I/O for the W8-v2 pack and bounding per-chunk memory on the
    /// physical device.
    ///
    /// - Returns: the staging URL (already in the install directory, ready to
    ///   be secured + atomically installed) and the matched variant.
    /// - Throws on read/write failure, size mismatch at stat or EOF, or a
    ///   digest that matches no accepted variant. On any throw the staging
    ///   file is removed.
    static func verifyAndStage(
        from source: URL, entry: ModelManifestEntry,
        secureInstalls: Bool, installDirectory: URL, chunkBytes: Int = 1 << 20
    ) throws -> (staging: URL, matched: ModelVariant) {
        guard chunkBytes > 0 else {
            throw AnimapkError.validation("model import chunk size must be positive")
        }
        guard let sourceSize = (try? FileManager.default.attributesOfItem(atPath: source.path))?[.size] as? NSNumber else {
            throw AnimapkError.validation("unable to read model file size")
        }
        let expectedSize: UInt64 = sourceSize.uint64Value
        // Cheap size gate: the source must match one accepted variant size
        // BEFORE we stream anything.
        guard entry.allVariants.contains(where: { $0.size == expectedSize }) else {
            throw AnimapkError.validation(
                "model size does not match any manifest variant for \(entry.component.rawValue)")
        }

        let staging = installDirectory.appendingPathComponent(
            ".\(entry.filename).staging-\(UUID().uuidString)")

        // On success the CALLER owns `staging` (it installs/moves it, and its
        // own defer cleans up on later failure). On any failure BEFORE a clean
        // return, remove the partial staging file here so no `.<name>.staging-*`
        // is ever left behind (size mismatch, read/write failure, digest
        // mismatch, or a thrown `secure()`).
        var handoff = false
        defer {
            if !handoff {
                try? FileManager.default.removeItem(at: staging)
            }
        }

        let sourceHandle = try FileHandle(forReadingFrom: source)
        defer { try? sourceHandle.close() }
        let created = FileManager.default.createFile(atPath: staging.path, contents: nil)
        guard created else {
            throw AnimapkError.validation("failed to create staging file for model import")
        }
        let stagingHandle = try FileHandle(forWritingTo: staging)
        defer { try? stagingHandle.close() }

        var digest = SHA256()
        var bytesCopied: UInt64 = 0
        while true {
            let reachedEOF = try autoreleasepool { () throws -> Bool in
                let data = try sourceHandle.read(upToCount: chunkBytes) ?? Data()
                guard !data.isEmpty else { return true }
                digest.update(data: data)
                try stagingHandle.write(contentsOf: data)
                bytesCopied += UInt64(data.count)
                if bytesCopied > expectedSize {
                    throw AnimapkError.validation(
                        "model size changed while importing (\(entry.component.rawValue))")
                }
                return false
            }
            if reachedEOF { break }
        }
        guard bytesCopied == expectedSize else {
            throw AnimapkError.validation(
                "model size mismatch: got \(bytesCopied), expected \(expectedSize)")
        }
        let hex = digest.finalize().map { String(format: "%02x", $0) }.joined()
        guard let variant = entry.allVariants.first(where: { $0.sha256.lowercased() == hex }) else {
            throw AnimapkError.validation(
                "model SHA-256 does not match manifest for \(entry.component.rawValue)")
        }
        if secureInstalls {
            try secure(staging)
        }
        // Hand the staging file to the caller (it installs/moves it into
        // place and cleans up on any later failure). Suppress the local
        // cleanup so the file survives past this function's return.
        handoff = true
        return (staging, variant)
    }

    private func downloadAndInstall(
        _ entry: ModelManifestEntry, destination: URL
    ) async throws -> URL {
        do {
            let capacity: Int64
            do {
                capacity = try availableCapacity(directory)
            } catch {
                throw AnimapkError.validation(
                    "unable to determine available disk space: \(error.localizedDescription)")
            }
            guard capacity >= Int64(entry.size) + Self.diskReserve else {
                throw AnimapkError.validation(
                    "insufficient disk space for model download (need \(entry.size) bytes)")
            }
            states[entry.component] = .downloading
            let downloaded = try await downloader(entry.url)
            // The downloader staging file is app-owned and only needed until
            // verification + install complete.
            defer { try? FileManager.default.removeItem(at: downloaded) }
            states[entry.component] = .verifying
            let variant = try Self.verifyAndStage(
                from: downloaded, entry: entry, secureInstalls: secureInstalls,
                installDirectory: directory, chunkBytes: chunkBytes)
            let staging = variant.staging
            defer { try? FileManager.default.removeItem(at: staging) }
            let finalURL = try installVerified(staging, to: destination, entry: entry)
            writeReceipt(for: entry, at: finalURL, variant: variant.matched)
            states[entry.component] = .ready(finalURL)
            return finalURL
        } catch {
            states[entry.component] = .failed(error.localizedDescription)
            throw error
        }
    }

    /// Stages, secures, and atomically moves a verified file into place.
    /// Replaces an existing (possibly corrupt) destination instead of failing:
    /// `moveItem` cannot overwrite, so a corrupt destination would otherwise
    /// block every repair/import forever.
    /// - Returns: the final installed URL (may differ from `destination` when
    ///   the system performs a true replace).
    private func installVerified(_ source: URL, to destination: URL, entry: ModelManifestEntry) throws -> URL {
        // The staging file is already secured inside `verifyAndStage` when
        // `secureInstalls` is true; `source` here is that staging file.
        if FileManager.default.fileExists(atPath: destination.path) {
            // Replace (not move): moveItem cannot overwrite an existing file,
            // which would block every repair/import over a corrupt destination.
            guard let replaced = try FileManager.default.replaceItemAt(
                destination, withItemAt: source) else {
                throw AnimapkError.validation(
                    "failed to replace existing model file at \(destination.path)")
            }
            return replaced
        }
        try FileManager.default.moveItem(at: source, to: destination)
        return destination
    }

    private static func secure(_ url: URL) throws {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try mutableURL.setResourceValues(values)
        try FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
            ofItemAtPath: url.path)
    }

    // MARK: - Verification receipts

    private func receiptIsValid(for entry: ModelManifestEntry, at url: URL) -> Bool {
        guard let receipt = receiptStore.receipt(for: entry.filename),
              receipt.schemaVersion == VerificationReceipt.currentSchemaVersion,
              // The receipt records the ACTUAL matched variant (W4 or W8-v2 for
              // the DiT slot), which must still be an accepted variant today.
              entry.allVariants.contains(where: {
                  $0.size == receipt.expectedSize && $0.sha256.lowercased() == receipt.expectedSHA256
              }),
              receipt.installedSize == receipt.expectedSize,
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value,
              size == receipt.expectedSize else {
            return false
        }
        // File metadata must still match the receipt: a rewrite (or touch)
        // invalidates the receipt and forces one full verification.
        if let modificationDate = attributes[.modificationDate] as? Date,
           let receiptDate = receipt.installedModificationDate {
            return abs(modificationDate.timeIntervalSince(receiptDate)) < 0.5
        }
        return receipt.installedModificationDate == nil
    }

    /// Best-effort: a failed receipt write only costs one full verification on
    /// the next launch — it never blocks readiness. The receipt records the
    /// matched variant (size + SHA-256) so discovery accepts whichever pack
    /// (W4 or W8-v2) was actually installed into the slot.
    private func writeReceipt(
        for entry: ModelManifestEntry, at url: URL, variant: ModelVariant
    ) {
        var receipts = receiptStore.load()
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        receipts[entry.filename] = VerificationReceipt(
            schemaVersion: VerificationReceipt.currentSchemaVersion,
            component: entry.component,
            filename: entry.filename,
            expectedSize: variant.size,
            expectedSHA256: variant.sha256.lowercased(),
            verifiedAt: Date(),
            installedSize: (attributes?[.size] as? NSNumber)?.uint64Value ?? variant.size,
            installedModificationDate: attributes?[.modificationDate] as? Date)
        try? receiptStore.save(receipts)
    }
}

/// Hashing-free model file facts for the cheap diagnostics snapshot.
struct ModelFileStatus: Equatable {
    let exists: Bool
    let sizeBytes: UInt64
    let receiptValid: Bool
}
