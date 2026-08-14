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
    typealias ModelVerifier = (URL, ModelManifestEntry) throws -> Void

    private let directory: URL
    private let downloader: Downloader
    private let availableCapacity: CapacityProvider
    private let secureInstalls: Bool
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
        receiptsDirectory: URL? = nil
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
        self.verifier = verifier ?? ModelManifest.verify
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
            try verifier(destination, entry)
            writeReceipt(for: entry, at: destination)
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
        var byComponent: [ModelComponent: URL] = [:]
        for entry in entries {
            if case .ready(let url) = await discover(entry) {
                byComponent[entry.component] = url
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
            try verifier(destination, entry)
            writeReceipt(for: entry, at: destination)
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
    /// is verified (size + SHA-256 against the manifest) exactly like a
    /// download, then installed atomically, replacing any existing (possibly
    /// corrupt) destination. Rejects arbitrary mismatched files. The caller is
    /// responsible for holding security-scoped access to `source` for the
    /// whole operation.
    func importPack(_ entry: ModelManifestEntry, from source: URL) async throws -> URL {
        // Verify the source first.
        try verifier(source, entry)
        let destination = localURL(for: entry)
        let finalURL = try installVerified(source, to: destination, entry: entry)
        writeReceipt(for: entry, at: finalURL)
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
            try verifier(downloaded, entry)
            let finalURL = try installVerified(downloaded, to: destination, entry: entry)
            writeReceipt(for: entry, at: finalURL)
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
        let staging = directory.appendingPathComponent(
            ".\(entry.filename).staging-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: source, to: staging)
        if secureInstalls {
            try secure(staging)
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            return try FileManager.default.replaceItemAt(destination, withItemAt: staging)
                ?? destination
        }
        try FileManager.default.moveItem(at: staging, to: destination)
        return destination
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

    // MARK: - Verification receipts

    private func receiptIsValid(for entry: ModelManifestEntry, at url: URL) -> Bool {
        guard let receipt = receiptStore.receipt(for: entry.filename),
              receipt.schemaVersion == VerificationReceipt.currentSchemaVersion,
              receipt.expectedSize == entry.size,
              receipt.expectedSHA256 == entry.sha256.lowercased(),
              receipt.installedSize == entry.size,
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value,
              size == entry.size else {
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
    /// the next launch — it never blocks readiness.
    private func writeReceipt(for entry: ModelManifestEntry, at url: URL) {
        var receipts = receiptStore.load()
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        receipts[entry.filename] = VerificationReceipt(
            schemaVersion: VerificationReceipt.currentSchemaVersion,
            component: entry.component,
            filename: entry.filename,
            expectedSize: entry.size,
            expectedSHA256: entry.sha256.lowercased(),
            verifiedAt: Date(),
            installedSize: (attributes?[.size] as? NSNumber)?.uint64Value ?? entry.size,
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
