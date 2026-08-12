import Foundation

/// Downloads, verifies, and installs the three production model packs, and
/// discovers valid already-installed packs after a cold launch.
///
/// State machine per component: `missing → downloading → verifying → ready`
/// or `failed`. A corrupt existing file is left `failed` until a repair or
/// re-import is requested; both re-validate via `ModelManifest.verify`.
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

    private let directory: URL
    private let downloader: Downloader
    private let availableCapacity: CapacityProvider
    private let secureInstalls: Bool
    private var states: [ModelComponent: State] = [:]
    private static let diskReserve: Int64 = 256 * 1_024 * 1_024

    init(
        directory: URL? = nil,
        downloader: Downloader? = nil,
        availableCapacity: CapacityProvider? = nil,
        secureInstalls: Bool = true
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
        self.downloader = downloader ?? { url in
            let (temporary, _) = try await URLSession.shared.download(from: url)
            return temporary
        }
        self.availableCapacity = availableCapacity ?? { url in
            let values = try url.resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey,
            ])
            return values.volumeAvailableCapacityForImportantUsage
                ?? Int64(values.volumeAvailableCapacity ?? 0)
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

    /// Returns an already verified model or downloads, verifies, and installs it.
    /// A component cannot start a second transfer while its first one is active.
    func prepare(_ entry: ModelManifestEntry) async throws -> URL {
        if case let .ready(url) = state(for: entry.component) { return url }
        switch state(for: entry.component) {
        case .downloading, .verifying:
            throw AnimapkError.validation("model transfer is already active")
        default:
            break
        }
        let destination = localURL(for: entry)
        if FileManager.default.fileExists(atPath: destination.path) {
            do {
                try ModelManifest.verify(destination, against: entry)
                states[entry.component] = .ready(destination)
                return destination
            } catch {
                // Leave a mismatched existing file for the user to repair/import.
                states[entry.component] = .failed(error.localizedDescription)
                throw error
            }
        }
        return try await downloadAndInstall(entry, destination: destination)
    }

    /// Repair path (K003/K004): remove/quarantine a mismatched existing file,
    /// then redownload and install it. Throws if the existing file is absent
    /// or there is nothing to repair.
    func repair(_ entry: ModelManifestEntry) async throws -> URL {
        let destination = localURL(for: entry)
        guard FileManager.default.fileExists(atPath: destination.path) else {
            states[entry.component] = .missing
            return try await downloadAndInstall(entry, destination: destination)
        }
        // Remove the invalid file before reinstalling so the state machine is
        // not left pointing at a stale corrupt path.
        try FileManager.default.removeItem(at: destination)
        return try await downloadAndInstall(entry, destination: destination)
    }

    /// Legal-block fallback: import a user-provided `.animapk` file. The file
    /// is verified (size + SHA-256 against the manifest) exactly like a
    /// download, then installed atomically. Rejects arbitrary mismatched files.
    func importPack(_ entry: ModelManifestEntry, from source: URL) async throws -> URL {
        // Verify the source first.
        try ModelManifest.verify(source, against: entry)
        let destination = localURL(for: entry)
        try installVerified(source, to: destination, entry: entry)
        states[entry.component] = .ready(destination)
        return destination
    }

    /// The single validated result for the three production packs (K002 §5.1):
    /// textEncoder, dit, vae. The adapter reads the DiT pack, so only these
    /// three URLs are exposed.
    ///
    /// - Parameter entries: Manifest entries to resolve. Defaults to the
    ///   production `ModelManifest.entries`; a test seam may inject tiny
    ///   synthetic entries to avoid the multi-GB real packs.
    func resolvedModels(entries: [ModelManifestEntry] = ModelManifest.entries) async throws -> ResolvedModels {
        let byComponent = Dictionary(
            uniqueKeysWithValues: try await withThrowingTaskGroup(of: (ModelComponent, URL).self) { group in
                for entry in entries {
                    group.addTask { (entry.component, try await self.prepare(entry)) }
                }
                var result: [(ModelComponent, URL)] = []
                for try await pair in group { result.append(pair) }
                return result
            })
        guard let textEncoder = byComponent[.textEncoder],
              let dit = byComponent[.dit],
              let vae = byComponent[.vae] else {
            throw AnimapkError.validation("incomplete production model set")
        }
        return ResolvedModels(textEncoder: textEncoder, dit: dit, vae: vae)
    }

    // MARK: - Private

    private func downloadAndInstall(
        _ entry: ModelManifestEntry, destination: URL
    ) async throws -> URL {
        do {
            let capacity = try availableCapacity(directory)
            guard capacity >= Int64(entry.size) + Self.diskReserve else {
                throw AnimapkError.validation(
                    "insufficient disk space for model download (need \(entry.size) bytes)")
            }
            states[entry.component] = .downloading
            let downloaded = try await downloader(entry.url)
            states[entry.component] = .verifying
            try ModelManifest.verify(downloaded, against: entry)
            try installVerified(downloaded, to: destination, entry: entry)
            states[entry.component] = .ready(destination)
            return destination
        } catch {
            states[entry.component] = .failed(error.localizedDescription)
            throw error
        }
    }

    /// Stages, secures, and atomically moves a verified file into place.
    private func installVerified(_ source: URL, to destination: URL, entry: ModelManifestEntry) throws {
        let staging = directory.appendingPathComponent(
            ".\\(entry.filename).staging-\\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: source, to: staging)
        if secureInstalls {
            try secure(staging)
        }
        try FileManager.default.moveItem(at: staging, to: destination)
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
