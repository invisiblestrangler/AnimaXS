import Foundation

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
    private var states: [ModelComponent: State] = [:]
    private static let diskReserve: Int64 = 256 * 1_024 * 1_024

    init(
        directory: URL? = nil,
        downloader: Downloader? = nil,
        availableCapacity: CapacityProvider? = nil
    ) throws {
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
                // Leave a mismatched existing file untouched. The user can diagnose/remove it.
                states[entry.component] = .failed(error.localizedDescription)
                throw error
            }
        }
        do {
            let capacity = try availableCapacity(directory)
            guard capacity >= Int64(entry.size) + Self.diskReserve else {
                throw AnimapkError.validation("insufficient disk space for model download")
            }
            states[entry.component] = .downloading
            let downloaded = try await downloader(entry.url)
            states[entry.component] = .verifying
            try ModelManifest.verify(downloaded, against: entry)

            let staging = directory.appendingPathComponent(
                ".\(entry.filename).staging-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: staging) }
            try FileManager.default.copyItem(at: downloaded, to: staging)
            try secure(staging)
            try FileManager.default.moveItem(at: staging, to: destination)
            states[entry.component] = .ready(destination)
            return destination
        } catch {
            states[entry.component] = .failed(error.localizedDescription)
            throw error
        }
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
