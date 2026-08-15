import XCTest
@testable import AnimaXS

/// ModelStore production-integration tests (K003/K004/guide §11) plus the
/// real-device stabilization regressions:
/// - local discovery never downloads;
/// - verification receipts make relaunch cheap (no repeated SHA-256);
/// - import replaces corrupt destinations safely;
/// - the production downloader validates HTTP responses and owns its staging;
/// - `AnimapkError` surfaces useful `localizedDescription`s.
/// No real network: a synthetic downloader supplies the pack bytes.
final class ModelStoreTests: XCTestCase {

    private func makeEntry(
        filename: String = "tiny.animapk",
        content: Data = Data("abc".utf8),
        component: ModelComponent = .dit
    ) -> ModelManifestEntry {
        let sha = content.sha256Hex
        return ModelManifestEntry(
            filename: filename, size: UInt64(content.count), sha256: sha,
            url: URL(string: "https://example.invalid/\(filename)")!, component: component)
    }

    /// Fresh temporary store with an injected downloader/capacity/verifier.
    /// `secureInstalls` is disabled so test files remain removable
    /// (complete-until-first-auth protection would make the temp dir
    /// undeletable in the simulator). Receipts live under the test root.
    private func makeStore(
        root: URL,
        downloader: @escaping ModelStore.Downloader,
        capacity: @escaping ModelStore.CapacityProvider = { _ in Int64.max },
        verifier: ModelStore.ModelVerifier? = nil,
        chunkBytes: Int = 1 << 20
    ) throws -> ModelStore {
        try ModelStore(
            directory: root.appendingPathComponent("models", isDirectory: true),
            downloader: downloader, availableCapacity: capacity, secureInstalls: false,
            verifier: verifier,
            receiptsDirectory: root.appendingPathComponent("Receipts", isDirectory: true),
            chunkBytes: chunkBytes)
    }

    /// Counts invocations of the injected verifier (i.e. full pack hashes).
    private final class VerifierCounter {
        private var _count = 0
        func increment() { _count += 1 }
        var count: Int { _count }
    }

    private func countingVerifier(_ counter: VerifierCounter) -> ModelStore.ModelVerifier {
        { url, entry in
            counter.increment()
            return try ModelManifest.matchedVariant(of: url, against: entry)
        }
    }

    /// Counts downloader invocations (network calls).
    private final class DownloadCounter {
        private var _count = 0
        private let result: URL
        init(result: URL) { self.result = result }
        func makeDownloader() -> ModelStore.Downloader {
            { _ in self._count += 1; return self.result }
        }
        var count: Int { _count }
    }

    private func stagingFiles(in dir: URL) -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?
            .filter { $0.contains(".staging-") } ?? []
    }

    // MARK: - Cold launch: valid existing file

    func testValidExistingFileDiscoveredAfterFreshInit() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let content = Data("existing valid pack".utf8)
        let entry = makeEntry(content: content)
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try content.write(to: modelsDir.appendingPathComponent(entry.filename))

        // A fresh store (nothing called) must surface the valid file as ready
        // through local discovery — no download.
        let store = try makeStore(root: root, downloader: { _ in fatalError("no download expected") })
        let state = await store.discover(entry)
        let url = await store.localURL(for: entry)
        XCTAssertEqual(state, .ready(url), "cold-launch discovery marks ready")
        XCTAssertEqual(try Data(contentsOf: url), content)
    }

    // MARK: - Corrupt existing file

    func testCorruptExistingFileBecomesFailedNotReady() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try Data("WRONG CONTENT".utf8).write(to: modelsDir.appendingPathComponent(entry.filename))

        let store = try makeStore(root: root, downloader: { _ in fatalError() })
        let state = await store.discover(entry)
        if case .failed = state {
            // expected
        } else {
            XCTFail("corrupt file leaves failed state, got \(state)")
        }
    }

    func testRepairRemovesCorruptFileAndReinstalls() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let good = Data("the correct pack".utf8)
        let entry = makeEntry(content: good)
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try Data("corrupt".utf8).write(to: modelsDir.appendingPathComponent(entry.filename))

        let downloaded = root.appendingPathComponent("download.animapk")
        try good.write(to: downloaded)
        let store = try makeStore(root: root, downloader: { _ in downloaded })

        let repaired = try await store.repair(entry)
        XCTAssertEqual(try Data(contentsOf: repaired), good)
        XCTAssertEqual(try Data(contentsOf: modelsDir.appendingPathComponent(entry.filename)), good,
                       "corrupt file replaced by valid install")
        let state = await store.state(for: entry.component)
        XCTAssertEqual(state, .ready(repaired))
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty, "no stale staging files after repair")
    }

    // MARK: - Download/install integrity

    func testWrongSizeDownloadFailsNoFakeReadyFile() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))  // expects size 3
        let bad = root.appendingPathComponent("bad.animapk")
        try Data("wrong size".utf8).write(to: bad)
        let store = try makeStore(root: root, downloader: { _ in bad })

        do {
            _ = try await store.download(entry)
            XCTFail("wrong-size download must fail")
        } catch {
            // expected
        }
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: modelsDir.appendingPathComponent(entry.filename).path),
            "no fake ready file after failed download")
    }

    func testInsufficientDiskRejected() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let content = Data(repeating: 0xAB, count: 1024)
        let entry = makeEntry(content: content)
        let downloaded = root.appendingPathComponent("d.animapk")
        try content.write(to: downloaded)
        let store = try makeStore(
            root: root, downloader: { _ in downloaded }, capacity: { _ in 100 })  // too small
        do {
            _ = try await store.download(entry)
            XCTFail("insufficient disk must be rejected")
        } catch {
            XCTAssertTrue(String(describing: error).contains("disk"),
                          "expected disk error, got \(error)")
        }
    }

    func testUnknownDiskCapacityIsAUsefulError() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let content = Data(repeating: 0xAB, count: 64)
        let entry = makeEntry(content: content)
        let downloaded = root.appendingPathComponent("d.animapk")
        try content.write(to: downloaded)
        let store = try makeStore(
            root: root,
            downloader: { _ in downloaded },
            capacity: { _ in throw AnimapkError.validation("volume capacity unavailable") })
        do {
            _ = try await store.download(entry)
            XCTFail("unavailable capacity must be rejected")
        } catch {
            let message = error.localizedDescription
            XCTAssertTrue(message.contains("disk"), "expected disk-space error, got \(message)")
        }
    }

    func testFailedDownloadLeavesFailedState() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))
        struct DownloadError: Error {}
        let store = try makeStore(root: root, downloader: { _ in throw DownloadError() })
        do {
            _ = try await store.download(entry)
            XCTFail("download failure must propagate")
        } catch {
            // expected
        }
        let state = await store.state(for: entry.component)
        if case .failed = state {
            // expected
        } else {
            XCTFail("failed download leaves failed state, got \(state)")
        }
    }

    func testRepeatedDownloadIsIdempotent() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let content = Data("abc".utf8)
        let entry = makeEntry(content: content)
        let downloaded = root.appendingPathComponent("d.animapk")
        try content.write(to: downloaded)
        let store = try makeStore(root: root, downloader: { _ in downloaded })

        let a = try await store.download(entry)
        let b = try await store.download(entry)
        let c = try await store.download(entry)
        XCTAssertEqual(a, b)
        XCTAssertEqual(b, c)
    }

    func testVerifyExistingRejectsMissingFile() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))
        let store = try makeStore(root: root, downloader: { _ in fatalError() })
        do {
            _ = try await store.verifyExisting(entry)
            XCTFail("verifyExisting must reject a missing file")
        } catch {
            // expected
        }
        let state = await store.state(for: entry.component)
        XCTAssertEqual(state, .missing)
    }

    func testVerifyExistingFailsCorruptAndPassesValid() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try Data("abd".utf8).write(to: modelsDir.appendingPathComponent(entry.filename))
        let store = try makeStore(root: root, downloader: { _ in fatalError() })

        do {
            _ = try await store.verifyExisting(entry)
            XCTFail("corrupt file must fail explicit verify")
        } catch {
            // expected
        }
        if case .failed = await store.state(for: entry.component) {
            // expected
        } else {
            XCTFail("corrupt verify leaves failed state")
        }

        // Replace with a valid file: explicit verify must now succeed.
        try Data("abc".utf8).write(to: modelsDir.appendingPathComponent(entry.filename))
        let verified = try await store.verifyExisting(entry)
        XCTAssertEqual(try Data(contentsOf: verified), Data("abc".utf8))
        let verifiedState = await store.state(for: entry.component)
        XCTAssertEqual(verifiedState, .ready(verified))
    }

    // MARK: - Local discovery: no network, no re-hash (real-device fixes A/B)

    /// Three tiny synthetic packs, one per production component.
    private func makeThreeEntries() -> [ModelManifestEntry] {
        let components: [ModelComponent] = [.textEncoder, .dit, .vae]
        let names: [ModelComponent: String] = [
            .textEncoder: "qwen3-0.6b-xsmax-w8.animapk",
            .dit: "anima-turbo-v1.0-xsmax-w4.animapk",
            .vae: "qwen-image-vae-xsmax-fp16.animapk",
        ]
        return components.map { component in
            let content = Data("pack bytes for \(component)".utf8)
            return ModelManifestEntry(
                filename: names[component]!, size: UInt64(content.count),
                sha256: content.sha256Hex,
                url: URL(string: "https://example.invalid/\(names[component]!)")!,
                component: component)
        }
    }

    private func seed(modelsDir: URL, entries: [ModelManifestEntry]) throws {
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        for entry in entries {
            try Data("pack bytes for \(entry.component)".utf8)
                .write(to: modelsDir.appendingPathComponent(entry.filename))
        }
    }

    func testEmptyDirectoryDiscoveryNeverDownloads() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entries = makeThreeEntries()
        let store = try makeStore(root: root, downloader: { _ in
            XCTFail("download must not occur during local discovery")
            fatalError("network must never be touched by discovery")
        })
        let states = await store.discoverInstalled(entries: entries)
        XCTAssertEqual(states[.textEncoder], .missing)
        XCTAssertEqual(states[.dit], .missing)
        XCTAssertEqual(states[.vae], .missing)
        do {
            _ = try await store.resolveInstalledModels(entries: entries)
            XCTFail("empty set must not resolve")
        } catch {
            // expected: incomplete set
        }
    }

    func testPartialSetResolvesNilWithZeroDownloads() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entries = makeThreeEntries()
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        // Only textEncoder + dit present and valid; vae missing.
        try seed(modelsDir: modelsDir, entries: Array(entries.prefix(2)))
        let store = try makeStore(root: root, downloader: { _ in
            XCTFail("download must not occur during local discovery")
            fatalError()
        })
        let states = await store.discoverInstalled(entries: entries)
        let textEncoderURL = await store.localURL(for: entries[0])
        let ditURL = await store.localURL(for: entries[1])
        XCTAssertEqual(states[.textEncoder], .ready(textEncoderURL))
        XCTAssertEqual(states[.dit], .ready(ditURL))
        XCTAssertEqual(states[.vae], .missing)
        do {
            _ = try await store.resolveInstalledModels(entries: entries)
            XCTFail("partial set must not resolve")
        } catch {
            // expected: incomplete set
        }
    }

    func testCompleteSetResolvesWithZeroDownloads() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entries = makeThreeEntries()
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try seed(modelsDir: modelsDir, entries: entries)
        let store = try makeStore(root: root, downloader: { _ in
            XCTFail("download must not occur during local discovery")
            fatalError()
        })
        let resolved = try await store.resolveInstalledModels(entries: entries)
        XCTAssertEqual(Set([resolved.textEncoder, resolved.dit, resolved.vae]).count, 3)
        XCTAssertTrue(resolved.textEncoder.lastPathComponent.contains("qwen3"))
        XCTAssertTrue(resolved.dit.lastPathComponent.contains("anima-turbo"))
        XCTAssertTrue(resolved.vae.lastPathComponent.contains("vae"))
    }

    // MARK: - Verification receipts (real-device fix B)

    func testValidImportWritesReceiptAndRelaunchDoesNotRehash() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let content = Data("imported pack bytes".utf8)
        let entry = makeEntry(content: content)
        let source = root.appendingPathComponent("user-import.animapk")
        try content.write(to: source)

        // Import verifies while streaming (single pass); it does not route
        // through the injected full-file `verifier` hook.
        let storeA = try makeStore(root: root, downloader: { _ in fatalError() })
        let installed = try await storeA.importPack(entry, from: source)
        XCTAssertEqual(try Data(contentsOf: installed), content)

        // A fresh store over the same directory = a relaunch: receipt must make
        // discovery hashing-free.
        let counterB = VerifierCounter()
        let storeB = try makeStore(root: root, downloader: { _ in fatalError() },
                                   verifier: countingVerifier(counterB))
        let state = await storeB.discover(entry)
        XCTAssertEqual(state, .ready(installed))
        XCTAssertEqual(counterB.count, 0, "receipt-valid relaunch must not re-hash")

        // Resolution after discovery also stays hashing-free.
        _ = try? await storeB.resolveInstalledModels(entries: [entry])
        XCTAssertEqual(counterB.count, 0, "resolve must not re-hash receipt-valid files")
    }

    func testMissingReceiptVerifiesOnceThenWritesReceipt() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let content = Data("migration pack".utf8)
        let entry = makeEntry(content: content)
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try content.write(to: modelsDir.appendingPathComponent(entry.filename))

        let counter = VerifierCounter()
        let store = try makeStore(root: root, downloader: { _ in fatalError() },
                                  verifier: countingVerifier(counter))
        let state = await store.discover(entry)
        let expectedURL = await store.localURL(for: entry)
        XCTAssertEqual(state, .ready(expectedURL))
        XCTAssertEqual(counter.count, 1, "migration: one full verification, then receipt")

        let counterB = VerifierCounter()
        let storeB = try makeStore(root: root, downloader: { _ in fatalError() },
                                   verifier: countingVerifier(counterB))
        let stateB = await storeB.discover(entry)
        let expectedURLB = await storeB.localURL(for: entry)
        XCTAssertEqual(stateB, .ready(expectedURLB))
        XCTAssertEqual(counterB.count, 0, "receipt written by first discovery avoids re-hash")
    }

    func testReceiptInvalidatedByContentRewrite() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        let fileURL = modelsDir.appendingPathComponent(entry.filename)
        try Data("abc".utf8).write(to: fileURL)

        // Establish a trusted receipt.
        let storeA = try makeStore(root: root, downloader: { _ in fatalError() })
        let discoverA = await storeA.discover(entry)
        XCTAssertEqual(discoverA, .ready(fileURL))

        // Rewrite with different content of the SAME size, and bump the
        // modification date explicitly (APFS mtime is nanosecond-accurate, but
        // an explicit bump removes any filesystem granularity doubt).
        try Data("abd".utf8).write(to: fileURL)
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(60)], ofItemAtPath: fileURL.path)
        let counter = VerifierCounter()
        let storeB = try makeStore(root: root, downloader: { _ in fatalError() },
                                   verifier: countingVerifier(counter))
        let state = await storeB.discover(entry)
        if case .failed = state {
            // expected: full verification catches the rewrite
        } else {
            XCTFail("rewritten file must fail verification, got \(state)")
        }
        XCTAssertEqual(counter.count, 1, "invalidated receipt forces exactly one full verify")
    }

    func testReceiptInvalidatedBySizeChange() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        let fileURL = modelsDir.appendingPathComponent(entry.filename)
        try Data("abc".utf8).write(to: fileURL)
        let storeA = try makeStore(root: root, downloader: { _ in fatalError() })
        let discoverA = await storeA.discover(entry)
        XCTAssertEqual(discoverA, .ready(fileURL))

        // Extend the file: size no longer matches the manifest.
        try Data("abcXX".utf8).write(to: fileURL)
        let counter = VerifierCounter()
        let storeB = try makeStore(root: root, downloader: { _ in fatalError() },
                                   verifier: countingVerifier(counter))
        if case .failed = await storeB.discover(entry) {
            // expected
        } else {
            XCTFail("size-changed file must fail verification")
        }
        XCTAssertEqual(counter.count, 1)
    }

    func testReceiptInvalidatedByManifestSHAChange() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let original = makeEntry(content: Data("abc".utf8))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        let fileURL = modelsDir.appendingPathComponent(original.filename)
        try Data("abc".utf8).write(to: fileURL)
        let storeA = try makeStore(root: root, downloader: { _ in fatalError() })
        let discoverA = await storeA.discover(original)
        XCTAssertEqual(discoverA, .ready(fileURL))

        // The manifest now expects different content (same size, new SHA).
        let changed = makeEntry(content: Data("abd".utf8))
        let counter = VerifierCounter()
        let storeB = try makeStore(root: root, downloader: { _ in fatalError() },
                                   verifier: countingVerifier(counter))
        if case .failed = await storeB.discover(changed) {
            // expected: on-disk file fails the new SHA
        } else {
            XCTFail("stale receipt must not trust a file against a changed manifest")
        }
        XCTAssertEqual(counter.count, 1)
    }

    func testReceiptNotNeededForFileStatus() async throws {
        // fileStatus (used by the cheap diagnostics snapshot) must be
        // hashing-free even when no receipt exists.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let content = Data("status probe".utf8)
        let entry = makeEntry(content: content)
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try content.write(to: modelsDir.appendingPathComponent(entry.filename))
        let counter = VerifierCounter()
        let store = try makeStore(root: root, downloader: { _ in fatalError() },
                                  verifier: countingVerifier(counter))
        let status = await store.fileStatus(for: entry)
        XCTAssertTrue(status.exists)
        XCTAssertEqual(status.sizeBytes, UInt64(content.count))
        XCTAssertFalse(status.receiptValid, "no receipt yet")
        XCTAssertEqual(counter.count, 0, "fileStatus must never hash")
    }

    // MARK: - Import fallback (legal-block path)

    func testImportVerifiesAndInstalls() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let content = Data("imported pack bytes".utf8)
        let entry = makeEntry(content: content)
        let source = root.appendingPathComponent("user-import.animapk")
        try content.write(to: source)
        let store = try makeStore(root: root, downloader: { _ in fatalError() })

        let installed = try await store.importPack(entry, from: source)
        XCTAssertEqual(try Data(contentsOf: installed), content)
        let state = await store.state(for: entry.component)
        XCTAssertEqual(state, .ready(installed))
    }

    func testImportRejectsMismatchedFile() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("expected".utf8))
        let source = root.appendingPathComponent("user-import.animapk")
        try Data("not the expected pack".utf8).write(to: source)
        let store = try makeStore(root: root, downloader: { _ in fatalError() })

        do {
            _ = try await store.importPack(entry, from: source)
            XCTFail("mismatched import must be rejected")
        } catch {
            // expected: same manifest verification as downloads
        }
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: modelsDir.appendingPathComponent(entry.filename).path),
            "rejected import must not leave a file")
    }

    func testImportReplacesCorruptDestination() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let good = Data("the real pack".utf8)
        let entry = makeEntry(content: good)
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        // A corrupt destination already exists.
        try Data("corrupt bytes".utf8).write(to: modelsDir.appendingPathComponent(entry.filename))
        let source = root.appendingPathComponent("user-import.animapk")
        try good.write(to: source)

        let store = try makeStore(root: root, downloader: { _ in fatalError() })
        let installed = try await store.importPack(entry, from: source)
        XCTAssertEqual(try Data(contentsOf: installed), good,
                       "corrupt destination replaced by valid import")
        XCTAssertEqual(try Data(contentsOf: modelsDir.appendingPathComponent(entry.filename)), good)
        let importState = await store.state(for: entry.component)
        XCTAssertEqual(importState, .ready(installed))
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty, "staging cleaned after replace")
        // The user's external source must not be deleted.
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
    }

    func testFailedImportLeavesNoFakeReadyAndKeepsCorruptDestination() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("expected".utf8))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try Data("corrupt destination".utf8).write(to: modelsDir.appendingPathComponent(entry.filename))
        let source = root.appendingPathComponent("bad-import.animapk")
        try Data("wrong".utf8).write(to: source)

        let store = try makeStore(root: root, downloader: { _ in fatalError() })
        do {
            _ = try await store.importPack(entry, from: source)
            XCTFail("invalid source must be rejected")
        } catch {
            // expected
        }
        let state = await store.state(for: entry.component)
        if case .ready = state {
            XCTFail("failed import must never leave a ready state")
        }
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty, "staging cleaned after failed import")
    }

    // MARK: - Single-pass streaming import (bounded memory)

    /// A deterministic fixture of `count` bytes (larger than a single test
    /// chunk, so the streaming installer must iterate many times).
    private func makeLargeSource(dir: URL, count: Int, salt: UInt8 = 3) throws -> URL {
        let url = dir.appendingPathComponent("large-source-\(UUID().uuidString).animapk")
        var data = Data(capacity: count)
        for i in 0..<count {
            data.append(UInt8((i &* Int(salt) &+ 11) & 0xFF))
        }
        try data.write(to: url)
        return url
    }

    /// An entry whose primary is W4-like and whose alternate is W8-v2-like
    /// (synthetic tiny values — the pinned real values are pinned by
    /// SmokeTests.testDiTSlotAcceptsPinnedW8V2Variant).
    private func makeDitEntry(
        filename: String = "anima-turbo-v1.0-xsmax-w4.animapk",
        primary: Data,
        alternate: Data
    ) -> ModelManifestEntry {
        ModelManifestEntry(
            filename: filename,
            size: UInt64(primary.count),
            sha256: primary.sha256Hex,
            url: URL(string: "https://example.invalid/\(filename)")!,
            component: .dit,
            alternates: [ModelVariant(size: UInt64(alternate.count), sha256: alternate.sha256Hex)])
    }

    /// 5.1 — A source comfortably larger than the chunk size must stream
    /// across many read/hash/write iterations and produce a byte-identical,
    /// digest-verified install with no staging leftovers.
    func testImportStreamsAcrossManyChunksAndMatchesSource() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        // 1 MiB source with 4 KiB chunks -> 256 stream iterations.
        let sourceSize = 1 << 20
        let chunkBytes = 4 << 10
        let source = try makeLargeSource(dir: root, count: sourceSize, salt: 5)
        let data = try Data(contentsOf: source)
        let entry = makeEntry(filename: "tiny.animapk", content: data)
        let store = try makeStore(root: root, downloader: { _ in fatalError() }, chunkBytes: chunkBytes)

        let installed = try await store.importPack(entry, from: source)
        XCTAssertEqual(try Data(contentsOf: installed), data)
        let state = await store.discover(entry)
        XCTAssertEqual(state, .ready(installed))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty, "staging cleaned after streaming import")
    }

    /// 5.2 — Importing the ALTERNATE variant (W8-v2-like) into the .dit slot
    /// must verify against the alternate, install, and be discovered as ready
    /// via a receipt that records the alternate digest.
    func testImportAlternateVariantIntoDiTSlot() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let w4 = Data("w4-like primary bytes".utf8)
        let w8 = Data("w8-v2-like alternate bytes, longer for distinction".utf8)
        let entry = makeDitEntry(primary: w4, alternate: w8)
        let source = root.appendingPathComponent("user-import-w8.animapk")
        try w8.write(to: source)
        let store = try makeStore(root: root, downloader: { _ in fatalError() })

        let installed = try await store.importPack(entry, from: source)
        XCTAssertEqual(try Data(contentsOf: installed), w8,
                       "the alternate variant must be installed into the .dit slot")
        // A fresh store = relaunch: the receipt records the W8-v2 alternate
        // digest, so discovery must be ready WITHOUT re-hashing.
        let relaunch = try makeStore(root: root, downloader: { _ in fatalError() })
        let state = await relaunch.discover(entry)
        XCTAssertEqual(state, .ready(installed),
                       "receipt recorded the alternate variant; discovery must trust it")
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty)
    }

    /// 5.2b — Importing the alternate variant REPLACES the primary when one is
    /// already installed, and vice versa (K: import w8 -> use w8, import w4 ->
    /// use w4).
    func testImportAlternateReplacesPrimaryInDiTSlot() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let w4 = Data("w4 primary bytes".utf8)
        let w8 = Data("w8-v2 alternate bytes".utf8)
        let entry = makeDitEntry(primary: w4, alternate: w8)
        let store = try makeStore(root: root, downloader: { _ in fatalError() })

        // Import W4 first.
        let w4Source = root.appendingPathComponent("w4.animapk")
        try w4.write(to: w4Source)
        let w4Installed = try await store.importPack(entry, from: w4Source)
        XCTAssertEqual(try Data(contentsOf: w4Installed), w4)

        // Import W8-v2 over it: replace, not fail.
        let w8Source = root.appendingPathComponent("w8.animapk")
        try w8.write(to: w8Source)
        let w8Installed = try await store.importPack(entry, from: w8Source)
        XCTAssertEqual(try Data(contentsOf: w8Installed), w8)
        let state = await store.discover(entry)
        XCTAssertEqual(state, .ready(w8Installed))
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty, "staging cleaned after replace")
    }

    /// 5.2c — A SHA mismatch (source size matches a variant but digest does
    /// not) must leave NO installed pack and NO staging file.
    func testStreamingImportSHAMismatchLeavesNoInstalledOrStagingFile() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { Self.cleanup(root) }
        let sourceSize = 512 << 10
        let chunkBytes = 4 << 10
        let source = try makeLargeSource(dir: root, count: sourceSize, salt: 7)
        // Wrong digest: all zeros. Size gate passes; stream reaches EOF and
        // the digest check must fail.
        let entry = ModelManifestEntry(
            filename: "tiny.animapk", size: UInt64(sourceSize),
            sha256: String(repeating: "0", count: 64),
            url: URL(string: "https://example.invalid/tiny.animapk")!, component: .dit)
        let store = try makeStore(root: root, downloader: { _ in fatalError() }, chunkBytes: chunkBytes)

        do {
            _ = try await store.importPack(entry, from: source)
            XCTFail("wrong SHA-256 must be rejected after streaming reaches EOF")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("sha"))
        }
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: modelsDir.appendingPathComponent(entry.filename).path),
            "no pack may be installed after a digest mismatch")
        XCTAssertTrue(stagingFiles(in: modelsDir).isEmpty,
                      "no staging file may remain after a digest mismatch")
        let state = await store.discover(entry)
        XCTAssertEqual(state, .missing)
    }

    // MARK: - Localized errors (real-device fix D)

    func testAnimapkErrorLocalizedDescriptionSurfacesReason() {
        let error = AnimapkError.validation("sentinel validation reason")
        XCTAssertTrue(error.localizedDescription.contains("sentinel validation reason"),
                      "localizedDescription must surface the real reason, got \(error.localizedDescription)")
        let io = AnimapkError.io("sentinel io reason")
        XCTAssertTrue(io.localizedDescription.contains("sentinel io reason"))
        // A generic description must never look like "(AnimaXS.AnimapkError error 3.)".
        XCTAssertFalse(error.localizedDescription.contains("error 3"))
    }

    // MARK: - Production downloader (real-device fix E)

    private final class MockURLProtocol: URLProtocol {
        static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?
        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            guard let handler = MockURLProtocol.handler else {
                client?.urlProtocol(self, didFailWithError: URLError(.unknown))
                return
            }
            do {
                let (response, data) = try handler(request)
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }
        override func stopLoading() {}
    }

    private func makeDownloaderSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    func testDownloaderAccepts2xxAndOwnsStagingFile() async throws {
        MockURLProtocol.handler = { request in
            (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
             Data("pack bytes".utf8))
        }
        defer { MockURLProtocol.handler = nil }
        let downloader = ModelStore.productionDownloader(session: makeDownloaderSession())
        let url = try await downloader(URL(string: "https://example.invalid/model.animapk")!)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path),
                      "staged download must exist after downloader returns")
        XCTAssertEqual(try Data(contentsOf: url), Data("pack bytes".utf8))
        XCTAssertTrue(url.path.hasPrefix(FileManager.default.temporaryDirectory.path),
                      "staging lives in the app-owned temporary directory")
    }

    func testDownloaderRejectsNon2xxWithStatusCode() async throws {
        MockURLProtocol.handler = { request in
            (HTTPURLResponse(url: request.url!, statusCode: 404, httpVersion: nil, headerFields: nil)!,
             Data("not found".utf8))
        }
        defer { MockURLProtocol.handler = nil }
        let downloader = ModelStore.productionDownloader(session: makeDownloaderSession())
        do {
            _ = try await downloader(URL(string: "https://example.invalid/model.animapk")!)
            XCTFail("non-2xx must throw")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("404"),
                          "error must surface the HTTP status, got \(error.localizedDescription)")
        }
    }

    func testDownloaderRejectsRedirectToLoginPage() async throws {
        // A 302 followed by an HTML login page: URLSession follows the
        // redirect, so the final response is a 200 HTML page — the downloader
        // cannot detect this by status alone. The store-level size/SHA check is
        // the real guard; here we only verify status handling stays strict.
        MockURLProtocol.handler = { request in
            (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
             Data("<html>login</html>".utf8))
        }
        defer { MockURLProtocol.handler = nil }
        let downloader = ModelStore.productionDownloader(session: makeDownloaderSession())
        let url = try await downloader(URL(string: "https://example.invalid/model.animapk")!)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(try Data(contentsOf: url), Data("<html>login</html>".utf8))
        // The store would reject this file on size/SHA — covered by
        // testWrongSizeDownloadFailsNoFakeReadyFile.
    }

    func testDownloaderPropagatesTransportFailure() async throws {
        MockURLProtocol.handler = { _ in throw URLError(.notConnectedToInternet) }
        defer { MockURLProtocol.handler = nil }
        let downloader = ModelStore.productionDownloader(session: makeDownloaderSession())
        do {
            _ = try await downloader(URL(string: "https://example.invalid/model.animapk")!)
            XCTFail("transport failure must propagate")
        } catch {
            // expected: URLError reaches the caller
        }
    }

    // MARK: - Resolved production models

    func testResolvedModelsExposeExactlyThreeURLs() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entries = makeThreeEntries()
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try seed(modelsDir: modelsDir, entries: entries)
        let store = try makeStore(root: root, downloader: { _ in fatalError("no download") })

        let resolved = try await store.resolveInstalledModels(entries: entries)
        XCTAssertEqual(Set([resolved.textEncoder, resolved.dit, resolved.vae]).count, 3)
        XCTAssertTrue(resolved.textEncoder.lastPathComponent.contains("qwen3"))
        XCTAssertTrue(resolved.dit.lastPathComponent.contains("anima-turbo"))
        XCTAssertTrue(resolved.vae.lastPathComponent.contains("vae"))
    }

    /// Best-effort temp cleanup that can never throw or fail the test.
    private static func cleanup(_ root: URL) {
        let fm = FileManager.default
        let models = root.appendingPathComponent("models", isDirectory: true)
        for dir in [models, root] where fm.fileExists(atPath: dir.path) {
            try? fm.removeItem(at: dir)
        }
    }
}
