import XCTest
@testable import AnimaXS

/// ModelStore production-integration tests (K003/K004/guide §11).
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

    /// Fresh temporary store with an injected downloader/capacity. `secureInstalls`
    /// is disabled so test files remain removable (complete-until-first-auth
    /// protection would make the temp dir undeletable in the simulator).
    private func makeStore(
        root: URL,
        downloader: @escaping ModelStore.Downloader,
        capacity: @escaping ModelStore.CapacityProvider = { _ in Int64.max }
    ) throws -> ModelStore {
        try ModelStore(
            directory: root.appendingPathComponent("models", isDirectory: true),
            downloader: downloader, availableCapacity: capacity, secureInstalls: false)
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

        // A fresh store (nothing called) must surface the valid file as ready.
        let store = try makeStore(root: root, downloader: { _ in fatalError("no download expected") })
        let state = await store.state(for: entry.component)
        XCTAssertEqual(state, .missing, "state resolves lazily on prepare")
        let url = try await store.prepare(entry)
        XCTAssertEqual(try Data(contentsOf: url), content)
        let after = await store.state(for: entry.component)
        XCTAssertEqual(after, .ready(url), "cold-launch discovery marks ready")
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
        do {
            _ = try await store.prepare(entry)
            XCTFail("corrupt file must not be treated as ready")
        } catch {
            // expected
        }
        let state = await store.state(for: entry.component)
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
    }

    // MARK: - Download/install integrity

    func testWrongSizeDownloadFailsNoFakeReadyFile() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let entry = makeEntry(content: Data("abc".utf8))  // expects size 3
        let bad = root.appendingPathComponent("bad.animapk")
        try Data("wrong size".utf8).write(to: bad)
        let store = try makeStore(root: root, downloader: { _ in bad })

        do {
            _ = try await store.prepare(entry)
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
        defer { Self.cleanup(root) }
        let content = Data(repeating: 0xAB, count: 1024)
        let entry = makeEntry(content: content)
        let downloaded = root.appendingPathComponent("d.animapk")
        try content.write(to: downloaded)
        let store = try makeStore(
            root: root, downloader: { _ in downloaded }, capacity: { _ in 100 })  // too small
        do {
            _ = try await store.prepare(entry)
            XCTFail("insufficient disk must be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("disk"),
                          "expected disk error, got \(error)")
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
            _ = try await store.prepare(entry)
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

    func testRepeatedPrepareIsIdempotent() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        let content = Data("abc".utf8)
        let entry = makeEntry(content: content)
        let downloaded = root.appendingPathComponent("d.animapk")
        try content.write(to: downloaded)
        let store = try makeStore(root: root, downloader: { _ in downloaded })

        let a = try await store.prepare(entry)
        let b = try await store.prepare(entry)
        let c = try await store.prepare(entry)
        XCTAssertEqual(a, b)
        XCTAssertEqual(b, c)
    }

    // MARK: - Resolved production models

    func testResolvedModelsExposeExactlyThreeURLs() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        defer { Self.cleanup(root) }
        // Three tiny synthetic packs, one per production component. The DiT
        // pack serves the adapter and sampler (ResolvedModels has no adapter
        // field), so the resolved set is exactly three URLs.
        let components: [ModelComponent] = [.dit, .textEncoder, .vae]
        let names: [ModelComponent: String] = [
            .dit: "anima-turbo-v1.0-xsmax-w4.animapk",
            .textEncoder: "qwen3-0.6b-xsmax-w8.animapk",
            .vae: "qwen-image-vae-xsmax-fp16.animapk",
        ]
        let entries = components.map { component -> ModelManifestEntry in
            let content = Data("pack bytes for \(component)".utf8)
            return ModelManifestEntry(
                filename: names[component]!, size: UInt64(content.count),
                sha256: content.sha256Hex,
                url: URL(string: "https://example.invalid/\(names[component]!)")!,
                component: component)
        }
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        // Pre-seed valid files so resolution discovers them (no download).
        for entry in entries {
            let content = Data("pack bytes for \(entry.component)".utf8)
            try content.write(to: modelsDir.appendingPathComponent(entry.filename))
        }
        let store = try makeStore(root: root, downloader: { _ in fatalError("no download") })

        let resolved = try await store.resolvedModels(entries: entries)
        XCTAssertEqual(Set([resolved.textEncoder, resolved.dit, resolved.vae]).count, 3)
        XCTAssertTrue(resolved.textEncoder.lastPathComponent.contains("qwen3"))
        XCTAssertTrue(resolved.dit.lastPathComponent.contains("anima-turbo"))
        XCTAssertTrue(resolved.vae.lastPathComponent.contains("vae"))
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
/// Best-effort temp cleanup that can never throw or fail the test.
    private static func cleanup(_ root: URL) {
        let fm = FileManager.default
        let models = root.appendingPathComponent("models", isDirectory: true)
        for dir in [models, root] where fm.fileExists(atPath: dir.path) {
            try? fm.removeItem(at: dir)
        }
    }
}
