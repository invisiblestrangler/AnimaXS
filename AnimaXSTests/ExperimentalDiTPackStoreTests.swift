import Dispatch
import XCTest
import CryptoKit
@testable import AnimaXS

/// Experimental W8 store tests (§18.8) using a tiny injected spec — the
/// 2.23 GB production pack is never touched in CI.
final class ExperimentalDiTPackStoreTests: XCTestCase {

    /// Tiny synthetic spec: a 256-byte fixture pack.
    private struct TinySpec: ExperimentalDiTPackSpec {
        let filename: String
        let byteCount: UInt64
        let sha256: String

        init(filename: String = "tiny-w8.animapk", byteCount: UInt64 = 256, sha256: String) {
            self.filename = filename
            self.byteCount = byteCount
            self.sha256 = sha256
        }
    }

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ExperimentalDiTStoreTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The pinned W8 v2 manifest values must match the HuggingFace LFS
    /// metadata exactly (verified 2026-08-15 via the tree API). This pins the
    /// constants so a digit-transcription error in the manifest (like the
    /// 2_232_973_560 vs 2_232_975_360 flip that rejected every import) is
    /// caught in CI without downloading the 2.23 GB pack.
    func testPinnedW8ManifestMatchesLFSMetadata() {
        XCTAssertEqual(ExperimentalDiTManifest.filename, "anima-turbo-v1.0-xsmax-w8-v2.animapk")
        XCTAssertEqual(ExperimentalDiTManifest.byteCount, 2_232_975_360,
                       "must match HuggingFace LFS size for the pinned revision")
        XCTAssertEqual(
            ExperimentalDiTManifest.sha256,
            "8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130")
        XCTAssertEqual(
            ExperimentalDiTManifest.revision,
            "589d028122f872e66ee20cdd12cb55eb3b816add")
    }

    private func sha256Hex(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A 256-byte deterministic fixture file.
    private func makeSource(dir: URL, salt: UInt8 = 1) throws -> URL {
        let url = dir.appendingPathComponent("source-\(UUID().uuidString).animapk")
        let data = Data((0..<256).map { UInt8(($0 &* Int(salt) &+ 7) & 0xFF) })
        try data.write(to: url)
        return url
    }

    /// A deterministic fixture of `count` bytes (larger than a single test
    /// chunk, so the streaming installer must iterate many times). The byte
    /// pattern is derived from a salt so each fixture is distinct.
    private func makeLargeSource(dir: URL, count: Int, salt: UInt8 = 3) throws -> URL {
        let url = dir.appendingPathComponent("large-source-\(UUID().uuidString).animapk")
        var data = Data(capacity: count)
        data.reserveCapacity(count)
        for i in 0..<count {
            data.append(UInt8((i &* Int(salt) &+ 11) & 0xFF))
        }
        try data.write(to: url)
        return url
    }

    private func makeSpec(for data: Data) -> TinySpec {
        TinySpec(sha256: sha256Hex(of: data))
    }

    private func makeStore(
        dir: URL, spec: TinySpec,
        capacity: Int64 = 10_000_000_000,
        secureInstalls: Bool = false,
        chunkBytes: Int = 1 << 20
    ) throws -> ExperimentalDiTPackStore {
        try ExperimentalDiTPackStore(
            directory: dir,
            spec: spec,
            availableCapacity: { _ in capacity },
            secureInstalls: secureInstalls,
            chunkBytes: chunkBytes)
    }

    func testMissingWhenNothingInstalled() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let data = Data((0..<256).map { UInt8($0 & 0xFF) })
        let store = try makeStore(dir: dir, spec: makeSpec(for: data))
        let state = await store.discover()
        XCTAssertEqual(state, .missing)
    }

    func testSuccessfulStagedImportAndReceipt() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        let store = try makeStore(dir: dir, spec: makeSpec(for: data))
        let url = try await store.importPack(from: source)
        XCTAssertEqual(url.lastPathComponent, "tiny-w8.animapk")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        // No staging leftovers.
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.contains("staging") }
        XCTAssertTrue(leftovers.isEmpty, "staging temp must be cleaned on success")
        let state = await store.discover()
        XCTAssertEqual(state, .ready(url))
    }

    func testWrongByteCountRejected() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        let spec = makeSpec(for: data)
        // Shrink the expected count so the real 256-byte file mismatches.
        let wrongSpec = TinySpec(byteCount: 255, sha256: spec.sha256)
        let store = try ExperimentalDiTPackStore(
            directory: dir, spec: wrongSpec,
            availableCapacity: { _ in 10_000_000_000 },
            secureInstalls: false)
        do {
            _ = try await store.importPack(from: source)
            XCTFail("wrong byte count must be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("size"))
        }
        // Nothing installed.
        let state = await store.discover()
        XCTAssertEqual(state, .missing)
    }

    func testWrongSHARejected() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        let spec = makeSpec(for: data)
        // Wrong digest: all zeros.
        let wrongSpec = TinySpec(sha256: String(repeating: "0", count: 64))
        let store = try ExperimentalDiTPackStore(
            directory: dir, spec: wrongSpec,
            availableCapacity: { _ in 10_000_000_000 },
            secureInstalls: false)
        do {
            _ = try await store.importPack(from: source)
            XCTFail("wrong SHA-256 must be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("sha"))
        }
        let state = await store.discover()
        XCTAssertEqual(state, .missing)
    }

    func testInsufficientDiskRejectedBeforeCopy() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        // Capacity below pack + 256 MB reserve.
        let store = try makeStore(dir: dir, spec: makeSpec(for: data), capacity: 100)
        do {
            _ = try await store.importPack(from: source)
            XCTFail("insufficient disk must be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("disk")
                          || error.localizedDescription.localizedCaseInsensitiveContains("space"))
        }
        let state = await store.discover()
        XCTAssertEqual(state, .missing)
    }

    func testChangedFileInvalidatesReceipt() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        let store = try makeStore(dir: dir, spec: makeSpec(for: data))
        let url = try await store.importPack(from: source)
        let state = await store.discover()
        XCTAssertEqual(state, .ready(url))

        // Corrupt the installed file (same size, different bytes) AND give it
        // a clearly different modification date: the cheap receipt check
        // compares size AND mod-date identity, so either change invalidates it.
        let corrupted = Data((0..<256).map { UInt8(($0 &* 3 &+ 1) & 0xFF) })
        try corrupted.write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 1_600_000_000)],
            ofItemAtPath: url.path)
        let stateAfterCorruption = await store.discover()
        XCTAssertEqual(stateAfterCorruption, .unverified)
    }

    func testRemoveDeletesPackAndReceipt() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        let store = try makeStore(dir: dir, spec: makeSpec(for: data))
        let url = try await store.importPack(from: source)
        let state = await store.discover()
        XCTAssertEqual(state, .ready(url))
        await store.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        let afterRemove = await store.discover()
        XCTAssertEqual(afterRemove, .missing)
    }

    func testReimportReplacesExistingPack() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source1 = try makeSource(dir: dir, salt: 1)
        let data1 = try Data(contentsOf: source1)
        let store1 = try makeStore(dir: dir, spec: makeSpec(for: data1))
        let first = try await store1.importPack(from: source1)
        let firstState = await store1.discover()
        XCTAssertEqual(firstState, .ready(first))

        // Re-import with the SAME spec from a second source: replace, not fail.
        let source2 = try makeSource(dir: dir, salt: 2)
        let data2 = try Data(contentsOf: source2)
        let spec2 = makeSpec(for: data2)
        let store2 = try ExperimentalDiTPackStore(
            directory: dir, spec: spec2,
            availableCapacity: { _ in 10_000_000_000 },
            secureInstalls: false)
        let second = try await store2.importPack(from: source2)
        let secondState = await store2.discover()
        XCTAssertEqual(secondState, .ready(second))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path))
        // The installed bytes must be exactly the new source (old pack replaced).
        XCTAssertEqual(try Data(contentsOf: second), data2)
        XCTAssertNotEqual(try Data(contentsOf: second), data1)
        // No staging leftovers after the replacement.
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.contains("staging") }
        XCTAssertTrue(leftovers.isEmpty, "staging temp must be cleaned on re-import")
    }

    // MARK: - Multi-chunk streaming import regressions

    /// 5.1 — A source comfortably larger than the chunk size must stream
    /// across many read/hash/write iterations and produce a byte-identical,
    /// digest-verified install with no staging leftovers.
    func testSuccessfulImportStreamsAcrossManyChunksAndMatchesSource() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // 1 MiB source with 4 KiB chunks -> 256 stream iterations.
        let sourceSize = 1 << 20
        let chunkBytes = 4 << 10
        let source = try makeLargeSource(dir: dir, count: sourceSize, salt: 5)
        let data = try Data(contentsOf: source)
        let spec = TinySpec(byteCount: UInt64(sourceSize), sha256: sha256Hex(of: data))
        let store = try makeStore(dir: dir, spec: spec, chunkBytes: chunkBytes)

        let url = try await store.importPack(from: source)
        let state = await store.discover()
        XCTAssertEqual(state, .ready(url))
        XCTAssertEqual(url.lastPathComponent, spec.filename)

        // Installed size and contents must match the source exactly.
        let installed = try Data(contentsOf: url)
        XCTAssertEqual(installed.count, sourceSize)
        XCTAssertEqual(installed, data)
        XCTAssertEqual(sha256Hex(of: installed), spec.sha256)

        // No staging leftovers.
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.contains("staging") }
        XCTAssertTrue(leftovers.isEmpty, "staging temp must be cleaned on success")
    }

    /// 5.2 — A byte-count-correct source with a wrong expected SHA streams to
    /// EOF, fails the digest check, and leaves NO installed pack and NO
    /// staging file behind.
    func testStreamingImportSHAMismatchLeavesNoInstalledOrStagingFile() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let sourceSize = 512 << 10
        let chunkBytes = 4 << 10
        let source = try makeLargeSource(dir: dir, count: sourceSize, salt: 7)
        let wrongSpec = TinySpec(
            byteCount: UInt64(sourceSize),
            sha256: String(repeating: "0", count: 64))
        let store = try makeStore(dir: dir, spec: wrongSpec, chunkBytes: chunkBytes)

        do {
            _ = try await store.importPack(from: source)
            XCTFail("wrong SHA-256 must be rejected after streaming reaches EOF")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("sha"))
        }

        // No final pack, no staging file, and discovery reports missing.
        let contents = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertFalse(contents.contains { $0 == wrongSpec.filename },
                       "no pack may be installed after a digest mismatch")
        XCTAssertTrue(contents.filter { $0.contains("staging") }.isEmpty,
                      "no staging file may remain after a digest mismatch")
        let state = await store.discover()
        XCTAssertEqual(state, .missing)
    }

    /// 5.3 — Byte-count mismatch is rejected cheaply from the initial stat
    /// BEFORE any streaming/staging occurs (covered by
    /// `testWrongByteCountRejected`); this variant additionally pins that the
    /// stat-based size gate creates no staging file at all when it fires.
    func testStreamingImportSizeGateCreatesNoStaging() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        // Declare an expected count 16 bytes larger than the real file so the
        // initial stat rejects before any staging exists.
        let tooLargeSpec = TinySpec(
            byteCount: UInt64(data.count) + 16, sha256: sha256Hex(of: data))
        let store = try makeStore(dir: dir, spec: tooLargeSpec, chunkBytes: 4 << 10)
        do {
            _ = try await store.importPack(from: source)
            XCTFail("size mismatch must be rejected before streaming")
        } catch {
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("size"))
        }
        let contents = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertTrue(contents.filter { $0.contains("staging") }.isEmpty,
                      "no staging file may be created when size gate fails")
        let state = await store.discover()
        XCTAssertEqual(state, .missing)
    }

    // MARK: - Catalog in-progress state

    /// 5.7 — The catalog publishes `.verifying` while the store import is
    /// in-flight (before completion/failure), so SwiftUI can hide the Import
    /// button for the whole multi-gigabyte operation. Uses the store's
    /// capacity provider as a deterministic in-flight gate — no mocking
    /// framework.
    @MainActor
    func testCatalogPublishesVerifyingWhileImportInFlight() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let source = try makeSource(dir: dir)
        let data = try Data(contentsOf: source)
        let spec = makeSpec(for: data)

        // The capacity provider blocks until released, pinning the import
        // in-flight so we can observe the intermediate `.verifying` state.
        let release = DispatchSemaphore(value: 0)
        let store = try ExperimentalDiTPackStore(
            directory: dir, spec: spec,
            availableCapacity: { _ in
                release.wait()
                return 10_000_000_000
            },
            secureInstalls: false)
        let catalog = ExperimentalDiTPackCatalog(store: store)

        let task = Task { await catalog.importPack(from: source) }
        // Always release the capacity gate (idempotent) so a failed assertion
        // can never leave the import blocked and deadlock the test run.
        defer { release.signal() }
        // Yield until the import task has run far enough to publish `.verifying`
        // (it then suspends inside the store's blocked capacity provider).
        var attempts = 0
        while catalog.state != .verifying && attempts < 1_000 {
            await Task.yield()
            attempts += 1
        }
        XCTAssertEqual(catalog.state, .verifying,
                       "catalog must publish in-progress state before awaiting the store")
        XCTAssertEqual(catalog.message, "Importing and verifying W8 v2…")

        // Unblock the store's capacity gate BEFORE awaiting completion.
        release.signal()
        await task.value
        if case .ready = catalog.state {
            XCTAssertEqual(catalog.message, "Imported experimental W8 v2 (verified size + SHA-256).")
        } else {
            XCTFail("catalog must settle to .ready after a successful import, got \(catalog.state)")
        }
    }
}
