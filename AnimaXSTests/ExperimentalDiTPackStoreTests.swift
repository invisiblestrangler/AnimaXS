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

    private func makeSpec(for data: Data) -> TinySpec {
        TinySpec(sha256: sha256Hex(of: data))
    }

    private func makeStore(
        dir: URL, spec: TinySpec,
        capacity: Int64 = 10_000_000_000,
        secureInstalls: Bool = false
    ) throws -> ExperimentalDiTPackStore {
        try ExperimentalDiTPackStore(
            directory: dir,
            spec: spec,
            verifier: { try ModelManifest.sha256(of: $0) },
            availableCapacity: { _ in capacity },
            secureInstalls: secureInstalls)
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
            verifier: { try ModelManifest.sha256(of: $0) },
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
            verifier: { try ModelManifest.sha256(of: $0) },
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
            verifier: { try ModelManifest.sha256(of: $0) },
            availableCapacity: { _ in 10_000_000_000 },
            secureInstalls: false)
        let second = try await store2.importPack(from: source2)
        let secondState = await store2.discover()
        XCTAssertEqual(secondState, .ready(second))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path))
    }
}
