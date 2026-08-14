import XCTest
@testable import AnimaXS

/// K005 diagnostics tests for the redesigned engine:
/// - opening diagnostics (snapshot) runs zero heavy work and zero hashing;
/// - basic self-tests are pack-free and compute-free;
/// - hardware tests run each Metal probe exactly once (or skip cleanly);
/// - deep integrity is explicit and hashes each pack exactly once;
/// - JSON serialization reuses an existing report and runs zero tests;
/// - Codable round trip and status representation stay stable.
final class DiagnosticsTests: XCTestCase {

    private final class VerifierCounter {
        private var _count = 0
        func increment() { _count += 1 }
        var count: Int { _count }
    }

    /// A store over a temp dir whose verifier counts full hashes.
    private func makeCountingStore(
        root: URL,
        counter: VerifierCounter
    ) throws -> ModelStore {
        try ModelStore(
            directory: root.appendingPathComponent("models", isDirectory: true),
            downloader: { _ in fatalError("no network in diagnostics tests") },
            secureInstalls: false,
            verifier: { url, entry in
                counter.increment()
                try ModelManifest.verify(url, against: entry)
            },
            receiptsDirectory: root.appendingPathComponent("Receipts", isDirectory: true))
    }

    private func makeEntry(
        filename: String = "tiny.animapk",
        content: Data = Data("abc".utf8),
        component: ModelComponent = .dit
    ) -> ModelManifestEntry {
        ModelManifestEntry(
            filename: filename, size: UInt64(content.count), sha256: content.sha256Hex,
            url: URL(string: "https://example.invalid/\(filename)")!, component: component)
    }

    private func makeNoMetalEngine(storeProvider: (() -> ModelStore?)? = nil) -> DiagnosticsEngine {
        DiagnosticsEngine(
            context: nil, forceMetalUnavailable: true, storeProvider: storeProvider)
    }

    // MARK: - Snapshot is cheap

    func testSnapshotReportsStableRequiredFields() async throws {
        let engine = makeNoMetalEngine()
        let report = await engine.snapshot()
        XCTAssertFalse(report.appVersion.isEmpty)
        XCTAssertFalse(report.osVersion.isEmpty)
        XCTAssertFalse(report.deviceModel.isEmpty)
        XCTAssertGreaterThan(report.physicalMemoryBytes, 0)
        XCTAssertFalse(report.thermalState.isEmpty)
        XCTAssertEqual(report.modelPacks.count, 3)
        XCTAssertEqual(report.modelPacks.map(\.component),
                       [ModelComponent.dit, .textEncoder, .vae].map(\.rawValue),
                       "packs in manifest order")
        XCTAssertTrue(report.selfTests.isEmpty,
                      "opening diagnostics must not run any heavy test")
    }

    func testSnapshotDoesNotHashPacks() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-diag-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let counter = VerifierCounter()
        let store = try makeCountingStore(root: root, counter: counter)
        let engine = makeNoMetalEngine(storeProvider: { store })
        _ = await engine.snapshot()
        XCTAssertEqual(counter.count, 0, "snapshot must never SHA-256 packs")
    }

    func testSnapshotReportsReceiptStateWithoutHashing() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-diag-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let counter = VerifierCounter()
        let store = try makeCountingStore(root: root, counter: counter)
        // Establish a trusted receipt via a normal import using a PRODUCTION
        // filename (the snapshot only reports production manifest entries).
        let content = Data("imported pack".utf8)
        let entry = makeEntry(filename: "qwen3-0.6b-xsmax-w8.animapk", content: content)
        let source = root.appendingPathComponent("source.animapk")
        try content.write(to: source)
        _ = try await store.importPack(entry, from: source)
        XCTAssertEqual(counter.count, 1, "import verifies once")

        let engine = makeNoMetalEngine(storeProvider: { store })
        let report = await engine.snapshot()
        let pack = try XCTUnwrap(report.modelPacks.first { $0.filename == entry.filename })
        XCTAssertTrue(pack.installed, "imported file is present")
        XCTAssertEqual(pack.sizeBytes, UInt64(content.count))
        // The receipt cannot match the production manifest (synthetic size/sha),
        // so the cheap flag must NOT claim verified — and the snapshot must not
        // hash the file to discover that.
        XCTAssertFalse(pack.verified, "receipt vs production manifest mismatch is reported truthfully")
        XCTAssertFalse(pack.sha256Verified, "snapshot never claims deep SHA verification")
        XCTAssertEqual(counter.count, 1, "snapshot adds no hashing on top of import")
    }

    // MARK: - Basic self-tests are pack-free

    func testBasicSelfTestsArePackFreeAndDeterministic() async throws {
        let engine = makeNoMetalEngine()
        let items = engine.basicSelfTests()
        let names = items.map(\.name)
        for required in ["W4 vector", "W8 vector", "Golden-noise RNG", "mmap benchmark", "Metal capability"] {
            XCTAssertTrue(names.contains(required), "missing basic test \(required)")
        }
        XCTAssertFalse(names.contains { $0.contains("Deep SHA") },
                       "basic tests must never deep-hash packs")
        let byName = Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.status) })
        XCTAssertEqual(byName["W4 vector"], .pass)
        XCTAssertEqual(byName["W8 vector"], .pass)
        XCTAssertEqual(byName["Golden-noise RNG"], .pass)
        XCTAssertEqual(byName["mmap benchmark"], .pass)
    }

    // MARK: - Hardware tests

    func testHardwareTestsSkippedWithoutMetal() async throws {
        let engine = makeNoMetalEngine()
        let items = await engine.hardwareTests(
            marker: DiagnosticRunMarker(defaults: makeDefaults()),
            progress: { _ in },
            thermalGate: { false })
        XCTAssertEqual(items.count, 3)
        for item in items {
            XCTAssertEqual(item.status, .skipped, "\(item.name) must be SKIPPED without Metal")
        }
        XCTAssertEqual(Set(items.map(\.name)),
                       Set(["MPS precision", "GEMM", "Attention tile"]))
    }

    func testHardwareTestsRespectThermalGate() async throws {
        let engine = DiagnosticsEngine()
        try XCTSkipUnless(engine.isMetalAvailable,
                          "SKIPPED_NO_METAL: default Metal device/library unavailable")
        // With a gate that always trips, every hardware test must be skipped
        // with an explicit reason — never run, never fail.
        let items = await engine.hardwareTests(
            marker: DiagnosticRunMarker(defaults: makeDefaults()),
            progress: { _ in },
            thermalGate: { true })
        XCTAssertEqual(items.count, 3)
        for item in items {
            XCTAssertEqual(item.status, .skipped)
            XCTAssertTrue(item.detail.contains("too warm"),
                          "thermal skip must explain itself: \(item.detail)")
        }
    }

    func testFullRunExecutesEachTestOnce() async throws {
        let engine = makeNoMetalEngine()
        let marker = DiagnosticRunMarker(defaults: makeDefaults())
        let items = await engine.fullRun(marker: marker, progress: { _ in }, thermalGate: { false })
        let names = items.map(\.name)
        let occurrences = Dictionary(grouping: names, by: { $0 }).mapValues(\.count)
        for name in ["W4 vector", "W8 vector", "Golden-noise RNG", "mmap benchmark",
                     "MPS precision", "GEMM", "Attention tile"] {
            XCTAssertEqual(occurrences[name], 1, "\(name) must run exactly once")
        }
    }

    /// The MPS probe must genuinely execute when Metal is available (as it
    /// does on the CI simulator) — never silently skip. A failure here is a
    /// real MPS regression with the corrected rowBytes/status handling, not an
    /// A12 claim: physical A12 behavior still requires the device retest.
    func testHardwareTestsExecuteRealMPSOnMetal() async throws {
        let engine = DiagnosticsEngine()
        try XCTSkipUnless(engine.isMetalAvailable,
                          "SKIPPED_NO_METAL: default Metal device/library unavailable")
        let items = await engine.hardwareTests(
            marker: DiagnosticRunMarker(defaults: makeDefaults()),
            progress: { _ in },
            thermalGate: { false })
        let mps = try XCTUnwrap(items.first { $0.name == "MPS precision" })
        XCTAssertNotEqual(mps.status, .skipped,
                          "MPS precision must actually run when Metal is available")
        XCTAssertTrue(mps.detail.contains("thermal"), "per-test resource facts recorded")
        XCTAssertEqual(mps.status, .pass, "MPS precision detail: \(mps.detail)")
    }

    // MARK: - JSON serialization runs zero tests

    func testJSONSerializationRunsZeroTests() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-diag-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let counter = VerifierCounter()
        let store = try makeCountingStore(root: root, counter: counter)
        let engine = makeNoMetalEngine(storeProvider: { store })
        let report = await engine.snapshot()

        let json = try engine.json(report: report)
        XCTAssertTrue(json.contains("selfTests"), "report JSON contains selfTests")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("anima-xs-diag-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try engine.writeJSON(report, to: url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(object, "report is a JSON object")
        XCTAssertNotNil(object?["modelPacks"], "report has modelPacks array")
        XCTAssertEqual(counter.count, 0, "export/serialization must run zero tests")
    }

    func testReportCodableRoundTrip() async throws {
        let engine = makeNoMetalEngine()
        let report = await engine.snapshot()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(report)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(DiagnosticsReport.self, from: data)
        XCTAssertEqual(decoded.appVersion, report.appVersion)
        XCTAssertEqual(decoded.osVersion, report.osVersion)
        XCTAssertEqual(decoded.metalAvailable, report.metalAvailable)
        XCTAssertEqual(decoded.modelPacks.map(\.filename), report.modelPacks.map(\.filename))
    }

    // MARK: - Deep integrity is explicit

    func testDeepIntegrityHashesEachPackExactlyOnce() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-diag-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let content = Data("deep integrity pack".utf8)
        let entries = [
            makeEntry(filename: "qwen3-0.6b-xsmax-w8.animapk", content: content, component: .textEncoder),
            makeEntry(filename: "anima-turbo-v1.0-xsmax-w4.animapk", content: content, component: .dit),
            makeEntry(filename: "qwen-image-vae-xsmax-fp16.animapk", content: content, component: .vae),
        ]
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        for entry in entries {
            try content.write(to: modelsDir.appendingPathComponent(entry.filename))
        }
        let counter = VerifierCounter()
        let store = try makeCountingStore(root: root, counter: counter)
        let engine = makeNoMetalEngine(storeProvider: { store })
        let marker = DiagnosticRunMarker(defaults: makeDefaults())

        let items = await engine.deepIntegrity(marker: marker, progress: { _ in }, entries: entries)
        XCTAssertEqual(counter.count, 3, "deep integrity hashes each of the three packs once")
        let passes = items.filter { $0.status == .pass }.map(\.name)
        XCTAssertTrue(passes.contains("Deep model SHA-256"), "summary pass item present")
        XCTAssertEqual(items.filter { $0.status == .pass }.count, 4, "3 per-pack + 1 summary")
    }

    func testDeepIntegritySkipsMissingPacks() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-diag-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let counter = VerifierCounter()
        let store = try makeCountingStore(root: root, counter: counter)
        let engine = makeNoMetalEngine(storeProvider: { store })
        let items = await engine.deepIntegrity(
            marker: DiagnosticRunMarker(defaults: makeDefaults()),
            progress: { _ in },
            entries: [makeEntry()])
        XCTAssertEqual(counter.count, 0, "missing pack is rejected before any hashing")
        XCTAssertTrue(items.contains { $0.name == "Deep model SHA-256" && $0.status == .fail },
                      "summary reports the missing pack")
    }

    // MARK: - Status representation

    func testStatusRepresentation() {
        XCTAssertEqual(DiagnosticStatus.pass.rawValue, "pass")
        XCTAssertEqual(DiagnosticStatus.fail.rawValue, "fail")
        XCTAssertEqual(DiagnosticStatus.skipped.rawValue, "skipped")
    }

    private func makeDefaults() -> UserDefaults {
        UserDefaults(suiteName: "AnimaXS-diag-tests-\(UUID().uuidString)")!
    }
}
