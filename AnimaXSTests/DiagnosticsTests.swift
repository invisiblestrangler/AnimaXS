import XCTest
@testable import AnimaXS

/// K005 diagnostics tests: Codable round trip, stable required fields,
/// pass/fail/skip representation, missing models, unavailable Metal,
/// deterministic RNG self-test, and JSON file creation.
final class DiagnosticsTests: XCTestCase {

    func testReportCodableRoundTrip() async throws {
        let engine = DiagnosticsEngine()
        let report = await engine.report()
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
        XCTAssertEqual(decoded.selfTests.map(\.name), report.selfTests.map(\.name))
    }

    func testRequiredFieldsArePresent() async throws {
        let report = await DiagnosticsEngine().report()
        XCTAssertFalse(report.appVersion.isEmpty)
        XCTAssertFalse(report.osVersion.isEmpty)
        XCTAssertFalse(report.deviceModel.isEmpty)
        XCTAssertGreaterThan(report.physicalMemoryBytes, 0)
        XCTAssertFalse(report.thermalState.isEmpty)
        // Exactly three production packs in manifest order.
        XCTAssertEqual(report.modelPacks.count, 3)
        XCTAssertEqual(report.modelPacks.map(\.component),
                       [ModelComponent.dit, .textEncoder, .vae].map(\.rawValue))
        // Self-tests cover the documented set.
        let names = report.selfTests.map(\.name)
        for required in ["Pack validation", "W4 vector", "W8 vector", "Golden-noise RNG",
                         "MPS precision", "GEMM", "Attention tile", "mmap benchmark"] {
            XCTAssertTrue(names.contains(required), "missing self-test \(required)")
        }
    }

    func testSelfTestsHaveExpectedStatusByDefault() async throws {
        // Pack validation reports the truth about the current installation:
        // FAIL when packs are absent, PASS when a valid model-assets-v1 set is
        // already installed. The W4/W8 vector and RNG tests must always PASS
        // because they are deterministic and pack-free.
        let report = await DiagnosticsEngine().report()
        let byName = Dictionary(uniqueKeysWithValues: report.selfTests.map { ($0.name, $0.status) })
        XCTAssertEqual(byName["W4 vector"], .pass, "deterministic W4 decode must pass")
        XCTAssertEqual(byName["W8 vector"], .pass, "deterministic W8 decode must pass")
        XCTAssertEqual(byName["Golden-noise RNG"], .pass, "deterministic RNG must pass")
        // The runner may retain verified assets independently of simulator
        // app state, so accept either truthful result but never a missing or
        // skipped pack-validation item.
        guard let packStatus = byName["Pack validation"] else {
            return XCTFail("Pack validation self-test is missing")
        }
        XCTAssertTrue(packStatus == .pass || packStatus == .fail,
                      "pack validation must report pass or fail, got \(packStatus)")
    }

    func testStatusRepresentation() {
        XCTAssertEqual(DiagnosticStatus.pass.rawValue, "pass")
        XCTAssertEqual(DiagnosticStatus.fail.rawValue, "fail")
        XCTAssertEqual(DiagnosticStatus.skipped.rawValue, "skipped")
    }

    func testUnavailableMetalYieldsSkipped() async throws {
        // Simulate an environment with no Metal (recoverable, not a crash).
        let engine = DiagnosticsEngine(context: nil, forceMetalUnavailable: true)
        let report = await engine.report()
        XCTAssertFalse(report.metalAvailable)
        let names = report.selfTests.map(\.name)
        let statuses = report.selfTests.map(\.status)
        // The Metal-dependent tests must be SKIPPED, not PASS or FAIL.
        for (index, name) in names.enumerated() {
            if name == "MPS precision" || name == "GEMM"
                || name == "Attention tile" || name == "mmap benchmark" {
                XCTAssertEqual(statuses[index], .skipped,
                               "\(name) must be SKIPPED when Metal unavailable")
            }
        }
    }

    func testJSONFileCreation() async throws {
        let engine = DiagnosticsEngine()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("anima-xs-diag-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try await engine.writeJSON(to: url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(object, "report is a JSON object")
        XCTAssertNotNil(object?["selfTests"], "report has selfTests array")
        XCTAssertNotNil(object?["modelPacks"], "report has modelPacks array")
    }
}
