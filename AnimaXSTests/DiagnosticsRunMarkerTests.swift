import XCTest
@testable import AnimaXS

/// Runbook §14 regression tests: the persistent diagnostic-run marker must
/// localize a crash to the exact test that was running, and must never report
/// a false crash for a clean session.
final class DiagnosticsRunMarkerTests: XCTestCase {

    private func makeMarker() -> (DiagnosticRunMarker, UserDefaults) {
        let defaults = UserDefaults(suiteName: "AnimaXS-marker-tests-\(UUID().uuidString)")!
        return (DiagnosticRunMarker(defaults: defaults), defaults)
    }

    func testFreshDefaultsShowNoCrashMarker() {
        let (marker, _) = makeMarker()
        XCTAssertNil(marker.unfinishedTest())
    }

    func testStartedTestIsReportedAsUnfinished() {
        let (marker, _) = makeMarker()
        marker.beginSession()
        XCTAssertNil(marker.unfinishedTest(), "no current test yet after beginSession")
        marker.markStarted("MPS precision")
        XCTAssertEqual(marker.unfinishedTest(), "MPS precision",
                       "crash mid-test must be attributable")
    }

    func testCompletedTestClearsMarker() {
        let (marker, _) = makeMarker()
        marker.beginSession()
        marker.markStarted("GEMM")
        marker.markCompleted("GEMM")
        XCTAssertNil(marker.unfinishedTest())
    }

    func testSequentialTestsAttributeCrashToCurrentTest() {
        let (marker, _) = makeMarker()
        marker.beginSession()
        marker.markStarted("W4 vector")
        marker.markCompleted("W4 vector")
        marker.markStarted("MPS precision")
        // App "relaunch" simulation: only the marker survives.
        XCTAssertEqual(marker.unfinishedTest(), "MPS precision",
                       "crash must be attributed to the CURRENT test, not the last completed one")
    }

    func testCleanSessionShowsNoFalseCrashMarker() {
        let (marker, _) = makeMarker()
        marker.beginSession()
        marker.markStarted("W4 vector")
        marker.markCompleted("W4 vector")
        marker.markStarted("MPS precision")
        marker.markCompleted("MPS precision")
        marker.markSessionClean()
        XCTAssertNil(marker.unfinishedTest(),
                     "a fully completed session must not show a crash marker")
    }

    func testBeginSessionClearsPreviousCurrentTest() {
        let (marker, _) = makeMarker()
        marker.beginSession()
        marker.markStarted("Deep SHA: anima-turbo-v1.0-xsmax-w4.animapk")
        XCTAssertEqual(marker.unfinishedTest(),
                       "Deep SHA: anima-turbo-v1.0-xsmax-w4.animapk")
        // A new run starts a fresh session: the stale test attribution is gone.
        marker.beginSession()
        XCTAssertNil(marker.unfinishedTest())
    }

    func testStateTransitionsAreDurableAcrossMarkerInstances() {
        // The marker reads UserDefaults, so a NEW instance (fresh launch) sees
        // the same state.
        let defaults = UserDefaults(suiteName: "AnimaXS-marker-tests-\(UUID().uuidString)")!
        var marker = DiagnosticRunMarker(defaults: defaults)
        marker.beginSession()
        marker.markStarted("Attention tile")
        marker = DiagnosticRunMarker(defaults: defaults)
        XCTAssertEqual(marker.unfinishedTest(), "Attention tile")
        marker.markSessionClean()
        XCTAssertNil(marker.unfinishedTest())
    }
}
