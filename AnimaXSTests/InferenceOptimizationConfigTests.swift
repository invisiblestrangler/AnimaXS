import XCTest
@testable import AnimaXS

/// Tests for the immutable per-generation inference configuration and its
/// persistent settings wrapper (§18.1). The baseline must reproduce current
/// HEAD behavior exactly; invalid persisted values sanitize to baseline.
final class InferenceOptimizationConfigTests: XCTestCase {

    // MARK: - Baseline

    func testBaselineIsExactlyCurrentHeadBehavior() {
        let baseline = InferenceOptimizationConfig.currentBaseline
        XCTAssertEqual(baseline.linearTileRows, 128)
        XCTAssertEqual(baseline.attentionTileRows, 128)
        XCTAssertFalse(baseline.directLinearMPSIO)
        XCTAssertTrue(baseline.pingPongWeightStreaming)
        XCTAssertTrue(baseline.numericalMonitoring)
        XCTAssertEqual(baseline.ditPackVariant, .productionW4)
    }

    func testBaselineEnablesCheckpointing() {
        XCTAssertTrue(InferenceOptimizationConfig.currentBaseline.checkpointingEnabled)
    }

    func testExperimentalW8DisablesCheckpointing() {
        var config = InferenceOptimizationConfig.currentBaseline
        config.ditPackVariant = .experimentalW8V2
        XCTAssertFalse(config.checkpointingEnabled)
    }

    // MARK: - Allowed tile rows

    func testAllowedTileRowsAreExactlyTheFourExperimentValues() {
        XCTAssertEqual(InferenceOptimizationConfig.allowedTileRows, [128, 256, 512, 1024])
    }

    func testAllowedTileRowsSurviveSanitization() {
        for rows in [128, 256, 512, 1024] {
            XCTAssertEqual(InferenceOptimizationConfig.sanitizedTileRows(rows), rows)
        }
    }

    func testInvalidTileRowsSanitizeToBaseline() {
        XCTAssertEqual(InferenceOptimizationConfig.sanitizedTileRows(0), 128)
        XCTAssertEqual(InferenceOptimizationConfig.sanitizedTileRows(-1), 128)
        XCTAssertEqual(InferenceOptimizationConfig.sanitizedTileRows(64), 128)
        XCTAssertEqual(InferenceOptimizationConfig.sanitizedTileRows(300), 256)
        XCTAssertEqual(InferenceOptimizationConfig.sanitizedTileRows(9999), 1024)
    }

    // MARK: - Value semantics / immutability

    func testConfigIsValueSemantic() {
        var a = InferenceOptimizationConfig.currentBaseline
        var b = a
        b.linearTileRows = 1024
        b.ditPackVariant = .experimentalW8V2
        XCTAssertEqual(a.linearTileRows, 128)
        XCTAssertEqual(a.ditPackVariant, .productionW4)
        XCTAssertEqual(b.linearTileRows, 1024)
        XCTAssertEqual(b.ditPackVariant, .experimentalW8V2)
        XCTAssertEqual(a, InferenceOptimizationConfig.currentBaseline)
    }

    // MARK: - Persistent settings (UserDefaults)

    private func makeDefaults() -> UserDefaults {
        let suite = "InferenceOptimizationConfigTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    @MainActor
    func testDefaultsMatchBaseline() {
        let settings = InferenceOptimizationSettings(defaults: makeDefaults())
        XCTAssertEqual(settings.snapshot, InferenceOptimizationConfig.currentBaseline)
    }

    @MainActor
    func testAllTileRowsSurvivePersistence() {
        for rows in [128, 256, 512, 1024] {
            let defaults = makeDefaults()
            let settings = InferenceOptimizationSettings(defaults: defaults)
            settings.setLinearTileRows(rows)
            settings.setAttentionTileRows(rows)
            let reloaded = InferenceOptimizationSettings(defaults: defaults)
            XCTAssertEqual(reloaded.linearTileRows, rows, "linear tile rows persist")
            XCTAssertEqual(reloaded.attentionTileRows, rows, "attention tile rows persist")
        }
    }

    @MainActor
    func testInvalidPersistedTileRowsSanitizeToBaseline() {
        let defaults = makeDefaults()
        defaults.set(9999, forKey: InferenceOptimizationSettings.Keys.linearTileRows)
        defaults.set(-5, forKey: InferenceOptimizationSettings.Keys.attentionTileRows)
        let settings = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertEqual(settings.linearTileRows, 1024)
        XCTAssertEqual(settings.attentionTileRows, 128)
    }

    @MainActor
    func testCorruptPersistedVariantSanitizesToBaseline() {
        let defaults = makeDefaults()
        defaults.set("not-a-variant", forKey: InferenceOptimizationSettings.Keys.ditPackVariant)
        let settings = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertEqual(settings.ditPackVariant, .productionW4)
    }

    @MainActor
    func testResetRestoresBaselineAndPersists() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setLinearTileRows(1024)
        settings.setAttentionTileRows(512)
        settings.setDirectLinearMPSIO(true)
        settings.setPingPongWeightStreaming(false)
        settings.setNumericalMonitoring(false)
        settings.setDiTPackVariant(.experimentalW8V2)
        settings.resetToBaseline()
        XCTAssertEqual(settings.snapshot, InferenceOptimizationConfig.currentBaseline)
        let reloaded = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertEqual(reloaded.snapshot, InferenceOptimizationConfig.currentBaseline)
    }

    @MainActor
    func testSnapshotIsIndependentOfLaterMutations() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        let snapshot = settings.snapshot
        settings.setLinearTileRows(1024)
        settings.setNumericalMonitoring(false)
        XCTAssertEqual(snapshot, InferenceOptimizationConfig.currentBaseline,
                       "snapshot must not observe later toggle changes")
    }
}
