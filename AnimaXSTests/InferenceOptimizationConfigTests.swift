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
    }

    func testBaselineEnablesCheckpointing() {
        XCTAssertTrue(InferenceOptimizationConfig.currentBaseline.checkpointingEnabled)
    }

    // P3: the fused activation toggles default OFF and do not change the
    // known-good baseline behavior (legacy paths remain the default).
    func testP3FusedTogglesDefaultOff() {
        let baseline = InferenceOptimizationConfig.currentBaseline
        XCTAssertFalse(baseline.fusedNormModulation)
        XCTAssertFalse(baseline.fusedMLPActivation)
    }

    // P4: the strided token-major attention toggle defaults OFF, so the
    // legacy head-major transpose path remains the baseline behavior.
    func testP4StridedTokenMajorAttentionDefaultsOff() {
        let baseline = InferenceOptimizationConfig.currentBaseline
        XCTAssertFalse(baseline.stridedTokenMajorAttention)
    }

    // P6: the mmap no-copy weight source toggle defaults OFF, so the
    // memcpy'd slot-ring path remains the baseline behavior (experimental).
    func testP6NoCopyWeightSourceDefaultsOff() {
        let baseline = InferenceOptimizationConfig.currentBaseline
        XCTAssertFalse(baseline.noCopyWeightSource)
    }

    func testP6NoCopyWeightSourceIsIndependent() {
        var config = InferenceOptimizationConfig.currentBaseline
        config.noCopyWeightSource = true
        XCTAssertTrue(config.noCopyWeightSource)
        XCTAssertFalse(config.crossKVCache)          // independent
        XCTAssertFalse(config.stridedTokenMajorAttention) // independent
    }

    func testP4StridedTokenMajorAttentionIsIndependent() {
        var config = InferenceOptimizationConfig.currentBaseline
        config.stridedTokenMajorAttention = true
        XCTAssertTrue(config.stridedTokenMajorAttention)
        XCTAssertFalse(config.fusedNormModulation) // independent
        XCTAssertFalse(config.fusedMLPActivation)  // independent
    }

    func testP3FusedTogglesAreIndependentAndExplicit() {
        var config = InferenceOptimizationConfig.currentBaseline
        config.fusedNormModulation = true
        XCTAssertTrue(config.fusedNormModulation)
        XCTAssertFalse(config.fusedMLPActivation) // independent
        config.fusedMLPActivation = true
        XCTAssertTrue(config.fusedMLPActivation)
    }

    // The DiT pack is whichever variant (W4 or W8-v2) was imported into the
    // .dit slot; it is not a config choice, and checkpointing is always on
    // because the slot holds exactly one verified pack.
    func testCheckpointingIsAlwaysEnabled() {
        var config = InferenceOptimizationConfig.currentBaseline
        config.linearTileRows = 1024
        config.numericalMonitoring = false
        XCTAssertTrue(config.checkpointingEnabled)
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
        b.numericalMonitoring = false
        XCTAssertEqual(a.linearTileRows, 128)
        XCTAssertTrue(a.numericalMonitoring)
        XCTAssertEqual(b.linearTileRows, 1024)
        XCTAssertFalse(b.numericalMonitoring)
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
    func testResetRestoresBaselineAndPersists() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setLinearTileRows(1024)
        settings.setAttentionTileRows(512)
        settings.setDirectLinearMPSIO(true)
        settings.setPingPongWeightStreaming(false)
        settings.setNumericalMonitoring(false)
        settings.setFusedNormModulation(true)
        settings.setFusedMLPActivation(true)
        settings.setStridedTokenMajorAttention(true)
        settings.setCrossKVCache(true)
        settings.setNoCopyWeightSource(true)
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
