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

    // QUARANTINED (Task 4): the baseline linear backend always resolves to
    // the known-good dequantized-MPS path.
    @MainActor
    func testBaselineLinearBackendIsDequantizedMPS() {
        XCTAssertEqual(InferenceOptimizationConfig.currentBaseline.linearBackend,
                       .dequantizedMPS)
        let settings = InferenceOptimizationSettings(defaults: makeDefaults())
        XCTAssertEqual(settings.linearBackend, .dequantizedMPS)
        XCTAssertEqual(settings.snapshot.linearBackend, .dequantizedMPS)
    }

    // QUARANTINED (Task 4): a persisted .directQuantized / .hybrid selection
    // (e.g. from a pre-quarantine build) is migrated back to .dequantizedMPS
    // on load, and the sanitized value is re-persisted so the store is clean.
    @MainActor
    func testPersistedQuarantinedLinearBackendSanitizesToBaseline() {
        for rawValue in ["directQuantized", "hybrid"] {
            let defaults = makeDefaults()
            defaults.set(rawValue, forKey: InferenceOptimizationSettings.Keys.linearBackend)
            let settings = InferenceOptimizationSettings(defaults: defaults)
            XCTAssertEqual(settings.linearBackend, .dequantizedMPS,
                           "persisted \(rawValue) must migrate to dequantizedMPS")
            XCTAssertEqual(settings.snapshot.linearBackend, .dequantizedMPS)
            XCTAssertEqual(defaults.string(forKey: InferenceOptimizationSettings.Keys.linearBackend),
                           "dequantizedMPS",
                           "the sanitized value must be re-persisted for \(rawValue)")
            let reloaded = InferenceOptimizationSettings(defaults: defaults)
            XCTAssertEqual(reloaded.linearBackend, .dequantizedMPS)
        }
    }

    // QUARANTINED (Task 4): a manual selection of .directQuantized / .hybrid
    // is rejected/normalized to .dequantizedMPS and NEVER reaches the
    // persisted store.
    @MainActor
    func testSetLinearBackendRejectsQuarantinedValues() {
        for backend in [DiTLinearBackend.directQuantized, .hybrid] {
            let defaults = makeDefaults()
            let settings = InferenceOptimizationSettings(defaults: defaults)
            settings.setLinearBackend(backend)
            XCTAssertEqual(settings.linearBackend, .dequantizedMPS,
                           "selecting \(backend) must normalize to dequantizedMPS")
            XCTAssertEqual(settings.snapshot.linearBackend, .dequantizedMPS)
            XCTAssertEqual(defaults.string(forKey: InferenceOptimizationSettings.Keys.linearBackend),
                           "dequantizedMPS",
                           "\(backend) must never be persisted")
        }
        // The known-good backend still round-trips normally.
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setLinearBackend(.dequantizedMPS)
        XCTAssertEqual(settings.linearBackend, .dequantizedMPS)
        XCTAssertEqual(defaults.string(forKey: InferenceOptimizationSettings.Keys.linearBackend),
                       "dequantizedMPS")
    }

    // QUARANTINED (Task 4): the quarantined flag covers exactly the two
    // disabled P8 backends.
    func testQuarantineFlag() {
        XCTAssertTrue(DiTLinearBackend.directQuantized.isQuarantined)
        XCTAssertTrue(DiTLinearBackend.hybrid.isQuarantined)
        XCTAssertFalse(DiTLinearBackend.dequantizedMPS.isQuarantined)
        XCTAssertTrue(InferencePreset.directQGEMMCandidate.containsQuarantinedLinearBackend)
        XCTAssertTrue(InferencePreset.allCandidate.containsQuarantinedLinearBackend)
        XCTAssertFalse(InferencePreset.baseline.containsQuarantinedLinearBackend)
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

    // MARK: - P9 presets

    // The baseline preset is exactly the current known-good baseline.
    @MainActor
    func testBaselinePresetEqualsCurrentBaseline() {
        XCTAssertEqual(InferencePreset.baseline.makeConfig(),
                       InferenceOptimizationConfig.currentBaseline)
    }

    // The §17 first-pass control: linear/attention 1024, direct MPS I/O on,
    // ping-pong on, all new optimizations off.
    @MainActor
    func testCurrent1024ControlPreset() {
        let c = InferencePreset.current1024Control.makeConfig()
        XCTAssertEqual(c.linearTileRows, 1024)
        XCTAssertEqual(c.attentionTileRows, 1024)
        XCTAssertTrue(c.directLinearMPSIO)
        XCTAssertTrue(c.pingPongWeightStreaming)
        // All new optimizations OFF.
        XCTAssertFalse(c.fusedNormModulation)
        XCTAssertFalse(c.fusedMLPActivation)
        XCTAssertFalse(c.stridedTokenMajorAttention)
        XCTAssertFalse(c.crossKVCache)
        XCTAssertFalse(c.noCopyWeightSource)
        XCTAssertEqual(c.attentionBackend, .legacyHeadMajorMPS)
        XCTAssertEqual(c.linearBackend, .dequantizedMPS)
    }

    // Fused = control + both fused activation kernels.
    @MainActor
    func testFusedTrafficPreset() {
        let c = InferencePreset.fusedTraffic.makeConfig()
        XCTAssertTrue(c.fusedNormModulation)
        XCTAssertTrue(c.fusedMLPActivation)
        XCTAssertEqual(c.linearTileRows, 1024)
        XCTAssertEqual(c.attentionTileRows, 1024)
    }

    // Strided = fused + strided token-major MPS attention backend.
    @MainActor
    func testStridedMPSPreset() {
        let c = InferencePreset.stridedMPS.makeConfig()
        XCTAssertTrue(c.fusedNormModulation)
        XCTAssertTrue(c.fusedMLPActivation)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertEqual(c.attentionBackend, .stridedTokenMajorMPS)
    }

    // Strided+KV = strided + cross-attention K/V cache.
    @MainActor
    func testStridedMPSKVPreset() {
        let c = InferencePreset.stridedMPSKV.makeConfig()
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertEqual(c.attentionBackend, .stridedTokenMajorMPS)
    }

    // No-copy candidate = KV + mmap no-copy (experimental).
    @MainActor
    func testNoCopyCandidatePreset() {
        let c = InferencePreset.noCopyCandidate.makeConfig()
        XCTAssertTrue(c.noCopyWeightSource)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.stridedTokenMajorAttention)
    }

    // Streaming MPS candidate = fused + KV + streaming MPS attention backend,
    // which REQUIRES the token-major layout.
    @MainActor
    func testStreamingMPSCandidatePreset() {
        let c = InferencePreset.streamingMPSCandidate.makeConfig()
        XCTAssertTrue(c.fusedNormModulation)
        XCTAssertTrue(c.fusedMLPActivation)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertEqual(c.attentionBackend, .streamingMPS)
    }

    // Metal Flash candidate = fused + KV + pure-Metal Flash attention backend
    // (also requires the token-major layout).
    @MainActor
    func testMetalFlashCandidatePreset() {
        let c = InferencePreset.metalFlashCandidate.makeConfig()
        XCTAssertTrue(c.fusedNormModulation)
        XCTAssertTrue(c.fusedMLPActivation)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertEqual(c.attentionBackend, .metalFlash)
    }

    // Direct QGEMM candidate = best attention (strided+KV) + hybrid linear
    // (MLP-only QGEMM), per §17 preset 8. QUARANTINED (Task 4): the QGEMM
    // part is disabled (measured ~10x A12 regression vs dequantizedMPS), so
    // the config keeps every other component but forces linearBackend back
    // to .dequantizedMPS — a device preset can never silently run the
    // 10x-slower direct path.
    @MainActor
    func testDirectQGEMMCandidatePreset() {
        let c = InferencePreset.directQGEMMCandidate.makeConfig()
        XCTAssertEqual(c.linearBackend, .dequantizedMPS)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertEqual(c.attentionBackend, .stridedTokenMajorMPS)
    }

    // allCandidate combines the winning components but is NOT forced as best.
    // QUARANTINED (Task 4): every other combined setting stays as-is EXCEPT
    // the hybrid/QGEMM part, which is forced back to .dequantizedMPS.
    @MainActor
    func testAllCandidatePreset() {
        let c = InferencePreset.allCandidate.makeConfig()
        XCTAssertTrue(c.fusedNormModulation)
        XCTAssertTrue(c.fusedMLPActivation)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.noCopyWeightSource)
        XCTAssertEqual(c.attentionBackend, .streamingMPS)
        XCTAssertEqual(c.linearBackend, .dequantizedMPS)
    }

    // QUARANTINED (Task 4): no preset a normal device user can apply may ever
    // produce a direct/hybrid linear backend.
    @MainActor
    func testNoPresetProducesQuarantinedLinearBackend() {
        for preset in InferencePreset.allCases {
            let c = preset.makeConfig()
            XCTAssertFalse(c.linearBackend.isQuarantined,
                           "\(preset) must not produce a quarantined linear backend")
            XCTAssertEqual(c.linearBackend, .dequantizedMPS,
                           "\(preset) must resolve to the baseline linear backend")
        }
    }

    // QUARANTINED (Task 4): the quarantined presets keep every NON-QGEMM
    // component of their combination — only linearBackend is neutralized.
    @MainActor
    func testQuarantinedPresetsKeepNonQGEMMComponents() {
        let direct = InferencePreset.directQGEMMCandidate.makeConfig()
        XCTAssertTrue(direct.crossKVCache)
        XCTAssertTrue(direct.stridedTokenMajorAttention)
        XCTAssertEqual(direct.attentionBackend, .stridedTokenMajorMPS)
        XCTAssertEqual(direct.linearBackend, .dequantizedMPS)

        let all = InferencePreset.allCandidate.makeConfig()
        XCTAssertTrue(all.fusedNormModulation)
        XCTAssertTrue(all.fusedMLPActivation)
        XCTAssertTrue(all.stridedTokenMajorAttention)
        XCTAssertTrue(all.crossKVCache)
        XCTAssertTrue(all.noCopyWeightSource)
        XCTAssertEqual(all.attentionBackend, .streamingMPS)
        XCTAssertEqual(all.linearBackend, .dequantizedMPS)
    }

    // Every preset's config is internally consistent with respect to backend
    // requirements (streaming/flash require token-major layout).
    @MainActor
    func testAllPresetsSatisfyBackendLayoutInvariant() {
        for preset in InferencePreset.allCases {
            let c = preset.makeConfig()
            switch c.attentionBackend {
            case .legacyHeadMajorMPS, .stridedTokenMajorMPS:
                // Strided MPS honors the stridedTokenMajorAttention bool for
                // exact P4 semantics; no hard invariant here.
                break
            case .streamingMPS, .metalFlash:
                XCTAssertTrue(c.stridedTokenMajorAttention,
                              "\(preset): \(c.attentionBackend) requires token-major")
            }
        }
    }

    // setPreset applies the preset's combination and records it as active.
    @MainActor
    func testSetPresetAppliesCombination() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.streamingMPSCandidate)
        XCTAssertEqual(settings.activePreset, .streamingMPSCandidate)
        XCTAssertEqual(settings.snapshot,
                       InferencePreset.streamingMPSCandidate.makeConfig())
        XCTAssertTrue(settings.snapshot.stridedTokenMajorAttention)
        XCTAssertEqual(settings.snapshot.attentionBackend, .streamingMPS)
    }

    // A preset persists across a relaunch (controls + active marker).
    @MainActor
    func testPresetPersistsAcrossReload() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.metalFlashCandidate)
        let reloaded = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertEqual(reloaded.activePreset, .metalFlashCandidate)
        XCTAssertEqual(reloaded.snapshot,
                       InferencePreset.metalFlashCandidate.makeConfig())
        XCTAssertEqual(reloaded.snapshot.attentionBackend, .metalFlash)
    }

    // An invalid persisted preset raw value sanitizes to nil (no active marker)
    // without disturbing the independently-sanitized controls.
    @MainActor
    func testInvalidPersistedPresetSanitizesToNil() {
        let defaults = makeDefaults()
        defaults.set("not-a-real-preset", forKey: InferenceOptimizationSettings.Keys.activePreset)
        defaults.set(1024, forKey: InferenceOptimizationSettings.Keys.linearTileRows)
        let settings = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertNil(settings.activePreset)
        XCTAssertEqual(settings.linearTileRows, 1024)
    }

    // resetToBaseline restores baseline AND clears the active preset marker.
    @MainActor
    func testResetClearsActivePreset() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.allCandidate)
        XCTAssertNotEqual(settings.snapshot, InferenceOptimizationConfig.currentBaseline)
        settings.resetToBaseline()
        XCTAssertEqual(settings.snapshot, InferenceOptimizationConfig.currentBaseline)
        XCTAssertNil(settings.activePreset)
        let reloaded = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertNil(reloaded.activePreset)
        XCTAssertEqual(reloaded.snapshot, InferenceOptimizationConfig.currentBaseline)
    }

    // Applying the baseline preset is equivalent to resetToBaseline.
    @MainActor
    func testBaselinePresetRestoresBaseline() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.allCandidate)
        settings.setPreset(.baseline)
        XCTAssertEqual(settings.snapshot, InferenceOptimizationConfig.currentBaseline)
        XCTAssertEqual(settings.activePreset, .baseline)
    }

    // A preset application does not change the immutable-snapshot contract.
    @MainActor
    func testSetPresetSnapshotStable() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        let before = settings.snapshot
        settings.setPreset(.directQGEMMCandidate)
        XCTAssertEqual(before, InferenceOptimizationConfig.currentBaseline,
                       "snapshot taken before setPreset must stay baseline")
    }
}
