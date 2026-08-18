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

    // The ANE hybrid backend is an explicit, persistable diagnostic choice;
    // unlike the failed P8 direct-QGEMM experiments it must not sanitize away.
    @MainActor
    func testANEHybridLinearBackendPersists() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setLinearBackend(.aneHybridW8)
        XCTAssertEqual(settings.linearBackend, .aneHybridW8)
        XCTAssertEqual(settings.snapshot.linearBackend, .aneHybridW8)
        XCTAssertEqual(defaults.string(forKey: InferenceOptimizationSettings.Keys.linearBackend),
                       DiTLinearBackend.aneHybridW8.rawValue)
        let reloaded = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertEqual(reloaded.linearBackend, .aneHybridW8)
    }

    // DISABLED (Task 5): a persisted noCopyWeightSource == true (e.g. from a
    // pre-disable build) is migrated back to false on load, and the sanitized
    // value is re-persisted so the store is clean — a normal device launch
    // can never hand `true` to Metal.
    @MainActor
    func testPersistedNoCopyTrueMigratesToFalse() {
        let defaults = makeDefaults()
        defaults.set(true, forKey: InferenceOptimizationSettings.Keys.noCopyWeightSource)
        let settings = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertFalse(settings.noCopyWeightSource,
                       "persisted true must migrate to false on load")
        XCTAssertFalse(settings.snapshot.noCopyWeightSource)
        XCTAssertEqual(defaults.bool(forKey: InferenceOptimizationSettings.Keys.noCopyWeightSource),
                       false,
                       "the sanitized value must be re-persisted")
        let reloaded = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertFalse(reloaded.noCopyWeightSource)
        XCTAssertFalse(reloaded.snapshot.noCopyWeightSource)
    }

    // DISABLED (Task 5): a manual setNoCopyWeightSource(true) is
    // normalized to false and NEVER reaches the persisted store.
    @MainActor
    func testSetNoCopyWeightSourceNormalizesTrueToFalse() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setNoCopyWeightSource(true)
        XCTAssertFalse(settings.noCopyWeightSource,
                       "setNoCopyWeightSource(true) must normalize to false")
        XCTAssertFalse(settings.snapshot.noCopyWeightSource)
        XCTAssertEqual(defaults.bool(forKey: InferenceOptimizationSettings.Keys.noCopyWeightSource),
                       false,
                       "true must never be persisted")
        // Setting false stays false and round-trips.
        settings.setNoCopyWeightSource(false)
        XCTAssertFalse(settings.noCopyWeightSource)
        XCTAssertEqual(defaults.bool(forKey: InferenceOptimizationSettings.Keys.noCopyWeightSource),
                       false)
    }

    // DISABLED (Task 5): applying a preset whose P6 no-copy part is disabled
    // persists noCopyWeightSource == false (makeConfig already neutralizes
    // it; the settings layer must never reintroduce true).
    @MainActor
    func testSetPresetNeverPersistsNoCopyTrue() {
        for preset in [InferencePreset.noCopyCandidate, .allCandidate] {
            let defaults = makeDefaults()
            let settings = InferenceOptimizationSettings(defaults: defaults)
            settings.setPreset(preset)
            XCTAssertFalse(settings.noCopyWeightSource)
            XCTAssertFalse(settings.snapshot.noCopyWeightSource)
            XCTAssertEqual(defaults.bool(forKey: InferenceOptimizationSettings.Keys.noCopyWeightSource),
                           false, "\(preset) must never persist noCopyWeightSource == true")
            let reloaded = InferenceOptimizationSettings(defaults: defaults)
            XCTAssertFalse(reloaded.noCopyWeightSource)
        }
    }

    // QUARANTINED (Task 4): the quarantined flag covers exactly the two
    // disabled P8 backends.
    func testQuarantineFlag() {
        XCTAssertTrue(DiTLinearBackend.directQuantized.isQuarantined)
        XCTAssertTrue(DiTLinearBackend.hybrid.isQuarantined)
        XCTAssertFalse(DiTLinearBackend.dequantizedMPS.isQuarantined)
        XCTAssertFalse(DiTLinearBackend.aneHybridW8.isQuarantined)
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

    // No-copy candidate = KV + mmap no-copy (experimental). DISABLED
    // (Task 5): the no-copy part is forced back to false — a physical A12
    // run hit a real GPU page fault (kIOGPUCommandBufferCallbackErrorPageFault)
    // while no-copy bytes were being served. Every other component of the
    // combination still applies unchanged.
    @MainActor
    func testNoCopyCandidatePreset() {
        let c = InferencePreset.noCopyCandidate.makeConfig()
        XCTAssertFalse(c.noCopyWeightSource)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertEqual(c.attentionBackend, .stridedTokenMajorMPS)
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
    // DISABLED (Task 5): the P6 mmap no-copy part is also forced back to
    // false (real A12 GPU page fault while no-copy bytes were being served).
    @MainActor
    func testAllCandidatePreset() {
        let c = InferencePreset.allCandidate.makeConfig()
        XCTAssertTrue(c.fusedNormModulation)
        XCTAssertTrue(c.fusedMLPActivation)
        XCTAssertTrue(c.stridedTokenMajorAttention)
        XCTAssertTrue(c.crossKVCache)
        XCTAssertFalse(c.noCopyWeightSource)
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
    // DISABLED (Task 5): the noCopy/all presets also keep every NON-no-copy
    // component — only noCopyWeightSource is neutralized.
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
        XCTAssertFalse(all.noCopyWeightSource)
        XCTAssertEqual(all.attentionBackend, .streamingMPS)
        XCTAssertEqual(all.linearBackend, .dequantizedMPS)
    }

    // DISABLED (Task 5): no preset a normal device user can apply may ever
    // produce a `noCopyWeightSource == true` — the P6 mmap no-copy path is
    // blocked after a real A12 GPU page fault
    // (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy bytes were
    // being served. Correctness/safety hardening, not a proof of the
    // historical root cause.
    @MainActor
    func testNoPresetProducesNoCopyWeightSource() {
        for preset in InferencePreset.allCases {
            let c = preset.makeConfig()
            XCTAssertFalse(c.noCopyWeightSource,
                           "\(preset) must not produce noCopyWeightSource == true")
        }
    }

    // DISABLED (Task 5): the disabled-no-copy marker covers exactly the two
    // presets whose P6 part is neutralized.
    func testDisabledNoCopyMarker() {
        XCTAssertTrue(InferencePreset.noCopyCandidate.containsDisabledNoCopy)
        XCTAssertTrue(InferencePreset.allCandidate.containsDisabledNoCopy)
        XCTAssertFalse(InferencePreset.baseline.containsDisabledNoCopy)
        XCTAssertFalse(InferencePreset.stridedMPSKV.containsDisabledNoCopy)
        XCTAssertFalse(InferencePreset.streamingMPSCandidate.containsDisabledNoCopy)
        XCTAssertFalse(InferencePreset.metalFlashCandidate.containsDisabledNoCopy)
    }

    // DISABLED (Task 5): the disabled-no-copy presets keep every NON-no-copy
    // component of their combination — only noCopyWeightSource is neutralized.
    @MainActor
    func testDisabledNoCopyPresetsKeepOtherComponents() {
        let noCopy = InferencePreset.noCopyCandidate.makeConfig()
        XCTAssertFalse(noCopy.noCopyWeightSource)
        XCTAssertTrue(noCopy.crossKVCache)
        XCTAssertTrue(noCopy.stridedTokenMajorAttention)
        XCTAssertEqual(noCopy.attentionBackend, .stridedTokenMajorMPS)

        let all = InferencePreset.allCandidate.makeConfig()
        XCTAssertFalse(all.noCopyWeightSource)
        XCTAssertTrue(all.fusedNormModulation)
        XCTAssertTrue(all.fusedMLPActivation)
        XCTAssertTrue(all.stridedTokenMajorAttention)
        XCTAssertTrue(all.crossKVCache)
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

    // MARK: - Preset marker drift (Task 9)

    // A manual individual control mutation after applying a preset clears the
    // active-preset marker and removes its persisted key, so Diagnostics
    // shows "Custom" instead of a stale preset name.
    @MainActor
    func testManualSetterClearsActivePresetMarker() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.baseline)
        XCTAssertEqual(settings.activePreset, .baseline)
        XCTAssertEqual(defaults.string(forKey: InferenceOptimizationSettings.Keys.activePreset),
                       InferencePreset.baseline.rawValue)
        settings.setLinearTileRows(256)
        XCTAssertNil(settings.activePreset,
                     "a manual setter must clear the preset marker")
        XCTAssertNil(defaults.string(forKey: InferenceOptimizationSettings.Keys.activePreset),
                     "a manual setter must remove the persisted preset marker key")
    }

    // After a manual edit clears the marker, a relaunch (fresh settings
    // instance over the same store) stays nil/Custom — the marker is not
    // restored from anywhere.
    @MainActor
    func testManualEditSurvivesRelaunchAsCustom() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.stridedMPSKV)
        settings.setCrossKVCache(false)
        XCTAssertNil(settings.activePreset)
        let reloaded = InferenceOptimizationSettings(defaults: defaults)
        XCTAssertNil(reloaded.activePreset,
                     "relaunch must stay Custom after a manual edit")
        XCTAssertEqual(reloaded.crossKVCache, false,
                       "the manual control value itself still persists")
        XCTAssertEqual(reloaded.stridedTokenMajorAttention, true,
                       "the preset's other controls remain applied")
    }

    // Every individual setter clears the marker (representative sweep over a
    // Bool toggle, a tile row, and both backend selectors).
    @MainActor
    func testEveryIndividualSetterClearsMarker() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.allCandidate)
        XCTAssertNotNil(settings.activePreset)

        settings.setDirectLinearMPSIO(false)
        XCTAssertNil(settings.activePreset)
        settings.setPreset(.allCandidate)

        settings.setAttentionTileRows(512)
        XCTAssertNil(settings.activePreset)
        settings.setPreset(.allCandidate)

        settings.setAttentionBackend(.stridedTokenMajorMPS)
        XCTAssertNil(settings.activePreset)
        settings.setPreset(.allCandidate)

        settings.setLinearBackend(.dequantizedMPS)
        XCTAssertNil(settings.activePreset)
        XCTAssertNil(defaults.string(forKey: InferenceOptimizationSettings.Keys.activePreset))
    }

    // resetToBaseline goes through the same clearPresetMarker path: exact
    // baseline values + nil marker + persisted key removed.
    @MainActor
    func testResetToBaselineClearsPersistedMarkerKey() {
        let defaults = makeDefaults()
        let settings = InferenceOptimizationSettings(defaults: defaults)
        settings.setPreset(.metalFlashCandidate)
        XCTAssertNotNil(defaults.string(forKey: InferenceOptimizationSettings.Keys.activePreset))
        settings.resetToBaseline()
        XCTAssertNil(settings.activePreset)
        XCTAssertNil(defaults.string(forKey: InferenceOptimizationSettings.Keys.activePreset),
                     "resetToBaseline must remove the persisted marker key")
        XCTAssertEqual(settings.snapshot, InferenceOptimizationConfig.currentBaseline)
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

    // MARK: - Central compatibility validator (Task 9)

    private func config(
        linearTileRows: Int = 128,
        attentionTileRows: Int = 128,
        directLinearMPSIO: Bool = false,
        pingPongWeightStreaming: Bool = true,
        numericalMonitoring: Bool = true,
        fusedNormModulation: Bool = false,
        fusedMLPActivation: Bool = false,
        stridedTokenMajorAttention: Bool = false,
        crossKVCache: Bool = false,
        noCopyWeightSource: Bool = false,
        attentionBackend: DiTAttentionBackend = .legacyHeadMajorMPS,
        linearBackend: DiTLinearBackend = .dequantizedMPS
    ) -> InferenceOptimizationConfig {
        InferenceOptimizationConfig(
            linearTileRows: linearTileRows,
            attentionTileRows: attentionTileRows,
            directLinearMPSIO: directLinearMPSIO,
            pingPongWeightStreaming: pingPongWeightStreaming,
            numericalMonitoring: numericalMonitoring,
            fusedNormModulation: fusedNormModulation,
            fusedMLPActivation: fusedMLPActivation,
            stridedTokenMajorAttention: stridedTokenMajorAttention,
            crossKVCache: crossKVCache,
            noCopyWeightSource: noCopyWeightSource,
            attentionBackend: attentionBackend,
            linearBackend: linearBackend)
    }

    // The baseline configuration is fully compatible.
    func testBaselineConfigHasNoBlockingReason() {
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: .currentBaseline))
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: config()))
    }

    // A no-copy config is blocked with a reason referencing the A12 GPU page
    // fault / no-copy disablement.
    func testNoCopyConfigBlockedWithPageFaultReason() {
        let reason = InferenceOptimizationConfig.blockingReason(
            for: config(noCopyWeightSource: true))
        XCTAssertNotNil(reason)
        XCTAssertTrue(reason!.contains("page fault"), reason!)
        XCTAssertTrue(reason!.contains("no-copy"), reason!)
    }

    // The quarantined P8 linear backends are blocked with a reason referencing
    // the ~10x A12 regression.
    func testQuarantinedLinearBackendsBlocked() {
        for backend in [DiTLinearBackend.directQuantized, .hybrid] {
            let reason = InferenceOptimizationConfig.blockingReason(
                for: config(linearBackend: backend))
            XCTAssertNotNil(reason, "\(backend) must be blocked")
            XCTAssertTrue(reason!.contains("10x"), reason!)
        }
    }

    // Without a resolved variant id the pure settings validator keeps its
    // historical behavior. Once the model is resolved, the native pack/backend
    // pair is enforced explicitly.
    func testANEHybridBackendPassesConfigOnlyGateWithoutResolvedPack() {
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: config(linearBackend: .aneHybridW8), numerics: .w8LegacyStabilized))
    }

    func testANEHybridRequiresNativePackVariant() {
        let c = config(linearBackend: .aneHybridW8)
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: c, numerics: .w8LegacyStabilized, ditVariantID: "w8-ane-v1"))
        for id in ["w4", "w8-v2"] {
            let reason = InferenceOptimizationConfig.blockingReason(
                for: c, numerics: .w8LegacyStabilized, ditVariantID: id)
            XCTAssertEqual(reason, InferenceOptimizationConfig.aneNativePackRequiredReason)
        }
    }

    func testNativeANEPackBlockedFromNonANEBackends() {
        let reason = InferenceOptimizationConfig.blockingReason(
            for: config(linearBackend: .dequantizedMPS),
            numerics: .w8LegacyStabilized, ditVariantID: "w8-ane-v1")
        XCTAssertEqual(reason, InferenceOptimizationConfig.aneNativeBackendRequiredReason)
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: config(linearBackend: .dequantizedMPS),
            numerics: .w8LegacyStabilized, ditVariantID: "w8-v2"))
    }

    // Experimental BF16 numerics + strided token-major attention is blocked:
    // AttentionExecutor refuses bf16Compute for the strided layout.
    func testBF16ExperimentalWithStridedAttentionBlocked() {
        for backend in [DiTAttentionBackend.stridedTokenMajorMPS,
                        .streamingMPS, .metalFlash] {
            let c = config(
                stridedTokenMajorAttention: true,
                attentionBackend: backend)
            let reason = InferenceOptimizationConfig.blockingReason(
                for: c, numerics: .w8BF16Experimental)
            XCTAssertNotNil(reason, "\(backend) with BF16 must be blocked")
            XCTAssertTrue(reason!.contains("BF16"), reason!)
        }
    }

    // Production W8 (w8LegacyStabilized -> legacy/legacy numerics) with
    // strided attention must NOT be blocked — the check is on the RESOLVED
    // numerics, never on the pack name.
    func testW8LegacyStabilizedWithStridedAttentionIsCompatible() {
        for backend in [DiTAttentionBackend.stridedTokenMajorMPS,
                        .streamingMPS, .metalFlash] {
            let c = config(
                stridedTokenMajorAttention: true,
                attentionBackend: backend)
            XCTAssertNil(InferenceOptimizationConfig.blockingReason(
                for: c, numerics: .w8LegacyStabilized),
                "\(backend) with w8LegacyStabilized must stay compatible")
        }
    }

    // W4 legacy numerics with strided attention is compatible too.
    func testW4LegacyWithStridedAttentionIsCompatible() {
        let c = config(
            stridedTokenMajorAttention: true,
            attentionBackend: .stridedTokenMajorMPS)
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: c, numerics: .w4Legacy))
    }

    // Experimental BF16 numerics with the legacy head-major layout is
    // compatible (the executor only rejects the strided layout).
    func testBF16ExperimentalWithHeadMajorAttentionIsCompatible() {
        XCTAssertNil(InferenceOptimizationConfig.blockingReason(
            for: config(), numerics: .w8BF16Experimental))
    }

    // The validator is read-only: it never mutates the config it checks.
    func testValidatorNeverMutatesConfig() {
        var c = config(noCopyWeightSource: true, linearBackend: .hybrid)
        let original = c
        _ = InferenceOptimizationConfig.blockingReason(for: c)
        XCTAssertEqual(c, original,
                       "blockingReason must not mutate the config")
        c.linearBackend = .dequantizedMPS
        c.noCopyWeightSource = false
        c.stridedTokenMajorAttention = true
        c.attentionBackend = .streamingMPS
        let original2 = c
        _ = InferenceOptimizationConfig.blockingReason(for: c, numerics: .w8BF16Experimental)
        XCTAssertEqual(c, original2,
                       "blockingReason must not mutate the config")
    }
}
