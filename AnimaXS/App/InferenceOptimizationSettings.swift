import Foundation

/// Persistent, user-editable runtime optimization settings for the Diagnostics
/// screen. Converts to an immutable `InferenceOptimizationConfig` snapshot at
/// Generate time; a generation never observes a mid-run toggle change.
///
/// Values persist to `UserDefaults` under a stable key. Invalid/corrupt
/// persisted values are sanitized back to baseline so they never reach the
/// executors.
@MainActor
final class InferenceOptimizationSettings: ObservableObject {
    enum Keys {
        static let linearTileRows = "inference.linearTileRows"
        static let attentionTileRows = "inference.attentionTileRows"
        static let directLinearMPSIO = "inference.directLinearMPSIO"
        static let pingPongWeightStreaming = "inference.pingPongWeightStreaming"
        static let numericalMonitoring = "inference.numericalMonitoring"
        static let fusedNormModulation = "inference.fusedNormModulation"
        static let fusedMLPActivation = "inference.fusedMLPActivation"
        static let stridedTokenMajorAttention = "inference.stridedTokenMajorAttention"
        static let crossKVCache = "inference.crossKVCache"
        static let noCopyWeightSource = "inference.noCopyWeightSource"
        static let attentionBackend = "inference.attentionBackend"
        static let linearBackend = "inference.linearBackend"
        static let activePreset = "inference.activePreset"
    }

    private let defaults: UserDefaults

    /// The P9 preset most recently applied, if any. Persisted so a relaunch
    /// restores the user's chosen configuration. `nil` until a preset is
    /// applied (e.g. fresh install or after `resetToBaseline`). Adjusting an
    /// individual control afterwards CLEARS the marker (Task 9): the preset
    /// combination no longer exactly describes the current controls, so
    /// Diagnostics shows "Custom" instead of a stale preset name.
    @Published private(set) var activePreset: InferencePreset?

    @Published private(set) var linearTileRows: Int
    @Published private(set) var attentionTileRows: Int
    @Published private(set) var directLinearMPSIO: Bool
    @Published private(set) var pingPongWeightStreaming: Bool
    @Published private(set) var numericalMonitoring: Bool
    @Published private(set) var fusedNormModulation: Bool
    @Published private(set) var fusedMLPActivation: Bool
    @Published private(set) var stridedTokenMajorAttention: Bool
    @Published private(set) var crossKVCache: Bool
    @Published private(set) var noCopyWeightSource: Bool
    @Published private(set) var attentionBackend: DiTAttentionBackend
    @Published private(set) var linearBackend: DiTLinearBackend

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        // Load + sanitize every value at init so a corrupt persisted set can
        // never escape to the executors.
        let baseline = InferenceOptimizationConfig.currentBaseline
        linearTileRows = InferenceOptimizationSettings.loadTileRows(
            from: defaults, key: Keys.linearTileRows, baseline: baseline.linearTileRows)
        attentionTileRows = InferenceOptimizationSettings.loadTileRows(
            from: defaults, key: Keys.attentionTileRows, baseline: baseline.attentionTileRows)
        directLinearMPSIO = defaults.object(forKey: Keys.directLinearMPSIO) as? Bool
            ?? baseline.directLinearMPSIO
        pingPongWeightStreaming = defaults.object(forKey: Keys.pingPongWeightStreaming) as? Bool
            ?? baseline.pingPongWeightStreaming
        numericalMonitoring = defaults.object(forKey: Keys.numericalMonitoring) as? Bool
            ?? baseline.numericalMonitoring
        fusedNormModulation = defaults.object(forKey: Keys.fusedNormModulation) as? Bool
            ?? baseline.fusedNormModulation
        fusedMLPActivation = defaults.object(forKey: Keys.fusedMLPActivation) as? Bool
            ?? baseline.fusedMLPActivation
        stridedTokenMajorAttention = defaults.object(forKey: Keys.stridedTokenMajorAttention) as? Bool
            ?? baseline.stridedTokenMajorAttention
        crossKVCache = defaults.object(forKey: Keys.crossKVCache) as? Bool
            ?? baseline.crossKVCache
        noCopyWeightSource = InferenceOptimizationSettings.loadNoCopyWeightSource(
            from: defaults, baseline: baseline.noCopyWeightSource)
        attentionBackend = InferenceOptimizationSettings.loadAttentionBackend(
            from: defaults, baseline: baseline.attentionBackend)
        linearBackend = InferenceOptimizationSettings.loadLinearBackend(
            from: defaults, baseline: baseline.linearBackend)
        activePreset = InferenceOptimizationSettings.loadPreset(
            from: defaults)
    }

    // MARK: - Mutations (validate before persist)

    /// Task 9: clears the active-preset marker after an INDIVIDUAL manual
    /// control mutation. The named preset no longer exactly describes the
    /// current controls, so the persisted marker is removed and Diagnostics
    /// shows "Custom" — a stale preset name must never be claimed. The
    /// persisted control value the caller just wrote is NOT touched (only the
    /// marker key is removed); `setPreset` re-establishes a marker.
    private func clearPresetMarker() {
        activePreset = nil
        defaults.removeObject(forKey: Keys.activePreset)
    }

    func setLinearTileRows(_ value: Int) {
        linearTileRows = InferenceOptimizationConfig.sanitizedTileRows(value)
        defaults.set(linearTileRows, forKey: Keys.linearTileRows)
        clearPresetMarker()
    }

    func setAttentionTileRows(_ value: Int) {
        attentionTileRows = InferenceOptimizationConfig.sanitizedTileRows(value)
        defaults.set(attentionTileRows, forKey: Keys.attentionTileRows)
        clearPresetMarker()
    }

    func setDirectLinearMPSIO(_ value: Bool) {
        directLinearMPSIO = value
        defaults.set(value, forKey: Keys.directLinearMPSIO)
        clearPresetMarker()
    }

    func setPingPongWeightStreaming(_ value: Bool) {
        pingPongWeightStreaming = value
        defaults.set(value, forKey: Keys.pingPongWeightStreaming)
        clearPresetMarker()
    }

    func setNumericalMonitoring(_ value: Bool) {
        numericalMonitoring = value
        defaults.set(value, forKey: Keys.numericalMonitoring)
        clearPresetMarker()
    }

    func setFusedNormModulation(_ value: Bool) {
        fusedNormModulation = value
        defaults.set(value, forKey: Keys.fusedNormModulation)
        clearPresetMarker()
    }

    func setFusedMLPActivation(_ value: Bool) {
        fusedMLPActivation = value
        defaults.set(value, forKey: Keys.fusedMLPActivation)
        clearPresetMarker()
    }

    func setStridedTokenMajorAttention(_ value: Bool) {
        stridedTokenMajorAttention = value
        defaults.set(value, forKey: Keys.stridedTokenMajorAttention)
        clearPresetMarker()
    }

    func setCrossKVCache(_ value: Bool) {
        crossKVCache = value
        defaults.set(value, forKey: Keys.crossKVCache)
        clearPresetMarker()
    }

    func setNoCopyWeightSource(_ value: Bool) {
        // DISABLED (Task 5): the P6 mmap no-copy weight source is blocked for
        // normal production/device settings after a physical A12 run hit a
        // real GPU page fault (kIOGPUCommandBufferCallbackErrorPageFault)
        // while no-copy bytes were being served. `true` is normalized to
        // `false` and NEVER persisted; the research implementation
        // (WeightNoCopyPolicy.makeAlias / WeightStreamer) stays intact so an
        // isolated hardware test can re-enable it later. This is
        // correctness/safety hardening, NOT a proof of the historical root
        // cause.
        noCopyWeightSource = false
        defaults.set(false, forKey: Keys.noCopyWeightSource)
        // Task 9: an individual manual mutation clears the active-preset
        // marker — the preset combination no longer exactly describes the
        // current controls, so Diagnostics shows "Custom".
        clearPresetMarker()
    }

    func setAttentionBackend(_ value: DiTAttentionBackend) {
        attentionBackend = value
        defaults.set(value.rawValue, forKey: Keys.attentionBackend)
        clearPresetMarker()
    }

    /// QUARANTINED (Task 4): the P8 direct packed QGEMM backends
    /// (`.directQuantized`, `.hybrid`) measured ~10x SLOWER than
    /// `.dequantizedMPS` on the physical A12 device — a measured PERFORMANCE
    /// regression, NOT a proven correctness failure (the research kernel
    /// stays intact and directly testable via `LinearExecutor`). Normal
    /// device settings must never select them while the optimization search
    /// runs, so a manual selection is rejected/normalized to
    /// `.dequantizedMPS` and never persisted. `.aneHybridW8` is a separate
    /// explicit A12/H11 backend and is allowed to persist.
    static let quarantineReason = "P8 direct QGEMM (.directQuantized / .hybrid) is temporarily disabled: it measured ~10x slower than dequantized MPS on the A12 device (performance regression, not a proven correctness failure). The research kernel remains testable directly via LinearExecutor."

    /// DISABLED (Task 5): human-readable reason for the P6 mmap no-copy
    /// weight-source block. A physical A12 run hit a real GPU page fault
    /// (`kIOGPUCommandBufferCallbackErrorPageFault`) while no-copy bytes
    /// were being served, so normal production/device settings can never
    /// select the no-copy path again until an isolated hardware test proves
    /// it safe. This is correctness/safety hardening, NOT a proof of the
    /// historical root cause — the research implementation
    /// (`WeightNoCopyPolicy.makeAlias` / `WeightStreamer`) stays intact.
    static let p6NoCopyDisabledReason = "P6 mmap no-copy weight source is temporarily disabled: a physical A12 run hit a real GPU page fault (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy bytes were being served. This is correctness/safety hardening, not a proof of the historical root cause; the research implementation remains intact for an isolated hardware test."

    /// Task 9: the central-validator reasons live on `InferenceOptimizationConfig`
    /// (`noCopyBlockingReason`, `linearBackendBlockingReason`,
    /// `bf16StridedAttentionBlockingReason`) — the single source of truth for
    /// Generate-time blocking text. These aliases keep the Diagnostics-side
    /// wording identical to the validator's wording so the UI never shows two
    /// different explanations for the same block.
    static var noCopyBlockingReason: String { InferenceOptimizationConfig.noCopyBlockingReason }
    static var linearBackendBlockingReason: String { InferenceOptimizationConfig.linearBackendBlockingReason }
    static var bf16StridedAttentionBlockingReason: String { InferenceOptimizationConfig.bf16StridedAttentionBlockingReason }

    func setLinearBackend(_ value: DiTLinearBackend) {
        // QUARANTINED (Task 4): reject direct/hybrid selections for normal
        // device settings — normalize to the known-good baseline and never
        // let a quarantined rawValue reach the persisted store. The P8
        // research path remains reachable only through a manually
        // constructed LinearExecutor (research tests).
        let sanitized = value.isQuarantined ? DiTLinearBackend.dequantizedMPS : value
        linearBackend = sanitized
        defaults.set(sanitized.rawValue, forKey: Keys.linearBackend)
        clearPresetMarker()
    }

    /// P9 (runbook §14): applies a named preset by setting every underlying
    /// optimization control to the preset's combination. The preset is
    /// recorded (and persisted) as the active preset so a relaunch restores
    /// it. After applying, the individual controls remain fully adjustable;
    /// adjusting any single control clears the marker (Task 9) so Diagnostics
    /// shows "Custom" rather than a stale preset name.
    func setPreset(_ preset: InferencePreset) {
        let config = preset.makeConfig()
        linearTileRows = config.linearTileRows
        attentionTileRows = config.attentionTileRows
        directLinearMPSIO = config.directLinearMPSIO
        pingPongWeightStreaming = config.pingPongWeightStreaming
        numericalMonitoring = config.numericalMonitoring
        fusedNormModulation = config.fusedNormModulation
        fusedMLPActivation = config.fusedMLPActivation
        stridedTokenMajorAttention = config.stridedTokenMajorAttention
        crossKVCache = config.crossKVCache
        noCopyWeightSource = config.noCopyWeightSource
        attentionBackend = config.attentionBackend
        linearBackend = config.linearBackend
        activePreset = preset
        // Persist every control so a relaunch restores the exact combination.
        defaults.set(config.linearTileRows, forKey: Keys.linearTileRows)
        defaults.set(config.attentionTileRows, forKey: Keys.attentionTileRows)
        defaults.set(config.directLinearMPSIO, forKey: Keys.directLinearMPSIO)
        defaults.set(config.pingPongWeightStreaming, forKey: Keys.pingPongWeightStreaming)
        defaults.set(config.numericalMonitoring, forKey: Keys.numericalMonitoring)
        defaults.set(config.fusedNormModulation, forKey: Keys.fusedNormModulation)
        defaults.set(config.fusedMLPActivation, forKey: Keys.fusedMLPActivation)
        defaults.set(config.stridedTokenMajorAttention, forKey: Keys.stridedTokenMajorAttention)
        defaults.set(config.crossKVCache, forKey: Keys.crossKVCache)
        defaults.set(config.noCopyWeightSource, forKey: Keys.noCopyWeightSource)
        defaults.set(config.attentionBackend.rawValue, forKey: Keys.attentionBackend)
        defaults.set(config.linearBackend.rawValue, forKey: Keys.linearBackend)
        defaults.set(preset.rawValue, forKey: Keys.activePreset)
    }

    /// Applies the `.baseline` preset (identical to `resetToBaseline`) and
    /// clears the active-preset marker, restoring the exact default
    /// configuration.
    func resetToPresetBaseline() {
        setPreset(.baseline)
    }

    /// Immutable snapshot of the current settings, used by a single
    /// generation. Never mutated mid-run.
    var snapshot: InferenceOptimizationConfig {
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
            linearBackend: linearBackend
        )
    }

    /// Restores exactly `InferenceOptimizationConfig.currentBaseline` and
    /// persists it.
    func resetToBaseline() {
        let baseline = InferenceOptimizationConfig.currentBaseline
        linearTileRows = baseline.linearTileRows
        attentionTileRows = baseline.attentionTileRows
        directLinearMPSIO = baseline.directLinearMPSIO
        pingPongWeightStreaming = baseline.pingPongWeightStreaming
        numericalMonitoring = baseline.numericalMonitoring
        fusedNormModulation = baseline.fusedNormModulation
        fusedMLPActivation = baseline.fusedMLPActivation
        stridedTokenMajorAttention = baseline.stridedTokenMajorAttention
        crossKVCache = baseline.crossKVCache
        noCopyWeightSource = baseline.noCopyWeightSource
        attentionBackend = baseline.attentionBackend
        linearBackend = baseline.linearBackend
        defaults.set(baseline.linearTileRows, forKey: Keys.linearTileRows)
        defaults.set(baseline.attentionTileRows, forKey: Keys.attentionTileRows)
        defaults.set(baseline.directLinearMPSIO, forKey: Keys.directLinearMPSIO)
        defaults.set(baseline.pingPongWeightStreaming, forKey: Keys.pingPongWeightStreaming)
        defaults.set(baseline.numericalMonitoring, forKey: Keys.numericalMonitoring)
        defaults.set(baseline.fusedNormModulation, forKey: Keys.fusedNormModulation)
        defaults.set(baseline.fusedMLPActivation, forKey: Keys.fusedMLPActivation)
        defaults.set(baseline.stridedTokenMajorAttention, forKey: Keys.stridedTokenMajorAttention)
        defaults.set(baseline.crossKVCache, forKey: Keys.crossKVCache)
        defaults.set(baseline.noCopyWeightSource, forKey: Keys.noCopyWeightSource)
        defaults.set(baseline.attentionBackend.rawValue, forKey: Keys.attentionBackend)
        defaults.set(baseline.linearBackend.rawValue, forKey: Keys.linearBackend)
        // Manual reset: semantically a "Custom" configuration — clear the
        // preset marker and its persisted key (same path every individual
        // setter uses).
        clearPresetMarker()
    }

    // MARK: - Loading

    private static func loadTileRows(from defaults: UserDefaults, key: String, baseline: Int) -> Int {
        guard let raw = defaults.object(forKey: key) as? Int else { return baseline }
        return InferenceOptimizationConfig.sanitizedTileRows(raw)
    }

    private static func loadAttentionBackend(from defaults: UserDefaults,
                                             baseline: DiTAttentionBackend) -> DiTAttentionBackend {
        guard let raw = defaults.string(forKey: Keys.attentionBackend),
              let value = DiTAttentionBackend(rawValue: raw) else { return baseline }
        return value
    }

    /// QUARANTINED (Task 4): the P8 direct packed QGEMM backends
    /// (`.directQuantized`, `.hybrid`) measured ~10x SLOWER than
    /// `.dequantizedMPS` on the physical A12 device — a measured PERFORMANCE
    /// regression, NOT a proven correctness failure (the research kernel
    /// stays intact and directly testable via `LinearExecutor`). Normal
    /// device settings must never select them while the optimization search
    /// runs. Persisted bad values are migrated back to `.dequantizedMPS` on
    /// load so they can never reach the executors.
    static func loadLinearBackend(from defaults: UserDefaults,
                                  baseline: DiTLinearBackend) -> DiTLinearBackend {
        guard let raw = defaults.string(forKey: Keys.linearBackend),
              let value = DiTLinearBackend(rawValue: raw) else { return baseline }
        // QUARANTINED (Task 4): a persisted direct/hybrid selection (e.g.
        // from a pre-quarantine build) is migrated back to the baseline.
        // The sanitized value is also re-persisted so the next launch reads
        // a clean store.
        if value.isQuarantined {
            defaults.set(baseline.rawValue, forKey: Keys.linearBackend)
            return baseline
        }
        return value
    }

    /// DISABLED (Task 5): the P6 mmap no-copy weight source is blocked for
    /// normal production/device settings after a physical A12 run hit a real
    /// GPU page fault (`kIOGPUCommandBufferCallbackErrorPageFault`) while
    /// no-copy bytes were being served. A persisted `true` (e.g. from a
    /// pre-disable build) is migrated back to `false` on load, and the
    /// sanitized value is re-persisted so the store is clean. This is
    /// correctness/safety hardening, NOT a proof of the historical root
    /// cause — the research implementation stays intact for an isolated
    /// hardware test.
    static func loadNoCopyWeightSource(from defaults: UserDefaults,
                                       baseline: Bool) -> Bool {
        guard let persisted = defaults.object(forKey: Keys.noCopyWeightSource) as? Bool else {
            return baseline
        }
        if persisted {
            defaults.set(false, forKey: Keys.noCopyWeightSource)
            return false
        }
        return persisted
    }

    /// Loads the persisted active preset. An invalid/corrupt value sanitizes
    /// to `nil` (no preset marker) so it can never reach the executors; the
    /// underlying controls were each sanitized independently already.
    private static func loadPreset(from defaults: UserDefaults) -> InferencePreset? {
        guard let raw = defaults.string(forKey: Keys.activePreset),
              let value = InferencePreset(rawValue: raw) else { return nil }
        return value
    }
}
