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
    /// individual control afterwards does NOT clear this marker — the picker
    /// simply continues to show the last preset applied.
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
        noCopyWeightSource = defaults.object(forKey: Keys.noCopyWeightSource) as? Bool
            ?? baseline.noCopyWeightSource
        attentionBackend = InferenceOptimizationSettings.loadAttentionBackend(
            from: defaults, baseline: baseline.attentionBackend)
        linearBackend = InferenceOptimizationSettings.loadLinearBackend(
            from: defaults, baseline: baseline.linearBackend)
        activePreset = InferenceOptimizationSettings.loadPreset(
            from: defaults)
    }

    // MARK: - Mutations (validate before persist)

    func setLinearTileRows(_ value: Int) {
        linearTileRows = InferenceOptimizationConfig.sanitizedTileRows(value)
        defaults.set(linearTileRows, forKey: Keys.linearTileRows)
    }

    func setAttentionTileRows(_ value: Int) {
        attentionTileRows = InferenceOptimizationConfig.sanitizedTileRows(value)
        defaults.set(attentionTileRows, forKey: Keys.attentionTileRows)
    }

    func setDirectLinearMPSIO(_ value: Bool) {
        directLinearMPSIO = value
        defaults.set(value, forKey: Keys.directLinearMPSIO)
    }

    func setPingPongWeightStreaming(_ value: Bool) {
        pingPongWeightStreaming = value
        defaults.set(value, forKey: Keys.pingPongWeightStreaming)
    }

    func setNumericalMonitoring(_ value: Bool) {
        numericalMonitoring = value
        defaults.set(value, forKey: Keys.numericalMonitoring)
    }

    func setFusedNormModulation(_ value: Bool) {
        fusedNormModulation = value
        defaults.set(value, forKey: Keys.fusedNormModulation)
    }

    func setFusedMLPActivation(_ value: Bool) {
        fusedMLPActivation = value
        defaults.set(value, forKey: Keys.fusedMLPActivation)
    }

    func setStridedTokenMajorAttention(_ value: Bool) {
        stridedTokenMajorAttention = value
        defaults.set(value, forKey: Keys.stridedTokenMajorAttention)
    }

    func setCrossKVCache(_ value: Bool) {
        crossKVCache = value
        defaults.set(value, forKey: Keys.crossKVCache)
    }

    func setNoCopyWeightSource(_ value: Bool) {
        noCopyWeightSource = value
        defaults.set(value, forKey: Keys.noCopyWeightSource)
    }

    func setAttentionBackend(_ value: DiTAttentionBackend) {
        attentionBackend = value
        defaults.set(value.rawValue, forKey: Keys.attentionBackend)
    }

    func setLinearBackend(_ value: DiTLinearBackend) {
        linearBackend = value
        defaults.set(value.rawValue, forKey: Keys.linearBackend)
    }

    /// P9 (runbook §14): applies a named preset by setting every underlying
    /// optimization control to the preset's combination. The preset is
    /// recorded (and persisted) as the active preset so a relaunch restores
    /// it. After applying, the individual controls remain fully adjustable;
    /// adjusting any single control does NOT silently clear the preset label.
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
        activePreset = nil
        defaults.removeObject(forKey: Keys.activePreset)
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

    private static func loadLinearBackend(from defaults: UserDefaults,
                                          baseline: DiTLinearBackend) -> DiTLinearBackend {
        guard let raw = defaults.string(forKey: Keys.linearBackend),
              let value = DiTLinearBackend(rawValue: raw) else { return baseline }
        return value
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
