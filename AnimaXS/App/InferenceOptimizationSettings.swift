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
        static let ditPackVariant = "inference.ditPackVariant"
    }

    private let defaults: UserDefaults

    @Published private(set) var linearTileRows: Int
    @Published private(set) var attentionTileRows: Int
    @Published private(set) var directLinearMPSIO: Bool
    @Published private(set) var pingPongWeightStreaming: Bool
    @Published private(set) var numericalMonitoring: Bool
    @Published private(set) var ditPackVariant: DiTPackVariant

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
        if let raw = defaults.string(forKey: Keys.ditPackVariant),
           let variant = DiTPackVariant(rawValue: raw) {
            ditPackVariant = variant
        } else {
            ditPackVariant = baseline.ditPackVariant
        }
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

    func setDiTPackVariant(_ value: DiTPackVariant) {
        ditPackVariant = value
        defaults.set(value.rawValue, forKey: Keys.ditPackVariant)
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
            ditPackVariant: ditPackVariant
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
        ditPackVariant = baseline.ditPackVariant
        defaults.set(baseline.linearTileRows, forKey: Keys.linearTileRows)
        defaults.set(baseline.attentionTileRows, forKey: Keys.attentionTileRows)
        defaults.set(baseline.directLinearMPSIO, forKey: Keys.directLinearMPSIO)
        defaults.set(baseline.pingPongWeightStreaming, forKey: Keys.pingPongWeightStreaming)
        defaults.set(baseline.numericalMonitoring, forKey: Keys.numericalMonitoring)
        defaults.set(baseline.ditPackVariant.rawValue, forKey: Keys.ditPackVariant)
    }

    // MARK: - Loading

    private static func loadTileRows(from defaults: UserDefaults, key: String, baseline: Int) -> Int {
        guard let raw = defaults.object(forKey: key) as? Int else { return baseline }
        return InferenceOptimizationConfig.sanitizedTileRows(raw)
    }
}
