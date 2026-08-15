import Foundation

/// DiT weight-pack variant used for one generation run.
///
/// Only `.productionW4` is part of the normal three-pack production topology.
/// `.experimentalW8V2` is a Diagnostics-only experiment that substitutes a
/// separately imported, verified W8 pack for the DiT URL in a run-specific
/// resolved-model snapshot. It deliberately lives OUTSIDE `ModelManifest.entries`.
enum DiTPackVariant: String, CaseIterable, Codable, Identifiable {
    case productionW4
    case experimentalW8V2

    var id: String { rawValue }
}

/// Immutable, value-semantic runtime configuration captured for a single
/// generation when Generate is pressed.
///
/// A generation must never observe mid-run toggle changes, so the mutable UI
/// settings (`InferenceOptimizationSettings`) are converted into this snapshot
/// at generation start. No `ObservableObject` is ever passed into Metal/runtime
/// code — only this plain struct (or a derived control value) is threaded down.
///
/// `currentBaseline` MUST reproduce current HEAD behavior exactly:
/// - Linear tile rows: 128
/// - Attention tile rows: 128
/// - Direct MPS linear I/O: OFF (per-tile copy path)
/// - Ping-pong weight streaming: ON (existing two-slot behavior)
/// - Numerical monitor: ON (existing production probes)
/// - DiT: production W4
struct InferenceOptimizationConfig: Equatable {
    static let allowedTileRows = [128, 256, 512, 1024]

    var linearTileRows: Int
    var attentionTileRows: Int
    var directLinearMPSIO: Bool
    var pingPongWeightStreaming: Bool
    var numericalMonitoring: Bool
    var ditPackVariant: DiTPackVariant

    static let currentBaseline = InferenceOptimizationConfig(
        linearTileRows: 128,
        attentionTileRows: 128,
        directLinearMPSIO: false,
        pingPongWeightStreaming: true,
        numericalMonitoring: true,
        ditPackVariant: .productionW4
    )

    /// Sanitizes a tile-row value down to the nearest allowed value (or the
    /// baseline 128 when it does not map to an allowed setting). Used when
    /// persisted/corrupt values must never reach the executors.
    static func sanitizedTileRows(_ value: Int) -> Int {
        if allowedTileRows.contains(value) { return value }
        // Clamp to the allowed range, then round to the nearest allowed row.
        let clamped = min(1024, max(128, value))
        return allowedTileRows.min(by: { abs($0 - clamped) < abs($1 - clamped) }) ?? 128
    }

    /// True when checkpoint persistence/resume is meaningful for this run.
    /// Experimental W8 has checkpointing disabled (its production-W4 hash set
    /// does not describe the W8 pack, and a W8 checkpoint must never resurrect
    /// unrelated production resume state).
    var checkpointingEnabled: Bool {
        ditPackVariant == .productionW4
    }
}
