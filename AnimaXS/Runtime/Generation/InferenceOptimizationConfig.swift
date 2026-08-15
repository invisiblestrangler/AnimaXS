import Foundation

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
/// - Fused LayerNorm+AdaLN+to-half: OFF (separate norm/modulated fp32 buffers)
/// - Fused MLP in-place half GELU: OFF (fp32 hidden GELU intermediate)
///
/// The DiT pack is whichever variant (W4 or W8-v2) was imported into the
/// `.dit` slot; it is resolved by `ModelStore` and is not a config choice.
struct InferenceOptimizationConfig: Equatable {
    static let allowedTileRows = [128, 256, 512, 1024]

    var linearTileRows: Int
    var attentionTileRows: Int
    var directLinearMPSIO: Bool
    var pingPongWeightStreaming: Bool
    var numericalMonitoring: Bool
    /// P3-A: fuse LayerNorm + AdaLN modulation + compute-boundary + fp16
    /// conversion into ONE kernel, eliminating the `dit.norm.f32` and
    /// `dit.modulated.f32` (16 MiB total) intermediates. False keeps the
    /// legacy three-pass path exactly for A/B.
    var fusedNormModulation: Bool
    /// P3-B: evaluate GELU in-place on the fp16 MLP hidden activations
    /// (fp32 register arithmetic, optional BF16 rounding), eliminating the
    /// ~32 MiB `dit.hidden.f32` intermediate and its four passes. False
    /// keeps the legacy half→float→GELU→round→to-half path.
    var fusedMLPActivation: Bool
    /// P4: remove the DiT attention token↔head transposes (3 in + 1 out per
    /// block) by feeding MPS strided token-major matrix views directly to the
    /// attention executor. False keeps the legacy head-major transpose path
    /// exactly for A/B. Only affects DiT attention; Qwen/VAE/adapter attention
    /// is always head-major.
    var stridedTokenMajorAttention: Bool

    static let currentBaseline = InferenceOptimizationConfig(
        linearTileRows: 128,
        attentionTileRows: 128,
        directLinearMPSIO: false,
        pingPongWeightStreaming: true,
        numericalMonitoring: true,
        fusedNormModulation: false,
        fusedMLPActivation: false,
        stridedTokenMajorAttention: false
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
    ///
    /// The DiT slot holds exactly one verified pack (W4 or W8-v2) at a time,
    /// so there is no separate experimental state that a checkpoint could
    /// collide with: checkpoint identity is the stable production hash set.
    var checkpointingEnabled: Bool {
        true
    }
}
