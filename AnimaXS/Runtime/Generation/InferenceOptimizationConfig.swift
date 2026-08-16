import Foundation

/// P8: runtime-selectable DiT linear (GEMM) backends (runbook §13).
/// Defaults to `.dequantizedMPS` — the effective current behavior — so the
/// known-good W4 path is byte-for-byte unchanged. The physical device
/// selects the winner later; `currentBaseline` must keep the legacy path.
enum DiTLinearBackend: String, Codable, CaseIterable {
    /// Legacy path: dequantize the packed [N,K] weight ONCE into a reusable
    /// fp16 scratch buffer, then tile input rows through MPS. The P0-P7
    /// known-good behavior.
    case dequantizedMPS
    /// P8: direct packed W4/W8 GEMM (qgemm_8x8x64 / qgemm_8x16x64 tile
    /// profiles). The W tile for one K group is decoded into threadgroup
    /// memory and consumed immediately — no full [N,K] fp16 weight scratch.
    /// M=1 matvec keeps the existing direct matvec kernels (never routed
    /// here).
    case directQuantized
    /// P8: family-based hybrid dispatch. MLP up/down matrices (the largest
    /// decompressed-weight scratch traffic) run the direct QGEMM; attention
    /// projections and everything else keep the dequantized-MPS path until
    /// A12 data proves otherwise.
    case hybrid
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
    /// P5: cache the invariant cross-attention K/V for the whole generation.
    /// Cross context is fixed for a generation, so after the first executed
    /// step each DiT block's cross K/V (post-projection, post-static-boundary,
    /// post-K-normalization) are reused from a per-generation `CrossKVCache`
    /// instead of being re-projected every step. EXACT reuse — no
    /// approximation; Q stays dynamic and is never cached. False keeps the
    /// legacy per-step cross K/V projection path exactly for A/B.
    var crossKVCache: Bool
    /// P6 (EXPERIMENTAL): hand Metal an `MTLBuffer` that ALIASES the already
    /// mmap'd pack region via `bytesNoCopy` instead of memcpy'ing the range
    /// into the slot ring, so the GPU reads weights directly from the file
    /// mapping (no CPU weight copy). Only applied when the range's file
    /// offset is 4096-byte page-aligned AND the device accepts the alias;
    /// every other range falls back to the exact copied path. The copied
    /// path is byte-for-byte unchanged, so this toggle only ever adds the
    /// no-copy fast path. False keeps the current memcpy path exactly for
    /// A/B. EXPERIMENTAL — never a production default (device decides later).
    var noCopyWeightSource: Bool
    /// P7: DiT attention backend selector. Defaults to `.legacyHeadMajorMPS`
    /// — the effective current behavior — so the known-good W4 path is
    /// byte-for-byte unchanged. Precedence (documented for the state file):
    /// - `.legacyHeadMajorMPS` → legacy transposed head-major MPS path
    ///   (ignores `stridedTokenMajorAttention`).
    /// - `.stridedTokenMajorMPS` → P4 strided token-major MPS path, but
    ///   HONORS the `stridedTokenMajorAttention` boolean: when the bool is
    ///   OFF this case selects the legacy head-major path, preserving the
    ///   P4 semantics exactly.
    /// - `.streamingMPS` / `.metalFlash` → the P7 backends REQUIRE the
    ///   token-major layout; DiTBlockExecutor throws unless
    ///   `stridedTokenMajorAttention` is ON (never a silent fallback).
    /// Only DiT attention is affected; Qwen/VAE/adapter attention always
    /// runs the legacy head-major path regardless of this selector.
    var attentionBackend: DiTAttentionBackend
    /// P8: DiT linear (GEMM) backend selector. Defaults to `.dequantizedMPS`
    /// — the effective current behavior — so the known-good W4 path is
    /// byte-for-byte unchanged. `.directQuantized` runs every non-M=1 DiT
    /// linear through the direct packed QGEMM; `.hybrid` routes only the
    /// MLP up/down matrices (largest decompressed-weight scratch traffic)
    /// through QGEMM and keeps attention projections on MPS until A12 data
    /// proves otherwise. M=1 modulation matvecs are never affected.
    var linearBackend: DiTLinearBackend

    static let currentBaseline = InferenceOptimizationConfig(
        linearTileRows: 128,
        attentionTileRows: 128,
        directLinearMPSIO: false,
        pingPongWeightStreaming: true,
        numericalMonitoring: true,
        fusedNormModulation: false,
        fusedMLPActivation: false,
        stridedTokenMajorAttention: false,
        crossKVCache: false,
        noCopyWeightSource: false,
        attentionBackend: .legacyHeadMajorMPS,
        linearBackend: .dequantizedMPS
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
}

/// P9 (runbook §14): named combinations of the existing per-toggle/per-backend
/// optimization controls, so the user can exercise realistic "combined" configs
/// (and, on the physical XS Max, the full §17 benchmark matrix) without
/// hand-setting every toggle. Each preset maps to a specific
/// `InferenceOptimizationConfig`. The individual A/B controls are NOT removed;
/// a preset is just a convenience that sets every underlying field at once.
///
/// `allCandidate` is not automatically "best" — it is simply one test
/// configuration. No preset claims physical-A12 victory; the user must measure
/// on the device (§17) before a `recommendedA12` preset can be added.
enum InferencePreset: String, CaseIterable {
    /// Everything at defaults: legacy W4-known-good path (runbook §17 #1 control
    /// with 128-tile rows and direct MPS I/O / ping-pong at their defaults).
    case baseline
    /// The §17 first-pass control: linear/attention 1024, direct MPS I/O on,
    /// ping-pong on, all new optimizations off.
    case current1024Control
    /// Control + fused LayerNorm+AdaLN+to-half + fused in-place MLP GELU.
    case fusedTraffic
    /// Fused + strided token-major MPS attention.
    case stridedMPS
    /// Strided + cross-attention K/V cache.
    case stridedMPSKV
    /// KV + mmap no-copy weight source (experimental).
    case noCopyCandidate
    /// Fused + KV + streaming MPS attention backend.
    case streamingMPSCandidate
    /// Fused + KV + pure-Metal Flash attention backend.
    case metalFlashCandidate
    /// Best attention candidate + direct packed QGEMM for the MLP only
    /// (`.hybrid` linear backend) — the §17 preset 8 "MLP QGEMM hybrid".
    case directQGEMMCandidate
    /// All currently-winning components combined — one test configuration,
    /// NOT an automatic "best".
    case allCandidate

    /// Human-friendly label for UI selection.
    var label: String {
        switch self {
        case .baseline: return "Baseline"
        case .current1024Control: return "1024 Control"
        case .fusedTraffic: return "Fused traffic"
        case .stridedMPS: return "Strided MPS"
        case .stridedMPSKV: return "Strided MPS + KV"
        case .noCopyCandidate: return "No-copy candidate"
        case .streamingMPSCandidate: return "Streaming MPS"
        case .metalFlashCandidate: return "Metal Flash"
        case .directQGEMMCandidate: return "Direct QGEMM"
        case .allCandidate: return "All candidates"
        }
    }

    /// Builds the `InferenceOptimizationConfig` this preset names, starting
    /// from `currentBaseline` and layering the intended combination on top.
    /// Immutable snapshot semantics are unchanged: callers keep the returned
    /// value for the whole generation and never mutate it mid-run.
    func makeConfig() -> InferenceOptimizationConfig {
        var c = InferenceOptimizationConfig.currentBaseline
        switch self {
        case .baseline:
            break
        case .current1024Control:
            c.linearTileRows = 1024
            c.attentionTileRows = 1024
            c.directLinearMPSIO = true
            c.pingPongWeightStreaming = true
        case .fusedTraffic:
            c = InferencePreset.current1024Control.makeConfig()
            c.fusedNormModulation = true
            c.fusedMLPActivation = true
        case .stridedMPS:
            c = InferencePreset.fusedTraffic.makeConfig()
            c.stridedTokenMajorAttention = true
            c.attentionBackend = .stridedTokenMajorMPS
        case .stridedMPSKV:
            c = InferencePreset.stridedMPS.makeConfig()
            c.crossKVCache = true
        case .noCopyCandidate:
            c = InferencePreset.stridedMPSKV.makeConfig()
            c.noCopyWeightSource = true
        case .streamingMPSCandidate:
            c = InferencePreset.fusedTraffic.makeConfig()
            c.crossKVCache = true
            c.stridedTokenMajorAttention = true
            c.attentionBackend = .streamingMPS
        case .metalFlashCandidate:
            c = InferencePreset.fusedTraffic.makeConfig()
            c.crossKVCache = true
            c.stridedTokenMajorAttention = true
            c.attentionBackend = .metalFlash
        case .directQGEMMCandidate:
            c = InferencePreset.stridedMPSKV.makeConfig()
            c.linearBackend = .hybrid
        case .allCandidate:
            c = InferencePreset.stridedMPSKV.makeConfig()
            c.noCopyWeightSource = true
            c.attentionBackend = .streamingMPS
            c.linearBackend = .hybrid
        }
        return c
    }
}
