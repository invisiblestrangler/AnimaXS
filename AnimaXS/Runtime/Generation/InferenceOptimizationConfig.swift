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
    /// A12/H11 hybrid backend proven by the V12-V14 device harness. Large W8
    /// DiT projection GEMMs (self QKV/O, cross Q/K/V/O, MLP up/down) execute
    /// on ANE; AdaLN, learned RMSNorm, RoPE, attention, GELU and residual math
    /// remain on the source-faithful Metal path. This is the original
    /// eight-model-per-block production control for the multiprocedure A/B.
    case aneHybridW8
    /// Stage2J/K production candidate: the same projection math, but each DiT
    /// block is represented by ONE `_ANEInMemoryModel` exposing ten
    /// single-output procedures. The scheduler uses the device-measured
    /// 6-pinned + 2-streaming residency policy while all nonlinear/attention
    /// math remains on the identical Metal path.
    case aneMultiProcW8

    /// QUARANTINED (device stabilization — Task 4): true for the P8 direct
    /// packed QGEMM backends (`.directQuantized`, `.hybrid`) that measured
    /// ~10x SLOWER than `.dequantizedMPS` on the physical A12 device
    /// (structural kernel issues — TM=8 against M=1024, repeated packed-weight
    /// revisits, scalar FMAs, many barriers). This is a measured PERFORMANCE
    /// regression, NOT a proven correctness failure: the research kernel
    /// (`LinearExecutor` / `AnimaKernels.metal`) stays intact and directly
    /// testable, but these backends must never be selected by normal device
    /// settings, presets, or combined candidates while the optimization
    /// search runs.
    var isQuarantined: Bool {
        self == .directQuantized || self == .hybrid
    }

    var isANEW8: Bool {
        self == .aneHybridW8 || self == .aneMultiProcW8
    }
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
///
/// Compatibility: `blockingReason(for:numerics:)` is the single central
/// validator for a resolved production configuration (Task 9). It returns a
/// user-visible blocking reason for any configuration that must NOT reach the
/// executors — P6 no-copy, the quarantined P8 linear backends, and explicit
/// experimental BF16 numerics combined with a strided token-major attention
/// layout (the `AttentionExecutor` constraint). It is DEFENSE-IN-DEPTH on top
/// of the Task 4/5 settings-layer sanitization: it never mutates a
/// user-selected config, it only reports a reason so Generate is blocked and
/// the user sees why.
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
    ///
    /// DISABLED (device stabilization — Task 5): normal production/device
    /// configuration can no longer set this to `true`. A physical A12 run
    /// hit a real GPU page fault (`kIOGPUCommandBufferCallbackErrorPageFault`)
    /// while no-copy bytes were being served, so `InferenceOptimizationSettings`
    /// migrates persisted `true` → `false`, `setNoCopyWeightSource(true)` is
    /// normalized to `false`, and the `noCopyCandidate` / `allCandidate`
    /// presets force it back to `false`. This is correctness/safety
    /// hardening, NOT a proof that the no-copy path caused the historical
    /// page fault. The research implementation (`WeightNoCopyPolicy.makeAlias`
    /// / `WeightStreamer`) stays fully intact so an isolated hardware test
    /// can re-enable the path later.
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
    ///
    /// QUARANTINED (Task 4): `.directQuantized` and `.hybrid` are measured
    /// ~10x slower than `.dequantizedMPS` on device (A12). Normal device
    /// settings, presets, and persisted values are sanitized to
    /// `.dequantizedMPS`; the research kernel remains directly testable via
    /// `LinearExecutor`.
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
        let clamped = min(1024, max(128, value))
        return allowedTileRows.min(by: { abs($0 - clamped) < abs($1 - clamped) }) ?? 128
    }

    // MARK: - Central compatibility validator (Task 9)

    static let noCopyBlockingReason = "P6 mmap no-copy weight source is disabled for device use: a physical A12 run hit a real GPU page fault (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy bytes were being served. Correctness/safety hardening, not a proof of the historical root cause."
    static let linearBackendBlockingReason = "P8 direct QGEMM (.directQuantized / .hybrid) is quarantined for device use: it measured ~10x slower than dequantized MPS on the A12 device (performance regression, not a proven correctness failure)."
    static let aneNativePackRequiredReason = "ANE W8 requires the ANE-native W8 pack (w8-ane-v1). Import that pack before using this backend."
    static let aneNativeBackendRequiredReason = "The ANE-native W8 pack can only run with an ANE W8 backend because its projection tensors use ANE row-wise quantization."
    static let bf16StridedAttentionBlockingReason = "Experimental BF16 numerics (w8BF16Experimental / bf16Compute) are not supported with strided token-major attention: the BF16 boundary round would corrupt the strided layout. Select legacy numerics or disable the strided token-major attention layout."

    var resolvesToStridedTokenMajorAttention: Bool {
        switch attentionBackend {
        case .legacyHeadMajorMPS:
            return false
        case .stridedTokenMajorMPS:
            return stridedTokenMajorAttention
        case .streamingMPS, .metalFlash:
            return true
        }
    }

    static func blockingReason(
        for config: InferenceOptimizationConfig,
        numerics: DiTNumericsPolicy = .w4Legacy,
        ditVariantID: String? = nil
    ) -> String? {
        if config.noCopyWeightSource {
            return noCopyBlockingReason
        }
        if let ditVariantID {
            if config.linearBackend.isANEW8, ditVariantID != "w8-ane-v1" {
                return aneNativePackRequiredReason
            }
            if ditVariantID == "w8-ane-v1", !config.linearBackend.isANEW8 {
                return aneNativeBackendRequiredReason
            }
        }
        if config.linearBackend.isQuarantined {
            return linearBackendBlockingReason
        }
        if numerics == .w8BF16Experimental {
            let (activation, attention) = DiffusionSampler.resolvedNumerics(for: numerics)
            if activation == .bf16Compute || attention == .bf16Compute {
                if config.resolvesToStridedTokenMajorAttention {
                    return bf16StridedAttentionBlockingReason
                }
            }
        }
        return nil
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
    case baseline
    case current1024Control
    case fusedTraffic
    case stridedMPS
    case stridedMPSKV
    case noCopyCandidate
    case streamingMPSCandidate
    case metalFlashCandidate
    case directQGEMMCandidate
    case allCandidate

    var containsQuarantinedLinearBackend: Bool {
        self == .directQGEMMCandidate || self == .allCandidate
    }

    var containsDisabledNoCopy: Bool {
        self == .noCopyCandidate || self == .allCandidate
    }

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
            c.noCopyWeightSource = false
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
            c.linearBackend = .dequantizedMPS
        case .allCandidate:
            c = InferencePreset.stridedMPSKV.makeConfig()
            c.noCopyWeightSource = false
            c.attentionBackend = .streamingMPS
            c.linearBackend = .dequantizedMPS
        }
        return c
    }
}
