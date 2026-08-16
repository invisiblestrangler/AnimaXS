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
        // Clamp to the allowed range, then round to the nearest allowed row.
        let clamped = min(1024, max(128, value))
        return allowedTileRows.min(by: { abs($0 - clamped) < abs($1 - clamped) }) ?? 128
    }

    // MARK: - Central compatibility validator (Task 9)

    /// DISABLED (Task 5) — P6 mmap no-copy weight source. A physical A12 run
    /// hit a real GPU page fault (`kIOGPUCommandBufferCallbackErrorPageFault`)
    /// while no-copy bytes were being served, so production configuration can
    /// never run the no-copy path. The Task 5 settings layer already
    /// normalizes `true` → `false` and never persists it; this reason is the
    /// central-validator wording for any config that still carries `true`
    /// (defense-in-depth — a normal device user cannot currently produce one).
    static let noCopyBlockingReason = "P6 mmap no-copy weight source is disabled for device use: a physical A12 run hit a real GPU page fault (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy bytes were being served. Correctness/safety hardening, not a proof of the historical root cause."

    /// QUARANTINED (Task 4) — P8 direct packed QGEMM linear backends
    /// (`.directQuantized` / `.hybrid`). They measured ~10x SLOWER than
    /// `.dequantizedMPS` on the physical A12 device — a measured PERFORMANCE
    /// regression, NOT a proven correctness failure (the research kernel
    /// stays intact and directly testable via `LinearExecutor`). The Task 4
    /// settings layer already normalizes them to `.dequantizedMPS`; this
    /// reason is the central-validator wording for any config that still
    /// carries one (defense-in-depth).
    static let linearBackendBlockingReason = "P8 direct QGEMM (.directQuantized / .hybrid) is quarantined for device use: it measured ~10x slower than dequantized MPS on the A12 device (performance regression, not a proven correctness failure)."

    /// EXPERIMENTAL BF16 numerics + strided token-major attention layout.
    /// `AttentionExecutor` throws "P4 strided token-major attention does not
    /// support bf16Compute numerics" for this exact combination — the BF16
    /// boundary round is contiguous in the legacy layout but would corrupt
    /// the strided token-major layout, so it refuses loudly instead of
    /// silently producing wrong results. Production W8-v2 resolves to
    /// `w8LegacyStabilized` → legacy/legacy numerics (compatible with strided
    /// attention); only an EXPLICIT experimental BF16 policy triggers this.
    static let bf16StridedAttentionBlockingReason = "Experimental BF16 numerics (w8BF16Experimental / bf16Compute) are not supported with strided token-major attention: the BF16 boundary round would corrupt the strided layout. Select legacy numerics or disable the strided token-major attention layout."

    /// True when the resolved DiT attention layout is strided token-major.
    /// Mirrors `DiTBlockExecutor`'s backend selection exactly:
    /// `.legacyHeadMajorMPS` never uses the strided layout; `.stridedTokenMajorMPS`
    /// honors the `stridedTokenMajorAttention` boolean; `.streamingMPS` /
    /// `.metalFlash` REQUIRE the token-major layout (they throw unless the
    /// toggle is ON). Qwen/VAE/adapter attention is always head-major and is
    /// never affected.
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

    /// The single central compatibility validator for a RESOLVED production
    /// configuration (Task 9). Returns a user-visible blocking reason when
    /// the configuration must NOT reach the executors, or `nil` when it is
    /// compatible.
    ///
    /// It blocks:
    /// 1. `noCopyWeightSource == true` (P6 disabled — A12 GPU page fault).
    /// 2. `linearBackend` other than `.dequantizedMPS` (P8 quarantined —
    ///    ~10x A12 regression).
    /// 3. An explicit experimental BF16 numerical policy — resolved
    ///    attention/activation numerics `bf16Compute` — combined with a
    ///    strided token-major attention layout (the `AttentionExecutor`
    ///    constraint).
    ///
    /// The BF16 check is based on the ACTUAL resolved numerical policy
    /// (`DiffusionSampler.resolvedNumerics(for:)`), NOT the pack name: a
    /// production W8 pack resolves to `w8LegacyStabilized` → legacy/legacy
    /// numerics, which IS compatible with strided attention and must not be
    /// blocked. `numerics` defaults to the W4 pack policy (the most common
    /// resolved case, legacy/legacy).
    ///
    /// The validator is read-only: it NEVER mutates a user-selected config.
    /// Persisted bad settings are migrated at app initialization (Tasks 4/5);
    /// an explicit current incompatible selection is surfaced as a blocking
    /// reason and left exactly as the user set it.
    static func blockingReason(
        for config: InferenceOptimizationConfig,
        numerics: DiTNumericsPolicy = .w4Legacy
    ) -> String? {
        // 1. P6 no-copy (Task 5): a physical A12 run hit a real GPU page
        // fault while no-copy bytes were being served.
        if config.noCopyWeightSource {
            return noCopyBlockingReason
        }
        // 2. P8 quarantined linear backends (Task 4): measured ~10x slower
        // than dequantized MPS on the A12 device.
        if config.linearBackend != .dequantizedMPS {
            return linearBackendBlockingReason
        }
        // 3. Experimental BF16 numerics + strided token-major attention:
        // AttentionExecutor refuses bf16Compute for the strided layout.
        // Only the explicit experimental policy resolves to BF16 numerics
        // (w8LegacyStabilized -> legacy/legacy, compatible with strided).
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
    /// KV + mmap no-copy weight source (experimental). DISABLED (Task 5):
    /// the no-copy part is forced back to `false` in `makeConfig()` — a
    /// physical A12 run hit a real GPU page fault
    /// (`kIOGPUCommandBufferCallbackErrorPageFault`) while no-copy bytes
    /// were being served. This is correctness/safety hardening, NOT a proof
    /// of the historical root cause; every other component of the
    /// combination still applies unchanged.
    case noCopyCandidate
    /// Fused + KV + streaming MPS attention backend.
    case streamingMPSCandidate
    /// Fused + KV + pure-Metal Flash attention backend.
    case metalFlashCandidate
    /// QUARANTINED (Task 4) — §17 preset 8 "MLP QGEMM hybrid". The QGEMM
    /// part of this preset is temporarily disabled: `.hybrid` measured ~10x
    /// SLOWER than `.dequantizedMPS` on the physical A12 device (a measured
    /// performance regression, NOT a proven correctness failure — the P8
    /// research kernel stays intact and testable). `makeConfig()` therefore
    /// FORCES `linearBackend` back to `.dequantizedMPS` so a device preset
    /// can never silently run the 10x-slower direct path; every other
    /// component of the combination still applies unchanged. The preset is
    /// kept selectable (with a visible note in Diagnostics) so the
    /// combination can be re-enabled instantly once the QGEMM kernel is
    /// structurally fixed and re-measured.
    case directQGEMMCandidate
    /// All currently-winning components combined — one test configuration,
    /// NOT an automatic "best". QUARANTINED (Task 4): like
    /// `directQGEMMCandidate`, the hybrid/QGEMM part is temporarily disabled
    /// (measured ~10x A12 regression vs `.dequantizedMPS`; performance, not
    /// correctness). `makeConfig()` keeps every other combined setting as-is
    /// EXCEPT `linearBackend` is forced back to `.dequantizedMPS`. DISABLED
    /// (Task 5): the P6 mmap no-copy part is also forced back to `false` —
    /// a physical A12 run hit a real GPU page fault
    /// (`kIOGPUCommandBufferCallbackErrorPageFault`) while no-copy bytes
    /// were being served (correctness/safety hardening, not a proof of the
    /// historical root cause).
    case allCandidate

    /// QUARANTINED (Task 4): true for presets whose QGEMM part is temporarily
    /// disabled (measured ~10x A12 regression vs `.dequantizedMPS` —
    /// performance, not correctness). They remain selectable: `makeConfig()`
    /// forces `linearBackend` to `.dequantizedMPS` while keeping every other
    /// component of the combination. The UI marks them with a visible warning
    /// so a device preset can never silently run the 10x-slower direct path.
    var containsQuarantinedLinearBackend: Bool {
        self == .directQGEMMCandidate || self == .allCandidate
    }

    /// DISABLED (Task 5): true for presets whose P6 mmap no-copy part is
    /// temporarily disabled — a physical A12 run hit a real GPU page fault
    /// (`kIOGPUCommandBufferCallbackErrorPageFault`) while no-copy bytes
    /// were being served. This is correctness/safety hardening, NOT a proof
    /// that the no-copy path caused the historical page fault. They remain
    /// selectable: `makeConfig()` forces `noCopyWeightSource` back to `false`
    /// while keeping every other component of the combination. The UI marks
    /// them so the disabled part is impossible to miss. The research
    /// implementation (`WeightNoCopyPolicy.makeAlias` / `WeightStreamer`)
    /// stays intact so an isolated hardware test can re-enable it later.
    var containsDisabledNoCopy: Bool {
        self == .noCopyCandidate || self == .allCandidate
    }

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
            // DISABLED (Task 5): the P6 mmap no-copy part is forced back to
            // `false` — a physical A12 run hit a real GPU page fault
            // (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy
            // bytes were being served. Correctness/safety hardening, not a
            // proof of the historical root cause. Every other component of
            // the combination still applies unchanged.
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
            // QUARANTINED (Task 4): the hybrid/QGEMM part is disabled — a
            // measured ~10x A12 regression vs .dequantizedMPS. Force the
            // backend back to the known-good path; never persist/run hybrid.
            c.linearBackend = .dequantizedMPS
        case .allCandidate:
            c = InferencePreset.stridedMPSKV.makeConfig()
            // DISABLED (Task 5): the P6 mmap no-copy part is forced back to
            // `false` — a physical A12 run hit a real GPU page fault
            // (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy
            // bytes were being served. Correctness/safety hardening, not a
            // proof of the historical root cause.
            c.noCopyWeightSource = false
            c.attentionBackend = .streamingMPS
            // QUARANTINED (Task 4): keep every other combined setting as-is,
            // EXCEPT the hybrid/QGEMM part — measured ~10x A12 regression vs
            // .dequantizedMPS. A device preset must never silently run the
            // 10x-slower direct path.
            c.linearBackend = .dequantizedMPS
        }
        return c
    }
}
