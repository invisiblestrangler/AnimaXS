# AnimaXS A12 optimization implementation report

## Repository
Branch: `opt/a12-sustained-io`
Final HEAD: `d16e317` (P9 presets, CI green on run 31924467039)
origin branch: `origin/opt/a12-sustained-io` == local `d16e317`
Working tree: clean

## Correctness fixes
W8 identity: P1 resolved-pack identity via `models.hashes`, checkpoint identity via `ModelManifestEntry`, W8-v2 normal alternate of the `.dit` slot; receipts record the matched variant. (CI 31904926712)
W8 numerical boundary: P1 `DiTNumericsPolicy` — W8 final-layer BF16-RNE-in-FP32 overflow fix keeps residuals >65,504 finite; W4 known-good path unchanged.
checkpoint identity: P1 cross W4/W8 resume rejects; same-variant resume passes.
error attribution: P1 `NumericalLocation` — real error location, no fake block 1.
cancellation attribution: P1 cancel reasons distinguishable (user/background/memory-warning).

## Optimization backends implemented
Fused norm/modulation: P3 `dit_layernorm_modulate_to_half` (eliminates `dit.norm.f32` + `dit.modulated.f32`, 16 MiB), behind `fusedNormModulation` toggle (default OFF).
Fused MLP GELU: P3 `dit_gelu_half_inplace` (eliminates ~32 MiB `dit.hidden.f32`), behind `fusedMLPActivation` toggle (default OFF).
Strided MPS: P4 strided token-major MPS attention (`AttentionInputLayout`/`encodeTokenMajor`) eliminates 3-in+1-out head transposes, behind `stridedTokenMajorAttention` (default OFF).
Cross-KV cache: P5 `CrossKVCache` (one ~112 MiB private buffer, per-block ready, blit hit/miss) behind `crossKVCache` (default OFF).
mmap no-copy: P6 `WeightStorageMode` no-copy `bytesNoCopy` MTLBuffer alias over the mmap'd pack region (page-aligned only, copied fallback) behind `noCopyWeightSource` (default OFF, experimental).
Streaming MPS: P7 streaming/online-softmax MPS attention (chunked keys, FP32 running max/sum, non-causal refusal), `.streamingMPS` backend.
Metal Flash: P7 DiT-specialized pure-Metal Flash (`dit_flash_attention_h128_q4_k32`/`_k16`, token-major, FP32 acc, no simdgroup_matrix), `.metalFlash` backend.
Direct QGEMM: P8 direct packed W4/W8 GEMM (`qgemm_8x8x64`/`qgemm_8x16x64`, macro-generated MSL, exact dequant decode, threadgroup W-tile, FP32 acc, no full [N,K] fp16 scratch), `.directQuantized`/`.hybrid` linear backends.
Hybrid selection: P8 `.hybrid` — MLP up/down matrices run direct QGEMM, attention projections stay dequantized-MPS until A12 data.

P9 presets: `InferencePreset` (baseline/current1024Control/fusedTraffic/stridedMPS/stridedMPSKV/noCopyCandidate/streamingMPSCandidate/metalFlashCandidate/directQGEMMCandidate/allCandidate) maps to concrete combinations; `setPreset` persists + sanitizes; individual A/B controls retained. `currentBaseline` UNCHANGED — no `recommendedA12` until physical A12 data.

## Tests
Normal CI: ci.yml green on P9 final — run 31924467039.
Simulator tests: 324 tests, 14 expected skips, 0 failures.
Full-inference W4: NOT run locally on VPS (no toolchain/models per §0.2). Milestone `full-inference-refine.yml` gates remain available for the final candidate.
Full-inference W8: NOT run (deferred to physical device; W8 correctness covered by CI parity/unit tests, not timing).
Artifact/run IDs: P0 31896517851 · P1 31904926712 · P2 31906565105 · P3 31908033162 · P4 31911189760 · P5 31913876755 · P6 31915690478 · P7 31918808645 · P8 31922667679 · P9 31924467039. (PR #17 draft.)

## Traffic changes proven by tests
Dequant traffic: P8 direct QGEMM avoids the full dequantized [N,K] fp16 weight scratch; `recordQGEMMCall`; parity (cosine 0.9999998).
Transpose traffic: P4 strided token-major → `transposeBytes` 0 per block on the strided path.
Conversion traffic: P3 fused norm/GELU eliminate FP32 norm/modulated/hidden intermediates (16 MiB + 32 MiB).
KV hit/miss: P5 cross-KV cache hit=blit (skip cross K/V projection + boundary + K RMSNorm); miss=project+store; `recordCrossKVHit/Miss`.

## Known limitations
- No physical XS Max measurements exist on the VPS; backend winners are NOT claimed. CI timing is not device proof (§16).
- mmap no-copy is experimental: page-aligned ranges only, else copied fallback; device must accept the `bytesNoCopy` alias.
- Metal Flash is gated to threadExecutionWidth==32 and DiT shapes (headDim 128, heads 16, self 1024 / cross 512); non-DiT rejected.
- Streaming MPS / Metal Flash require the token-major layout (DiTBlockExecutor throws otherwise — no silent fallback).
- Simulator/macOS timing does not predict A12 sustained behavior; charging changes device thermal/sustained behavior (test unplugged, low-power off).

## Physical XS Max tests still required
Runbook §17, first-pass configurations (fixed model W4, fixed prompt, fixed seed, 8 steps, unchanged resolution, unplugged, low-power off). Record full metrics text after each run.
Preset 1: Control (linear 1024, attention 1024, direct MPS I/O on, ping-pong on, all new opts off)
Preset 2: Fused (control + fused norm/modulation + fused MLP GELU)
Preset 3: Strided (Fused + strided token-major MPS)
Preset 4: KV (Strided + cross-K/V cache)
Preset 5: No-copy (KV + mmap no-copy)
Preset 6: Streaming MPS (Fused + KV + streaming MPS) — compare vs Strided
Preset 7: Metal Flash (Fused + KV + Metal Flash)
Preset 8: MLP QGEMM hybrid (best attention + direct QGEMM for MLP only)
Preset 9: All candidate (all currently-winning components)
After W4 stable: run W8-v2 correctness first; do not use W8 as the first performance control.
Exact prompt/seed instructions: use the same fixed model (W4 first), fixed prompt, fixed seed, 8 steps, unchanged resolution, device unplugged, low-power mode off; record full metrics text after each run.
These map directly to the `InferencePreset` enum (`.current1024Control` … `.allCandidate`).

## Context/state files
OPTIMIZATION_IMPLEMENTATION_STATE.md: updated — P9 complete, "ready for physical A12 validation".
OPTIMIZATION_EVIDENCE.md: updated — all phase run IDs, P9 green (31924467039), prior failing runs documented.
OPTIMIZATION_DECISIONS.md: updated — D003 P9 preset decision.

## Final recommendation
Do NOT claim which backend is fastest until physical A12 measurements are supplied. Use the `InferencePreset` selector on the Diagnostics screen to run the §17 matrix on the physical XS Max; then add a `recommendedA12` preset with the proven winner combination.
