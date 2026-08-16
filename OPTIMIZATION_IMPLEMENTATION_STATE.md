# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: opt/a12-sustained-io
Current HEAD: d16e317 (P9 presets + tests, CI green on 31924467039)
## Current phase: P9 COMPLETE — combine winners + presets + final integration/report (runbook §14, §23)
Current phase gate: P9 COMPLETE — InferencePreset enum; diagnostics presets; no winner forced without A12 data; final report written; READY FOR PHYSICAL A12 VALIDATION (runbook §17)
Working tree: clean (at d16e317)

## Completed phases
- P0 COMPLETE: branch created from origin/main f89b1d8; state files committed; PR #17; normal CI green (31896517851).
- P1 COMPLETE (HEAD f42230d): P1-A..P1-H implemented + tested. Normal CI green (31904926712).
- P2 COMPLETE (HEAD 738831e): per-step DiffusionStepMetrics + activeStep accumulation + partial-step recording + traffic counters. Normal CI green (31906565105).
- P3 COMPLETE (HEAD be38161): fused LayerNorm+AdaLN+to-half and in-place half GELU kernels behind fusedNormModulation/fusedMLPActivation toggles (default OFF); P3-C propagates optimization snapshot to final-layer/preparation LinearExecutors; fused-traffic-saved metric. Normal CI green (31908033162: simulator-tests ✓ 280/0-fail).
- P4 COMPLETE (commits 0234a1e..34ba332): strided token-major MPS attention behind stridedTokenMajorAttention toggle (default OFF); AttentionInputLayout enum; strided MPS per-head matrix views eliminate the 3-in + 1-out transposes; strict validation rejects GQA/fp32/bf16 on strided path (P4-F); UI toggle; tests (strided-vs-legacy parity, full DiT shapes self 1024/1024 + cross 1024/512, no head mixing, zero transpose bytes, unsupported-combo rejection, config toggles). CI green on run 31911189760 (287 tests, 0 failures).
- P5 COMPLETE (commits 287555e..7062064): generation-local cross-attention K/V cache (CrossKVCache, one contiguous ~112 MiB .storageModePrivate buffer, per-block ready flags); hit=blit cache→scratch (skip cross K/V projection + static boundary + K RMSNorm), miss=project+store+markReady, Q always dynamic; threaded DiffusionSampler→DitForward→DiTBlockExecutor behind crossKVCache toggle (default OFF); recordCrossKVHit/Miss metrics + summary line; graceful alloc-failure fallback. CI green on run 31913876755 (292 tests, 0 failures).
- P6 COMPLETE (commits cad818f..e0b6109): mmap-backed no-copy weight source (WeightStorageMode copied/noCopy + WeightLoadResult + WeightNoCopyPolicy page-aligned eligibility + bytesNoCopy MTLBuffer alias over the mmap'd pack region, deallocator nil so AnimapkFile retains the mmap); noCopyWeightSource toggle (default OFF) wired through DiTBlock/Preparation/FinalLayer executors; buffer(for:) returns the alias on no-copy path; recordMmapNoCopyBytes + summary; safe fallback to copied for non-aligned/out-of-bounds/device-refusal; Qwen/VAE/LLMAdapter unchanged. CI green on run 31915690478 (302 tests, 0 failures).
- P7 COMPLETE (commits 55ce7bc..153ac1f): attention backends — DiTAttentionBackend selector (legacyHeadMajorMPS/stridedTokenMajorMPS/streamingMPS/metalFlash, default legacy preserves W4); P7-A streaming/online-softmax MPS (chunked keys 64/128/256, running FP32 max/sum, FP32 accumulator, no per-chunk wait, non-causal refusal); P7-B DiT-specialized pure-Metal Flash (headDim=128, heads=16, self 1024/cross 512, token-major, FP32 accumulation, threadExecutionWidth==32 gate, running-max/rescale mandatory, K=32 + K=16 profiles, no simdgroup_matrix); parity vs strided + non-DiT rejection tests; DiagnosticsView backend picker. CI green on run 31918808645 (305 tests, 0 failures).
- P8 COMPLETE (commits 781a54a..1cb1d93): direct packed W4/W8 GEMM — DiTLinearFamily (attentionProjection/mlpUp/mlpDown/other) tagged at DiT call sites; DiTLinearBackend selector (dequantizedMPS/directQuantized/hybrid, default dequantizedMPS preserves W4); qgemm_8x8x64/qgemm_8x16x64 (+ W8 variants) macro-generated MSL kernels with EXACT dequant_w4_to_half/dequant_w8_to_half decode (group K=64), threadgroup W-tile dequant reused across TM rows, FP32 accumulator, no full [N,K] fp16 scratch; hybrid dispatch (MLP→QGEMM, attention→MPS); M=1 matvec preserved; qgemmCalls + per-family metrics; DiagnosticsView selector. CI green on run 31922667679 (307 tests, 0 failures).

## Current exact objective
- P9 COMPLETE (normal CI green on run 31924467039: 324 tests, 14 expected skips, 0 failures). InferencePreset enum (10 cases) + setPreset persistence/sanitization + DiagnosticsView preset picker implemented and CI-verified. Final report written to OPTIMIZATION_FINAL_REPORT.md (runbook §23). PROJECT IS READY FOR PHYSICAL A12 VALIDATION: use the Diagnostics-screen preset selector to run the runbook §17 benchmark matrix on a physical XS Max, then add a `recommendedA12` preset with the proven winner.

## Current files being modified
- P9 complete: InferencePreset (InferenceOptimizationConfig.swift), setPreset + activePreset + loadPreset (InferenceOptimizationSettings.swift), preset picker (DiagnosticsView.swift), P9 preset tests (InferenceOptimizationConfigTests.swift), OPTIMIZATION_FINAL_REPORT.md. Working tree clean at d16e317.

## Invariants that must not regress
- W4 known-good path
- 8 diffusion steps
- 28 DiT blocks
- no app thermal gating
- no automatic model download
- bounded-memory weight streaming
- no model packs on VPS (never git lfs pull, never download models)
- no Xcode/Metal/PyTorch installs on VPS; build/test only via GitHub Actions CI

## Tests already passed at current HEAD
- P0 normal CI green (31896517851).
- P1 normal CI green (31904926712): 273 tests, 0 failures.
- P2 normal CI green (31906565105): 277 tests, 0 failures.
- P3 normal CI green (31908033162): 280 tests, 0 failures.
- P4 normal CI green (31911189760): 287 tests, 0 failures.
- P5 normal CI green (31913876755): 292 tests, 0 failures. Cross-KV cache unit tests + metrics accumulation.
- P6 normal CI green (31915690478): 302 tests, 0 failures. WeightStreamer no-copy/mmap alias + metrics.
- P7 normal CI green (31918808645): 305 tests, 0 failures. Streaming MPS + Metal Flash parity + non-DiT rejection.
- P8 normal CI green (31922667679): 307 tests, 0 failures. Direct QGEMM parity + qgemmCalls metrics.

## Tests still required
- P9: preset selection maps to the right combination; reset-to-baseline restores; immutable snapshot mid-run unchanged.

## Known unresolved items
- None.

## Exact next command / next code edit
- Begin P9: add InferencePreset enum (baseline/current1024Control/fusedTraffic/stridedMPS/stridedMPSKV/noCopyCandidate/streamingMPSCandidate/metalFlashCandidate/directQGEMMCandidate/allCandidate); diagnostics preset selector + reset-to-baseline; do NOT change currentBaseline to recommendedA12 (no A12 data yet); final report per runbook §23.

## Last safe continuation point
commit: 1cb1d93 (P8 complete, CI green) on opt/a12-sustained-io — HEAD at 1cb1d93 (remote == local)
notes: P8 done and green (307 tests). P9 (combine winners + presets + final report) is the LAST phase. Reminders: after adding/removing .swift files run bootstrap-project.yml + pull bot commit before ci.yml; push every commit; verify git ls-remote origin opt/a12-sustained-io == git rev-parse HEAD after each push. Full handoff in HANDOFF.md.
