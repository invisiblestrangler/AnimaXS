# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: fix/device-stability-no-checkpoint (device-stabilization patch on top of the completed opt/a12-sustained-io work)
Current HEAD: 605300f (device-stabilization patch: checkpoint removed, W8 legacy policy, P8/P6 quarantined/disabled, fatal-Metal poisoning, Import local-only, prompt/seed persistence, monitor-label cleanup)
## Current phase: DEVICE STABILIZATION — source committed; CI Gate A green; PHYSICAL A12 VALIDATION PENDING (user XS Max test) + W4/W8 full-inference gate pending
Current phase gate: P0–P9 optimization work (branch opt/a12-sustained-io) is IMPLEMENTED and CI-green, but NOT device-proven. The device-stabilization patch removed checkpointing, locked W8 production to `w8LegacyStabilized`, quarantined P8 `directQuantized`/hybrid and disabled P6 mmap no-copy after physical A12 failures, added fatal-Metal context poisoning, made Import local-only with single-flight model ops, and persisted prompt/seed. CI green on simulator/macOS does NOT prove real-device success — physical XS Max validation is PENDING user test (stabilization plan §17), and the W4/W8 full-inference gate under the corrected production policy is PENDING (plan §15).
Working tree: clean (at 605300f)

## Completed phases
- P0 COMPLETE: branch created from origin/main f89b1d8; state files committed; PR #17; normal CI green (31896517851).
- P1 COMPLETE (HEAD f42230d): P1-A..P1-H implemented + tested. Normal CI green (31904926712).
- P2 COMPLETE (HEAD 738831e): per-step DiffusionStepMetrics + activeStep accumulation + partial-step recording + traffic counters. Normal CI green (31906565105).
- P3 COMPLETE (HEAD be38161): fused LayerNorm+AdaLN+to-half and in-place half GELU kernels behind fusedNormModulation/fusedMLPActivation toggles (default OFF); P3-C propagates optimization snapshot to final-layer/preparation LinearExecutors; fused-traffic-saved metric. Normal CI green (31908033162: simulator-tests ✓ 280/0-fail).
- P4 COMPLETE (commits 0234a1e..34ba332): strided token-major MPS attention behind stridedTokenMajorAttention toggle (default OFF); AttentionInputLayout enum; strided MPS per-head matrix views eliminate the 3-in + 1-out transposes; strict validation rejects GQA/fp32/bf16 on strided path (P4-F); UI toggle; tests (strided-vs-legacy parity, full DiT shapes self 1024/1024 + cross 1024/512, no head mixing, zero transpose bytes, unsupported-combo rejection, config toggles). CI green on run 31911189760 (287 tests, 0 failures).
- P5 COMPLETE (commits 287555e..7062064): generation-local cross-attention K/V cache (CrossKVCache, one contiguous ~112 MiB .storageModePrivate buffer, per-block ready flags); hit=blit cache→scratch (skip cross K/V projection + static boundary + K RMSNorm), miss=project+store+markReady, Q always dynamic; threaded DiffusionSampler→DitForward→DiTBlockExecutor behind crossKVCache toggle (default OFF); recordCrossKVHit/Miss metrics + summary line; graceful alloc-failure fallback. CI green on run 31913876755 (292 tests, 0 failures).
- P6 COMPLETE (commits cad818f..e0b6109): mmap-backed no-copy weight source (WeightStorageMode copied/noCopy + WeightLoadResult + WeightNoCopyPolicy page-aligned eligibility + bytesNoCopy MTLBuffer alias over the mmap'd pack region, deallocator nil so AnimapkFile retains the mmap); noCopyWeightSource toggle (default OFF) wired through DiTBlock/Preparation/FinalLayer executors; buffer(for:) returns the alias on no-copy path; recordMmapNoCopyBytes + summary; safe fallback to copied for non-aligned/out-of-bounds/device-refusal; Qwen/VAE/LLMAdapter unchanged. CI green on run 31915690478 (302 tests, 0 failures). **NOTE (stabilization): the no-copy path is now DISABLED in production/device settings** after a physical A12 GPU page fault (`kIOGPUCommandBufferCallback` ErrorPageFault, commit b14b88b); the research implementation remains but cannot be enabled without a future GPU-read hardware proof.
- P7 COMPLETE (commits 55ce7bc..153ac1f): attention backends — DiTAttentionBackend selector (legacyHeadMajorMPS/stridedTokenMajorMPS/streamingMPS/metalFlash, default legacy preserves W4); P7-A streaming/online-softmax MPS (chunked keys 64/128/256, running FP32 max/sum, FP32 accumulator, no per-chunk wait, non-causal refusal); P7-B DiT-specialized pure-Metal Flash (headDim=128, heads=16, self 1024/cross 512, token-major, FP32 accumulation, threadExecutionWidth==32 gate, running-max/rescale mandatory, K=32 + K=16 profiles, no simdgroup_matrix); parity vs strided + non-DiT rejection tests; DiagnosticsView backend picker. CI green on run 31918808645 (305 tests, 0 failures).
- P8 COMPLETE (commits 781a54a..1cb1d93): direct packed W4/W8 GEMM — DiTLinearFamily (attentionProjection/mlpUp/mlpDown/other) tagged at DiT call sites; DiTLinearBackend selector (dequantizedMPS/directQuantized/hybrid, default dequantizedMPS preserves W4); qgemm_8x8x64/qgemm_8x16x64 (+ W8 variants) macro-generated MSL kernels with EXACT dequant_w4_to_half/dequant_w8_to_half decode (group K=64), threadgroup W-tile dequant reused across TM rows, FP32 accumulator, no full [N,K] fp16 scratch; hybrid dispatch (MLP→QGEMM, attention→MPS); M=1 matvec preserved; qgemmCalls + per-family metrics; DiagnosticsView selector. CI green on run 31922667679 (307 tests, 0 failures). **NOTE (stabilization): `directQuantized`/`hybrid` are now QUARANTINED from production/device presets** after a physical A12 measurement showed ~10× slowdown vs `dequantizedMPS` (commit e536bd7); the kernel remains research code, testable directly, but cannot be selected for normal device generation.

## Current exact objective
- **Device stabilization (fix/device-stability-no-checkpoint, HEAD 605300f):** source committed and CI Gate A green (run 31932848703). Checkpoint/resume REMOVED (no per-step latent CPU snapshot remains); W8-v2 production resolves to `w8LegacyStabilized` (legacy numerics, `w8BF16Experimental` not production-selected); full-inference CI derives the production numerical policy from `DiTNumericsPolicy.fromVariantID` and gates per-variant markers; P8 `directQuantized`/`hybrid` quarantined (~10× A12 slowdown); P6 mmap no-copy disabled (A12 GPU page fault); fatal Metal faults poison the generation context until restart; manual Import is local-only with per-component single-flight model ops; prompt/seed persist via `generation.lastPrompt`/`generation.lastSeed`. **PENDING: physical XS Max validation (user test) and the W4/W8 full-inference gate under the corrected production policy.** Simulator/macOS CI green does NOT prove device success.

## Current files being modified
- (None in flight — working tree clean at 605300f.) Last stabilization phase modified: GenerationCoordinator/GenerationEligibility/GenerationEngine (checkpoint removal + `metalContextPoisoned`), DiTNumericsPolicy/DiffusionSampler/DiTFinalLayerExecutor (W8 legacy policy + `resolvedNumerics`), InferenceOptimizationConfig/Settings/DiagnosticsView (P8 quarantine + P6 block + compat validator + preset drift), ModelStore/ContentView (single-flight + local-only Import + prompt/seed persistence), FullInferenceTests/full-inference-refine.yml (production-policy gate), NumericalMonitor (probe labels).

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
- (Historical P9 requirement — DONE at d16e317: preset mapping/persistence/reset covered, 324 tests green.)
- Remaining: CI Gate B (final normal CI after all stabilization tasks); W4/W8 full-inference gate under the corrected production policy (plan §15); physical XS Max retest by the user (plan §17).

## Known unresolved items
- None in the P0–P9 optimization scope (all phases implemented, CI green).
- **Open from stabilization:** physical XS Max validation is PENDING user test; W4/W8 full-inference gate under the corrected production policy is PENDING; P6 no-copy has no GPU-read hardware proof yet; BF16 emulation is not claimed range-safe internally.

## Exact next command / next code edit
- (None — the stabilization source phase is complete at 605300f and the tree is clean.)
- Next actions are CI/validation, not code: run CI Gate B on fix/device-stability-no-checkpoint, then the W4/W8 full-inference workflow (plan §15); hand the build to the user for the physical XS Max test (plan §17).

## Last safe continuation point
commit: 605300f (device-stabilization patch complete: checkpoint removal 90ca169, W8 legacy policy 82026cc, full-inference policy 27d3eb7, P8 quarantine e536bd7, P6 disable + fatal-Metal poisoning b14b88b, Import local-only/single-flight f64eb5c, prompt/seed persistence a4a1c60, fresh-run ownership b25b845, preset drift/compat validator f647908, monitor labels 605300f) on fix/device-stability-no-checkpoint
notes: CI Gate A green (31932848703). P0–P9 optimization work (opt/a12-sustained-io, d16e317) remains CI-proven only, NOT device-proven — physical XS Max validation is PENDING user test. P6 no-copy and P8 direct/hybrid are quarantined from production/device settings; checkpointing is removed. Remaining: CI Gate B, W4/W8 full-inference gate, physical retest.
