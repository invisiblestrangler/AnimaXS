# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: opt/a12-sustained-io
Current HEAD: 153ac1f (P7 complete, CI green)
## Current phase: P8 — direct packed W4/W8 GEMM (runbook §13, highest-risk)
Current phase gate: P8 — DiTLinearFamily tagging; qgemm_8x8x64/qgemm_8x16x64 tile profiles; dispatch dequantizedMPS/directQuantized/hybrid; parity + CI green
Working tree: clean (at 153ac1f)

## Completed phases
- P0 COMPLETE: branch created from origin/main f89b1d8; state files committed; PR #17; normal CI green (31896517851).
- P1 COMPLETE (HEAD f42230d): P1-A..P1-H implemented + tested. Normal CI green (31904926712).
- P2 COMPLETE (HEAD 738831e): per-step DiffusionStepMetrics + activeStep accumulation + partial-step recording + traffic counters. Normal CI green (31906565105).
- P3 COMPLETE (HEAD be38161): fused LayerNorm+AdaLN+to-half and in-place half GELU kernels behind fusedNormModulation/fusedMLPActivation toggles (default OFF); P3-C propagates optimization snapshot to final-layer/preparation LinearExecutors; fused-traffic-saved metric. Normal CI green (31908033162: simulator-tests ✓ 280/0-fail).
- P4 COMPLETE (commits 0234a1e..34ba332): strided token-major MPS attention behind stridedTokenMajorAttention toggle (default OFF); AttentionInputLayout enum; strided MPS per-head matrix views eliminate the 3-in + 1-out transposes; strict validation rejects GQA/fp32/bf16 on strided path (P4-F); UI toggle; tests (strided-vs-legacy parity, full DiT shapes self 1024/1024 + cross 1024/512, no head mixing, zero transpose bytes, unsupported-combo rejection, config toggles). CI green on run 31911189760 (287 tests, 0 failures).
- P5 COMPLETE (commits 287555e..7062064): generation-local cross-attention K/V cache (CrossKVCache, one contiguous ~112 MiB .storageModePrivate buffer, per-block ready flags); hit=blit cache→scratch (skip cross K/V projection + static boundary + K RMSNorm), miss=project+store+markReady, Q always dynamic; threaded DiffusionSampler→DitForward→DiTBlockExecutor behind crossKVCache toggle (default OFF); recordCrossKVHit/Miss metrics + summary line; graceful alloc-failure fallback. CI green on run 31913876755 (292 tests, 0 failures).
- P6 COMPLETE (commits cad818f..e0b6109): mmap-backed no-copy weight source (WeightStorageMode copied/noCopy + WeightLoadResult + WeightNoCopyPolicy page-aligned eligibility + bytesNoCopy MTLBuffer alias over the mmap'd pack region, deallocator nil so AnimapkFile retains the mmap); noCopyWeightSource toggle (default OFF) wired through DiTBlock/Preparation/FinalLayer executors; buffer(for:) returns the alias on no-copy path; recordMmapNoCopyBytes + summary; safe fallback to copied for non-aligned/out-of-bounds/device-refusal; Qwen/VAE/LLMAdapter unchanged. CI green on run 31915690478 (302 tests, 0 failures).
- P7 COMPLETE (commits 55ce7bc..153ac1f): attention backends — DiTAttentionBackend selector (legacyHeadMajorMPS/stridedTokenMajorMPS/streamingMPS/metalFlash, default legacy preserves W4); P7-A streaming/online-softmax MPS (chunked keys 64/128/256, running FP32 max/sum, FP32 accumulator, no per-chunk wait, non-causal refusal); P7-B DiT-specialized pure-Metal Flash (headDim=128, heads=16, self 1024/cross 512, token-major, FP32 accumulation, threadExecutionWidth==32 gate, running-max/rescale mandatory, K=32 + K=16 profiles, no simdgroup_matrix); parity vs strided + non-DiT rejection tests; DiagnosticsView backend picker. CI green on run 31918808645 (305 tests, 0 failures).

## Current exact objective
- P7 COMPLETE (normal CI green on run 31918808645: 305 tests, 0 failures). Next: P8 — direct packed W4/W8 GEMM (runbook §13, highest-risk).

## Current files being modified
- P7 complete: working tree clean at 153ac1f.

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

## Tests still required
- P8: tiny W4/W8 QGEMM vs dequant+MPS parity; odd M/N; tile boundary; no full FP16 weight scratch on direct path; qgemmCalls metric.

## Known unresolved items
- None.

## Exact next command / next code edit
- Begin P8: add DiTLinearFamily to LinearExecutor call sites; implement qgemm_8x8x64 + qgemm_8x16x64 (TK=64, group K=64) tile profiles with threadgroup dequant of W tile reused across TM activation rows + FP32 accumulator + exact dequant_w4_to_half/dequant_w8_to_half decode; dispatch .dequantizedMPS/.directQuantized/.hybrid (MLP→QGEMM initially); no full FP16 weight scratch on direct path.

## Last safe continuation point
commit: 153ac1f (P7 complete, CI green) on opt/a12-sustained-io — HEAD at 153ac1f (remote == local)
notes: P7 done and green (305 tests). P8 (direct packed W4/W8 QGEMM, highest-risk) next. Reminders: after adding/removing .swift files run bootstrap-project.yml + pull bot commit before ci.yml; push every commit; verify git ls-remote origin opt/a12-sustained-io == git rev-parse HEAD after each push. Full handoff in HANDOFF.md.
