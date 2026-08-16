# OPTIMIZATION_EVIDENCE

Append-only evidence log for the A12 sustained-performance optimization implementation.

## 2026-08-15 (P0) — Baseline verification
HEAD: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main)
command: git fetch origin; git rev-parse origin/main; git branch -a
configuration: none (no toolchain on VPS)
result: origin/main == f89b1d8 (planner-verified baseline). Local stale branch `fix/w8-import-refactor` (a646a12) is a divergent pre-squash line and was NOT used.
key metrics: repo clean; branch opt/a12-sustained-io created from f89b1d8.
artifact/run id: n/a
interpretation: Baseline confirmed as origin/main f89b1d8. Starting implementation from there.

## 2026-08-15 (P3) — Normal CI green on P3 (fused activation fusion)
HEAD: be38161 (P3 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31908033162
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (280 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Adds dit_layernorm_modulate_to_half(+probe) and dit_gelu_half_inplace(+probe) Metal kernels (shared BF16 RNE helper), fusedNormModulation/fusedMLPActivation config toggles (default OFF, baseline unchanged), DiTBlockExecutor fused paths (no dit.norm/modulated/hiddenFloat intermediates), P3-C optimization-snapshot propagation to final-layer/preparation LinearExecutors, fusedTrafficSavedBytes metric + summary line. Tests: fused toggles default-off/independence, fused-traffic-saved accumulation.
artifact/run id: 31908033162 (PR #17 draft)
interpretation: P3 gate met. Fused paths gated behind toggles (W4 default unchanged); device measures speed benefit later. Next: P4.

## 2026-08-15 (P4) — Normal CI green on P4 (strided token-major MPS attention)
HEAD: 34ba332 (P4 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31911367183
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (287 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Adds AttentionInputLayout (.headMajor default/.tokenMajor), AttentionExecutor.encodeTokenMajor + tokenMajorHeadMatrix (strided MPS head views, tight score scratch), stridedTokenMajorAttention toggle (default OFF), DiTBlockExecutor gated path eliminating the 3-in+1-out head transposes, strict rejection of GQA/fp32/bf16 on strided path (P4-F), UI toggle. Tests: strided-vs-legacy parity (token-major reference transpose fix 4e29342), full DiT shapes self 1024/1024 + cross 1024/512, no-head-mixing (tolerance loosened 0.002→0.05 in 378df32 for fp16-vs-fp32 rounding), zero transpose bytes, unsupported-combo rejection, config toggles.
artifact/run id: 31911367183 (PR #17 draft); also 31911189760 (workflow_dispatch, 287/0)
interpretation: P4 gate met. Strided path gated behind toggle (W4 default unchanged). Device decides strided vs legacy speed later. Next: P5.

## 2026-08-15 (P4) — Strided token-major MPS attention implemented (transpose elimination)
HEAD: b3aeb14 (P4 code + tests committed on opt/a12-sustained-io)
command: edits to AttentionExecutor/DiTBlockExecutor/InferenceOptimizationConfig(+Settings+DiagnosticsView)/AttentionExecutorTests/InferenceOptimizationConfigTests; commit + push; CI run triggered (id pending)
configuration: stridedTokenMajorAttention toggle (default OFF) — legacy head-major path byte-for-byte unchanged for A/B
result: AttentionInputLayout enum (.headMajor/.tokenMajor(tokenStride:)); encodeTokenMajor + tokenMajorHeadMatrix build strided per-head MPSMatrix views (row stride = tokenStride*2, head offset = head*headDim*2) so Q/K/V + attended output stay token-major — 3-in + 1-out transpose kernels and the 4 head buffers are skipped when ON; score scratch stays tight rows×keyCount (no 2048-dim stride); strict validation rejects GQA (keyValueHeads != heads), fp32ScoresAndSoftmax, and bf16Compute on the strided path (P4-F loud failure); 16-byte alignment validated. DiTBlockExecutor gates the 4 transposes on the toggle. Tests: strided-vs-legacy parity (4 heads, distinguishable per-token/head values), full DiT shapes self 1024/1024 + cross 1024/512 (tokenStride 2048), no head mixing, transpose bytes == 0 + query tiles == 4, unsupported-combo rejection, config toggle default-off/independence/reset.
key metrics: transposeBytes → 0 on the strided path per block (legacy records 4×2 MiB per attention branch); legacy path untouched.
artifact/run id: CI run pending
interpretation: P4 gate candidate. Numerical parity of the strided path is proven by the parity test on real MPS; physical-device perf measurement (and any auto-fallback) deferred per runbook §9 (P4-F).

## 2026-08-15 (P4) — Normal CI green on P4 (strided token-major MPS attention)
HEAD: 34ba332 (P4 complete on opt/a12-sustained-io)
command: gh run watch 31911189760 (workflow_dispatch on f811e38, includes all P4 code + tolerance fix)
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (287 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Adds stridedTokenMajorAttention toggle (default OFF), AttentionInputLayout, strided MPS per-head matrix views eliminating DiT head transposes; strict rejection of GQA/fp32/bf16 on strided path; tests (strided-vs-legacy parity, full DiT shapes, no head mixing, zero transpose bytes, unsupported rejection). Fixed strided MPS parity by transposing legacy head-major reference to token-major before comparison.
artifact/run id: 31911189760 (PR #17 draft)
interpretation: P4 gate met. Strided path gated behind toggle (W4 default unchanged). Next: P5.

## 2026-08-15 (P5) — Normal CI green on P5 (cross-attention K/V cache)
HEAD: 7062064 (P5 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31913876755
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (292 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Adds CrossKVCache (one contiguous ~112 MiB .storageModePrivate buffer, 28 blocks × 4 MiB, per-block ready flags, generation-local). crossKVCache toggle (default OFF). DiTBlockExecutor cross path: hit = blit cache→scratch (skip cross K/V projection + static boundary + K RMSNorm), miss = project + store + markReady; Q always dynamic, self attention never cached. Threaded DiffusionSampler→DitForward→DiTBlockExecutor. recordCrossKVHit/Miss metrics + summary "cross-KV cache hits/misses: X/Y". Graceful alloc-failure fallback to legacy.
artifact/run id: 31913876755 (PR #17 draft)
interpretation: P5 gate met. Cache gated behind toggle (W4 default unchanged). Exact reuse by construction (blit cached post-transform K/V); device measures speed benefit later. Next: P6.

## 2026-08-16 (P6) — Normal CI green on P6 (mmap-backed no-copy weight source)
HEAD: e0b6109 (P6 complete on opt/a12-sustained-io)
command: gh run watch 31915690478 (workflow_dispatch on e0b6109, includes optimization-property fix)
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. simulator-tests ✓ (302 tests, 14 expected skips, 0 failures), iphone-build ✓, project-consistency ✓.
key metrics: 10 new P6 tests (WeightStreamerTests testP6*). Adds noCopyWeightSource toggle (default OFF), WeightStorageMode/WeightNoCopyPolicy, WeightStorageView, mmap no-copy MTLBuffer source with safe fallback; recordMmapNoCopyBytes metric. Fixed: added optimization property to DiTPreparation/DiTFinalLayer executors; added GenerationMetrics.mmapNoCopyBytes global field.
artifact/run id: 31915690478 (PR #17 draft)
interpretation: P6 gate met. mmap no-copy is experimental (behind toggle, default OFF); copied ping-pong retained. Next: P7.

## 2026-08-15 (P6) — Normal CI green on P6 (mmap no-copy weight source)
HEAD: e0b6109 (P6 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31915690478
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (302 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Adds WeightStorageMode (copied/noCopy) + WeightLoadResult + WeightNoCopyPolicy (page-aligned eligibility; makeBuffer(bytesNoCopy:) MTLBuffer alias over the mmap'd pack region, deallocator nil so AnimapkFile retains the mmap). noCopyWeightSource toggle (default OFF) wired through DiTBlock/DiTPreparation/DiTFinalLayer executors; WeightStreamer.load(mode:) with no-copy fast path + copied fallback; buffer(for:) returns the alias on the no-copy path; recordMmapNoCopyBytes + summary "Weight bytes served mmap no-copy". Qwen/VAE/LLMAdapter unchanged (DiT-scoped). Fixed: added GenerationMetrics.mmapNoCopyBytes global field + stored optimization snapshot in DiTPreparation/DiTFinalLayer executors.
artifact/run id: 31915690478 (PR #17 draft)
interpretation: P6 gate met. no-copy gated behind toggle (W4 default unchanged); page-aligned-only with safe copied fallback; device decides speed benefit later. Next: P7.

## 2026-08-16 (P7) — Normal CI green on P7 (streaming MPS + pure Metal Flash attention)
HEAD: 153ac1f (P7 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31918808645
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (305 tests, 14 expected skips, 0 failures), iphone-build ✓ (Metal Flash + streaming kernels compile).
key metrics: 0 failures. Adds DiTAttentionBackend selector (legacyHeadMajorMPS/stridedTokenMajorMPS/streamingMPS/metalFlash, default legacy preserves W4). P7-A streaming/online-softmax MPS (chunked keys 64/128/256, running FP32 max/sum, FP32 accumulator, no per-chunk wait, non-causal refusal). P7-B DiT-specialized pure-Metal Flash (dit_flash_attention_h128_q4_k32 + _k16; headDim=128, heads=16, token-major, FP32 accumulation, running-max/rescale, no simdgroup_matrix). Tests: streaming MPS parity, Metal Flash parity (DiT shape), non-DiT headDim rejection. DiagnosticsView backend picker.
artifact/run id: 31918808645 (PR #17 draft)
interpretation: P7 gate met. Backends runtime-selectable behind DiTAttentionBackend (default legacy preserves W4). Physical A12 (P7-D) selects the winner on sustained later steps — not done by CI. Next: P8.

## 2026-08-16 (P8) — Normal CI green on P8 (direct packed W4/W8 GEMM)
HEAD: 1cb1d93 (P8 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31922667679
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (307 tests, 14 expected skips, 0 failures), iphone-build ✓ (QGEMM Metal kernels compile).
key metrics: 0 failures. Adds DiTLinearFamily (attentionProjection/mlpUp/mlpDown/other) + DiTLinearBackend (dequantizedMPS/directQuantized/hybrid, default dequantizedMPS preserves W4). qgemm_8x8x64/qgemm_8x16x64 + W8 variants as macro-generated MSL kernels (MSL rejects template-param structs), EXACT dequant_w4_to_half/dequant_w8_to_half decode (group K=64), threadgroup W-tile dequant reused across TM rows, FP32 accumulator, no full [N,K] fp16 scratch; hybrid dispatch MLP→QGEMM / attention→MPS; M=1 matvec preserved; qgemmCalls + per-family metrics; DiagnosticsView selector. Tests: direct QGEMM parity (cosine 0.9999998, fp16 accumulation drift), qgemmCalls per-step/family accumulation.
artifact/run id: 31922667679 (PR #17 draft)
interpretation: P8 gate met. Direct QGEMM gated behind selector (default dequantizedMPS preserves W4); device decides per-family enablement later. Next: P9 (combine winners + presets + final report).

## 2026-08-16 (P9) — Presets implemented + normal CI green
HEAD: d16e317 (P9 presets + tests committed on opt/a12-sustained-io; CI green on 31924467039)
command: edits to InferenceOptimizationConfig/Settings/DiagnosticsView/InferenceOptimizationConfigTests; commit + push; ci.yml run 31924467039
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (324 tests, 14 expected skips, 0 failures), iphone-build ✓. Adds InferencePreset enum (10 cases) mapping to concrete InferenceOptimizationConfig per runbook §14/§17; setPreset applies + persists every control + activePreset marker; invalid persisted preset sanitizes to nil; resetToBaseline clears marker; DiagnosticsView preset picker; individual A/B controls retained. Tests: preset→config mapping, persistence, sanitization, snapshot stability, reset clears marker, backend layout invariant (17 new tests).
key metrics: 0 failures. baseline preset == currentBaseline; allCandidate is one test config, NOT an automatic best; no recommendedA12 until physical A12 data (§14).
artifact/run id: 31924467039 (PR #17 draft). Prior failing runs: 31924014986 (escaped key path), 31924308805 (leading-dot type inference) — both fixed.
interpretation: P9 gate met. Presets ready for the §17 physical XS Max benchmark matrix. Final report (§23) + state files → "ready for physical A12 validation".

## 2026-08-16 (DEVICE STABILIZATION) — stabilization patch committed; CI Gate A green; physical validation PENDING
branch: fix/device-stability-no-checkpoint (off 1756ab8, the opt/a12-sustained-io merge)
HEAD: 605300f
command: gh run watch 31932848703 (ci.yml Gate A after the checkpoint-removal refactor)
configuration: normal CI (ci.yml): project-consistency, iphone-build, simulator-tests
result: ALL PASS. Stabilization commits: 90ca169 (checkpoint/resume REMOVED — Checkpoint.swift/CheckpointStore.swift deleted, no Resume UI, no cold-launch resume, GenerationEngine always starts at step 0, no per-step 256 KiB fp32 latent CPU readback), 82026cc (W8-v2 production → `w8LegacyStabilized`; `w8BF16Experimental` retained only for explicit diagnostics, NOT claimed range-safe), 27d3eb7 (full-inference CI derives numerics from `DiTNumericsPolicy.fromVariantID` via `DiffusionSampler.resolvedNumerics`; `full-inference-refine.yml` asserts w4-v2→w4Legacy / w8-v2→w8LegacyStabilized markers), e536bd7 (P8 `directQuantized`/`hybrid` quarantined — ~10× A12 slowdown vs dequantizedMPS), b14b88b (P6 mmap no-copy disabled after A12 GPU page fault; `metalContextPoisoned` on fatal Metal faults), f64eb5c (Import local-only + per-component single-flight), a4a1c60 (prompt/seed persist), b25b845 (fresh-run image ownership), f647908 (preset drift + compat validator), 605300f (monitor probe labels).
key metrics: 0 failures. Previous evidence entries above remain valid history for what CI proved about the optimization branch — they do NOT claim physical-device success.
artifact/run id: 31932848703 (CI Gate A); prior branch CI: full-inference W4/W8 PASS at run 31925685619 (DEFAULT baseline config only, §16).
interpretation: The stabilization patch is source-complete and CI Gate A green. PHYSICAL XS MAX VALIDATION REMAINS **PENDING** (user test, plan §17): simulator/macOS CI green does not prove real-device success. The W4/W8 full-inference gate under the corrected production policy (plan §15) is also still pending. This corrects the earlier "ready for physical A12 validation" phrasing: the branch is ready to BE validated, not already validated.

## 2026-08-16 (P9/FULL-INFERENCE) — Final §16 milestone full-inference gate: PASS (W4 + W8)
HEAD: ff171f0 (opt/a12-sustained-io, P0-P9 complete)
command: gh workflow run full-inference-refine.yml --ref opt/a12-sustained-io; run 31925685619
configuration: full-inference-refine.yml matrix (w4-v2 + w8-v2 parallel), real production prompt→image chain (Qwen→adapter→DiffusionSampler→VAE), latent+RGB regression floors (cosine ≥ 0.65), FULL_INFERENCE=PASS required. Runs DEFAULT baseline config (all P1-P9 optimizations off, per §16 synthetic-vs-device separation). Runner downloaded pinned packs (~3.4 GB total).
result: ALL PASS. Run conclusion success. W4 job 12m, W8 job 20m56s. Both produced FULL_INFERENCE=PASS with generated/reference/comparison PNG + metrics.txt artifacts (downloaded to /root/fullinf-w4, /root/fullinf-w8). This closes the previously-missing runbook §16 milestone gate (after P1, after P3/4/5, and final candidate were all skipped).
key metrics (DEFAULT baseline config, regression floor cosine ≥ 0.65):
- W4 (w4-v2): latent_cosine 0.8231 (≥0.65 ✓), rgb_cosine 0.7818 (≥0.65 ✓), latent_rmse 0.7500, rgb_rmse 0.4302; diffusion 67.76s, total 82.09s.
- W8 (w8-v2): latent_cosine 0.9105 (≥0.65 ✓), rgb_cosine 0.8613 (≥0.65 ✓), latent_rmse 0.5488, rgb_rmse 0.3576; diffusion 101.54s, total 120.23s.
- Both: 8 completed Euler steps × 28 block callbacks, 512×512 RGBA, alpha 255, non-degenerate; attention/activation numerics legacy (baseline).
- Visual check of comparison.png: reference + generated show the same subject/scene (girl, brown hair, blue eyes, cherry blossom) — coherent, correct output.
interpretation: The P1-P9 runtime changes did NOT regress the known-good full-pipeline output on either W4 or W8. Full-inference evidence gap closed. (This is a correctness gate at DEFAULT config; it does NOT time or validate the optimized presets — that remains physical-device §17 work.)
artifact/run id: 31925685619 (PR #17 draft). Artifacts: anima-xs-refine-w4-v2-images (9257881115), anima-xs-refine-w8-v2-images (9257990324).

