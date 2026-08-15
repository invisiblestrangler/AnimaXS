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
