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

