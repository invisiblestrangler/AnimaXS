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

## 2026-08-15 (P1) — Normal CI green on P1 (all sub-items A-H)
HEAD: f42230d (P1 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31904926712
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (273 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Includes new P1 tests: W8 final-layer overflow (>65,504 finite), checkpoint cross-variant accept/reject, final-layer error attribution (never block 1), thrown-diffusion nonzero stage time, cancellation reason, numerical bookkeeping.
artifact/run id: 31904926712 (PR #17 draft)
interpretation: P1 gate met. Baseline + P1 changes all green. Next: P2.

