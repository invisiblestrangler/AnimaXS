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

## 2026-08-15 (P2) — Normal CI green on P2 (per-step telemetry)
HEAD: 738831e (P2 complete on opt/a12-sustained-io)
command: gh workflow run ci.yml --ref opt/a12-sustained-io; gh run watch 31906565105
configuration: normal CI (ci.yml): project-consistency, simulator-tests, iphone-build
result: ALL PASS. project-consistency ✓, simulator-tests ✓ (277 tests, 14 expected skips, 0 failures), iphone-build ✓.
key metrics: 0 failures. Adds DiffusionStepMetrics (per-step gpu/copy/encode/host/memory/traffic), activeStep accumulation (per-step sums == globals), partial-step recording on throw, conversion/transpose/dequant traffic counters, per-step summary table. Tests: 8 completed steps, failed partial step, sums-match-globals, summary renders.
artifact/run id: 31906565105 (PR #17 draft)
interpretation: P2 gate met. Next: P3.

