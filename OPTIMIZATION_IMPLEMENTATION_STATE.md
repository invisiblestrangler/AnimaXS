# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: opt/a12-sustained-io
Current HEAD: f89b1d867883f04b2d9aa4b59e9322b36111c8b4
## Current phase: P0 — branch, state files, compile baseline
Current phase gate: P0 — branch created from origin/main f89b1d8, state files committed, normal CI green
Working tree: clean (at P0 commit; local edits to state file not yet committed)

## Completed phases
- P0 COMPLETE: branch created from origin/main f89b1d8; state files committed; PR #17; normal CI green (31896517851).
- P1 COMPLETE (HEAD f42230d): P1-A..P1-H implemented + tested. Normal CI green (31904926712).
- P2 COMPLETE (HEAD 738831e): per-step DiffusionStepMetrics + activeStep accumulation + partial-step recording + traffic counters. Normal CI green (31906565105).
- P3 COMPLETE (HEAD be38161): fused LayerNorm+AdaLN+to-half and in-place half GELU kernels behind fusedNormModulation/fusedMLPActivation toggles (default OFF); P3-C propagates optimization snapshot to final-layer/preparation LinearExecutors; fused-traffic-saved metric. Normal CI green (31908033162: simulator-tests ✓ 280/0-fail).

## Current exact objective
- P3 gate met. Next: P4 — remove DiT attention head transposes via strided token-major MPS views (runbook §9).

## Current files being modified
- P3 complete: working tree clean at be38161.

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
- P3 normal CI green (31908033162): 280 tests, 0 failures. Fused-toggled + metrics tests pass.

## Tests still required
- P4: strided token-major MPS attention vs legacy head-major; self 1024/1024 descriptor test; cross 1024/512 descriptor test; no head mixing; transpose bytes drop to zero in strided backend.

## Known unresolved items
- None at P0.

## Exact next command / next code edit
- Create OPTIMIZATION_EVIDENCE.md and OPTIMIZATION_DECISIONS.md, then commit all three state files:
  `git add OPTIMIZATION_IMPLEMENTATION_STATE.md OPTIMIZATION_EVIDENCE.md OPTIMIZATION_DECISIONS.md`
  `git commit -m "docs: bootstrap A12 sustained-performance optimization state"`
- Then push branch and trigger normal CI (ci.yml / bootstrap-project.yml not needed since no .swift added).

## Last safe continuation point
commit: be38161 (P3 complete, normal CI green) on opt/a12-sustained-io — HEAD at be38161
notes: P3 done and green. Next: P4 strided token-major MPS attention. Reminders: after adding/removing .swift files run bootstrap-project.yml + pull bot commit before ci.yml; push every commit; delegation provider works (delegating phases to single subagents); subagents hit iteration budget mid-large-phases — brief them for small commits and expect to finish some phases myself.
