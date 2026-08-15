# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: opt/a12-sustained-io
Current HEAD: 9d16490 (P5 code + tests committed; CI confirming)
## Current phase: P5 — generation-local cross-attention K/V cache (runbook §10)
Current phase gate: P5 — implement CrossKVCache; tests + normal CI green
Working tree: clean (at 9d16490)

## Completed phases
- P0 COMPLETE: branch created from origin/main f89b1d8; state files committed; PR #17; normal CI green (31896517851).
- P1 COMPLETE (HEAD f42230d): P1-A..P1-H implemented + tested. Normal CI green (31904926712).
- P2 COMPLETE (HEAD 738831e): per-step DiffusionStepMetrics + activeStep accumulation + partial-step recording + traffic counters. Normal CI green (31906565105).
- P3 COMPLETE (HEAD be38161): fused LayerNorm+AdaLN+to-half and in-place half GELU kernels behind fusedNormModulation/fusedMLPActivation toggles (default OFF); P3-C propagates optimization snapshot to final-layer/preparation LinearExecutors; fused-traffic-saved metric. Normal CI green (31908033162: simulator-tests ✓ 280/0-fail).
- P4 COMPLETE (commits 0234a1e..34ba332): strided token-major DiT attention behind stridedTokenMajorAttention toggle (default OFF); AttentionInputLayout enum; strided MPS per-head matrix views eliminate the 3-in + 1-out transposes; strict validation rejects GQA/fp32/bf16 on strided path (P4-F); UI toggle; tests (strided-vs-legacy parity, full DiT shapes self 1024/1024 + cross 1024/512, no head mixing, zero transpose bytes, unsupported-combo rejection, config toggles). CI green on run 31911189760 (287 tests, 0 failures).

## Current exact objective
- P4 COMPLETE (normal CI green on run 31911189760: 287 tests, 0 failures). Next: P5 — generation-local cross-attention K/V cache (runbook §10).

## Current files being modified
- P4 complete: working tree clean at 34ba332.

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

## Tests still required
- P5: cross-KV miss→hit; 28-block readiness; cache isolated between generations; resume first-step fills; disabled path unchanged; memory ~112 MiB (not multiple copies).

## Known unresolved items
- None.

## Exact next command / next code edit
- Begin P5: implement CrossKVCache (one contiguous ~112 MiB private buffer, per-block ready flags), wire into DiffusionSampler/DiTBlockExecutor behind crossKVCache toggle (default OFF), tests, normal CI.

## Last safe continuation point
commit: 34ba332 (P4 complete, CI green) on opt/a12-sustained-io — HEAD at 34ba332 (remote == local)
notes: P4 done and green (287 tests). P5 next. Reminders: after adding/removing .swift files run bootstrap-project.yml + pull bot commit before ci.yml; push every commit; verify git ls-remote origin opt/a12-sustained-io == git rev-parse HEAD after each push. Full handoff in HANDOFF.md.
