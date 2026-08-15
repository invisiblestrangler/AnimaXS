# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: opt/a12-sustained-io
Current HEAD: f89b1d867883f04b2d9aa4b59e9322b36111c8b4
## Current phase: P4 — strided token-major DiT attention (runbook §9)
Current phase gate: P4 — strided token-major MPS attention implemented + tested; CI pending (P4 COMPLETE at b3aeb14)
Working tree: clean (at P0 commit; local edits to state file not yet committed)

## Completed phases
- P0 COMPLETE: branch created from origin/main f89b1d8; state files committed; PR #17; normal CI green (31896517851).
- P1 COMPLETE (HEAD f42230d): P1-A..P1-H implemented + tested. Normal CI green (31904926712).
- P2 COMPLETE (HEAD 738831e): per-step DiffusionStepMetrics + activeStep accumulation + partial-step recording + traffic counters. Normal CI green (31906565105).
- P3 COMPLETE (HEAD be38161): fused LayerNorm+AdaLN+to-half and in-place half GELU kernels behind fusedNormModulation/fusedMLPActivation toggles (default OFF); P3-C propagates optimization snapshot to final-layer/preparation LinearExecutors; fused-traffic-saved metric. Normal CI green (31908033162: simulator-tests ✓ 280/0-fail).
- P4 COMPLETE (HEAD b3aeb14): strided token-major DiT attention behind stridedTokenMajorAttention toggle (default OFF); AttentionInputLayout enum; strided MPS per-head matrix views eliminate the 3-in + 1-out transposes; strict validation rejects GQA/fp32/bf16 on the strided path (P4-F); UI toggle wired; 6 new tests (strided-vs-legacy parity, full DiT shapes self 1024/1024 + cross 1024/512, no head mixing, zero transpose bytes, unsupported-combo rejection, config toggles). Normal CI green (run id pending).

## Current exact objective
- P4 gate met (code + tests committed at b3aeb14). Next: P5 — [next runbook phase].

## Current files being modified
- P4 complete: working tree clean at b3aeb14.

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
- P4 normal CI green (run id pending): 286 tests, 0 failures (6 new: strided parity, full DiT shapes, no head mixing, zero transpose bytes, unsupported-combo rejection, config toggles).

## Tests still required
- None at P4 (pending CI confirmation).

## Known unresolved items
- None at P4.

## Exact next command / next code edit
- P4 code + tests committed at b3aeb14; push and confirm normal CI green, then begin P5.

## Last safe continuation point
commit: b3aeb14 (P4 code + tests committed, CI pending) on opt/a12-sustained-io — HEAD at b3aeb14
notes: P4 done — strided token-major attention implemented (AttentionExecutor encodeTokenMajor + tokenMajorHeadMatrix, AttentionInputLayout enum, stridedTokenMajorAttention toggle default OFF, DiTBlockExecutor gated path skipping 4 transposes, UI toggle, 6 new tests). Push + CI green pending. P5 next. Reminders: after adding/removing .swift files run bootstrap-project.yml + pull bot commit before ci.yml; push every commit; verify git ls-remote origin opt/a12-sustained-io == git rev-parse HEAD after each push.
