# Current implementation state

Baseline main SHA: f89b1d867883f04b2d9aa4b59e9322b36111c8b4 (origin/main, planner-verified baseline)
Working branch: opt/a12-sustained-io
Current HEAD: f89b1d867883f04b2d9aa4b59e9322b36111c8b4
## Current phase: P0 — branch, state files, compile baseline
Current phase gate: P0 — branch created from origin/main f89b1d8, state files committed, normal CI green
Working tree: clean (at P0 commit; local edits to state file not yet committed)

## Completed phases
- P0 COMPLETE: branch `opt/a12-sustained-io` created from `origin/main` `f89b1d8`; three state files committed and pushed; draft PR #17; normal CI green (run 31896517851).
- P1 IN PROGRESS: P1-A foundation committed as WIP `48cf0a7` (ModelVariantDescriptor, ditW4/ditW8V2, descriptor(for:), ResolvedModels→ResolvedModelPack reshape with .hashes, GenerationMetrics.recordDiTPackIdentity). First subagent attempt (deleg_f4593fac9) interrupted mid-inference (no changes); second attempt (deleg_f6307e41) hit iteration budget after P1-A foundation.

## Current exact objective
- P1 CODE COMPLETE (committed 48cf0a7, 892ffd1, 041eefe, 7597ea4): P1-A..P1-H implemented.
- P1 TESTS being added: P1-D overflow (DiTFinalLayerExecutorTests), P1-B checkpoint cross-variant (new CheckpointIdentityTests.swift), P1-E final-layer (NumericalFailureTests), P1-F stage-timing (GenerationCoordinatorTests), P1-G/H metrics (GenerationMetricsTests).
- Next: commit tests; regenerate xcodeproj via bootstrap-project.yml (2 new .swift test files); run normal CI; fix any failures; then P1 gate.

## Current files being modified
- P1 code committed: ModelManifest.swift, GenerationEngine.swift, GenerationMetrics.swift, ModelStore.swift, GenerationCoordinator.swift, DiffusionSampler.swift, DiTFinalLayerExecutor.swift, NumericalFailure.swift, NumericalMonitor.swift, DitForward (via sampler), scripts/dit_final_oracle.py.
- P1 tests (uncommitted): DiTFinalLayerExecutorTests.swift, CheckpointIdentityTests.swift (NEW), NumericalFailureTests.swift, GenerationMetricsTests.swift, GenerationCoordinatorTests.swift.

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
- P0 normal CI (run 31896517851, PR #17 draft): project-consistency PASS, iphone-build PASS, simulator-tests PASS. Baseline compiles/tests green.

## Tests still required
- P1: W8 identity/checkpoint/numerical/error-attribution/failure-telemetry tests (see runbook P1 gate).

## Known unresolved items
- None at P0.

## Exact next command / next code edit
- Create OPTIMIZATION_EVIDENCE.md and OPTIMIZATION_DECISIONS.md, then commit all three state files:
  `git add OPTIMIZATION_IMPLEMENTATION_STATE.md OPTIMIZATION_EVIDENCE.md OPTIMIZATION_DECISIONS.md`
  `git commit -m "docs: bootstrap A12 sustained-performance optimization state"`
- Then push branch and trigger normal CI (ci.yml / bootstrap-project.yml not needed since no .swift added).

## Last safe continuation point
commit: 48cf0a7 (wip P1-A foundation) on opt/a12-sustained-io — HEAD at 48cf0a7
notes: P1-A foundation committed but build is BROKEN at 48cf0a7 until consumers updated (ModelStore.swift:173, GenerationCoordinator.swift:212/304/350). Next dispatch continues P1 from here. Two subagent interruptions observed (model inference stalls + iteration budget); work in small steps, commit frequently.
