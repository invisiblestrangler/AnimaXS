# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-12 after J004 refactor, K002 GenerationCoordinator, and L001 FullInferenceTests repair.

## Repository state

```
branch: main
HEAD: 9e2393c (chore: regenerate AnimaXS.xcodeproj from project.yml)
origin/main: 9e2393c
working-tree: clean (except untracked scripts/oracle_out/block0/ — do not commit)
```

## Last known green CI

```
CI run: 31601722959 (refactor: share validated VAE decode path, fix RGBA8 test)
  ✓ project-consistency
  ✓ iphone-build
  ✓ simulator-tests (106 tests, 12 skipped, 0 failures)
```

K002 and L001 commits (`6e7f042`, `594a04b`) were pushed after; CI triggered by `workflow_dispatch` is run `31602869235` (check status). Bootstrap was run to regenerate xcodeproj for new files.

## Completed work this phase

1. **Commit `ddcf409`** — `fix: restore J004 iOS build` — UIKit import fix for VAEDecoder compile failure.
2. **Commit `1a32112`** — `refactor: share validated VAE decode path, fix RGBA8 test` — Extracted `decodeToPositionMajorRGB(latent:)` shared method; both `execute(latent:rgb:)` and `rgba8(latent:)` call it. Added `DecodedRGBA8` struct and `decode(latent:)` method. Fixed `testPositionToRGBA8MatchesCPUReference` (HWC↔CHW layout bug). CI green `31601722959`.
3. **Commit `594a04b`** — `test: repair full inference integration scaffold (L001)` — Rewrote `FullInferenceTests.swift` with correct production APIs, `TokenizerLoader` semantics, fixture gating.
4. **Commit `6e7f042`** — `feat: add generation coordinator (K002)` — `GenerationCoordinator.swift` + `ContentView` wired. Stage-scoped lifetime, `GenerationState` enum, `Task.checkCancellation()`.

## Key architecture decisions

- **One VAE decoder graph**: `decodeToPositionMajorRGB(latent:)` is the single implementation. RGB and RGBA8 outputs are thin adapters.
- **Platform-neutral output**: `DecodedRGBA8` struct keeps UIKit out of the VAE runtime. `image(latent:)` is a thin UI adapter.
- **Stage-scoped ownership**: `GenerationCoordinator.generate()` creates and releases all heavy model objects (Qwen, adapter, sampler, VAE + AnimapkFile mmaps) within the method body.
- **L001 fixture gating**: `FullInferenceTests` skips cleanly when model packs are unavailable (A005 gates `model-assets-v1`).

## What cannot be proven in this phase

```
physical iPhone XS Max launch
real A12 GPU behavior
actual A12 Metal throughput
actual device peak RSS
iOS jetsam behavior
thermal behavior
real-device GPU memory reclamation timing
real-device second-generation memory stability
```

Progress files say: **CI-validated; physical A12 acceptance pending.**

## Remaining tasks (ordered by dependency)

1. **A005** — Resolve license review (CircleStone + NVIDIA Cosmos) to unblock `model-assets-v1`.
2. **L001 full run** — When A005 resolves and model packs are available, run `full-inference.yml` workflow to execute end-to-end inference in Actions. Establish measured final-image regression metrics from the real full-pack run.
3. **K003** — Cancellation at safe boundaries (stop scheduling, finish safe work, checkpoint, release).
4. **K004** — Memory warning handler (checkpoint + graceful cancel + free buffers).
5. **Physical A12 acceptance** — Build in Xcode, install on iPhone XS Max, record stage timings, peak memory, second-generation stability, thermal behavior. Tune VAE tile size only if device measurements justify it.

## First command for next agent

```bash
cd /root/AnimaXS
git pull --rebase origin main
gh run list --limit 5
# Verify CI is green on the latest commit
# If A005 is resolved, run: gh workflow run full-inference.yml
```
