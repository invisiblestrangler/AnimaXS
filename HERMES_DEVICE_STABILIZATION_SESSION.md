# Hermes Device Stabilization Session Ledger

**Repo:** /root/AnimaXS
**Branch:** fix/device-stability-no-checkpoint
**Base:** 1756ab8 (opt: A12 sustained-performance + reliability implementation (#17))

## Task Checklist
- [x] T1 Checkpoint/resume removal (verified + committed + CI Gate A green)
- [x] T2 W8 legacy numerical policy (verified + committed)
- [x] T3 Full-inference CI policy alignment (verified + committed)
- [x] T4 Quarantine directQuantized/hybrid (verified + committed)
- [x] T5 Disable P6 mmap no-copy + fatal-Metal poisoning (verified + committed)
- [x] T6 Import vs Download + single-flight (verified + committed)
- [x] T7 Persist prompt/seed (verified + committed)
- [x] T8 Fresh-run image ownership (verified + committed)
- [x] T9 Diagnostics preset drift + compat validator (verified + committed)
- [x] T10 Numerical-monitor label cleanup (verified + committed)
- [x] T11 Docs/state correction (verified + committed)
- [x] CI Gate A (PASS after T1)
- [x] CI Gate B (PASS after T1-T11, run 31940846369 at 86c5904)
- [x] Full-inference W4/W8 gate (PASS, run 31941211295 at 86c5904)

## Commits
- 90ca169 feat: remove checkpoint/resume from production entirely (T1 source)
- 0f86b67 chore: regenerate AnimaXS.xcodeproj from project.yml (bootstrap #31932812543)
- 5e9933b docs: add device-stabilization runbook and session ledger
- 82026cc fix: W8-v2 production resolves to stabilized legacy numerics (T2)
- 27d3eb7 fix: full-inference CI executes the production numerical policy (T3)
- fd3e31f docs: update session ledger (T2, T3 committed)
- e536bd7 fix: quarantine P8 directQuantized/hybrid from production device settings (T4)
- b14b88b fix: disable P6 mmap no-copy and poison context on fatal Metal faults (T5)
- f64eb5c fix: Import is local-only and per-component model ops are single-flight (T6)
- a4a1c60 feat: persist latest prompt and seed across relaunch (T7)
- b25b845 fix: a fresh Generate owns the image/metrics surface (T8)
- f48deca docs: update session ledger (T7, T8 committed)
- f647908 fix: Diagnostics preset drift + central compatibility validator (T9)
- 605300f fix: numerical-monitor gate/add probe labels identify the gate input boundary (T10)
- de61aa6 docs: correct project state — device stabilization decisions recorded (T11)
- bee68e5 fix: fatal-Metal classification compares raw NSError code to case rawValues (CI fix)
- f26a1be fix: fatal-Metal test helper takes MTLCommandBufferError.Code (CI fix)
- 86c5904 fix: correct Task 6 ModelStore single-flight tests for real store semantics (CI fix)

## CI Runs
- bootstrap-project #31932812543: success (pbxproj regenerated for T1)
- ci.yml Gate A #31932848703: success (project-consistency, iphone-build, simulator-tests)
- ci.yml Gate B #31940846369: success (project-consistency, iphone-build, simulator-tests) at 86c5904
- full-inference-refine #31941211295: PASS both W4 (w4Legacy, latent cos 0.8231, RGB cos 0.7818) and W8 (w8LegacyStabilized, latent cos 0.9105, RGB cos 0.8613); generated images coherent, no checkerboard/NaN

## Deviations
- (none)

## Physical-device validation
PENDING — requires user's XS Max test after final build.
