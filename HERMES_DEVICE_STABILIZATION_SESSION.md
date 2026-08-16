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
- [ ] T10 Numerical-monitor label cleanup (pending)
- [ ] T11 Docs/state correction (pending)
- [x] CI Gate A (PASS after T1)
- [ ] CI Gate B (after T1-T10)
- [ ] Full-inference W4/W8 gate

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

## CI Runs
- bootstrap-project #31932812543: success (pbxproj regenerated for T1)
- ci.yml Gate A #31932848703: success (project-consistency, iphone-build, simulator-tests)

## Deviations
- (none)

## Physical-device validation
PENDING — requires user's XS Max test after final build.
