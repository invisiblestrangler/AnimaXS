# Hermes Device Stabilization Session Ledger

**Repo:** /root/AnimaXS
**Branch:** fix/device-stability-no-checkpoint
**Base:** 1756ab8 (opt: A12 sustained-performance + reliability implementation (#17))

## Task Checklist
- [x] T1 Checkpoint/resume removal (verified + committed + CI Gate A green)
- [x] T2 W8 legacy numerical policy (verified + committed)
- [x] T3 Full-inference CI policy alignment (verified + committed)
- [ ] T4 Quarantine directQuantized/hybrid (pending)
- [ ] T5 Disable P6 mmap no-copy + fatal-Metal poisoning (pending)
- [ ] T6 Import vs Download + single-flight (pending)
- [ ] T7 Persist prompt/seed (pending)
- [ ] T8 Fresh-run image ownership (pending)
- [ ] T9 Diagnostics preset drift + compat validator (pending)
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

## CI Runs
- bootstrap-project #31932812543: success (pbxproj regenerated for T1)
- ci.yml Gate A #31932848703: success (project-consistency, iphone-build, simulator-tests)

## Deviations
- (none)

## Physical-device validation
PENDING — requires user's XS Max test after final build.
