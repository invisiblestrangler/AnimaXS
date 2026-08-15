# AnimaXS TODO

## Real-device stabilization (2026-08-15) — IN PROGRESS (PR #8, branch fix/real-device-stabilization)

Physical iPhone XS Max run exposed: Files import permission error, generic `AnimapkError error 3`,
unwanted auto-downloads, no keyboard dismissal, silent Generate no-op, Diagnostics crash, idle warmth.

### Repository side — DONE
- [x] Fix A: model discovery local-only (`discover`/`discoverInstalled`/`resolveInstalledModels`);
  acquisition explicit (`download`/`repair`/`importPack`/`verifyExisting`); `prepare()` removed
- [x] Fix B: verification receipts — launch does cheap stat checks; no ~2.07 GB re-hash, no double-hash
- [x] Fix C: security-scoped Files import held for the whole async verify+copy
- [x] Fix D: `AnimapkError` is a `LocalizedError`
- [x] Fix E: downloader validates final HTTP status (2xx), owns staging, useful disk-capacity errors
- [x] Fix F: `installVerified` replaces corrupt destinations via `replaceItemAt`; staging cleaned
- [x] Fix G: `GenerationEligibility` single source (button/visible reason/start guard);
  Generate taps log thermal/memory/coordinator state
- [x] Fix H: `@FocusState` + Done toolbar for prompt and number pad; interactive scroll dismissal
- [x] Fix I: Diagnostics — cheap snapshot on open; explicit basic/hardware/deep levels;
  `json(report:)`/`writeJSON(_:to:)` run zero tests; thermal-gated hardware tests
- [x] Crash marker: `DiagnosticRunMarker` persists running test; "Previous run ended during X" warning
- [x] MPS audit: `MPSMatrixDescriptor.rowBytes(fromColumns:dataType:)` (64-byte aligned) +
  command-buffer status/error inspection after `waitUntilCompleted`
- [x] Resource instrumentation: per-hardware-test thermal before/after, process memory, elapsed
- [x] Regression tests (ModelStoreTests/DiagnosticsTests/GenerationEligibilityTests/
  DiagnosticsRunMarkerTests/SmokeTests)
- [x] Endpoint checks: all three `model-assets-v1` assets → 206 partial, content-length matches manifest
- [x] Docs updated (README/STATUS/DEVICE_TESTS/NEXT_TASK_HANDOFF/DECISIONS/TEST_MATRIX)

### CI — IN PROGRESS
- [x] project-consistency PASS (bootstrap bot commit `92b544c`)
- [ ] iphone-build PASS
- [ ] simulator-tests PASS (incl. new stabilization tests)
- [ ] PR #8 merged to main after green

### Physical retest (K, ordered) — REQUIRED before claiming device fixes
- [ ] Phase 1 idle: no downloads, quick ready, no warmth drift
- [ ] Phase 2: import Qwen3/DiT/VAE, relaunch, quick ready
- [ ] Phase 3: Generate (seed 1337) → first visible state; record last stage
- [ ] Phase 4: Diagnostics open (no auto-run), basic, hardware, crash marker if it dies
- [ ] Phase 5: explicit Download success or precise failure

## Closed: 8px grid root cause (2026-08-14) — DONE

- [x] Wan21 process_out boundary proven (cos 1.0 / rmse 0.0) and fixed on CUDA + Metal
- [x] macOS full-inference E2E PASS; evidence durable (HF, GRID_ROOT_CAUSE_FINAL_REPORT.md)
- [x] Clore CUDA work terminated (order 2024354) — all CUDA work durable

## Backlog (not scheduled)

- [ ] Physical A12 acceptance per DEVICE_TESTS.md (microbenchmarks, memory record, thermal)
- [ ] Residual W8 texture investigation if quality work resumes (see NEXT_TASK_HANDOFF.md)
