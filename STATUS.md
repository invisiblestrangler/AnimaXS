# Current Project Status

- **Base main:** `d2c4fad` (Wan21 latent-format boundary fix, 8px grid root cause closed).
- **Active branch:** `fix/real-device-stabilization` (PR #8) — physical-device stabilization
  from the first iPhone XS Max run (2026-08-15).
- **Current phase:** stabilization fixes implemented; CI verification in progress.

## Real-device stabilization (2026-08-15)

The first physical-device run exposed bugs that simulator/CI did not catch:

1. manual `.animapk` import failed with an iOS Files permission error (security-scoped access);
2. Download failed with a generic `AnimaXS.AnimapkError error 3` (now `LocalizedError`);
3. missing-model launch auto-triggered downloads (discovery is now local-only);
4. prompt keyboard had no dismissal path (now `@FocusState` + Done toolbar);
5. Generate could silently no-op (now single-source `GenerationEligibility` + visible reasons);
6. Diagnostics crashed the app on device (crash-localizing marker, MPS rowBytes audit, no auto-run);
7. app warmed at idle after launch (verification receipts remove ~2.07 GB launch re-hash;
   discovery/verification moved off the main actor).

### Repository-side status

- [x] Model discovery local-only, zero network requests on launch (Fix A)
- [x] Verification receipts: launch = cheap stat checks; missing/stale receipt = one off-main verify (Fix B)
- [x] Security-scoped Files import held for the whole async copy (Fix C)
- [x] `AnimapkError` conforms to `LocalizedError` (Fix D)
- [x] Downloader validates final HTTP response, owns staging, useful disk errors (Fix E)
- [x] Import/repair replace corrupt destinations atomically (Fix F)
- [x] Generate blocked reasons visible; single-source eligibility (Fix G); thermal policy removed (D102)
- [x] Keyboard dismissal (Fix H)
- [x] Diagnostics: cheap snapshot on open, explicit levels, JSON export runs zero tests (Fix I)
- [x] DiagnosticRunMarker persists running test across launches (crash localization)
- [x] MPS precision diagnostic uses `MPSMatrixDescriptor.rowBytes(fromColumns:dataType:)`
      (64-byte alignment) + command-buffer status/error inspection
- [x] Regression tests: no-network discovery, receipt validity, corrupt-destination import,
      downloader HTTP handling, eligibility matrix, diagnostics-once, crash marker
- [ ] CI green (project-consistency PASS; iphone-build / simulator-tests in flight)
- [ ] Physical iPhone retest (DEVICE_TESTS.md checklist) — required before claiming
      "Diagnostics crash fixed" on A12

## Prior phase (closed): 8px grid root cause

- **Root cause (D097):** missing Wan21 `process_out` latent postprocess at the sampler→VAE
  boundary. Fixed on CUDA and Metal; comfy-ref pinned; carrier 245–254× → ~1.3×.
- **Full-inference macOS E2E PASS** (run `31724606040`, branch `investigate/animapk-cuda-parity`):
  all 8 Euler steps + 224 block callbacks, latent cosine 0.971, evidence durable on HF.
- See `GRID_ROOT_CAUSE_FINAL_REPORT.md` and `HERMES_ANIMAPK_CUDA_PARITY.md`.

---

## Runtime inference-optimization experiments (D207)

- **Branch:** `perf/runtime-experiments`
- **Base SHA:** `c424427` (origin/main `3bf7297` + the D206 completed-8/8
  checkpoint fix, which is NOT on origin/main and must be preserved)
- **Implementation:** all runtime experiments landed in one batch (config,
  settings, metrics, linear/attention tiles, direct MPS I/O, ping-pong OFF,
  monitor OFF, experimental W8 store, Diagnostics UI, tests). The D206
  completed-run checkpoint cleanup and the existing two-slot ping-pong remain
  the baseline.
- **Xcode project:** regenerated via `bootstrap-project.yml` (bot commit
  `64c11e0`).
- **Normal CI:** PASS (run `31874533573`) — project-consistency ✓,
  iphone-build ✓, simulator-tests ✓ (264 tests, 0 failures, 14 expected
  real-pack-gated skips).
- **Full-inference correctness gate:** PASS (run `31874824950`) —
  `FULL_INFERENCE=PASS`, latent cosine 0.866, RGB cosine 0.8198 (matches the
  pre-change baseline), `FULL_STRESS=PASS`, 2/2 tests.
- **Real-device results:** PENDING — no device speedup is claimed from
  simulator/macOS tests. The seven-run matrix is in DEVICE_TESTS.md.

## W8 v2 import crash fix (2026-08-15)

- **Branch:** `fix/w8-import-stream` (off `origin/main` `f25e257`)
- **Problem:** importing `anima-turbo-v1.0-xsmax-w8-v2.animapk` (2.233 GB) on the
  physical iPhone XS Max terminated the app during import.
- **Fix:** `ExperimentalDiTPackStore.importPack` now streams the source into
  staging exactly once (read → hash → write per 1 MiB chunk inside an explicit
  `autoreleasepool`), then verifies pinned byte count + SHA-256 before atomic
  install + receipt. Removes the prior full SHA pass + full `copyItem` pass
  (~4.47 GB source I/O → ~2.23 GB) and bounds per-chunk temporary lifetime.
- **Also hardened:** `ModelManifest.sha256(of:chunkBytes:)` now bounds each
  chunk's `Data` lifetime with an `autoreleasepool` (behavior unchanged).
- **UI races fixed:** catalog publishes `.verifying` before awaiting; W8 row
  hides Import while `.verifying`; W8 Import/Remove disabled during active
  generation; security-scoped access held for the whole stream.
- **Tests:** synthetic multi-chunk fixtures (no 2.23 GB pack in CI) — streaming
  success, SHA-mismatch cleanup, size-gate no-staging, re-import replace,
  catalog in-flight state, `ModelManifest` multi-chunk SHA regression.
- **CI:** PASS (run `31883123045`) — project-consistency ✓, iphone-build ✓,
  simulator-tests ✓ (270 tests, 0 failures, 14 real-pack-gated skips). Final
  acceptance = physical device retest (DEVICE_TESTS.md checklist).
