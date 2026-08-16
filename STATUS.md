# Current Project Status

- **Base main:** `d2c4fad` (Wan21 latent-format boundary fix, 8px grid root cause closed).
- **Active branch:** `fix/device-stability-no-checkpoint` — device-stabilization patch on
  top of the A12 optimization work (`1756ab8`, PR #17). Supersedes the PR #8
  stabilization line for device testing.
- **Current phase:** stabilization patch committed (checkpoint removed, W8 legacy
  numerics, P8/P6 quarantined/disabled, fatal-Metal poisoning, Import local-only,
  prompt/seed persistence); CI Gate A green. Physical XS Max validation and the
  W4/W8 full-inference gate are **PENDING**.

## Device-stabilization patch (2026-08-16, `fix/device-stability-no-checkpoint`)

Committed on top of `1756ab8`. These are the ACTUAL decisions now in the code; they are
proven at the source/CI level only — no physical-device success is claimed until the
user re-tests on the XS Max.

1. **Checkpoint/resume removed from production** (`90ca169`) — `Checkpoint.swift` /
   `CheckpointStore.swift` and all wiring (Resume UI, cold-launch load, `canResume`,
   `checkpointingEnabled` metrics, per-step latent snapshot) deleted. `GenerationEngine`
   always starts at step 0; the 256 KiB fp32 per-step CPU latent readback is gone. No
   per-step latent CPU snapshot remains in production. Background/memory warning =
   cooperative cancel.
2. **W8-v2 production temporarily uses stabilized legacy numerics** (`82026cc`) —
   `DiTNumericsPolicy` = `w4Legacy` / `w8LegacyStabilized` / `w8BF16Experimental`;
   `fromVariantID("w8-v2")` → `w8LegacyStabilized` (legacy attention/activation). The
   FP16-backed BF16 emulation (`w8BF16Experimental`) remains experimental and is NOT
   claimed range-safe internally; only explicit diagnostics reach it.
3. **Full-inference CI executes the production policy** (`27d3eb7`) —
   `testCanonicalProductionInference` derives numerics from
   `DiTNumericsPolicy.fromVariantID(ANIMAXS_DIT_VARIANT)` via
   `DiffusionSampler.resolvedNumerics` (the production resolver); `full-inference-refine.yml`
   asserts the per-variant marker (w4-v2 → `w4Legacy`, w8-v2 → `w8LegacyStabilized`).
4. **P8 `directQuantized`/`hybrid` quarantined** (`e536bd7`) — physical A12 measurement
   showed ~10× slowdown vs `dequantizedMPS` (performance regression, not proven
   incorrect). Device settings sanitize/reject it; `directQGEMMCandidate`/`allCandidate`
   force `dequantizedMPS`; Diagnostics hides them with a visible note. Kernel remains
   research code.
5. **P6 mmap no-copy disabled** (`b14b88b`) — physical A12 GPU page fault
   (`kIOGPUCommandBufferCallback` ErrorPageFault). `noCopyWeightSource` cannot be true in
   production/device settings; not re-enabled without a future GPU-read hardware proof.
6. **Fatal Metal faults poison the generation context** (`b14b88b`) —
   `metalContextPoisoned` on `.pageFault`/`.invalidResource`/`.internal` (or narrow IOGPU
   text fallback): state fails with "Fatal GPU fault. Restart AnimaXS before generating
   again.", Generate blocked until restart. No context recreate/retry. Cooperative
   cancellation and ordinary failures never poison.
7. **Manual Import is local-only** (`f64eb5c`) — `.borderless` buttons so Import can
   never cross-trigger Download; `ModelStore` per-component single-flight guard
   (`activeOperations`); app launch never auto-downloads model packs.
8. **Prompt/seed persist across relaunch** (`a4a1c60`) — `generation.lastPrompt` /
   `generation.lastSeed` (`@AppStorage`), Randomize included.
9. Also on this branch: fresh Generate owns the image/metrics surface (`b25b845`);
   Diagnostics preset marker cannot lie after manual edits + central compatibility
   validator (`f647908`); numerical-monitor gate/add probe labels disambiguated
   (`605300f`).

### Status

- [x] Stabilization source committed; CI Gate A green (run `31932848703`:
  project-consistency, iphone-build, simulator-tests)
- [ ] CI Gate B (final normal CI after all stabilization tasks)
- [ ] Full-inference W4/W8 gate under the corrected production policy
- [ ] **Physical iPhone XS Max retest (user) — PENDING.** Simulator/macOS CI green does
      NOT prove device success. Use the stabilization-plan §17 configuration (W8-v2,
      baseline preset, `dequantizedMPS`, no-copy unavailable, checkpointing absent).

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
- **Superseded** by the W8-as-normal-pack refactor below.

## W8 v2 is now a normal DiT pack (main page), not a Diagnostics experiment (2026-08-15)

- **Branch:** `fix/w8-import-refactor` (off `origin/main` `15f9c81`)
- **Decision (K):** "have it load the normal way like w4 at main page and
  remove it from the diagnostics page. The user can then import either w4 or
  w8-v2."
- The experimental W8 special path is **removed** (deleted
  `ExperimentalDiTPackStore`/`ExperimentalDiTPackCatalog`/
  `ExperimentalDiTManifest` and the `DiTPackVariant` picker/substitution).
- W8-v2 is now an **alternate variant** of the production `.dit` slot in
  `ModelManifest`: whichever pack (W4 or W8-v2) you import into the main-page
  Models row is the DiT a generation uses. Importing W8 over W4 replaces it;
  discovery is receipt-cheap for either variant.
- The **single-pass streaming import** (the crash fix) now lives in the normal
  `ModelStore.verifyAndStage` (1 MiB chunks, autoreleasepool per chunk),
  shared by download and manual import. The old full-SHA + `copyItem` two-pass
  is gone from the normal path too.
- Generation config no longer carries a DiT variant; checkpointing is always
  on; run metrics report the actual DiT pack filename. *(Historical for this
  2026-08-15 branch — checkpointing is now REMOVED on
  `fix/device-stability-no-checkpoint`; see the section at the top of this file.)*
- **CI:** PASS (run `31888994142`) — project-consistency ✓, iphone-build ✓,
  simulator-tests ✓ (262 tests, 0 failures, 14 real-pack-gated skips).
  Final acceptance = physical device retest (DEVICE_TESTS.md checklist).
