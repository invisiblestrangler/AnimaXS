# AnimaXS Handoff — Real-Device Stabilization (2026-08-15)

The first physical iPhone XS Max run (2026-08-15) is complete. It proved the app builds,
installs, and launches on-device, and it exposed a set of bugs that simulator/CI could not
catch. The stabilization pass (PR #8, branch `fix/real-device-stabilization`, base `main` @
`d2c4fad`) implements repository-side fixes for all of them.

## What changed (repository side, all in PR #8)

1. **Local-only model discovery (Fix A).** `ModelStore` now splits discovery
   (`discover`/`discoverInstalled`/`resolveInstalledModels`) from explicit acquisition
   (`download`/`repair`/`importPack`/`verifyExisting`). Discovery never touches the network;
   the mixed `prepare()` API was removed. A missing-model launch no longer auto-downloads.
2. **Verification receipts (Fix B).** After a full size+SHA install, a small JSON receipt is
   persisted next to the models directory. Normal launch trusts the file with stat/metadata
   checks only. A missing/stale receipt triggers exactly one full verification on the
   `ModelStore` actor (off the main actor) and writes a receipt — the migration path for
   pre-receipt installs. No more ~2.07 GB re-hash on every launch and no double-hash from
   refresh+resolve.
3. **Security-scoped Files import (Fix C).** `startAccessingSecurityScopedResource()` is held
   for the entire async size-check + SHA-256 + copy, released in a `defer`.
4. **`AnimapkError` → `LocalizedError` (Fix D).** `error.localizedDescription` now surfaces the
   real reason instead of "The operation couldn't be completed. (AnimaXS.AnimapkError error 3.)".
5. **Downloader hardening (Fix E).** Production downloader validates the final HTTP response
   (2xx required, redirects preserved), moves the completed temporary download into an
   app-owned staging file, and turns unknown disk capacity into a useful error instead of zero.
6. **Corrupt-destination replacement (Fix F).** `installVerified` replaces an existing corrupt
   destination via `replaceItemAt` (plain `moveItem` cannot overwrite); staging is always
   cleaned; receipts are written only after the final install succeeds.
7. **No silent Generate (Fix G).** `GenerationEligibility` is the single source of truth for the
   button state, the visible blocked reason, and the start guard. Every tap logs prompt/seed/
   models/thermal/process-memory/coordinator-state; "generation accepted" is logged before the
   coordinator starts and the post-start state after.
8. **Keyboard dismissal (Fix H).** `@FocusState` + keyboard-toolbar Done for prompt and number pad.
9. **Diagnostics redesign (Fix I).** Opening the screen runs only a cheap snapshot (device facts
   + model presence/size/receipt state — zero hashing). Explicit levels: basic self-tests,
   hardware tests (sequential, per-test progress, crash-marked; thermal recorded as passive
   telemetry only), deep SHA-256
   (opt-in, warns ~2.07 GB). `json(report:)`/`writeJSON(_:to:)` serialize an existing report —
   export runs zero tests.
10. **Crash localization (§14).** `DiagnosticRunMarker` persists the currently-running test in
    UserDefaults; after a native crash the next launch shows "Previous diagnostic run ended
    unexpectedly while: <test>".
11. **MPS A12 audit (§15).** The MPS precision diagnostic now computes rowBytes via
    `MPSMatrixDescriptor.rowBytes(fromColumns:dataType:)` (64-byte aligned — the old hand-
    computed 2-byte result rowBytes is an A12 assertion risk) and inspects command-buffer
    status/error after `waitUntilCompleted`.

## What is NOT yet proven

- **Physical A12 behavior.** Repository-side fixes are done and CI is being verified, but the
  Diagnostics crash root cause is NOT proven and the app has NOT been retested on the device
  with this build. Do not write "Diagnostics crash fixed" until the retest passes.
- The original explicit-Download failure root cause (it failed before the useful error was
  visible). Endpoint checks (all three assets reachable, sizes match the manifest) passed; the
  device retest must confirm Download works or shows a precise failure.

## Next steps

1. Merge PR #8 after CI is green (project-consistency + iphone-build + simulator-tests).
2. Run the ordered physical retest in `DEVICE_TESTS.md` (Phases 1–5) and record observed facts.
3. If Diagnostics crashes on-device again, relaunch and read the persisted "previous run ended
   during X" marker, then ship that evidence back here.

## Provenance

- Pinned source revision for model assets: `circlestone-labs/Anima` `f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b`
  (`split_files/diffusion_models/anima-turbo-v1.0.safetensors`, LFS SHA
  `c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174`).
- Model packs: `model-assets-v1` release; sizes/hashes in `ModelManifest` (verified against
  live endpoints 2026-08-15).
- Do not change the 0.65 regression floor or claim A12 validation from simulator CI.
