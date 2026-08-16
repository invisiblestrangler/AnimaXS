# AnimaXS TODO

## Current P0 — W8 final-residual boundary + fused AdaLN offset (fix/w8-range-and-fused-adaln) — source/CI DONE, physical retest PENDING

- [x] W8 block attention/activation numerics = legacy (`w8LegacyStabilized`, `.legacy`/`.legacy`)
- [x] W8 final residual ENTRY boundary = `.bf16RNEInFP32` via `FinalResidualBoundary` — large FP32
      residual rounded to BF16 RNE, retained in FP32, LayerNorm in FP32; NEVER FP16 (pre-fix
      `float_to_half` overflowed above 65,504 after block 28/28 on the physical XS Max)
- [x] W4 final residual boundary = `.fp16Legacy` (byte-for-byte unchanged)
- [x] Full `w8BF16Experimental` = still NOT production
- [x] Fused LayerNorm+AdaLN+to-half modulation offset = float elements (2048 via
      `fusedModulationElementOffset(columns:)`; old byte count 8192 walked past the 6144-float buffer)
- [x] `fusedNormModulation` = OFF by default (`currentBaseline.fusedNormModulation == false`)
- [x] W8 pack identity unchanged — no repack (sha256 8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130)
- [x] CI Gate A PASS (run 31950731666); CI Gate B PASS (run 31951006939 — W8 coherent image,
      latent_cos 0.9108)
- [ ] **Physical iPhone XS Max W8-v2 retest (user) — PENDING.** macOS Metal runner CI does NOT
      prove device correctness; perform per the P0 runbook §13.

## Device-stabilization patch (2026-08-16, branch fix/device-stability-no-checkpoint) — source DONE, validation PENDING

Physical iPhone XS Max run (2026-08-15) exposed: Files import permission error, generic
`AnimapkError error 3`, unwanted auto-downloads, no keyboard dismissal, silent Generate
no-op, Diagnostics crash, idle warmth. The stabilization patch on top of the A12
optimization work (`1756ab8`) addresses these and removes the ambiguities that made
optimization benchmarking unreliable.

### Source — DONE (committed; CI Gate A green on 31932848703)
- [x] Checkpoint/resume removed from production (no Resume UI, no per-step latent CPU snapshot)
- [x] W8-v2 production resolves to stabilized legacy numerics (`w8LegacyStabilized`);
      BF16 emulation stays experimental and unproven range-safe
- [x] Full-inference CI derives the production numerical policy from
      `DiTNumericsPolicy.fromVariantID`; `full-inference-refine.yml` asserts per-variant markers
- [x] P8 `directQuantized`/`hybrid` quarantined from production/device presets (~10× A12 slowdown)
- [x] P6 mmap no-copy disabled (A12 GPU page fault); not re-enabled without a hardware proof
- [x] Fatal Metal command-buffer faults poison generation context until restart
- [x] Manual Import local-only (no accidental Download); per-component model ops single-flight
- [x] Prompt/seed persist across relaunch (`generation.lastPrompt` / `generation.lastSeed`)
- [x] Fresh Generate owns the image/metrics surface; Diagnostics preset marker cannot lie;
      central compatibility validator; monitor probe labels disambiguated
- [x] Regression tests for the above (GenerationCoordinatorTests, ModelStoreTests,
      InferenceOptimizationConfigTests, WeightStreamerTests, GenerationEligibilityTests)

### CI / validation — PENDING
- [x] CI Gate A PASS (project-consistency, iphone-build, simulator-tests — run 31932848703)
- [ ] CI Gate B (final normal CI after all stabilization tasks)
- [ ] Full-inference W4/W8 gate under the corrected production policy (plan §15)
- [ ] **Physical iPhone XS Max retest (user) — REQUIRED before claiming device fixes.**
      CI green on simulator/macOS does NOT prove device success. Use the stabilization-plan
      §17 configuration: W8-v2, baseline preset, `dequantizedMPS`, no-copy unavailable,
      checkpointing absent.

## Closed: real-device stabilization (2026-08-15, PR #8) — historical, superseded by the patch above

The PR #8 stabilization line (`fix/real-device-stabilization`) is superseded for device
testing by `fix/device-stability-no-checkpoint`. Its fixes are recorded for history:

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
  `json(report:)`/`writeJSON(_:to:)` run zero tests; thermal state recorded as passive telemetry only
- [x] Crash marker: `DiagnosticRunMarker` persists running test; "Previous run ended during X" warning
- [x] MPS audit: `MPSMatrixDescriptor.rowBytes(fromColumns:dataType:)` (64-byte aligned) +
  command-buffer status/error inspection after `waitUntilCompleted`
- [x] Resource instrumentation: per-hardware-test thermal before/after, process memory, elapsed
- [x] Regression tests (ModelStoreTests/DiagnosticsTests/GenerationEligibilityTests/
  DiagnosticsRunMarkerTests/SmokeTests)
- [x] Endpoint checks: all three `model-assets-v1` assets → 206 partial, content-length matches manifest
- [x] Docs updated (README/STATUS/DEVICE_TESTS/NEXT_TASK_HANDOFF/DECISIONS/TEST_MATRIX)

### PR #8 CI (historical)
- [x] project-consistency PASS (bootstrap bot commit `92b544c`)
- [x] iphone-build PASS
- [x] simulator-tests PASS (incl. new stabilization tests)
- [x] PR #8 merged to main after green

### Physical retest of PR #8 (historical — superseded; the new patch is what the user tests)
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
      — after the stabilization retest passes
- [ ] Residual W8 texture investigation if quality work resumes (see NEXT_TASK_HANDOFF.md)
- [ ] P6 no-copy re-enable only with a future GPU-read hardware proof
