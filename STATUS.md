# STATUS — AnimaXS (keep short & current)

- **Current milestone:** The project has entered the runbook-defined refine-and-improve phase. Legacy three-pack inference is connected and captured, but its patterned/dull W4 image is not the refinement target. Work is now focused on reproducible DiT W4-v2/W8 packing, one shared W4/W8 runtime graph, and canonical visual comparison.
- **Final HEAD:** `bbf5cc1` (main, origin/main). Last source commit `3dae37d`; regenerated `AnimaXS.xcodeproj` bot commit `bbf5cc1`.
- **Normal CI:** run `31634877232` (workflow_dispatch on `bbf5cc1`): **SUCCESS** — project-consistency ✓, iphone-build ✓, simulator-tests ✓ (151 tests, 13 expected fixture-gated skips, 0 failures). (An earlier run `31633936959` failed only `DiagnosticsTests.testSelfTestsArePassOrSkippedNotFailByDefault` because the runner VM's simulator Application Support retained valid packs from a prior pack-backed workflow, so pack validation legitimately returned PASS — an environmental flake, confirmed by the clean re-run; see D072.)
- **L002 release verification:** `model-assets-v1` — all three packs independently re-downloaded unauthenticated and SHA-256-verified; manifest, LICENSE, NOTICE confirmed (see DECISIONS D070).
- **What works (2026-08-12):** K001 full UI; K002 3-pack generation architecture (seed→RNG, stage lifetime, recoverable Metal); I004 real checkpoint resume; K003 lifecycle (background safe-cancel + foreground resume); K004 memory/thermal policy; D006 ModelStore repair/3-pack/import; J003 production primitive coverage; K005 diagnostics; full production pipeline (Qwen/adapter/DiT/sampler/VAE) CI-validated. L001 final-image RGB8 regression fixture + gate added (`case1_decoded_rgb8.bin`, cosine ≥ 0.65 calibrated D074).
- **A005 license (D069):** non-commercial redistribution of the three converted/quantized packs is permitted with license copies + notices (`MODEL_LICENSE.md`, `MODEL_NOTICE.txt`, `docs/model-licenses/`). Commercial use remains blocked without separate licenses.
- **L003 full-inference:** workflow fixed (Bash-3.2 + bundled-resource pack injection). Final run `31639624310` (commit `494ffb1`) **PASS** — Metal smoke PASS, 3 packs verified, full production inference on the simulator in 161.28 s: latent cosine 0.6946 (≥ 0.65 floor), RGB cosine 0.7035 (≥ 0.65 gate, D074); FULL_INFERENCE=PASS. Metrics bit-identical to calibration run `31638695924` (deterministic).
- **Known incomplete / in progress:** none CI-verifiable. J005 tiled VAE deferred pending physical-device evidence.
- **Device-only unknowns (NOT CI-provable):** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, thermal stability, second-generation stability, real-device install/launch. Hosted Metal functional success does not answer them.
- **Next tasks:**
  1. Complete the v2 packer/verifier and W8 runtime generalization.
  2. Pack and publish exact-source W4-v2/W8 on separate HF-backed CI jobs.
  3. Run both canonical image captures and choose the visually acceptable variant.
  4. Physical A12 acceptance remains outside this refinement cycle and is still pending.
