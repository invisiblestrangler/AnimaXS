# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Implementation, model distribution (L002), and CI-verifiable inference work are COMPLETE. Normal CI is green on the final HEAD. L003 full-inference run and final documentation reconciliation are in progress. Physical A12 acceptance remains explicitly PENDING (no iPhone XS Max hardware available).
- **Final HEAD:** `bbf5cc1` (main, origin/main). Last source commit `3dae37d`; regenerated `AnimaXS.xcodeproj` bot commit `bbf5cc1`.
- **Normal CI:** run `31634877232` (workflow_dispatch on `bbf5cc1`): **SUCCESS** — project-consistency ✓, iphone-build ✓, simulator-tests ✓ (151 tests, 13 expected fixture-gated skips, 0 failures). (An earlier run `31633936959` failed only `DiagnosticsTests.testSelfTestsArePassOrSkippedNotFailByDefault` because the runner VM's simulator Application Support retained valid packs from a prior pack-backed workflow, so pack validation legitimately returned PASS — an environmental flake, confirmed by the clean re-run; see D072.)
- **L002 release verification:** `model-assets-v1` — all three packs independently re-downloaded unauthenticated and SHA-256-verified; manifest, LICENSE, NOTICE confirmed (see DECISIONS D070).
- **What works (2026-08-12):** K001 full UI; K002 3-pack generation architecture (seed→RNG, stage lifetime, recoverable Metal); I004 real checkpoint resume; K003 lifecycle (background safe-cancel + foreground resume); K004 memory/thermal policy; D006 ModelStore repair/3-pack/import; J003 production primitive coverage; K005 diagnostics; full production pipeline (Qwen/adapter/DiT/sampler/VAE) CI-validated. L001 final-image RGB8 regression fixture + gate added (`case1_decoded_rgb8.bin`, cosine ≥ 0.9).
- **A005 license (D069):** non-commercial redistribution of the three converted/quantized packs is permitted with license copies + notices (`MODEL_LICENSE.md`, `MODEL_NOTICE.txt`, `docs/model-licenses/`). Commercial use remains blocked without separate licenses.
- **Known incomplete / in progress:** L003 manual full-inference workflow run (fixed Bash-3.2 pack-download loop; result pending Metal availability). J005 tiled VAE deferred pending physical-device evidence.
- **Device-only unknowns (NOT CI-provable):** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, thermal stability, second-generation stability, real-device install/launch. Hosted Metal functional success does not answer them.
- **Next tasks:**
  1. Run L003 full-inference workflow (PASS or explicit SKIPPED_NO_METAL).
  2. Finalize README/TEST_MATRIX/DEVICE_TESTS/STATUS/handoff + M003 report.
  3. Physical A12 acceptance — build in Xcode, install on iPhone XS Max, record timings/memory/thermal.
