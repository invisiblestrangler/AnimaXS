# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-12 (final completion stage: L002 verified, L003 workflow fixed, docs reconciled).

## Repository state

```
branch: main
HEAD: bbf5cc1 (chore: regenerate AnimaXS.xcodeproj from project.yml)
origin/main: bbf5cc1
parent source commit: 3dae37d (fix(L001/L003): Bash-3.2-safe pack download loop + RGB8 fixture)
working-tree: clean except untracked README.md (M001) and scripts/oracle_out/block0/ (gitignored oracle artifacts — do not commit)
```

## Last known green CI

```
Normal CI (workflow_dispatch on bbf5cc1): run 31634877232 SUCCESS
  ✓ project-consistency
  ✓ iphone-build
  ✓ simulator-tests (151 tests, 13 expected fixture-gated skips, 0 failures)
The run 31633790701 (on 3dae37d) failed ONLY project-consistency because the
xcodeproj had not yet been regenerated to include the new case1_decoded_rgb8.bin
fixture; the bootstrap-project run 31633797289 regenerated it and pushed bbf5cc1.
Run 31633936959 failed ONLY DiagnosticsTests pack-validation (environmental
flake — leftover packs in runner simulator Application Support; D072); clean
re-run 31634877232 passed.
```

## Completed work this stage

1. **L002 (commit none needed — verification only):** `model-assets-v1` release independently verified as an unauthenticated user: all three packs downloaded (one at a time) and SHA-256 matched the production `ModelManifest.swift` and the release `model-manifest.json`; LICENSE/NOTICE byte-identical to committed copies. Evidence in DECISIONS D070.
2. **Commit `3dae37d`** — `fix(L001/L003)`: full-inference.yml Bash 3.2 associative-array bug fixed (macOS Bash 3.2 has no `declare -A`); replaced with a `printf | while read` line-oriented loop and hardened curl (`--fail --location --retry --retry-all-errors`), verifying exact filename + byte count + SHA-256.
3. **Commit `3dae37d` (RGB gap, L001):** added canonical `case1_decoded_rgb8.bin` (512×512×3 UInt8, SHA `a396c4ae…7019`) derived from the canonical NPZ `decoded_rgb` via the exact production display transform; `FullInferenceTests` now computes FULL_RGB_COSINE/RMSE/MAE/MAXABS with a cosine ≥ 0.9 gate; `scripts/extract_golden_fixtures.py` regenerates it. Evidence in DECISIONS D071.
4. **Commit `bbf5cc1` (bot):** regenerated `AnimaXS.xcodeproj` to include `case1_decoded_rgb8.bin` as a test resource.

## Key architecture decisions

- **Three production packs** (K002): qwen text encoder, DiT (serves BOTH adapter and sampler), VAE. No separate adapter pack in production inference.
- **L001 RGB regression**: `case1_decoded_rgb8.bin` is the canonical final-image reference; both production RGBA8 and the reference go through the identical `(v+1)*0.5→*255→round` display transform, so the cosine/RMSE/MAE/maxAbs comparison is apples-to-apples. Gate: cosine ≥ 0.9 (DECISIONS D071).

## What cannot be proven in this phase

```
physical iPhone XS Max launch
real A12 GPU behavior
actual A12 Metal throughput
actual device peak RSS
iOS jetsam behavior
thermal behavior
real-device GPU memory reclamation timing
real-device second-generation memory stability
```

Progress files say: **CI-validated; physical A12 acceptance pending.**

## Remaining tasks (ordered by dependency)

1. **L003** — Run the `full-inference` workflow (`gh workflow run full-inference.yml --ref main`) and record PASS or explicit `SKIPPED_NO_METAL` + metrics (commit SHA, run ID, runner image, Xcode, Metal smoke, 3-pack verification, model load, 8 diffusion steps, final latent cosine/RMSE, RGB parity metrics, image-health, timings, overall result).
2. **M001 README** — `README.md` is drafted (untracked); review/commit it.
3. **M002** — Update `TEST_MATRIX.md` (remove stale run IDs) and confirm `DEVICE_TESTS.md` still clearly separates CI-proven vs A12-pending.
4. **M003** — Final report with the required fields.
5. **Physical A12 acceptance** — Build in Xcode, install on iPhone XS Max, record stage timings, peak memory, second-generation stability, thermal behavior. Tune VAE tile size (J005) only if device measurements justify it.

## First command for next agent

```bash
cd /root/AnimaXS
git pull --rebase origin main
gh run list --limit 5
# Confirm normal CI is green on the latest commit, then:
gh workflow run full-inference.yml --ref main   # L003
# Collect the run's FULL_INFERENCE result + metrics from the log/job summary.
```
