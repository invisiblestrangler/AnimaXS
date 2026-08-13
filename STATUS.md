# Current Project Status

- **Base main:** `45c28f44697253e20385edeab662c8db8098c55f`.
- **Investigation branch/worktree:** `investigate/dit-quality-runtime` at
  `/root/AnimaXS-quality`.
- **Current phase:** BF16 compute-mode quality matrix (handoff §8/§9).
- **Latest evidence — run `31701142683` (run #29, commit `357bfb1`) is a
  LEGACY control, NOT a BF16 run:** `FULL_ACTIVATION_NUMERICS=legacy`,
  `FULL_ATTENTION_NUMERICS=legacy`, `FULL_QWEN_CONTEXT=production`. Metrics:
  latent cosine `0.8120` (RMSE `0.8696`, maxAbs `3.4669`), RGB cosine `0.7921`
  (RMSE `0.4391`, MAE `0.3708`), 384.10 s total. This is the highest-fidelity
  legacy baseline. The true `bf16_compute` experiment had NOT run as of this
  status.
- **Defect baseline (run #29 PNGs, `scripts/measure_grid_carrier.py`,
  2026-08-13):** generated exact-8px carrier total `0.01345363`, reference
  `0.00005484`, ratio `~245x`; horizontal-stripe bins `1082x`, vertical-stripe
  `113x`. (Calibration differs from the handoff's quoted `0.01228284`/`347.6x`
  because normalization/axis conventions differ; same strong periodic
  signature, orders of magnitude above reference.)
- **Provenance fixed in workflow commit:** candidate artifacts now carry
  `provenance.json` (commit/run_id/run_attempt/workflow/ref/variant + exact
  Qwen/DiT/VAE pack identity incl. source revision + packed SHA-256/bytes) and
  `metrics.txt` records real commit/run ID instead of `<injected-by-workflow>`.
- **Workflow efficiency fixed:** `quality-investigation.yml` is manual-dispatch
  only (push trigger removed); `quality-qwen-fp16.yml` packs once and runs the
  macOS full-image variants in a parallel matrix; pack reuse via
  `pack_artifact_run_id`/`pack_artifact_sha` skips repacking.
- **Historical decisions:** D082–D087 recorded (attention FP32 rejected,
  block parity tight, BF16 residual-boundary rejected, source-FP16 ceilings
  leave the grid). Qwen golden-context isolation run `31691143106` was block
  captures only — no comparable full-image golden result existed, so the new
  matrix includes `legacy-golden-qwen` as a directly comparable control.
- **Completion truth:** old 0.65 regression floors are not the acceptance
  gate; no current image is reference-comparable (severe woven/etched grid).
