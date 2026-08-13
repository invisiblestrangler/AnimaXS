# Current Project Status

- **Base main:** `45c28f44697253e20385edeab662c8db8098c55f`.
- **Investigation branch/worktree:** `investigate/dit-quality-runtime` at
  `/root/AnimaXS-quality`.
- **Current phase:** source-oracle trajectory parity (the decisive DiT
  forward-vs-source experiment, run `31724606040` in flight).
- **BF16 matrix result (run `31711438180`):** BF16 arithmetic is contributory
  but NOT sufficient. bf16+golden-Qwen: latent `0.8344`, RGB `0.8420`;
  legacy+golden: `0.8120`/`0.7924`; legacy+production (run #29 control):
  `0.8120`/`0.7921`; bf16+production: `0.8086`/`0.7853`. The exact-8px grid
  carrier is unchanged across ALL variants (~0.0132–0.0140 vs reference
  `0.000055`, ~245–254x) and the image is still visibly etched/woven.
- **Qwen branch closed (Case E):** golden Qwen context ≈ production Qwen under
  legacy arithmetic (`0.8120` vs `0.8120`); the validated W8/fp16 Qwen path is
  not the grid cause.
- **Sampler/scheduler audit cleared (D090):** Swift sigma schedule == golden
  trace exactly; Euler/denoised/initial-latent/final-handoff match the pinned
  source (old-form flux_time_shift shift=3.0/timesteps=1000 vs the pinned
  snapshot's exp-form — a ComfyUI-version difference, not a Swift bug).
- **Source oracle built (`scripts/dit_source_oracle.py`):** runs the pinned
  Anima model (predict2.py + position_embedding.py + anima/model.py LLMAdapter)
  in torch on the native-bf16 safetensors; per-step velocity parity vs Swift's
  captured trajectory + end-to-end source run vs the golden fixtures.
- **Carrier baseline (`scripts/measure_grid_carrier.py`):** generated total
  `0.01345` vs reference `0.0000548` (~245x; dominant horizontal-stripe bins
  ~1082x).
- **Provenance:** workflow-injected `provenance.json` (commit/run_id/attempt/
  ref/variant + Qwen/DiT/VAE identity incl. packed SHA-256/bytes) — artifacts
  are self-describing; metrics.txt carries real commit/run IDs.
- **Completion truth:** old 0.65 regression floors are not the acceptance
  gate; no current image is reference-comparable (severe woven/etched grid).
