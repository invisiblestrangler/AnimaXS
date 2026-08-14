# AnimaXS 8px Grid — Final Root Cause Report

**Date:** 2026-08-14
**Branch:** `investigate/animapk-cuda-parity`
**Clore:** order 2024354 (RTX 3060 12 GB)
**Decision:** D097 (DECISIONS.md)

## Root cause
The custom CUDA/Metal runtime decoded raw **sampler-space** latents as if they
were **VAE-space** latents, omitting ComfyUI's model latent-format boundary
transform. ComfyUI applies `Wan21.process_out` to the sampler return inside
`CFGGuider.inner_sample` (samplers.py:1238) before the workflow/VAE receives
the latent; Anima is registered `latent_format = Wan21` (supported_models.py).
Missing that per-channel affine (`vae = sampler*std + mean`, scale 1.0) meant
the VAE decoded 2–3× too-small, wrongly-centered latents — the visible 8px
woven grid.

## Proof
- golden callback raw (`step_latents[7]`) vs golden `final_latent`:
  cos **0.836949**, RMSE 0.844841 (the old "golden inconsistency", D096)
- after exact pinned Wan21 `process_out`: cos **1.000000**, RMSE **0.000000**,
  maxAbs **0.000000** — bit-exact ⇒ `final_latent == process_out(step7)`
- decode through the validated CUDA VAE (exact-8px carrier vs clean ref
  0.0000548):
  - golden step7 raw: **254.4×** → converted: **1.0×** (rgb_cos 0.999989)
  - double-apply control (golden + process_out): 2.8× ⇒ exactly-once semantics
- fixed-pipeline rerun (real DiT runs, boundary applied in code):

| Lane | latent cos before→after | carrier before→after |
|---|---|---|
| G1 official BF16 | 0.811 → **0.971** | 246.7× → **1.3×** |
| G2 FP16-all | 0.813 → **0.973** | 246.3× → **1.3×** |
| G3 W8 | 0.809 → **0.969** | 244.5× → **1.3×** |
| G4 Metal fp16-all | 0.812 → **0.973** | 245.6× → **1.3×** |
| G4 Metal w8 | 0.810 → **0.970** | 245.5× → **1.3×** |
| G4 Metal w4 | 0.660 → **0.851** | 493.3× → **1.8×** |

Vision review (mimo-v2.5): no grid/checkerboard in ANY fixed lane, including
W4; clean anime output matching the reference.

## Code fix
- **CUDA:** `scripts/animapk_cuda/wan21_latent_format.py` (pinned constants,
  `anima_sampler_to_vae` / `anima_vae_to_sampler`, torch variant);
  `grid_repro.py` converts `sampler_latent` → `vae_latent` exactly once at the
  decode boundary (explicit naming; G0 golden stays as-is — it is already
  VAE-space).
- **Metal/iOS:** `AnimaXS/Runtime/VAE/Wan21LatentFormat.swift` (pinned
  constants, `processOut`/`processIn`, in-place `applyProcessOutInPlace`);
  `GenerationEngine.generate` applies it exactly once before
  `VAEDecoder.decode`; checkpoints remain sampler-space.
  `FullInferenceTests` converts before the latent regression (golden is
  VAE-space) and before decode.
- **Tests:** `scripts/test_wan21_latent_format.py` (18.1 formula, 18.2
  inverse, 18.3 channel mapping) — PASS locally;
  `Wan21LatentFormatTests.swift` (18.4 known-vector golden crop, zero==mean,
  distinct-channel mapping, inverse, buffer parity) — CI pending.

## Remaining quality differences
- **FP16/BF16:** none material — carrier 1.3×, cos 0.97.
- **W8:** same as FP16 — carrier 1.3×, cos 0.969 (W8-v2 remains the selected
  production candidate).
- **W4:** carrier 1.8× (grid gone; no regular 8px pattern), latent cos 0.851 —
  ordinary W4 quantization degradation, documented separately (D079-D081
  rejected W4 as the production candidate).

## CI
- PR #7 (branch `investigate/animapk-cuda-parity`):
  - project-consistency: PASS
  - iphone-build: (re-run after pointer-binding fix)
  - simulator-tests: (re-run after pointer-binding fix)
- NOTE: this run has no physical iPhone XS Max — "macOS/CI validated" only;
  physical A12 device validation remains a separate acceptance item.

## Durable artifacts
- HF repo: `ScalingBiz/AnimaXS-investigation-artifacts` (private dataset)
  — `experiments/2026-08-14_wan21-process-out-fix/` (bundle built on Clore at
  `/workspace/out/hf_wan21_fix`, 27 files + SHA256SUMS; **upload blocked on
  HF token — see session note**)
- Clore paths: `/workspace/out/wan21_boundary_proof/` (zero-inference proof),
  `/workspace/out/grid_repro_fixed/` (fixed pipeline, manifest.json)
- Repo: `scripts/animapk_cuda/{wan21_boundary_proof,wan21_latent_format,grid_repro}.py`,
  `scripts/test_wan21_latent_format.py`,
  `AnimaXS/Runtime/VAE/Wan21LatentFormat.swift`, `AnimaXSTests/Wan21LatentFormatTests.swift`

## Repo
- branch: `investigate/animapk-cuda-parity`
- final HEAD: `7092400` (docs follow)
- working tree: clean (except untracked `HERMES_GRID_ROOT_CAUSE.md`)

## Clore
- terminated: **no** (upload + any residual CUDA work pending; per K's new
  instructions the instance is to be terminated once all CUDA work is durable —
  will terminate after HF upload completes)
