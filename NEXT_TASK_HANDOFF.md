# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after F007/G003. The next dependency work is **I001/I002 sampler integration**, with **J001 VAE fold validation** safe to do in parallel.

## Proven baseline

Work in `/root/AnimaXS`. Preserve and never commit untracked `scripts/oracle_out/block0/` (~347 MB). No temporary fixture release/tag/workflow remains.

The production Metal paths are now hosted-parity green:

- Qwen F007 run `31491046871`: selected W8 embedding rows, 28 streamed layers, grouped 16Q/8KV attention, final cosine `0.9999992405`, 5.19 s.
- Adapter G003 run `31492451065`: selected W4 rows, six streamed ~9.6 MB blocks, exact GELU/bias and theta-10000 RoPE, final cosine `0.9999984505`, 1.46 s, padded tail exactly zero.
- DiT E009/H006/H007: real W4 block parity, serial 28-block loop, and final source-order velocity; H007 run `31488934459` cosine `0.9999999646`.

`QwenEncoderMetal.swift`, `LLMAdapterMetal.swift`, `DitForward.swift`, and `DiTFinalLayerExecutor.swift` are production paths. Their CPU counterparts are oracles only. Hosted simulator results establish functionality, not A12 memory/performance/thermal/watchdog acceptance.

## Immediate I001/I002 work

`EulerSampler.swift` was started after G003. Verify it in normal CI before checking I001: its nine Float32 sigmas must exactly equal golden `sigmas_comfy`:

```text
[1.0, 0.9546938, 0.90035903, 0.8339981, 0.7511211,
 0.64468634, 0.50298506, 0.30500895, 0.0]
```

The Euler contract is `x_next = x + (x-denoised)/sigma * (nextSigma-sigma)`. The DiT final layer returns raw FLOW velocity, not `denoised`. Preserve the pinned model-sampling conversion before Euler:

```text
velocity = DiT(...)
denoised = latent - sigma * velocity
latentNext = Euler(latent, denoised, sigma, nextSigma)
```

Do not silently pass raw velocity into `euler_step_f32`. Although the equations simplify algebraically to `latent + dSigma*velocity`, first match the source's explicit Float32 conversion/update order against the golden `step_latents`; only fuse after proving equivalent error. CFG is exactly 1, so execute one conditional model pass and no unconditional pass.

Implement an eight-step non-reentrant coordinator around the existing production pieces. Keep latent/velocity/denoised fp32 `[1,16,1,64,64]`, use the full adapter context `[512,1024]`, regenerate timestep/AdaLN inputs for each sigma, run `DitForward.executeVelocity`, reject NaN/Inf after every step, and expose step/block progress. For isolated parity, inject golden conditioning and `init_noise_randn`; compare every post-step latent to golden `step_latents`. The final element is `final_latent`. Do not add checkpoint persistence until the core I002 parity loop is green.

Before coding the loop, inspect the production availability of DiT input, timestep embedding, AdaLN and RoPE. Some of H001–H004 remain CPU oracles; any missing production-Metal bridge must use existing kernels/common linears and metadata ranges, never nested Swift matrices or full dequantized weights. This audit may reveal a small prerequisite that TODO currently compresses into I002; document it explicitly rather than hiding a CPU fallback.

## Parallel J001 work

Implement `scripts/validate_vae_fold.py` against the real VAE pack and pinned decoder source. Enumerate every decoder causal-convolution variant used at T=1, including initial cache/padding behavior and temporal kernel slices. Numerically compare the true 3-D operation with the proposed 2-D folded kernel on deterministic tensors. Record per-variant maxAbs/RMSE and either prove the fold or choose a true T=1 3-D path. Do not begin a full decoder on an assumed fold, and do not implement tiling without A12 evidence.

## Verification and hygiene

Normal push CI remains pack-free and must pass project consistency, generic iOS build, and all simulator tests. Real-pack tests stay fixture-gated; temporary release assets/workflows must be removed after proof. If adding a source file, change `project.yml`, let the pinned XcodeGen workflow regenerate the project, pull the bot commit, then dispatch CI because bot pushes do not trigger another workflow. Keep the retained real parity gates at cosine ≥ `0.999`.

After I002, proceed through J002–J004, A006 fixtures, and L001 full inference. A005 license review gates model pack release, not local/temporary validation.
