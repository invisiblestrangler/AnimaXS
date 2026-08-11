# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after F007/G003/J001. The next dependency work is **I001/I002 sampler integration**, with **J002/J003 VAE decoding** safe to do in parallel.

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

## Parallel J002/J003 work

J001 is complete and corrected a dangerous stale assumption. The model uses pinned Wan `comfy/ldm/wan/vae.py`, not Cosmos tokenizer replication semantics. For uncached T=1, causal zero padding makes every executed rank-5 convolution fold to its **final temporal slice**. All 34 decoder/post-quant tensors pass; temporal sums are wrong for all 32 kt=3 weights. The two decoder `time_conv` tensors are not executed at T=1 because no feature cache exists. Read `docs/VAE_FOLD_REPORT.md` and D052 before implementing.

Build the straightforward full-frame decoder from the real VAE inventory and pinned Wan source. Apply Wan21 decode normalization first (`z / 0.5 * std + mean`, chunk-0 values for T=1), then `conv2`, decoder conv/residual/attention/resample/head in exact order. Native rank-4 resample/attention projections remain 2-D; executed rank-5 weights use final-slice folds. Preserve RMS norm vs GroupNorm semantics from the actual source—the current pack names (`*.gamma`, attention `to_qkv/proj`) reveal this is the Wan decoder and older prose claiming “no attention” is wrong. Validate layer shapes and final RGB against the golden before considering tiling.

## Verification and hygiene

Normal push CI remains pack-free and must pass project consistency, generic iOS build, and all simulator tests. Real-pack tests stay fixture-gated; temporary release assets/workflows must be removed after proof. If adding a source file, change `project.yml`, let the pinned XcodeGen workflow regenerate the project, pull the bot commit, then dispatch CI because bot pushes do not trigger another workflow. Keep the retained real parity gates at cosine ≥ `0.999`.

After I002, proceed through J002–J004, A006 fixtures, and L001 full inference. A005 license review gates model pack release, not local/temporary validation.
