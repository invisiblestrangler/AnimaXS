# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after production sampler/support infrastructure and the J001 VAE audit. The next critical path is **J002/J003 full-frame Wan VAE decoding**, followed by J004/K002 full-pipeline wiring.

## Start here

Work in `/root/AnimaXS`. Preserve and never commit untracked `scripts/oracle_out/block0/` (~347 MB). Read `STATUS.md`, `TODO.md`, D052–D058 in `DECISIONS.md`, and `docs/VAE_FOLD_REPORT.md` before touching the VAE.

Normal CI baseline `31496280087` passed the generic iOS build, project consistency, and 92 simulator tests (11 expected fixture/real-pack skips). A final I002 full-pack regression rerun may be newer than this handoff; inspect the latest `I002 diffusion parity` run and record it before continuing. Delete its temporary release/tag/workflow after a green proof.

## Proven production inference path

These are real Metal paths, not CPU placeholders:

- Qwen F007 run `31491046871`: final same-W8 cosine `0.9999992405`, 5.19 s.
- Adapter G003 run `31492451065`: final same-W4 cosine `0.9999984505`, 1.46 s; padded tail exactly zero.
- DiT preparation run `31494520040`: residual cosine `0.9999999598`; timestep embedding/AdaLN effectively 1.0, 1.67 s.
- DiT block/final gates: E009 `0.9999999798`, H007 `0.9999999646`; full 28-block residual was finite.
- `DiffusionSampler` converts the adapter's fp32 `[512,1024]` output to fp16 once, then performs eight iterations of preparation → 28 blocks → final FLOW velocity → explicit fp32 `denoised = latent - sigma*velocity` → fp32 Euler. It rejects nonfinite post-step states and exposes block progress plus a post-step checkpoint callback.

The first complete I002 execution ran all 224 blocks and eight finite Euler updates in 96.40 s. Its source-BF16 observational comparison was final cosine `0.6919034`; individual callback cosines were `0.9670` down to `0.8665`. Do not confuse this cumulative quantized-vs-source quality measurement with graph parity: every component has a tight same-W4 gate, while recurrent W4 error compounds. The retained regression floor is `0.65`; decoded-image quality remains an L001 release decision.

The old golden `step_latents` is internally contradictory (D055): it cannot be post-step state, and treating it as denoised violates the final Euler invariant. Do not tune code to it. Regenerate the trace later with separately named callback `x`, `denoised`, post-step state, commit, and hashes. Full noise/final latent/context and compact anchors now live under `AnimaXSTests/Fixtures/Case1Binary` (A006).

## Immediate J002/J003 work

The model is pinned Wan `comfy/ldm/wan/vae.py`, not the Cosmos tokenizer. At T=1 with no feature cache:

- `conv2` runs first, then `decoder.conv1`, middle residual → one-head spatial attention → residual, 15 `decoder.upsamples` modules, then head RMSNorm → SiLU → convolution.
- Every executed rank-5 causal convolution folds to its **final temporal slice** because the source sees `[0,0,x]`. All 34 folds passed J001; temporal sums are wrong. The two decoder `time_conv` tensors do not execute at T=1.
- Native rank-4 resample/attention weights stay ordinary 2-D weights.
- Latent preprocessing is `z / 0.5 * std + mean`, channel-wise for the 16 chunk-0 values.
- Wan RMS normalization is `F.normalize` across C at each pixel, then `×sqrt(C)×gamma`; it is not GroupNorm. `VAENumerics` contains the tested CPU equations.
- Upsampling is integer 2× `nearest-exact` followed by the packed 3×3 convolution.
- Middle attention is one head over 64×64 = 4096 positions at C=384. Reuse/query-tile `AttentionExecutor`; never allocate a 4096×4096 full score matrix.

Implement the straightforward full-frame decoder first, stream/stage one fp16 weight at a time, and reuse activation scratch. Hosted simulator Metal can validate functional kernels and real-pack parity, so add a fixture-gated decoder test and use temporary release assets exactly like earlier gates. Do not implement VAE tiling unless A12 evidence later requires it. The required gate is decoded RGB maxAbs ≤0.05 and PSNR ≥30 dB versus canonical `decoded_rgb`.

`RGBConverter` already implements CHW `(rgb+1)/2`, clamp, RGBA8, and sRGB `UIImage`; J004 only needs the real decoder buffer wired and its fp32 storage released at the stage boundary.

## Support work already landed

- D005: exact built-in three-pack manifest plus incremental CryptoKit SHA-256.
- D006 partial: actor download/verify/install state machine, disk reserve, Application Support, backup exclusion and file protection. Synthetic CI is green. A real release download cannot pass until A005 license review authorizes `model-assets-v1`; do not publish packs early.
- I003: SplitMix64 + Box–Muller deterministic app RNG. It intentionally does not match ComfyUI; golden paths inject recorded noise.
- I004: versioned atomic JSON checkpoint with bit-exact Float32 latent and model hashes. Use `DiffusionSampler.stepCompleted` as the persistence boundary.
- J003 references and J004 RGB conversion have pack-free tests; production VAE execution remains missing.

## Hygiene and verification

Normal push CI must remain pack-free. Adding a source/resource requires changing `project.yml`, letting `bootstrap-project` regenerate the committed Xcode project, pulling the bot commit, then manually dispatching CI because bot pushes do not trigger another run.

Real-pack tests stay fixture-gated. Temporary releases, tags, and workflows must be removed immediately after proof. A12 memory, speed, thermal, page-cache and watchdog behavior remain device-only unknowns; hosted simulator success is functional evidence, not device acceptance.

After J002–J004, implement K002 stage ownership and L001 full canonical inference. A005 gates public model-pack release, not local or temporary private validation.
