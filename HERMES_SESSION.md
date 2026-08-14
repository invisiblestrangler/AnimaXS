# HERMES_SESSION.md

# Current state

Branch: investigate/animapk-cuda-parity (HEAD a6d4657, clean, pushed)
origin/main: 686e5a8 (infra-only workflow registration; do not dirty)
Working tree: clean
Last known green normal CI: 31671071004 (main)

Clore status: ACTIVE order 2024354 (RTX 3060 12GB, $0.70/h). SSH via
  ssh -i /root/key root@n1.msk.cloreai.ru -p 1500
  repo at /workspace/repo (NOW UNSHALLOWED; remote.origin.fetch fixed to
  "+refs/heads/*:refs/remotes/origin/*"; commit identity set).
GPU: RTX 3060 12GB, torch 2.7.1+cu126, venv at /workspace/venv
Clore free disk: 755G avail
Clore environment: /workspace/{repo,source,fixtures,packs,comfy-ref,out}
HF artifact repo: ScalingBiz/AnimaXS-investigation-artifacts (dataset, private)
HF latest experiment: experiments/2026-08-14_grid-repro (upload in progress)
HF pack repos: ScalingBiz/AnimaXS-DiT-W8, -W4, -FP16-ALL, AnimaXS-VAE-XSMAX-FP16

## THE HEADLINE RESULT (2026-08-14)
The 8px grid is REPRODUCED on CUDA/Clore in image space. Decoder is clean;
the grid is ENCODED IN THE FINAL LATENT produced by the real pinned CUDA graph
(official BF16 source, 8-step Euler), completely off Apple hardware.

Grid carrier (exact-8px normalized Fourier energy) vs reference:
  G0 golden latent -> CUDA VAE:   0.0000548  (1.0x, CLEAN)
  G1 official BF16 CUDA:          0.01353    (~246.7x, GRID)
  G2 FP16-all CUDA:               0.01351    (~246.3x, GRID)
  G3 W8 CUDA:                     0.01341    (~244.5x, GRID)
  G4 Metal fp16-all -> CUDA VAE:  0.01347    (~245.6x, GRID)
  G4 Metal w8 -> CUDA VAE:        0.01346    (~245.5x, GRID)
  G4 Metal w4 -> CUDA VAE:        0.02705    (~493x, GRID)
  (reference/baseline == 0.0000548)

Decode the same canonical reference: G0 rgb_cos vs decoded_rgb 0.999989.

## Root-cause isolation
1. G0 clean => VAE/PNG decode path is NOT the grid source (Case 2).
2. Decoder-sensitivity: golden + k*delta for k=0.25/0.5/0.75/1.0 gives carrier
   3.3x/18.3x/72.4x/246.7x; a NORM-MATCHED RANDOM perturbation gives 2.1x.
   => the grid is structured error already in the latent, NOT decoder
   amplification. The VAE is clean.
3. denoised_step0 already grids (415x) => the grid enters at the FIRST model
   forward, not by Euler accumulation.
4. harness trajectory matches golden's own (mislabeled, D055) step_latents at
   0.974 (step 7), but NOT golden final_latent (0.811).
5. Block-0 step-7 probe: raw_sigma + ctx512 reproduces golden block_00_out at
   0.999064 (RMSE 0.0805). Model forward is faithful; Euler matches ComfyUI
   sample_euler algebraically. => NOT a model-forward or Euler bug.

Current hypothesis: the grid-producing trajectory is faithful to the golden's
recorded trace; the golden's CLEAN final_latent (and reference image) came from
a different capture than its grid-carrying step_latents. The produced latent
diverges from golden final_latent (0.811) starting at step 1. Need to resolve
whether the golden final was produced by a differing ComfyUI path (trace_anima
callback arg mismatch per D055, or conditioning/context difference).

## Key discoveries
- Clore repo was a SHALLOW clone (caused stale fetches); unshallowed.
- Clore remote.origin.fetch refspec only tracked dit-quality-runtime; fixed.
- VAE pack qwen-image-vae-xsmax-fp16.animapk SHA 10171af0...c447 now durable
  on HF (ScalingBiz/AnimaXS-VAE-XSMAX-FP16) and on Clore.
- canonical PNG conversion = to_display_uint8 (clamp((v+1)*0.5,0,1), floor*255+0.5)
- Golden step_latents are DENOISED captures (trace_anima callback arg mismatch,
  D055), not post-step latents.

## Proven/rejected hypotheses
PROMOTE: grid is latent-encoded, reproduced on CUDA, decoder clean.
PROMOTE: model forward faithful (block00 step7 cos 0.999), Euler faithful.
REJECT: VAE/decoder as grid source (G0 clean + random-perturb clean).
REJECT: input noise as grid source (94.7x on pure noise is VAE noise response,
  not the 246x structured grid; golden_final clean at 1.0x).

## Open hypotheses
- Golden final_latent vs produced latent divergence from step 1 (cos 0.17 at
  step1) — the true differentiator. Candidate: trace_anima's cfg=1.0 single-cond
  path vs the harness, or a conditioning/context nuance.

## Next 3 actions
1. Check Clore grid evidence upload status; verify on HF.
2. Decide whether the golden final_latent clean path is reproducible on CUDA;
   if the divergence is a harness/golden-path artifact, establish the true
   clean CUDA control via the actual trace_anima ComfyUI path (real ComfyUI).
3. Update DECISIONS.md + TODO + this file; report milestone to K.

Important source revisions: circlestone-labs/Anima@f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b
  -> anima-turbo-v1.0.safetensors SHA c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
  pinned MiniTrainDIT/comfy cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44
Important pack SHAs: fp16-all (see manifest), w8-v2, w4-v2, vae 10171af0...c447
Important Clore paths: /workspace/out/grid_repro (all evidence)
