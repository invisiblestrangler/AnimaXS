# HERMES_SESSION.md

# Current state (2026-08-14, root cause CLOSED)

Branch: investigate/animapk-cuda-parity (HEAD 5c485eb + docs pending; pushed)
origin/main: 686e5a8 (infra-only; do not dirty)
Working tree: clean (docs commit pending)
Bootstrap-project workflow dispatched: run 31802354612 (regenerates xcodeproj)
PR to open after bot commit lands -> CI (project-consistency + build + tests)

## THE HEADLINE RESULT (2026-08-14)
**ROOT CAUSE CLOSED (D097):** the 8px grid = missing Wan21 latent-format
boundary transform. ComfyUI applies latent_format.process_out to the sampler
return (samplers.py CFGGuider.inner_sample) before the workflow/VAE receives
the latent; the custom CUDA/Metal runtime decoded raw sampler-space latents as
VAE-space.

Proof (zero-inference, scripts/animapk_cuda/wan21_boundary_proof.py):
  process_out(golden step_latents[7]) vs golden final_latent:
    cos 1.000000, RMSE 0.000000, maxAbs 0.000000  (BIT-EXACT)
Fixed pipeline (grid_repro.py rerun, /workspace/out/grid_repro_fixed):
  carrier ratio vs clean ref (0.0000548):
    G0 1.0x | G1 BF16 1.3x | G2 FP16 1.3x | G3 W8 1.3x |
    G4 metal fp16-all 1.3x | w8 1.3x | w4 1.8x   (was 244-493x)
  latent cos vs golden after conversion: G1 0.971, G2 0.973, G3 0.969,
    metal 0.973/0.970/0.851

## Code fix
- CUDA: scripts/animapk_cuda/wan21_latent_format.py (pinned constants,
  process_in/out) + grid_repro.py converts sampler_latent -> vae_latent once.
- Metal: AnimaXS/Runtime/VAE/Wan21LatentFormat.swift; GenerationEngine applies
  process_out in place before VAEDecoder.decode; FullInferenceTests converts
  before latent regression + decode. Checkpoints stay sampler-space.
- Tests: scripts/test_wan21_latent_format.py (local, PASS);
  Wan21LatentFormatTests.swift (known-vector golden crop, zero==mean, channel
  mapping, inverse, buffer parity) — pending CI.

## Pinned constants (do NOT use ComfyUI master)
Wan21 (comfy-ref cbbc9dab…, file snapshot at /workspace/comfy-ref):
  scale_factor 1.0; latents_mean [-0.7571,-0.7089,-0.9113,0.1075,-0.1745,
  0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,
  0.2503,-0.2921]; latents_std [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,
  2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]
  process_out: x*std/scale + mean; process_in: (x-mean)*scale/std
  comfy-ref file SHAs: supported_models a97d56c8…, latent_formats f13259cb…,
  samplers 69498999…, model_base 39022ddc…

## Clore state
ACTIVE order 2024354 (RTX 3060 12GB) — terminate AFTER all CUDA work durable.
ssh -i /root/key root@n1.msk.cloreai.ru -p 1500; repo /workspace/repo @ 5c485eb
Evidence: /workspace/out/wan21_boundary_proof (zero-inference),
  /workspace/out/grid_repro_fixed (fixed pipeline, full manifests)
Fixture dir: /workspace/out/fixture (x_in.f32, context512.f32)
Metal latents: /workspace/metal_latents/{fp16-all,w8,w4}/step07_denoised.f32
HF: ScalingBiz/AnimaXS-investigation-artifacts — upload
  experiments/2026-08-14_wan21-process-out-fix/ pending (from Clore), then
  SHA256SUMS + hash verify.

## Next 3 actions
1. Check bootstrap-project run -> pull bot commit -> open PR -> CI
2. HF upload of wan21 evidence from Clore + verify
3. Docs commit (DECISIONS D097 done; TODO/HERMES_SESSION) -> final report -> Clore shutdown

## Older conclusions (superseded where they conflict)
- D096's "golden trace artifact/inconsistency" interpretation is superseded by
  D097: step_latents[7] raw vs final_latent differed only by Wan21.process_out.
- D060's "latent fed unchanged" is true INSIDE the VAE only; sampler output
  must be converted to VAE-space before decode.

## Update 2026-08-14 (afternoon): closure status
- PR #7 CI ALL GREEN: project-consistency PASS, iphone-build PASS,
  simulator-tests PASS (7m6s, includes all 5 Wan21LatentFormatTests passing).
- Full-inference macOS E2E dispatched: run 31804551959 (W4 pack, REQUIRED
  mode) — validates the fixed Metal pipeline end-to-end.
- HF evidence uploaded + verified: experiments/2026-08-14_wan21-process-out-fix
  (27 files, SHA256SUMS, hash MATCH 27/27) — token at /root/HUGGINGFACE_TOKEN.
- Final report: GRID_ROOT_CAUSE_FINAL_REPORT.md (repo root).
- Branch HEAD: 9eea6c0 (all pushed).
- Remaining: full-inference run result; then terminate Clore per K's new
  instructions (supersedes the older "keep Clore alive" note).

## Update 2026-08-14 (late): COMPLETE
- Full-inference macOS E2E PASS (run 31805850085, 13m53s): W4 pack,
  latent_cosine 0.8660 (was 0.6946 pre-fix), rgb_cosine 0.8198 (was 0.7035),
  full_inference: PASS, FULL_INFERENCE=PASS marker. Vision: no grid, clean.
  (Fixed workflow fixture-name bug: DiT pack bundled as anima-turbo-refine.animapk.)
- comfy-ref pinned snapshot (4 files) uploaded to HF + SHA verified (SHAs match
  recorded provenance exactly).
- Clore termination: pending (cancel_order 2024354) — all evidence durable:
  HF experiments/2026-08-14_wan21-process-out-fix (27 files + comfy-ref-pinned,
  hash-verified), repo pushed (branch investigate/animapk-cuda-parity, PR #7).
