# AnimaXS 8px Grid — Root-Cause Closure TODO

Reset 2026-08-14 per K's instructions (Hermes_AnimaXS_Test_Grid_Hypothesis_Instructions.md).
Historical items live in git history + DECISIONS.md. This list is ONLY this run's closure plan.

## Wan21 boundary proof (Phase A) — DONE
- [x] Pin exact comfy-ref Wan21 semantics (constants, call site samplers.py:1238, Anima registration)
- [x] Zero-inference proof: process_out(step_latents[7]) == final_latent BIT-EXACT (cos 1.0, rmse 0.0)
- [x] Decode raw/converted/golden through CUDA VAE; carrier 254.4x -> 1.0x
- [x] Saved G1/G2/G3/G4 latents: carrier ~245x -> 1.3x all lanes (W4 493x -> 1.8x)
- [x] Vision review: grid gone in all fixed lanes

## CUDA fix + rerun (Phases B/C) — DONE
- [x] wan21_latent_format.py (pinned constants, process_in/out) + unit tests (18.1-18.3) PASS
- [x] grid_repro.py boundary: sampler_latent -> vae_latent exactly once; explicit naming
- [x] Fixed-pipeline rerun: G1 0.811->0.971, carrier 246.7x->1.3x; G2/G3 same; W4 1.8x
- [x] Comfy-ref file SHAs + provenance JSON recorded

## Metal/iOS port (Phase D) — IN PROGRESS
- [x] Wan21LatentFormat.swift (pinned constants, processOut/processIn, in-place buffer op)
- [x] GenerationEngine applies process_out exactly once before VAEDecoder.decode
- [x] FullInferenceTests converts before latent regression + decode
- [x] Wan21LatentFormatTests (known-vector golden crop, zero==mean, channel mapping, inverse, buffer parity)
- [ ] xcodeproj regenerated via bootstrap-project workflow (bot commit) — dispatched 31802354612
- [ ] PR -> CI green (project consistency + iPhone build + simulator tests incl. new tests)
- [ ] macOS full-inference validation (FP16-all/W8) via pack-backed workflow if practical

## Evidence + docs + shutdown (Phase E)
- [ ] DECISIONS.md D097 appended (DONE)
- [ ] HERMES_SESSION.md updated
- [ ] This TODO updated
- [ ] Final summary report (section 20 format)
- [ ] HF upload experiments/2026-08-14_wan21-process-out-fix/ + SHA256SUMS + hash verify
- [ ] Commit + push docs
- [ ] Terminate Clore instance (all CUDA work durable)
