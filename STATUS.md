# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Full production inference pipeline is implemented and CI-validated. Qwen, adapter, DiT, sampler, VAE, GenerationCoordinator, and FullInferenceTests are all in place. A12 physical-device acceptance remains deferred.
- **Current task:** Reconcile progress docs; await A005 license review to unblock model-assets-v1 for full-pack CI inference.
- **Last real-pack baselines:** F007 `31491046871` (final cosine `0.9999992405`, 5.19 s); G003 `31492451065` (final cosine `0.9999984505`, 1.46 s); H007 `31488934459` (velocity cosine `0.9999999646`); J002 VAE `31593343788` (cosine `0.99981`, RMSE `0.01333`, maxAbs `0.04272`, 37 s).
- **CI baseline:** `31601722959` — all three jobs green (project-consistency, iphone-build, simulator-tests) after J004 refactor.
- **What works:** ANMA parser/CRC and metadata locators; tokenizer parity; streamed Metal Qwen, adapter, DiT block/final, sampler; full-frame VAE decoder with shared decode path; RGBA8 conversion; GenerationCoordinator (K002); FullInferenceTests (L001) compiles and skips cleanly without packs.
- **J004 refactor (commit `1a32112`):** VAE decoder graph exists in ONE implementation (`decodeToPositionMajorRGB`); `execute(latent:rgb:)` and `rgba8(latent:)` both call it. Platform-neutral `DecodedRGBA8` struct separates runtime from UIKit. RGBA8 test fixed (HWC↔CHW layout).
- **K002 GenerationCoordinator (commit `6e7f042`):** Orchestrates prompt→tokenizers→Qwen→adapter→diffusion→VAE→UIImage with stage-scoped object lifetime and `GenerationState` progress. ContentView wired to coordinator.
- **L001 FullInferenceTests (commit `594a04b`):** Compiles in normal CI. Uses production APIs: `QwenEncoderMetal.execute(tokenIDs:output:)`, `LLMAdapterMetal.execute(qwenContext:contextTokens:t5IDs:t5Weights:output:)`, `DiffusionSampler.execute(initialLatent:crossContext:outputLatent:)`, `VAEDecoder.decode(latent:)`. Production TokenizerLoader semantics. Fixture-gated (skips when packs unavailable).
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** A005 license review blocks model-assets-v1 release (and thus full-pack CI inference). L001 runs end-to-end only when legitimate model packs are available. Physical A12 acceptance deferred.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. A005 — resolve license review to unblock model-assets-v1 and full-pack L001 CI run.
  2. K003/K004 — cancellation and memory-warning handling (safe-boundary cancel, checkpoint, recover).
  3. Physical A12 acceptance — build in Xcode, install on iPhone XS Max, record timings/memory/thermal.
