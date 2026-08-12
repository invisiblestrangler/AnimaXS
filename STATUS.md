# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Qwen, adapter, DiT, and complete eight-step sampler paths are implemented; support RNG/checkpoint/manifest/store foundations are in place.
- **Current task:** Finish/record the I002 full-pack regression rerun, then implement J002/J003 full-frame Wan VAE and wire J004 RGB.
- **Last real-pack baselines:** F007 `31491046871` (final cosine `0.9999992405`, 5.19 s); G003 `31492451065` (final cosine `0.9999984505`, 1.46 s); H007 `31488934459` (velocity cosine `0.9999999646`).
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC and metadata locators; tokenizer parity; streamed Metal Qwen and adapter; E001–E009 primitives/block; H006 28-block loop; H007 final layer/source-order velocity unpatchify.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** production VAE, full stage integration, diagnostics/UI, real release download and release work. The current golden `step_latents` trace is internally inconsistent with Euler/final_latent and must be regenerated before it can gate intermediate steps (D055).
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. J002/J003 — full-frame VAE with final-temporal-slice folds, channel RMS, one-head tiled attention and nearest-exact upsampling.
  2. J004/K002 — connect decoded RGB and explicit stage-memory release/ownership.
  3. L001 — full canonical inference and decoded-image quality gate.
