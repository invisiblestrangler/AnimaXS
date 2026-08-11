# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Qwen, adapter, DiT-through-velocity, and I001 Euler paths are implemented and hosted/normal-CI green.
- **Current task:** Finish I002 full-pack eight-step sampler parity, then J002–J004 VAE decoder/RGB.
- **Last real-pack baselines:** F007 `31491046871` (final cosine `0.9999992405`, 5.19 s); G003 `31492451065` (final cosine `0.9999984505`, 1.46 s); H007 `31488934459` (velocity cosine `0.9999999646`).
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC and metadata locators; tokenizer parity; streamed Metal Qwen and adapter; E001–E009 primitives/block; H006 28-block loop; H007 final layer/source-order velocity unpatchify.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** full-pack eight-step sampler validation, VAE, full integration, diagnostics/UI/model store/release work. The current golden `step_latents` trace is internally inconsistent with Euler/final_latent and must be regenerated before it can gate intermediate steps (D055).
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. I002 — verify explicit FLOW velocity→denoised conversion and all eight production model/Euler steps against a corrected trace/final latent.
  2. J002/J003 — full-frame VAE with final-temporal-slice folds and exact norms/attention.
  3. J004 — RGB conversion and release of the float buffer.
