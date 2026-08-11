# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Qwen, adapter, and complete DiT-through-velocity paths are implemented and hosted-parity green.
- **Current task:** I001/I002 sampler integration, then J001–J004 VAE validation/decoder.
- **Last real-pack baselines:** F007 `31491046871` (final cosine `0.9999992405`, 5.19 s); G003 `31492451065` (final cosine `0.9999984505`, 1.46 s); H007 `31488934459` (velocity cosine `0.9999999646`).
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC and metadata locators; tokenizer parity; streamed Metal Qwen and adapter; E001–E009 primitives/block; H006 28-block loop; H007 final layer/source-order velocity unpatchify.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** sampler/model-sampling conversion, VAE, full integration, diagnostics/UI/model store/release work. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. I001/I002 — exact schedule, FLOW velocity→denoised conversion, and eight Euler steps.
  2. J001 — prove or reject every T=1 causal-conv 3-D→2-D fold.
  3. J002–J004 — VAE decode and RGB conversion.
