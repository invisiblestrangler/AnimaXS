# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production streamed Metal inference. The complete DiT model path through final velocity is implemented and hosted-parity green.
- **Current task:** F007 streamed Metal Qwen encoder, followed by G003 streamed adapter.
- **Last green H007 baseline:** feature `acd953d`; normal CI `31488187793` and real-W4 parity `31488934459` pass. A later normal-CI push failure was only a transient runner TLS certificate error while downloading pinned XcodeGen, not a source/test failure.
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC plus D007/D008 zero-copy locators; tokenizer parity; CPU Qwen/adapter and H005; E001–E009 Metal primitives/block; H006 real-pack 28-block loop; H007 streamed final layer and source-order velocity unpatchify.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** streamed Qwen/adapter, sampler/VAE integration, diagnostics/UI/release work. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. F007 — streamed Metal Qwen, embedding-row gather through final RMSNorm.
  2. G003 — streamed Metal adapter using the Qwen output.
  3. I001/I002 and J001 — sampler integration and VAE fold validation.
