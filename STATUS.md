# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Metal vertical slice. H001–H005 CPU/reference work is complete; H005 fixed and proved row-aware matrix quantization.
- **Current task:** H007 final DiT norm/modulation/projection and unpatchify. E009 and H006 are complete.
- **Last green feature commit:** `011cd38`, CI `31482950188`: project consistency, generic Xcode 26.3 iPhone build, and simulator tests including E006–E008 all pass.
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC plus D007/D008 zero-copy locators; tokenizer parity; CPU Qwen/adapter and H005; E001–E009 Metal primitives/block; H006 real-pack 28-block streamed Metal loop.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** H007 final DiT layer, streamed Qwen/adapter, sampler/VAE integration, diagnostics/UI/release work. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. H007 — final DiT layer and unpatchify16.
  2. F007/G003 — streamed Metal Qwen and adapter.
  3. Sampler/VAE/integration after their production dependencies.
