# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Metal vertical slice. H001–H005 CPU/reference work is complete; H005 fixed and proved row-aware matrix quantization.
- **Current task:** E009 production Metal DiT block-0 vertical slice. E001–E008 are complete.
- **Last green feature commit:** `011cd38`, CI `31482950188`: project consistency, generic Xcode 26.3 iPhone build, and simulator tests including E006–E008 all pass.
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC plus D007/D008 zero-copy locators; tokenizer parity; CPU Qwen/adapter and H005 DiT block 0; E001–E008 Metal primitives, including row-aware quantized MPS linear and bounded self/cross attention.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** E009 production Metal block 0, streamed Qwen/adapter, full Metal DiT, sampler/VAE integration, diagnostics/UI/release work. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. E009 — production Metal block 0 using D007 ranges and the E001–E008 primitives.
  2. F007/G003 — streamed Metal Qwen and adapter.
  3. H006/H007 — streamed 28-block DiT and final projection.
