# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Metal vertical slice. H001–H005 CPU/reference work is complete; H005 fixed and proved row-aware matrix quantization.
- **Current task:** E006 common MPS linear, then E007–E009. Exact arithmetic, RoPE/patch/Euler, and direct W4 matvec are complete.
- **Last green main commit:** `0c41ecf`, CI `31455561792`: deterministic XcodeGen 2.46.0, generic Xcode 26.3 iPhone build, and 61 simulator tests (4 expected pack skips, 0 failures).
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC plus D007/D008 zero-copy DiT/Qwen locators; tokenizer parity; CPU Qwen/adapter and H005 DiT block 0; E001 diagnostics; E002 row-aware dequant; E003 exact fp32-sensitive arithmetic; E004 fused split-half RoPE, patch/unpatch and Euler; E005 direct W4 matvec. E005 hosted result: maxAbs `2.09e-6`, cosine `0.9999999999999986` vs fp64 at K=68.
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete:** Common MPS linear/precision characterization/attention, streamed Qwen/adapter, and production Metal block-0/full-DiT parity are unfinished. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. E006 — reusable row-aware dequant scratch plus transposed tiled MPS linear.
  2. E007/E008 — precision characterization and bounded query-tiled attention.
  3. E009 — production Metal block 0 using the completed locator and one-slot ring.
