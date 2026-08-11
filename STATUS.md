# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Metal vertical slice. H001–H005 CPU/reference work is complete; H005 fixed and proved row-aware matrix quantization.
- **Current task:** E003, then E004–E009. Exact fp32-sensitive kernels are the next gate before the bounded-memory production Metal block 0.
- **Last green main commit:** `0c41ecf`, CI `31455561792`: deterministic XcodeGen 2.46.0, generic Xcode 26.3 iPhone build, and 61 simulator tests (4 expected pack skips, 0 failures).
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC plus D007/D008 zero-copy DiT/Qwen locators; tokenizer parity; CPU Qwen + final RMSNorm; CPU adapter; DiT input/timestep/AdaLN/RoPE; H005 CPU block 0; E001 diagnostics and E002 bounded row-aware W4/W8 Metal dequant. Swift-W4≈NumPy-W4 cosine `1.000000000`; Swift-W4≈source-BF16 golden `0.998712139`, with original BF16≈golden `0.999992303` (D035).
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete scaffold:** Metal tanh-GELU and half gate-add are not reference-correct; E003 norms/arithmetic, RoPE/matvec/linear/attention, streamed Qwen/adapter, and Metal block-0 parity are unfinished. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. E003 — exact erf GELU, fp32-stat norms, SiLU, modulation and gate/residual kernels with hosted parity tests.
  2. E004–E008 — RoPE/patch/sampler, direct matvec, MPS linear/precision/attention.
  3. E009 — production Metal block 0 using the completed locator and one-slot ring.
