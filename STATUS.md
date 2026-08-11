# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Production Metal vertical slice. H001–H005 CPU/reference work is complete; H005 fixed and proved row-aware matrix quantization.
- **Current task:** D007, then E001–E009. Build a zero-copy block/tensor locator and tested bounded-memory Metal/MPS block 0 before attempting H006’s 28-block loop.
- **Last green main commit:** `6e72f4f` — main CI `31436850938` green. H005/audit remain local/uncommitted, but an exact non-invasive snapshot `510b3e1` passed CI run `31452206651`: deterministic XcodeGen 2.46.0, generic Xcode 26.3 iPhone build, and 56 simulator tests (3 expected pack skips, 0 failures).
- **Hosted Metal fact:** Final snapshot run `31452206651` executed the permanent W4 Metal kernel test and fp16 `MPSMatrixMultiplication` on `Apple iOS simulator GPU`; both passed. Pack-free Metal/MPS parity belongs in normal CI.
- **What works:** ANMA parser/CRC/ranges; tokenizer parity; CPU Qwen + final RMSNorm; CPU adapter; DiT input/timestep/AdaLN/RoPE; H005 CPU block 0. Swift-W4≈NumPy-W4 cosine `1.000000000`; Swift-W4≈source-BF16 golden `0.998712139`, with original BF16≈golden `0.999992303` (D035).
- **Production boundary:** `DiTBlockCPU` and its dequantized Swift arrays are oracles only. Production must use mmap spans, a one-slot ~39 MB ring, one fp16 dequant scratch, fp32 residual/modulation/gates, Metal kernels, MPS linears, and query-tiled attention.
- **Known incomplete scaffold:** Metal tanh-GELU and half gate-add are not reference-correct; D007, row-reset GPU regressions, norms/RoPE/matvec/linear/attention, streamed Qwen/adapter, and Metal block-0 parity are unfinished. Full inference test/release assets do not exist yet.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior, watchdog limits, page cache, and thermal stability. Hosted Metal functional success does not answer them.
- **Next three tasks:**
  1. D007 — validated zero-copy block ranges and tensor-relative mmap/ring spans.
  2. E001/E002/E003 — finish diagnostics/harness and row-aware dequant + exact fp32-sensitive kernels in hosted CI.
  3. E004–E009 — RoPE/patch/sampler, direct matvec, MPS linear/precision/attention, then production Metal block 0 parity.
