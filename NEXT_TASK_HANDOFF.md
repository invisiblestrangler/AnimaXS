# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after completing H005, D007/D008, and E001–E005. The next task is **E006: the common transposed MPS linear executor**. Do not start H006’s 28-block loop before E009 passes.

## Current branch and proof

Work in `/root/AnimaXS`. E003–E005 are on `codex/e003-exact-arithmetic` at `433d064`; GitHub Actions run `31478699877` passed:

- deterministic XcodeGen 2.46.0 project check;
- generic iPhone build with Xcode 26.3;
- 64 simulator tests, 4 expected no-pack skips, 0 failures;
- hosted execution of exact arithmetic, split-half RoPE, patch/unpatch, Euler and direct W4 matvec on `Apple iOS simulator GPU`.

Local `scripts/oracle_out/block0/` contains ~347 MB of ignored/regenerable H005 output and must not be committed.

## Reload context

Read `RUNBOOK.md` (especially §§18–24, 32–33, 48–49, 53), `STATUS.md`, E003–E009 in `TODO.md`, `DECISIONS.md`, and `TEST_MATRIX.md`. Read `/root/H005_ADVANCED_AI_GUIDANCE.md` only when touching H005 equations. Frozen ComfyUI source is `/root/comfy-ref` at `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`; the Hermes-compatible Swift harness is `/root/anima-harness` and Swift is `/opt/swift/usr/bin/swift`.

## Completed foundation

H005 is resolved. Matrix quantization groups reset per row:

```text
groupsPerRow = ceil(K / 64)
group = row * groupsPerRow + column / 64
```

Final H005 metrics remain:

```text
Swift W4 vs NumPy W4:        cosine 1.000000000, RMSE 6.18e-6, maxAbs 1.91e-4
Swift W4 vs BF16 golden:     cosine 0.998712139
Original BF16 vs golden:     cosine 0.999992303
```

`AnimapkRangeLocator.swift` now provides checked relative spans, `DiTBlockLocator`, and `QwenLayerLocator`. Real-pack audit results:

```text
DiT:  28 logical blocks, 560 tensors, maximum range 38,993,920 bytes
Qwen: 28 logical layers, 308 layer tensors, 151,936 checked embedding rows
```

Ranges use exact prefixes and logical numerical order even though physical order is lexicographic. `AnimapkFile.bytes(in:)` is zero-copy and valid while the file owns its mmap. Production should copy each execution range once into one ring; do not create per-tensor `Data` copies.

`MetalContext` now probes Apple5, buffer/threadgroup/working-set/allocation/physical-memory/thermal values. E002–E005 now provide bounded row-aware dequant, fp32-stat norms, erf-form exact GELU, fp32 modulation/gates/residuals, fused split-half RoPE, patch/unpatch, Euler, and direct packed W4 matvec. The E005 K=68 result versus fp64 is maxAbs `2.09e-6`, cosine `0.9999999999999986`.

## Next implementation order

```text
E006  transposed MPS linear with one reusable fp16 dequant scratch
E007  K=2048/8192 precision characterization and recorded decision
E008  bounded query-tiled self/cross attention with fp32 softmax
E009  one production Metal block 0 against the H005 same-W4 oracle
F007/G003  streamed Metal Qwen and adapter using D008
H006  streamed 28-block Metal loop
```

For E006, implement `LinearExecutor.swift` around the existing row-aware dequant kernels and `MPSMatrixMultiplication`. Reuse one fp16 weight scratch, interpret weights as `[N,K]`, and multiply `[M,K] × [N,K]ᵀ`. Start with an M tile of 128, support a smaller final tile, preserve asynchronous command-buffer completion, and do not retain dequantized matrices. Add pack-free transpose/stride tests and a representative tiled shape; keep giant pack-backed shapes out of normal CI unless measured safe.

H005 semantics that must survive E003–E009: split-half `(p,p+64)` DiT RoPE, shared `[128]` Q/K RMSNorm weights, all 512 padded cross rows, no DiT mask/GQA/extra position embedding, SiLU before modulation linears, and exact erf GELU. `DiTBlockCPU` is an oracle only and must not become a production fallback.

## Validation commands

Linux checks:

```bash
/opt/swift/usr/bin/swiftc -parse AnimaXS/Runtime/Animapk/*.swift
/opt/swift/usr/bin/swiftc -parse AnimaXS/Runtime/Animapk/QuantDecoders.swift \
  AnimaXS/Runtime/Text/DiTBlockCPU.swift AnimaXSTests/DiTBlockTests.swift
python3 -m py_compile scripts/dit_block0_oracle.py
git diff --check
```

Every Metal change must also run hosted CI: pinned XcodeGen leaves the project clean, generic iPhone build passes, and simulator tests execute the kernels. Keep `generateEmptyDirectories: false`. Do not add extra rings, heaps, bytes-no-copy, or cache tricks without A12 measurements. A12 speed, jetsam, thermal behavior, and Apple5 performance remain device-only acceptance work.
