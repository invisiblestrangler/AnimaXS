# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after E006–E008. The next task is **E009: production Metal DiT block 0**. Do not start H006’s 28-block loop until E009 matches the H005 same-W4 oracle.

## Proven baseline

Work in `/root/AnimaXS`. Feature head `011cd38` passed hosted CI `31482950188`: deterministic Xcode project, generic iPhone build, and all simulator tests, including real Metal/MPS execution. Local untracked `scripts/oracle_out/block0/` is ~347 MB of regenerable H005 output; preserve it and never commit it.

Completed production foundations:

- D007/D008 metadata-derived, checked, zero-copy DiT/Qwen execution ranges; maximum DiT block range `38,993,920` bytes.
- E001–E005 diagnostics, row-aware dequant, exact fp32-sensitive arithmetic, split-half RoPE, patch/unpatch, Euler, and direct W4 fp32 matvec.
- E006 `LinearExecutor`: `[M,K] × [N,K]ᵀ`, 128-row tiles, reusable padded activation/result buffers and one reusable fp16 dequant matrix, async completion.
- E007: simple accumulation retained. At K=2048/8192, direct fp32 cosine is effectively 1.0; MPS cosine is at least `0.999999975`, maxAbs at most `1.76e-4`.
- E008 `AttentionExecutor`: one 128-query score tile, fp32 softmax max/sum, self K=1024 and cross K=512. The test proves cross row 511 participates.

Hosted CI is a functional Metal gate, not an A12 performance/memory substitute. Repeat E007 and benchmark E008 on A12 before device acceptance.

## E009 objective

Build one bounded-memory production Metal block-0 vertical slice using the real D007 block range and E002–E008. It must not materialize large Swift arrays/Data or retain dequantized matrices. Use one range/ring slot, reusable GPU buffers, fp32 residual/modulation/gates, and fp16 branch/MPS boundaries. Validate finite output and parity against the H005 same-W4 block-0 oracle with a recorded experimental tolerance.

Read `RUNBOOK.md` §§18–24, 32–33, 48–49, 53; E009/H006 in `TODO.md`; D030–D045 in `DECISIONS.md`; `DiTBlockCPU.swift`; `DiTWeights.swift`; and D007 locator APIs. `/root/H005_ADVANCED_AI_GUIDANCE.md` is required only when changing H005 equations. Frozen ComfyUI is `/root/comfy-ref` at `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`; Hermes harness is `/root/anima-harness`.

## Non-negotiable semantics and pitfalls

- Quant groups reset per matrix row: `group = row*ceil(K/64) + column/64`.
- Never guess MPS row stride. Use `MPSMatrixDescriptor.rowBytes(fromColumns:dataType:)`; tight buffers need explicit staging/copy.
- Register Metal completion handlers before `commit()`.
- DiT self-attention: 16/16 heads, head dim 128, shared Q/K RMSNorm weights, split-half RoPE pairs `(p,p+64)`, bidirectional, no GQA/mask/extra position embedding.
- Cross-attention receives all 512 adapter rows, including zero-padded rows; no RoPE or mask.
- Modulation is `Linear2(Linear1(SiLU(embedding))) + adaln_lora`, then shift/scale/gate chunks. Norm is mean-centering LayerNorm; residual gate-add is fp32.
- Exact erf GELU; block residual stays fp32; branch inputs/outputs cross fp16 compute boundaries.
- `DiTBlockCPU` is an oracle only, never a production fallback.

Canonical H005 step-7 proof remains:

```text
Swift W4 vs NumPy W4:    cosine 1.000000000, RMSE 6.18e-6, maxAbs 1.91e-4
Swift W4 vs BF16 golden: cosine 0.998712139
Original BF16 vs golden: cosine 0.999992303
```

Start E009 with a small pack-free orchestration test, then a manual/pack-backed block-0 test when the fixture is available. Every Metal edit must run normal hosted CI. Keep giant packs and the 347 MB oracle directory out of git. After E009 passes, proceed in dependency order: H006/H007, F007/G003 where their E009 dependency is satisfied, then sampler/VAE/integration tasks.
