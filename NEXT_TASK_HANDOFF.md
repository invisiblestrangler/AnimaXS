# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after the H005 recovery and production-readiness audit. The next task is **D007: a zero-copy DiT block/tensor range locator**. Do not begin H006’s full 28-block loop until the production Metal block-0 gate E009 passes.

## 1. Reload context

Work from `/root/AnimaXS` and read:

1. `RUNBOOK.md`, especially the current checkpoint and §§18–24, 32–33, 48–49, 53
2. `STATUS.md`
3. unchecked `TODO.md` items D007 and E001–E009
4. `DECISIONS.md`, especially D030–D041
5. `TEST_MATRIX.md`
6. `/root/H005_ADVANCED_AI_GUIDANCE.md` only if changing H005 math/oracle behavior

The clean frozen ComfyUI source is `/root/comfy-ref` at commit `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`; `comfy/ldm/cosmos/predict2.py` is present there. Pinned source and recorded numerical decisions override old prose.

## 2. Preserve the current worktree

H005 and this audit are local/uncommitted. Do not reset, discard, or overwrite them. Important changes/files include:

- `AnimaXS/Runtime/Animapk/QuantDecoders.swift` and `AnimaXSTests/QuantDecoderTests.swift` — row-aware W4/W8 matrix group reset.
- `AnimaXS/Runtime/Text/DiTWeights.swift`, `LLMAdapter.swift`, `QwenEncoderCPU.swift` — matrix loads use row-aware decoding.
- `AnimaXS/Runtime/Text/DiTBlockCPU.swift` and `AnimaXSTests/DiTBlockTests.swift` — H005 CPU oracle and primitive tests.
- `scripts/dit_block0_oracle.py` and `scripts/oracle_out/block0/` — canonical same-W4 oracle/provenance. Raw `.f32` and the regenerable ~8 MB block NPZ are local-only/ignored; they exceed the committed A006 fixture budget.
- `AnimaXSTests/MetalExecutionTests.swift` — permanent pack-free Metal-kernel and MPS execution smoke tests.
- `.github/workflows/ci.yml` / `full-inference.yml` — current action majors and fail-fast/explicit-skip behavior.
- `RUNBOOK.md`, `TODO.md`, `STATUS.md`, `DECISIONS.md`, `TEST_MATRIX.md` — audited production order.

The untracked `scripts/diag_deltas.py`, `diag_mlp.py`, and `diag_stages.py` came from H005 diagnostics; preserve them unless deliberately superseded. The external Hermes-compatible Swift harness is `/root/anima-harness`; Swift is `/opt/swift/usr/bin/swift` and is not on `PATH`.

## 3. H005 is resolved

Quantization groups reset independently for every `[out,in]` matrix row:

```text
groupsPerRow = ceil(K / 64)
group = row * groupsPerRow + column / 64
```

Never restore `flatIndex / 64` for a matrix. DiT `x_embedder.proj.1.weight [2048,68]` exposed this bug; W4 `[2,68]` and W8 `[2,65]` are required regressions.

The supplied `block_00_out` is the final hook invocation (sampler step 7, sigma `0.3050089478492737`), not step 0. Final H005 metrics:

```text
Swift W4 vs NumPy W4:          cosine 1.000000000, RMSE 6.18e-6, maxAbs 1.91e-4
Swift W4 vs original golden:   cosine 0.998712139
Original BF16 vs golden:       cosine 0.999992303
```

D035 records why the lower W4-vs-BF16 result is proven quantization error. Do not reopen H005 without contradictory evidence.

## 4. Production boundary and execution order

`DiTBlockCPU` is a correctness oracle. It materializes dequantized Swift arrays and must **not** become the production 28-block implementation.

Proceed in this order:

```text
D007  validated block ranges + tensor-relative mmap/ring spans
E001  finish MetalContext diagnostics and permanent CI harness
E002  row-aware W4/W8 GPU dequant tests (including non-aligned K)
E003  exact erf GELU, fp32 norms/modulation/gate-add
E004  Q/K RMSNorm + split-half RoPE, patch/unpatch, Euler
E005  direct packed W4 fp32 matvec
E006  common transposed MPS linear with one reusable fp16 scratch
E007  precision characterization
E008  bounded query-tiled self/cross attention
E009  one production Metal block 0 vs the H005 same-W4 oracle
D008/F007/G003  streamed Metal Qwen + adapter vs CPU oracles
H006  streamed 28-block Metal loop
```

The existing `AnimaKernels.metal` is incomplete scaffolding. Its tanh GELU and half gate-add are not valid H005 parity implementations. Preserve exact H005 semantics: split-half `(p,p+64)` RoPE, shared `[128]` Q/K RMSNorm, full 512-row padded cross context, no DiT mask/GQA/extra position embedding, mean-centered LayerNorm, SiLU before modulation linears, exact erf GELU, and Float32 residual/modulation/gates.

## 5. D007 acceptance

Identify each block only by exact prefix `model.diffusion_model.blocks.N.`. Physical order is lexicographic (`0,1,10…19,2,20…27,3…9`), so execute logical `0..<28` and never derive `first + index*size` or use old `block_index` metadata.

D007 should return:

- exactly 28 validated physical block ranges;
- each block’s tensors with data/scale/zero spans relative to the block start;
- checked arithmetic for every offset and length;
- enough metadata to copy a block once from read-only mmap into one shared ~39 MB ring and address every packed tensor without another `Data` copy.

Add synthetic order/range tests and run a real-pack metadata audit when available. The CPU reference decoders may continue using safe `Data`; production storage may not create huge `Data` or `[[Float]]` values. Do not add ring=2/3, `MTLHeap`, bytes-no-copy, or page-cache tricks without A12 measurements.

## 6. Verified CI capability

Standard hosted `macos-15` arm64 Simulator Metal is usable. Final non-invasive snapshot run `31452206651` returned `Apple iOS simulator GPU`, passed the permanent project W4 kernel and 2×2 fp16 `MPSMatrixMultiplication` tests, passed deterministic XcodeGen and generic iPhone build gates, and finished 56 tests with 0 failures (3 expected pack skips). Main remains at green commit `6e72f4f`; local H005/audit changes are uncommitted.

Therefore:

- put every pack-free Metal/MPS parity test in normal simulator CI;
- `SKIPPED_NO_METAL` is allowed only when no device/library exists;
- a wrong GPU result is a failure;
- manual full inference must fail on real errors and must never use `continue-on-error`;
- A12 speed, memory/jetsam, thermal, watchdog, and Apple5 behavior remain device-only.

The local H005/audit changes ran together successfully in the non-invasive snapshot `510b3e1` (`31452206651`). They are still uncommitted on main. Commit the intended source/docs/workflow set without raw artifacts or packs, push, and require the resulting **main** project-consistency, generic iPhone build, and simulator jobs to be green before calling main current.

## 7. Useful validation

Linux/Hermes checks:

```bash
/opt/swift/usr/bin/swiftc -parse \
  AnimaXS/Runtime/Animapk/QuantDecoders.swift \
  AnimaXS/Runtime/Text/DiTBlockCPU.swift \
  AnimaXSTests/QuantDecoderTests.swift \
  AnimaXSTests/DiTBlockTests.swift

python3 -m py_compile scripts/dit_block0_oracle.py

cd /root/anima-harness
/opt/swift/usr/bin/swift build -c release
```

On macOS/CI, require `xcodegen generate` to leave the project clean, build with Xcode 26.3, and run the simulator test target. Update `TODO.md`, `STATUS.md`, `DECISIONS.md`, `TEST_MATRIX.md`, and rewrite this handoff when handing off E001/E002 or a later task.

Keep `project.yml` option `generateEmptyDirectories: false`: git does not preserve local empty runtime folders, and enabling it makes local/clean-CI project generation disagree (D040).

CI deliberately pins XcodeGen 2.46.0 and verifies the official release zip checksum (D041). Do not replace this with unpinned `brew install xcodegen`.
