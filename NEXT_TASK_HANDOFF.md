# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after E009/H006. The next task is **H007: final DiT layer and unpatchify**.

## Proven baseline

Work in `/root/AnimaXS`. Preserve and never commit untracked `scripts/oracle_out/block0/` (~347 MB). E009 real-W4 hosted run `31485374918` matched H005 at cosine `0.999999979842237`, RMSE `0.0003750332`, maxAbs `0.02190399`. H006 run `31486134420` streamed all 28 real block ranges through one slot; all checkpoints were finite, final residual min/max `-5063.061`/`39658.49`, GPU test time `19.456s`. Normal hosted CI also compiles/runs the pack-free Metal suite; A12 performance/memory acceptance remains outstanding.

Production files are `WeightStreamer.swift`, `DiTBlockExecutor.swift`, `DitForward.swift`, `LinearExecutor.swift`, `AttentionExecutor.swift`, and `AnimaKernels.metal`. `DiTBlockCPU` remains an oracle only.

## H007 objective

After `DitForward.execute`, implement the pinned `predict2.py` `FinalLayer` exactly and produce fp32 `[1,16,1,64,64]` velocity via `unpatchify16`. The verified real-pack tensors are:

```text
model.diffusion_model.final_layer.adaln_modulation.1.weight [256,2048] W4
model.diffusion_model.final_layer.adaln_modulation.2.weight [4096,256] W4
model.diffusion_model.final_layer.linear.weight             [64,2048] W4
```

Pinned source is `/root/comfy-ref` commit `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`, `comfy/ldm/cosmos/predict2.py` `FinalLayer` lines 338–395 and `unpatchify` lines 829–838. Semantics: `SiLU(emb) → 2048→256 → 256→4096`, add only `adaln_lora[0..<4096]`, chunk shift then scale, mean-centered LayerNorm residual, apply `norm*(1+scale)+shift`, project 2048→64, then rearrange 1024 spatial tokens to 16 channels ×64×64. No gate exists in the final layer.

Use metadata-derived spans and one small streamed final-layer range; do not guess offsets or materialize weights as Swift matrices/Data. Reuse direct fp32 W4 matvec for M=1 modulation and the common MPS linear for 1024 rows. The existing `unpatchify16` kernel consumes fp32, so make the fp16 projection boundary explicit before conversion/unpatchify. Add pack-free kernel/orchestration tests and a pack-backed finite/shape test. Compare block 15/27 or final output fixtures when available; record measured tolerances rather than inventing them.

After H007, proceed by dependency: F007 streamed Qwen, G003 streamed adapter, then I001/J001/L001 integration. Keep exact erf GELU, per-row quant groups, recommended MPS row strides, completion-before-commit, full 512-row cross attention, split-half RoPE, and fp32 residual/gates unchanged.
