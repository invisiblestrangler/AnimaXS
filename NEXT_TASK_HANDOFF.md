# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-11 after H007. The next dependency task is **F007: production streamed Metal Qwen encoder**.

## Proven baseline

Work in `/root/AnimaXS`. Preserve and never commit untracked `scripts/oracle_out/block0/` (~347 MB). The complete DiT path is now production Metal: E009 block parity (`31485374918`), H006 all 28 real blocks finite in 19.456 s (`31486134420`), and H007 final velocity parity (`31488934459`: cosine `0.9999999645956833`, RMSE `0.0003068520`, maxAbs `0.001953125`). Normal CI `31488187793` passed project consistency, generic iOS build, and all pack-free Metal/MPS tests. A later push encountered a transient runner TLS certificate error before build; it was not a code failure.

H007 lives in `DiTFinalLayerExecutor.swift`, `DitForward.executeVelocity`, `DiTFinalLayerLocator`, `unpatchify_velocity16`, and `scripts/dit_final_oracle.py`. Its temporary release/workflow was removed; the gated real-fixture test remains. Preserve D048: FinalLayer has explicit residual and LayerNorm-output fp16 boundaries, and output unpatchify order is `(p1,p2,t,C)`, unlike input patchify.

## F007 objective and fixed architecture

Implement a non-reentrant `QwenEncoderMetal`/executor that accepts tokenizer IDs (1…512), gathers only those W8 embedding rows, streams each logical layer 0…27 through one reusable ring, applies final `model.norm.weight`, and returns finite fp32 `[sequence,1024]` context. Production must not call `QwenEncoderCPU`, materialize full weights, or copy the 165 MB embedding tensor.

Use `QwenLayerLocator`; physical layers are string-sorted, so never derive offsets arithmetically. The real pack is `/root/anima-xsmax/results/packs/qwen3-0.6b-xsmax-w8.animapk`. Each layer has exactly 11 tensors and a ~16 MB contiguous range:

```text
input_layernorm.weight             [1024]       fp16
post_attention_layernorm.weight    [1024]       fp16
self_attn.q_proj.weight            [2048,1024]  W8
self_attn.k_proj.weight            [1024,1024]  W8
self_attn.v_proj.weight            [1024,1024]  W8
self_attn.o_proj.weight            [1024,2048]  W8
self_attn.q_norm.weight            [128]        fp16
self_attn.k_norm.weight            [128]        fp16
mlp.gate_proj.weight               [3072,1024]  W8
mlp.up_proj.weight                 [3072,1024]  W8
mlp.down_proj.weight               [1024,3072]  W8
```

The embedding is `model.embed_tokens.weight [151936,1024] W8`; each selected row has 1024 packed bytes plus 16 fp16 scales and 16 fp16 zeros. `QwenLayerLocator.embeddingRow` already returns checked row-relative spans. Gather selected rows directly from mmap into small packed/scale/zero staging buffers, dequantize to activation storage, and retain no full embedding copy. Final norm is `model.norm.weight [1024] fp16`, outside layer ranges.

The authoritative readable implementation is `QwenEncoderCPU.swift` plus `QwenNumerics.swift`; pinned-source resolutions are D018–D020 and `docs/QWEN_ENCODER_DEBUG.md`. Per layer:

```text
fp32 residual
RMSNorm(1024, eps 1e-6)
Q 1024→2048, K/V 1024→1024
per-head Q/K RMSNorm(128, shared weight)
split-half GPT-NeoX RoPE, theta 1_000_000
causal attention, 16 Q heads / 8 KV heads
grouped GQA mapping kvHead = qHead / 2 (never modulo)
O 2048→1024 + fp32 residual
RMSNorm(1024)
SiLU(gate 1024→3072) * up 1024→3072
down 3072→1024 + fp32 residual
```

Reuse `LinearExecutor` and `AttentionExecutor`. The existing `rms_rope_split_half` kernel is shape-dynamic and matches the Qwen 128-d split-half operation when supplied a theta-1e6 sequence rope; explicit token↔head-major transposes are still required around attention. Either expand 8-head K/V to 16 heads with grouped repeat or extend attention with an explicit grouped KV-head mapping; lock it with a nonzero pack-free test. Keep causal masking (`queryCount == keyCount`) and dynamic sequence length. Add only the small fused kernels actually needed, likely W8 embedding-row gather/dequant, gated SiLU multiply, grouped GQA copy/mapping, and fp32 residual conversion/add.

## Validation and continuation

First add pack-free tests for exact embedding row/group indexing, grouped GQA head mapping, causal row behavior, dynamic short/tail sequence, buffer guards, and a synthetic one-layer orchestration path. Then run a compact or full real-pack hosted test against the existing F005 same-W8 oracle, preferably layer 0 plus final checkpoints from `scripts/oracle_out/qwen_oracle_layers.npz`. Record maxAbs/RMSE/cosine and bounded ring/scratch sizes. Normal push CI must remain pack-free; temporary fixture releases/workflows must be removed after proof.

After F007, implement G003 streamed adapter, then I001/I002 sampler integration and J001 VAE fold validation. Do not start UI polish or pack release first. Preserve exact erf GELU, group-64 reset per matrix row, recommended MPS row strides, command completion before ring overwrite, and fp32 residual arithmetic. A12 performance/memory/thermal/watchdog acceptance remains pending and must not be inferred from hosted simulator success.
