# Qwen Encoder — Debugging State

## Status (2026-08-10)
The CPU-reference Qwen3-0.6B encoder (`QwenEncoderCPU.swift` + `QwenNumerics.swift`) has
**layer-0 numerics verified exact** (cosine 1.0, maxAbs 4.7e-6) against an independent
Python reference built from the same dequantized W8 weights. The full 28-layer forward
does NOT yet match the golden `cond_context` (cosine ≈ -0.04), and the SAME wrong output
is produced by both the Swift and a hand-written Python reference — so the bug is a
**shared misunderstanding of one reference detail**, not a Swift-specific typo.

## What is verified correct (layer 0)
- Embedding gather from W8 table (row slice + fp16 scale/zero, 32-byte stride per row)
- RMSNorm (fp32 accumulation)
- GQA attention: q_proj [2048,1024] (16 heads×128), k/v_proj [1024,1024] (8 KV×128),
  o_proj [1024,2048]; Q head h reads KV head (h%8)
- gemma3 per-head Q/K RMSNorm (add=False for Qwen3_06B → no weight+1)
- Half-split RoPE (rotate first-half vs second-half), theta 1e6, freq over even indices
- Causal mask (triu(1))
- Gated SiLU MLP (gate/up [3072,1024], down [1024,3072])
- Residual add order (residual + attn, residual + mlp)
- fp16-vs-W8 storage dispatch (norm vectors are fp16, projections are W8)

## Debugging lead (next step)
Both Swift and Python produce identical wrong output (magnitude ~100-200 vs golden ~2-25,
cosine ~ -0.04 ≈ orthogonal). Layer 0 is self-consistent, so the error compounds through
later layers. Prime suspects to check against the reference `optimized_attention` /
`BaseLlama.forward`:
1. The exact `optimized_attention` scale + `small_input` path semantics.
2. Whether the mask value (`finfo.min/4`) or mask reshaping differs from a plain -inf mask.
3. Whether `position_ids`/RoPE is applied to Q and K with the SAME position or Q gets a
   shifted position (autoregressive offset) — the reference uses `position_ids =
   arange(0, seq_len)` so no shift, but confirm.
4. bf16 golden vs fp32 reference is NOT the cause (that would be cosine ~0.99+, not -0.04).

To isolate: compute the golden's per-layer block outputs (`block_00_out`..`block_27_out`
in the golden npz) and compare each layer's hidden state against them. That pinpoints the
FIRST layer where the reference diverges from both my implementations.

## Fixture/refs on this box
- `/root/anima-harness/cond_context_case1.f32`, `qwen_ids_case1.json`, `layer0_input_emb.npy`,
  `layer0_h1.npy`, `layer0_ref_out.npy`, `layer0_*.f32`
- Python oracle: `/root/anima-xsmax/scripts/gen_tokenizer_ref.py`
