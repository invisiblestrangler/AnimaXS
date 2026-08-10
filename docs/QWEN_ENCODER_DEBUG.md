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

---

## GQA grouping — ROOT CAUSE CONFIRMED (2026-08-10, continuation agent)

Pinned source:
- ComfyUI commit: `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44` (`/root/comfy-ref`)
- Function: `comfy/ops.py::repeat_kv_for_gqa` + `comfy/ops.py::gqa_repeat_factor`

Observed GQA expansion:
```python
def gqa_repeat_factor(query_heads, key_heads, value_heads):
    assert key_heads == value_heads
    return 1 if query_heads == key_heads else (query_heads // key_heads)

def repeat_kv_for_gqa(k, v, query_heads, head_dim):
    n_rep = gqa_repeat_factor(query_heads, k.shape[head_dim], v.shape[head_dim])
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=head_dim)
        v = v.repeat_interleave(n_rep, dim=head_dim)
    return k, v
```
For Qwen3-0.6B: query_heads=16, kv_heads=8 → `n_rep = 16//8 = 2`.
`repeat_interleave(2, dim=head_dim)` on `[B, seq, 8, 128]` yields 16 heads ordered
`[K0,K0,K1,K1,K2,K2,K3,K3,K4,K4,K5,K5,K6,K6,K7,K7]` (contiguous grouped repeats).
`torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` repeats K/V with
`key.repeat_interleave(query_heads // kv_heads, dim=-3)` — identical grouping.

Derived Q→KV mapping (the CORRECT one):
```
Q0->KV0  Q1->KV0  Q2->KV1  Q3->KV1  Q4->KV2  Q5->KV2  Q6->KV3  Q7->KV3
Q8->KV4  Q9->KV4  Q10->KV5 Q11->KV5 Q12->KV6 Q13->KV6 Q14->KV7 Q15->KV7
```
Equivalent formula: `kvHead = queryHead / (queryHeads / kvHeads) = queryHead / 2`.

**The previous `kvHead = queryHead % 8` mapping was WRONG.** It paired
`[Q0,KV0],[Q1,KV1],...` (round-robin), which scrambles which V-vector each Q head reads.
This is the shared Swift==Python misunderstanding that produced cosine ≈ −0.04 vs the
golden `cond_context` even though layer-0 was internally self-consistent. Both implementations
used the same wrong `% 8`.

Fixed in: `AnimaXS/Runtime/Text/QwenEncoderCPU.swift` and `/root/anima-harness/Sources/harness/QwenEncoderCPU.swift`.
Regression test: `GqaHeadMappingTests` (see TEST_MATRIX.md).

---

## RESOLVED (2026-08-10, continuation agent) — full-28 parity achieved

Two bugs were fixed and validated against a TRUE pinned-ComfyUI structural oracle
(`scripts/qwen_comfy_oracle.py`, pinned commit `cbbc9da`):

1. **GQA grouping** (`h % 8` → `h / 2`), see above. Fixed cosine vs golden from −0.04 → 0.62.
2. **Missing final RMSNorm on `cond_context`.** The golden `cond_context` is the
   **POST-final-norm** output. `Qwen3_06BConfig.final_norm=True` (llama.py:130) and
   `SDClipModel` with `layer=="last"` returns `outputs[0]` = the main `x` AFTER
   `if self.norm is not None: x = self.norm(x)` (llama.py:800-801). The
   `layer_norm_hidden_state=False` in `anima.py` (Qwen3_06BModel) only suppresses the final
   norm on the *intermediate* path, NOT the main output. HANDOFF.json's "NO final norm"
   note is incorrect about the actual capture point. Adding the final norm moved cosine
   vs golden from 0.62 → 0.992.

Final measured results (case1, 46 tokens):
```
pinned ComfyUI commit: cbbc9da   animapk sha256: ba59e4d1...4ceab
Swift W8 vs pinned-Comfy W8 (structural, both post-final-norm):
    cosine=1.000000  rmse=0.00001  maxAbs=0.0004        ← kernel is CORRECT
Swift W8 (post-final-norm) vs golden cond_context:
    cosine=0.992164  rmse=0.434   maxAbs=47.53          ← W8 quantization deviation only
oracle pre-final-norm vs golden: cosine=0.619639        ← confirms final norm is required
```
The residual 0.008 cosine gap vs golden is 100% the expected W8-quantization-vs-original
bf16 deviation (comparison B). It is NOT a structural error — Swift and the pinned ComfyUI
W8 oracle agree to cosine 1.000000. Per GOLDENS.md §5, cosine ≥ 0.99 is the "kernel is
wrong" boundary; 0.992 with structural 1.0 proves correctness.

Per-layer oracle outputs (embedding + layer_00..27 + post-final-norm) saved at
`scripts/oracle_out/qwen_oracle_layers.npz` for future layer bring-up.

Qwen encoder gate: PASS (see TODO.md F005/F006).
