#!/usr/bin/env python3
"""
qwen_comfy_oracle.py — TRUE pinned-ComfyUI structural oracle for the Qwen3-0.6B text encoder.

Purpose (AnimaXS runbook §10-§16):
  Build a durable, reproducible oracle that executes the EXACT equations of the pinned
  ComfyUI Qwen3-0.6B implementation, using the SAME dequantized W8 animapk weights as the
  Swift encoder. This isolates model-implementation correctness (Swift vs ComfyUI math)
  from quantization fidelity (W8 vs original bf16).

Why self-contained instead of importing /root/comfy-ref directly:
  The pinned `comfy` package's import chain (comfy.utils -> comfy.memory_management ->
  comfy_aimdo, plus optional flash_attn/xformers/sageattention) is not importable on this
  CPU-only box. The continuation brief's escape hatch (runbook §13) permits a small adapter
  that CALLS the pinned math. Here the math is TRANSCRIBED VERBATIM from the pinned source
  with line citations, so this script fails if my understanding differs from ComfyUI.

Pinned source (commit cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44):
  comfy/text_encoders/llama.py          : RMSNorm, precompute_freqs_cis, apply_rope,
                                          Attention.forward, MLP, TransformerBlock,
                                          BaseLlama.forward, Qwen3_06BConfig
  comfy/ldm/modules/attention.py        : attention_pytorch / attention_basic
  comfy/ops.py                          : repeat_kv_for_gqa, gqa_repeat_factor
  comfy/sd1_clip.py                     : SDClipModel.encode (cond_context capture)
  comfy/text_encoders/anima.py          : Qwen3_06BModel (layer_norm_hidden_state=False)

Weights: dequantized W8 TE pack via inspect_animapk.Animapk.decode (group=64, fp16 scale/zero).

Output (--out fixtures dir):
  qwen_oracle_layers.npz  with keys: embedding, layer_00..layer_27, layer27_pre_final_norm,
                                     layer27_post_final_norm, token_ids, commit, pack_sha256
  Prints whole-tensor metrics vs the golden cond_context for both variants.

Usage:
  python qwen_comfy_oracle.py [--pack PATH] [--ids PATH] [--golden PATH] [--out DIR]
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/anima-xsmax/scripts")
import inspect_animapk as R

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"

# ---------------------------------------------------------------------------
# 1. Weight loading from the W8 animapk pack
# ---------------------------------------------------------------------------
class PackWeights:
    def __init__(self, pack_path):
        self.pk = R.Animapk(pack_path)
        self.recs = self.pk.read_table()
        byname = {}
        for rec in self.recs:
            byname[rec["name"]] = rec
        self.byname = byname
        sha = hashlib.sha256()
        with open(pack_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        self.pack_sha256 = sha.hexdigest()

    def tensor(self, name):
        rec = self.byname.get(name)
        if rec is None:
            raise KeyError(name)
        return torch.from_numpy(self.pk.decode(rec)).float()

    def has(self, name):
        return name in self.byname


# ---------------------------------------------------------------------------
# 2. Pinned ComfyUI equations — VERBATIM transcriptions with line citations
# ---------------------------------------------------------------------------
# comfy/ops.py:42-47
def gqa_repeat_factor(query_heads, key_heads, value_heads):
    if query_heads == key_heads:
        return 1
    if key_heads == value_heads:
        return query_heads // key_heads
    raise ValueError(f"key_heads {key_heads} != value_heads {value_heads}")

# comfy/ops.py:49-54
def repeat_kv_for_gqa(k, v, query_heads, head_dim):
    n_rep = gqa_repeat_factor(query_heads, k.shape[head_dim], v.shape[head_dim])
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=head_dim)
        v = v.repeat_interleave(n_rep, dim=head_dim)
    return k, v

# comfy/text_encoders/llama.py:424-469 (precompute_freqs_cis, non-interleaved path)
def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None):
    if not isinstance(theta, list):
        theta = [theta]
    out = []
    for index, t in enumerate(theta):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (t ** (theta_numerator / head_dim))
        if rope_scale is not None:
            if isinstance(rope_scale, list):
                inv_freq /= rope_scale[index]
            else:
                inv_freq /= rope_scale
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        sin_split = sin.shape[-1] // 2
        out.append((cos, sin[..., : sin_split], -sin[..., sin_split:]))
    if len(out) == 1:
        return out[0]
    return out

# comfy/text_encoders/llama.py:471-487
def apply_rope(xq, xk, freqs_cis):
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    nsin = freqs_cis[2]
    q_embed = (xq * cos)
    q_split = q_embed.shape[-1] // 2
    q_embed[..., : q_split].addcmul_(xq[..., q_split:], nsin)
    q_embed[..., q_split:].addcmul_(xq[..., : q_split], sin)
    k_embed = (xk * cos)
    k_split = k_embed.shape[-1] // 2
    k_embed[..., : k_split].addcmul_(xk[..., k_split:], nsin)
    k_embed[..., k_split:].addcmul_(xk[..., : k_split], sin)
    return q_embed.to(org_dtype), k_embed.to(org_dtype)

# comfy/text_encoders/llama.py:415-420
def rms_norm(x, w, eps):
    # comfy.ldm.common_dit.rms_norm == comfy.rmsnorm.rms_norm
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps)

# comfy/ldm/modules/attention.py:505-546 (attention_pytorch, mask path)
# uses comfy.ops.scaled_dot_product_attention -> which (comfy/ops.py:56-61) when
# enable_gqa=True and attn_mask is not None calls repeat_kv_for_gqa first, then
# torch.nn.functional.scaled_dot_product_attention with enable_gqa=False.
def attention_pytorch(q, k, v, heads, mask=None, skip_reshape=True, scale=None, enable_gqa=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    if enable_gqa and mask is not None:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
        enable_gqa = False
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
        scale=(dim_head ** -0.5 if scale is None else scale),
        enable_gqa=enable_gqa)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

# comfy/ldm/modules/attention.py:167-229 (attention_basic) — equivalent reference used when
# pytorch attention is disabled; reproduced to cross-check the GQA grouping exactly.
def attention_basic(q, k, v, heads, mask=None, scale=None, enable_gqa=False, skip_reshape=True):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    h = heads
    if enable_gqa:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
    q, k, v = map(lambda t: t.reshape(b * heads, -1, dim_head), (q, k, v))
    scale = scale if scale is not None else dim_head ** -0.5
    sim = torch.einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
    if mask is not None:
        # comfy/ldm/modules/attention.py:198-210 — additive mask
        if mask.dtype == torch.bool:
            raise NotImplementedError("bool mask not used for Qwen")
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
        sim.add_(mask)
    sim = sim.softmax(dim=-1)
    out = torch.einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)
    out = out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out


# ---------------------------------------------------------------------------
# 3. Model forward — transcribed from comfy/text_encoders/llama.py
# ---------------------------------------------------------------------------
class QwenOracle:
    def __init__(self, pack_path, device="cpu", dtype=torch.float32):
        self.pw = PackWeights(pack_path)
        self.device = device
        self.dtype = dtype
        self.head_dim = 128
        self.num_heads = 16
        self.num_kv_heads = 8
        self.hidden = 1024
        self.intermediate = 3072
        self.num_layers = 28
        self.eps = 1e-6
        self.rope_theta = 1e6
        self.rms_norm_add = False
        self.mlp_activation = "silu"
        self.qkv_bias = False
        self.final_norm = True          # Qwen3_06BConfig.final_norm=True (llama.py:130)
        self.layer_norm_hidden_state = False   # anima.py:41 — affects ONLY intermediate

        self.embed = self.pw.tensor("model.embed_tokens.weight")
        self.norm_w = self.pw.tensor("model.norm.weight")
        self.layers = []
        for l in range(self.num_layers):
            p = f"model.layers.{l}."
            self.layers.append({
                "in_norm": self.pw.tensor(p + "input_layernorm.weight"),
                "post_norm": self.pw.tensor(p + "post_attention_layernorm.weight"),
                "q": self.pw.tensor(p + "self_attn.q_proj.weight"),
                "k": self.pw.tensor(p + "self_attn.k_proj.weight"),
                "v": self.pw.tensor(p + "self_attn.v_proj.weight"),
                "o": self.pw.tensor(p + "self_attn.o_proj.weight"),
                "q_norm": self.pw.tensor(p + "self_attn.q_norm.weight"),
                "k_norm": self.pw.tensor(p + "self_attn.k_norm.weight"),
                "gate": self.pw.tensor(p + "mlp.gate_proj.weight"),
                "up": self.pw.tensor(p + "mlp.up_proj.weight"),
                "down": self.pw.tensor(p + "mlp.down_proj.weight"),
            })
        for name, t in [("embed", self.embed), ("norm_w", self.norm_w)]:
            pass

    def forward(self, input_ids, attention_mask=None, capture_intermediates=True):
        """Transcribed from BaseLlama.forward (llama.py:728-816)."""
        x = self.embed[input_ids.long()]  # [1, seq, 1024] gather; embed_tokens(x)
        seq_len = x.shape[1]
        device = x.device
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)
        else:
            mask = None
        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask = mask + causal_mask
            else:
                mask = causal_mask

        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(self.head_dim, position_ids, self.rope_theta,
                                         rope_scale=None, rope_dims=None, device=device)

        layers_out = {}
        for i, layer in enumerate(self.layers):
            x = self._transformer_block(x, layer, mask, freqs_cis)
            if capture_intermediates:
                layers_out[f"layer_{i:02d}"] = x.detach().clone()
        pre_final_norm = x.detach().clone()
        if self.final_norm:
            x = rms_norm(x, self.norm_w, self.eps)
        post_final_norm = x.detach().clone()
        return x, {"embedding": self.embed[input_ids.long()].detach().clone(),
                   **layers_out,
                   "layer27_pre_final_norm": pre_final_norm,
                   "layer27_post_final_norm": post_final_norm}

    def _transformer_block(self, x, layer, mask, freqs_cis):
        # TransformerBlock.forward (llama.py:591-617)
        residual = x
        h1 = rms_norm(x, layer["in_norm"], self.eps)
        # self_attn
        B, S, _ = h1.shape
        xq = h1 @ layer["q"].T
        xk = h1 @ layer["k"].T
        xv = h1 @ layer["v"].T
        xq = xq.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xq = rms_norm(xq, layer["q_norm"], self.eps)
        xk = rms_norm(xk, layer["k_norm"], self.eps)
        xq, xk = apply_rope(xq, xk, freqs_cis)
        # optimized_attention(skip_reshape=True, enable_gqa=True)  [B,heads,S,128]
        attn = attention_pytorch(xq, xk, xv, self.num_heads, mask=mask, skip_reshape=True, enable_gqa=True)
        attn = attn @ layer["o"].T        # o_proj [1024,2048]; out [B,S,1024]
        x = residual + attn
        # MLP (llama.py:568-581)
        residual = x
        h2 = rms_norm(x, layer["post_norm"], self.eps)
        gate = h2 @ layer["gate"].T
        up = h2 @ layer["up"].T
        if self.mlp_activation == "silu":
            act = torch.nn.functional.silu(gate)
        else:
            act = torch.nn.functional.gelu(gate, approximate="tanh")
        down = (act * up) @ layer["down"].T
        x = residual + down
        return x


# ---------------------------------------------------------------------------
# 4. Metrics + main
# ---------------------------------------------------------------------------
def cosine(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return (a @ b) / (a.norm() * b.norm())

def maxabs(a, b):
    return (a - b).abs().max().item()

def rmse(a, b):
    return ((a - b).float().pow(2).mean()).sqrt().item()

def report(name, a, b):
    print(f"{name:38s} cosine={cosine(a,b):+.6f}  rmse={rmse(a,b):.5f}  maxAbs={maxabs(a,b):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/root/anima-xsmax/results/packs/qwen3-0.6b-xsmax-w8.animapk")
    ap.add_argument("--ids", default="/root/anima-harness/qwen_ids_case1.json")
    ap.add_argument("--golden", default="/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz")
    ap.add_argument("--out", default="/root/AnimaXS/scripts/oracle_out")
    args = ap.parse_args()

    with open(args.ids) as f:
        ids = json.load(f)
    input_ids = torch.tensor([ids], dtype=torch.long)
    print(f"Pinned ComfyUI commit: {PINNED_COMMIT}")
    print(f"ids: {len(ids)} tokens  first6={ids[:6]}")

    orc = QwenOracle(args.pack)
    print(f"pack SHA256: {orc.pw.pack_sha256}")
    t0 = time.time()
    final, cap = orc.forward(input_ids)
    print(f"forward done in {time.time()-t0:.1f}s  final shape {tuple(final.shape)}")

    os.makedirs(args.out, exist_ok=True)
    out_npz = os.path.join(args.out, "qwen_oracle_layers.npz")
    save = {"token_ids": np.array(ids, dtype=np.int32),
            "commit": PINNED_COMMIT, "pack_sha256": orc.pw.pack_sha256}
    for k, v in cap.items():
        if isinstance(v, torch.Tensor):
            save[k] = v.float().numpy()
    np.savez_compressed(out_npz, **save)
    print("saved", out_npz)

    # Compare against golden cond_context
    golden = np.load(args.golden)["cond_context"]  # [1,46,1024]
    g = torch.from_numpy(golden).float()
    print("\n=== vs golden cond_context ===")
    report("oracle POST-final-norm", final, g)
    report("oracle PRE-final-norm (layer27)", cap["layer27_pre_final_norm"], g)
    print("\nGOLDEN anchors:")
    gg = golden[0]
    print(f"  token0 d0..5:    {gg[0,:6]}")
    print(f"  token23 d0..5:   {gg[23,:6]}")
    print(f"  token45 d0..5:   {gg[45,:6]}")
    print(f"  token0 d128..129:{gg[0,128:130]}")
    print(f"  token0 d1023:    {gg[0,1023]:.6f}")

    # Optional: compare against the Swift harness full output if provided
    swift_path = os.path.join(args.out, "swift_full28_case1.f32")
    if os.path.exists(swift_path):
        sw = np.fromfile(swift_path, dtype=np.float32).reshape(1, 46, 1024)
        swt = torch.from_numpy(sw).float()
        print("\n=== Swift full-28 vs oracle POST-final-norm ===")
        report("Swift vs oracle post-norm", swt, final)
        print("=== Swift full-28 vs golden ===")
        report("Swift vs golden", swt, g)

if __name__ == "__main__":
    main()
