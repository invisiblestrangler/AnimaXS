#!/usr/bin/env python3
"""
anima_adapter_oracle.py — pinned-ComfyUI structural oracle for the LLMAdapter (lllite).

Purpose (AnimaXS runbook §28, G001/G002):
  Compute the adapter conditioning output [1,512,1024] using the EXACT equations of the
  pinned ComfyUI LLMAdapter, with the SAME W4-dequantized adapter weights from the DiT
  animapk that the Swift app consumes. Validates model-implementation correctness
  (Swift vs ComfyUI math) independently of quantization fidelity.

Pinned source (commit cbbc9da):
  comfy/ldm/anima/model.py   : LLMAdapter, TransformerBlock, Attention, RotaryEmbedding,
                                rotate_half, apply_rotary_pos_emb, Anima.preprocess_text_embeds
  comfy/ops.py               : repeat_kv_for_gqa (not used here — adapter is MHA, 16 heads)

Inputs (case1):
  context  = Qwen last hidden [1, 46, 1024]  (from golden cond_context, the actual adapter input)
  target   = T5 token ids [47]
  weights  = t5xxl_weights (all 1.0)

Adapter config (RUNTIME_CONSTANTS adapter_lllite + MODEL_ARCHITECTURE adapter):
  source_dim=target_dim=model_dim=1024, num_layers=6, num_heads=16, head_dim=64,
  mlp_ratio=4.0 (mlp_size 4096), rope_theta=10000, in_proj=Identity, use_self_attn=True,
  final RMSNorm eps 1e-6. out_proj/MLP have biases (default nn.Linear behavior).

Usage:
  python anima_adapter_oracle.py [--dit-pack PATH] [--golden PATH] [--qwen-ctx PATH] [--out DIR]
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/anima-xsmax/scripts")
import inspect_animapk as R

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"


# ---------------------------------------------------------------------------
# 1. Weight loading from the W4 DiT animapk (adapter subset)
# ---------------------------------------------------------------------------
class AdapterWeights:
    def __init__(self, pack_path):
        self.pk = R.Animapk(pack_path)
        self.recs = self.pk.read_table()
        # JSON tensor_meta carries FULL names (binary table truncates to 64 chars);
        # match by blob_offset (unique) to build a name -> rec map.
        self.byname = {}
        meta = {m["blob_offset"]: m.get("name", "") for m in self.pk.meta.get("tensor_meta", [])}
        for rec in self.recs:
            full = meta.get(rec["blob_offset"], rec["name"])
            self.byname[full] = rec
        sha = hashlib.sha256()
        with open(pack_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        self.pack_sha256 = sha.hexdigest()

    def get(self, name):
        full = "model.diffusion_model." + name
        rec = self.byname.get(full)
        if rec is None:
            raise KeyError(full)
        return torch.from_numpy(self.pk.decode(rec)).float()


# ---------------------------------------------------------------------------
# 2. Pinned ComfyUI equations — VERBATIM (comfy/ldm/anima/model.py)
# ---------------------------------------------------------------------------
# model.py:7-10
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# model.py:13-17
def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

# model.py:20-39 (RotaryEmbedding.forward)
def rotary_emb(inv_freq, x, position_ids):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# RMSNorm (comfy/rmsnorm.py + comfy/ops)
def rms_norm(x, w, eps=1e-6):
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps)


# ---------------------------------------------------------------------------
# 3. Adapter forward — transcribed from comfy/ldm/anima/model.py
# ---------------------------------------------------------------------------
class AdapterOracle:
    def __init__(self, dit_pack):
        self.w = AdapterWeights(dit_pack)
        self.model_dim = 1024
        self.num_heads = 16
        self.head_dim = self.model_dim // self.num_heads  # 64
        self.num_layers = 6
        self.mlp_size = 4096
        self.eps = 1e-6
        self.rope_theta = 10000.0
        # RotaryEmbedding inv_freq (model.py:24)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim))
        self.inv_freq = inv_freq

    def embed(self):
        return self.w.get("llm_adapter.embed.weight")

    def block(self, i):
        p = f"llm_adapter.blocks.{i}."
        d = {
            "self_attn": {
                "q_proj": self.w.get(p + "self_attn.q_proj.weight"),
                "k_proj": self.w.get(p + "self_attn.k_proj.weight"),
                "v_proj": self.w.get(p + "self_attn.v_proj.weight"),
                "o_proj": self.w.get(p + "self_attn.o_proj.weight"),
                "q_norm": self.w.get(p + "self_attn.q_norm.weight"),
                "k_norm": self.w.get(p + "self_attn.k_norm.weight"),
            },
            "cross_attn": {
                "q_proj": self.w.get(p + "cross_attn.q_proj.weight"),
                "k_proj": self.w.get(p + "cross_attn.k_proj.weight"),
                "v_proj": self.w.get(p + "cross_attn.v_proj.weight"),
                "o_proj": self.w.get(p + "cross_attn.o_proj.weight"),
                "q_norm": self.w.get(p + "cross_attn.q_norm.weight"),
                "k_norm": self.w.get(p + "cross_attn.k_norm.weight"),
            },
            "norm_self_attn": self.w.get(p + "norm_self_attn.weight"),
            "norm_cross_attn": self.w.get(p + "norm_cross_attn.weight"),
            "norm_mlp": self.w.get(p + "norm_mlp.weight"),
            "mlp0_w": self.w.get(p + "mlp.0.weight"),
            "mlp0_b": self.w.get(p + "mlp.0.bias"),
            "mlp2_w": self.w.get(p + "mlp.2.weight"),
            "mlp2_b": self.w.get(p + "mlp.2.bias"),
        }
        return d

    def out_proj(self):
        return (self.w.get("llm_adapter.out_proj.weight"),
                self.w.get("llm_adapter.out_proj.bias"))

    def norm_w(self):
        return self.w.get("llm_adapter.norm.weight")

    def forward(self, context, target_ids, target_weights, capture=True):
        """LLMAdapter.forward + Anima.preprocess_text_embeds (model.py:171-208)."""
        # x = in_proj(embed(target)) — in_proj is Identity
        x = self.embed()[target_ids]  # [T, 1024] → unsqueeze [1, T, 1024]
        x = x.unsqueeze(0)
        context = context.unsqueeze(0) if context.dim() == 2 else context
        T = x.shape[1]
        Ct = context.shape[1]

        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(Ct, device=x.device).unsqueeze(0)
        pos_emb = rotary_emb(self.inv_freq, x, position_ids)          # (cos,sin) for target len T
        pos_emb_ctx = rotary_emb(self.inv_freq, x, position_ids_context)  # for context len Ct

        layers = {}
        for i in range(self.num_layers):
            b = self.block(i)
            x = self._block(x, context, b, pos_emb, pos_emb_ctx)
            if capture:
                layers[f"adapter_block_{i:02d}"] = x.detach().clone()

        ow, ob = self.out_proj()
        x = (x @ ow.T) + ob        # out_proj (with bias)
        x = rms_norm(x, self.norm_w(), self.eps)   # final RMSNorm
        out = x.detach().clone()

        # Anima.preprocess_text_embeds (model.py:198-208)
        out_w = out * target_weights.unsqueeze(0).unsqueeze(-1).float()
        T = out_w.shape[1]
        if T < 512:
            out_pad = F.pad(out_w, (0, 0, 0, 512 - T))
        else:
            out_pad = out_w
        return out_w, out_pad, layers

    def _block(self, x, context, b, pos_emb, pos_emb_ctx):
        """TransformerBlock.forward (model.py:125-136)."""
        # self-attn
        if True:  # use_self_attn=True
            normed = rms_norm(x, b["norm_self_attn"], self.eps)
            attn_out = self._attn_mha(normed, normed, b["self_attn"], None, pos_emb, pos_emb)
            x = x + attn_out
        # cross-attn
        normed = rms_norm(x, b["norm_cross_attn"], self.eps)
        attn_out = self._attn_mha(normed, context, b["cross_attn"], None, pos_emb, pos_emb_ctx)
        x = x + attn_out
        # mlp
        x = x + self._mlp(rms_norm(x, b["norm_mlp"], self.eps), b)
        return x

    def _attn_mha(self, qx, ctx, attn, mask, pos_emb, pos_emb_ctx):
        """Attention.forward (model.py:62-84): MHA (16 heads, no GQA)."""
        B, S, _ = qx.shape
        C = ctx.shape[1]
        q = qx @ attn["q_proj"].T
        k = ctx @ attn["k_proj"].T
        v = ctx @ attn["v_proj"].T
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)   # [B,16,S,64]
        k = k.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q, attn["q_norm"], self.eps)
        k = rms_norm(k, attn["k_norm"], self.eps)
        cos, sin = pos_emb
        q = apply_rotary_pos_emb(q, cos, sin)
        cos, sin = pos_emb_ctx
        k = apply_rotary_pos_emb(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, S, -1).contiguous()
        return out @ attn["o_proj"].T

    def _mlp(self, x, b):
        """MLP (model.py:119-123): Linear→GELU→Linear with biases."""
        h = (x @ b["mlp0_w"].T) + b["mlp0_b"]
        h = F.gelu(h)
        return (h @ b["mlp2_w"].T) + b["mlp2_b"]


# ---------------------------------------------------------------------------
# 4. Metrics + main
# ---------------------------------------------------------------------------
def cosine(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return (a @ b) / (a.norm() * b.norm())

def maxabs(a, b): return (a - b).abs().max().item()
def rmse(a, b): return ((a - b).float().pow(2).mean()).sqrt().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-pack", default="/root/anima-xsmax/results/packs/anima-turbo-v1.0-xsmax-w4.animapk")
    ap.add_argument("--golden", default="/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz")
    ap.add_argument("--qwen-ctx", default="/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz")
    ap.add_argument("--out", default="/root/AnimaXS/scripts/oracle_out")
    args = ap.parse_args()

    orc = AdapterOracle(args.dit_pack)
    print(f"Pinned ComfyUI commit: {PINNED_COMMIT}")
    print(f"DiT pack SHA256: {orc.w.pack_sha256}")

    g = np.load(args.golden)
    context = torch.from_numpy(g["cond_context"]).float()   # [1,46,1024] (adapter input)
    t5_ids = torch.from_numpy(g["cond_meta_t5xxl_ids"].astype(np.int64))
    t5_w = torch.from_numpy(g["cond_meta_t5xxl_weights"].astype(np.float32))
    print(f"context {tuple(context.shape)}  t5 ids {len(t5_ids)}  t5 weights all1={(t5_w==1.0).all().item()}")

    t0 = time.time()
    out_w, out_pad, layers = orc.forward(context, t5_ids, t5_w)
    print(f"adapter forward done in {time.time()-t0:.1f}s")

    print(f"adapter out (weighted) shape {tuple(out_w.shape)}  padded {tuple(out_pad.shape)}")
    print(f"adapter out [0,0,:6] = {out_w[0,0,:6].tolist()}")
    print(f"adapter out global min/max = {out_w.min():.4f} / {out_w.max():.4f}")
    print(f"adapter out allFinite = {bool(torch.isfinite(out_w).all().item())}")
    # per-token L2 norms of the weighted output
    norms = out_w[0].pow(2).sum(-1).sqrt()
    print(f"adapter out per-token L2 (0..5) = {[round(float(x),3) for x in norms[:6]]}")

    os.makedirs(args.out, exist_ok=True)
    save = {
        "t5_ids": t5_ids.numpy(),
        "t5_weights": t5_w.numpy(),
        "context": context.float().numpy(),
        "adapter_out_weighted": out_w.float().numpy(),
        "adapter_out_padded512": out_pad.float().numpy(),
        "commit": PINNED_COMMIT, "pack_sha256": orc.w.pack_sha256,
    }
    for k, v in layers.items():
        save[k] = v.float().numpy()
    npz = os.path.join(args.out, "adapter_oracle_case1.npz")
    np.savez_compressed(npz, **save)
    print("saved", npz)

if __name__ == "__main__":
    main()
