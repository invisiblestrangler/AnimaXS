#!/usr/bin/env python3
"""dit_source_oracle.py — run the PINNED Anima source model (predict2.py + anima
model.py) in torch on CPU, for source-vs-Swift trajectory parity.

This answers the question the same-input block oracles cannot: does the Swift
Metal implementation reproduce the actual source model computation over the
diffusion trajectory? All prior parity tests compared Swift against a
transcription of the same source; a transcription blind spot would be shared by
both. This script executes the pinned equations directly in torch.

The pinned source model is native bfloat16 (supported_inference_dtypes lists
bfloat16 first; all 685 DiT tensors are bf16 in the safetensors), so the
default compute dtype is bf16 (predict2.py keeps a bf16 residual stream; the
fp32 residual promotion at predict2.py:918 only fires for float16 inputs).

Modes:
  --mode trajectory --capture DIR [--dtype bf16|fp16|fp32]
      For each captured stepXX_x_in.f32 + stepXX_denoised.f32 (Swift capture),
      run the source forward(x_i, sigma_i, context) and compare the source
      velocity with Swift's v_i = (x_i - denoised_i)/sigma_i. Emits per-step
      cosine/RMSE/maxAbs plus the Nyquist (latent-checkerboard) energy of the
      error, which isolates whether the known 8px output grid is a per-step
      structural error or accumulated numeric drift.

  --mode endtoend --fixtures DIR [--dtype bf16|fp16|fp32]
      Reproduce the canonical 8-step trajectory from the committed fixtures
      (case1_noise.f32 + case1_cond_context.f32 + case1_t5_ids.i64, t5 weights
      all 1.0 per D063) through the source adapter + DiT + Euler, and compare
      the final latent against case1_final_latent.f32. Also reports the
      Nyquist energy of the source final latent itself.

Usage:
  python dit_source_oracle.py --source PATH/anima-turbo-v1.0.safetensors \
      --mode trajectory --capture DIR --out OUT
  python dit_source_oracle.py --source PATH/anima-turbo-v1.0.safetensors \
      --mode endtoend --fixtures DIR --out OUT

Pinned source: circlestone-labs/Anima @ f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b
  comfy/ldm/cosmos/predict2.py (MiniTrainDIT, Block, FinalLayer, PatchEmbed,
  Timesteps, TimestepEmbedding, Attention)
  comfy/ldm/cosmos/position_embedding.py (VideoRopePosition3DEmb)
  comfy/ldm/anima/model.py (LLMAdapter, Anima.preprocess_text_embeds)
  comfy/model_detection.py Anima config: 28 blocks, 16 heads, head_dim 128,
  mlp_ratio 4, concat_padding_mask, rope h/w ratio 4.0, t ratio 1.0,
  use_adaln_lora, adaln_lora_dim 256.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

DIM = 2048
HEADS = 16
HEAD_DIM = 128
CTX_DIM = 1024
MLP_HID = 8192
LORA = 256
NUM_BLOCKS = 28
TOKENS = 1024          # 1 x 32 x 32
CTX_TOKENS = 512
LATENT_EL = 16 * 64 * 64
EPS = 1e-6

PINNED_SOURCE_SHA = "c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174"


# ---------------------------------------------------------------------------
# Numerics helpers (fp32 compute, cast to model dtype — CUDA bf16 semantics)
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=EPS):
    """RMSNorm over the last dim, fp32 statistics, output in x dtype."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (out * weight.float()).to(x.dtype)


def layer_norm_mean_center(x, eps=EPS):
    """LayerNorm(elementwise_affine=False, eps=1e-6) — mean CENTERING."""
    return F.layer_norm(x.float(), (x.shape[-1],), None, None, eps).to(x.dtype)


def linear(x, w, b=None):
    """Matmul with fp32 accumulation, output rounded to x dtype (CUDA bf16)."""
    out = F.linear(x.float(), w.float(), b)
    return out.to(x.dtype)


def gelu(x):
    """nn.GELU() default (exact erf)."""
    return F.gelu(x.float()).to(x.dtype)


def silu(x):
    return F.silu(x.float()).to(x.dtype)


def scaled_dot_product(q, k, v):
    """optimized_attention (skip_reshape=True): heads folded into the batch,
    scale 1/sqrt(head_dim), fp32 softmax, bf16 PV."""
    scale = HEAD_DIM ** -0.5
    b, s_q, h, d = q.shape
    s_k = k.shape[1]
    q2 = q.reshape(b * h, s_q, d)
    k2 = k.reshape(b * h, s_k, d)
    v2 = v.reshape(b * h, s_k, d)
    scores = (q2.float() @ k2.float().transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    out = (probs.to(q.dtype) @ v2).reshape(b, s_q, h, d)
    return out


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def dit_rope(h=32, w=32, t=1, head_dim=HEAD_DIM, dtype=torch.float32):
    """VideoRopePosition3DEmb for T=1,H=32,W=32 (position_embedding.py:57-163).

    Returns [T*H*W, 64, 2, 2] float32; each block [[cos,-sin],[sin,cos]].
    """
    dim_h = head_dim // 6 * 2   # 42
    dim_w = dim_h               # 42
    dim_t = head_dim - 2 * dim_h  # 44
    h_ratio = w_ratio = 4.0
    t_ratio = 1.0
    h_ntk = h_ratio ** (dim_h / (dim_h - 2))
    w_ntk = w_ratio ** (dim_w / (dim_w - 2))
    t_ntk = t_ratio ** (dim_t / (dim_t - 2))
    # Float32 rounding of the thetas matches Swift Float(10000*pow(...)) (D029)
    h_theta = torch.tensor(10000.0 * h_ntk, dtype=torch.float32)
    w_theta = torch.tensor(10000.0 * w_ntk, dtype=torch.float32)
    t_theta = torch.tensor(10000.0 * t_ntk, dtype=torch.float32)

    dim_h_range = (torch.arange(0, dim_h, 2)[:dim_h // 2].float() / dim_h)  # 21
    dim_w_range = (torch.arange(0, dim_w, 2)[:dim_w // 2].float() / dim_w)  # 21
    dim_t_range = (torch.arange(0, dim_t, 2)[:dim_t // 2].float() / dim_t)  # 22

    h_freqs = 1.0 / (h_theta ** dim_h_range)
    w_freqs = 1.0 / (w_theta ** dim_w_range)
    t_freqs = 1.0 / (t_theta ** dim_t_range)

    seq = torch.arange(max(h, w, t), dtype=torch.float32)
    half_h = torch.outer(seq[:h], h_freqs)
    half_w = torch.outer(seq[:w], w_freqs)
    half_t = torch.outer(seq[:t], t_freqs)

    def blocks(half):
        return torch.stack([torch.cos(half), -torch.sin(half),
                            torch.sin(half), torch.cos(half)], dim=-1)  # [..,4]

    emb_t = blocks(half_t)  # [t,22,4]
    emb_h = blocks(half_h)  # [h,21,4]
    emb_w = blocks(half_w)  # [w,21,4]

    em = torch.cat([
        emb_t.unsqueeze(1).unsqueeze(2).expand(t, h, w, 22, 4),
        emb_h.unsqueeze(0).unsqueeze(2).expand(t, h, w, 21, 4),
        emb_w.unsqueeze(0).unsqueeze(1).expand(t, h, w, 21, 4),
    ], dim=-2)  # [t,h,w,64,4]
    rope = em.reshape(t * h * w, 64, 2, 2).float()
    return rope.to(dtype)


def apply_split_half_rope(q, k, rope):
    """rms_rope_split_half: rotate pairs (p, p+64) per head using rope blocks.

    q/k: [B, S, H, 128] (already RMSNormed with the shared [128] weight).
    rope: [S, 64, 2, 2].
    """
    b, s, heads, hd = q.shape
    cos = rope[:, :, 0, 0].unsqueeze(0).unsqueeze(2)  # [1,S,1,64]
    sin = rope[:, :, 1, 0].unsqueeze(0).unsqueeze(2)
    a = q[..., :64]
    bb = q[..., 64:]
    qr = torch.empty_like(q)
    qr[..., :64] = cos * a - sin * bb
    qr[..., 64:] = sin * a + cos * bb
    a = k[..., :64]
    bb = k[..., 64:]
    kr = torch.empty_like(k)
    kr[..., :64] = cos * a - sin * bb
    kr[..., 64:] = sin * a + cos * bb
    return qr, kr


def adapter_rope(seq_len, head_dim=64, dtype=torch.float32):
    """LLMAdapter RotaryEmbedding (anima/model.py:20-40), theta 10000."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float().unsqueeze(1)
    freqs = pos @ inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotate_half(q, cos, sin):
    """apply_rotary_pos_emb (anima/model.py:12-18): x*cos + rotate_half(x)*sin
    with FULL-width cos/sin (head_dim long)."""
    half = q.shape[-1] // 2
    x1 = q[..., :half]
    x2 = q[..., half:]
    rh = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(1)  # [S,1,D] broadcast over heads
    sin = sin.unsqueeze(1)
    return q * cos + rh * sin


# ---------------------------------------------------------------------------
# DiT components (predict2.py)
# ---------------------------------------------------------------------------
class DitTimesteps:
    """Timesteps(2048): sigma -> sinusoidal; cos [0:1024], sin [1024:2048]."""

    def __call__(self, timesteps, dtype):
        half_dim = DIM // 2
        exponent = -np.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.float().flatten()[:, None] * emb[None, :]
        out = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return out.to(dtype)


class DitTimestepEmbedding:
    """TimestepEmbedding(2048, 2048, use_adaln_lora=True)."""

    def __init__(self, w):
        self.w1 = w["t_embedder.1.linear_1.weight"]
        self.w2 = w["t_embedder.1.linear_2.weight"]

    def __call__(self, sample):
        emb = linear(sample, self.w1)
        emb = silu(emb)
        emb = linear(emb, self.w2)
        return sample, emb  # emb_B_T_D = sample (raw), adaln_lora = emb


class DitAttention:
    def __init__(self, w, prefix, is_self):
        self.is_self = is_self
        self.q_proj = w[f"{prefix}.q_proj.weight"]
        self.k_proj = w[f"{prefix}.k_proj.weight"]
        self.v_proj = w[f"{prefix}.v_proj.weight"]
        self.o_proj = w[f"{prefix}.output_proj.weight"]
        self.q_norm = w[f"{prefix}.q_norm.weight"]
        self.k_norm = w[f"{prefix}.k_norm.weight"]

    def forward(self, x, context, rope):
        # x: [B, S, 2048] in compute dtype
        b, s, _ = x.shape
        c = context if context is not None else x
        q = linear(x, self.q_proj).view(b, s, HEADS, HEAD_DIM)
        k = linear(c, self.k_proj).view(b, c.shape[1], HEADS, HEAD_DIM)
        v = linear(c, self.v_proj).view(b, c.shape[1], HEADS, HEAD_DIM)
        q = rms_norm(q, self.q_norm)
        k = rms_norm(k, self.k_norm)
        if self.is_self and rope is not None:
            q, k = apply_split_half_rope(q, k, rope)
        out = scaled_dot_product(q, k, v)  # [B, S, H, HD]
        out = out.reshape(b, s, -1)
        return linear(out, self.o_proj)


class DitBlock:
    def __init__(self, w, i):
        p = f"blocks.{i}"
        self.norm_s = lambda x: layer_norm_mean_center(x)
        self.norm_c = lambda x: layer_norm_mean_center(x)
        self.norm_m = lambda x: layer_norm_mean_center(x)
        self.self_attn = DitAttention(w, f"{p}.self_attn", True)
        self.cross_attn = DitAttention(w, f"{p}.cross_attn", False)
        self.mlp1 = w[f"{p}.mlp.layer1.weight"]
        self.mlp2 = w[f"{p}.mlp.layer2.weight"]
        self.mod_s = (w[f"{p}.adaln_modulation_self_attn.1.weight"],
                      w[f"{p}.adaln_modulation_self_attn.2.weight"])
        self.mod_c = (w[f"{p}.adaln_modulation_cross_attn.1.weight"],
                      w[f"{p}.adaln_modulation_cross_attn.2.weight"])
        self.mod_m = (w[f"{p}.adaln_modulation_mlp.1.weight"],
                      w[f"{p}.adaln_modulation_mlp.2.weight"])

    def modulate(self, emb, adaln, mod):
        w1, w2 = mod
        m = linear(silu(emb), w1)
        m = linear(m, w2)
        m = m + adaln
        shift, scale, gate = m.chunk(3, dim=-1)
        return shift, scale, gate

    def forward(self, x, emb, adaln, context, rope):
        # x: [B,1,H,W,D] residual (compute dtype), emb/adaln [B,1,D]/[B,1,6144]
        b, t, h, w, d = x.shape
        xs = x.reshape(b, t * h * w, d)

        def branch(norm, attn, ctx, rope_use, mod):
            shift, scale, gate = self.modulate(emb, adaln, mod)
            normed = norm(x) * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)
            nt = normed.reshape(b, t * h * w, d)
            out = attn.forward(nt, ctx, rope_use)
            out = out.reshape(b, t, h, w, d)
            return out, gate

        out, gate = branch(self.norm_s, self.self_attn, None, rope, self.mod_s)
        x = x + gate.unsqueeze(2).unsqueeze(3) * out
        out, gate = branch(self.norm_c, self.cross_attn, context, None, self.mod_c)
        x = x + gate.unsqueeze(2).unsqueeze(3) * out
        shift, scale, gate = self.modulate(emb, adaln, self.mod_m)
        normed = self.norm_m(x) * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)
        hidden = linear(normed.reshape(b, t * h * w, d), self.mlp1)
        hidden = gelu(hidden)
        out = linear(hidden, self.mlp2).reshape(b, t, h, w, d)
        x = x + gate.unsqueeze(2).unsqueeze(3) * out
        return x


class DitFinalLayer:
    def __init__(self, w):
        self.w1 = w["final_layer.adaln_modulation.1.weight"]
        self.w2 = w["final_layer.adaln_modulation.2.weight"]
        self.linear_w = w["final_layer.linear.weight"]

    def forward(self, x, emb, adaln):
        m = linear(silu(emb), self.w1)
        m = linear(m, self.w2)
        m = m + adaln[:, :, : 2 * DIM]
        shift, scale = m.chunk(2, dim=-1)
        b, t, h, w, d = x.shape
        normed = layer_norm_mean_center(x) * (1 + scale.unsqueeze(2).unsqueeze(3)) \
            + shift.unsqueeze(2).unsqueeze(3)
        out = linear(normed.reshape(b, t * h * w, d), self.linear_w)
        return out.reshape(b, t, h, w, 64)


class DitModel:
    """MiniTrainDIT forward for T=1, 64x64, 17 input channels."""

    def __init__(self, w, dtype):
        self.dtype = dtype
        self.x_proj = w["x_embedder.proj.1.weight"]
        self.t_steps = DitTimesteps()
        self.t_emb = DitTimestepEmbedding(w)
        self.t_norm = w["t_embedding_norm.weight"]
        self.blocks = [DitBlock(w, i) for i in range(NUM_BLOCKS)]
        self.final = DitFinalLayer(w)
        self.rope = dit_rope().to(dtype)

    def patchify(self, x):
        # x: [1, 17, 1, 64, 64] -> [1, 1, 32, 32, 68]
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h // 2, 2, w // 2, 2)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)  # b t h w c m n
        x = x.reshape(b, t, h // 2, w // 2, c * 4)
        return x

    def unpatchify(self, x):
        # x: [1, 1, 32, 32, 64] -> [1, 16, 1, 64, 64]
        b, t, h, w, m = x.shape
        x = x.reshape(b, t, h, w, 2, 2, 16)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # b C t H p1 W p2
        x = x.reshape(b, 16, t, h * 2, w * 2)
        return x

    def forward(self, x, sigma, context):
        """x: [1,16,1,64,64] latent (fp32), sigma: float, context [1,512,1024]."""
        dt = self.dtype
        x = x.to(dt)
        # padding-mask channel appended as zeros (predict2.py:806)
        pm = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4],
                         dtype=dt, device=x.device)
        x17 = torch.cat([x, pm], dim=1)
        tokens = self.patchify(x17)                       # [1,1,32,32,68]
        emb_tok = linear(tokens.reshape(1, TOKENS, 68), self.x_proj)
        emb_tok = emb_tok.reshape(1, 1, 32, 32, DIM)
        # source does NOT promote bf16 residual to fp32 (predict2.py:918)
        t_emb_raw = self.t_steps(torch.tensor([sigma]), dt)
        emb, adaln = self.t_emb(t_emb_raw)                # [1,D], [1,6144]
        emb = emb.unsqueeze(1)                            # -> [1,1,D] (B,T,D)
        adaln = adaln.unsqueeze(1)                        # -> [1,1,6144]
        emb = rms_norm(emb, self.t_norm)
        context = context.to(dt)
        for blk in self.blocks:
            emb_tok = blk.forward(emb_tok, emb, adaln, context, self.rope)
        emb_tok = emb_tok.to(context.dtype)               # final_layer input cast
        out = self.final.forward(emb_tok, emb, adaln)
        vel = self.unpatchify(out)[:, :, :, :64, :64]
        return vel.float()                                # [1,16,1,64,64] fp32


# ---------------------------------------------------------------------------
# LLMAdapter (anima/model.py)
# ---------------------------------------------------------------------------
class AdapterAttention:
    def __init__(self, w, prefix):
        self.q_proj = w[f"{prefix}.q_proj.weight"]
        self.k_proj = w[f"{prefix}.k_proj.weight"]
        self.v_proj = w[f"{prefix}.v_proj.weight"]
        self.o_proj = w[f"{prefix}.o_proj.weight"]
        self.q_norm = w[f"{prefix}.q_norm.weight"]
        self.k_norm = w[f"{prefix}.k_norm.weight"]

    def forward(self, x, context, cos, sin, cos_ctx, sin_ctx, mask=None):
        if context is None:
            context = x
        b, s, _ = x.shape
        cs = context.shape[1]
        q = linear(x, self.q_proj).view(b, s, 16, 64)
        k = linear(context, self.k_proj).view(b, cs, 16, 64)
        v = linear(context, self.v_proj).view(b, cs, 16, 64)
        q = rms_norm(q, self.q_norm).transpose(1, 2)
        k = rms_norm(k, self.k_norm).transpose(1, 2)
        v = v.transpose(1, 2)
        if cos is not None:
            q = apply_rotate_half(q, cos, sin)
            k = apply_rotate_half(k, cos_ctx, sin_ctx)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = attn.transpose(1, 2).reshape(b, s, -1)
        return linear(out, self.o_proj)


class AdapterBlock:
    def __init__(self, w, i):
        p = f"llm_adapter.blocks.{i}"
        self.norm_s = w[f"{p}.norm_self_attn.weight"]
        self.norm_c = w[f"{p}.norm_cross_attn.weight"]
        self.norm_m = w[f"{p}.norm_mlp.weight"]
        self.self_attn = AdapterAttention(w, f"{p}.self_attn")
        self.cross_attn = AdapterAttention(w, f"{p}.cross_attn")
        self.mlp0 = w[f"{p}.mlp.0.weight"]
        self.mlp0b = w[f"{p}.mlp.0.bias"]
        self.mlp2 = w[f"{p}.mlp.2.weight"]
        self.mlp2b = w[f"{p}.mlp.2.bias"]

    def forward(self, x, context, cos, sin, cos_ctx, sin_ctx):
        # self-attn uses position_embeddings for BOTH q and k (context=x)
        attn = self.self_attn.forward(rms_norm(x, self.norm_s), None,
                                      cos, sin, cos, sin)
        x = x + attn
        attn = self.cross_attn.forward(rms_norm(x, self.norm_c), context,
                                       cos, sin, cos_ctx, sin_ctx)
        x = x + attn
        hidden = linear(rms_norm(x, self.norm_m), self.mlp0, self.mlp0b)
        hidden = gelu(hidden)
        out = linear(hidden, self.mlp2, self.mlp2b)
        x = x + out
        return x


class Adapter:
    def __init__(self, w, dtype):
        self.dtype = dtype
        self.embed = w["llm_adapter.embed.weight"]
        self.out_proj = w["llm_adapter.out_proj.weight"]
        self.out_proj_b = w["llm_adapter.out_proj.bias"]
        self.norm = w["llm_adapter.norm.weight"]
        self.blocks = [AdapterBlock(w, i) for i in range(6)]

    def __call__(self, context, t5_ids, t5_weights):
        """preprocess_text_embeds (anima/model.py:198-208)."""
        dt = self.dtype
        context = context.to(dt)
        x = F.embedding(t5_ids, self.embed.float()).to(dt)  # [1,47,1024]
        n = x.shape[1]
        cos, sin = adapter_rope(n, dtype=dt)
        cos_ctx, sin_ctx = adapter_rope(context.shape[1], dtype=dt)
        for blk in self.blocks:
            x = blk.forward(x, context, cos, sin, cos_ctx, sin_ctx)
        out = rms_norm(linear(x, self.out_proj, self.out_proj_b), self.norm)
        w = t5_weights.float().unsqueeze(0).unsqueeze(-1).to(dt)
        out = out * w
        if out.shape[1] < CTX_TOKENS:
            out = F.pad(out, (0, 0, 0, CTX_TOKENS - out.shape[1]))
        return out  # [1,512,1024]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nyquist_energy(t, normalize=True):
    """Latent checkerboard/stripe energy at the 64x64 Nyquist bins.

    t: [16, 64, 64] fp32. Returns (row_nyquist, col_nyquist, total) energy
    fractions after per-channel mean removal (row = kx=±32, col = ky=±32).
    The known 8px output grid maps to latent Nyquist (1 latent px).
    """
    scores = []
    for c in range(t.shape[0]):
        x = t[c].numpy().astype(np.float64)
        x = x - x.mean()
        F = np.fft.rfft2(x)
        energy = np.abs(F) ** 2
        tot = energy.sum()
        # rfft2 on 64x64: kx in 0..63 (kx=32 is its own -32 alias), ky in 0..32
        row_nyq = energy[32, 0]      # horizontal stripes at latent Nyquist
        col_nyq = energy[0, 32]      # vertical stripes at latent Nyquist
        diag = energy[32, 32]        # checkerboard at latent Nyquist
        scores.append((row_nyq / tot, col_nyq / tot,
                       (row_nyq + col_nyq + diag) / tot))
    arr = np.array(scores)
    return arr.mean(axis=0)


def compare(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    rmse = float((a - b).pow(2).mean().sqrt())
    maxabs = float((a - b).abs().max())
    return cos, rmse, maxabs


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_weights(path):
    if safe_open is None:
        sys.exit("safetensors not installed (pip install safetensors)")
    w = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            key = k.replace("model.diffusion_model.", "")
            w[key] = f.get_tensor(k)
    return w


def load_f32(path, count=None):
    a = np.fromfile(path, dtype=np.float32)
    if count is not None:
        assert a.size == count, f"{path}: {a.size} != {count}"
    return torch.from_numpy(a.copy())


def capture_file(capture, base):
    """Locate a captured tensor by its canonical name; tolerate the
    XCTest-mangled no-extension form (step00_x_in vs step00_x_in.f32)."""
    cand = os.path.join(capture, base)
    if os.path.isfile(cand):
        return cand
    for f in os.listdir(capture):
        if f == base or f.startswith(base.split(".")[0] + "."):
            return os.path.join(capture, f)
        if f.startswith(base.split(".")[0]) and not f.endswith((".png", ".txt", ".json", ".log")):
            return os.path.join(capture, f)
    return cand


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_trajectory(model, w, capture, out_dir, dtype):
    os.makedirs(out_dir, exist_ok=True)
    sigmas_path = capture_file(capture, "sigmas.txt")
    sigmas = [float(x) for x in open(sigmas_path).read().split(",")]
    assert len(sigmas) == 9, f"expected 9 sigmas, got {len(sigmas)}"
    context = load_f32(capture_file(capture, "cross-context.f32"),
                       CTX_TOKENS * CTX_DIM).view(1, CTX_TOKENS, CTX_DIM)

    rows = []
    for step in range(8):
        x_in = load_f32(capture_file(capture, f"step{step:02d}_x_in.f32"),
                        LATENT_EL).view(1, 16, 1, 64, 64)
        denoised = load_f32(capture_file(capture, f"step{step:02d}_denoised.f32"),
                            LATENT_EL).view(1, 16, 1, 64, 64)
        sigma = sigmas[step]
        swift_v = ((x_in - denoised) / sigma).float()
        with torch.no_grad():
            src_v = model.forward(x_in, sigma, context)
        cos, rmse, maxabs = compare(src_v, swift_v)
        err = (src_v - swift_v)[0, :, 0]          # [16,64,64]
        nyq = nyquist_energy(err)
        nyq_v = nyquist_energy(swift_v[0, :, 0])
        rows.append({"step": step, "sigma": sigma, "cosine": cos,
                     "rmse": rmse, "maxabs": maxabs,
                     "err_nyquist_row": nyq[0], "err_nyquist_col": nyq[1],
                     "err_nyquist_total": nyq[2],
                     "swift_v_nyquist_row": nyq_v[0],
                     "swift_v_nyquist_col": nyq_v[1],
                     "swift_v_nyquist_total": nyq_v[2]})
        print(f"step {step}: cosine={cos:.6f} rmse={rmse:.6f} "
              f"maxabs={maxabs:.6f} err_nyq_row={nyq[0]:.3e} "
              f"col={nyq[1]:.3e} tot={nyq[2]:.3e} "
              f"swift_v_nyq_tot={nyq_v[2]:.3e}", flush=True)
    with open(os.path.join(out_dir, "trajectory-parity.json"), "w") as fh:
        json.dump({"dtype": str(dtype), "sigmas": sigmas, "steps": rows},
                  fh, indent=2)
    print("wrote", os.path.join(out_dir, "trajectory-parity.json"))


def mode_endtoend(model, adapter, fixtures, out_dir, dtype):
    os.makedirs(out_dir, exist_ok=True)
    noise = load_f32(os.path.join(fixtures, "case1_noise.f32"),
                     LATENT_EL).view(1, 16, 1, 64, 64)
    cond = load_f32(os.path.join(fixtures, "case1_cond_context.f32"),
                    46 * CTX_DIM).view(1, 46, CTX_DIM)
    t5 = torch.from_numpy(
        np.fromfile(os.path.join(fixtures, "case1_t5_ids.i64"),
                    dtype=np.int64)).unsqueeze(0)
    t5w = torch.ones(47, dtype=torch.float32)
    ref = load_f32(os.path.join(fixtures, "case1_final_latent.f32"), LATENT_EL)

    sigmas = [1.0, 0.9546938, 0.90035903, 0.8339981, 0.7511211,
              0.64468634, 0.50298506, 0.30500895, 0.0]
    with torch.no_grad():
        context = adapter(cond, t5, t5w)
        x = noise.clone().float()
        for i in range(8):
            sigma = sigmas[i]
            v = model.forward(x, sigma, context)
            denoised = x - sigma * v
            d = (x - denoised) / sigma
            x = x + d * (sigmas[i + 1] - sigma)
    cos, rmse, maxabs = compare(x, ref)
    nyq = nyquist_energy(x[0, :, 0])
    nyq_ref = nyquist_energy(ref.view(1, 16, 1, 64, 64)[0, :, 0])
    result = {"dtype": str(dtype), "final_cosine": cos, "final_rmse": rmse,
              "final_maxabs": maxabs,
              "final_nyquist_row": nyq[0], "final_nyquist_col": nyq[1],
              "final_nyquist_total": nyq[2],
              "ref_nyquist_row": nyq_ref[0], "ref_nyquist_col": nyq_ref[1],
              "ref_nyquist_total": nyq_ref[2]}
    with open(os.path.join(out_dir, "endtoend.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("endtoend:", json.dumps(result, indent=2))
    torch.save(x.float(), os.path.join(out_dir, "source_final_latent.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", required=True, choices=["trajectory", "endtoend"])
    ap.add_argument("--capture", default=None)
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    torch.set_num_threads(4)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    print(f"loading weights from {args.source} ...", flush=True)
    w = load_weights(args.source)
    print(f"loaded {len(w)} tensors", flush=True)
    model = DitModel(w, dtype)
    if args.mode == "trajectory":
        assert args.capture, "--capture required"
        mode_trajectory(model, w, args.capture, args.out, dtype)
    else:
        assert args.fixtures, "--fixtures required"
        adapter = Adapter(w, dtype)
        mode_endtoend(model, adapter, args.fixtures, args.out, dtype)


if __name__ == "__main__":
    main()
