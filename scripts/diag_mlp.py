#!/usr/bin/env python3
"""Diagnostic 2: isolate the MLP branch — is the MLP delta correct?"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dit_block0_oracle import (load_f32, load_vec, layer_norm_modulated, rms_rope_split_half,
                               attention, split_heads, concat_heads, gelu_fast,
                               TOKENS, DIM, HEADS, HEAD_DIM, CTX, CTX_TOKENS, LORA, MLP_HID)

D = "/root/AnimaXS/scripts/oracle_out/block0/"
x = load_f32(D + "block0_input_x.f32", (TOKENS, DIM))
emb = load_vec(D + "block0_emb.f32", DIM)
adaln = load_vec(D + "block0_adaln_lora.f32", 3 * DIM)
ctx = load_f32(D + "block0_cross_ctx.f32", (CTX_TOKENS, CTX))
rope = load_f32(D + "block0_rope.f32", (TOKENS, HEAD_DIM // 2, 2, 2))

def w(name, shape): return load_f32(D + f"block0_{name}.f32", shape)
def vec(name, n): return load_vec(D + f"block0_{name}.f32", n)

mod_self_w1 = w("mod_self_w1", (LORA, DIM)); mod_self_w2 = w("mod_self_w2", (3 * DIM, LORA))
mod_cross_w1 = w("mod_cross_w1", (LORA, DIM)); mod_cross_w2 = w("mod_cross_w2", (3 * DIM, LORA))
mod_mlp_w1 = w("mod_mlp_w1", (LORA, DIM)); mod_mlp_w2 = w("mod_mlp_w2", (3 * DIM, LORA))
self_q = w("self_q", (DIM, DIM)); self_k = w("self_k", (DIM, DIM))
self_v = w("self_v", (DIM, DIM)); self_o = w("self_o", (DIM, DIM))
self_qn = vec("self_q_norm", HEAD_DIM); self_kn = vec("self_k_norm", HEAD_DIM)
cross_q = w("cross_q", (DIM, DIM)); cross_k = w("cross_k", (DIM, CTX))
cross_v = w("cross_v", (DIM, CTX)); cross_o = w("cross_o", (DIM, DIM))
cross_qn = vec("cross_q_norm", HEAD_DIM); cross_kn = vec("cross_k_norm", HEAD_DIM)
mlp_w1 = w("mlp_w1", (MLP_HID, DIM)); mlp_w2 = w("mlp_w2", (DIM, MLP_HID))

def modulate(w1, w2):
    silu = emb * (1.0 / (1.0 + np.exp(-emb)))
    h1 = silu @ w1.T
    mod = (h1 @ w2.T) + adaln
    return tuple(np.split(mod, 3))
self_shift, self_scale, self_gate = modulate(mod_self_w1, mod_self_w2)
cross_shift, cross_scale, cross_gate = modulate(mod_cross_w1, mod_cross_w2)
mlp_shift, mlp_scale, mlp_gate = modulate(mod_mlp_w1, mod_mlp_w2)

def cosine(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
def maxabs(a, b): return np.abs(a - b).max()

g = np.load("/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz")["block_00_out"].reshape(TOKENS, DIM)

# --- self branch ---
self_norm = layer_norm_modulated(x, self_scale, self_shift)
q = self_norm @ self_q.T; k = self_norm @ self_k.T; v = self_norm @ self_v.T
q = split_heads(q, TOKENS, HEADS, HEAD_DIM); k = split_heads(k, TOKENS, HEADS, HEAD_DIM); v = split_heads(v, TOKENS, HEADS, HEAD_DIM)
q, k = rms_rope_split_half(q, k, rope, self_qn, self_kn)
self_out = concat_heads(attention(q, k, v)) @ self_o.T
x1 = x + self_gate * self_out

# --- cross branch ---
cross_norm = layer_norm_modulated(x1, cross_scale, cross_shift)
q = cross_norm @ cross_q.T; k = ctx @ cross_k.T; v = ctx @ cross_v.T
q = split_heads(q, TOKENS, HEADS, HEAD_DIM); k = split_heads(k, CTX_TOKENS, HEADS, HEAD_DIM); v = split_heads(v, CTX_TOKENS, HEADS, HEAD_DIM)
q = q / np.sqrt((q**2).mean(-1, keepdims=True) + 1e-6) * cross_qn
k = k / np.sqrt((k**2).mean(-1, keepdims=True) + 1e-6) * cross_kn
cross_out = concat_heads(attention(q, k, v)) @ cross_o.T
x2 = x1 + cross_gate * cross_out

# --- MLP branch ---
mlp_norm = layer_norm_modulated(x2, mlp_scale, mlp_shift)
h = mlp_norm @ mlp_w1.T
h_gelu = gelu_fast(h)
y = h_gelu @ mlp_w2.T
mlp_delta = mlp_gate * y
x3 = x2 + mlp_delta

print("=== MLP isolation ===")
print("my x3 - my x2   (my MLP delta)  norm=%.4f" % np.linalg.norm(mlp_delta))
print("golden - my x2  (implied golden MLP delta)  norm=%.4f" % np.linalg.norm(g - x2))
print("cosine(my mlp_delta, golden_implied_mlp_delta) = %.6f" % cosine(mlp_delta, g - x2))
print("maxabs = %.4f" % maxabs(mlp_delta, g - x2))

# Also: try WITHOUT the LayerNorm+AdaLN before MLP (just x2 directly)
mlp_norm2 = layer_norm_modulated(x2, mlp_scale, mlp_shift)
h2 = mlp_norm2 @ mlp_w1.T
y2 = gelu_fast(h2) @ mlp_w2.T
print("cosine(my mlp_delta, golden_implied) using gelu_fast: %.6f" % cosine(mlp_gate*y2, g - x2))

# Sanity: what if MLP gate orientation is wrong (gate applied before norm etc)? Just print stats
print("mlp_gate min/max: %.4f/%.4f" % (mlp_gate.min(), mlp_gate.max()))
print("mlp_scale min/max: %.4f/%.4f" % (mlp_scale.min(), mlp_scale.max()))
print("y min/max: %.4f/%.4f  h min/max: %.4f/%.4f" % (y.min(), y.max(), h.min(), h.max()))

# Check golden x3 vs my x2 directly — if golden x3 ≈ my x2 + small, then golden has NO mlp contribution?
print("golden vs my x2: cosine=%.6f maxAbs=%.4f" % (cosine(g, x2), maxabs(g, x2)))
