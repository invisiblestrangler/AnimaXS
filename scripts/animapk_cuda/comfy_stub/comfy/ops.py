"""Minimal comfy.ops for predict2.py: manual_cast-style nn operations.

Semantics mirror comfy.ops.manual_cast: weights cast to the input dtype,
compute via torch ops (fp32 accumulation on CUDA for fp16/bf16 GEMMs).
"""

import torch
from torch import nn
import torch.nn.functional as F


class Embedding(nn.Embedding):
    """manual_cast-style embedding: output in out_dtype (default input dtype)."""

    def forward(self, x, out_dtype=None):
        dt = out_dtype if out_dtype is not None else x.dtype
        return F.embedding(x, self.weight.to(dt))


class Linear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight.to(x.dtype), eps=self.eps)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype) if self.weight is not None else None,
                            self.bias.to(x.dtype) if self.bias is not None else None, self.eps)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                          is_causal=is_causal, scale=scale)


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False,
                     compute_dtype=None, want_requant=False):
    """Non-quantized path: return the norm weight in the input dtype."""
    if hasattr(s, "weight") and input is not None:
        w = s.weight.to(input.dtype)
        return w, None, None
    return None, None, None


def uncast_bias_weight(s, scale=None, bias=None, stream=None):
    return None
