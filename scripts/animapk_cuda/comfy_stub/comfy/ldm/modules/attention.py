"""SDPA optimized_attention (pinned comfy attention_pytorch path, CUDA)."""

import torch
import torch.nn.functional as F


def optimized_attention(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kwargs):
    """q/k/v [B,H,S,D] when skip_reshape=True (predict2 torch_attention_op)."""
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        raise NotImplementedError("stub supports skip_reshape=True only")
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out
