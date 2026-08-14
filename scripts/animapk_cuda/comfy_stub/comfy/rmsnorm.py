"""Minimal comfy.rmsnorm stub (pinned upstream uses F.rms_norm)."""

import torch
import torch.nn.functional as F

RMSNorm = torch.nn.RMSNorm


def rms_norm(x, weight=None, eps=1e-6):
    if weight is None:
        return F.rms_norm(x, (x.shape[-1],), eps=eps)
    return F.rms_norm(x, weight.shape, weight=weight.to(x.dtype), eps=eps)
