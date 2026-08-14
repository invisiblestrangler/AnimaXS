"""Stub of comfy.quant_ops for the non-quantized path.

ck.rms_rope_split_half is the plain fallback: RMS norm (fp32 statistics) then
split-half RoPE rotation, matching the comfy-kitchen kernel's non-quantized
behavior (q_scale/k_scale = norm weights in input dtype).
"""

import torch


class _Ck:
    @staticmethod
    def rms_rope_split_half(q, k, rope_emb, q_scale=None, k_scale=None, eps=1e-6):
        def rms(x, w):
            xf = x.float()
            v = xf.pow(2).mean(dim=-1, keepdim=True)
            return (xf * torch.rsqrt(v + eps) * w.float()).to(x.dtype)

        if q_scale is not None:
            q = rms(q, q_scale)
        if k_scale is not None:
            k = rms(k, k_scale)
        # rope_emb arrives in one of several shapes ([L,64,2,2] from the
        # pos_embedder, or with leading singleton dims after the _forward's
        # unsqueeze chain: [1,L,1,1,64,2,2] / [1,L,1,64,2,2]).
        rope = rope_emb.squeeze()  # -> [L, 64, 2, 2]
        cos = rope[..., 0, 0].unsqueeze(0).unsqueeze(2)  # [1,S,1,64]
        sin = rope[..., 1, 0].unsqueeze(0).unsqueeze(2)
        for t in (q, k):
            # IMPORTANT: clone halves — t[..., :64] is a VIEW; writing it first
            # would alias the second-half computation (rotate uses original a).
            a = t[..., :64].clone()
            b = t[..., 64:].clone()
            t[..., :64] = cos * a - sin * b
            t[..., 64:] = sin * a + cos * b
        return q, k


ck = _Ck()
