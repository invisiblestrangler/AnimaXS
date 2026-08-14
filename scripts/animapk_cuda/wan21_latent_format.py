#!/usr/bin/env python3
"""Anima/Wan21 model latent-format boundary (sampler space <-> VAE space).

Root cause (2026-08-14, proven by scripts/animapk_cuda/wan21_boundary_proof.py):
ComfyUI applies `latent_format.process_out` to the sampler return BEFORE the
workflow receives the latent (comfy/samplers.py CFGGuider.inner_sample:
`return self.inner_model.process_latent_out(samples.to(torch.float32))`).
Anima is registered with `latent_format = Wan21` (comfy/supported_models.py).

So:
    sampler_latent  (raw DiT/Euler output)      -- ComfyUI process_out -->  vae_latent
    vae_latent      (what the VAE decoder eats) -- ComfyUI process_in  -->  sampler_latent

The custom CUDA/Metal runtime previously decoded sampler-space latents as if
they were VAE-space, which produced the 8px grid. Apply `anima_sampler_to_vae`
EXACTLY ONCE at the sampler->VAE boundary, never inside the VAE itself.

Constants are read VERBATIM from the project's pinned comfy-ref snapshot
(MiniTrainDIT/comfy cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44, checked out at
/workspace/comfy-ref on Clore as a file snapshot). DO NOT replace them with
current ComfyUI master values — the pinned Wan21 constants differ.

Pinned comfy-ref file SHAs (recorded 2026-08-14):
  supported_models.py a97d56c8efeb9aea...
  latent_formats.py   f13259cbd815a339...
  samplers.py         69498999dd1198c0...
  model_base.py       39022ddcbb8a1683...
"""
from __future__ import annotations

import numpy as np

SCALE_FACTOR = 1.0

LATENTS_MEAN = np.array([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
], dtype=np.float32)

LATENTS_STD = np.array([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
], dtype=np.float32)


def _affine_shapes(x: np.ndarray) -> tuple:
    """Return broadcastable (mean, std) for [16,64,64] or [1,16,1,64,64] x."""
    if x.ndim == 5:  # [B,C,T,H,W]
        return (LATENTS_MEAN.reshape(1, 16, 1, 1, 1),
                LATENTS_STD.reshape(1, 16, 1, 1, 1))
    if x.ndim == 4:  # [B,C,H,W]
        return (LATENTS_MEAN.reshape(1, 16, 1, 1),
                LATENTS_STD.reshape(1, 16, 1, 1))
    if x.ndim == 3:  # [C,H,W]
        return (LATENTS_MEAN.reshape(16, 1, 1),
                LATENTS_STD.reshape(16, 1, 1))
    raise ValueError(f"unsupported latent ndim {x.ndim}, shape {x.shape}")


def anima_sampler_to_vae(x: np.ndarray) -> np.ndarray:
    """Wan21.process_out: sampler-space -> VAE-space.

    vae_latent = sampler_latent * latents_std / scale_factor + latents_mean
    (scale_factor == 1.0 for the pinned Wan21).
    """
    mean, std = _affine_shapes(x)
    return (x.astype(np.float64) * std / SCALE_FACTOR + mean).astype(np.float32)


def anima_vae_to_sampler(x: np.ndarray) -> np.ndarray:
    """Wan21.process_in: VAE-space -> sampler-space (exact inverse)."""
    mean, std = _affine_shapes(x)
    return ((x.astype(np.float64) - mean) * SCALE_FACTOR / std).astype(np.float32)


def anima_sampler_to_vae_torch(x):
    """Torch variant of anima_sampler_to_vae (device/dtype-preserving)."""
    import torch
    c = x.shape[-3] if x.dim() == 5 else (x.shape[1] if x.dim() == 4 else x.shape[0])
    mean = torch.tensor(LATENTS_MEAN, dtype=x.dtype, device=x.device).reshape(
        [1, c, 1, 1, 1] if x.dim() == 5 else [1, c, 1, 1] if x.dim() == 4 else [c, 1, 1])
    std = torch.tensor(LATENTS_STD, dtype=x.dtype, device=x.device).reshape(
        [1, c, 1, 1, 1] if x.dim() == 5 else [1, c, 1, 1] if x.dim() == 4 else [c, 1, 1])
    return x * std / SCALE_FACTOR + mean
