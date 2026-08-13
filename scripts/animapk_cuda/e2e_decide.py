#!/usr/bin/env python3
"""Decisive end-to-end test: run the FULL 8-step Euler trajectory with the
REAL pinned upstream (fixed SDPA attention) vs the golden final latent.

Also runs the oracle-transcription trajectory for comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from animapk_cuda import upstream as up
import dit_source_oracle as dso
from animapk_cuda.runtime import LadderDiT

FIX = "/workspace/out/fixture"
NPZ = "/workspace/fixtures/case1_danbooru_seed1337.npz"


def cos(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def euler_loop(model_fn, x0, sigmas, context):
    x = x0.clone().to("cuda")
    for i in range(8):
        s = sigmas[i]
        s_next = sigmas[i + 1]
        v = model_fn(x, s, context)
        if v.device != x.device:
            v = v.to(x.device)
        denoised = x - s * v
        x = x + (x - denoised) / s * (s_next - s)
    return x.cpu()


def main():
    device = "cuda"
    z = np.load(NPZ, allow_pickle=True)
    golden = torch.from_numpy(z["final_latent"]).float()
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    x0 = torch.from_numpy(np.fromfile(os.path.join(FIX, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(FIX, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)
    print("golden final norm:", golden.norm().item())

    # A: real upstream (fixed stub, SDPA)
    model, _ = up.load_upstream("animapk_cuda/comfy_stub",
                                "/workspace/source/anima-turbo-v1.0.safetensors",
                                dtype=torch.bfloat16, device=device)
    def a_forward(x, s, c):
        return up.capture_forward(model, x, s, c, device=device)[0]
    with torch.no_grad():
        lat_a = euler_loop(a_forward, x0, sigmas, context)
    print("A (upstream SDPA) final vs golden: cos", cos(lat_a, golden), "rmse", (lat_a - golden).float().norm() / golden.numel() ** 0.5)

    # A2: oracle transcription (manual fp32 attention)
    w = dso.load_weights("/workspace/source/anima-turbo-v1.0.safetensors")
    m2 = LadderDiT(lambda p: w, torch.bfloat16, device)
    def a2_forward(x, s, c):
        return m2.forward(x, s, c, capture=False)[0]
    with torch.no_grad():
        lat_a2 = euler_loop(a2_forward, x0, sigmas, context)
    print("A2 (oracle manual-attn) final vs golden: cos", cos(lat_a2, golden), "rmse", (lat_a2 - golden).float().norm() / golden.numel() ** 0.5)
    print("A vs A2 final: cos", cos(lat_a, lat_a2))


if __name__ == "__main__":
    main()
