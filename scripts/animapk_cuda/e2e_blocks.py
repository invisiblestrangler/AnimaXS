#!/usr/bin/env python3
"""Run the 8-step upstream trajectory capturing per-block outputs at the LAST
step, compare vs golden block_00_out..block_27_out."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from animapk_cuda import upstream as up

NPZ = "/workspace/fixtures/case1_danbooru_seed1337.npz"
FIX = "/workspace/out/fixture"


def cos(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def main():
    device = "cuda"
    z = np.load(NPZ, allow_pickle=True)
    golden_blocks = {k: torch.from_numpy(z[k]).float() for k in z.files if k.startswith("block_")}
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    x0 = torch.from_numpy(np.fromfile(os.path.join(FIX, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(FIX, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)

    model, _ = up.load_upstream("animapk_cuda/comfy_stub",
                                "/workspace/source/anima-turbo-v1.0.safetensors",
                                dtype=torch.bfloat16, device=device)
    x = x0.to(device, torch.bfloat16)
    c = context.to(device, torch.bfloat16)
    last_blocks = {}
    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(
            lambda m, i, o, idx=i: last_blocks.__setitem__(idx, o.detach().float().cpu())))
    with torch.no_grad():
        for step in range(8):
            s = sigmas[step]
            s_next = sigmas[step + 1]
            v = up.capture_forward(model, x.float().cpu(), s, context, device=device)[0]
            v = v.to(device, torch.bfloat16)
            denoised = x - s * v
            x = x + (x - denoised) / s * (s_next - s)
    for h in handles:
        h.remove()

    print(f"{'block':>10} {'A_step7_vs_golden':>16}")
    for i in range(28):
        gk = f"block_{i:02d}_out"
        if gk in golden_blocks:
            print(f"{gk:>10} {cos(last_blocks[i].reshape(-1), golden_blocks[gk].reshape(-1)):16.6f}")


if __name__ == "__main__":
    main()
