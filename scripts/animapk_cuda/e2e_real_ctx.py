#!/usr/bin/env python3
"""Compute the canonical context with the REAL pinned LLMAdapter (bf16 CUDA)
and rerun the 8-step upstream trajectory with it vs the golden final latent.

This closes the context-precision gap in e2e_decide.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from animapk_cuda import upstream as up
from animapk_cuda.e2e_decide import cos, euler_loop

FIX = "/workspace/out/fixture"
NPZ = "/workspace/fixtures/case1_danbooru_seed1337.npz"


def main():
    device = "cuda"
    z = np.load(NPZ, allow_pickle=True)
    golden = torch.from_numpy(z["final_latent"]).float()
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    x0 = torch.from_numpy(np.fromfile(os.path.join(FIX, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    cond = torch.from_numpy(z["cond_context"]).float()  # [1,46,1024]
    t5_ids = torch.from_numpy(z["cond_meta_t5xxl_ids"].astype(np.int64)).unsqueeze(0)  # [1,47]
    t5_w = torch.from_numpy(z["cond_meta_t5xxl_weights"]).float()

    sys.path.insert(0, "animapk_cuda/comfy_stub")
    from comfy.ldm.anima.model import LLMAdapter
    from comfy import ops as _ops
    from safetensors import safe_open

    state = {}
    with safe_open("/workspace/source/anima-turbo-v1.0.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            key = k.replace("model.diffusion_model.", "", 1) if k.startswith("model.diffusion_model.") else k
            if key.startswith("llm_adapter."):
                state[key[len("llm_adapter."):]] = f.get_tensor(k)
    adapter = LLMAdapter(device=device, dtype=torch.bfloat16, operations=_ops)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    print("adapter missing:", len(missing), "unexpected:", len(unexpected))
    adapter.to(torch.bfloat16).to(device).eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        ctx = adapter(cond.to(device, torch.bfloat16), t5_ids.to(device))
        ctx = ctx * t5_w.to(device, torch.bfloat16).unsqueeze(0).unsqueeze(-1)
        if ctx.shape[1] < 512:
            ctx = torch.nn.functional.pad(ctx, (0, 0, 0, 512 - ctx.shape[1]))
    print("real adapter context: shape", tuple(ctx.shape), "norm", ctx.float().norm().item())
    ctx_cpu = ctx.float().cpu()

    prev = torch.from_numpy(np.fromfile(os.path.join(FIX, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)
    print("real-bf16 ctx vs fp32-oracle ctx: cos", cos(ctx_cpu, prev),
          "relL2", (ctx_cpu - prev).float().norm() / prev.float().norm())

    model, _ = up.load_upstream("animapk_cuda/comfy_stub",
                                "/workspace/source/anima-turbo-v1.0.safetensors",
                                dtype=torch.bfloat16, device=device)
    def a_forward(x, s, c):
        return up.capture_forward(model, x, s, c, device=device)[0]
    with torch.no_grad():
        lat_a = euler_loop(a_forward, x0, sigmas, ctx_cpu)
    print("A (real upstream + real-bf16 ctx) vs golden: cos", cos(lat_a, golden))


if __name__ == "__main__":
    main()
