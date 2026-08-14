#!/usr/bin/env python3
"""Fine-grained A (actual upstream) vs A2 (oracle) stage diagnostic.

Hooks the REAL pinned MiniTrainDIT at the same boundaries the oracle
captures, then compares stage by stage to localize the first divergence.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from animapk_cuda import upstream as up
from animapk_cuda import compare as cmp

FIX = "/workspace/out/fixture"
A2 = "/workspace/out/ladder/caps_A2_oracle_bf16.npz"


def main():
    device = "cuda"
    model, _ = up.load_upstream("animapk_cuda/comfy_stub",
                                "/workspace/source/anima-turbo-v1.0.safetensors",
                                dtype=torch.bfloat16, device=device)
    x_in = torch.from_numpy(np.fromfile(os.path.join(FIX, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(FIX, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)

    caps = {}
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            if isinstance(out, tuple):
                # TimestepEmbedding returns (emb_B_T_D, adaln_lora_B_T_3D)
                out = out[1] if name == "adaln_lora" else out[0]
            caps[name] = out.detach().float().cpu()
        return fn

    def prehook(name):
        def fn(mod, inp):
            caps[name] = inp[0].detach().float().cpu()
        return fn

    handles.append(model.x_embedder.register_forward_hook(hook("x_embedder_out")))
    handles.append(model.t_embedding_norm.register_forward_hook(hook("timestep_emb")))
    handles.append(model.t_embedder[1].register_forward_hook(hook("adaln_lora")))
    handles.append(model.blocks[0].register_forward_pre_hook(prehook("block00_in")))
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(hook(f"block{i:02d}_out")))
    handles.append(model.final_layer.register_forward_pre_hook(prehook("pre_final_norm")))
    handles.append(model.final_layer.register_forward_hook(hook("post_final_projection_patched")))

    x = x_in.to(device=device, dtype=model.dtype)
    c = context.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        vel = model.forward(x, torch.tensor([1.0], dtype=model.dtype, device=device), c)
    caps["post_unpatchify_velocity"] = vel.detach().float().cpu()
    for h in handles:
        h.remove()

    a2 = {k: torch.from_numpy(v) for k, v in np.load(A2).items()}

    stages = ["x_embedder_out", "timestep_emb", "adaln_lora", "block00_in"] + \
             [f"block{i:02d}_out" for i in range(28)] + \
             ["pre_final_norm", "post_final_projection_patched", "post_unpatchify_velocity"]

    print(f"{'stage':>32} {'cosine':>10} {'rmse':>12} {'relL2':>10} {'a_norm':>12} {'b_norm':>12}")
    for s in stages:
        if s not in caps or s not in a2:
            print(f"{s:>32} MISSING")
            continue
        a, b = caps[s].reshape(-1).float(), a2[s].reshape(-1).float()
        m = cmp.metrics(a.view(1, -1), b.view(1, -1))
        print(f"{s:>32} {m['cosine']:10.6f} {m['rmse']:12.4f} {m['rel_l2']:10.4f} {m['a_norm']:12.1f} {m['b_norm']:12.1f}")


if __name__ == "__main__":
    main()
