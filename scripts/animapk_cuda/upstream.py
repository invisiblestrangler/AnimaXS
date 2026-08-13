"""Variant A: run the ACTUAL pinned upstream graph (ComfyUI MiniTrainDIT at
cbbc9da) on CUDA with the official BF16 safetensors, capturing the same
per-block and final-path boundaries as the oracle and pack runtimes.

Loads the pinned comfy package from --comfy (path to the comfy repo checkout).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open


def load_upstream(comfy_dir: str, source: str, dtype=torch.bfloat16, device="cuda"):
    sys.path.insert(0, comfy_dir)
    from comfy.ldm.cosmos.predict2 import MiniTrainDIT  # noqa: E402

    model = MiniTrainDIT(
        max_img_h=64,
        max_img_w=64,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        model_channels=2048,
        num_blocks=28,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        min_fps=1,
        max_fps=30,
        use_adaln_lora=True,
        adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False,
    )
    state = {}
    with safe_open(source, framework="pt", device="cpu") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        # x_embedder etc. all map; anything unexpected is a real mismatch
        print("UNEXPECTED STATE KEYS:", unexpected[:20], file=sys.stderr)
    model.to(dtype=dtype, device=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, missing


def capture_forward(model, x, sigma, context, device="cuda"):
    """One step-0 forward with block/final hooks.  x/context fp32 CPU in,
    captures returned as fp32 CPU tensors."""
    caps = {}
    handles = []

    def make_hook(name, kind):
        def hook(mod, inp, out):
            caps[f"{name}"] = out.detach().float().cpu()

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(make_hook(f"block{i:02d}_out", "out")))
    handles.append(model.final_layer.register_forward_hook(make_hook("final_layer_out", "out")))

    x = x.to(device=device, dtype=model.dtype)
    context = context.to(device=device, dtype=model.dtype)
    sigma_t = torch.tensor([sigma], dtype=model.dtype, device=device)
    with torch.no_grad():
        vel = model.forward(x, sigma_t, context)
    for h in handles:
        h.remove()
    return vel.detach().float().cpu(), caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="pinned comfy checkout root (repo dir)")
    ap.add_argument("--source", required=True, help="official anima-turbo-v1.0.safetensors")
    ap.add_argument("--fixture", required=True, help="prepare_fixture.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, missing = load_upstream(args.comfy, args.source, dtype=dtype, device=device)

    x_in = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)
    with open(os.path.join(args.fixture, "sigmas.txt")) as f:
        sigmas = [float(s) for s in f.read().strip().split(",")]

    vel, caps = capture_forward(model, x_in, sigmas[0], context, device=device)

    os.makedirs(os.path.join(args.out, "captures"), exist_ok=True)
    for k, v in caps.items():
        v.contiguous().numpy().tofile(os.path.join(args.out, "captures", f"{k}.f32"))
    vel.contiguous().numpy().tofile(os.path.join(args.out, "captures", "velocity.f32"))
    report = {
        "variant": f"A-{args.dtype}",
        "graph": "comfy MiniTrainDIT cbbc9da",
        "source": args.source,
        "dtype": str(dtype),
        "device": device,
        "sigma": sigmas[0],
        "velocity": {"shape": list(vel.shape), "sha256": __import__("hashlib").sha256(vel.numpy().tobytes()).hexdigest()},
        "captures": sorted(caps.keys()),
        "missing_state_keys": missing,
    }
    with open(os.path.join(args.out, "upstream_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({"variant": report["variant"], "velocity_sha": report["velocity"]["sha256"], "captures": len(caps)}, indent=2))


if __name__ == "__main__":
    main()
