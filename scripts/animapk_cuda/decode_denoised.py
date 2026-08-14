#!/usr/bin/env python3
"""Decode harness per-step DENOISED (not x) to see where the grid enters.

If the model's denoised output at step 7 is clean but the Euler-accumulated x
grids, the grid is an accumulation artifact of the Euler step, not the model.
If the denoised itself grids, the model output carries it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from animapk_cuda.grid_repro import (  # noqa: E402
    load_decoder_tensors_torch, decode_latent_torch, rgb_f32_to_png, latent_metrics, rgb_metrics,
)
from measure_grid_carrier import carrier_scores, load_rgb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-pack", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    final = z["final_latent"].astype(np.float32).reshape(16, 64, 64)
    ref_rgb = z["decoded_rgb"][0, :, 0, :, :].astype(np.float32)
    ref_png = os.path.join(args.out, "reference.png")
    rgb_f32_to_png(ref_rgb, ref_png)
    vae_t = load_decoder_tensors_torch(args.vae_pack)

    # re-run the 8-step Euler, capturing per-step DENOISED
    from animapk_cuda import ladder_real
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    x0 = torch.from_numpy(np.fromfile("/workspace/out/fixture/x_in.f32",
                                      dtype=np.float32)).view(1, 16, 1, 64, 64).to(device)
    ctx = torch.from_numpy(np.fromfile("/workspace/out/fixture/context512.f32",
                                       dtype=np.float32)).view(1, 512, 1024).to(device)
    w = ladder_real.load_src_weights("/workspace/source/anima-turbo-v1.0.safetensors")
    model = ladder_real.build_real(w, torch.bfloat16, device)
    model.eval()

    x = x0
    with torch.no_grad():
        for i in range(8):
            s, s_next = sigmas[i], sigmas[i + 1]
            v = model(x, torch.tensor([s], dtype=x.dtype, device=device), ctx)
            denoised = x - s * v
            if i in (0, 3, 6, 7):
                d = denoised.float().cpu().numpy().reshape(16, 64, 64)
                name = f"denoised_step{i}"
                rgb = decode_latent_torch(np.ascontiguousarray(d, np.float32), vae_t)
                png = os.path.join(args.out, f"{name}.png")
                rgb_f32_to_png(rgb, png)
                gen = carrier_scores(load_rgb(png))
                ref = carrier_scores(load_rgb(ref_png))
                ratio = gen["total"] / ref["total"] if ref["total"] > 0 else float("inf")
                print(f"{name}: carrier {gen['total']:.6f} ratio {ratio:.1f}x "
                      f"latent_cos_vs_final {latent_metrics(d, final)['cosine']:.4f} "
                      f"rgb_cos {rgb_metrics(rgb, ref_rgb)['cosine']:.4f}")
            x = x + (x - denoised) / s * (s_next - s)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
