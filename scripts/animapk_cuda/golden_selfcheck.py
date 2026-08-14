#!/usr/bin/env python3
"""Resolve the golden self-inconsistency: step_latents[7] vs final_latent.

D055 says golden step_latents is internally inconsistent with final_latent.
This decodes BOTH golden step_latents[7] and golden final_latent through the
same VAE, measures the 8px carrier of each, and reports the cosine between
the two golden latents. Whichever decodes clean is the true reference; if
step_latents[7] decodes with the grid, the golden trace itself carries it.
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
    load_decoder_tensors_torch, decode_latent_torch, rgb_f32_to_png,
    latent_metrics, rgb_metrics,
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
    step7 = z["step_latents"][7].astype(np.float32).reshape(16, 64, 64)
    final = z["final_latent"].astype(np.float32).reshape(16, 64, 64)
    ref_rgb = z["decoded_rgb"][0, :, 0, :, :].astype(np.float32)

    m = latent_metrics(step7, final)
    print(f"golden step_latents[7] vs golden final_latent: cos {m['cosine']:.6f} "
          f"rmse {m['rmse']:.6f} rel_l2 {m['rel_l2']:.6f}")

    ref_png = os.path.join(args.out, "reference.png")
    rgb_f32_to_png(ref_rgb, ref_png)

    vae_t = load_decoder_tensors_torch(args.vae_pack)
    for name, lat in (("golden_step7", step7), ("golden_final", final)):
        rgb = decode_latent_torch(np.ascontiguousarray(lat, np.float32), vae_t)
        png = os.path.join(args.out, f"{name}.png")
        rgb_f32_to_png(rgb, png)
        gen = carrier_scores(load_rgb(png))
        ref = carrier_scores(load_rgb(ref_png))
        ratio = gen["total"] / ref["total"] if ref["total"] > 0 else float("inf")
        print(f"{name}: carrier {gen['total']:.6f} ratio {ratio:.1f}x "
              f"rgb_cos {rgb_metrics(rgb, ref_rgb)['cosine']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
