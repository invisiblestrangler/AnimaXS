#!/usr/bin/env python3
"""Check whether the input noise (x0) already carries the 8px grid pattern.

Decode x0 through the VAE and measure carrier. If x0 decodes to a grid, the
defect is in the input noise, not the DiT. Also check the golden's own
step_latents[0] (the first mislabeled capture) and the harness step-0 x.
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
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    final = z["final_latent"].astype(np.float32).reshape(16, 64, 64)
    ref_rgb = z["decoded_rgb"][0, :, 0, :, :].astype(np.float32)
    ref_png = os.path.join(args.out, "reference.png")
    rgb_f32_to_png(ref_rgb, ref_png)
    vae_t = load_decoder_tensors_torch(args.vae_pack)

    x0 = np.fromfile(os.path.join(args.fixture, "x_in.f32"), dtype=np.float32).reshape(16, 64, 64)
    noise = z["init_noise_randn"].astype(np.float32).reshape(16, 64, 64)
    golden_step0 = z["step_latents"][0].astype(np.float32).reshape(16, 64, 64)

    print("x0 == init_noise_randn?", np.array_equal(x0, noise),
          "maxdiff", np.abs(x0 - noise).max())

    for name, lat in (("x0_input_noise", x0), ("golden_step0", golden_step0), ("golden_final", final)):
        rgb = decode_latent_torch(np.ascontiguousarray(lat, np.float32), vae_t)
        png = os.path.join(args.out, f"{name}.png")
        rgb_f32_to_png(rgb, png)
        gen = carrier_scores(load_rgb(png))
        ref = carrier_scores(load_rgb(ref_png))
        ratio = gen["total"] / ref["total"] if ref["total"] > 0 else float("inf")
        print(f"{name}: carrier {gen['total']:.6f} ratio {ratio:.1f}x "
              f"rgb_cos {rgb_metrics(rgb, ref_rgb)['cosine']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
