#!/usr/bin/env python3
"""Decoder-sensitivity experiment (MD section 24).

delta = bad_latent - golden_latent. Decode golden + k*delta for k in
{0.25, 0.5, 0.75, 1.0} plus a norm-matched random perturbation, through the
SAME VAE. Measure 8px carrier + RGB metrics for each.

Interpretation:
- carrier growing with the real structured delta => grid is latent-encoded
- equal-norm random perturbation producing the same grid => decoder sensitivity
- sharp threshold => explains high cosine but pathological image
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from animapk_cuda.grid_repro import (  # noqa: E402
    load_decoder_tensors_torch, decode_latent_torch, rgb_f32_to_png,
    latent_metrics, rgb_metrics, sha256_bytes,
)
from measure_grid_carrier import carrier_scores, load_rgb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-pack", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bad-latent", required=True, help="G1_bf16_final_latent.f32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    golden = z["final_latent"].astype(np.float32).reshape(16, 64, 64)
    ref_rgb = z["decoded_rgb"][0, :, 0, :, :].astype(np.float32)
    bad = np.fromfile(args.bad_latent, dtype=np.float32).reshape(16, 64, 64)
    delta = bad.astype(np.float64) - golden.astype(np.float64)
    delta_norm = np.linalg.norm(delta)

    print("loading VAE...")
    vae_t = load_decoder_tensors_torch(args.vae_pack)

    # reference PNG via canonical conversion
    ref_png = os.path.join(args.out, "reference.png")
    rgb_f32_to_png(ref_rgb, ref_png)

    results = {}
    rng = np.random.default_rng(args.seed)

    def run(name, latent):
        rgb = decode_latent_torch(np.ascontiguousarray(latent, np.float32), vae_t)
        png = os.path.join(args.out, f"{name}.png")
        rgb_f32_to_png(rgb, png)
        gen = carrier_scores(load_rgb(png))
        ref = carrier_scores(load_rgb(ref_png))
        ratio = gen["total"] / ref["total"] if ref["total"] > 0 else float("inf")
        m = {
            "latent_cos_vs_golden": latent_metrics(latent, golden)["cosine"],
            "carrier_total": gen["total"],
            "carrier_h": gen["horizontal"],
            "carrier_v": gen["vertical"],
            "carrier_ratio": ratio,
            "rgb_vs_ref_cosine": rgb_metrics(rgb, ref_rgb)["cosine"],
            "rgb_vs_ref_rmse": rgb_metrics(rgb, ref_rgb)["rmse"],
        }
        results[name] = m
        print(f"{name}: carrier {gen['total']:.6f} ratio {ratio:.1f}x "
              f"latent_cos {m['latent_cos_vs_golden']:.4f} rgb_cos {m['rgb_vs_ref_cosine']:.4f}")
        return rgb

    # baseline: golden alone
    run("k0_golden", golden)
    # golden + k*delta
    for k in (0.25, 0.5, 0.75, 1.0):
        run(f"k{k:g}", golden + k * delta)
    # norm-matched random perturbation at k=1.0 and k=0.5
    rnd = rng.standard_normal(golden.shape).astype(np.float64)
    rnd *= delta_norm / (np.linalg.norm(rnd) + 1e-30)
    run("rand_k1", golden + rnd)
    run("rand_k0.5", golden + 0.5 * rnd)

    with open(os.path.join(args.out, "decoder_sensitivity.json"), "w") as fh:
        json.dump({"delta_norm": float(delta_norm), "results": results}, fh, indent=2)
    print("done ->", os.path.join(args.out, "decoder_sensitivity.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
