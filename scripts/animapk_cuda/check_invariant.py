#!/usr/bin/env python3
"""Check the Euler invariant on the harness trajectory.

final post-step latent should equal final denoised (D055 invariant, since
sigma_next=0). Also compare harness denoised[i] against golden step_latents[i]
(which are actually denoised captures, not x captures).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from animapk_cuda.grid_repro import latent_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--traj", required=True, help="dir with harness stepNN_latent.f32 (post-step)")
    ap.add_argument("--denoised", required=True, help="dir with harness denoised_stepN_rgb only (png); recompute if missing")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    golden_steps = z["step_latents"].astype(np.float32)  # [8,1,16,1,64,64]
    golden_final = z["final_latent"].astype(np.float32).reshape(16, 64, 64)

    # harness post-step latents
    post = []
    for i in range(8):
        p = os.path.join(args.traj, f"step{i:02d}_latent.f32")
        post.append(np.fromfile(p, dtype=np.float32).reshape(16, 64, 64))

    # invariant: post[7] == denoised[7]. Recompute denoised via Euler from post[6]
    # denoised_7 = x_7 - sigma_7 * v_7; but we don't have v. Instead use the
    # relation: post[7] = post[6] + (post[6]-denoised7)/s7 * (0 - s7) => denoised7 = post[7]
    m = latent_metrics(post[7], golden_final)
    print(f"harness post[7] vs golden final_latent: cos {m['cosine']:.6f} rmse {m['rmse']:.6f}")

    # golden step_latents[i] are DENOISED captures. Compare against harness
    # denoised[i] reconstructed from post-step states:
    # denoised_i = x_i - sigma_i * (x_i - x_{i+1})/(sigma_i - sigma_{i+1})
    # From Euler: x_{i+1} = x_i + (x_i - denoised_i)/sigma_i * (sigma_{i+1} - sigma_i)
    # => (x_i - denoised_i) = (x_{i+1} - x_i) * sigma_i / (sigma_{i+1} - sigma_i)
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    print("\nharness-reconstructed denoised[i] vs golden step_latents[i]:")
    for i in range(8):
        x_i, x_ip1 = post[i], post[i + 1] if i < 7 else post[7]
        # for i=7, x_{8} doesn't exist; skip reconstruction (post[7]==denoised7 by invariant)
        if i == 7:
            d_i = post[7]
        else:
            d_i = x_i - (x_ip1 - x_i) * sigmas[i] / (sigmas[i + 1] - sigmas[i])
        g = golden_steps[i].reshape(16, 64, 64)
        m = latent_metrics(d_i, g)
        print(f"  step{i}: cos {m['cosine']:.6f} rmse {m['rmse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
