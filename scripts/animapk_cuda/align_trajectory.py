#!/usr/bin/env python3
"""Align harness trajectory vs golden step_latents (pre-step x) and final.

trace_anima.py's denoise_cb captures x BEFORE the Euler update, so
golden step_latents[i] == model input at step i == harness post-step (i-1).
Compare:
  harness step_{i-1} (post) vs golden step_latents[i]
  harness x_7 (step-7 input) vs golden step_latents[7]
  harness final vs golden final_latent
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
    ap.add_argument("--traj", required=True, help="dir with harness stepNN_latent.f32")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    golden_steps = z["step_latents"].astype(np.float32)  # [8,1,16,1,64,64]
    golden_final = z["final_latent"].astype(np.float32).reshape(16, 64, 64)

    print("golden step_latents[i] == harness post-step (i-1):")
    for i in range(8):
        gp = os.path.join(args.traj, f"step{i:02d}_latent.f32")
        if not os.path.isfile(gp):
            print(f"  step{i}: harness file missing")
            continue
        harness_post = np.fromfile(gp, dtype=np.float32).reshape(16, 64, 64)
        # golden step_latents[i] is the model input at step i = harness post (i-1)
        if i == 0:
            g = z["init_noise_randn"].astype(np.float32).reshape(16, 64, 64)
            label = f"step0 input (init_noise) vs harness post-step0"
        else:
            g = golden_steps[i].reshape(16, 64, 64)
            label = f"step{i} input (golden step_latents[{i}]) vs harness post-step{i-1}"
        m = latent_metrics(harness_post, g)
        print(f"  {label}: cos {m['cosine']:.6f} rmse {m['rmse']:.6f}")

    # harness final (post-step7) vs golden final
    hf = np.fromfile(os.path.join(args.traj, "step07_latent.f32"), dtype=np.float32).reshape(16, 64, 64)
    m = latent_metrics(hf, golden_final)
    print(f"harness final (post-step7) vs golden final_latent: cos {m['cosine']:.6f} rmse {m['rmse']:.6f}")
    # golden step_latents[7] vs golden final (D055 anomaly)
    m2 = latent_metrics(golden_steps[7].reshape(16, 64, 64), golden_final)
    print(f"golden step_latents[7] vs golden final_latent: cos {m2['cosine']:.6f} rmse {m2['rmse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
