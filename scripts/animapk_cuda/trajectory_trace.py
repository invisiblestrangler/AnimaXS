#!/usr/bin/env python3
"""Per-step latent trajectory comparison: harness real-graph vs golden step_latents.

Runs the full 8-step Euler (as grid_repro G1) but records x after every step,
and compares each against golden step_latents[i] (the ComfyUI-captured
per-step latents). Also computes the step-0 block-00 cosine under the harness
convention (raw sigma, ctx512) to reproduce the handoff's "0.98-0.999 at step7"
claim and isolate where divergence begins.

If per-step latents agree well at early steps and diverge later, the error is
accumulation. If they diverge from step 0, it's a single-step systematic diff.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from animapk_cuda import ladder_real  # noqa: E402
from animapk_cuda import upstream as up  # noqa: E402


def cos(a, b):
    a = a.detach().float().cpu().reshape(-1).double()
    b = b.detach().float().cpu().reshape(-1).double()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.npz, allow_pickle=True)
    golden_steps = torch.from_numpy(z["step_latents"].astype(np.float32))  # [8,1,16,1,64,64]
    sigmas = [float(s) for s in z["sigmas_comfy"]]

    x0 = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "x_in.f32"),
                                      dtype=np.float32)).view(1, 16, 1, 64, 64).to(device)
    ctx = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "context512.f32"),
                                       dtype=np.float32)).view(1, 512, 1024).to(device)

    print("building real graph (bf16)...")
    w = ladder_real.load_src_weights(args.source)
    model = ladder_real.build_real(w, torch.bfloat16, device)
    model.eval()

    x = x0
    per_step = []
    with torch.no_grad():
        for i in range(8):
            s, s_next = sigmas[i], sigmas[i + 1]
            v = model(x, torch.tensor([s], dtype=x.dtype, device=device), ctx)
            denoised = x - s * v
            x = x + (x - denoised) / s * (s_next - s)
            per_step.append(x.float().cpu())
            g = golden_steps[i]
            c = cos(x, g)
            rmse = float((x.float().cpu() - g).pow(2).mean().sqrt())
            print(f"step {i}: harness latent cos vs golden {c:.6f} rmse {rmse:.4f}")

    # Save per-step latents
    for i, t in enumerate(per_step):
        t.reshape(16, 64, 64).numpy().astype(np.float32).tofile(
            os.path.join(args.out, f"step{i:02d}_latent.f32"))
    print("wrote per-step latents ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
