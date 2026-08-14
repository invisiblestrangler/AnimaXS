#!/usr/bin/env python3
"""Timestep + context convention probe (step-0) vs golden block_00_out.

The golden npz captured block_00_out at step 0. ComfyUI ModelSamplingDiscreteFlow
feeds the model `timestep = sigma * multiplier` (multiplier=1000) — the harness
feeds raw sigma. The golden also recorded cond_context [1,46,1024] while the
harness fixture uses context512.f32 [1,512,1024] (47 nonzero rows, different
values). This probe runs the REAL pinned graph step 0 with every combination
and reports which reproduces golden block_00_out best.

Combinations:
  timestep: raw sigma (s), sigma*1000, sigma*multiplier_fit (if golden old-form)
  context:  context512 (harness fixture), cond_context padded to 512, cond_context 46
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
    # D032: block_00_out is from sampler step 7 at sigma 0.3050089478492737
    golden_b00 = torch.from_numpy(z["block_00_out"].astype(np.float32)).to(device)  # [1,1,32,32,2048]
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    s0 = sigmas[0]
    s7 = sigmas[7]
    print("sigma0:", s0, "sigma7:", s7, "(block_00_out is from step 7 per D032)")

    x0 = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "x_in.f32"),
                                      dtype=np.float32)).view(1, 16, 1, 64, 64).to(device)
    ctx512 = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "context512.f32"),
                                          dtype=np.float32)).view(1, 512, 1024).to(device)
    cc46 = torch.from_numpy(z["cond_context"].astype(np.float32)).to(device)  # [1,46,1024]

    # cond_context padded to 512 with zeros (harness mask semantics unknown; zero pad)
    cc512 = torch.zeros(1, 512, 1024, device=device)
    cc512[:, :46] = cc46

    print("loading weights + building real graph...")
    w = ladder_real.load_src_weights(args.source)
    model = ladder_real.build_real(w, torch.bfloat16, device)
    model.eval()

    def block0(x, t, ctx):
        with torch.no_grad():
            vel, caps = up.capture_forward(model, x, t, ctx, device=device)
        return vel, caps

    # candidates on the timestep convention at step 7 (where golden block00 was captured)
    candidates = []
    # timestep conventions at sigma7
    t_raw = s7
    t_x1000 = s7 * 1000.0
    shift = 3.0
    u = shift / s7 - shift
    t_old = 1.0 / (1.0 + u)
    t_old_x1000 = t_old * 1000.0
    t_old_x10000 = t_old * 10000.0

    # step-7 model input = post-step-6 latent from the trajectory trace run
    step7_in = os.path.join(args.out, "step06_latent.f32")
    if not os.path.isfile(step7_in) and os.path.isdir(os.path.join(args.out, "traj")):
        step7_in = os.path.join(args.out, "traj", "step06_latent.f32")
    if os.path.isfile(step7_in):
        x_in7 = torch.from_numpy(np.fromfile(step7_in, dtype=np.float32)).view(1, 16, 1, 64, 64).to(device)
        print("using step06 latent as step-7 model input")
    else:
        # run trajectory to get post-step-6 latent
        x = x0
        with torch.no_grad():
            for i in range(7):
                s, s_next = sigmas[i], sigmas[i + 1]
                v = model(x, torch.tensor([s], dtype=x.dtype, device=device), ctx512)
                denoised = x - s * v
                x = x + (x - denoised) / s * (s_next - s)
        x_in7 = x
        print("computed step-7 model input by running 7 steps")

    for tname, t in [("raw_sigma", t_raw), ("sigma_x1000", t_x1000),
                     ("oldform_t", t_old), ("oldform_t_x1000", t_old_x1000),
                     ("oldform_t_x10000", t_old_x10000)]:
        for cname, ctx in [("ctx512", ctx512), ("cc_pad512", cc512), ("cc46_as512", cc46)]:
            try:
                vel, caps = block0(x_in7, t, ctx)
                b0 = caps.get("block00_out")
                if b0 is None:
                    b0 = caps.get("block_00_out")
                if b0 is None:
                    b0 = caps.get("block0")
                if b0 is None:
                    print(f"t={tname:16s} ctx={cname:10s} -> no block capture; keys={list(caps)[:8]}")
                    continue
                c = cos(b0, golden_b00)
                rmse = float((b0.detach().float().cpu() - golden_b00.float().cpu()).pow(2).mean().sqrt())
                print(f"t={tname:16s} ctx={cname:10s} -> block00 cos {c:.6f} rmse {rmse:.4f}")
                candidates.append((c, tname, cname))
            except Exception as e:
                print(f"t={tname:16s} ctx={cname:10s} -> ERROR {e}")

    candidates.sort(reverse=True)
    print("\nBest:", candidates[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
