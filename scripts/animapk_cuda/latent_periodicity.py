#!/usr/bin/env python3
"""Latent-space spatial diagnostics for the grid repro (MD section 23).

For golden vs each produced final latent: per-channel MAE map, signed mean
error map, latent FFT magnitude, checker/parity statistics, horizontal/vertical
neighbor correlation, row/column periodicity. Saves PNGs + JSON.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="grid_repro out dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = args.dir

    golden = np.fromfile(os.path.join(d, "G0_golden_rgb.f32"), dtype=np.float32)
    # golden latent from manifest lane G0 (we saved latent via lane_record but
    # not bytes; use the golden npz-verified f32: G0 uses golden_latent)
    lat_files = {
        "G1_bf16": "G1_bf16_final_latent.f32",
        "G2_fp16all": "G2_fp16all_final_latent.f32",
        "G3_w8": "G3_w8_final_latent.f32",
    }
    import glob
    g0lat = None
    # golden latent == case1 final latent (npz final_latent, sha cc56061d...)
    for cand in ("/workspace/fixtures/case1_danbooru_seed1337.npz",
                 "/workspace/fixtures/case1_danbooru_seed1337.npz"):
        if os.path.isfile(cand):
            zz = np.load(cand, allow_pickle=True)
            g0lat = zz["final_latent"].astype(np.float32).reshape(16, 64, 64)
            break
    if g0lat is None:
        raise SystemExit("golden latent not found")

    results = {}
    for name, fn in lat_files.items():
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            print("skip", fn)
            continue
        lat = np.fromfile(p, dtype=np.float32).reshape(16, 64, 64)
        delta = lat.astype(np.float64) - g0lat.astype(np.float64)

        # per-channel MAE + signed mean
        mae_c = np.mean(np.abs(delta), axis=(1, 2))
        sme_c = np.mean(delta, axis=(1, 2))

        # checker/parity statistics on delta (sum over channels)
        dsum = delta.sum(axis=0)  # [64,64]
        # parity masks
        even_even = dsum[0::2, 0::2]
        even_odd = dsum[0::2, 1::2]
        odd_even = dsum[1::2, 0::2]
        odd_odd = dsum[1::2, 1::2]
        checker = float(np.abs((even_even.mean() + odd_odd.mean())
                               - (even_odd.mean() + odd_even.mean())))
        # full checker: mean of (d[i,j] * (-1)^(i+j))
        sign = np.ones((64, 64))
        sign[1::2, :] *= -1
        sign[:, 1::2] *= -1
        checker_signed = float(np.mean(dsum * sign))

        # horizontal/vertical neighbor correlation of delta (per channel avg)
        def neigh_corr(a):
            c = 0.0
            for ch in range(16):
                x = a[ch]
                hc = np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1]
                vc = np.corrcoef(x[:-1, :].ravel(), x[1:, :].ravel())[0, 1]
                c += hc + vc
            return c / 32.0

        # row/column periodicity: autocorrelation at lag 1..8 of mean row/col
        mrow = dsum.mean(axis=1)  # [64] per-row mean of delta
        mcol = dsum.mean(axis=0)
        row_ac = [float(np.corrcoef(mrow[:-l], mrow[l:])[0, 1]) for l in range(1, 9)]
        col_ac = [float(np.corrcoef(mcol[:-l], mcol[l:])[0, 1]) for l in range(1, 9)]

        # FFT magnitude peak bins of dsum
        F = np.fft.fft2(dsum)
        mag = np.abs(F)
        peak = np.unravel_index(np.argmax(mag[1:, 1:]), mag[1:, 1:].shape)
        peak = (peak[0] + 1, peak[1] + 1)

        # save maps
        from PIL import Image

        def norm8(x):
            x = x.astype(np.float32)
            x = (x - x.min()) / (x.max() - x.min() + 1e-12)
            return (x * 255).astype(np.uint8)

        mae_map = np.mean(np.abs(delta), axis=0)
        sme_map = np.mean(delta, axis=0)
        Image.fromarray(norm8(mae_map), mode="L").save(os.path.join(args.out, f"{name}_mae.png"))
        Image.fromarray(norm8(sme_map), mode="L").save(os.path.join(args.out, f"{name}_sme.png"))

        results[name] = {
            "mae_per_channel": [float(v) for v in mae_c],
            "signed_mean_per_channel": [float(v) for v in sme_c],
            "checker_bias": checker,
            "checker_signed_mean": checker_signed,
            "neighbor_corr_avg": float(neigh_corr(delta)),
            "row_autocorr_lag1_8": row_ac,
            "col_autocorr_lag1_8": col_ac,
            "fft_peak_bin": [int(peak[0]), int(peak[1])],
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_l2_frac": float(np.linalg.norm(delta) / np.linalg.norm(g0lat)),
        }
        print(name, json.dumps(results[name], indent=1))

    with open(os.path.join(args.out, "latent_periodicity.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
