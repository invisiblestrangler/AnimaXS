#!/usr/bin/env python3
"""Measure the exact-8-pixel periodic carrier in generated vs reference images.

The AnimaXS image-quality defect shows up as a severe regular woven/etched/grid
pattern. It has a strong machine-detectable periodic signature: normalized
Fourier energy at the exact 1/8-cycle-per-pixel bins (8-pixel period on a
512x512 image) after per-channel mean removal.

Usage:
  measure_grid_carrier.py <generated.png> <reference.png> [--json OUT.json]

Output (text or JSON):
  horizontal score (energy at kx=+/-1/8 cyc/px, ky=0)
  vertical score   (energy at kx=0, ky=+/-1/8)
  diagonal score   (energy at kx=+/-1/8 AND ky=+/-1/8)
  total exact-8px score
  generated/reference ratio

Reference baseline (defective run #29 artifact, 2026-08-13):
  generated_total ~= 0.01228284
  reference_total ~= 0.0000353322
  ratio           ~= 347.6
  horizontal ratio ~= 1183x, vertical ratio ~= 112.8x

These are calibration numbers for THIS normalization (energy fraction after
per-channel mean removal), not universal thresholds. A winning candidate must
reduce the total by orders of magnitude and lose the lattice visually.
"""

import json
import sys

import numpy as np
from PIL import Image

PERIOD_PX = 8  # exact carrier period in pixels


def carrier_scores(rgb: np.ndarray) -> dict:
    """rgb: float32 array [H, W, C] in [0, 1] (or any linear scale).

    Returns normalized energy at the exact +/-1/8-cycle-per-pixel bins after
    per-channel mean removal. Normalization = carrier energy / total energy
    (Parseval: energy fraction), averaged over channels.
    """
    h, w, _ = rgb.shape
    k = w // PERIOD_PX  # exact bin index for an 8-px period along width
    assert h == w == 512 and k * PERIOD_PX == w, \
        f"expected 512x512 image, got {h}x{w}"

    channel_scores = {"horizontal": [], "vertical": [], "diagonal": [], "total": []}
    for c in range(rgb.shape[2]):
        x = rgb[:, :, c].astype(np.float64)
        x = x - x.mean()
        F = np.fft.rfft2(x)
        energy = np.abs(F) ** 2
        total = energy.sum()

        # rfft2 layout: rows kx in 0..511 (negative = 512-kx), cols ky in 0..256
        # (negative = 256-ky).
        kx_pos, kx_neg = k, h - k
        ky_pos, ky_neg = k, w // 2 - k  # w//2 == 256 for 512

        def e(kx, ky):
            return float(energy[kx, ky])

        horizontal = e(kx_pos, 0) + e(kx_neg, 0)
        vertical = e(0, ky_pos) + e(0, ky_neg)
        diagonal = (
            e(kx_pos, ky_pos) + e(kx_pos, ky_neg)
            + e(kx_neg, ky_pos) + e(kx_neg, ky_neg)
        )
        total_carrier = horizontal + vertical + diagonal

        for key, val in (("horizontal", horizontal), ("vertical", vertical),
                         ("diagonal", diagonal), ("total", total_carrier)):
            channel_scores[key].append(val / total if total > 0 else 0.0)

    return {key: float(np.mean(vals)) for key, vals in channel_scores.items()}


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    generated_path, reference_path = sys.argv[1], sys.argv[2]
    json_out = None
    if len(sys.argv) >= 5 and sys.argv[3] == "--json":
        json_out = sys.argv[4]

    gen = carrier_scores(load_rgb(generated_path))
    ref = carrier_scores(load_rgb(reference_path))

    ratio_total = gen["total"] / ref["total"] if ref["total"] > 0 else float("inf")
    ratio_h = gen["horizontal"] / ref["horizontal"] if ref["horizontal"] > 0 else float("inf")
    ratio_v = gen["vertical"] / ref["vertical"] if ref["vertical"] > 0 else float("inf")
    ratio_d = gen["diagonal"] / ref["diagonal"] if ref["diagonal"] > 0 else float("inf")

    result = {
        "generated": gen,
        "reference": ref,
        "ratio": {
            "total": ratio_total,
            "horizontal": ratio_h,
            "vertical": ratio_v,
            "diagonal": ratio_d,
        },
        "note": "normalized Fourier energy at exact +/-1/8-cycle-per-pixel bins "
                "(8px period), per-channel mean removed; score = carrier "
                "energy / total energy, channel-averaged",
        "baseline": {
            "generated_total": 0.01228284,
            "reference_total": 0.0000353322,
            "ratio_total": 347.6,
        },
    }

    if json_out:
        with open(json_out, "w") as fh:
            json.dump(result, fh, indent=2)

    print(f"generated total exact-8px carrier score: {gen['total']:.8f}")
    print(f"reference total exact-8px carrier score: {ref['total']:.8f}")
    print(f"ratio: ~{ratio_total:.1f}x")
    print(f"  horizontal: generated {gen['horizontal']:.8f} vs ref {ref['horizontal']:.8f} "
          f"-> ~{ratio_h:.1f}x")
    print(f"  vertical:   generated {gen['vertical']:.8f} vs ref {ref['vertical']:.8f} "
          f"-> ~{ratio_v:.1f}x")
    print(f"  diagonal:   generated {gen['diagonal']:.8f} vs ref {ref['diagonal']:.8f} "
          f"-> ~{ratio_d:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
