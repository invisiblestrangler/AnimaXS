#!/usr/bin/env python3
"""Quick CUDA real-graph vs Metal simulator per-block comparison at step 0.

Compares the Metal step-0 captures (f32 files from the macOS workflow) against
the CUDA real-graph step-0 captures (npz from real_step0_caps.py) for the same
packs.  This is the C/D/E-Metal vs C/D/E-CUDA backend-parity check.
"""

import json
import os
import sys

import numpy as np

META = "/tmp/step0"          # Metal captures: /tmp/step0/{fp16-all,w8,w4}/*.f32
CUDA = "/tmp/cuda_step0"     # CUDA caps npz (to be pulled locally)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    variant_map = {"fp16-all": "C_fp16all_real", "w8": "D_w8_real", "w4": "E_w4_real"}
    out = {}
    for metal_dir, cuda_name in variant_map.items():
        npz_path = os.path.join(CUDA, f"caps_{cuda_name}.npz")
        if not os.path.exists(npz_path):
            print(f"missing CUDA caps {npz_path}")
            continue
        z = np.load(npz_path)
        rows = []
        for blk in [0, 1, 13, 27]:
            metal_f = os.path.join(META, metal_dir, f"w8-step0-block{blk}-output.f32")
            if not os.path.exists(metal_f):
                continue
            m = np.fromfile(metal_f, dtype=np.float32).reshape(1, 1, 32, 32, 2048)
            c = z[f"block{blk:02d}_out"]
            rows.append({"block": blk, "cosine": cos(m, c)})
        out[metal_dir] = rows
        print(f"== {metal_dir} (CUDA {cuda_name}) ==")
        for r in rows:
            print(f"   block{r['block']:>2}: cos {r['cosine']:.6f}")
    with open("/tmp/cuda_vs_metal_block0_27.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
