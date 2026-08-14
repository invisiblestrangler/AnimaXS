#!/usr/bin/env python3
"""Confirm decoded vs streaming bit-equality for D(w8) and E(w4) through the
REAL graph at step 0 — completes the ladder_real streaming checks that keep
dying in the long run. Checkpoints each result as it goes."""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animapk_cuda import compare as cmp  # noqa: E402
from animapk_cuda.reader import PackFile  # noqa: E402
from animapk_cuda.runtime import DecodedReference, StreamingAnimapk  # noqa: E402

FIX = "/workspace/out/fixture"
OUT = "/workspace/out/ladder_real"


def main():
    device = "cuda"
    x0 = torch.from_numpy(np.fromfile(os.path.join(FIX, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(FIX, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)
    with open(os.path.join(FIX, "sigmas.txt")) as f:
        sigma = float(f.read().strip().split(",")[0])

    packs = {
        "D_w8": "/workspace/packs/anima-turbo-v1.0-xsmax-w8-v2.animapk",
        "E_w4": "/workspace/packs/anima-turbo-v1.0-xsmax-w4-v2.animapk",
    }
    results = {}
    for label, path in packs.items():
        print(f"== {label} ==", flush=True)
        with PackFile(path) as pk:
            dr = DecodedReference(pk, torch.float16, device)
            sm = StreamingAnimapk(pk, torch.float16, device)
            with torch.no_grad():
                _, c_dr = dr.forward(x0, sigma, context)
                _, c_sm = sm.forward(x0, sigma, context)
            v = cmp.metrics(c_dr["post_unpatchify_velocity"], c_sm["post_unpatchify_velocity"])
            results[label + "_decoded_vs_streaming"] = v
            print(f"   {label} decoded vs streaming: cos {v['cosine']:.8f} maxAbs {v['max_abs']:.3e} relL2 {v['rel_l2']:.3e}", flush=True)
            with open(os.path.join(OUT, "ladder_real_checkpoint.json"), "w") as fh:
                json.dump(results, fh, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
