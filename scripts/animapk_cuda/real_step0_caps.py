#!/usr/bin/env python3
"""Save step-0 per-block captures for the REAL upstream graph with each
weight source (A bf16 official, C fp16-all pack, D w8 pack, E w4 pack) —
the CUDA side of the backend-parity comparison against the Metal captures.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dit_source_oracle as dso  # noqa: E402

from animapk_cuda.reader import PackFile  # noqa: E402
from animapk_cuda import upstream as up  # noqa: E402
from animapk_cuda.ladder_real import build_real, decode_pack, load_src_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--fp16-pack", required=True)
    ap.add_argument("--w8-pack", required=True)
    ap.add_argument("--w4-pack", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    x0 = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)
    with open(os.path.join(args.fixture, "sigmas.txt")) as f:
        sigma = float(f.read().strip().split(",")[0])

    jobs = {
        "A_bf16_real": ("bf16", None, torch.bfloat16),
        "C_fp16all_real": ("pack", args.fp16_pack, torch.float16),
        "D_w8_real": ("pack", args.w8_pack, torch.float16),
        "E_w4_real": ("pack", args.w4_pack, torch.float16),
    }
    provenance = {}
    for label, (kind, path, dtype) in jobs.items():
        print(f"== {label} ==", flush=True)
        if kind == "bf16":
            w = load_src_weights(args.source)
        else:
            w, prov = decode_pack(path)
            provenance[label] = prov
        model = build_real(w, dtype, device)
        vel, caps = up.capture_forward(model, x0, sigma, context, device=device)
        caps["post_unpatchify_velocity"] = vel
        npz = os.path.join(args.out, f"caps_{label}.npz")
        np.savez(npz, **{k: v.numpy() for k, v in caps.items()})
        print(f"   saved {npz} ({len(caps)} captures)", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump({"source": args.source, "source_sha256": dso.PINNED_SOURCE_SHA,
                   "sigma": sigma, "packs": {k: {"sha256": v["sha256"], "size": v["size"]} for k, v in provenance.items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
