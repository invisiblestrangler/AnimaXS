"""Prepare the canonical step-0 fixture from the golden NPZ.

Extracts:
  x_in          = init_noise_randn [1,16,1,64,64] (sigma=1.0, step 0)
  sigmas        = sigmas_comfy [9]
  context512    = canonical [1,512,1024] fp32 cross-attention context computed
                  by the SOURCE LLM adapter (anima/model.py preprocess_text_embeds)
                  from cond_context + t5 ids/weights — identical for all ladder
                  variants (each variant casts to its own compute dtype).
Outputs a small fixture dir consumed by ladder.py / upstream.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dit_source_oracle as dso  # noqa: E402

CTX_TOKENS = 512
CTX_DIM = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="golden case1_danbooru_seed1337.npz")
    ap.add_argument("--source", required=True, help="official anima-turbo-v1.0.safetensors")
    ap.add_argument("--out", required=True, help="fixture output dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)
    x_in = torch.from_numpy(z["init_noise_randn"]).float()
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    cond_context = torch.from_numpy(z["cond_context"]).float()  # [1,46,1024]
    t5_ids = torch.from_numpy(z["cond_meta_t5xxl_ids"].astype(np.int64))
    t5_weights = torch.from_numpy(z["cond_meta_t5xxl_weights"]).float()

    # canonical fp32 context from the source adapter
    w = dso.load_weights(args.source)
    adapter = dso.Adapter(w, dtype=torch.float32)
    with torch.no_grad():
        context512 = adapter(cond_context, t5_ids, t5_weights)  # [1,512,1024] fp32
    assert context512.shape == (1, CTX_TOKENS, CTX_DIM)

    def save(name, t):
        p = os.path.join(args.out, name)
        t.contiguous().numpy().tofile(p)
        return p, hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

    manifest = {}
    for fn, t in [
        ("x_in.f32", x_in),
        ("context512.f32", context512),
        ("cond_context.f32", cond_context),
    ]:
        p, h = save(fn, t)
        manifest[fn] = {"bytes": t.numel() * 4, "sha256": h}
    with open(os.path.join(args.out, "sigmas.txt"), "w") as f:
        f.write(",".join(f"{s:.10g}" for s in sigmas))
    manifest["sigmas.txt"] = {"values": sigmas}
    manifest["sigma0"] = sigmas[0]
    manifest["x_in"] = {"shape": list(x_in.shape)}
    manifest["context512"] = {"shape": list(context512.shape)}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"fixture ready in {args.out}: x_in {list(x_in.shape)} sigma0 {sigmas[0]} context {list(context512.shape)}")


if __name__ == "__main__":
    main()
