"""CUDA-vs-Metal step-0 backend parity.

Consumes:
  - Metal captures exported from the XCTest result bundle
    (w8-step0-block{N}-output.f32, w8-step0-block0-input.f32,
    w8-step0-embedding.f32, w8-step0-adaln.f32, w8-rope.f32,
    step00_x_in.f32, step00_denoised.f32, cross-context.f32)
  - CUDA ladder captures (caps_<variant>_decoded.npz on the Clore box,
    or a local mirror dir)

Emits backend_parity_{fp16,w8,w4}.{csv,json,md} + backend_parity_overview.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from animapk_cuda import compare as cmp

STAGES = (
    ["pre_x_embedder_tokens", "x_embedder_out", "timestep_emb", "adaln_lora"]
    + [f"block{i:02d}_in" for i in range(28)]
    + [f"block{i:02d}_out" for i in range(28)]
    + ["pre_final_norm", "post_final_projection_patched", "post_unpatchify_velocity"]
)


def load_metal(metal_dir: str, variant: str) -> dict:
    """Load Metal step-0 captures into oracle stage names."""
    caps = {}
    f = os.path.join(metal_dir, "w8-step0-block0-input.f32")
    if os.path.exists(f):
        caps["block00_in"] = torch_from_file(f, (1, 1, 32, 32, 2048))
    f = os.path.join(metal_dir, "w8-step0-embedding.f32")
    if os.path.exists(f):
        caps["x_embedder_out"] = torch_from_file(f, (1, 1, 32, 32, 2048))
    f = os.path.join(metal_dir, "w8-step0-adaln.f32")
    if os.path.exists(f):
        caps["adaln_lora"] = torch_from_file(f, (1, 1, 6144))
    f = os.path.join(metal_dir, "w8-rope.f32")
    if os.path.exists(f):
        caps["rope_sha256_meta"] = os.path.basename(f)
    for i in range(28):
        f = os.path.join(metal_dir, f"w8-step0-block{i:02d}-output.f32")
        if os.path.exists(f):
            caps[f"block{i:02d}_out"] = torch_from_file(f, (1, 1, 32, 32, 2048))
    # trajectory (step 0)
    f = os.path.join(metal_dir, "step00_x_in.f32")
    if os.path.exists(f):
        caps["step0_x_in"] = torch_from_file(f, (1, 16, 1, 64, 64))
    f = os.path.join(metal_dir, "step00_denoised.f32")
    if os.path.exists(f):
        caps["step0_denoised"] = torch_from_file(f, (1, 16, 1, 64, 64))
    f = os.path.join(metal_dir, "cross-context.f32")
    if os.path.exists(f):
        caps["context"] = torch_from_file(f, (1, 512, 1024))
    return caps


def torch_from_file(path, shape):
    import torch
    return torch.from_numpy(np.fromfile(path, dtype=np.float32)).view(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metal-dir", required=True, help="exported Metal captures")
    ap.add_argument("--cuda-dir", required=True, help="dir with caps_<V>_decoded.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="all", choices=["all", "fp16", "w8", "w4"])
    args = ap.parse_args()

    import torch
    os.makedirs(args.out, exist_ok=True)

    variant_packs = {"fp16": "C", "w8": "D", "w4": "E"}
    results = {}
    rows = []
    for label, vcode in variant_packs.items():
        npz_path = os.path.join(args.cuda_dir, f"caps_{vcode}_decoded.npz")
        if not os.path.exists(npz_path):
            print(f"missing CUDA captures {npz_path}")
            continue
        cuda = {k: torch.from_numpy(v) for k, v in np.load(npz_path).items()}
        metal = load_metal(args.metal_dir, label)
        if not metal:
            print(f"no Metal captures for {label}")
            continue
        # velocity from trajectory
        if "step0_x_in" in metal and "step0_denoised" in metal:
            metal["post_unpatchify_velocity"] = (
                (metal["step0_x_in"] - metal["step0_denoised"]) / 1.0
            ).float()
        table = {}
        for s in STAGES:
            if s in cuda and s in metal:
                table[s] = cmp.metrics(cuda[s], metal[s])
        results[label] = table
        for s, m in table.items():
            rows.append({"backend": f"CUDA_vs_Metal_{label}", "stage": s,
                         **{k: m.get(k) for k in ("cosine", "rmse", "rel_l2", "max_abs", "a_norm", "b_norm")}})
        cmp.write_csv(os.path.join(args.out, f"backend_parity_{label}.csv"),
                      [r for r in rows if r["backend"] == f"CUDA_vs_Metal_{label}"])
        cmp.write_json(os.path.join(args.out, f"backend_parity_{label}.json"), table)
        cmp.write_md(os.path.join(args.out, f"backend_parity_{label}.md"),
                     f"# Backend parity {label} (CUDA vs Metal, step 0)\n\n" + cmp.stage_table(STAGES, table))

    if results:
        series = {k: {s: v[s]["cosine"] for s in v} for k, v in results.items()}
        cmp.plot_ladder(os.path.join(args.out, "backend_parity_overview.png"), STAGES, series,
                        "CUDA vs Metal step-0 block cosine", "cosine")
    print(json.dumps({k: {"velocity_cosine": v.get("post_unpatchify_velocity", {}).get("cosine"),
                          "block27_cosine": v.get("block27_out", {}).get("cosine")} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
