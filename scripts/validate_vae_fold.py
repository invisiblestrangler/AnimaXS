#!/usr/bin/env python3
"""Validate every T=1 decoder CausalConv3d fold against pinned Wan VAE semantics.

Pinned source: comfy/ldm/wan/vae.py at cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44.
For an uncached single frame, Wan CausalConv3d uses causal ZERO padding. A kt=3
kernel therefore folds to weight[:, :, 2, :, :], not a temporal sum. Decoder
upsample3d time_conv branches are not executed when feat_cache is None (T=1).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/anima-xsmax/scripts")
from inspect_animapk import Animapk

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"


def conv3d_single_frame(x, weight):
    """Direct N=1,T=1 causal-zero cross correlation, sampled channels."""
    co, ci, kt, kh, kw = weight.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    temporal = np.zeros((ci, kt, padded.shape[1], padded.shape[2]), np.float64)
    temporal[:, -1] = padded
    out = np.empty((co, x.shape[1], x.shape[2]), np.float64)
    for y in range(x.shape[1]):
        for z in range(x.shape[2]):
            patch = temporal[:, :, y:y + kh, z:z + kw]
            out[:, y, z] = np.einsum("cthw,octhw->o", patch, weight, optimize=True)
    return out


def conv2d(x, weight):
    co, ci, kh, kw = weight.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    out = np.empty((co, x.shape[1], x.shape[2]), np.float64)
    for y in range(x.shape[1]):
        for z in range(x.shape[2]):
            patch = padded[:, y:y + kh, z:z + kw]
            out[:, y, z] = np.einsum("chw,ochw->o", patch, weight, optimize=True)
    return out


def metrics(a, b):
    error = a - b
    return {
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        default="/root/anima-xsmax/results/packs/qwen-image-vae-xsmax-fp16.animapk")
    parser.add_argument("--report", default="docs/VAE_FOLD_REPORT.md")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    pack = Animapk(args.pack)
    records = pack.read_table()
    metadata = {m["blob_offset"]: m for m in pack.meta["tensor_meta"]}
    selected = []
    for record in records:
        meta = metadata[record["blob_offset"]]
        name, shape = meta["name"], meta["shape"]
        if name == "conv2.weight" or (
                name.startswith("decoder.") and name.endswith(".weight") and len(shape) == 5):
            selected.append((record, meta))

    rng = np.random.default_rng(20260811)
    rows = []
    for record, meta in selected:
        full = pack.decode(record)
        co, ci = min(2, full.shape[0]), min(3, full.shape[1])
        weight = full[:co, :ci].astype(np.float64)
        x = rng.standard_normal((ci, 5, 6)).astype(np.float64)
        true = conv3d_single_frame(x, weight)
        causal_fold = conv2d(x, weight[:, :, -1])
        sum_fold = conv2d(x, weight.sum(axis=2))
        rows.append({
            "name": meta["name"], "shape": meta["shape"],
            "runtime_t1": "skipped by decoder" if ".time_conv." in meta["name"] else "executed",
            "causal_last_slice": metrics(true, causal_fold),
            "replication_sum": metrics(true, sum_fold),
        })

    # The two einsum paths reduce a different number of explicit zero terms;
    # tolerate only a few Float64 ulps from reduction order.
    failures = [r for r in rows if r["causal_last_slice"]["max_abs"] > 2e-15]
    sum_distinguishers = [
        r for r in rows if r["shape"][2] > 1 and
        r["replication_sum"]["max_abs"] > 1e-7]
    hasher = hashlib.sha256()
    with open(args.pack, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    result = {
        "pinned_commit": PINNED_COMMIT, "pack_sha256": digest,
        "tensor_count": len(rows), "failures": len(failures),
        "sum_fold_distinguishers": len(sum_distinguishers), "tensors": rows,
    }
    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or ".", exist_ok=True)
        with open(args.json_path, "w") as handle:
            json.dump(result, handle, indent=2)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as handle:
        handle.write("# VAE T=1 causal-convolution fold report\n\n")
        handle.write(f"Pinned ComfyUI commit: `{PINNED_COMMIT}`  \n")
        handle.write(f"VAE pack SHA-256: `{digest}`  \n")
        handle.write(f"Validated tensors: **{len(rows)}**; last-slice failures: **{len(failures)}**.\n\n")
        handle.write("The decoder is Wan `vae.py`, not the similarly named Cosmos tokenizer. "
                     "For uncached T=1, `CausalConv3d` uses causal zero padding, so a temporal "
                     "kernel folds to its final temporal slice. Summing temporal slices models "
                     "replication padding and is wrong here. The two `time_conv` tensors are "
                     "not executed because T=1 decoding creates no feature cache.\n\n")
        handle.write("| Tensor | Shape | T=1 runtime | last-slice maxAbs | sum-fold maxAbs |\n")
        handle.write("|---|---:|---|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['name']}` | `{row['shape']}` | {row['runtime_t1']} | "
                f"{row['causal_last_slice']['max_abs']:.3g} | "
                f"{row['replication_sum']['max_abs']:.6g} |\n")
        handle.write("\n`VAE_2D_FOLD_VALIDATED=LAST_TEMPORAL_SLICE_CAUSAL_ZERO`\n")

    print(f"validated {len(rows)} decoder/post-quant tensors; failures={len(failures)}")
    print(f"replication-sum contradicted by {len(sum_distinguishers)} kt>1 tensors")
    print("VAE_2D_FOLD_VALIDATED=LAST_TEMPORAL_SLICE_CAUSAL_ZERO")
    if failures or not sum_distinguishers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
