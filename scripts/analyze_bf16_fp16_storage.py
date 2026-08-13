"""BF16 -> FP16 storage audit (§15).

For every official tensor: compare
    source BF16 -> FP32 reference
vs
    source BF16 -> FP32 -> FP16 -> FP32
Outputs bf16_fp16_storage_summary.json, bf16_fp16_storage_tensors.csv,
bf16_fp16_top20.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    total_elems = 0
    exact_total = 0
    with safe_open(args.source, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype != torch.bfloat16:
                continue
            ref = t.float()
            conv = t.float().half().float()
            n = ref.numel()
            total_elems += n
            exact = (ref == conv).sum().item()
            exact_total += exact
            diff = ref - conv
            denom = ref.abs().clamp_min(1e-30)
            rows.append({
                "tensor": k,
                "shape": list(t.shape),
                "elements": n,
                "exact_count": exact,
                "exact_pct": 100.0 * exact / n,
                "cosine": float((ref * conv).sum() / (ref.norm() * conv.norm() + 1e-30)),
                "rmse": float(diff.pow(2).mean().sqrt()),
                "rel_l2": float(diff.norm() / conv.norm().clamp_min(1e-30)),
                "max_abs": float(diff.abs().max()),
                "max_rel": float((diff.abs() / denom).max()),
                "src_zeros": int((ref == 0).sum()),
                "new_zeros": int((conv == 0).sum()) - int((ref == 0).sum()),
                "fp16_subnormals": int((conv.abs() > 0) & (conv.abs() < 2 ** -14)),
                "inf": int((conv == float("inf")).sum()) + int((conv == float("-inf")).sum()),
                "nan": int(torch.isnan(conv).sum()),
                "mag_min": float(conv.abs().min()),
                "mag_max": float(conv.abs().max()),
            })
            del t, ref, conv, diff

    with open(os.path.join(args.out, "bf16_fp16_storage_tensors.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    top = sorted(rows, key=lambda r: r["rel_l2"], reverse=True)[:20]
    md = ["# BF16 -> FP16 storage — top 20 by relative L2\n",
          "| tensor | shape | rel L2 | cosine | RMSE | maxAbs | exact% |", "|---|---|---|---|---|---|---|"]
    for r in top:
        md.append(f"| {r['tensor']} | {r['shape']} | {r['rel_l2']:.4e} | {r['cosine']:.8f} | {r['rmse']:.4e} | {r['max_abs']:.4e} | {r['exact_pct']:.2f} |")
    with open(os.path.join(args.out, "bf16_fp16_top20.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    summary = {
        "tensors": len(rows),
        "total_elements": total_elems,
        "exact_total": exact_total,
        "exact_total_pct": 100.0 * exact_total / max(total_elems, 1),
        "any_inf": any(r["inf"] for r in rows),
        "any_nan": any(r["nan"] for r in rows),
        "any_subnormal": any(r["fp16_subnormals"] for r in rows),
        "max_rel_l2": max(r["rel_l2"] for r in rows),
        "max_max_abs": max(r["max_abs"] for r in rows),
        "worst_tensor": max(rows, key=lambda r: r["rel_l2"])["tensor"],
        "focus_tensors": [r["tensor"] for r in rows if r["rel_l2"] > 0.01],
    }
    with open(os.path.join(args.out, "bf16_fp16_storage_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
