#!/usr/bin/env python3
"""Compare Oracle E2 block-0 internal stage tensors.

Accepts either an unpacked physical-device `device_manifest.json` or a CUDA
`oracle_e_checkpoints.json` for each side. The nine canonical block-0 self
stages are compared in execution order and the first material mismatch is
reported without conflating it with later residual accumulation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

STAGE_ORDER = [
    "stage_b00_self_modulation",
    "stage_b00_self_projection_input",
    "stage_b00_self_q_raw",
    "stage_b00_self_k_raw",
    "stage_b00_self_v_raw",
    "stage_b00_self_q_post_rope",
    "stage_b00_self_k_post_rope",
    "stage_b00_self_attended",
    "stage_b00_self_branch",
]
DTYPES = {
    "float16-le": np.dtype("<f2"),
    "float32-le": np.dtype("<f4"),
}


def _stage_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "block0_stage_payloads" in manifest:
        return list(manifest.get("block0_stage_payloads") or [])
    if "block0_stage_records" in manifest:
        return list(manifest.get("block0_stage_records") or [])
    raise ValueError("manifest has no Oracle E2 block-0 stage records")


def _canonical_name(record: dict[str, Any]) -> str:
    if record.get("name"):
        return str(record["name"])
    if record.get("stage"):
        return "stage_b00_" + str(record["stage"]).replace(".", "_")
    raise ValueError(f"stage record has no name/stage: {record}")


def _load(manifest_path: Path) -> dict[str, tuple[dict[str, Any], np.ndarray]]:
    manifest = json.loads(manifest_path.read_text())
    records = _stage_records(manifest)
    result: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for record in records:
        name = _canonical_name(record)
        if name in result:
            raise ValueError(f"duplicate stage {name} in {manifest_path}")
        dtype_name = str(record.get("dtype"))
        if dtype_name not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r} for {name}")
        filename = record.get("file")
        if not filename:
            raise ValueError(f"stage {name} has no file in {manifest_path}")
        path = manifest_path.parent / str(filename)
        if not path.is_file():
            raise ValueError(f"stage file missing for {name}: {path}")
        values = np.fromfile(path, dtype=DTYPES[dtype_name])
        expected = int(record.get("elementCount", record.get("element_count", values.size)))
        if values.size != expected:
            raise ValueError(f"stage {name} has {values.size} values; manifest says {expected}")
        result[name] = (record, values.astype(np.float64))
    return result


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"stage shape mismatch: {reference.shape} vs {candidate.shape}")
    diff = candidate - reference
    ref_l2_sq = float(np.dot(reference, reference))
    cand_l2_sq = float(np.dot(candidate, candidate))
    diff_l2_sq = float(np.dot(diff, diff))
    denom = math.sqrt(ref_l2_sq * cand_l2_sq)
    return {
        "relative_rmse": math.sqrt(diff_l2_sq / max(ref_l2_sq, 1e-300)),
        "cosine": float(np.dot(reference, candidate)) / denom if denom else 1.0,
        "rmse": math.sqrt(diff_l2_sq / reference.size),
        "mean_absolute_error": float(np.mean(np.abs(diff))),
        "max_absolute_error": float(np.max(np.abs(diff))),
        "reference_rms": math.sqrt(ref_l2_sq / reference.size),
        "candidate_rms": math.sqrt(cand_l2_sq / candidate.size),
    }


def compare(
    reference_manifest: Path,
    candidate_manifest: Path,
    *,
    rel_threshold: float = 0.01,
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    reference = _load(reference_manifest)
    candidate = _load(candidate_manifest)
    missing_ref = [name for name in STAGE_ORDER if name not in reference]
    missing_cand = [name for name in STAGE_ORDER if name not in candidate]
    if missing_ref or missing_cand:
        raise ValueError(
            f"incomplete stage manifests: reference missing={missing_ref}, candidate missing={missing_cand}")

    rows = []
    first = None
    for ordinal, name in enumerate(STAGE_ORDER):
        rr, rv = reference[name]
        cr, cv = candidate[name]
        if str(rr.get("dtype")) != str(cr.get("dtype")):
            raise ValueError(
                f"stage dtype mismatch for {name}: {rr.get('dtype')} vs {cr.get('dtype')}")
        metrics = _metrics(rv, cv)
        crossed = metrics["relative_rmse"] > rel_threshold or metrics["cosine"] < cosine_threshold
        row = {
            "ordinal": ordinal,
            "stage": name.removeprefix("stage_b00_"),
            "name": name,
            "dtype": rr.get("dtype"),
            "element_count": int(rv.size),
            **metrics,
            "threshold_crossed": crossed,
        }
        rows.append(row)
        if crossed and first is None:
            first = dict(row)

    return {
        "schema": 1,
        "oracle": "AnimaXS Oracle E2 block0 stage parity",
        "reference_manifest": str(reference_manifest.resolve()),
        "candidate_manifest": str(candidate_manifest.resolve()),
        "relative_rmse_threshold": rel_threshold,
        "cosine_threshold": cosine_threshold,
        "stage_count": len(rows),
        "first_stage_threshold_crossing": first,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--relative-threshold", type=float, default=0.01)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    args = ap.parse_args()
    result = compare(
        args.reference,
        args.candidate,
        rel_threshold=args.relative_threshold,
        cosine_threshold=args.cosine_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage_count": result["stage_count"],
        "first_stage_threshold_crossing": result["first_stage_threshold_crossing"],
        "output": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
