#!/usr/bin/env python3
"""Build bounded-memory ANMA v1 DiT packs.

The v1 container is deliberately kept byte-compatible with the recovered
packer/runtime.  v2 changes the quantizer and writer, not the file format:
rank-2 matrices use row-reset group-64 W4/W8 data with fp16 scale/zero
parameters, while rank<=1 tensors remain fp16.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

MAGIC = b"ANMA"
VERSION = 1
ALIGN = 16_384
HEADER_SIZE = 256
RECORD_SIZE = 128
DEFAULT_METADATA_RESERVE = 8 * 1024 * 1024
COMPONENT = {"dit": 1, "te": 2, "vae": 3}
STORAGE_CODE = {"w4": 0, "w8": 1, "fp16": 2, "fp32": 3}


def align_up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path, "rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise RuntimeError("source is not a complete safetensors file")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 256 * 1024 * 1024:
            raise RuntimeError("safetensors header is unreasonably large")
        raw_header = source.read(header_length)
    header = json.loads(raw_header)
    return {name: value for name, value in header.items() if name != "__metadata__"}


def product(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def parse_precision_map(path: str | None, default: str) -> tuple[str, list[dict[str, str]], str | None]:
    if not path:
        return default, [], None
    with open(path, "rb") as source:
        raw = source.read()
    value = json.loads(raw)
    base = value.get("default", default)
    if base not in ("w4", "w8"):
        raise ValueError("precision-map default must be w4 or w8")
    overrides: list[dict[str, str]] = []
    for entry in value.get("overrides", []):
        match = entry.get("match")
        storage = entry.get("storage")
        if not isinstance(match, str) or storage not in ("w4", "w8", "fp16"):
            raise ValueError("precision-map overrides require match and w4/w8/fp16 storage")
        overrides.append({"match": match, "storage": storage})
    return base, overrides, hashlib.sha256(raw).hexdigest()


def choose_storage(name: str, rank: int, default: str, overrides: list[dict[str, str]]) -> str:
    if rank <= 1:
        return "fp16"
    if rank != 2:
        raise ValueError(f"unsupported rank {rank} tensor {name}; only rank<=1/rank2 is allowed")
    for override in overrides:
        if fnmatch.fnmatchcase(name, override["match"]):
            return override["storage"]
    return default


def make_plan(
    header: dict[str, dict[str, Any]],
    default_quant: str,
    overrides: list[dict[str, str]],
    group: int,
    metadata_reserve: int,
) -> list[dict[str, Any]]:
    if group != 64:
        raise ValueError("this refinement cycle requires group size 64")
    plans: list[dict[str, Any]] = []
    for execution_index, name in enumerate(sorted(header)):
        spec = header[name]
        shape = [int(x) for x in spec["shape"]]
        rank = len(shape)
        storage = choose_storage(name, rank, default_quant, overrides)
        count = product(shape)
        if rank <= 1 or storage == "fp16":
            data_size = count * 2
            scale_size = zero_size = 0
            logical = "fp16"
        else:
            rows, columns = shape
            groups_per_row = (columns + group - 1) // group
            data_size = rows * ((columns + 1) // 2 if storage == "w4" else columns)
            scale_size = zero_size = rows * groups_per_row * 2
            logical = storage
        blob_data_size = data_size + scale_size + zero_size
        block_match = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
        plans.append({
            "name": name,
            "shape": shape,
            "logical_dtype": logical,
            "storage_dtype": storage,
            "numel": count,
            "data_size": data_size,
            "scale_size": scale_size,
            "zero_size": zero_size,
            "blob_size": align_up(blob_data_size),
            "block_index": int(block_match.group(1)) if block_match else -1,
            "execution_index": execution_index,
            "source_dtype": spec.get("dtype"),
            "source_offsets": spec.get("data_offsets"),
        })

    table_offset = HEADER_SIZE + metadata_reserve
    payload_offset = align_up(table_offset + RECORD_SIZE * len(plans))
    cursor = payload_offset
    for plan in plans:
        plan["blob_offset"] = cursor
        cursor += plan["blob_size"]
    return plans


def _candidate_ranges(mn: Any, mx: Any) -> list[tuple[Any, Any]]:
    span = mx - mn
    candidates: list[tuple[Any, Any]] = [(mn, mx)]
    for fraction in (0.005, 0.01, 0.02, 0.03, 0.05):
        candidates.append((mn + span * fraction, mx - span * fraction))
    for fraction in (0.01, 0.02, 0.05):
        candidates.append((mn + span * fraction, mx))
    for fraction in (0.01, 0.02, 0.05):
        candidates.append((mn, mx - span * fraction))
    return candidates


def _quantize_chunk(chunk: Any, bits: int, group: int, optimize_w4: bool) -> tuple[bytes, bytes, bytes, dict[str, float]]:
    import numpy as np

    rows, columns = chunk.shape
    groups = (columns + group - 1) // group
    padded_columns = groups * group
    padded = np.zeros((rows, padded_columns), dtype=np.float32)
    padded[:, :columns] = chunk
    grouped = padded.reshape(rows, groups, group)
    mask = np.broadcast_to(
        np.arange(padded_columns, dtype=np.int64).reshape(1, groups, group) < columns,
        (rows, groups, group))
    qmax = 15.0 if bits == 4 else 255.0
    mn = np.min(np.where(mask, grouped, np.inf), axis=2)
    mx = np.max(np.where(mask, grouped, -np.inf), axis=2)

    if bits == 4 and optimize_w4:
        candidates = _candidate_ranges(mn, mx)
    else:
        candidates = [(mn, mx)]

    best_mse = np.full((rows, groups), np.inf, dtype=np.float64)
    best_q = np.zeros((rows, groups, group), dtype=np.uint8)
    best_scale = np.ones((rows, groups), dtype=np.float16)
    best_zero = np.zeros((rows, groups), dtype=np.float16)
    for lo, hi in candidates:
        span = hi - lo
        valid_range = span >= 1e-8
        raw_scale = np.where(valid_range, span / qmax, 1.0)
        scale16 = raw_scale.astype(np.float16)
        zero16 = lo.astype(np.float16)
        # Quantize against the exact values that the runtime will read back.
        scale = scale16.astype(np.float32)
        zero = zero16.astype(np.float32)
        safe_scale = np.where(scale > 0, scale, 1.0)
        q = np.clip(np.rint((grouped - zero[..., None]) / safe_scale[..., None]), 0, qmax)
        q = q.astype(np.uint8)
        q = np.where(mask, q, 0)
        recon = q.astype(np.float32) * scale[..., None] + zero[..., None]
        squared = np.where(mask, (recon - grouped) ** 2, 0.0)
        mse = squared.sum(axis=2, dtype=np.float64) / mask.sum(axis=2)
        better = mse < best_mse
        best_mse = np.where(better, mse, best_mse)
        best_q = np.where(better[..., None], q, best_q)
        best_scale = np.where(better, scale16, best_scale)
        best_zero = np.where(better, zero16, best_zero)

    best_scale_f32 = best_scale.astype(np.float32)
    best_zero_f32 = best_zero.astype(np.float32)
    best_recon = best_q.astype(np.float32) * best_scale_f32[..., None] + best_zero_f32[..., None]
    valid_original = grouped[mask]
    valid_recon = best_recon[mask]
    error = valid_recon - valid_original
    stats = {
        "sum_x2": float(np.dot(valid_original, valid_original)),
        "sum_y2": float(np.dot(valid_recon, valid_recon)),
        "sum_xy": float(np.dot(valid_original, valid_recon)),
        "sum_squared_error": float(np.dot(error, error)),
        "sum_absolute_error": float(np.abs(error).sum()),
        "max_absolute_error": float(np.abs(error).max(initial=0.0)),
        "element_count": float(error.size),
        "q_zero_count": float((best_q[mask] == 0).sum()),
        "q_max_count": float((best_q[mask] == int(qmax)).sum()),
    }

    q_valid = best_q.reshape(rows, padded_columns)[:, :columns]
    if bits == 4:
        packed_columns = (columns + 1) // 2
        packed = np.zeros((rows, packed_columns), dtype=np.uint8)
        even = q_valid[:, 0::2]
        odd = q_valid[:, 1::2]
        packed[:, :even.shape[1]] = even & 0x0F
        if odd.shape[1]:
            packed[:, :odd.shape[1]] |= (odd & 0x0F) << 4
        data = packed.tobytes()
    else:
        data = q_valid.tobytes()
    scale_data = best_scale.astype("<f2").tobytes()
    zero_data = best_zero.astype("<f2").tobytes()
    return data, scale_data, zero_data, stats


def _merge_stats(total: dict[str, float], current: dict[str, float]) -> None:
    for key, value in current.items():
        if key == "max_absolute_error":
            total[key] = max(total.get(key, 0.0), value)
        else:
            total[key] = total.get(key, 0.0) + value


def _finalize_stats(stats: dict[str, float]) -> dict[str, Any]:
    count = max(stats.get("element_count", 0.0), 1.0)
    denom = math.sqrt(max(stats.get("sum_x2", 0.0), 0.0) * max(stats.get("sum_y2", 0.0), 0.0))
    return {
        **stats,
        "element_count": int(stats.get("element_count", 0.0)),
        "q_zero_count": int(stats.get("q_zero_count", 0.0)),
        "q_max_count": int(stats.get("q_max_count", 0.0)),
        "cosine": stats.get("sum_xy", 0.0) / denom if denom else 1.0,
        "rmse": math.sqrt(stats.get("sum_squared_error", 0.0) / count),
        "relative_l2": math.sqrt(stats.get("sum_squared_error", 0.0) / max(stats.get("sum_x2", 0.0), 1e-30)),
        "mae": stats.get("sum_absolute_error", 0.0) / count,
    }


def _write_region(dst: Any, offset: int, data: bytes) -> None:
    dst.seek(offset)
    dst.write(data)


def quantize_tensor_to_file(src: Any, plan: dict[str, Any], dst: Any, group: int, optimize_w4: bool) -> dict[str, Any]:
    import numpy as np

    array = src.detach().to(dtype=__import__("torch").float32, device="cpu").numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"source tensor contains NaN/Inf: {plan['name']}")
    blob_hash = hashlib.sha256()
    crc = 0
    stats: dict[str, float] = {}
    data_cursor = int(plan["blob_offset"])

    if plan["storage_dtype"] == "fp16":
        data = array.astype("<f2").tobytes()
        blob_hash.update(data)
        crc = zlib.crc32(data)
        _write_region(dst, data_cursor, data)
    else:
        rows, columns = array.shape
        # 128 rows keeps candidate tensors bounded while using vectorized groups.
        for row_start in range(0, rows, 128):
            chunk = array[row_start:min(row_start + 128, rows), :]
            data, scale, zero, chunk_stats = _quantize_chunk(
                chunk, 4 if plan["storage_dtype"] == "w4" else 8,
                group, optimize_w4 and plan["storage_dtype"] == "w4")
            blob_hash.update(data)
            crc = zlib.crc32(data, crc)
            _write_region(dst, data_cursor, data)
            data_cursor += len(data)
            stats_before = stats
            _merge_stats(stats_before, chunk_stats)
            # Parameters are written after all packed data, so retain only their
            # small per-group arrays, never a second full quantized tensor.
            plan.setdefault("_scale_parts", bytearray()).extend(scale)
            plan.setdefault("_zero_parts", bytearray()).extend(zero)

        scale_data = bytes(plan.pop("_scale_parts"))
        zero_data = bytes(plan.pop("_zero_parts"))
        blob_hash.update(scale_data)
        blob_hash.update(zero_data)
        _write_region(dst, int(plan["blob_offset"]) + int(plan["data_size"]), scale_data)
        _write_region(dst, int(plan["blob_offset"]) + int(plan["data_size"]) + int(plan["scale_size"]), zero_data)

    result = {
        "crc32": crc & 0xFFFFFFFF,
        "blob_sha256": blob_hash.hexdigest(),
    }
    if stats:
        result["stats"] = _finalize_stats(stats)
    return result


def build_table(plans: list[dict[str, Any]]) -> bytes:
    table = bytearray()
    for plan in plans:
        name = plan["name"].encode("utf-8")[:63].ljust(64, b"\0")
        shape = list(plan["shape"])[:4] + [0] * max(0, 4 - len(plan["shape"]))
        logical = STORAGE_CODE[plan["logical_dtype"]]
        storage = STORAGE_CODE[plan["storage_dtype"]]
        data_offset = 0
        scale_offset = plan["data_size"]
        zero_offset = scale_offset + plan["scale_size"]
        table.extend(struct.pack(
            "<64sI4IBB2xQQQIIII",
            name, len(plan["shape"]), shape[0], shape[1], shape[2], shape[3],
            logical, storage, plan["numel"], plan["blob_offset"], plan["blob_size"],
            data_offset, plan["data_size"], scale_offset, zero_offset))
    return bytes(table)


def build_metadata(
    plans: list[dict[str, Any]], component: str, quant: str, group: int,
    source_path: str, source_sha: str, source_repo: str, source_revision: str,
    precision_map_sha: str | None, script_sha: str, packer_commit: str,
    numpy_version: str, torch_version: str, safetensors_version: str,
    w4_algorithm: str,
) -> dict[str, Any]:
    tensor_meta = []
    for plan in plans:
        entry = {
            "name": plan["name"], "shape": plan["shape"],
            "logical_dtype": plan["logical_dtype"], "storage_dtype": plan["storage_dtype"],
            "crc32": plan.get("crc32"), "blob_sha256": plan.get("blob_sha256"),
            "block_index": plan["block_index"], "execution_index": plan["execution_index"],
            "blob_offset": plan["blob_offset"], "blob_size": plan["blob_size"],
            "data_offset": 0, "data_size": plan["data_size"],
            "scale_offset": plan["data_size"], "scale_size": plan["scale_size"],
            "zero_offset": plan["data_size"] + plan["scale_size"], "zero_size": plan["zero_size"],
        }
        tensor_meta.append(entry)
    return {
        "component": component,
        "quant": {"scheme": quant, "group": group, "w4_algorithm": w4_algorithm},
        "tensor_meta": tensor_meta,
        "source_hashes": {os.path.basename(source_path): source_sha},
        "packer": {
            "version": 2, "script_sha256": script_sha, "git_commit": packer_commit,
            "python": sys.version.split()[0], "numpy": numpy_version,
            "torch": torch_version, "safetensors": safetensors_version,
        },
        "source": {
            "repo": source_repo, "revision": source_revision,
            "path": "split_files/diffusion_models/anima-turbo-v1.0.safetensors",
            "sha256": source_sha,
        },
        "quantization": {
            "default": quant, "group": group, "w4_algorithm": w4_algorithm,
            "scale_dtype": "fp16", "zero_dtype": "fp16",
        },
        "precision_map_sha256": precision_map_sha,
    }


def write_header_and_metadata(
    dst: Any, metadata: dict[str, Any], table: bytes,
    component: str, tensor_count: int, metadata_reserve: int,
    payload_offset: int, file_size: int,
) -> None:
    json_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(json_bytes) > metadata_reserve:
        raise RuntimeError(f"metadata is {len(json_bytes)} bytes, reserve is only {metadata_reserve}")
    table_offset = HEADER_SIZE + metadata_reserve
    header = bytearray(HEADER_SIZE)
    struct.pack_into(
        "<4sHHIQQQQQQQI", header, 0,
        MAGIC, VERSION, COMPONENT[component], ALIGN, tensor_count,
        HEADER_SIZE, len(json_bytes), table_offset, len(table),
        payload_offset, file_size, RECORD_SIZE)
    _write_region(dst, 0, bytes(header))
    _write_region(dst, HEADER_SIZE, json_bytes)
    _write_region(dst, table_offset, table)


def package(args: argparse.Namespace) -> int:
    try:
        import numpy as np
        import torch
        import safetensors
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(f"packing dependencies unavailable: {error}") from error

    header = read_safetensors_header(args.input)
    source_sha = sha256_file(args.input)
    if args.source_sha256 and source_sha != args.source_sha256:
        raise RuntimeError(f"source SHA mismatch: expected {args.source_sha256}, got {source_sha}")
    default, overrides, precision_map_sha = parse_precision_map(args.precision_map, args.quant)
    plans = make_plan(header, default, overrides, args.group, args.metadata_reserve)
    table_offset = HEADER_SIZE + args.metadata_reserve
    payload_offset = align_up(table_offset + len(plans) * RECORD_SIZE)
    file_size = payload_offset + sum(int(p["blob_size"]) for p in plans)
    print(f"plan: {len(plans)} tensors, payload={file_size - payload_offset:,} bytes, file={file_size:,} bytes", flush=True)
    if args.dry_plan:
        plan_output = {
            "component": args.component, "quant": args.quant, "group": args.group,
            "source": args.input, "source_sha256": source_sha,
            "tensor_count": len(plans), "payload_offset": payload_offset,
            "file_size": file_size,
            "storage_counts": {storage: sum(p["storage_dtype"] == storage for p in plans)
                               for storage in ("w4", "w8", "fp16")},
        }
        print(json.dumps(plan_output, indent=2, sort_keys=True))
        if args.report:
            Path(args.report).write_text(json.dumps(plan_output, indent=2, sort_keys=True) + "\n")
        return 0

    started = time.monotonic()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w+b") as dst:
        dst.truncate(file_size)
        with safe_open(args.input, framework="pt", device="cpu") as source:
            names = list(source.keys())
            if names != sorted(names):
                raise RuntimeError("safetensors key order is not deterministic")
            for index, plan in enumerate(plans, start=1):
                tensor = source.get_tensor(plan["name"])
                result = quantize_tensor_to_file(
                    tensor, plan, dst, args.group,
                    args.w4_algorithm == "mseclip")
                plan.update(result)
                del tensor
                if index == 1 or index % 25 == 0 or index == len(plans):
                    print(f"tensor {index} / {len(plans)}: {plan['name']} bytes={output.stat().st_size:,} elapsed={time.monotonic() - started:.1f}s", flush=True)

        metadata = build_metadata(
            plans, args.component, args.quant, args.group, args.input, source_sha,
            args.source_repo, args.source_revision, precision_map_sha,
            sha256_file(Path(__file__)), os.environ.get("GITHUB_SHA", "local"),
            np.__version__, torch.__version__, getattr(safetensors, "__version__", "unknown"),
            args.w4_algorithm)
        table = build_table(plans)
        write_header_and_metadata(
            dst, metadata, table, args.component, len(plans), args.metadata_reserve,
            payload_offset, file_size)
        dst.flush()
        os.fsync(dst.fileno())

    summary_stats: dict[str, float] = {}
    tensor_reports = []
    for plan in plans:
        entry = {k: plan[k] for k in ("name", "shape", "storage_dtype", "data_size", "scale_size", "zero_size", "blob_size", "crc32", "blob_sha256")}
        if "stats" in plan:
            entry.update(plan["stats"])
            _merge_stats(summary_stats, {
                k: v for k, v in plan["stats"].items()
                if isinstance(v, (int, float)) and (
                    k.startswith("sum_") or
                    k in ("element_count", "q_zero_count", "q_max_count", "max_absolute_error")
                )
            })
        tensor_reports.append(entry)
    report = {
        "format": "ANMA-v1-refinement-v2",
        "component": args.component, "default_quant": args.quant, "group": args.group,
        "w4_algorithm": args.w4_algorithm, "source_sha256": source_sha,
        "source_revision": args.source_revision, "tensor_count": len(plans),
        "output": {"filename": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "summary": _finalize_stats(summary_stats) if summary_stats else {},
        "tensors": tensor_reports,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}: {output.stat().st_size:,} bytes; sha256={report['output']['sha256']}", flush=True)
    if args.verify:
        from verify_animapk import verify_file
        verification = verify_file(str(output))
        if not verification["ok"]:
            raise RuntimeError("self-verification failed: " + "; ".join(verification["errors"]))
        print("verify: PASS", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=COMPONENT)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quant", required=True, choices=("w4", "w8"))
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--w4-algorithm", choices=("mseclip", "minmax"), default="mseclip")
    parser.add_argument("--precision-map")
    parser.add_argument("--report")
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--metadata-reserve", type=int, default=DEFAULT_METADATA_RESERVE)
    parser.add_argument("--source-repo", default=os.environ.get("ANIMA_SOURCE_REPO", "circlestone-labs/Anima"))
    parser.add_argument("--source-revision", default=os.environ.get("ANIMA_SOURCE_REVISION", "unknown"))
    parser.add_argument("--source-sha256", default=os.environ.get("ANIMA_SOURCE_SHA256"))
    args = parser.parse_args()
    try:
        return package(args)
    except Exception as error:
        print(f"pack_anima.py: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
