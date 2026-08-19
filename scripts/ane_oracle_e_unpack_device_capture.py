#!/usr/bin/env python3
"""Unpack an AnimaXS `.oraclee` A12 capture into portable parity inputs.

The Swift producer writes one bounded single-file bundle.  This tool validates
all offsets before touching payload bytes, extracts the exact device initial
latent, exact device cross-context, and 84 deterministic branch samples, and
optionally converts the raw initial latent into the safetensors shape expected
by the proven Oracle-V2 runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

MAGIC = b"AXOECAP1"
VERSION = 1
HEADER_BYTES = 64
EXPECTED_CHECKPOINTS = 84
EXPECTED_CROSS_BYTES = 512 * 1024 * 4
EXPECTED_LATENT_BYTES = 16 * 64 * 64 * 4


def _load_capture(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) < HEADER_BYTES:
        raise ValueError("Oracle E capture is shorter than its 64-byte header")
    if raw[:8] != MAGIC:
        raise ValueError(f"bad Oracle E magic: {raw[:8]!r}")
    version = struct.unpack_from("<I", raw, 8)[0]
    if version != VERSION:
        raise ValueError(f"unsupported Oracle E capture version: {version}")
    manifest_offset = struct.unpack_from("<Q", raw, 16)[0]
    manifest_length = struct.unpack_from("<Q", raw, 24)[0]
    if manifest_offset < HEADER_BYTES:
        raise ValueError("manifest overlaps Oracle E header")
    if manifest_offset > len(raw) or manifest_length > len(raw) - manifest_offset:
        raise ValueError("manifest range is outside Oracle E capture")
    manifest_blob = raw[manifest_offset:manifest_offset + manifest_length]
    manifest = json.loads(manifest_blob.decode("utf-8"))
    if int(manifest.get("schema", -1)) != 1:
        raise ValueError(f"unsupported Oracle E manifest schema: {manifest.get('schema')!r}")
    return manifest, raw


def _slice(raw: bytes, *, offset: int, byte_count: int, manifest_offset: int, label: str) -> bytes:
    if offset < HEADER_BYTES or byte_count < 0:
        raise ValueError(f"invalid {label} payload range")
    end = offset + byte_count
    if end < offset or end > manifest_offset:
        raise ValueError(f"{label} payload is outside the raw-payload region")
    return raw[offset:end]


def unpack_capture(capture: Path, out_dir: Path) -> dict[str, Any]:
    manifest, raw = _load_capture(capture)
    manifest_offset = struct.unpack_from("<Q", raw, 16)[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_records: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in manifest.get("payloads", []):
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate Oracle E payload name: {name}")
        names.add(name)
        byte_count = int(item["byteCount"])
        blob = _slice(
            raw,
            offset=int(item["offset"]),
            byte_count=byte_count,
            manifest_offset=manifest_offset,
            label=f"payload {name}",
        )
        if item.get("dtype") != "float32-le" or byte_count % 4:
            raise ValueError(f"unsupported payload dtype/size for {name}")
        filename = f"{name}.f32"
        (out_dir / filename).write_bytes(blob)
        record = dict(item)
        record["file"] = filename
        payload_records.append(record)

    required = {"initial_latent", "cross_context"}
    if not required.issubset(names):
        raise ValueError(f"missing device payloads: {sorted(required - names)}")
    by_name = {item["name"]: item for item in payload_records}
    if int(by_name["initial_latent"]["byteCount"]) != EXPECTED_LATENT_BYTES:
        raise ValueError("device initial latent has unexpected byte length")
    if int(by_name["cross_context"]["byteCount"]) != EXPECTED_CROSS_BYTES:
        raise ValueError("device cross-context has unexpected byte length")

    checkpoint_records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    branch_order = {"self": 0, "cross": 1, "mlp": 2}
    for item in manifest.get("checkpoints", []):
        step = int(item["step"])
        block = int(item["block"])
        branch = str(item["branch"])
        key = (step, block, branch)
        if key in seen:
            raise ValueError(f"duplicate device checkpoint: {key}")
        seen.add(key)
        if step != 0 or block not in range(28) or branch not in branch_order:
            raise ValueError(f"invalid device checkpoint key: {key}")
        byte_count = int(item["byteCount"])
        sample_count = int(item["sampleCount"])
        if item.get("sampleDtype") != "float32-le" or byte_count != sample_count * 4:
            raise ValueError(f"bad sample encoding for device checkpoint {key}")
        blob = _slice(
            raw,
            offset=int(item["offset"]),
            byte_count=byte_count,
            manifest_offset=manifest_offset,
            label=f"checkpoint {key}",
        )
        filename = f"step00_block{block:02d}_{branch}.f32"
        (out_dir / filename).write_bytes(blob)
        record = dict(item)
        record["sample_file"] = filename
        record["sample_count"] = sample_count
        record["sample_stride"] = int(item["sampleStride"])
        checkpoint_records.append(record)

    checkpoint_records.sort(key=lambda r: (int(r["block"]), branch_order[str(r["branch"])]))
    completed = int(manifest.get("completedStep0Checkpoints", len(checkpoint_records)))
    status = str(manifest.get("status", "unknown"))
    if status == "completed":
        if completed != EXPECTED_CHECKPOINTS or len(checkpoint_records) != EXPECTED_CHECKPOINTS:
            raise ValueError(
                f"completed device capture is incomplete: manifest={completed}, files={len(checkpoint_records)}")

    portable = {
        "schema": 1,
        "source_capture": str(capture.resolve()),
        "status": status,
        "error": manifest.get("error"),
        "seed": int(manifest["seed"]),
        "dit_variant_id": manifest["ditVariantID"],
        "linear_backend": manifest["linearBackend"],
        "ping_pong_weight_streaming": bool(manifest["pingPongWeightStreaming"]),
        "conditioning_source": manifest["conditioningSource"],
        "initial_latent_source": manifest["initialLatentSource"],
        "payloads": payload_records,
        "expected_step0_checkpoints": EXPECTED_CHECKPOINTS,
        "completed_step0_checkpoints": len(checkpoint_records),
        "records": checkpoint_records,
    }
    (out_dir / "device_manifest.json").write_text(
        json.dumps(portable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return portable


def write_noise_safetensors(out_dir: Path, template: Path) -> Path:
    # Import lazily so capture validation/unpacking itself only needs NumPy.
    import torch
    from safetensors.torch import load_file, save_file

    template_data = load_file(str(template), device="cpu")
    if "noise" not in template_data:
        raise ValueError(f"noise template {template} does not contain tensor 'noise'")
    shape = tuple(template_data["noise"].shape)
    element_count = int(np.prod(shape, dtype=np.int64))
    values = np.fromfile(out_dir / "initial_latent.f32", dtype="<f4")
    if values.size != element_count:
        raise ValueError(
            f"device latent has {values.size} values but noise template shape {shape} needs {element_count}")
    noise = torch.from_numpy(values.copy()).reshape(shape).contiguous()
    destination = out_dir / "initial_noise.safetensors"
    save_file({"noise": noise}, str(destination))
    return destination


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--noise-template",
        type=Path,
        help="Oracle-V2 A1 initial_noise.safetensors; writes device values in the exact tensor shape",
    )
    args = ap.parse_args()

    portable = unpack_capture(args.capture, args.out_dir)
    noise_path = None
    if args.noise_template is not None:
        noise_path = write_noise_safetensors(args.out_dir, args.noise_template)
    print(json.dumps({
        "status": portable["status"],
        "checkpoints": portable["completed_step0_checkpoints"],
        "device_manifest": str(args.out_dir / "device_manifest.json"),
        "noise_safetensors": str(noise_path) if noise_path else None,
        "cross_context": str(args.out_dir / "cross_context.f32"),
    }, indent=2))


if __name__ == "__main__":
    main()
