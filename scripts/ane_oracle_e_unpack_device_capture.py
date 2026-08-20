#!/usr/bin/env python3
"""Unpack an AnimaXS `.oraclee` A12 capture into portable parity inputs.

The Swift producer writes one bounded single-file bundle. This tool validates
all offsets before touching payload bytes, extracts the exact device initial
latent, cross-context, step-0 prepared residual/embedding/AdaLN-LoRA state,
and all 84 deterministic branch samples.

Physical XS Max captures are the app's fixed 512x512 lane: initial latent
[1,16,64,64] and prepared DiT residual [1024,2048]. When requested, this tool
writes `initial_noise.safetensors` directly in the captured latent shape; it
must NOT reshape those values through Oracle V2's 1024x1024 noise template.
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
DEVICE_WIDTH = 512
DEVICE_HEIGHT = 512
EXPECTED_PAYLOAD_SHAPES = {
    "initial_latent": [1, 16, 64, 64],
    "cross_context": [1, 512, 1024],
    "prepared_residual": [1024, 2048],
    "prepared_embedding": [2048],
    "prepared_adaln_lora": [6144],
}
EXPECTED_PAYLOAD_BYTES = {
    name: int(np.prod(shape, dtype=np.int64)) * 4
    for name, shape in EXPECTED_PAYLOAD_SHAPES.items()
}
BLOCK0_STAGE_PAYLOADS = {
    "stage_b00_self_modulation": ("float32-le", [6144]),
    "stage_b00_self_projection_input": ("float16-le", [1024, 2048]),
    "stage_b00_self_q_raw": ("float16-le", [1024, 2048]),
    "stage_b00_self_k_raw": ("float16-le", [1024, 2048]),
    "stage_b00_self_v_raw": ("float16-le", [1024, 2048]),
    "stage_b00_self_q_post_rope": ("float16-le", [1024, 2048]),
    "stage_b00_self_k_post_rope": ("float16-le", [1024, 2048]),
    "stage_b00_self_attended": ("float16-le", [1024, 2048]),
    "stage_b00_self_branch": ("float16-le", [1024, 2048]),
}
DTYPE_INFO = {
    "float32-le": (4, "f32"),
    "float16-le": (2, "f16"),
}


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
        element_count = int(item["elementCount"])
        shape = [int(x) for x in item["shape"]]
        blob = _slice(
            raw,
            offset=int(item["offset"]),
            byte_count=byte_count,
            manifest_offset=manifest_offset,
            label=f"payload {name}",
        )
        dtype = str(item.get("dtype"))
        if dtype not in DTYPE_INFO:
            raise ValueError(f"unsupported payload dtype for {name}: {dtype}")
        bytes_per_element, extension = DTYPE_INFO[dtype]
        if byte_count != element_count * bytes_per_element:
            raise ValueError(f"unsupported payload dtype/size for {name}")
        if not shape or any(x <= 0 for x in shape) or int(np.prod(shape, dtype=np.int64)) != element_count:
            raise ValueError(f"payload shape/element mismatch for {name}")
        filename = f"{name}.{extension}"
        (out_dir / filename).write_bytes(blob)
        record = dict(item)
        record["file"] = filename
        payload_records.append(record)

    required = set(EXPECTED_PAYLOAD_BYTES)
    if not required.issubset(names):
        raise ValueError(f"missing device payloads: {sorted(required - names)}")
    by_name = {item["name"]: item for item in payload_records}
    for name, expected_bytes in EXPECTED_PAYLOAD_BYTES.items():
        actual_bytes = int(by_name[name]["byteCount"])
        actual_shape = [int(x) for x in by_name[name]["shape"]]
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"device payload {name} has {actual_bytes} bytes; expected {expected_bytes}")
        expected_shape = EXPECTED_PAYLOAD_SHAPES[name]
        if actual_shape != expected_shape:
            raise ValueError(
                f"device payload {name} has shape {actual_shape}; expected {expected_shape}")

    expected_stage_count = int(manifest.get("expectedBlock0StagePayloads", 0))
    present_stage_names = set(BLOCK0_STAGE_PAYLOADS).intersection(names)
    if expected_stage_count:
        missing_stages = set(BLOCK0_STAGE_PAYLOADS) - present_stage_names
        if expected_stage_count != len(BLOCK0_STAGE_PAYLOADS) or missing_stages:
            raise ValueError(f"incomplete block-0 stage payloads: missing {sorted(missing_stages)}")
    for name in present_stage_names:
        expected_dtype, expected_shape = BLOCK0_STAGE_PAYLOADS[name]
        item = by_name[name]
        actual_dtype = str(item["dtype"])
        actual_shape = [int(x) for x in item["shape"]]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * DTYPE_INFO[expected_dtype][0]
        if actual_dtype != expected_dtype or actual_shape != expected_shape or int(item["byteCount"]) != expected_bytes:
            raise ValueError(f"bad block-0 stage payload contract for {name}")

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
        "prepared_state_source": manifest.get("preparedStateSource"),
        "generation_width": DEVICE_WIDTH,
        "generation_height": DEVICE_HEIGHT,
        "latent_shape": EXPECTED_PAYLOAD_SHAPES["initial_latent"],
        "dit_token_count": EXPECTED_PAYLOAD_SHAPES["prepared_residual"][0],
        "payloads": payload_records,
        "expected_step0_checkpoints": EXPECTED_CHECKPOINTS,
        "completed_step0_checkpoints": len(checkpoint_records),
        "expected_block0_stage_payloads": expected_stage_count,
        "completed_block0_stage_payloads": len(present_stage_names),
        "block0_stage_payloads": [by_name[name] for name in sorted(present_stage_names)],
        "records": checkpoint_records,
    }
    (out_dir / "device_manifest.json").write_text(
        json.dumps(portable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return portable


def write_noise_safetensors(out_dir: Path, portable: dict[str, Any]) -> Path:
    """Write the captured A12 latent in its native 512x512 graph shape."""
    # Import lazily so capture validation/unpacking itself only needs NumPy.
    import torch
    from safetensors.torch import save_file

    shape = tuple(int(x) for x in portable["latent_shape"])
    if list(shape) != EXPECTED_PAYLOAD_SHAPES["initial_latent"]:
        raise ValueError(f"unexpected device latent shape for noise export: {shape}")
    values = np.fromfile(out_dir / "initial_latent.f32", dtype="<f4")
    element_count = int(np.prod(shape, dtype=np.int64))
    if values.size != element_count:
        raise ValueError(
            f"device latent has {values.size} values but captured shape {shape} needs {element_count}")
    noise = torch.from_numpy(values.copy()).reshape(shape).contiguous()
    destination = out_dir / "initial_noise.safetensors"
    save_file({"noise": noise}, str(destination))
    return destination


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--write-noise-safetensors",
        action="store_true",
        help="write captured [1,16,64,64] latent as initial_noise.safetensors for the 512x512 CUDA parity graph",
    )
    args = ap.parse_args()

    portable = unpack_capture(args.capture, args.out_dir)
    noise_path = None
    if args.write_noise_safetensors:
        noise_path = write_noise_safetensors(args.out_dir, portable)
    print(json.dumps({
        "status": portable["status"],
        "checkpoints": portable["completed_step0_checkpoints"],
        "generation_width": portable["generation_width"],
        "generation_height": portable["generation_height"],
        "dit_token_count": portable["dit_token_count"],
        "block0_stage_payloads": portable["completed_block0_stage_payloads"],
        "device_manifest": str(args.out_dir / "device_manifest.json"),
        "noise_safetensors": str(noise_path) if noise_path else None,
        "cross_context": str(args.out_dir / "cross_context.f32"),
        "prepared_residual": str(args.out_dir / "prepared_residual.f32"),
        "prepared_embedding": str(args.out_dir / "prepared_embedding.f32"),
        "prepared_adaln_lora": str(args.out_dir / "prepared_adaln_lora.f32"),
    }, indent=2))


if __name__ == "__main__":
    main()
