#!/usr/bin/env python3
"""Extract block-0 source or exact ANMA W8 weights for dit_block0_oracle.py.

This is intentionally a bounded, one-block utility for GitHub Actions. It does
not materialize the full model and preserves the pack's fp16 scale/zero decode.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
from pathlib import Path

import numpy as np

MAPPING = {
    "adaln_modulation_self_attn.1.weight": "mod_self_w1",
    "adaln_modulation_self_attn.2.weight": "mod_self_w2",
    "adaln_modulation_cross_attn.1.weight": "mod_cross_w1",
    "adaln_modulation_cross_attn.2.weight": "mod_cross_w2",
    "adaln_modulation_mlp.1.weight": "mod_mlp_w1",
    "adaln_modulation_mlp.2.weight": "mod_mlp_w2",
    "self_attn.q_proj.weight": "self_q",
    "self_attn.k_proj.weight": "self_k",
    "self_attn.v_proj.weight": "self_v",
    "self_attn.output_proj.weight": "self_o",
    "self_attn.q_norm.weight": "self_q_norm",
    "self_attn.k_norm.weight": "self_k_norm",
    "cross_attn.q_proj.weight": "cross_q",
    "cross_attn.k_proj.weight": "cross_k",
    "cross_attn.v_proj.weight": "cross_v",
    "cross_attn.output_proj.weight": "cross_o",
    "cross_attn.q_norm.weight": "cross_q_norm",
    "cross_attn.k_norm.weight": "cross_k_norm",
    "mlp.layer1.weight": "mlp_w1",
    "mlp.layer2.weight": "mlp_w2",
}


def write(array: np.ndarray, name: str, output: Path) -> None:
    # Own the storage so no final local can keep the pack mmap exported.
    value = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite tensor {name}")
    value.tofile(output / f"block0_{MAPPING[name]}.f32")


def extract_source(path: str, output: Path, block: int) -> None:
    prefix = f"model.diffusion_model.blocks.{block}."
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as source:
        for suffix in MAPPING:
            full = prefix + suffix
            if full not in source.keys():
                raise KeyError(full)
            write(source.get_tensor(full).float().numpy(), suffix, output)


def extract_pack(path: str, output: Path, block: int) -> None:
    prefix = f"model.diffusion_model.blocks.{block}."
    with open(path, "rb") as handle:
        blob = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if blob[:4] != b"ANMA":
                raise ValueError("bad ANMA magic")
            json_offset = struct.unpack_from("<Q", blob, 20)[0]
            json_size = struct.unpack_from("<Q", blob, 28)[0]
            metadata = json.loads(blob[json_offset:json_offset + json_size])
            items = {item["name"]: item for item in metadata["tensor_meta"]}
            group = int(metadata["quant"]["group"])
            if group != 64:
                raise ValueError(f"unexpected group {group}")
            for suffix in MAPPING:
                full = prefix + suffix
                item = items[full]
                base = int(item["blob_offset"])
                shape = tuple(int(x) for x in item["shape"])
                storage = item["storage_dtype"]
                data_start = base + int(item["data_offset"])
                data_size = int(item["data_size"])
                if storage == "fp16":
                    value = np.frombuffer(blob, dtype="<f2", count=data_size // 2,
                                          offset=data_start).astype(np.float32).reshape(shape)
                elif storage == "w8" and len(shape) == 2:
                    rows, columns = shape
                    groups = (columns + group - 1) // group
                    quantized = np.frombuffer(blob, dtype=np.uint8, count=data_size,
                                              offset=data_start).reshape(rows, columns)
                    scale = np.frombuffer(
                        blob, dtype="<f2", count=rows * groups,
                        offset=base + int(item["scale_offset"])).astype(np.float32).reshape(rows, groups)
                    zero = np.frombuffer(
                        blob, dtype="<f2", count=rows * groups,
                        offset=base + int(item["zero_offset"])).astype(np.float32).reshape(rows, groups)
                    group_index = np.arange(columns) // group
                    value = quantized.astype(np.float32) * scale[:, group_index] + zero[:, group_index]
                else:
                    raise ValueError(f"unsupported {full}: {storage} {shape}")
                write(value, suffix, output)
                del value
                if storage == "w8":
                    del quantized, scale, zero
        finally:
            blob.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack")
    parser.add_argument("--source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--swift-projection-rounding", action="store_true")
    args = parser.parse_args()
    if bool(args.pack) == bool(args.source):
        parser.error("provide exactly one of --pack or --source")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.pack:
        extract_pack(args.pack, output, args.block)
    else:
        extract_source(args.source, output, args.block)
    if args.swift_projection_rounding:
        for suffix, short in MAPPING.items():
            if suffix.startswith("adaln_modulation_") or suffix.endswith("_norm.weight"):
                continue
            target = output / f"block0_{short}.f32"
            values = np.fromfile(target, dtype=np.float32)
            values.astype(np.float16).astype(np.float32).tofile(target)
    print(json.dumps({"output": str(output), "block": args.block, "tensors": len(MAPPING),
                      "swift_projection_rounding": args.swift_projection_rounding}, sort_keys=True))


if __name__ == "__main__":
    main()
