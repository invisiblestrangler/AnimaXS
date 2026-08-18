#!/usr/bin/env python3
"""Independent offline verifier for ANMA v1 packs."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

MAGIC = b"ANMA"
ALIGN = 16_384
HEADER_SIZE = 256
RECORD_SIZE = 128
ANE_QUANT_SCHEME = "w8-ane-hybrid-v1"
ANE_TENSOR_FORMAT = "ane_u8_per_row_fp32_v1"
GROUP64_TENSOR_FORMAT = "group64_affine_fp16_v2"
FP16_TENSOR_FORMAT = "fp16"
ANE_PROJECTION_SUFFIXES = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.k_proj.weight",
    "cross_attn.v_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
}
_BLOCK_RE = re.compile(r"^model\.diffusion_model\.blocks\.(\d+)\.(.+)$")


def u16(blob: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<H", blob, offset)[0]


def u32(blob: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<I", blob, offset)[0]


def u64(blob: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", blob, offset)[0]


def expected_ane_names() -> set[str]:
    return {
        f"model.diffusion_model.blocks.{block}.{suffix}"
        for block in range(28)
        for suffix in ANE_PROJECTION_SUFFIXES
    }


def verify_file(path: str) -> dict[str, Any]:
    errors: list[str] = []
    tensor_count = 0
    sha_count = 0
    crc_count = 0
    native_names: set[str] = set()
    metal_only_block_layout = True
    actual_size = os.path.getsize(path)
    with open(path, "rb") as source:
        blob = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if actual_size < HEADER_SIZE:
                return {"ok": False, "errors": ["file is smaller than ANMA header"]}
            if blob[:4] != MAGIC:
                errors.append("bad magic")
            version = u16(blob, 4)
            component_code = u16(blob, 6)
            alignment = u32(blob, 8)
            tensor_count = u64(blob, 12)
            json_offset, json_size = u64(blob, 20), u64(blob, 28)
            table_offset, table_bytes = u64(blob, 36), u64(blob, 44)
            payload_offset, declared_size = u64(blob, 52), u64(blob, 60)
            record_size = u32(blob, 68)
            if version != 1: errors.append(f"version {version} != 1")
            if component_code not in (1, 2, 3): errors.append(f"invalid component code {component_code}")
            if alignment != ALIGN: errors.append(f"alignment {alignment} != {ALIGN}")
            if record_size != RECORD_SIZE: errors.append(f"record size {record_size} != {RECORD_SIZE}")
            if declared_size != actual_size: errors.append(f"declared file size {declared_size} != actual {actual_size}")
            for label, offset, size in (("json", json_offset, json_size), ("table", table_offset, table_bytes)):
                if offset > actual_size or size > actual_size - offset:
                    errors.append(f"{label} section out of bounds")
            if payload_offset > actual_size: errors.append("payload offset out of bounds")
            if json_offset + json_size > actual_size:
                return {"ok": False, "errors": errors + ["cannot parse out-of-bounds JSON"]}
            try:
                metadata = json.loads(bytes(blob[json_offset:json_offset + json_size]))
            except Exception as error:
                return {"ok": False, "errors": errors + [f"JSON parse failed: {error}"]}
            tensors = metadata.get("tensor_meta", [])
            quant = metadata.get("quant") or {}
            scheme = quant.get("scheme")
            is_hybrid = scheme == ANE_QUANT_SCHEME
            if is_hybrid:
                if metadata.get("profile") != "ane-hybrid-w8-v1":
                    errors.append("ANE hybrid pack is missing profile=ane-hybrid-w8-v1")
                if component_code != 1 or metadata.get("component") != "dit":
                    errors.append("ANE hybrid scheme is only valid for a DiT pack")
                if int(quant.get("group") or 0) != 64:
                    errors.append("ANE hybrid pack requires group=64 for non-native W8 tensors")
            if len(tensors) != tensor_count:
                errors.append(f"JSON tensor count {len(tensors)} != header {tensor_count}")
            if table_bytes != tensor_count * RECORD_SIZE:
                errors.append("table size does not match tensor count")
            by_offset = {int(item.get("blob_offset", -1)): item for item in tensors}
            if len(by_offset) != len(tensors):
                errors.append("duplicate JSON blob_offset")
            seen: list[tuple[int, int, str]] = []
            group = int(quant.get("group") or 64)
            table_records = min(
                int(tensor_count),
                (actual_size - table_offset) // RECORD_SIZE if table_offset <= actual_size else 0)
            for index in range(table_records):
                offset = int(table_offset) + index * RECORD_SIZE
                if offset + RECORD_SIZE > actual_size:
                    break
                table_blob_offset = u64(blob, offset + 96)
                if table_blob_offset not in by_offset:
                    errors.append(f"table blob offset {table_blob_offset} is missing from JSON")
                    continue
                item = by_offset[table_blob_offset]
                name = str(item.get("name", "<unnamed>"))
                shape = [int(x) for x in item.get("shape", [])]
                storage = item.get("storage_dtype")
                quant_format = item.get("quantization_format")
                blob_size = int(item.get("blob_size", 0))
                data_size = int(item.get("data_size", 0))
                scale_size = int(item.get("scale_size") or 0)
                zero_size = int(item.get("zero_size") or 0)
                data_offset = int(item.get("data_offset") or 0)
                scale_offset = int(item.get("scale_offset") or 0)
                zero_offset = int(item.get("zero_offset") or 0)
                if table_blob_offset % ALIGN: errors.append(f"{name}: blob is not {ALIGN}-aligned")
                if table_blob_offset + blob_size > actual_size: errors.append(f"{name}: blob is out of bounds")
                if data_offset + data_size > blob_size or scale_offset + scale_size > blob_size or zero_offset + zero_size > blob_size:
                    errors.append(f"{name}: subregion out of bounds")
                    continue

                native = quant_format == ANE_TENSOR_FORMAT
                if native:
                    native_names.add(name)
                    if not is_hybrid:
                        errors.append(f"{name}: ANE-native tensor appears outside ANE hybrid scheme")
                    if storage != "w8" or len(shape) != 2:
                        errors.append(f"{name}: ANE-native tensor must be rank-2 w8")
                    else:
                        rows, columns = shape
                        if data_size != rows * columns:
                            errors.append(f"{name}: ANE data size {data_size} != {rows * columns}")
                        if scale_size != rows * 4 or zero_size != rows * 4:
                            errors.append(f"{name}: ANE FP32 row scale/bias size mismatch")
                elif storage in ("w4", "w8") and len(shape) == 2:
                    rows, columns = shape
                    expected_data = rows * ((columns + 1) // 2 if storage == "w4" else columns)
                    expected_params = rows * ((columns + group - 1) // group) * 2
                    if data_size != expected_data: errors.append(f"{name}: data size {data_size} != {expected_data}")
                    if scale_size != expected_params or zero_size != expected_params:
                        errors.append(f"{name}: scale/zero size mismatch")
                    if is_hybrid and quant_format != GROUP64_TENSOR_FORMAT:
                        errors.append(f"{name}: hybrid ordinary quantized tensor missing {GROUP64_TENSOR_FORMAT}")
                elif storage == "fp16":
                    expected_data = 1
                    for dimension in shape: expected_data *= dimension
                    expected_data *= 2
                    if data_size != expected_data: errors.append(f"{name}: fp16 data size mismatch")
                    if scale_size or zero_size: errors.append(f"{name}: fp16 tensor has quant params")
                    if is_hybrid and quant_format != FP16_TENSOR_FORMAT:
                        errors.append(f"{name}: hybrid fp16 tensor missing {FP16_TENSOR_FORMAT}")
                elif storage == "fp32":
                    expected_data = 1
                    for dimension in shape: expected_data *= dimension
                    expected_data *= 4
                    if data_size != expected_data: errors.append(f"{name}: fp32 data size mismatch")
                else:
                    errors.append(f"{name}: unsupported storage/shape {storage}/{shape}")

                data_start = table_blob_offset + data_offset
                data_end = data_start + data_size
                scale_start = table_blob_offset + scale_offset
                scale_end = scale_start + scale_size
                zero_start = table_blob_offset + zero_offset
                zero_end = zero_start + zero_size
                data = bytes(blob[data_start:data_end])
                scale = bytes(blob[scale_start:scale_end])
                zero = bytes(blob[zero_start:zero_end])
                crc = zlib.crc32(data) & 0xFFFFFFFF
                crc_count += 1
                if item.get("crc32") is not None and int(item["crc32"]) != crc:
                    errors.append(f"{name}: CRC32 mismatch")
                digest = hashlib.sha256(data + scale + zero).hexdigest()
                if item.get("blob_sha256"):
                    sha_count += 1
                    if item["blob_sha256"] != digest: errors.append(f"{name}: blob SHA256 mismatch")
                elif is_hybrid and native:
                    errors.append(f"{name}: ANE-native tensor requires blob_sha256")

                if scale_size:
                    import numpy as np
                    if native:
                        scales = np.frombuffer(scale, dtype="<f4")
                        biases = np.frombuffer(zero, dtype="<f4")
                        if not np.isfinite(scales).all(): errors.append(f"{name}: ANE scale has NaN/Inf")
                        if not np.isfinite(biases).all(): errors.append(f"{name}: ANE bias has NaN/Inf")
                        if not (scales > 0).all(): errors.append(f"{name}: ANE scale must be > 0")
                    else:
                        if not np.isfinite(np.frombuffer(scale, dtype="<f2").astype("<f4")).all(): errors.append(f"{name}: scale has NaN/Inf")
                        if not np.isfinite(np.frombuffer(zero, dtype="<f2").astype("<f4")).all(): errors.append(f"{name}: zero has NaN/Inf")
                seen.append((table_blob_offset, table_blob_offset + blob_size, name))

            ordered = sorted(seen)
            for left, right in zip(ordered, ordered[1:]):
                if left[1] > right[0]: errors.append(f"blob overlap: {left[2]} / {right[2]}")

            if is_hybrid:
                expected = expected_ane_names()
                missing = sorted(expected - native_names)
                extra = sorted(native_names - expected)
                if len(native_names) != 280:
                    errors.append(f"ANE-native tensor count {len(native_names)} != 280")
                if missing:
                    errors.append(f"missing ANE-native tensors: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
                if extra:
                    errors.append(f"unexpected ANE-native tensors: {', '.join(extra[:5])}" + (" ..." if len(extra) > 5 else ""))
                for block in range(28):
                    prefix = f"model.diffusion_model.blocks.{block}."
                    block_items = [item for item in tensors if str(item.get("name", "")).startswith(prefix)]
                    metal = [item for item in block_items if item.get("quantization_format") != ANE_TENSOR_FORMAT]
                    native_items = [item for item in block_items if item.get("quantization_format") == ANE_TENSOR_FORMAT]
                    if len(native_items) != 10 or not metal:
                        metal_only_block_layout = False
                        errors.append(f"block {block}: expected 10 ANE tensors and non-empty Metal subset")
                        continue
                    metal_end = max(int(item["blob_offset"]) + int(item["blob_size"]) for item in metal)
                    ane_start = min(int(item["blob_offset"]) for item in native_items)
                    if ane_start < metal_end:
                        metal_only_block_layout = False
                        errors.append(f"block {block}: ANE blob interleaves Metal-only interval")

            return {
                "ok": not errors, "errors": errors, "path": path,
                "bytes": actual_size, "tensor_count": len(tensors),
                "sha256_entries": sha_count, "crc_entries": crc_count,
                "quant_scheme": scheme,
                "ane_native_tensor_count": len(native_names),
                "metal_only_block_layout": metal_only_block_layout if is_hybrid else None,
                "source": metadata.get("source"), "packer": metadata.get("packer"),
            }
        finally:
            blob.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = verify_file(args.path)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
