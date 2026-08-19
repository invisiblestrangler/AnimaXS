#!/usr/bin/env python3
"""Exact ANIMAPK arithmetic helpers for the ANE Oracle E controls.

This module intentionally does *not* model undocumented ANE instructions.  It
models the public contract AnimaXS itself supplies to the private runtime:

* native projection weights are unsigned U8 with one FP32 scale and FP32 bias
  per output row (``ane_u8_per_row_fp32_v1``);
* the mathematical reconstructed row is ``q * scale + bias``;
* ANE activation surfaces are FP16;
* residual checkpoints use a deterministic bounded sample so CUDA and A12 can
  be compared without exporting ~672 MiB of full step-0 residuals.

The helpers are shared by the external CUDA/ComfyUI oracle and by CPU unit
checks.  Keeping the ANIMAPK parser here prevents Oracle E from silently using
source weights or a second quantizer implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import mmap
from pathlib import Path
import struct
from typing import Any

import numpy as np

MAGIC = b"ANMA"
HEADER_SIZE = 256
ANE_QUANT_SCHEME = "w8-ane-hybrid-v1"
ANE_TENSOR_FORMAT = "ane_u8_per_row_fp32_v1"
GROUP64_TENSOR_FORMAT = "group64_affine_fp16_v2"
FP16_TENSOR_FORMAT = "fp16"
DEFAULT_SAMPLE_COUNT = 65_536


def _u64(blob: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", blob, offset)[0]


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    storage_dtype: str
    quantization_format: str | None
    blob_offset: int
    blob_size: int
    data_offset: int
    data_size: int
    scale_offset: int
    scale_size: int
    zero_offset: int
    zero_size: int

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "TensorRecord":
        return cls(
            name=str(item["name"]),
            shape=tuple(int(x) for x in item["shape"]),
            storage_dtype=str(item["storage_dtype"]),
            quantization_format=item.get("quantization_format"),
            blob_offset=int(item["blob_offset"]),
            blob_size=int(item["blob_size"]),
            data_offset=int(item.get("data_offset") or 0),
            data_size=int(item.get("data_size") or 0),
            scale_offset=int(item.get("scale_offset") or 0),
            scale_size=int(item.get("scale_size") or 0),
            zero_offset=int(item.get("zero_offset") or 0),
            zero_size=int(item.get("zero_size") or 0),
        )


class AnimapkReference:
    """Read-only, mmap-backed ANIMAPK reader for Oracle arithmetic."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._blob = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def _parse(self) -> None:
        if len(self._blob) < HEADER_SIZE or self._blob[:4] != MAGIC:
            raise ValueError("not an ANMA v1 pack")
        json_offset = _u64(self._blob, 20)
        json_size = _u64(self._blob, 28)
        if json_offset > len(self._blob) or json_size > len(self._blob) - json_offset:
            raise ValueError("ANIMAPK JSON metadata is out of bounds")
        self.metadata = json.loads(bytes(self._blob[json_offset:json_offset + json_size]))
        quant = self.metadata.get("quant") or {}
        self.quant_scheme = quant.get("scheme")
        self.group = int(quant.get("group") or 64)
        records = [TensorRecord.from_json(item) for item in self.metadata.get("tensor_meta", [])]
        self.by_name = {r.name: r for r in records}
        if len(self.by_name) != len(records):
            raise ValueError("duplicate tensor names in ANIMAPK metadata")
        if self.quant_scheme != ANE_QUANT_SCHEME:
            raise ValueError(
                f"Oracle E requires {ANE_QUANT_SCHEME}, got {self.quant_scheme!r}")

    def close(self) -> None:
        blob = getattr(self, "_blob", None)
        if blob is not None:
            blob.close()
            self._blob = None  # type: ignore[assignment]
        f = getattr(self, "_file", None)
        if f is not None:
            f.close()
            self._file = None  # type: ignore[assignment]

    def __enter__(self) -> "AnimapkReference":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def record(self, name: str) -> TensorRecord:
        try:
            return self.by_name[name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in ANIMAPK: {name}") from exc

    def native_rows(
        self, name: str, row_start: int = 0, row_end: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copied ``(q_u8, scale_f32, bias_f32)`` rows for a native tensor."""
        rec = self.record(name)
        if rec.quantization_format != ANE_TENSOR_FORMAT or rec.storage_dtype != "w8":
            raise ValueError(f"{name} is not an ANE-native tensor")
        if len(rec.shape) != 2:
            raise ValueError(f"{name} is not rank-2")
        rows, cols = rec.shape
        end = rows if row_end is None else int(row_end)
        start = int(row_start)
        if not (0 <= start <= end <= rows):
            raise IndexError(f"row slice [{start}:{end}] out of range 0...{rows}")
        count = end - start

        q_base = rec.blob_offset + rec.data_offset + start * cols
        q = np.frombuffer(self._blob, dtype=np.uint8, count=count * cols, offset=q_base)
        q = q.reshape(count, cols).copy()

        scale_base = rec.blob_offset + rec.scale_offset + start * 4
        bias_base = rec.blob_offset + rec.zero_offset + start * 4
        scale = np.frombuffer(self._blob, dtype="<f4", count=count, offset=scale_base).copy()
        bias = np.frombuffer(self._blob, dtype="<f4", count=count, offset=bias_base).copy()
        if not np.isfinite(scale).all() or not np.isfinite(bias).all():
            raise ValueError(f"{name} contains non-finite ANE parameters")
        if not (scale > 0).all():
            raise ValueError(f"{name} contains non-positive ANE scale")
        return q, scale, bias

    def native_reconstructed_rows(
        self, name: str, row_start: int = 0, row_end: int | None = None
    ) -> np.ndarray:
        q, scale, bias = self.native_rows(name, row_start, row_end)
        return q.astype(np.float32) * scale[:, None] + bias[:, None]


def fp16_roundtrip_numpy(values: np.ndarray) -> np.ndarray:
    """Device-surface storage boundary, returned in FP32 for stable comparison."""
    return np.asarray(values, dtype=np.float32).astype(np.float16).astype(np.float32)


def native_linear_numpy(
    pack: AnimapkReference,
    name: str,
    x: np.ndarray,
    *,
    output_row_chunk: int = 256,
) -> np.ndarray:
    """Reference native projection with FP16 surface input/output and FP32 dot products.

    This is a mathematical control, not a claim about the private ANE's exact
    accumulator implementation.  ``x`` may have any leading dimensions; the
    final dimension must equal the packed K dimension.  The returned array is
    FP32 containing values exactly representable in FP16.
    """
    rec = pack.record(name)
    if len(rec.shape) != 2:
        raise ValueError("native linear weight must be rank-2")
    out_features, in_features = rec.shape
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.shape[-1] != in_features:
        raise ValueError(
            f"input K={x_arr.shape[-1]} does not match {name} K={in_features}")
    x_surface = fp16_roundtrip_numpy(x_arr).reshape(-1, in_features)
    out = np.empty((x_surface.shape[0], out_features), dtype=np.float32)
    for row0 in range(0, out_features, output_row_chunk):
        row1 = min(out_features, row0 + output_row_chunk)
        weight = pack.native_reconstructed_rows(name, row0, row1)
        out[:, row0:row1] = x_surface @ weight.T
    out = fp16_roundtrip_numpy(out)
    return out.reshape(*x_arr.shape[:-1], out_features)


def deterministic_stride(length: int, sample_count: int = DEFAULT_SAMPLE_COUNT) -> int:
    if length < 0 or sample_count <= 0:
        raise ValueError("invalid checkpoint sample dimensions")
    return max(1, length // sample_count) if length else 1


def checkpoint_sample_numpy(
    values: np.ndarray, sample_count: int = DEFAULT_SAMPLE_COUNT
) -> tuple[np.ndarray, int]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    stride = deterministic_stride(flat.size, sample_count)
    sample = flat[::stride][:sample_count].copy()
    return sample, stride


def checkpoint_stats_numpy(values: np.ndarray) -> dict[str, Any]:
    """Full-tensor scalar statistics used by both Oracle and device manifests."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(flat)
    finite_values = flat[finite].astype(np.float64, copy=False)
    nan_count = int(np.isnan(flat).sum())
    posinf_count = int(np.isposinf(flat).sum())
    neginf_count = int(np.isneginf(flat).sum())
    if finite_values.size:
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        mean = float(finite_values.mean())
        centered = finite_values - mean
        std = float(math.sqrt(float(np.dot(centered, centered)) / finite_values.size))
        l2 = float(math.sqrt(float(np.dot(finite_values, finite_values))))
        max_abs = float(np.abs(finite_values).max())
    else:
        minimum = maximum = mean = std = l2 = max_abs = float("nan")
    return {
        "element_count": int(flat.size),
        "finite_count": int(finite.sum()),
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "l2": l2,
        "max_abs": max_abs,
    }


def compare_samples(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"sample shape mismatch: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"relative_rmse": float("inf"), "cosine": float("nan"), "max_abs_error": float("inf")}
    diff = b - a
    denom = float(np.dot(a, a))
    relative_rmse = math.sqrt(float(np.dot(diff, diff)) / max(denom, np.finfo(np.float64).tiny))
    norm_product = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    cosine = float(np.dot(a, b)) / norm_product if norm_product else 1.0
    return {
        "relative_rmse": relative_rmse,
        "cosine": cosine,
        "max_abs_error": float(np.abs(diff).max(initial=0.0)),
    }
