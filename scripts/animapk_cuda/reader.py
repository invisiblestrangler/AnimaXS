"""mmap-based .animapk reader — byte-faithful access to ANMA v1 packs.

The JSON tensor_meta is authoritative for names/shapes/offsets (binary table
truncates names to 64 chars — D008).  Blobs are 16 KiB aligned; each blob
contains data (offset 0), fp16 scale (at data_size), fp16 zero (after scale).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from pathlib import Path

from . import format as fmt

KEY_PREFIX = "model.diffusion_model."


def _u64(blob, off):
    return struct.unpack_from("<Q", blob, off)[0]


class PackFile:
    """Random-access ANMA pack backed by an mmap."""

    def __init__(self, path: str):
        self.path = str(path)
        self.size = os.path.getsize(self.path)
        self._fh = open(self.path, "rb")
        self.blob = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = fmt.parse_header(self.blob[: fmt.HEADER_SIZE])
        if self.header["declared_size"] != self.size:
            raise ValueError(
                f"declared size {self.header['declared_size']} != actual {self.size}"
            )
        raw_json = bytes(self.blob[self.header["json_offset"] : self.header["json_offset"] + self.header["json_size"]])
        self.metadata = json.loads(raw_json)
        self.tensor_meta = self.metadata.get("tensor_meta", [])
        if len(self.tensor_meta) != self.header["tensor_count"]:
            raise ValueError("JSON tensor count != header tensor count")
        self.quant = (self.metadata.get("quant") or {}).get("group", 64)
        self._by_offset = {}
        self._by_name = {}
        for item in self.tensor_meta:
            bo = int(item["blob_offset"])
            self._by_offset[bo] = item
            name = str(item["name"])
            if name.startswith(KEY_PREFIX):
                name = name[len(KEY_PREFIX) :]
            self._by_name[name] = item
        self.pack_sha256 = self._sha256_file()

    def _sha256_file(self):
        h = hashlib.sha256()
        self.blob.seek(0)
        # mmap read in chunks
        off = 0
        while off < self.size:
            chunk = self.blob[off : off + (1 << 24)]
            h.update(chunk)
            off += len(chunk)
        return h.hexdigest()

    def close(self):
        self.blob.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- lookup -----------------------------------------------------------
    def meta_by_name(self, name: str) -> dict:
        return self._by_name[name]

    def meta_by_offset(self, off: int) -> dict:
        return self._by_offset[off]

    def tensor_names(self, stripped=True) -> list[str]:
        names = [str(m["name"]) for m in self.tensor_meta]
        if stripped:
            return [n[len(KEY_PREFIX) :] if n.startswith(KEY_PREFIX) else n for n in names]
        return names

    # -- raw regions ------------------------------------------------------
    def region(self, item: dict, kind: str) -> bytes:
        """Return data/scale/zero raw bytes for a tensor metadata entry."""
        base = int(item["blob_offset"])
        if kind == "data":
            off, size = int(item["data_offset"]), int(item["data_size"])
        elif kind == "scale":
            off, size = int(item["scale_offset"]), int(item["scale_size"])
        elif kind == "zero":
            off, size = int(item["zero_offset"]), int(item["zero_size"])
        else:
            raise ValueError(kind)
        start, end = base + off, base + off + size
        if start < 0 or end > self.size or end < start:
            raise ValueError(f"out of bounds {item['name']} {kind}")
        return bytes(self.blob[start:end])

    def blob_bytes(self, item: dict) -> bytes:
        base = int(item["blob_offset"])
        return bytes(self.blob[base : base + int(item["blob_size"])])

    # -- provenance -------------------------------------------------------
    def provenance(self) -> dict:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.pack_sha256,
            "header": self.header,
            "source": self.metadata.get("source"),
            "packer": self.metadata.get("packer"),
            "quant": self.metadata.get("quant"),
        }


def build_execution_manifest(pack: PackFile, group_by: str = "block") -> dict:
    """Build animapk_execution_manifest.json.

    Ranges follow blob order (sorted-name execution order).  For the DiT pack,
    tensors naturally group by stage: x_embedder, t_embedder, adapter,
    blocks.N.*, final_layer.*.  group_by='block' splits per blocks.N prefix so
    each transformer block is one streaming range.
    """
    metas = sorted(pack.tensor_meta, key=lambda m: int(m["blob_offset"]))
    ranges = []
    current = None
    for m in metas:
        name = str(m["name"])
        if name.startswith(KEY_PREFIX):
            name = name[len(KEY_PREFIX) :]
        if group_by == "block":
            key = name.split(".")[0]
            if name.startswith("blocks."):
                key = ".".join(name.split(".")[:2])
        else:
            key = name
        if current is None or current["key"] != key:
            current = {"key": key, "start": int(m["blob_offset"]), "end": int(m["blob_offset"]), "tensors": []}
            ranges.append(current)
        current["end"] = int(m["blob_offset"]) + int(m["blob_size"])
        current["tensors"].append(
            {
                "name": name,
                "shape": [int(x) for x in m["shape"]],
                "storage_dtype": m["storage_dtype"],
                "global_offset": int(m["blob_offset"]),
                "blob_bytes": int(m["blob_size"]),
                "data_size": int(m["data_size"]),
                "scale_size": int(m["scale_size"]),
                "zero_size": int(m["zero_size"]),
            }
        )
    for r in ranges:
        base = r["start"]
        r["local_offsets"] = {t["name"]: t["global_offset"] - base for t in r["tensors"]}
    return {
        "pack": pack.provenance(),
        "group_by": group_by,
        "range_count": len(ranges),
        "ranges": ranges,
    }
