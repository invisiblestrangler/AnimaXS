#!/usr/bin/env python3
"""
pack_anima.py — build .animapk containers (XS-Max edition, v1).

Produces exactly:
  anima-turbo-v1.0-xsmax-w4.animapk     (DiT, W4 group64)
  qwen3-0.6b-xsmax-w8.animapk           (text encoder, W8 group64)
  qwen-image-vae-xsmax-fp16.animapk     (VAE, fp16)

Reads official safetensors, processes tensors ONE AT A TIME (streaming), never
creating another full in-memory copy of a whole model. 16 KB alignment.

W4: uint4 group-wise affine, group=64, even K idx -> low nibble, odd -> high nibble.
W8: uint8 group-wise affine, group=64. scale/zero fp16. value = q*scale + zero.
fp16/fp32: stored as-is.

Usage:
  pack_anima.py --component dit   --input anima-turbo-v1.0.safetensors --out out.animapk --quant w4
  pack_anima.py --component te    --input qwen_3_06b_base.safetensors  --out out.animapk --quant w8
  pack_anima.py --component vae   --input qwen_image_vae.safetensors   --out out.animapk --quant fp16

Dependency: safetensors (numpy for tensor parse). Prefer running where torch is available.
"""
import argparse, hashlib, json, os, struct, sys, zlib

MAGIC = b"ANMA"
VERSION = 1
ALIGN = 16384
HEADER_SIZE = 256
TENSOR_RECORD_SIZE = 128
GROUP = 64

COMPONENT = {"dit": 1, "te": 2, "vae": 3}
QUANT = {"w4": 1, "w8": 2, "fp16": 0}

def align_up(n, a=ALIGN):
    return (n + a - 1) & ~(a - 1)

# ---- quantization primitives (pure numpy) ----
def quantize_groupwise(x, bits):
    """x: 1D float32 array. group size GROUP along the leading/K dim.
       Returns (q_uint8_packed, scale_fp16_bytes, zero_fp16_bytes)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    ngroup = (n + GROUP - 1) // GROUP
    scale = np.zeros(ngroup, dtype=np.float16)
    zero = np.zeros(ngroup, dtype=np.float16)
    qfull = np.zeros(n, dtype=np.uint8)
    for g in range(ngroup):
        lo = g * GROUP
        hi = min(lo + GROUP, n)
        seg = x[lo:hi]
        mn = seg.min()
        mx = seg.max()
        if bits == 4:
            qmax = 15.0
        else:  # 8
            qmax = 255.0
        if mx - mn < 1e-8:
            s = 1.0
            z = mn
        else:
            s = (mx - mn) / qmax
            z = mn
        qfull[lo:hi] = np.clip(np.round((seg - z) / s), 0, qmax).astype(np.uint8)
        scale[g] = np.float16(s)
        zero[g] = np.float16(z)
    if bits == 4:
        # pack 2 uint4 per byte: even index -> low nibble, odd -> high nibble
        if n % 2:
            qfull = np.append(qfull, np.uint8(0))
        packed = (qfull[0::2] & 0x0F) | ((qfull[1::2] & 0x0F) << 4)
        return packed.tobytes(), scale.tobytes(), zero.tobytes()
    return qfull.tobytes(), scale.tobytes(), zero.tobytes()

def quantize_matrix(w, bits):
    """w: (out, in) float32. Quantize along in (K) dim with group GROUP."""
    import numpy as np
    w = np.asarray(w, dtype=np.float32)
    out, inn = w.shape
    data_parts = []
    scale_parts = []
    zero_parts = []
    for r in range(out):
        pd, sc, zr = quantize_groupwise(w[r], bits)
        data_parts.append(pd)
        scale_parts.append(sc)
        zero_parts.append(zr)
    return b"".join(data_parts), b"".join(scale_parts), b"".join(zero_parts)

# ---- container writer ----
def write_pack(path, component, json_obj, tensors, quant, source_hashes):
    """
    tensors: list of dicts in EXECUTION ORDER:
      {name, shape, logical_dtype, storage_dtype, data, scale, zero, crc32,
       block_index, execution_index}
    Assigns 16 KB-aligned blob offsets in execution order. Converges the
    JSON/table/payload layout (JSON size is stable across offset values because
    offsets stay small), then writes the container.
    """
    import numpy as np

    def compute_offsets(payload_offset):
        cur = payload_offset
        for t in tensors:
            blob = len(t["data"]) + len(t["scale"]) + len(t["zero"])
            t["_blob_aligned"] = align_up(blob)
            t["blob_offset"] = cur
            cur += t["_blob_aligned"]
        return cur

    def build_json():
        meta = []
        for t in tensors:
            meta.append({
                "name": t["name"], "shape": list(t["shape"]),
                "logical_dtype": t["logical_dtype"], "storage_dtype": t["storage_dtype"],
                "crc32": t["crc32"], "block_index": t["block_index"],
                "execution_index": t["execution_index"],
                "blob_offset": t["blob_offset"], "blob_size": t["_blob_aligned"],
                "data_size": len(t["data"]),
                "scale_offset": len(t["data"]), "scale_size": len(t["scale"]),
                "zero_offset": len(t["data"]) + len(t["scale"]), "zero_size": len(t["zero"]),
            })
        j = dict(json_obj)
        j["tensor_meta"] = meta
        j["quant"] = {"scheme": quant, "group": GROUP}
        j["source_hashes"] = source_hashes
        return json.dumps(j, indent=2).encode()

    # pass 1: provisional payload offset to assign blob offsets (JSON size not yet known)
    est_json = len(json.dumps(json_obj).encode())
    payload0 = align_up(HEADER_SIZE + align_up(est_json) + align_up(TENSOR_RECORD_SIZE * len(tensors)))
    compute_offsets(payload0)
    # build real JSON; recompute layout; if payload offset changed, re-assign offsets
    json_bytes = build_json()
    table_offset = HEADER_SIZE + align_up(len(json_bytes))
    payload_offset = align_up(table_offset + align_up(TENSOR_RECORD_SIZE * len(tensors)))
    if payload_offset != payload0:
        compute_offsets(payload_offset)
        json_bytes = build_json()
        table_offset = HEADER_SIZE + align_up(len(json_bytes))
        payload_offset = align_up(table_offset + align_up(TENSOR_RECORD_SIZE * len(tensors)))

    # build tensor table
    table = bytearray()
    for t in tensors:
        name_b = t["name"].encode()[:63].ljust(64, b"\x00")
        rank = len(t["shape"])
        shape_pad = list(t["shape"]) + [0] * (4 - rank)
        logical = {"fp16": 0, "w4": 1, "w8": 2, "fp32": 3}[t["logical_dtype"]]
        storage = {"fp16": 2, "w4": 0, "w8": 1, "fp32": 3}[t["storage_dtype"]]
        numel = 1
        for d in t["shape"]:
            numel *= d
        data_off = 0
        scale_off = len(t["data"])
        zero_off = scale_off + len(t["scale"])
        table += struct.pack(
            "<64sI4IBB2xQQQIIII",
            name_b, rank, shape_pad[0], shape_pad[1], shape_pad[2], shape_pad[3],
            logical, storage,
            numel, t["blob_offset"], t["_blob_aligned"],
            data_off, len(t["data"]), scale_off, zero_off,
        )

    payload_size = sum(t["_blob_aligned"] for t in tensors)
    file_size = payload_offset + payload_size

    final_pos = None
    with open(path, "wb") as f:
        hdr = bytearray(HEADER_SIZE)
        struct.pack_into(
            "<4sHHIQQQQQQQI",
            hdr, 0,
            MAGIC, VERSION, COMPONENT[component], ALIGN,
            len(tensors),
            HEADER_SIZE, len(json_bytes),
            table_offset, len(table),
            payload_offset,
            file_size,
            TENSOR_RECORD_SIZE,
        )
        f.write(hdr)
        f.write(json_bytes)
        f.write(b"\x00" * (align_up(len(json_bytes)) - len(json_bytes)))
        f.write(table)
        f.write(b"\x00" * (align_up(len(table)) - len(table)))
        for t in tensors:
            cur = f.tell()
            if cur != t["blob_offset"]:
                assert cur < t["blob_offset"]
                f.write(b"\x00" * (t["blob_offset"] - cur))
            f.write(t["data"])
            f.write(t["scale"])
            f.write(t["zero"])
            cur = f.tell()
            f.write(b"\x00" * (align_up(cur - t["blob_offset"]) - (cur - t["blob_offset"])))
        final_pos = f.tell()
    assert final_pos == file_size
    return file_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True, choices=["dit", "te", "vae"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", required=True, choices=["w4", "w8", "fp16"])
    ap.add_argument("--exclude-json", help="fp16 exclusion list JSON (names to keep fp16)")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy required")
    try:
        import torch
    except ImportError:
        sys.exit("torch required (for bf16 source handling)")
    from safetensors import safe_open

    exclude = set()
    if args.exclude_json and os.path.exists(args.exclude_json):
        exclude = set(json.load(open(args.exclude_json)).get("exclude", []))

    quant_bits = 4 if args.quant == "w4" else (8 if args.quant == "w8" else None)
    storage_dtype = args.quant if args.quant != "fp16" else "fp16"

    json_obj = {"component": args.component, "quant": {"scheme": args.quant, "group": GROUP}}

    tensors = []
    import re
    # Use torch framework: bf16 weights loaded natively; convert to fp32 numpy for quantize.
    with safe_open(args.input, framework="pt") as f:
        for name in f.keys():
            src = f.get_tensor(name)  # torch tensor (bf16/fp16/fp32)
            src_np_f32 = src.float().numpy()  # fp32 for quantization
            logical = args.quant
            storage = storage_dtype
            # small/sensitive tensors (rank<=1) + excluded -> keep fp16 (policy)
            if args.quant != "fp16" and (name in exclude or src.dim() <= 1):
                logical = "fp16"
                storage = "fp16"
            data = scale = zero = b""
            if logical == "fp16":
                data = src.float().numpy().astype(np.float16).tobytes()
            elif logical in ("w4", "w8"):
                data, scale, zero = quantize_matrix(src_np_f32, quant_bits)
            crc = zlib.crc32(data)
            # block index for streaming layout (DiT blocks.N.* ; -1 for non-block)
            m = re.search(r"blocks\.(\d+)\.", name)
            block_index = int(m.group(1)) if m else -1
            tensors.append({
                "name": name, "shape": list(src.shape),
                "logical_dtype": logical, "storage_dtype": storage,
                "data": data, "scale": scale, "zero": zero, "crc32": crc,
                "block_index": block_index, "execution_index": len(tensors),
            })

    source_hashes = {}
    try:
        h = hashlib.sha256()
        with open(args.input, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        source_hashes[os.path.basename(args.input)] = h.hexdigest()
    except Exception:
        pass

    size = write_pack(args.out, args.component, json_obj, tensors, args.quant, source_hashes)
    print(f"wrote {args.out}: {size} bytes, {len(tensors)} tensors")

if __name__ == "__main__":
    main()
