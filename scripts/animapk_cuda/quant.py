"""Byte-exact W4/W8/FP16 decoders for ANMA v1 packs (torch, CPU or CUDA).

Semantics mirrored from Swift QuantDecoders (byte-exact reference):

W4:  q = unsigned nibble 0..15; even K index -> LOW nibble, odd -> HIGH nibble.
W8:  q = unsigned byte 0..255.
Both: groups along K (input dim) with group size 64; groups RESET at every
matrix row; scale/zero are fp16;  value = q * scale + zero.

FP16: raw little-endian halfs (row-major).
"""

from __future__ import annotations

import torch

from . import format as fmt

GROUP = 64


def fp16_bytes_to_tensor(data: bytes, shape, device="cpu") -> torch.Tensor:
    """Decode a raw fp16 blob to a float32 tensor of the given shape."""
    n = 1
    for d in shape:
        n *= int(d)
    assert len(data) == n * 2, f"fp16 bytes {len(data)} != {n * 2}"
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(torch.float16)
    return t.float().reshape(shape)


def decode_w4_matrix(data: bytes, scale: bytes, zero: bytes, rows: int, cols: int,
                     group: int = GROUP, device="cpu") -> torch.Tensor:
    """Dequantize a row-reset W4 matrix [rows, cols] to float32."""
    bytes_per_row = (cols + 1) // 2
    groups_per_row = (cols + group - 1) // group
    assert len(data) >= rows * bytes_per_row
    assert len(scale) >= rows * groups_per_row * 2
    assert len(zero) >= rows * groups_per_row * 2

    b = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(rows, bytes_per_row)
    lo = (b & 0x0F).to(torch.float32)          # even K
    hi = (b >> 4).to(torch.float32)            # odd K
    # interleave lo/hi -> [rows, bytes_per_row, 2] -> flatten -> [rows, 2*bpr]
    q = torch.stack([lo, hi], dim=-1).reshape(rows, 2 * bytes_per_row)[:, :cols]

    s = torch.frombuffer(bytearray(scale), dtype=torch.uint8).view(torch.float16).float().reshape(rows, groups_per_row)
    z = torch.frombuffer(bytearray(zero), dtype=torch.uint8).view(torch.float16).float().reshape(rows, groups_per_row)

    col_groups = torch.arange(cols, dtype=torch.long, device=device) // group
    g = col_groups.unsqueeze(0).expand(rows, cols) + torch.arange(rows, dtype=torch.long, device=device).unsqueeze(1) * groups_per_row
    if device != "cpu":
        q, s, z, g = q.to(device), s.to(device), z.to(device), g.to(device)
    sg = s.flatten()[g.clamp(max=s.numel() - 1)]
    zg = z.flatten()[g.clamp(max=z.numel() - 1)]
    return q * sg + zg


def decode_w8_matrix(data: bytes, scale: bytes, zero: bytes, rows: int, cols: int,
                     group: int = GROUP, device="cpu") -> torch.Tensor:
    """Dequantize a row-reset W8 matrix [rows, cols] to float32."""
    groups_per_row = (cols + group - 1) // group
    assert len(data) >= rows * cols
    assert len(scale) >= rows * groups_per_row * 2
    assert len(zero) >= rows * groups_per_row * 2

    q = torch.frombuffer(bytearray(data), dtype=torch.uint8).float().reshape(rows, cols)
    s = torch.frombuffer(bytearray(scale), dtype=torch.uint8).view(torch.float16).float().reshape(rows, groups_per_row)
    z = torch.frombuffer(bytearray(zero), dtype=torch.uint8).view(torch.float16).float().reshape(rows, groups_per_row)
    col_groups = torch.arange(cols, dtype=torch.long, device=device) // group
    g = col_groups.unsqueeze(0).expand(rows, cols) + torch.arange(rows, dtype=torch.long, device=device).unsqueeze(1) * groups_per_row
    if device != "cpu":
        q, s, z, g = q.to(device), s.to(device), z.to(device), g.to(device)
    sg = s.flatten()[g.clamp(max=s.numel() - 1)]
    zg = z.flatten()[g.clamp(max=z.numel() - 1)]
    return q * sg + zg


def decode_tensor_from_pack(pack, item: dict, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Decode one pack tensor by its JSON metadata into a float32 torch tensor."""
    storage = item["storage_dtype"]
    shape = [int(x) for x in item["shape"]]
    data = pack.region(item, "data")
    if storage == "fp16":
        t = fp16_bytes_to_tensor(data, shape, device=device)
    elif storage == "fp32":
        n = 1
        for d in shape:
            n *= int(d)
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(torch.float32).reshape(shape)
    elif storage == "w4":
        if len(shape) == 2:
            rows, cols = shape
            t = decode_w4_matrix(data, pack.region(item, "scale"), pack.region(item, "zero"),
                                 rows, cols, device=device)
        else:
            raise ValueError(f"w4 non-2D shape {shape}")
    elif storage == "w8":
        if len(shape) == 2:
            rows, cols = shape
            t = decode_w8_matrix(data, pack.region(item, "scale"), pack.region(item, "zero"),
                                 rows, cols, device=device)
        else:
            raise ValueError(f"w8 non-2D shape {shape}")
    else:
        raise ValueError(f"unsupported storage {storage}")
    return t.to(dtype)


def decode_all_weights(pack, dtype=torch.float32, device="cpu",
                       name_map=lambda n: n) -> dict:
    """Decode every pack tensor into a weights dict keyed by stripped name.

    name_map: optional remap (e.g. strip 'model.diffusion_model.' happens in
    PackFile.meta_by_name already; this is a hook for special cases).
    """
    w = {}
    for item in pack.tensor_meta:
        name = str(item["name"])
        if name.startswith("model.diffusion_model."):
            name = name[len("model.diffusion_model."):]
        w[name] = decode_tensor_from_pack(pack, item, device=device, dtype=dtype)
    return w
