"""ANMA v1 pack format constants and header parsing (mirrors Swift AnimapkHeader)."""

MAGIC = b"ANMA"
ALIGN = 16_384
HEADER_SIZE = 256
RECORD_SIZE = 128

STORAGE_CODE = {"w4": 0, "w8": 1, "fp16": 2, "fp32": 3}
STORAGE_NAME = {v: k for k, v in STORAGE_CODE.items()}

import struct


def parse_header(blob: bytes) -> dict:
    """Parse the 256-byte ANMA header from raw bytes."""
    assert len(blob) >= HEADER_SIZE, f"header too small: {len(blob)}"
    if blob[:4] != MAGIC:
        raise ValueError(f"bad magic {blob[:4]!r}")
    magic, version, component_code, alignment = (
        blob[:4],
        struct.unpack_from("<H", blob, 4)[0],
        struct.unpack_from("<H", blob, 6)[0],
        struct.unpack_from("<I", blob, 8)[0],
    )
    tensor_count = struct.unpack_from("<Q", blob, 12)[0]
    json_offset, json_size = struct.unpack_from("<QQ", blob, 20)
    table_offset, table_bytes = struct.unpack_from("<QQ", blob, 36)
    payload_offset, declared_size = struct.unpack_from("<QQ", blob, 52)
    record_size = struct.unpack_from("<I", blob, 68)[0]
    h = {
        "magic": magic.decode("ascii"),
        "version": version,
        "component_code": component_code,
        "alignment": alignment,
        "tensor_count": tensor_count,
        "json_offset": json_offset,
        "json_size": json_size,
        "table_offset": table_offset,
        "table_bytes": table_bytes,
        "payload_offset": payload_offset,
        "declared_size": declared_size,
        "record_size": record_size,
    }
    if version != 1:
        raise ValueError(f"unsupported version {version}")
    if alignment != ALIGN:
        raise ValueError(f"alignment {alignment} != {ALIGN}")
    if record_size != RECORD_SIZE:
        raise ValueError(f"record size {record_size} != {RECORD_SIZE}")
    return h


def parse_record(blob: bytes, off: int) -> dict:
    """Parse one 128-byte binary table record."""
    name = bytes(blob[off : off + 64]).split(b"\0", 1)[0].decode("utf-8", "replace")
    storage_code = struct.unpack_from("<I", blob, off + 64)[0]
    logical_dtype = struct.unpack_from("<I", blob, off + 68)[0]
    numel = struct.unpack_from("<Q", blob, off + 72)[0]
    blob_offset = struct.unpack_from("<Q", blob, off + 96)[0]
    blob_size = struct.unpack_from("<Q", blob, off + 104)[0]
    data_offset = struct.unpack_from("<Q", blob, off + 112)[0]
    data_size = struct.unpack_from("<Q", blob, off + 120)[0]
    return {
        "name": name,
        "storage_code": storage_code,
        "storage_dtype": STORAGE_NAME.get(storage_code, f"?{storage_code}"),
        "logical_dtype_code": logical_dtype,
        "numel": numel,
        "blob_offset": blob_offset,
        "blob_size": blob_size,
        "data_offset": data_offset,
        "data_size": data_size,
    }
