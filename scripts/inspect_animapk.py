#!/usr/bin/env python3
"""
inspect_animapk.py — independent .animapk reader (v1, XS-Max).
Implemented independently of packer internals (only the documented spec).
  parse header, parse JSON, validate ranges, validate CRC, list tensors,
  decode fp16 / W8 / W4, export .npy.

Usage:
  inspect_animapk.py FILE [--json] [--validate] [--list] [--decode NAME --npy out.npy]
"""
import argparse, json, mmap, struct, sys, zlib

MAGIC = b"ANMA"
ALIGN = 16384

def rd_u16(b, o): return struct.unpack_from("<H", b, o)[0]
def rd_u32(b, o): return struct.unpack_from("<I", b, o)[0]
def rd_u64(b, o): return struct.unpack_from("<Q", b, o)[0]

class Animapk:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.blob = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.parse_header()
        self.parse_json()

    def parse_header(self):
        b = self.blob
        assert b[:4] == MAGIC, "bad magic"
        self.version = rd_u16(b, 4)
        self.component = rd_u16(b, 6)
        self.alignment = rd_u32(b, 8)
        self.tensor_count = rd_u64(b, 12)
        self.json_offset = rd_u64(b, 20)
        self.json_size = rd_u64(b, 28)
        self.table_offset = rd_u64(b, 36)
        self.table_bytes = rd_u64(b, 44)
        self.payload_offset = rd_u64(b, 52)
        self.file_size = rd_u64(b, 60)
        self.record_size = rd_u32(b, 68)
        assert self.magic_ok, "magic"
        assert self.alignment == ALIGN

    @property
    def magic_ok(self):
        return self.blob[:4] == MAGIC

    def parse_json(self):
        raw = self.blob[self.json_offset:self.json_offset + self.json_size]
        self.meta = json.loads(bytes(raw))

    def read_table(self):
        recs = []
        for i in range(self.tensor_count):
            o = self.table_offset + i * self.record_size
            b = self.blob
            name = bytes(b[o:o+64]).split(b"\x00")[0].decode()
            rank = rd_u32(b, o+64)
            shape = [rd_u32(b, o+68+4*j) for j in range(4)][:rank]
            logical = b[o+84]
            storage = b[o+85]
            numel = rd_u64(b, o+88)
            blob_offset = rd_u64(b, o+96)
            blob_size = rd_u64(b, o+104)
            data_off = rd_u32(b, o+112)
            data_size = rd_u32(b, o+116)
            scale_off = rd_u32(b, o+120)
            zero_off = rd_u32(b, o+124)
            recs.append(dict(name=name, shape=shape, logical_dtype=logical,
                             storage_dtype=storage, numel=numel, blob_offset=blob_offset,
                             blob_size=blob_size, data_off=data_off, data_size=data_size,
                             scale_off=scale_off, zero_off=zero_off))
        return recs

    def crc(self, rec):
        data = bytes(self.blob[rec["blob_offset"]+rec["data_off"]:
                              rec["blob_offset"]+rec["data_off"]+rec["data_size"]])
        return zlib.crc32(data)

    def decode(self, rec, n_groups_per_row=None):
        """Return float32 numpy array dequantized from storage."""
        import numpy as np
        base = rec["blob_offset"]
        data = bytes(self.blob[base+rec["data_off"]: base+rec["data_off"]+rec["data_size"]])
        storage = rec["storage_dtype"]
        shape = rec["shape"]
        # Prefer full shape + full name from JSON tensor_meta. The binary table truncates
        # names to 64 chars and shape to 4 dims; JSON tensor_meta carries full data and is
        # in the SAME execution order as the binary table (both written in order), so match
        # by execution index via blob_offset (unique, reliable).
        for m in self.meta.get("tensor_meta", []):
            if m.get("blob_offset") == rec["blob_offset"]:
                if len(m.get("shape", [])) >= len(shape):
                    shape = list(m["shape"])
                # also fix the (possibly truncated) name for the caller
                if len(m.get("name", "")) > len(rec["name"]):
                    rec["name"] = m["name"]
                break
        if storage == 2:  # fp16le
            arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
        elif storage == 3:  # fp32le
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
        elif storage == 0:  # uint4 packed
            n = rec["numel"]
            qfull = np.zeros((n + (n % 2)), dtype=np.uint8)
            packed = np.frombuffer(data, dtype=np.uint8)
            qfull[0:len(packed)*2:2] = packed & 0x0F
            qfull[1:len(packed)*2:2] = (packed >> 4) & 0x0F
            qfull = qfull[:n].reshape(shape)
            ngroups = (shape[-1] + 63) // 64
            nsc = int(np.prod(shape[:-1])) * ngroups * 2  # fp16 bytes
            scale = np.frombuffer(bytes(self.blob[base+rec["scale_off"]: base+rec["scale_off"]+nsc]),
                                  dtype=np.float16).astype(np.float32)
            zero = np.frombuffer(bytes(self.blob[base+rec["zero_off"]: base+rec["zero_off"]+nsc]),
                                 dtype=np.float16).astype(np.float32)
            arr = qfull.astype(np.float32)
            s = scale.reshape(list(shape[:-1]) + [ngroups])
            z = zero.reshape(list(shape[:-1]) + [ngroups])
            # broadcast group index along last dim
            gi = np.arange(shape[-1]) // 64
            arr = arr * s[..., gi] + z[..., gi]
        elif storage == 1:  # uint8
            q = np.frombuffer(data, dtype=np.uint8).astype(np.float32).reshape(shape)
            ngroups = (shape[-1] + 63) // 64
            nsc = int(np.prod(shape[:-1])) * ngroups * 2  # fp16 bytes
            scale = np.frombuffer(bytes(self.blob[base+rec["scale_off"]:
                                                  base+rec["scale_off"]+nsc]),
                                  dtype=np.float16).astype(np.float32)
            zero = np.frombuffer(bytes(self.blob[base+rec["zero_off"]:
                                                 base+rec["zero_off"]+nsc]),
                                 dtype=np.float16).astype(np.float32)
            s = scale.reshape(list(shape[:-1]) + [ngroups])
            z = zero.reshape(list(shape[:-1]) + [ngroups])
            gi = np.arange(shape[-1]) // 64
            arr = q * s[..., gi] + z[..., gi]
        else:
            raise ValueError("unknown storage dtype %d" % storage)
        return arr

    def close(self):
        self.blob.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--decode")
    ap.add_argument("--npy")
    args = ap.parse_args()

    pk = Animapk(args.file)
    print(f"magic ok, version {pk.version}, component {pk.component}, align {pk.alignment}")
    print(f"tensors {pk.tensor_count}, json @{pk.json_offset} size {pk.json_size}, "
          f"table @{pk.table_offset} {pk.table_bytes}B, payload @{pk.payload_offset}, "
          f"file {pk.file_size}B")
    recs = pk.read_table()

    if args.validate:
        ok = True
        for r in recs:
            if r["blob_offset"] % ALIGN != 0:
                print(f"  FAIL align {r['name']} @{r['blob_offset']}"); ok = False
            if r["blob_offset"] + r["blob_size"] > pk.file_size:
                print(f"  FAIL range {r['name']}"); ok = False
            want = r["numel"] * (1 if r["storage_dtype"] in (0,) else
                                 (1 if r["storage_dtype"] in (1,) else
                                  2 if r["storage_dtype"] in (2,) else 4))
            # uint4 packed bytes ~ numel//2; check within data_size
            if r["data_size"] < (r["numel"] // 2 if r["storage_dtype"] == 0 else
                                 r["numel"] * (1 if r["storage_dtype"] in (1,) else
                                               2 if r["storage_dtype"] == 2 else 4)):
                print(f"  FAIL data_size {r['name']}"); ok = False
        # CRC from JSON
        crcs = {m["name"]: m.get("crc32") for m in pk.meta.get("tensor_meta", [])}
        nbad = 0
        for r in recs:
            if r["name"] in crcs and crcs[r["name"]] is not None:
                if pk.crc(r) != crcs[r["name"]]:
                    print(f"  FAIL crc {r['name']}"); nbad += 1; ok = False
        print(f"validate: {'PASS' if ok else 'FAIL'} ({nbad} crc mismatches)")

    if args.json:
        print(json.dumps(pk.meta, indent=2)[:4000])

    if args.list:
        for r in recs:
            print(f"  {r['name']:50s} {str(r['shape']):22s} "
                  f"logical={r['logical_dtype']} storage={r['storage_dtype']} "
                  f"blob@{r['blob_offset']} crc={pk.crc(r):08x}")

    if args.decode:
        import numpy as np
        rec = next((r for r in recs if r["name"] == args.decode), None)
        if rec is None:
            sys.exit(f"tensor {args.decode} not found")
        arr = pk.decode(rec)
        print(f"decoded {args.decode} shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
        if args.npy:
            np.save(args.npy, arr)
            print("saved", args.npy)
    pk.close()

if __name__ == "__main__":
    main()
