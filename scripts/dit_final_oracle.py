#!/usr/bin/env python3
"""Same-W4 NumPy oracle for H007 FinalLayer and source-order unpatchify.

The equations and dtype boundaries follow pinned ComfyUI commit cbbc9dab:
`comfy/ldm/cosmos/predict2.py` FinalLayer.forward and MiniTrainDIT.unpatchify.
This script reads only the three final-layer tensors from the production pack.

W4-ONLY oracle: it casts the large fp32 residual to fp16 before LayerNorm,
which matches the known-good W4 pack path. It is NOT the numerical authority
for the W8-v2 pack — W8/source semantics keep the residual in BF16 range
(source model dtype is bfloat16; predict2.py keeps a bf16 residual stream).
For W8-v2 correctness use `dit_source_oracle.py` (the pinned torch source
oracle), which applies BF16 RNE rounding while retaining fp32 storage and
never converts the large residual to fp16.
"""
import argparse
import pathlib
import sys

import numpy as np

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"
PREFIX = "model.diffusion_model.final_layer."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", default=(
        "/root/anima-xsmax/results/packs/anima-turbo-v1.0-xsmax-w4.animapk"))
    parser.add_argument("--golden", default=(
        "/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz"))
    parser.add_argument("--oracle-dir", default="/root/AnimaXS/scripts/oracle_out/block0")
    parser.add_argument("--reader-dir", default="/root/anima-xsmax/scripts")
    parser.add_argument("--out", default="/tmp/dit_final_oracle_case1.npz")
    return parser.parse_args()


def silu(x):
    return x / (1.0 + np.exp(-x))


def load_weights(pack_path, reader_dir):
    sys.path.insert(0, reader_dir)
    import inspect_animapk as reader

    pack = reader.Animapk(pack_path)
    full_names = {
        item["blob_offset"]: item["name"]
        for item in pack.meta.get("tensor_meta", [])
    }
    records = {full_names.get(r["blob_offset"], r["name"]): r
               for r in pack.read_table()}

    def get(suffix, shape):
        name = PREFIX + suffix
        value = pack.decode(records[name]).astype(np.float32)
        if value.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        return value

    result = (
        get("adaln_modulation.1.weight", (256, 2048)),
        get("adaln_modulation.2.weight", (4096, 256)),
        get("linear.weight", (64, 2048)),
    )
    pack.close()
    return result


def unpatchify(projected):
    # Source: "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)".
    tokens = projected.reshape(1, 1, 32, 32, 2, 2, 1, 16)
    return tokens.transpose(0, 7, 1, 6, 2, 4, 3, 5).reshape(1, 16, 1, 64, 64)


def main():
    args = parse_args()
    w1, w2, projection = load_weights(args.pack, args.reader_dir)
    with np.load(args.golden, allow_pickle=True) as golden:
        residual = golden["block_27_out"].reshape(1024, 2048).astype(np.float32)
    oracle_dir = pathlib.Path(args.oracle_dir)
    emb = np.fromfile(oracle_dir / "block0_emb.f32", dtype="<f4")
    adaln = np.fromfile(oracle_dir / "block0_adaln_lora.f32", dtype="<f4")[:4096]
    if emb.shape != (2048,) or adaln.shape != (4096,):
        raise ValueError("invalid H005 timestep fixtures")

    modulation = (silu(emb).astype(np.float32) @ w1.T).astype(np.float32)
    modulation = (modulation @ w2.T).astype(np.float32) + adaln
    shift, scale = np.split(modulation, 2)

    # predict2.py casts the fp32 residual to cross-attention fp16. PyTorch
    # LayerNorm computes stable statistics but preserves fp16 output dtype.
    x = residual.astype(np.float16).astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normalized = (centered / np.sqrt(variance + np.float32(1e-6))).astype(np.float16)
    modulated = (normalized.astype(np.float32) * (1.0 + scale) + shift).astype(np.float16)

    # LinearExecutor dequantizes W4 weights to fp16 and MPS returns fp16.
    projected = (modulated.astype(np.float32)
                 @ projection.astype(np.float16).astype(np.float32).T)
    projected = projected.astype(np.float16).astype(np.float32)
    velocity = unpatchify(projected)
    if not np.isfinite(velocity).all():
        raise ValueError("non-finite H007 oracle output")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, pinned_commit=np.array(PINNED_COMMIT), residual=residual,
        emb=emb, adaln=adaln, projected=projected, velocity=velocity)
    print(f"H007_ORACLE=PASS shape={velocity.shape} min={velocity.min():.8g} "
          f"max={velocity.max():.8g} mean={velocity.mean():.8g} out={out}")


if __name__ == "__main__":
    main()
