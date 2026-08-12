#!/usr/bin/env python3
"""Validate the same-pack Wan T=1 VAE decoder against canonical `decoded_rgb`.

Lane B oracle (section 27): reads the real fp16 .animapk weights, executes the
pinned T=1 Wan decoder graph, consumes `case1_final_latent`, emits RGB, and
compares against the canonical source `decoded_rgb`. This isolates
source-vs-pack/precision difference (Lane B); the Swift/Metal decoder must match
THIS pack-executed output (Lane A), not necessarily the BF16 source pixel-exact.

Layout: activations are [C, H, W] float32. Rank-5 causal weights fold to their
FINAL temporal slice (payload slice index 2 for kt=3) per D052. time_conv is not
executed (no feature cache at T=1). Upsample stages: nearest-exact 2x then 3x3
convolution.

The latent decode normalization is parameterized (scale_factor 0.5 vs 1.0) and
resolved empirically against `decoded_rgb`.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/anima-xsmax/scripts")
from inspect_animapk import Animapk

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"

# Wan21 latent statistics (also ModelConstants).
LATENTS_MEAN = np.array([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
    dtype=np.float32).reshape(16, 1, 1)
LATENTS_STD = np.array([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
    dtype=np.float32).reshape(16, 1, 1)


def conv2d(x, w, b=None):
    """[Cin,H,W] \u00d7 [Cout,Cin,kh,kw] -> [Cout,H,W], stride 1, padding same."""
    co, ci, kh, kw = w.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    # Sliding-window view is (ci, H, W, kh, kw); einsum contracts ci/kh/kw.
    view = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(1, 2))
    rows, cols = x.shape[1], x.shape[2]
    out = np.empty((co, rows, cols), dtype=np.float32)
    tile = 64
    for c0 in range(0, cols, tile):
        c1 = min(cols, c0 + tile)
        patch = view[:, :, c0:c1]  # (ci, H, cspan, kh, kw)
        out[:, :, c0:c1] = np.einsum(
            "cYXab,ocab->oYX", patch, w, optimize=True).astype(np.float32)
    if b is not None:
        out += b.reshape(co, 1, 1)
    return out


def conv1x1(x, w, b=None):
    """[Cin,H,W] x [Cout,Cin] or [Cout,Cin,1,1] -> [Cout,H,W]."""
    if w.ndim == 4:
        w = w.reshape(w.shape[0], w.shape[1])
    co, ci = w.shape
    out = np.tensordot(w.reshape(co, ci), x, axes=([1], [0])).astype(np.float32)
    if b is not None:
        out += b.reshape(co, 1, 1)
    return out


def channel_rms_norm(x, gamma):
    """F.normalize over C at each pixel \u00d7 sqrt(C) \u00d7 gamma. x:[C,H,W]."""
    c = x.shape[0]
    sq = np.sum(x.astype(np.float32) * x.astype(np.float32), axis=0)
    inv = np.reciprocal(np.maximum(np.sqrt(sq), 1e-12))
    scale = np.sqrt(float(c))
    return (x.astype(np.float32) * inv * scale) * gamma.reshape(c, 1, 1)


def silu(x):
    return x / (1.0 + np.exp(-np.asarray(x, np.float32)))


def nearest_exact_2x(x):
    c, h, w = x.shape
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def residual_block(x, tensors, fold=True):
    """Wan ResidualBlock. tensors keyed by suffix index."""
    in_c, out_c = x.shape[0], tensors["w2"].shape[0]
    old = x
    nx = channel_rms_norm(x, tensors["g0"])
    nx = silu(nx)
    nx = conv2d(nx, fold2d(tensors["w2"], fold), tensors["b2"])
    nx = channel_rms_norm(nx, tensors["g3"])
    nx = silu(nx)
    nx = conv2d(nx, fold2d(tensors["w6"], fold), tensors["b6"])
    if "shortcut" in tensors:
        sc = conv2d(old, fold2d(tensors["shortcut"], fold), tensors.get("shortcut_b"))
    else:
        sc = old
    return nx + sc


def attention_block(x, tensors, fold=True):
    """Single-head spatial attention over [H*W] positions at C."""
    c, h, w = x.shape
    identity = x
    nx = channel_rms_norm(x, tensors["norm_gamma"])
    qkv = conv1x1(nx, fold2d(tensors["to_qkv"], fold), tensors["to_qkv_b"])  # [3C,H,W]
    q, k, v = [np.ascontiguousarray(qkv[j * c:(j + 1) * c].reshape(c, h * w).T)
               for j in range(3)]  # [HW,C]
    scale = 1.0 / np.sqrt(c)
    scores = (q @ k.T) * scale  # [HW,HW]
    # softmax (stable)
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / e.sum(axis=1, keepdims=True)
    att = probs.astype(np.float32) @ v  # [HW,C]
    att = att.T.reshape(c, h, w)
    att = conv1x1(att, fold2d(tensors["proj"], fold), tensors["proj_b"])
    return att + identity


def fold2d(w, fold):
    """Rank-5 -> 2-D final temporal slice (D052). Rank-4 native passed through."""
    if w.ndim == 5:
        return np.ascontiguousarray(w[:, :, -1]) if fold else np.ascontiguousarray(w[:, :, -1])
    return w


def resample_up(x, w, b):
    """nearest-exact 2x then 3x3 conv C->C/2 (T=1, no time_conv)."""
    up = nearest_exact_2x(x)
    return conv2d(up, w, b)


def load_decoder_tensors(pack):
    """Materialize every decoder/post-quant tensor keyed by full name."""
    recs = pack.read_table()
    tensors = {}
    for rec in recs:
        name = rec["name"]
        if name.startswith("decoder.") or name in ("conv2.weight", "conv2.bias"):
            tensors[name] = pack.decode(rec)
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/root/anima-xsmax/results/packs/qwen-image-vae-xsmax-fp16.animapk")
    ap.add_argument("--latent", default="/root/AnimaXS/AnimaXSTests/Fixtures/Case1Binary/case1_final_latent.f32")
    ap.add_argument("--golden", default="/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz")
    ap.add_argument("--scale-factor", type=float, default=0.0, help="latent decode scale factor (0=identity, or 0.5/1.0 mean-std denorm)")
    ap.add_argument("--json", dest="json_path")
    args = ap.parse_args()

    # Pack hash + integrity
    hasher = hashlib.sha256()
    with open(args.pack, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            hasher.update(ch)
    pack_sha = hasher.hexdigest()
    print("pack_sha256", pack_sha)

    t = load_decoder_tensors(Animapk(args.pack))
    latent = np.fromfile(args.latent, dtype=np.float32).reshape(16, 64, 64)
    golden = np.load(args.golden, allow_pickle=True)
    ref_rgb = golden["decoded_rgb"]  # [1,3,1,512,512]
    print("ref_rgb", ref_rgb.shape, "min/max",
          float(golden["decoded_rgb_min"]), float(golden["decoded_rgb_max"]))

    # ---- latent decode normalization ----
    # The reference trace feeds the sampler's `samples` directly into the raw
    # WanVAE module (`vae_model.decode(vae_in)`), bypassing latent_format.
    # Resolve empirically: sf>0 applies mean/std denormalization, 0 = identity.
    sf = args.scale_factor
    if sf > 0:
        z = latent / sf * LATENTS_STD + LATENTS_MEAN
    else:
        z = latent

    # ---- conv2 (1x1x1 -> 2-D 1x1, 16->16) ----
    x = conv2d(z, fold2d(t["conv2.weight"], True), t["conv2.bias"])
    # ---- decoder.conv1 (final-slice 3x3, 16->384) ----
    x = conv2d(x, fold2d(t["decoder.conv1.weight"], True), t["decoder.conv1.bias"])
    # ---- middle ----
    mid0 = lambda n: f"decoder.middle.0.residual.{n}"
    mid2 = lambda n: f"decoder.middle.2.residual.{n}"
    x = residual_block(x, {
        "g0": t[mid0("0.gamma")], "w2": t[mid0("2.weight")], "b2": t[mid0("2.bias")],
        "g3": t[mid0("3.gamma")], "w6": t[mid0("6.weight")], "b6": t[mid0("6.bias")]})
    x = attention_block(x, {
        "norm_gamma": t["decoder.middle.1.norm.gamma"],
        "to_qkv": t["decoder.middle.1.to_qkv.weight"], "to_qkv_b": t["decoder.middle.1.to_qkv.bias"],
        "proj": t["decoder.middle.1.proj.weight"], "proj_b": t["decoder.middle.1.proj.bias"]})
    x = residual_block(x, {
        "g0": t[mid2("0.gamma")], "w2": t[mid2("2.weight")], "b2": t[mid2("2.bias")],
        "g3": t[mid2("3.gamma")], "w6": t[mid2("6.weight")], "b6": t[mid2("6.bias")]})

    # ---- upsample stages ----
    # stage 0 (384@64): 3 residuals, resample 384->192 @128
    for m in (0, 1, 2):
        pre = f"decoder.upsamples.{m}.residual."
        x = residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = resample_up(x, t["decoder.upsamples.3.resample.1.weight"], t["decoder.upsamples.3.resample.1.bias"])
    # stage 1 (192->384@128): upsamples.4 residual + shortcut, 5,6 residuals, resample 384->192 @256
    pre = "decoder.upsamples.4.residual."
    x = residual_block(x, {
        "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
        "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"],
        "shortcut": t["decoder.upsamples.4.shortcut.weight"],
        "shortcut_b": t["decoder.upsamples.4.shortcut.bias"]})
    for m in (5, 6):
        pre = f"decoder.upsamples.{m}.residual."
        x = residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = resample_up(x, t["decoder.upsamples.7.resample.1.weight"], t["decoder.upsamples.7.resample.1.bias"])
    # stage 2 (192@256): residuals 8,9,10, resample 192->96 @512
    for m in (8, 9, 10):
        pre = f"decoder.upsamples.{m}.residual."
        x = residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = resample_up(x, t["decoder.upsamples.11.resample.1.weight"], t["decoder.upsamples.11.resample.1.bias"])
    # stage 3 (96@512): residuals 12,13,14
    for m in (12, 13, 14):
        pre = f"decoder.upsamples.{m}.residual."
        x = residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    # ---- head: RMS norm -> SiLU -> 3x3 conv 96->3 ----
    x = channel_rms_norm(x, t["decoder.head.0.gamma"])
    x = silu(x)
    rgb = conv2d(x, fold2d(t["decoder.head.2.weight"], True), t["decoder.head.2.bias"])
    print("decoded rgb  min/max", float(rgb.min()), float(rgb.max()))

    ref = ref_rgb.reshape(3, 512, 512).astype(np.float32)
    error = rgb.astype(np.float64) - ref.astype(np.float64)
    metrics = {
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "psnr": float(20 * np.log10(255.0 / (np.sqrt(np.mean(error * error)) + 1e-12))),
        "cosine": float(np.dot(rgb.ravel(), ref.ravel()) /
                         (np.linalg.norm(rgb.ravel()) * np.linalg.norm(ref.ravel()))),
    }
    print("scale_factor", sf, "metrics", {k: round(v, 6) for k, v in metrics.items()})

    result = {
        "pinned_commit": PINNED_COMMIT, "pack_sha256": pack_sha,
        "scale_factor": sf, "metrics": metrics,
        "decoded_rgb_source_sha256": hashlib.sha256(ref.astype(np.float32).tobytes()).hexdigest(),
    }
    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or ".", exist_ok=True)
        with open(args.json_path, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
