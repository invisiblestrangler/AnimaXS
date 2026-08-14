#!/usr/bin/env python3
"""Build the G0-G4 labeled contact sheet + difference/high-gain/FFT panels.

Panels (one contact sheet, labels in-image):
  REFERENCE | G0 GOLDEN | G1 BF16 | G2 FP16-ALL | G3 W8 | G4 METAL FP16
  ABS DIFF (high gain) | FFT magnitude comparison

Also writes side_by_side.png, difference.png, difference_high_gain.png,
center_crop.png, texture_crop.png, fft_magnitude.png, fft_comparison.png.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageOps

CANONICAL = {"G0_golden", "G1_bf16", "G2_fp16all", "G3_w8",
             "G4_metal_fp16-all", "G4_metal_w8", "G4_metal_w4"}


def load_rgb_f32(path):
    arr = np.fromfile(path, dtype=np.float32)
    # [3,512,512] CHW -> HWC [0,1]
    return arr.reshape(3, 512, 512).transpose(1, 2, 0)


def to_uint8(hwc):
    x = (hwc + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)
    return np.floor(x * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="grid_repro out dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = args.dir

    ref_f = load_rgb_f32(os.path.join(d, "G0_golden_rgb.f32"))
    # reference rgb == G0 decode of golden latent (cos 0.999989); use it as ref
    ref_u8 = to_uint8(ref_f)
    Image.fromarray(ref_u8, mode="RGB").save(os.path.join(args.out, "reference.png"))

    lanes = ["G0_golden", "G1_bf16", "G2_fp16all", "G3_w8", "G4_metal_fp16-all"]
    images = {}
    for name in lanes:
        p = os.path.join(d, f"{name}_rgb.f32")
        if not os.path.isfile(p):
            print("skip", p)
            continue
        images[name] = to_uint8(load_rgb_f32(p))

    # side-by-side
    w, h = 512, 512
    n = len(images) + 1
    canvas = Image.new("RGB", (w * n, h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(ref_u8, mode="RGB"), (0, 0))
    draw.text((8, 8), "REFERENCE", fill=(255, 255, 0))
    for i, (name, img) in enumerate(images.items(), start=1):
        canvas.paste(Image.fromarray(img, mode="RGB"), (i * w, 0))
        draw.text((i * w + 8, 8), name, fill=(255, 255, 0))
    canvas.save(os.path.join(args.out, "side_by_side.png"))

    # contact sheet with 2 rows: 4 per row (REF, G0, G1, G2 / G3, G4f, G4w8, G4w4)
    sheet_lanes = ["REFERENCE"] + lanes + ["G4_metal_w8", "G4_metal_w4"]
    sheet = Image.new("RGB", (w * 4, h * 2), (32, 32, 32))
    sdraw = ImageDraw.Draw(sheet)
    for idx, name in enumerate(sheet_lanes):
        if name == "REFERENCE":
            im = Image.fromarray(ref_u8, mode="RGB")
        elif name in images:
            im = Image.fromarray(images[name], mode="RGB")
        else:
            im = Image.fromarray(to_uint8(load_rgb_f32(os.path.join(d, f"{name}_rgb.f32"))), mode="RGB")
        r, c = divmod(idx, 4)
        sheet.paste(im, (c * w, r * h))
        sdraw.text((c * w + 8, r * h + 8), name, fill=(255, 255, 0))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))

    # differences vs reference
    diff = np.abs(ref_f.astype(np.float64) - images["G1_bf16"].astype(np.float64))
    diff_img = (diff / diff.max() * 255).astype(np.uint8)
    Image.fromarray(np.repeat(diff_img[:, :, None], 3, axis=2), mode="RGB").save(
        os.path.join(args.out, "difference.png"))
    gain = np.clip(diff * 8.0, 0, 255).astype(np.uint8)
    Image.fromarray(np.repeat(gain[:, :, None], 3, axis=2), mode="RGB").save(
        os.path.join(args.out, "difference_high_gain.png"))

    # center + texture crops of G1
    g1 = images["G1_bf16"]
    center = g1[176:336, 176:336]
    Image.fromarray(center, mode="RGB").save(os.path.join(args.out, "center_crop.png"))
    Image.fromarray(g1[32:160, 32:160], mode="RGB").save(os.path.join(args.out, "texture_crop.png"))

    # FFT magnitude (log) per image
    def fft_mag(hwc):
        out = []
        for c in range(3):
            x = hwc[:, :, c].astype(np.float64)
            x = x - x.mean()
            F = np.fft.fftshift(np.fft.fft2(x))
            out.append(np.log1p(np.abs(F)))
        return np.mean(out, axis=0)

    fm_ref = fft_mag(ref_f)
    fm_g1 = fft_mag(images["G1_bf16"])
    for name, fm in (("fft_magnitude_ref.png", fm_ref), ("fft_magnitude_g1.png", fm_g1)):
        norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-12)
        Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(
            os.path.join(args.out, name))
    # comparison: 3 panels side by side
    comp = Image.new("L", (512 * 3, 512), 0)
    for i, fm in enumerate((fm_ref, fm_g1, np.abs(fm_g1 - fm_ref))):
        norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-12)
        comp.paste(Image.fromarray((norm * 255).astype(np.uint8), mode="L"), (i * 512, 0))
    comp.save(os.path.join(args.out, "fft_comparison.png"))

    print("wrote:", sorted(os.listdir(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
