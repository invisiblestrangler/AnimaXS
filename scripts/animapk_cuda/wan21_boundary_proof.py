#!/usr/bin/env python3
"""Phase A: zero-inference proof of the missing Wan21 process_out boundary.

Hypothesis (K's instruction file, 2026-08-14): the custom CUDA/Metal path
decodes raw SAMPLER-space latents as if they were VAE-space latents, omitting
ComfyUI's model-latent-format boundary transform. ComfyUI applies
`latent_format.process_out` to the sampler return inside CFGGuider.inner_sample
(comfy/samplers.py) BEFORE the workflow receives the latent, so the golden
`final_latent` is VAE-space while the harness's saved final latents are
sampler-space. The 8px grid is the decoded result of that missing affine
per-channel transform.

This script is PURELY zero-inference: it reuses already-saved tensors and the
already-validated CUDA VAE decode path (grid_repro.py semantics, D060/D061).

Tests:
  A1 provenance JSON: exact pinned Wan21 constants + call sites + file SHAs
  A2 golden relationship: process_out(step_latents[7]) vs final_latent
  A3 decode raw step7 / converted step7 / golden final through CUDA VAE
  A4 saved G1/G2/G3/G4 final latents: before/after process_out -> VAE -> carrier

Controls:
  - process_out(golden_final)      [double-apply control: must get WORSE]
  - process_in(process_out(x)) ~= x [identity]
  - harness G1 raw vs golden step7  [both sampler-space: must be HIGH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from animapk_cuda.grid_repro import (  # noqa: E402
    load_decoder_tensors_torch, decode_latent_torch, rgb_f32_to_png,
    latent_metrics, rgb_metrics, sha256_file, sha256_bytes,
)
from measure_grid_carrier import carrier_scores, load_rgb  # noqa: E402

# ---------------------------------------------------------------------------
# Exact pinned Wan21 constants — read verbatim from the project's pinned
# comfy-ref snapshot (MiniTrainDIT/comfy cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44,
# /workspace/comfy-ref on Clore; not a git repo — file SHAs recorded in A1).
# DO NOT replace with current ComfyUI master values.
# ---------------------------------------------------------------------------
WAN21_LATENTS_MEAN = np.array([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
], dtype=np.float32)

WAN21_LATENTS_STD = np.array([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
], dtype=np.float32)

WAN21_SCALE_FACTOR = 1.0


def wan21_process_out(x: np.ndarray) -> np.ndarray:
    """x: [16,64,64] or [1,16,1,64,64] f32 sampler-space -> VAE-space."""
    c = x.shape[-3] if x.ndim == 5 else x.shape[0]
    std = WAN21_LATENTS_STD.reshape((1, c, 1, 1) if x.ndim == 5 else (c, 1, 1))
    mean = WAN21_LATENTS_MEAN.reshape((1, c, 1, 1) if x.ndim == 5 else (c, 1, 1))
    return (x.astype(np.float64) * std / WAN21_SCALE_FACTOR + mean).astype(np.float32)


def wan21_process_in(x: np.ndarray) -> np.ndarray:
    """Exact inverse (process_in)."""
    c = x.shape[-3] if x.ndim == 5 else x.shape[0]
    std = WAN21_LATENTS_STD.reshape((1, c, 1, 1) if x.ndim == 5 else (c, 1, 1))
    mean = WAN21_LATENTS_MEAN.reshape((1, c, 1, 1) if x.ndim == 5 else (c, 1, 1))
    return ((x.astype(np.float64) - mean) * WAN21_SCALE_FACTOR / std).astype(np.float32)


def chan_stats(x: np.ndarray) -> dict:
    x = x.reshape(16, -1)
    return {
        "mean_per_channel": [float(v) for v in x.mean(axis=1)],
        "std_per_channel": [float(v) for v in x.std(axis=1)],
    }


def sha(x: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(x, np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vae-pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-repro", required=True, help="dir with G1-G3 final latents")
    ap.add_argument("--metal-dir", required=True,
                    help="dir with {fp16-all,w8,w4}/step07_denoised.f32 (G4)")
    ap.add_argument("--comfy-ref", required=True, help="pinned comfy-ref checkout dir")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results: dict = {"experiment": "2026-08-14_wan21-process-out-fix",
               "phase": "A_zero-inference-proof",
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---------------- A1 provenance ----------------
    comfy_files = ["supported_models.py", "latent_formats.py",
                   "samplers.py", "model_base.py"]
    prov = {
        "comfy_ref_pin": "MiniTrainDIT/comfy cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44 "
                         "(per HERMES_SESSION.md; checkout is a file snapshot, not a git repo)",
        "anima_latent_format": "Wan21",
        "scale_factor": WAN21_SCALE_FACTOR,
        "latents_mean": [float(v) for v in WAN21_LATENTS_MEAN],
        "latents_std": [float(v) for v in WAN21_LATENTS_STD],
        "process_out": "latent * latents_std / scale_factor + latents_mean",
        "process_in": "(latent - latents_mean) * scale_factor / latents_std",
        "call_site": ("comfy/samplers.py CFGGuider.inner_sample: "
                      "'return self.inner_model.process_latent_out(samples.to(torch.float32))' "
                      "right after sampler.sample returns; model_base.py process_latent_out "
                      "delegates to latent_format.process_out; Anima.latent_format = Wan21 "
                      "(supported_models.py)"),
        "comfy_ref_file_sha256": {f: sha256_file(os.path.join(args.comfy_ref, f))
                                  for f in comfy_files},
        "note": "constants read verbatim from pinned comfy-ref; not current master",
    }
    results["A1_provenance"] = prov

    # ---------------- load fixture + VAE ----------------
    z = np.load(args.npz, allow_pickle=True)
    step7 = z["step_latents"][7].astype(np.float32).reshape(16, 64, 64)
    final = z["final_latent"].astype(np.float32).reshape(16, 64, 64)
    ref_rgb = z["decoded_rgb"][0, :, 0, :, :].astype(np.float32)  # [3,512,512]
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    results["sigmas"] = sigmas
    results["last_sigma_next"] = sigmas[-1]  # 0.0 => final x == final denoised

    # reference PNG + carrier
    ref_png = os.path.join(args.out, "reference.png")
    rgb_f32_to_png(ref_rgb, ref_png)
    ref_carrier = carrier_scores(load_rgb(ref_png))
    results["reference_carrier"] = ref_carrier

    print("loading VAE pack tensors...")
    vae_t = load_decoder_tensors_torch(args.vae_pack)

    def decode_carrier(name: str, latent: np.ndarray) -> dict:
        """Decode [16,64,64] -> save rgb.f32 + png; return carrier + rgb metrics."""
        rgb = decode_latent_torch(np.ascontiguousarray(latent, np.float32), vae_t)
        rgb.tofile(os.path.join(args.out, f"{name}_rgb.f32"))
        png = os.path.join(args.out, f"{name}.png")
        rgb_f32_to_png(rgb, png)
        c = carrier_scores(load_rgb(png))
        rm = rgb_metrics(rgb, ref_rgb)
        return {"carrier": c,
                "carrier_ratio_vs_ref": c["total"] / ref_carrier["total"],
                "rgb_cos_vs_ref": rm["cosine"], "rgb_rmse_vs_ref": rm["rmse"],
                "png": os.path.basename(png)}

    # ---------------- A2 golden relationship ----------------
    raw_m = latent_metrics(step7, final)
    conv7 = wan21_process_out(step7)
    conv_m = latent_metrics(conv7, final)
    identity_m = latent_metrics(wan21_process_in(wan21_process_out(step7)), step7)
    results["A2_golden"] = {
        "step7_sha256": sha(step7), "final_sha256": sha(final),
        "converted_sha256": sha(conv7),
        "shapes": {"step7": list(step7.shape), "final": list(final.shape),
                   "converted": list(conv7.shape)},
        "dtypes": {"step7": str(step7.dtype), "final": str(final.dtype)},
        "raw_vs_final": raw_m,
        "converted_vs_final": conv_m,
        "process_in_process_out_identity": identity_m,
        "channel_stats_raw": chan_stats(step7),
        "channel_stats_converted": chan_stats(conv7),
        "channel_stats_final": chan_stats(final),
    }
    print(f"[A2] raw step7 vs final:      cos {raw_m['cosine']:.6f} rmse {raw_m['rmse']:.6f}")
    print(f"[A2] converted vs final:      cos {conv_m['cosine']:.6f} rmse {conv_m['rmse']:.6f} "
          f"maxabs {np.abs(conv7 - final).max():.6f}")
    print(f"[A2] process_in(process_out) ~= id: cos {identity_m['cosine']:.6f} "
          f"rmse {identity_m['rmse']:.6f}")

    # ---------------- A3 decode both sides ----------------
    print("== A3: decode raw step7 / converted step7 / golden final ==")
    results["A3_decode"] = {
        "raw_step7": decode_carrier("golden_step7_raw", step7),
        "converted_step7": decode_carrier("golden_step7_converted", conv7),
        "golden_final": decode_carrier("golden_final", final),
        "double_apply_control": decode_carrier("golden_final_double_processed",
                                                wan21_process_out(final)),
    }
    for k, v in results["A3_decode"].items():
        print(f"   {k}: carrier {v['carrier']['total']:.6f} "
              f"ratio {v['carrier_ratio_vs_ref']:.1f}x rgb_cos {v['rgb_cos_vs_ref']:.6f}")

    # ---------------- A4 saved G1/G2/G3/G4 latents ----------------
    lanes = {}
    lanes["G1_bf16"] = os.path.join(args.grid_repro, "G1_bf16_final_latent.f32")
    lanes["G2_fp16all"] = os.path.join(args.grid_repro, "G2_fp16all_final_latent.f32")
    lanes["G3_w8"] = os.path.join(args.grid_repro, "G3_w8_final_latent.f32")
    for pack in ("fp16-all", "w8", "w4"):
        lanes[f"G4_metal_{pack}"] = os.path.join(
            args.metal_dir, pack, "step07_denoised.f32")

    # control: harness raw G1 vs golden step7 (both sampler-space)
    g1 = np.fromfile(lanes["G1_bf16"], dtype=np.float32).reshape(16, 64, 64)
    results["A4_control_harness_G1_vs_golden_step7"] = latent_metrics(g1, step7)

    print("== A4: saved final latents, before/after process_out ==")
    results["A4_lanes"] = {}
    for name, path in lanes.items():
        if not os.path.isfile(path):
            print(f"   {name}: MISSING {path}, skip")
            continue
        raw = np.fromfile(path, dtype=np.float32).reshape(16, 64, 64)
        conv = wan21_process_out(raw)
        before = latent_metrics(raw, final)
        after = latent_metrics(conv, final)
        dec_before = decode_carrier(f"{name}_raw", raw)
        dec_after = decode_carrier(f"{name}_converted", conv)
        results["A4_lanes"][name] = {
            "source": path, "raw_sha256": sha(raw), "converted_sha256": sha(conv),
            "latent_cos_vs_golden_before": before["cosine"],
            "latent_cos_vs_golden_after": after["cosine"],
            "latent_rmse_vs_golden_before": before["rmse"],
            "latent_rmse_vs_golden_after": after["rmse"],
            "decode_before": dec_before,
            "decode_after": dec_after,
        }
        b, a = before["cosine"], after["cosine"]
        cb, ca = dec_before["carrier_ratio_vs_ref"], dec_after["carrier_ratio_vs_ref"]
        print(f"   {name}: cos {b:.4f}->{a:.4f}  carrier {cb:.1f}x -> {ca:.1f}x")

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---------------- contact sheet ----------------
    try:
        from PIL import Image, ImageDraw
        rows, cols, cell = 3, 6, 512
        sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        names = ["golden_step7_raw", "golden_step7_converted", "golden_final"]
        for lane in list(results["A4_lanes"].keys())[:3]:
            names += [f"{lane}_raw", f"{lane}_converted"]
        for i, n in enumerate(names):
            p = os.path.join(args.out, f"{n}.png")
            if not os.path.isfile(p):
                continue
            im = Image.open(p).resize((cell, cell))
            r, c = i // cols, i % cols
            sheet.paste(im, (c * cell, r * cell))
            draw.text((c * cell + 4, r * cell + 4), n, fill=(255, 0, 0))
        sheet_path = os.path.join(args.out, "contact_sheet.png")
        sheet.save(sheet_path)
        results["contact_sheet"] = os.path.basename(sheet_path)
        print("contact sheet:", sheet_path)
    except Exception as e:  # non-fatal
        print("contact sheet failed:", e)

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
