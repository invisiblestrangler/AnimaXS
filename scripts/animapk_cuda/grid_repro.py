#!/usr/bin/env python3
"""G0-G4 image-space grid reproduction on CUDA (Clore).

Lanes (one identical Python/CUDA VAE decode path for every latent):
  G0  golden final latent               -> VAE -> PNG   (clean decoder control)
  G1  official BF16 source, real CUDA graph, 8-step Euler -> latent -> VAE -> PNG
  G2  FP16-all .animapk, real CUDA graph, 8-step Euler   -> latent -> VAE -> PNG
  G3  W8 .animapk,      real CUDA graph, 8-step Euler   -> latent -> VAE -> PNG
  G4  Metal 8-step final latent (step07_denoised)       -> VAE -> PNG

The VAE decode is a torch/CUDA port of the validated scripts/vae_decoder_oracle.py
semantics (D060/D061): latent fed UNCHANGED (no mean/std transform), conv2 ->
decoder.conv1 -> middle (residual/attention/residual) -> 15 upsample modules ->
head -> RGB, final-slice folds (rank-5 -> w[:, :, -1]), channel RMS norm
(F.normalize over C x sqrt(C) x gamma), single-head spatial attention,
nearest-exact 2x resample. The port is validated against the NumPy oracle on the
golden latent before any DiT lane is decoded (--validate-vae).

PNG conversion is the canonical production transform (extract_golden_fixtures.py
to_display_uint8 == AnimaKernels.metal vae_position_to_rgba8):
  clamp((v+1)*0.5, 0, 1), floor(x*255+0.5), RGB interleave [512,512,3].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from animapk_cuda import ladder_real  # noqa: E402  (build_real / euler / decode_pack)


# ---------------------------------------------------------------------------
# Canonical PNG conversion (identical to Metal vae_position_to_rgba8)
# ---------------------------------------------------------------------------
def to_display_uint8(value: float) -> int:
    x = (float(value) + 1.0) * 0.5
    x = max(0.0, min(x, 1.0))
    return int(np.floor(x * 255.0 + 0.5))


def rgb_f32_to_png(rgb: np.ndarray, path: str) -> None:
    """rgb: float32 [3,512,512] CHW -> canonical uint8 [512,512,3] PNG."""
    assert rgb.shape == (3, 512, 512), rgb.shape
    hwc = rgb.transpose(1, 2, 0)  # [512,512,3]
    u8 = np.array([to_display_uint8(v) for v in hwc.reshape(-1)],
                  dtype=np.uint8).reshape(512, 512, 3)
    from PIL import Image
    Image.fromarray(u8, mode="RGB").save(path)
    return u8


# ---------------------------------------------------------------------------
# Torch CUDA VAE decoder (port of scripts/vae_decoder_oracle.py semantics)
# ---------------------------------------------------------------------------
def t_conv2d(x, w, b=None):
    """x [C,H,W] f32, w [Co,Ci,kh,kw] f32 -> [Co,H,W], padding same."""
    return F.conv2d(x.unsqueeze(0), w, b, padding=(w.shape[2] // 2, w.shape[3] // 2)).squeeze(0)


def t_channel_rms_norm(x, gamma):
    c = x.shape[0]
    inv = 1.0 / torch.clamp(x.norm(dim=0, keepdim=True), min=1e-12)
    return (x * inv * np.sqrt(float(c))) * gamma.reshape(c, 1, 1)


def t_silu(x):
    return x * torch.sigmoid(x)


def t_fold2d(w):
    if w.ndim == 5:
        return w[:, :, -1].contiguous()
    return w


def t_residual_block(x, tensors):
    in_c, out_c = x.shape[0], tensors["w2"].shape[0]
    old = x
    nx = t_channel_rms_norm(x, tensors["g0"])
    nx = t_silu(nx)
    nx = t_conv2d(nx, t_fold2d(tensors["w2"]), tensors["b2"])
    nx = t_channel_rms_norm(nx, tensors["g3"])
    nx = t_silu(nx)
    nx = t_conv2d(nx, t_fold2d(tensors["w6"]), tensors["b6"])
    if "shortcut" in tensors:
        sc = t_conv2d(old, t_fold2d(tensors["shortcut"]), tensors.get("shortcut_b"))
    else:
        sc = old
    return nx + sc


def t_attention_block(x, tensors):
    c, h, w = x.shape
    identity = x
    nx = t_channel_rms_norm(x, tensors["norm_gamma"])
    qkv = t_conv2d(nx, t_fold2d(tensors["to_qkv"]), tensors["to_qkv_b"])  # [3C,H,W]
    q, k, v = [qkv[j * c:(j + 1) * c].reshape(c, h * w).t().contiguous() for j in range(3)]
    scale = 1.0 / np.sqrt(c)
    scores = (q @ k.t()) * scale
    probs = torch.softmax(scores, dim=1)
    att = (probs @ v).t().reshape(c, h, w)
    att = t_conv2d(att, t_fold2d(tensors["proj"]), tensors["proj_b"])
    return att + identity


def t_resample_up(x, w, b):
    up = F.interpolate(x.unsqueeze(0), scale_factor=2.0, mode="nearest").squeeze(0)
    return t_conv2d(up, w, b)


def load_decoder_tensors_torch(pack_path):
    """Load decoder/post-quant tensors via inspect_animapk -> torch f32 CUDA."""
    from inspect_animapk import Animapk
    pk = Animapk(pack_path)
    tensors = {}
    for rec in pk.read_table():
        name = rec["name"]
        if name.startswith("decoder.") or name in ("conv2.weight", "conv2.bias"):
            tensors[name] = torch.from_numpy(pk.decode(rec)).float().cuda()
    return tensors


def decode_latent_torch(latent: np.ndarray, t: dict) -> np.ndarray:
    """latent [16,64,64] f32 (unchanged raw VAE input space, D060)."""
    z = torch.from_numpy(np.ascontiguousarray(latent, dtype=np.float32)).cuda()
    x = t_conv2d(z, t_fold2d(t["conv2.weight"]), t["conv2.bias"])
    x = t_conv2d(x, t_fold2d(t["decoder.conv1.weight"]), t["decoder.conv1.bias"])

    mid0 = lambda n: f"decoder.middle.0.residual.{n}"
    mid2 = lambda n: f"decoder.middle.2.residual.{n}"
    x = t_residual_block(x, {
        "g0": t[mid0("0.gamma")], "w2": t[mid0("2.weight")], "b2": t[mid0("2.bias")],
        "g3": t[mid0("3.gamma")], "w6": t[mid0("6.weight")], "b6": t[mid0("6.bias")]})
    x = t_attention_block(x, {
        "norm_gamma": t["decoder.middle.1.norm.gamma"],
        "to_qkv": t["decoder.middle.1.to_qkv.weight"], "to_qkv_b": t["decoder.middle.1.to_qkv.bias"],
        "proj": t["decoder.middle.1.proj.weight"], "proj_b": t["decoder.middle.1.proj.bias"]})
    x = t_residual_block(x, {
        "g0": t[mid2("0.gamma")], "w2": t[mid2("2.weight")], "b2": t[mid2("2.bias")],
        "g3": t[mid2("3.gamma")], "w6": t[mid2("6.weight")], "b6": t[mid2("6.bias")]})

    for m in (0, 1, 2):
        pre = f"decoder.upsamples.{m}.residual."
        x = t_residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = t_resample_up(x, t["decoder.upsamples.3.resample.1.weight"], t["decoder.upsamples.3.resample.1.bias"])
    pre = "decoder.upsamples.4.residual."
    x = t_residual_block(x, {
        "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
        "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"],
        "shortcut": t["decoder.upsamples.4.shortcut.weight"],
        "shortcut_b": t["decoder.upsamples.4.shortcut.bias"]})
    for m in (5, 6):
        pre = f"decoder.upsamples.{m}.residual."
        x = t_residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = t_resample_up(x, t["decoder.upsamples.7.resample.1.weight"], t["decoder.upsamples.7.resample.1.bias"])
    for m in (8, 9, 10):
        pre = f"decoder.upsamples.{m}.residual."
        x = t_residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = t_resample_up(x, t["decoder.upsamples.11.resample.1.weight"], t["decoder.upsamples.11.resample.1.bias"])
    for m in (12, 13, 14):
        pre = f"decoder.upsamples.{m}.residual."
        x = t_residual_block(x, {
            "g0": t[pre + "0.gamma"], "w2": t[pre + "2.weight"], "b2": t[pre + "2.bias"],
            "g3": t[pre + "3.gamma"], "w6": t[pre + "6.weight"], "b6": t[pre + "6.bias"]})
    x = t_channel_rms_norm(x, t["decoder.head.0.gamma"])
    x = t_silu(x)
    x = t_conv2d(x, t_fold2d(t["decoder.head.2.weight"]), t["decoder.head.2.bias"])
    return x.detach().cpu().numpy().astype(np.float32)  # [3,512,512]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def latent_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return {"cosine": cos, "rmse": rmse, "rel_l2": rel_l2}


def rgb_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    psnr = float(20 * np.log10(255.0 / (np.sqrt(np.mean((a - b) ** 2)) + 1e-12)))
    return {"cosine": cos, "rmse": rmse, "psnr": psnr}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="dir with x_in.f32, context512.f32")
    ap.add_argument("--source", required=True, help="official BF16 safetensors")
    ap.add_argument("--fp16-pack", required=True)
    ap.add_argument("--w8-pack", required=True)
    ap.add_argument("--vae-pack", required=True)
    ap.add_argument("--npz", required=True, help="golden npz")
    ap.add_argument("--metal-dir", default=None,
                    help="dir containing step07_denoised.f32 per pack (G4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate-vae", action="store_true",
                    help="validate torch VAE port vs NumPy oracle on golden latent")
    ap.add_argument("--skip-dit", action="store_true", help="skip G1-G3 DiT runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}")

    z = np.load(args.npz, allow_pickle=True)
    golden = z["final_latent"].astype(np.float32)          # [1,16,1,64,64]
    golden_latent = golden.reshape(16, 64, 64)
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    ref_rgb = z["decoded_rgb"][0, :, 0, :, :].astype(np.float32)  # [3,512,512]
    x0 = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "x_in.f32"),
                                      dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "context512.f32"),
                                           dtype=np.float32)).view(1, 512, 1024)

    manifest = {
        "experiment": "2026-08-14_grid-repro",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "source": args.source, "source_sha256": ladder_real.dso.PINNED_SOURCE_SHA,
        "npz": os.path.basename(args.npz), "npz_sha256": sha256_file(args.npz),
        "fixture_x_in_sha256": sha256_file(os.path.join(args.fixture, "x_in.f32")),
        "fixture_ctx512_sha256": sha256_file(os.path.join(args.fixture, "context512.f32")),
        "sigmas": sigmas,
        "golden_latent_sha256": sha256_bytes(golden.astype("<f4").tobytes()),
        "ref_rgb_sha256": sha256_bytes(ref_rgb.astype("<f4").tobytes()),
        "packs": {
            "fp16": {"path": args.fp16_pack, "sha256": sha256_file(args.fp16_pack)},
            "w8": {"path": args.w8_pack, "sha256": sha256_file(args.w8_pack)},
            "vae": {"path": args.vae_pack, "sha256": sha256_file(args.vae_pack)},
        },
        "lanes": {},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ---- reference PNG (canonical) ----
    ref_png = os.path.join(args.out, "reference.png")
    rgb_f32_to_png(ref_rgb, ref_png)
    manifest["lanes"]["reference"] = {"png": os.path.basename(ref_png),
                                      "png_sha256": sha256_file(ref_png)}

    # ---- VAE tensors (torch CUDA) ----
    print("loading VAE pack tensors...")
    vae_t = load_decoder_tensors_torch(args.vae_pack)

    def decode_save(name: str, latent: np.ndarray) -> np.ndarray:
        """Decode latent [16,64,64] -> RGB [3,512,512], save PNG + .f32, return rgb."""
        rgb = decode_latent_torch(latent, vae_t)
        rgb.tofile(os.path.join(args.out, f"{name}_rgb.f32"))
        png = os.path.join(args.out, f"{name}.png")
        rgb_f32_to_png(rgb, png)
        return rgb

    def lane_record(name, latent, rgb):
        m = latent_metrics(latent.ravel(), golden_latent.ravel())
        m["rgb_vs_ref_cosine"] = rgb_metrics(rgb, ref_rgb)["cosine"]
        m["rgb_vs_ref_rmse"] = rgb_metrics(rgb, ref_rgb)["rmse"]
        m["rgb_vs_ref_psnr"] = rgb_metrics(rgb, ref_rgb)["psnr"]
        m["latent_sha256"] = sha256_bytes(np.ascontiguousarray(latent, np.float32).tobytes())
        m["rgb_sha256"] = sha256_bytes(np.ascontiguousarray(rgb, np.float32).tobytes())
        manifest["lanes"][name] = m
        with open(os.path.join(args.out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    # ---- G0: golden latent -> VAE (clean decoder control) ----
    print("== G0: golden latent -> VAE ==")
    g0_rgb = decode_save("G0_golden", golden_latent)
    lane_record("G0_golden", golden_latent, g0_rgb)
    print("   G0 rgb vs ref cosine",
          round(manifest["lanes"]["G0_golden"]["rgb_vs_ref_cosine"], 6))

    # ---- validate torch VAE port vs NumPy oracle (golden latent) ----
    if args.validate_vae:
        print("== validating torch VAE port vs NumPy oracle ==")
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        import vae_decoder_oracle as vo
        from inspect_animapk import Animapk
        pk = Animapk(args.vae_pack)
        npt = vo.load_decoder_tensors(pk)
        np_rgb = vo.decode_latent(golden_latent, npt) if hasattr(vo, "decode_latent") else None
        if np_rgb is None:
            # replicate oracle main() decode inline
            x = vo.conv2d(golden_latent, vo.fold2d(npt["conv2.weight"], True), npt["conv2.bias"])
            x = vo.conv2d(x, vo.fold2d(npt["decoder.conv1.weight"], True), npt["decoder.conv1.bias"])
            mid0 = lambda n: f"decoder.middle.0.residual.{n}"
            mid2 = lambda n: f"decoder.middle.2.residual.{n}"
            x = vo.residual_block(x, {"g0": npt[mid0("0.gamma")], "w2": npt[mid0("2.weight")],
                                      "b2": npt[mid0("2.bias")], "g3": npt[mid0("3.gamma")],
                                      "w6": npt[mid0("6.weight")], "b6": npt[mid0("6.bias")]})
            x = vo.attention_block(x, {"norm_gamma": npt["decoder.middle.1.norm.gamma"],
                                       "to_qkv": npt["decoder.middle.1.to_qkv.weight"],
                                       "to_qkv_b": npt["decoder.middle.1.to_qkv.bias"],
                                       "proj": npt["decoder.middle.1.proj.weight"],
                                       "proj_b": npt["decoder.middle.1.proj.bias"]})
            x = vo.residual_block(x, {"g0": npt[mid2("0.gamma")], "w2": npt[mid2("2.weight")],
                                      "b2": npt[mid2("2.bias")], "g3": npt[mid2("3.gamma")],
                                      "w6": npt[mid2("6.weight")], "b6": npt[mid2("6.bias")]})
            for m in (0, 1, 2):
                pre = f"decoder.upsamples.{m}.residual."
                x = vo.residual_block(x, {"g0": npt[pre + "0.gamma"], "w2": npt[pre + "2.weight"],
                                          "b2": npt[pre + "2.bias"], "g3": npt[pre + "3.gamma"],
                                          "w6": npt[pre + "6.weight"], "b6": npt[pre + "6.bias"]})
            x = vo.resample_up(x, npt["decoder.upsamples.3.resample.1.weight"], npt["decoder.upsamples.3.resample.1.bias"])
            pre = "decoder.upsamples.4.residual."
            x = vo.residual_block(x, {"g0": npt[pre + "0.gamma"], "w2": npt[pre + "2.weight"],
                                      "b2": npt[pre + "2.bias"], "g3": npt[pre + "3.gamma"],
                                      "w6": npt[pre + "6.weight"], "b6": npt[pre + "6.bias"],
                                      "shortcut": npt["decoder.upsamples.4.shortcut.weight"],
                                      "shortcut_b": npt["decoder.upsamples.4.shortcut.bias"]})
            for m in (5, 6):
                pre = f"decoder.upsamples.{m}.residual."
                x = vo.residual_block(x, {"g0": npt[pre + "0.gamma"], "w2": npt[pre + "2.weight"],
                                          "b2": npt[pre + "2.bias"], "g3": npt[pre + "3.gamma"],
                                          "w6": npt[pre + "6.weight"], "b6": npt[pre + "6.bias"]})
            x = vo.resample_up(x, npt["decoder.upsamples.7.resample.1.weight"], npt["decoder.upsamples.7.resample.1.bias"])
            for m in (8, 9, 10):
                pre = f"decoder.upsamples.{m}.residual."
                x = vo.residual_block(x, {"g0": npt[pre + "0.gamma"], "w2": npt[pre + "2.weight"],
                                          "b2": npt[pre + "2.bias"], "g3": npt[pre + "3.gamma"],
                                          "w6": npt[pre + "6.weight"], "b6": npt[pre + "6.bias"]})
            x = vo.resample_up(x, npt["decoder.upsamples.11.resample.1.weight"], npt["decoder.upsamples.11.resample.1.bias"])
            for m in (12, 13, 14):
                pre = f"decoder.upsamples.{m}.residual."
                x = vo.residual_block(x, {"g0": npt[pre + "0.gamma"], "w2": npt[pre + "2.weight"],
                                          "b2": npt[pre + "2.bias"], "g3": npt[pre + "3.gamma"],
                                          "w6": npt[pre + "6.weight"], "b6": npt[pre + "6.bias"]})
            x = vo.channel_rms_norm(x, npt["decoder.head.0.gamma"])
            x = vo.silu(x)
            x = vo.conv2d(x, vo.fold2d(npt["decoder.head.2.weight"], True), npt["decoder.head.2.bias"])
            np_rgb = np.asarray(x, np.float32)
        np_rgb.tofile(os.path.join(args.out, "G0_numpy_oracle_rgb.f32"))
        err = np.abs(np_rgb.astype(np.float64) - g0_rgb.astype(np.float64))
        v = {"max_abs": float(err.max()), "rmse": float(np.sqrt(np.mean(err ** 2))),
             "cosine": rgb_metrics(np_rgb, g0_rgb)["cosine"]}
        manifest["vae_port_validation"] = v
        with open(os.path.join(args.out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"   torch port vs numpy oracle: {v}")
        if v["max_abs"] > 1e-3:
            print("WARNING: VAE port diverges from oracle beyond 1e-3; do not trust lanes")

    # ---- DiT lanes (G1-G3) ----
    if not args.skip_dit:
        def euler(model, dtype):
            x = x0.to(device, dtype)
            c = context.to(device, dtype)
            with torch.no_grad():
                for i in range(8):
                    s, s_next = sigmas[i], sigmas[i + 1]
                    v = model(x, torch.tensor([s], dtype=dtype, device=device), c)
                    denoised = x - s * v
                    x = x + (x - denoised) / s * (s_next - s)
            return x.float().cpu().numpy().reshape(16, 64, 64)

        # G1: official BF16 source
        print("== G1: official BF16 source ==")
        w = ladder_real.load_src_weights(args.source)
        model = ladder_real.build_real(w, torch.bfloat16, device)
        lat = euler(model, torch.bfloat16)
        del model
        lat.tofile(os.path.join(args.out, "G1_bf16_final_latent.f32"))
        rgb = decode_save("G1_bf16", lat)
        lane_record("G1_bf16", lat, rgb)
        print("   G1 latent cos vs golden", round(manifest["lanes"]["G1_bf16"]["cosine"], 6))

        # G2: FP16-all pack
        print("== G2: FP16-all pack ==")
        wdec, prov = ladder_real.decode_pack(args.fp16_pack)
        manifest["packs"]["fp16"]["packer"] = prov.get("packer")
        model = ladder_real.build_real(wdec, torch.float16, device)
        lat = euler(model, torch.float16)
        del model
        lat.tofile(os.path.join(args.out, "G2_fp16all_final_latent.f32"))
        rgb = decode_save("G2_fp16all", lat)
        lane_record("G2_fp16all", lat, rgb)
        print("   G2 latent cos vs golden", round(manifest["lanes"]["G2_fp16all"]["cosine"], 6))

        # G3: W8 pack
        print("== G3: W8 pack ==")
        wdec, prov = ladder_real.decode_pack(args.w8_pack)
        manifest["packs"]["w8"]["packer"] = prov.get("packer")
        model = ladder_real.build_real(wdec, torch.float16, device)
        lat = euler(model, torch.float16)
        del model
        lat.tofile(os.path.join(args.out, "G3_w8_final_latent.f32"))
        rgb = decode_save("G3_w8", lat)
        lane_record("G3_w8", lat, rgb)
        print("   G3 latent cos vs golden", round(manifest["lanes"]["G3_w8"]["cosine"], 6))
    else:
        print("--skip-dit: skipping G1-G3 DiT runs")

    # ---- G4: Metal final latents through the same VAE ----
    if args.metal_dir:
        for pack in ("fp16-all", "w8", "w4"):
            p = os.path.join(args.metal_dir, pack, "step07_denoised.f32")
            if not os.path.isfile(p):
                print(f"G4: {p} missing, skip")
                continue
            lat = np.fromfile(p, dtype=np.float32).reshape(16, 64, 64)
            name = f"G4_metal_{pack}"
            rgb = decode_save(name, lat)
            lane_record(name, lat, rgb)
            print(f"   {name} latent cos vs golden",
                  round(manifest["lanes"][name]["cosine"], 6))

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("manifest written:", os.path.join(args.out, "manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
