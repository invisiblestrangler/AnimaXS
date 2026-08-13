#!/usr/bin/env python3
"""REAL-GRAPH precision ladder (variant A/B/C/D/E through the actual pinned
upstream MiniTrainDIT with the fixed stub) — the definitive evidence chain:

  A  official BF16 safetensors  -> real upstream graph (bf16 CUDA)
  B  official weights -> FP16    -> real upstream graph (fp16)
  C  FP16-all .animapk           -> real upstream graph (decoded_reference + streaming)
  D  W8 .animapk                 -> real upstream graph (decoded_reference + streaming)
  E  W4 .animapk                 -> real upstream graph (decoded_reference + streaming)

Each variant runs the full 8-step Euler trajectory; the FINAL LATENT is
compared against the golden case1 final_latent (the actual ComfyUI capture) —
the ground-truth quality anchor.  Step-0 per-block captures are also emitted
for CUDA-vs-Metal backend parity later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dit_source_oracle as dso  # noqa: E402

from animapk_cuda import compare as cmp  # noqa: E402
from animapk_cuda.reader import PackFile  # noqa: E402
from animapk_cuda.runtime import DecodedReference, StreamingAnimapk  # noqa: E402
from animapk_cuda import upstream as up  # noqa: E402

STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_stub")


def cos(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def build_real(weights, dtype, device):
    """weights: {stripped_name: cpu tensor} (DiT only). Builds the REAL
    MiniTrainDIT and loads the weights."""
    sys.path.insert(0, STUB)
    from comfy.ldm.cosmos.predict2 import MiniTrainDIT  # noqa: E402
    from comfy import ops as _ops  # noqa: E402
    model = MiniTrainDIT(
        max_img_h=64, max_img_w=64, max_frames=1, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
        crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop", min_fps=1, max_fps=30,
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
        operations=_ops,
    )
    state = {k: v for k, v in weights.items() if not k.startswith("llm_adapter")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"real graph missing weights: {list(missing)[:8]}")
    model.to(dtype=dtype, device=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_src_weights(path):
    w = dso.load_weights(path)
    return {k: v for k, v in w.items() if not k.startswith("llm_adapter")}


def decode_pack(pack_path):
    with PackFile(pack_path) as pk:
        from animapk_cuda.quant import decode_tensor_from_pack
        out = {}
        for item in pk.tensor_meta:
            name = str(item["name"])
            if name.startswith("model.diffusion_model."):
                name = name[len("model.diffusion_model."):]
            if name.startswith("llm_adapter"):
                continue
            out[name] = decode_tensor_from_pack(pk, item, device="cpu")
        return out, pk.provenance()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--fp16-pack", required=True)
    ap.add_argument("--w8-pack", required=True)
    ap.add_argument("--w4-pack", required=True)
    ap.add_argument("--npz", required=True, help="golden npz for final-latent anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    z = np.load(args.npz, allow_pickle=True)
    golden = torch.from_numpy(z["final_latent"]).float()
    sigmas = [float(s) for s in z["sigmas_comfy"]]
    x0 = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(args.fixture, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)

    def euler(model, dtype):
        x = x0.to(device, dtype)
        c = context.to(device, dtype)
        with torch.no_grad():
            for i in range(8):
                s, s_next = sigmas[i], sigmas[i + 1]
                v = model(x, torch.tensor([s], dtype=dtype, device=device), c)
                denoised = x - s * v
                x = x + (x - denoised) / s * (s_next - s)
        return x.float().cpu()

    results = {}
    provenance = {"source": args.source, "source_sha256": dso.PINNED_SOURCE_SHA,
                  "npz": os.path.basename(args.npz), "packs": {}}

    variants = {
        "A_bf16": ("bf16", None, torch.bfloat16),
        "B_fp16": ("fp16", None, torch.float16),
    }
    for label, (_, _, dtype) in variants.items():
        print(f"== {label} ==")
        w = load_src_weights(args.source)
        if label.startswith("B"):
            w = {k: v.float().half() for k, v in w.items()}
        model = build_real(w, dtype, device)
        lat = euler(model, dtype)
        results[label] = {"final_cosine_vs_golden": cos(lat, golden),
                          "final_rmse": float((lat - golden).pow(2).mean().sqrt()),
                          "final_rel_l2": float((lat - golden).norm() / golden.norm())}
        print(f"   final vs golden cos {results[label]['final_cosine_vs_golden']:.6f}")
        del model

    packs = {
        "C_fp16all": args.fp16_pack,
        "D_w8": args.w8_pack,
        "E_w4": args.w4_pack,
    }
    for label, path in packs.items():
        print(f"== {label} ==")
        wdec, prov = decode_pack(path)
        provenance["packs"][label] = {"sha256": prov["sha256"], "size": prov["size"],
                                      "source": prov["source"], "packer": prov["packer"]}
        model = build_real(wdec, torch.float16, device)
        lat = euler(model, torch.float16)
        results[label] = {"final_cosine_vs_golden": cos(lat, golden),
                          "final_rmse": float((lat - golden).pow(2).mean().sqrt()),
                          "final_rel_l2": float((lat - golden).norm() / golden.norm())}
        print(f"   final vs golden cos {results[label]['final_cosine_vs_golden']:.6f}")
        del model

    # decoded vs streaming equivalence through the real graph (step-0 only, fast)
    step0_caps = {}
    for label, path in packs.items():
        with PackFile(path) as pk:
            dr = DecodedReference(pk, torch.float16, device)
            sm = StreamingAnimapk(pk, torch.float16, device)
            with torch.no_grad():
                _, c_dr = dr.forward(x0, sigmas[0], context)
                _, c_sm = sm.forward(x0, sigmas[0], context)
            v = cmp.metrics(c_dr["post_unpatchify_velocity"], c_sm["post_unpatchify_velocity"])
            results[label + "_decoded_vs_streaming"] = v
            print(f"   {label} decoded vs streaming velocity cos {v['cosine']:.6f} maxAbs {v['max_abs']:.3e}")

    cmp.write_json(os.path.join(args.out, "ladder_real_final.json"), {
        "golden_norm": float(golden.norm()),
        "results": results,
        "provenance": provenance,
    })
    rows = [{"variant": k, **v} for k, v in results.items() if isinstance(v, dict) and "final_cosine_vs_golden" in v]
    cmp.write_csv(os.path.join(args.out, "ladder_real_final.csv"), rows)
    md = ["# Real-graph precision ladder — final latent vs golden (case1 seed1337)\n",
          "| variant | cosine | RMSE | relL2 |", "|---|---|---|---|"]
    for k, v in results.items():
        if "final_cosine_vs_golden" in v:
            md.append(f"| {k} | {v['final_cosine_vs_golden']:.6f} | {v['final_rmse']:.6f} | {v['final_rel_l2']:.6f} |")
    md += ["", "decoded vs streaming (step-0 velocity):"]
    for k, v in results.items():
        if "decoded_vs_streaming" in k:
            md.append(f"- {k}: cos {v['cosine']:.8f} maxAbs {v['max_abs']:.3e} relL2 {v['rel_l2']:.3e}")
    with open(os.path.join(args.out, "ladder_real_final.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(json.dumps({k: (v.get("final_cosine_vs_golden") if isinstance(v, dict) else v) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
