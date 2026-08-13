"""Run the A/B/C/D/E step-0 precision ladder on CUDA and emit all artifacts.

Variants:
  A  official BF16 safetensors  -> actual pinned upstream ComfyUI graph (CUDA)
  A2 same weights               -> validated source-oracle transcription (bf16)
  B  official weights -> FP16   -> source-oracle transcription
  C  FP16-all .animapk          -> decoded_reference + streaming_animapk
  D  W8 .animapk                -> decoded_reference + streaming_animapk
  E  W4 .animapk                -> decoded_reference + streaming_animapk

Outputs (per §28):
  precision_ladder_stage_parity.{csv,json,md}
  precision_ladder_cosine.png / precision_ladder_relative_l2.png
  source_oracle_parity.{json,md}        (A vs A2)
  animapk_decoder_parity.{json,md}      (pack decode vs official weights)
  animapk_execution_manifest.json       (streaming ranges)
  animapk_runtime_provenance.json
  provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dit_source_oracle as dso  # noqa: E402

from animapk_cuda import compare as cmp  # noqa: E402
from animapk_cuda.reader import PackFile, build_execution_manifest  # noqa: E402
from animapk_cuda.runtime import DecodedReference, StreamingAnimapk  # noqa: E402
from animapk_cuda import upstream as up  # noqa: E402

STAGES = (
    ["pre_x_embedder_tokens", "x_embedder_out", "timestep_emb", "adaln_lora"]
    + [f"block{i:02d}_in" for i in range(28)]
    + [f"block{i:02d}_out" for i in range(28)]
    + ["pre_final_norm", "post_final_projection_patched", "post_unpatchify_velocity"]
)


def load_fixture(fixture_dir):
    x_in = torch.from_numpy(np.fromfile(os.path.join(fixture_dir, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64)
    context = torch.from_numpy(np.fromfile(os.path.join(fixture_dir, "context512.f32"), dtype=np.float32)).view(1, 512, 1024)
    with open(os.path.join(fixture_dir, "sigmas.txt")) as f:
        sigmas = [float(s) for s in f.read().strip().split(",")]
    with open(os.path.join(fixture_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return x_in, context, sigmas, manifest


def save_captures(out_dir, variant, caps):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"caps_{variant}.npz")
    np.savez(path, **{k: v.numpy() for k, v in caps.items()})
    return path


def load_captures(out_dir, variant):
    z = np.load(os.path.join(out_dir, f"caps_{variant}.npz"))
    return {k: torch.from_numpy(z[k]) for k in z.files}


def run_oracle_variant(w, dtype, x_in, context, sigma, device, mode="decoded"):
    provider = lambda keys: w  # noqa: E731
    model = _make_model(provider, dtype, device)
    with torch.no_grad():
        vel, caps = model.forward(x_in, sigma, context)
    return vel, caps


def _make_model(provider, dtype, device):
    from animapk_cuda.runtime import LadderDiT
    return LadderDiT(provider, dtype, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--source", required=True, help="official safetensors")
    ap.add_argument("--fp16-pack", required=True)
    ap.add_argument("--w8-pack", required=True)
    ap.add_argument("--w4-pack", required=True)
    ap.add_argument("--comfy", required=True, help="pinned comfy checkout for variant A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    x_in, context, sigmas, fixture_manifest = load_fixture(args.fixture)
    sigma = sigmas[0]

    provenance = {
        "fixture": fixture_manifest,
        "sigma0": sigma,
        "source": args.source,
        "source_sha256": dso.PINNED_SOURCE_SHA,
        "packs": {},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # ---- A: actual pinned upstream (ComfyUI) bf16 ----
    print("== A: upstream ComfyUI bf16 ==")
    model_up, _ = up.load_upstream(args.comfy, args.source, dtype=torch.bfloat16, device=device)
    vel_a, caps_a = up.capture_forward(model_up, x_in, sigma, context, device=device)
    save_captures(args.out, "A_upstream_bf16", caps_a)

    # ---- A2 / B: source-oracle transcription ----
    print("== A2/B: oracle transcription ==")
    w_src = dso.load_weights(args.source)  # bf16 CPU
    w_src = {k: v.to(device) for k, v in w_src.items()}
    w_bf16 = w_src
    w_fp16 = {k: v.float().half() for k, v in w_src.items()}

    vel_a2, caps_a2 = run_oracle_variant(w_bf16, torch.bfloat16, x_in, context, sigma, device)
    save_captures(args.out, "A2_oracle_bf16", caps_a2)
    vel_b, caps_b = run_oracle_variant(w_fp16, torch.float16, x_in, context, sigma, device)
    save_captures(args.out, "B_oracle_fp16", caps_b)

    # ---- C/D/E: packs, decoded_reference + streaming ----
    packs = {
        "C": args.fp16_pack,
        "D": args.w8_pack,
        "E": args.w4_pack,
    }
    caps_pack = {}
    for variant, path in packs.items():
        print(f"== {variant}: {os.path.basename(path)} ==")
        with PackFile(path) as pk:
            prov = pk.provenance()
            provenance["packs"][variant] = {"path": path, "sha256": prov["sha256"], "size": prov["size"],
                                            "source": prov["source"], "packer": prov["packer"]}
            dr = DecodedReference(pk, torch.float16, device)
            with torch.no_grad():
                vel_dr, caps_dr = dr.forward(x_in, sigma, context)
            save_captures(args.out, f"{variant}_decoded", caps_dr)
            sm = StreamingAnimapk(pk, torch.float16, device)
            with torch.no_grad():
                vel_sm, caps_sm = sm.forward(x_in, sigma, context)
            save_captures(args.out, f"{variant}_streaming", caps_sm)
            caps_pack[variant] = caps_dr

    # ---- per-stage comparisons ----
    variants = {
        "A_upstream_bf16": caps_a,
        "A2_oracle_bf16": caps_a2,
        "B_oracle_fp16": caps_b,
    }
    for v, caps in caps_pack.items():
        variants[f"{v}_decoded"] = caps
        variants[f"{v}_streaming"] = load_captures(args.out, f"{v}_streaming")

    def bundle(base, ref, label):
        c = {}
        for s in STAGES:
            if s in variants[base] and s in variants[ref]:
                c[s] = cmp.metrics(variants[base][s], variants[ref][s])
        c["post_unpatchify_velocity"] = cmp.metrics(variants[base]["post_unpatchify_velocity"],
                                                    variants[ref]["post_unpatchify_velocity"])
        c["__meta"] = {"base": base, "ref": ref, "label": label}
        return c

    tables = {
        "A_upstream_vs_A2_oracle": bundle("A_upstream_bf16", "A2_oracle_bf16", "A (upstream) vs A2 (oracle bf16)"),
        "A_vs_B": bundle("A2_oracle_bf16", "B_oracle_fp16", "A bf16 vs B fp16 (storage precision)"),
        "B_vs_C": bundle("B_oracle_fp16", "C_decoded", "B fp16 source vs C fp16-all pack"),
        "C_vs_D": bundle("C_decoded", "D_decoded", "C fp16-all vs D w8"),
        "C_vs_E": bundle("C_decoded", "E_decoded", "C fp16-all vs E w4"),
        "C_decoded_vs_streaming": bundle("C_decoded", "C_streaming", "C decoded_reference vs streaming"),
        "D_decoded_vs_streaming": bundle("D_decoded", "D_streaming", "D decoded_reference vs streaming"),
        "E_decoded_vs_streaming": bundle("E_decoded", "E_streaming", "E decoded_reference vs streaming"),
    }

    # ---- artifacts ----
    rows = []
    for label, tb in tables.items():
        for s in STAGES:
            if s in tb:
                rows.append({"comparison": label, "stage": s, **{k: tb[s].get(k) for k in ("cosine", "rmse", "rel_l2", "max_abs", "a_norm", "b_norm")}})
    cmp.write_csv(os.path.join(args.out, "precision_ladder_stage_parity.csv"), rows)

    ladder_meta = {label: {"meta": tb["__meta"]} for label, tb in tables.items()}
    for label, tb in tables.items():
        ladder_meta[label]["stages"] = {s: tb[s] for s in STAGES if s in tb}
    cmp.write_json(os.path.join(args.out, "precision_ladder_stage_parity.json"), ladder_meta)

    md = ["# Precision ladder — step-0 stage parity (CUDA)\n"]
    for label, tb in tables.items():
        md.append(f"\n## {label} — {tb['__meta']['label']}\n")
        md.append(cmp.stage_table(STAGES, tb))
    cmp.write_md(os.path.join(args.out, "precision_ladder_stage_parity.md"), "\n".join(md))

    cos_series = {label: {s: tb[s]["cosine"] for s in STAGES if s in tb} for label, tb in tables.items() if label.startswith(("A_", "B_vs", "C_vs"))}
    rel_series = {label: {s: tb[s]["rel_l2"] for s in STAGES if s in tb} for label, tb in tables.items() if label.startswith(("A_", "B_vs", "C_vs"))}
    cmp.plot_ladder(os.path.join(args.out, "precision_ladder_cosine.png"), STAGES, cos_series,
                    "Step-0 stage cosine (CUDA)", "cosine")
    cmp.plot_ladder(os.path.join(args.out, "precision_ladder_relative_l2.png"), STAGES, rel_series,
                    "Step-0 stage relative L2 (CUDA)", "rel L2")

    # ---- source oracle parity report ----
    src_par = {s: cmp.metrics(caps_a[s], caps_a2[s]) for s in STAGES if s in caps_a and s in caps_a2}
    cmp.write_json(os.path.join(args.out, "source_oracle_parity.json"), {"A_vs_A2": src_par})
    cmp.write_md(os.path.join(args.out, "source_oracle_parity.md"),
                 "# Source oracle parity (A upstream ComfyUI vs A2 transcription)\n\n" + cmp.stage_table(STAGES, src_par))

    # ---- decoder parity (pack decode vs official weights) ----
    dec_rows = []
    for variant, path in packs.items():
        with PackFile(path) as pk:
            from animapk_cuda.quant import decode_tensor_from_pack
            for item in pk.tensor_meta:
                name = str(item["name"])
                if not name.startswith("model.diffusion_model."):
                    continue
                sname = name[len("model.diffusion_model."):]
                if sname not in w_src:
                    continue
                src = w_src[sname].float().cpu()
                dec = decode_tensor_from_pack(pk, item, device="cpu")
                if dec.shape != src.shape:
                    dec_rows.append({"variant": variant, "tensor": sname, "error": "shape mismatch"})
                    continue
                m = cmp.metrics(dec, src)
                dec_rows.append({"variant": variant, "tensor": sname, "storage": item["storage_dtype"],
                                 "shape": list(src.shape), **{k: m[k] for k in ("cosine", "rmse", "rel_l2", "max_abs")}})
    cmp.write_csv(os.path.join(args.out, "animapk_decoder_parity.csv"), dec_rows)
    dec_md = ["# Animapk decoder parity — pack decode vs official source weights\n"]
    for v in packs:
        sub = [r for r in dec_rows if r["variant"] == v]
        worst = sorted(sub, key=lambda r: r.get("cosine", 0))[:5]
        dec_md.append(f"\n## {v}\n- tensors checked: {len(sub)}\n- worst 5 by cosine:")
        for r in worst:
            dec_md.append(f"  - {r['tensor']} [{r.get('storage')}] cos {r.get('cosine', float('nan')):.6f} rmse {r.get('rmse', float('nan')):.4e}")
    cmp.write_md(os.path.join(args.out, "animapk_decoder_parity.md"), "\n".join(dec_md))
    cmp.write_json(os.path.join(args.out, "animapk_decoder_parity.json"), {"rows": dec_rows})

    # ---- streaming manifest + runtime provenance ----
    with PackFile(args.fp16_pack) as pk:
        manifest = build_execution_manifest(pk, "block")
    cmp.write_json(os.path.join(args.out, "animapk_execution_manifest.json"), manifest)

    cmp.write_json(os.path.join(args.out, "animapk_runtime_provenance.json"),
                   {"modes": ["decoded_reference", "streaming_animapk"], "device": device,
                    "graph": "dit_source_oracle transcription", "pack_grouping": "block"})
    cmp.write_json(os.path.join(args.out, "provenance.json"), provenance)

    summary = {}
    for label, tb in tables.items():
        v = tb.get("post_unpatchify_velocity", {})
        summary[label] = {"velocity_cosine": v.get("cosine"), "velocity_rel_l2": v.get("rel_l2"),
                          "velocity_max_abs": v.get("max_abs")}
    print(json.dumps(summary, indent=2))
    cmp.write_json(os.path.join(args.out, "ladder_summary.json"), summary)


if __name__ == "__main__":
    main()
