#!/usr/bin/env python3
"""Extract the small committed A006 fixture set from the canonical NPZ."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_GOLDEN_SHA256 = "44d35d4f788c0a48411b0e68db66a84a79a8dcd8ef3beb842d800ceaff81a8dc"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden", type=Path,
        default=Path("/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz"))
    parser.add_argument(
        "--out", type=Path, default=Path("AnimaXSTests/Fixtures/Case1Binary"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    golden = np.load(args.golden)

    prompt = (
        "masterpiece, best quality, score_7, safe, 1girl, long brown hair, blue eyes, "
        "school uniform, cherry blossom, outdoors, looking at viewer, smile, depth of "
        "field, highres, absurdres"
    )
    arrays = {
        "noise": golden["init_noise_randn"].astype("<f4"),
        "final_latent": golden["final_latent"].astype("<f4"),
        "cond_context": golden["cond_context"].astype("<f4"),
        "t5_ids": golden["cond_meta_t5xxl_ids"].astype("<i8"),
    }
    source_hash = digest(args.golden.read_bytes())
    if source_hash != EXPECTED_GOLDEN_SHA256:
        raise RuntimeError(f"unexpected canonical golden SHA-256: {source_hash}")
    metadata = {
        "source": args.golden.name,
        "source_sha256": source_hash,
        "prompt": prompt,
        "seed": 1337,
        "arrays": {},
        "sigmas": golden["sigmas_comfy"].astype(np.float32).tolist(),
        "attention_mask_shape": list(golden["cond_meta_attention_mask"].shape),
        "attention_mask_sha256": digest(
            golden["cond_meta_attention_mask"].astype("<f4").tobytes()),
        "anchors": {},
        "legacy_step_callback_warning": (
            "step_latents is internally inconsistent with final_latent/Euler; see D055. "
            "These anchors identify the legacy trace only and are not an I002 parity gate."
        ),
    }
    for name, array in arrays.items():
        raw = array.tobytes(order="C")
        filename = f"case1_{name}.{'i64' if name == 't5_ids' else 'f32'}"
        (args.out / filename).write_bytes(raw)
        metadata["arrays"][name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": len(raw),
            "sha256": digest(raw),
        }

    for key in ["step_latents", "block_00_out", "block_15_out", "block_27_out", "decoded_rgb"]:
        array = golden[key].astype(np.float32)
        metadata["anchors"][key] = {
            "shape": list(array.shape),
            "sha256": digest(array.astype("<f4").tobytes()),
            "first16": array.reshape(-1)[:16].tolist(),
        }
    output = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    (args.out / "fixtures.json").write_text(output)
    total = sum(path.stat().st_size for path in args.out.iterdir())
    if total > 3 * 1_024 * 1_024:
        raise RuntimeError(f"fixture budget exceeded: {total} bytes")
    print(f"wrote {len(arrays)} arrays + metadata, {total} bytes")


if __name__ == "__main__":
    main()
