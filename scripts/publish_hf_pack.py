#!/usr/bin/env python3
"""Publish one verified refinement pack and its small provenance evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--sha-file", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    identity = api.whoami()
    owner = identity["name"]
    repo_id = f"{owner}/{args.suffix}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    pack = Path(args.pack)
    report = Path(args.report)
    verification = Path(args.verification)
    source_sha = os.environ.get("ANIMA_SOURCE_SHA256", "unknown")
    source_revision = os.environ.get("ANIMA_SOURCE_REVISION", "unknown")
    output_sha = sha256(pack)
    manifest = {
        "variant": args.variant,
        "repository": repo_id,
        "source": {
            "repo": os.environ.get("ANIMA_SOURCE_REPO", "circlestone-labs/Anima"),
            "revision": source_revision,
            "path": os.environ.get("ANIMA_SOURCE_FILE", "split_files/diffusion_models/anima-turbo-v1.0.safetensors"),
            "sha256": source_sha,
        },
        "output": {
            "filename": pack.name,
            "bytes": pack.stat().st_size,
            "sha256": output_sha,
        },
        "git_commit": os.environ.get("GITHUB_SHA", "local"),
    }
    out = pack.parent
    manifest_path = out / f"{args.variant}.packing-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with tempfile.TemporaryDirectory(prefix="animaxs-hf-") as temporary:
        staging = Path(temporary)
        shutil.copy2(report, staging / "quant-report.json")
        shutil.copy2(verification, staging / "verification-report.json")
        shutil.copy2(manifest_path, staging / "packing-manifest.json")
        shutil.copy2(Path(args.sha_file), staging / "SHA256SUMS")
        shutil.copy2(Path("MODEL_LICENSE.md"), staging / "MODEL_LICENSE.md")
        shutil.copy2(Path("MODEL_NOTICE.txt"), staging / "MODEL_NOTICE.txt")
        licenses = staging / "licenses"
        licenses.mkdir()
        shutil.copy2(Path("docs/model-licenses/CircleStone-NC-1.2.md"), licenses / "CircleStone-NC-1.2.md")
        shutil.copy2(Path("docs/model-licenses/NVIDIA-Open-Model-License.txt"), licenses / "NVIDIA-Open-Model-License.txt")
        readme = f"""# AnimaXS {args.variant} DiT pack

This is an experimental AnimaXS converted/quantized derivative for the iPhone XS Max / Apple Metal runtime.

- Non-commercial model use only; see `MODEL_LICENSE.md` and `MODEL_NOTICE.txt`.
- Source: `circlestone-labs/Anima`, revision `{source_revision}`.
- Source SHA-256: `{source_sha}`.
- Packing: ANMA v1 container, bounded-memory packer v2, group size 64, `{args.variant}`.
- Output SHA-256: `{output_sha}`.
- Built on NVIDIA Cosmos; this is not an official CircleStone release.

The accompanying reports and license files are part of the derivative's provenance record.
        """
        (staging / "README.md").write_text(readme)
        # Upload the multi-GB pack directly from its planned output path. The
        # staging directory contains only small evidence/license files.
        api.upload_file(
            path_or_fileobj=str(pack), path_in_repo=pack.name,
            repo_id=repo_id, repo_type="model",
            commit_message=f"Upload {args.variant} ANIMAPK")
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(staging),
            commit_message=f"Upload {args.variant} ANIMAPK")

    publish = {
        "repository": repo_id,
        "revision": getattr(commit, "oid", None) or getattr(commit, "commit_id", None),
        "variant": args.variant,
        "filename": pack.name,
        "bytes": pack.stat().st_size,
        "sha256": output_sha,
    }
    publish_path = out / f"{args.variant}.hf-publish.json"
    publish_path.write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")
    print(json.dumps(publish, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
