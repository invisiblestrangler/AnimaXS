#!/usr/bin/env python3
"""Collect small Oracle V2/D textual evidence from Hugging Face.

This is deliberately metadata/text-only. It never downloads the multi-GB
reconstructed checkpoint or large latent arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


TEXT_SUFFIXES = {".json", ".md", ".txt", ".py", ".csv", ".yaml", ".yml"}
KEYWORDS = (
    "oracle", "boundary", "manifest", "report", "golden", "provenance",
    "run", "summary", "metric", "capture", "probe", "script", "decision",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--v2-prefix", required=True)
    ap.add_argument("--d-prefix", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--max-file-bytes", type=int, default=2_000_000)
    ap.add_argument("--max-total-bytes", type=int, default=20_000_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    files_dir = out / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=args.token)
    entries = list(api.list_repo_tree(
        repo_id=args.repo,
        path_in_repo=args.v2_prefix,
        recursive=True,
        expand=True,
        repo_type="model",
    ))
    files = [e for e in entries if isinstance(e, RepoFile)]
    manifest = [
        {
            "path": f.path,
            "size": int(f.size or 0),
            "blob_id": getattr(f, "blob_id", None),
            "lfs": getattr(f, "lfs", None),
        }
        for f in files
    ]
    (out / "ORACLE_V2_TREE.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    selected = []
    total = 0
    for f in sorted(files, key=lambda x: x.path):
        size = int(f.size or 0)
        suffix = Path(f.path).suffix.lower()
        lower = f.path.lower()
        is_d = f.path.startswith(args.d_prefix + "/")
        interesting = is_d or any(k in lower for k in KEYWORDS)
        if not interesting or suffix not in TEXT_SUFFIXES:
            continue
        if size <= 0 or size > args.max_file_bytes:
            continue
        if total + size > args.max_total_bytes:
            continue
        selected.append(f)
        total += size

    dump = []
    selected_manifest = []
    for i, f in enumerate(selected, 1):
        print(f"[{i:03d}/{len(selected)}] {f.path} ({f.size} bytes)", flush=True)
        cached = hf_hub_download(
            repo_id=args.repo,
            filename=f.path,
            repo_type="model",
            token=args.token,
        )
        rel = Path(f.path).relative_to(args.v2_prefix)
        dest = files_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        raw = dest.read_bytes()
        selected_manifest.append({
            "path": f.path,
            "size": len(raw),
            "local": str(dest.relative_to(out)),
        })
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        dump.append(
            f"\n===== BEGIN {f.path} =====\n{text}\n===== END {f.path} =====\n"
        )

    (out / "SELECTED_MANIFEST.json").write_text(
        json.dumps(selected_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out / "SELECTED_TEXT_DUMP.txt").write_text("".join(dump), encoding="utf-8")

    d_files = [m for m in manifest if m["path"].startswith(args.d_prefix + "/")]
    (out / "ORACLE_D_TREE.json").write_text(
        json.dumps(d_files, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"V2 files={len(files)} D files={len(d_files)} selected={len(selected)} bytes={total}")


if __name__ == "__main__":
    main()
