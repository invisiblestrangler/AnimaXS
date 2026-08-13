#!/usr/bin/env python3
"""Map UUID-named xcresult attachments to canonical block-0 filenames."""
import json
import shutil
import sys
from pathlib import Path

BLOCKS = (0, 7, 14, 21, 27)
TARGETS = (
    "w8-step0-embedding.f32",
    "w8-step0-adaln.f32",
    "w8-cross-context.f16",
    "w8-rope.f32",
) + tuple(
    f"w8-step0-block{block}-{stage}.f32"
    for block in BLOCKS
    for stage in ("input", "after-self", "after-cross", "after-mlp", "output")
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_block0_attachments.py EXPORTED_DIR OUTPUT_DIR")
    exported, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((exported / "manifest.json").read_text())
    pairs = []
    stack = [manifest]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
            filename = node.get("exportedFileName") or node.get("filename")
            suggested = node.get("suggestedHumanReadableName") or node.get("suggestedName")
            if filename and suggested:
                pairs.append((str(filename), str(suggested)))
        elif isinstance(node, list):
            stack.extend(node)
    for target in TARGETS:
        stem, suffix = target.rsplit(".", 1)
        matches = [filename for filename, suggested in pairs
                   if suggested == target or
                   (suggested.startswith(stem + "_") and suggested.endswith("." + suffix))]
        if len(matches) != 1:
            raise SystemExit(f"{target}: expected one manifest match, found {matches}")
        source = exported / matches[0]
        if not source.is_file():
            raise SystemExit(f"{target}: exported file missing: {source}")
        shutil.copyfile(source, output / target)
        print(f"mapped {matches[0]} -> {target}")


if __name__ == "__main__":
    main()
