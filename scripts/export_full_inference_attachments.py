#!/usr/bin/env python3
"""Export xcresulttool image attachments into canonical artifact filenames.

Temporary capture-branch helper for the AnimaXS full-inference image artifact.
Reads the manifest.json produced by `xcresulttool export attachments` and maps
each exported UUID-named file to its canonical artifact name using the
suggested human-readable name recorded by the tool.

Usage:
  export_full_inference_attachments.py <exported_dir> <artifacts_dir>

Required outputs in <artifacts_dir>:
  generated.png, reference.png, comparison.png, metrics.txt, pack-metadata.json
"""

import json
import os
import shutil
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: export_full_inference_attachments.py <exported_dir> <artifacts_dir>")
        return 2
    exported_dir = sys.argv[1]
    artifacts_dir = sys.argv[2]
    os.makedirs(artifacts_dir, exist_ok=True)

    # canonical artifact name -> exported file name (UUID) from manifest.json
    targets = {
        "generated": "generated.png",
        "reference": "reference.png",
        "comparison": "comparison.png",
        "metrics": "metrics.txt",
        "pack-metadata": "pack-metadata.json",
    }
    found = {}
    exported_by_suggested = {}

    manifest_path = os.path.join(exported_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)
        # Schema varies by Xcode version. Observed (Xcode 26.3): a list of
        # objects each with an `attachments` array; each attachment is a dict
        # with exportedFileName + suggestedHumanReadableName. Older versions
        # may use a flat dict {id: entry} or a flat list of entries. Walk all
        # nested dicts and collect every exported-file/suggested-name pair.
        stack = [manifest]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
                fname = node.get("exportedFileName") or node.get("filename") or ""
                sname = node.get("suggestedHumanReadableName") or node.get("suggestedName") or ""
                if fname and sname:
                    exported_by_suggested[sname] = fname
            elif isinstance(node, list):
                stack.extend(node)
        for sname, fname in exported_by_suggested.items():
            for prefix, canonical in targets.items():
                if sname.startswith(prefix) and canonical not in found:
                    found[canonical] = fname

    # Fallback: match by file content where possible. metrics.txt is the only
    # .txt; the three PNGs cannot be told apart without the manifest, so this
    # only covers metrics when the manifest is missing.
    if "metrics.txt" not in found:
        for fname in os.listdir(exported_dir):
            if fname.endswith(".txt"):
                found["metrics.txt"] = fname

    for canonical, fname in sorted(found.items()):
        src = os.path.join(exported_dir, fname)
        dst = os.path.join(artifacts_dir, canonical)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            print(f"mapped {fname} -> {canonical}")
        else:
            print(f"WARNING: exported file missing: {src}")

    # Trajectory/source-oracle captures (per-step x_in/denoised, fp32
    # cross-context, sigma list) are exported under their own names so the
    # Linux source oracle can consume them without renaming. XCTest mangles
    # suggested names with a "_0_<UUID>" suffix; strip it back to the
    # canonical name (e.g. step07_x_in_0_UUID.f32 -> step07_x_in.f32).
    for sname, fname in exported_by_suggested.items():
        lname = sname.lower()
        if (sname.startswith("step") and sname.endswith(".f32")) \
                or lname.startswith("cross-context") or lname.startswith("sigmas"):
            src = os.path.join(exported_dir, fname)
            dst = os.path.join(artifacts_dir, sname)
            if os.path.isfile(src):
                shutil.copyfile(src, dst)
                print(f"mapped {fname} -> {sname}")
            canonical = sname.split("_0_")[0]
            if canonical != sname:
                cdst = os.path.join(artifacts_dir, canonical)
                if not os.path.isfile(cdst):
                    shutil.copyfile(src, cdst)
                    print(f"mapped {fname} -> {canonical}")

    # Fallback for a missing manifest: scan exported files by extension only
    # (ambiguous but better than nothing for the f32 trajectory tensors).
    if not exported_by_suggested:
        for fname in os.listdir(exported_dir):
            if fname.startswith("step") and fname.endswith(".f32"):
                shutil.copyfile(os.path.join(exported_dir, fname),
                                os.path.join(artifacts_dir, fname))
                print(f"fallback mapped {fname}")

    missing = [c for c in targets.values() if not os.path.isfile(os.path.join(artifacts_dir, c))]
    if missing:
        print(f"ERROR: missing canonical artifacts: {missing}")
        return 1
    print("all canonical artifacts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
