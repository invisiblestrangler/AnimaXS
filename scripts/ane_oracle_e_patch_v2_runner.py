#!/usr/bin/env python3
"""Create an Oracle-E runner by minimally patching the proven Oracle-V2 runner.

Only two semantic edits are allowed:
  1. instrument choices become E1/E2;
  2. the graph arm node becomes AnimaOracleEArm.

Everything else (pinned ComfyUI launch, reconstructed checkpoint selection,
prompt, sampler settings, exact-noise load, output collection) remains the
Oracle-V2 implementation.  The script fails closed if the expected anchors are
not present exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    original = source.read_bytes()
    text = original.decode("utf-8")

    text = replace_once(
        text,
        'ap.add_argument("--instrument",choices=["none","capture","fp16_boundaries"],default="none")',
        'ap.add_argument("--instrument",choices=["none","E1_native_ane","E2_device_residual"],default="none")',
        "instrument choices",
    )
    text = replace_once(
        text,
        '"class_type":"AnimaOracleArm"',
        '"class_type":"AnimaOracleEArm"',
        "arm node",
    )

    output.write_text(text, encoding="utf-8")
    manifest = output.with_suffix(output.suffix + ".patch.txt")
    manifest.write_text(
        "source=" + str(source) + "\n"
        + "source_sha256=" + sha256_bytes(original) + "\n"
        + "output=" + str(output) + "\n"
        + "output_sha256=" + sha256_bytes(text.encode("utf-8")) + "\n"
        + "semantic_edits=2\n"
        + "edit_1=instrument choices only\n"
        + "edit_2=AnimaOracleArm -> AnimaOracleEArm\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
