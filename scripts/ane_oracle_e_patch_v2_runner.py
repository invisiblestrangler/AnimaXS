#!/usr/bin/env python3
"""Create an Oracle-E runner by minimally patching the proven Oracle-V2 runner.

Semantic edits are intentionally narrow and fail closed:
  1. instrument choices become E1/E2;
  2. the graph arm node becomes AnimaOracleEArm;
  3. generation width/height may be overridden by Oracle-E environment vars.

The third edit is inert for ordinary Oracle-V2/D->E controls. It exists only so
a physical-device parity run can use the XS Max 512x512 / 1024-token geometry
instead of Oracle V2's original 1024x1024 / 4096-token geometry.

Everything else (pinned ComfyUI launch, reconstructed checkpoint selection,
prompt, sampler settings, exact-noise load, output collection) remains the
Oracle-V2 implementation. The script fails closed if any expected anchor is not
present exactly once.
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
    text = replace_once(
        text,
        '    g = CFG["generation"]',
        '    g = dict(CFG["generation"])\n'
        '    g["width"] = int(os.environ.get("ANIMA_ORACLE_E_WIDTH", g["width"]))\n'
        '    g["height"] = int(os.environ.get("ANIMA_ORACLE_E_HEIGHT", g["height"]))',
        "Oracle E geometry override",
    )

    output.write_text(text, encoding="utf-8")
    manifest = output.with_suffix(output.suffix + ".patch.txt")
    manifest.write_text(
        "source=" + str(source) + "\n"
        + "source_sha256=" + sha256_bytes(original) + "\n"
        + "output=" + str(output) + "\n"
        + "output_sha256=" + sha256_bytes(text.encode("utf-8")) + "\n"
        + "semantic_edits=3\n"
        + "edit_1=instrument choices only\n"
        + "edit_2=AnimaOracleArm -> AnimaOracleEArm\n"
        + "edit_3=optional ANIMA_ORACLE_E_WIDTH/HEIGHT generation geometry override\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
