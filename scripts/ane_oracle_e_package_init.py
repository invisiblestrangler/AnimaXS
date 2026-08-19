"""Package entry point for the Oracle E ComfyUI node.

The core node remains valid for D->E controls using Oracle-V2 conditioning.
When ANIMA_ORACLE_E_CROSS_CONTEXT points at a device-exported raw FP32
512x1024 buffer, this thin layer additionally replaces each block's external
cross-attention context before the core forward runs.  That makes A12-vs-E2
comparisons independent of Qwen/adapter quantization differences.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch

from .ane_oracle_e_custom_node import *  # noqa: F401,F403
from . import ane_oracle_e_custom_node as _core

_CROSS_PATH_TEXT = os.environ.get("ANIMA_ORACLE_E_CROSS_CONTEXT", "")
_CROSS_PATH = Path(_CROSS_PATH_TEXT) if _CROSS_PATH_TEXT else None
_EXPECTED_CROSS_ELEMENTS = 512 * 1024
_ORIGINAL_ARM = _core.AnimaOracleEArm.arm


def _arm_with_optional_device_context(self, model, mode):
    result = _ORIGINAL_ARM(self, model, mode)
    if _CROSS_PATH is None:
        return result
    if not _CROSS_PATH.is_file():
        raise RuntimeError(f"Oracle E device cross-context missing: {_CROSS_PATH}")
    raw = np.fromfile(_CROSS_PATH, dtype="<f4")
    if raw.size != _EXPECTED_CROSS_ELEMENTS:
        raise RuntimeError(
            f"Oracle E device cross-context has {raw.size} floats; "
            f"expected {_EXPECTED_CROSS_ELEMENTS}")
    if not np.isfinite(raw).all():
        raise RuntimeError("Oracle E device cross-context contains non-finite values")

    patcher = result[0]
    dm = _core._locate_diffusion_model(patcher)
    blocks = list(dm.blocks)
    cache: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}

    def context_hook(_module, args):
        mutable = list(args)
        if len(mutable) < 3:
            raise RuntimeError("Oracle E block call did not expose cross context as arg 2")
        template = mutable[2]
        if template.numel() != _EXPECTED_CROSS_ELEMENTS:
            raise RuntimeError(
                f"Oracle E cross-context shape mismatch: {tuple(template.shape)} "
                f"has {template.numel()} elements")
        key = (str(template.device), tuple(template.shape))
        replacement = cache.get(key)
        if replacement is None:
            replacement = torch.from_numpy(raw.copy()).reshape(template.shape)
            # Keep the device-exported FP32 values until the native projection
            # wrapper performs the real ANE FP16 input-surface roundtrip.
            replacement = replacement.to(device=template.device, dtype=torch.float32)
            cache[key] = replacement
        mutable[2] = replacement
        return tuple(mutable)

    handles = getattr(dm, "_anima_oracle_e_handles")
    for block in blocks:
        handles.append(block.register_forward_pre_hook(context_hook, prepend=True))

    arm_path = _core.RUN_DIR / "oracle_e_arm.json"
    if arm_path.is_file():
        import json
        meta = json.loads(arm_path.read_text())
        meta["device_cross_context_override"] = str(_CROSS_PATH)
        meta["device_cross_context_elements"] = int(raw.size)
        meta["device_cross_context_dtype"] = "float32-le"
        arm_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return result


_core.AnimaOracleEArm.arm = _arm_with_optional_device_context
AnimaOracleEArm = _core.AnimaOracleEArm
NODE_CLASS_MAPPINGS = {"AnimaOracleEArm": AnimaOracleEArm}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaOracleEArm": "Anima Oracle E Arm"}
