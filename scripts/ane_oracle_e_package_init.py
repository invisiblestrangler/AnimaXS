"""Package entry point for the Oracle E ComfyUI node.

The core node remains valid for D->E controls using Oracle-V2 preparation and
conditioning. For physical A12 parity, the runner supplies four raw FP32 files:

* exact device step-0 prepared residual [1024,2048]
* exact device step-0 embedding [2048]
* exact device step-0 AdaLN-LoRA base [6144]
* exact device adapter cross-context [512,1024]

This thin layer injects those values only into the *first* 28-block traversal.
The residual is injected only at block 0; the embedding, AdaLN-LoRA base and
cross-context are re-injected for every block because the source MiniTrainDIT
passes those global values separately into each block. That removes
text-encoder/adapter/preparation drift from E2-vs-A12 and leaves the comparison
scoped to the hybrid DiT blocks themselves.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import torch

from .ane_oracle_e_custom_node import *  # noqa: F401,F403
from . import ane_oracle_e_custom_node as _core

_DEVICE_SPECS = {
    "cross_context": ("ANIMA_ORACLE_E_CROSS_CONTEXT", 512 * 1024),
    "prepared_residual": ("ANIMA_ORACLE_E_PREPARED_RESIDUAL", 1024 * 2048),
    "prepared_embedding": ("ANIMA_ORACLE_E_PREPARED_EMBEDDING", 2048),
    "prepared_adaln_lora": ("ANIMA_ORACLE_E_PREPARED_ADALN_LORA", 6144),
}
_ORIGINAL_ARM = _core.AnimaOracleEArm.arm


def _configured_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    configured = []
    for name, (env_name, _count) in _DEVICE_SPECS.items():
        text = os.environ.get(env_name, "")
        configured.append(bool(text))
        if text:
            paths[name] = Path(text)
    if any(configured) and not all(configured):
        missing = [
            env_name for name, (env_name, _count) in _DEVICE_SPECS.items()
            if name not in paths
        ]
        raise RuntimeError(
            "Oracle E device parity requires all prepared-state overrides; missing "
            + ", ".join(missing))
    return paths


def _load_device_arrays(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Oracle E device payload missing: {path}")
        expected = _DEVICE_SPECS[name][1]
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != expected:
            raise RuntimeError(
                f"Oracle E device payload {name} has {raw.size} floats; expected {expected}")
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Oracle E device payload {name} contains non-finite values")
        arrays[name] = raw
    return arrays


def _arm_with_optional_device_state(self, model, mode):
    result = _ORIGINAL_ARM(self, model, mode)
    paths = _configured_paths()
    if not paths:
        return result

    arrays = _load_device_arrays(paths)
    patcher = result[0]
    dm = _core._locate_diffusion_model(patcher)
    blocks = list(dm.blocks)
    state = getattr(dm, "_anima_oracle_e_state")
    handles = getattr(dm, "_anima_oracle_e_handles")
    cache: dict[tuple[str, str, tuple[int, ...]], torch.Tensor] = {}
    first_block0_pending = True

    def tensor_for(name: str, template: torch.Tensor) -> torch.Tensor:
        raw = arrays[name]
        if template.numel() != raw.size:
            raise RuntimeError(
                f"Oracle E {name} shape mismatch: source {tuple(template.shape)} has "
                f"{template.numel()} elements, device payload has {raw.size}")
        key = (name, str(template.device), tuple(template.shape))
        replacement = cache.get(key)
        if replacement is None:
            replacement = torch.from_numpy(raw.copy()).reshape(template.shape)
            replacement = replacement.to(device=template.device, dtype=torch.float32)
            cache[key] = replacement
        return replacement

    def replace_global_block_inputs(args):
        if len(args) < 5:
            raise RuntimeError(
                "Oracle E block call must expose residual, embedding, cross context, rope, and AdaLN as positional args")
        mutable = list(args)
        mutable[1] = tensor_for("prepared_embedding", mutable[1])
        mutable[2] = tensor_for("cross_context", mutable[2])
        if mutable[4] is None:
            raise RuntimeError("Oracle E expected non-nil AdaLN-LoRA block input")
        mutable[4] = tensor_for("prepared_adaln_lora", mutable[4])
        return mutable

    def block0_device_hook(_module, args):
        nonlocal first_block0_pending
        if not first_block0_pending:
            return None
        mutable = replace_global_block_inputs(args)
        mutable[0] = tensor_for("prepared_residual", mutable[0])
        first_block0_pending = False
        return tuple(mutable)

    def later_block_device_hook(_module, args):
        # Core's block0 traversal hook has already incremented traversal to 0
        # when blocks 1...27 execute. Do not override any later diffusion step.
        if state.traversal != 0:
            return None
        return tuple(replace_global_block_inputs(args))

    # Registered after the core hook with prepend=True, therefore this block0
    # hook executes first and replaces the prepared values before core advances
    # traversal -1 -> 0.
    handles.append(blocks[0].register_forward_pre_hook(block0_device_hook, prepend=True))
    for block in blocks[1:]:
        handles.append(block.register_forward_pre_hook(later_block_device_hook, prepend=True))

    arm_path = _core.RUN_DIR / "oracle_e_arm.json"
    if arm_path.is_file():
        meta = json.loads(arm_path.read_text())
        meta["device_step0_override"] = {
            name: {
                "path": str(paths[name]),
                "elements": int(arrays[name].size),
                "dtype": "float32-le",
            }
            for name in sorted(paths)
        }
        meta["device_override_scope"] = (
            "first 28-block traversal only; residual at block0, embedding/AdaLN/cross at all 28 blocks")
        arm_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return result


_core.AnimaOracleEArm.arm = _arm_with_optional_device_state
AnimaOracleEArm = _core.AnimaOracleEArm
NODE_CLASS_MAPPINGS = {"AnimaOracleEArm": AnimaOracleEArm}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaOracleEArm": "Anima Oracle E Arm"}
