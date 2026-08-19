# AnimaXS Oracle E ComfyUI custom node.
#
# E1 isolates the real hybrid pack's ANE-native projection arithmetic from
# Oracle D.  E2 adds the iPhone's FP32 residual-stream policy.  The actual
# pinned ComfyUI Anima forward remains in control of all non-projection math;
# this node patches only the ten ANE projection modules per block, the exact
# half->float GELU->half handoff, Q/K/V post-normalization surface boundaries,
# and bounded diagnostic hooks.
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

try:
    from .ane_oracle_e_pack_reference import (
        ANE_TENSOR_FORMAT,
        DEFAULT_SAMPLE_COUNT,
        AnimapkReference,
        deterministic_stride,
    )
except Exception:
    from scripts.ane_oracle_e_pack_reference import (
        ANE_TENSOR_FORMAT,
        DEFAULT_SAMPLE_COUNT,
        AnimapkReference,
        deterministic_stride,
    )

PINNED_PACK_SHA256 = "f5c80a25114b62a6807996180d439c5d12828d7392c604e1eee15acb28977dc4"
PINNED_PACK_BYTES = 2_128_838_656
MODES = ("E1_native_ane", "E2_device_residual")
RUN_DIR = Path(os.environ.get("ANIMA_ORACLE_E_RUN_DIR", "/tmp/anima_oracle_e"))
RUN_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_COUNT = int(os.environ.get("ANIMA_ORACLE_E_SAMPLE_COUNT", str(DEFAULT_SAMPLE_COUNT)))
OUTPUT_ROW_CHUNK = int(os.environ.get("ANIMA_ORACLE_E_OUTPUT_ROW_CHUNK", "256"))
PACK_PATH = Path(os.environ.get("ANIMA_ORACLE_E_PACK", ""))
EXPECTED_PACK_SHA256 = os.environ.get("ANIMA_ORACLE_E_PACK_SHA256", PINNED_PACK_SHA256).lower()
NOISE_MODE = os.environ.get("ANIMA_ORACLE_E_NOISE_MODE", os.environ.get("ANIMA_ORACLE_NOISE_MODE", "none"))
NOISE_FILE = Path(os.environ.get(
    "ANIMA_ORACLE_E_NOISE_FILE",
    os.environ.get("ANIMA_ORACLE_NOISE_FILE", str(RUN_DIR / "initial_noise.safetensors")),
))

if SAMPLE_COUNT <= 0:
    raise RuntimeError("ANIMA_ORACLE_E_SAMPLE_COUNT must be positive")
if OUTPUT_ROW_CHUNK <= 0:
    raise RuntimeError("ANIMA_ORACLE_E_OUTPUT_ROW_CHUNK must be positive")
if NOISE_MODE not in ("none", "save", "load"):
    raise RuntimeError(f"invalid ANIMA_ORACLE_E_NOISE_MODE={NOISE_MODE!r}")

_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _locate_diffusion_model(patcher: Any) -> Any:
    base = getattr(patcher, "model", None)
    if base is None:
        raise RuntimeError("Oracle E: MODEL has no .model")
    dm = getattr(base, "diffusion_model", None)
    if dm is not None:
        return dm
    inner = getattr(base, "model", None)
    if inner is not None:
        dm = getattr(inner, "diffusion_model", None)
        if dm is not None:
            return dm
    raise RuntimeError("Oracle E: could not locate diffusion_model on MODEL")


def _native_name(block: int, suffix: str) -> str:
    return f"model.diffusion_model.blocks.{block}.{suffix}.weight"


def _native_modules(block: Any) -> list[tuple[str, Any]]:
    return [
        ("self_attn.q_proj", block.self_attn.q_proj),
        ("self_attn.k_proj", block.self_attn.k_proj),
        ("self_attn.v_proj", block.self_attn.v_proj),
        ("self_attn.output_proj", block.self_attn.output_proj),
        ("cross_attn.q_proj", block.cross_attn.q_proj),
        ("cross_attn.k_proj", block.cross_attn.k_proj),
        ("cross_attn.v_proj", block.cross_attn.v_proj),
        ("cross_attn.output_proj", block.cross_attn.output_proj),
        ("mlp.layer1", block.mlp.layer1),
        ("mlp.layer2", block.mlp.layer2),
    ]


def _ane_linear_torch(pack: AnimapkReference, name: str, x: torch.Tensor) -> torch.Tensor:
    """FP16 surface I/O + exact q*FP32-scale+FP32-bias weight reconstruction.

    Matmul is deliberately FP32 with TF32 disabled by the arm step.  That is a
    reference arithmetic choice, not a claim that Apple's private ANE uses
    instruction-for-instruction FP32 accumulation.
    """
    rec = pack.record(name)
    if rec.quantization_format != ANE_TENSOR_FORMAT or len(rec.shape) != 2:
        raise RuntimeError(f"Oracle E native projection contract mismatch: {name}")
    out_features, in_features = rec.shape
    if x.shape[-1] != in_features:
        raise RuntimeError(
            f"Oracle E input K mismatch for {name}: {x.shape[-1]} != {in_features}")
    original_dtype = x.dtype
    x_surface = x.to(torch.float16).reshape(-1, in_features).to(torch.float32)
    out = torch.empty(
        (x_surface.shape[0], out_features), device=x.device, dtype=torch.float32
    )
    for row0 in range(0, out_features, OUTPUT_ROW_CHUNK):
        row1 = min(out_features, row0 + OUTPUT_ROW_CHUNK)
        q_np, scale_np, bias_np = pack.native_rows(name, row0, row1)
        q = torch.from_numpy(q_np).to(device=x.device, dtype=torch.float32)
        scale = torch.from_numpy(scale_np).to(device=x.device, dtype=torch.float32)
        bias = torch.from_numpy(bias_np).to(device=x.device, dtype=torch.float32)
        weight = q.mul_(scale[:, None]).add_(bias[:, None])
        out[:, row0:row1] = torch.mm(x_surface, weight.t())
    # The private-runtime output IOSurface is FP16.  Return in the caller's
    # dtype only after that storage roundtrip so the source model can continue.
    out = out.to(torch.float16).to(original_dtype)
    return out.reshape(*x.shape[:-1], out_features)


def _fp16_rt(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


def _device_gelu(x: torch.Tensor) -> torch.Tensor:
    # Mirrors current baseline ANE path: half surface -> FP32 exact-erf GELU ->
    # half surface.  CUDA's exact GELU is the control; Metal's local erf helper
    # may differ by a few ulps and is therefore not asserted bit-identical.
    original_dtype = x.dtype
    y = F.gelu(x.to(torch.float16).to(torch.float32), approximate="none")
    return y.to(torch.float16).to(original_dtype)


def _checkpoint_stats(x: torch.Tensor) -> dict[str, Any]:
    flat = x.detach().reshape(-1).to(torch.float32)
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(flat).sum().item())
    posinf_count = int(torch.isposinf(flat).sum().item())
    neginf_count = int(torch.isneginf(flat).sum().item())
    if finite_count:
        work = flat if finite_count == flat.numel() else flat[finite]
        minimum = float(work.min().item())
        maximum = float(work.max().item())
        mean = float(work.mean().item())
        std = float(work.std(unbiased=False).item())
        l2 = float(torch.linalg.vector_norm(work).item())
        max_abs = float(work.abs().max().item())
    else:
        minimum = maximum = mean = std = l2 = max_abs = float("nan")
    return {
        "element_count": int(flat.numel()),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "l2": l2,
        "max_abs": max_abs,
    }


class _CaptureState:
    def __init__(self, mode: str, pack_sha256: str):
        self.mode = mode
        self.pack_sha256 = pack_sha256
        self.traversal = -1
        self.records: list[dict[str, Any]] = []
        self.checkpoint_dir = RUN_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def begin_traversal(self) -> None:
        self.traversal += 1

    def capture(self, block: int, branch: str, tensor: torch.Tensor) -> None:
        # CFG is pinned to 1.0 in the Oracle V2 graph, therefore traversal 0 is
        # exactly diffusion step 0.  Later traversals run normally but are not
        # exported in this localization pass.
        if self.traversal != 0:
            return
        flat = tensor.detach().reshape(-1)
        stride = deterministic_stride(flat.numel(), SAMPLE_COUNT)
        sample = flat[::stride][:SAMPLE_COUNT].to(torch.float32).cpu().contiguous()
        filename = f"step00_block{block:02d}_{branch}.f32"
        path = self.checkpoint_dir / filename
        np.asarray(sample.numpy(), dtype="<f4").tofile(path)
        record = {
            "schema": 1,
            "step": 0,
            "traversal": 0,
            "block": block,
            "branch": branch,
            "shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype),
            "sample_file": f"checkpoints/{filename}",
            "sample_dtype": "float32-le",
            "sample_count": int(sample.numel()),
            "sample_stride": stride,
            "stats": _checkpoint_stats(tensor),
        }
        with _LOCK:
            self.records.append(record)
            self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "schema": 1,
            "oracle": "AnimaXS Oracle E",
            "mode": self.mode,
            "pack_sha256": self.pack_sha256,
            "pack_bytes": PACK_PATH.stat().st_size if PACK_PATH.exists() else None,
            "sample_count_target": SAMPLE_COUNT,
            "sampling": "flatten C-order; stride=max(1,numel//sample_count); first sample_count values",
            "expected_step0_checkpoints": 84,
            "completed_step0_checkpoints": len(self.records),
            "complete": len(self.records) == 84,
            "records": self.records,
        }
        tmp = RUN_DIR / "oracle_e_checkpoints.json.tmp"
        final = RUN_DIR / "oracle_e_checkpoints.json"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        tmp.replace(final)


def _install_noise_control() -> None:
    if NOISE_MODE == "none":
        return
    import comfy.sample as sample_mod
    if getattr(sample_mod.prepare_noise, "_anima_oracle_e", False):
        return
    original = sample_mod.prepare_noise

    def oracle_prepare_noise(latent_image, seed, noise_inds=None):
        if NOISE_MODE == "load":
            if not NOISE_FILE.exists():
                raise RuntimeError(f"Oracle E noise file missing: {NOISE_FILE}")
            data = load_file(str(NOISE_FILE), device="cpu")
            noise = data["noise"]
            if tuple(noise.shape) != tuple(latent_image.shape):
                raise RuntimeError(
                    f"Oracle E noise shape {tuple(noise.shape)} != latent {tuple(latent_image.shape)}")
            return noise.to(dtype=latent_image.dtype)
        noise = original(latent_image, seed, noise_inds)
        if NOISE_MODE == "save":
            NOISE_FILE.parent.mkdir(parents=True, exist_ok=True)
            save_file({"noise": noise.detach().cpu().contiguous()}, str(NOISE_FILE))
        return noise

    oracle_prepare_noise._anima_oracle_e = True  # type: ignore[attr-defined]
    sample_mod.prepare_noise = oracle_prepare_noise


_install_noise_control()


class AnimaOracleEArm:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "mode": (list(MODES),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "arm"
    CATEGORY = "diagnostics/oracle"

    def arm(self, model, mode):
        if mode not in MODES:
            raise RuntimeError(f"invalid Oracle E mode: {mode}")
        if not PACK_PATH.is_file():
            raise RuntimeError(
                "ANIMA_ORACLE_E_PACK must point to the exact pinned hybrid ANIMAPK")
        if PACK_PATH.stat().st_size != PINNED_PACK_BYTES:
            raise RuntimeError(
                f"Oracle E pack size mismatch: {PACK_PATH.stat().st_size} != {PINNED_PACK_BYTES}")
        digest = _sha256(PACK_PATH)
        if digest.lower() != EXPECTED_PACK_SHA256:
            raise RuntimeError(
                f"Oracle E pack SHA256 mismatch: {digest} != {EXPECTED_PACK_SHA256}")

        # The CUDA reference must not silently switch to TF32 tensor-core math.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                torch.set_float32_matmul_precision("highest")
            except Exception:
                pass

        m = model.clone()
        dm = _locate_diffusion_model(m)
        if getattr(dm, "_anima_oracle_e_armed", False):
            raise RuntimeError("Oracle E model was already armed")
        blocks = list(getattr(dm, "blocks", []))
        if len(blocks) != 28:
            raise RuntimeError(f"Oracle E expected 28 DiT blocks, found {len(blocks)}")

        pack = AnimapkReference(PACK_PATH)
        native_records = [
            r for r in pack.by_name.values() if r.quantization_format == ANE_TENSOR_FORMAT
        ]
        if len(native_records) != 280:
            pack.close()
            raise RuntimeError(
                f"Oracle E expected 280 ANE-native tensors, found {len(native_records)}")

        state = _CaptureState(mode, digest)
        handles = []
        originals = []

        for block_index, block in enumerate(blocks):
            # Patch exactly the ten projection modules selected by pack_anima.py.
            for suffix, module in _native_modules(block):
                name = _native_name(block_index, suffix)
                rec = pack.record(name)
                if rec.quantization_format != ANE_TENSOR_FORMAT:
                    pack.close()
                    raise RuntimeError(f"Oracle E projection is not native in pack: {name}")
                if getattr(module, "bias", None) is not None:
                    pack.close()
                    raise RuntimeError(f"Oracle E expected bias-free native projection: {name}")
                original_forward = module.forward
                originals.append((module, original_forward))

                def native_forward(x, _name=name, _pack=pack):
                    return _ane_linear_torch(_pack, _name, x)

                module.forward = native_forward

            # Q/K/V leaving normalization/RoPE live in FP16 Metal/ANE handoff
            # buffers on device.  Wrapping compute_qkv catches self (norm+RoPE)
            # and cross (norm) at the source model's exact seam.
            for attention in (block.self_attn, block.cross_attn):
                original_compute_qkv = attention.compute_qkv
                originals.append((attention, original_compute_qkv))

                def compute_qkv(*args, _orig=original_compute_qkv, **kwargs):
                    q, k, v = _orig(*args, **kwargs)
                    return _fp16_rt(q), _fp16_rt(k), _fp16_rt(v)

                attention.compute_qkv = compute_qkv

            # Current ANE baseline uses half -> float GELU -> half with fused
            # activation disabled.  Patch only this activation seam.
            activation = block.mlp.activation
            original_activation = activation.forward
            originals.append((activation, original_activation))
            activation.forward = lambda x: _device_gelu(x)

            # These are exact post-branch seams in pinned MiniTrainDIT:
            # cross-LN input == post-self residual; MLP-LN input == post-cross;
            # block output == post-MLP residual.
            def make_pre_capture(b: int, branch: str):
                def hook(_module, args):
                    state.capture(b, branch, args[0])
                return hook

            def make_post_capture(b: int):
                def hook(_module, _args, output):
                    state.capture(b, "mlp", output)
                return hook

            handles.append(block.layer_norm_cross_attn.register_forward_pre_hook(
                make_pre_capture(block_index, "self")))
            handles.append(block.layer_norm_mlp.register_forward_pre_hook(
                make_pre_capture(block_index, "cross")))
            handles.append(block.register_forward_hook(make_post_capture(block_index)))

        # Block 0 is the stable traversal seam used by Oracle V2.  E2 converts
        # only the incoming residual to FP32; all later blocks naturally remain
        # FP32 because addcmul uses residual_dtype from x.
        def traversal_hook(_module, args):
            state.begin_traversal()
            if mode == "E2_device_residual":
                mutable = list(args)
                mutable[0] = mutable[0].to(torch.float32)
                return tuple(mutable)
            return None

        handles.append(blocks[0].register_forward_pre_hook(traversal_hook, prepend=True))

        dm._anima_oracle_e_armed = True
        dm._anima_oracle_e_pack = pack
        dm._anima_oracle_e_handles = handles
        dm._anima_oracle_e_originals = originals
        dm._anima_oracle_e_state = state

        meta = {
            "schema": 1,
            "mode": mode,
            "diffusion_class": f"{dm.__class__.__module__}.{dm.__class__.__name__}",
            "block_count": len(blocks),
            "native_projection_count": len(native_records),
            "pack_path": str(PACK_PATH),
            "pack_bytes": PACK_PATH.stat().st_size,
            "pack_sha256": digest,
            "native_format": ANE_TENSOR_FORMAT,
            "sample_count": SAMPLE_COUNT,
            "output_row_chunk": OUTPUT_ROW_CHUNK,
            "tf32_disabled": bool(torch.cuda.is_available()),
            "noise_mode": NOISE_MODE,
            "noise_file": str(NOISE_FILE),
            "reference_accumulation": "FP32 torch.mm; not claimed instruction-identical to private ANE",
            "gelu_reference": "FP16 input -> CUDA exact-erf FP32 GELU -> FP16 output",
        }
        (RUN_DIR / "oracle_e_arm.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n")
        # Emit an initial manifest even if the model fails before block 0.
        state._write_manifest()
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaOracleEArm": AnimaOracleEArm}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaOracleEArm": "Anima Oracle E Arm"}
