# AnimaXS Oracle E ComfyUI custom node.
#
# E1/E2 keep the pinned ComfyUI Anima forward in control of block structure,
# attention, LayerNorm/AdaLN, gating, and residual updates. Only the seams that
# differ in the physical hybrid pack/runtime are replaced:
#
# * 10 ANE projections/block: exact U8 + FP32 per-row scale/bias from ANIMAPK,
#   with FP16 activation surface input/output;
# * 6 Metal modulation matvecs/block: exact group64 W8 + FP16 scale/zero;
# * 4 learned attention RMSNorm vectors/block: exact FP16 pack payload;
# * MLP activation: FP16 -> FP32 exact-erf GELU -> FP16;
# * Q/K/V post-transform storage: FP16;
# * E2 residual stream: FP32.
#
# This is a software reference for the *visible* hybrid contract. FP32 torch.mm
# is deliberately used as stable reference accumulation; it is NOT a claim
# about undocumented private-ANE accumulator instructions/reduction order.
from __future__ import annotations

from pathlib import Path
import hashlib
import json
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
        GROUP64_TENSOR_FORMAT,
        DEFAULT_SAMPLE_COUNT,
        AnimapkReference,
        deterministic_stride,
    )
except Exception:
    from scripts.ane_oracle_e_pack_reference import (
        ANE_TENSOR_FORMAT,
        GROUP64_TENSOR_FORMAT,
        DEFAULT_SAMPLE_COUNT,
        AnimapkReference,
        deterministic_stride,
    )

PINNED_PACK_SHA256 = "f5c80a25114b62a6807996180d439c5d12828d7392c604e1eee15acb28977dc4"
PINNED_PACK_BYTES = 2_128_838_656
MODES = ("E1_native_ane", "E2_device_residual")
STEP0_SENTINEL = "__ANIMA_ORACLE_E_STEP0_COMPLETE__"
RUN_DIR = Path(os.environ.get("ANIMA_ORACLE_E_RUN_DIR", "/tmp/anima_oracle_e"))
RUN_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_COUNT = int(os.environ.get("ANIMA_ORACLE_E_SAMPLE_COUNT", str(DEFAULT_SAMPLE_COUNT)))
OUTPUT_ROW_CHUNK = int(os.environ.get("ANIMA_ORACLE_E_OUTPUT_ROW_CHUNK", "256"))
PACK_PATH = Path(os.environ.get("ANIMA_ORACLE_E_PACK", ""))
EXPECTED_PACK_SHA256 = os.environ.get("ANIMA_ORACLE_E_PACK_SHA256", PINNED_PACK_SHA256).lower()
STOP_AFTER_STEP0 = os.environ.get("ANIMA_ORACLE_E_STOP_AFTER_STEP0", "0") == "1"
CAPTURE_BLOCK0_STAGES = os.environ.get("ANIMA_ORACLE_E_CAPTURE_BLOCK0_STAGES", "0") == "1"
DEVICE_LEGACY_ATTENTION = os.environ.get("ANIMA_ORACLE_E_DEVICE_LEGACY_ATTENTION", "0") == "1"
NOISE_MODE = os.environ.get(
    "ANIMA_ORACLE_E_NOISE_MODE", os.environ.get("ANIMA_ORACLE_NOISE_MODE", "none"))
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


def _tensor_name(block: int, suffix: str) -> str:
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


def _modulation_modules(block: Any) -> list[tuple[str, Any]]:
    groups = [
        ("adaln_modulation_self_attn", block.adaln_modulation_self_attn),
        ("adaln_modulation_cross_attn", block.adaln_modulation_cross_attn),
        ("adaln_modulation_mlp", block.adaln_modulation_mlp),
    ]
    result: list[tuple[str, Any]] = []
    for prefix, sequence in groups:
        if len(sequence) != 3:
            raise RuntimeError(
                f"Oracle E expected SiLU+Linear+Linear for {prefix}, found length {len(sequence)}")
        result.append((f"{prefix}.1", sequence[1]))
        result.append((f"{prefix}.2", sequence[2]))
    return result


def _norm_modules(block: Any) -> list[tuple[str, Any]]:
    return [
        ("self_attn.q_norm", block.self_attn.q_norm),
        ("self_attn.k_norm", block.self_attn.k_norm),
        ("cross_attn.q_norm", block.cross_attn.q_norm),
        ("cross_attn.k_norm", block.cross_attn.k_norm),
    ]


def _ane_linear_torch(pack: AnimapkReference, name: str, x: torch.Tensor) -> torch.Tensor:
    """Exact packed native weights + FP16 surface I/O + FP32 reference dot."""
    rec = pack.record(name)
    if rec.quantization_format != ANE_TENSOR_FORMAT or len(rec.shape) != 2:
        raise RuntimeError(f"Oracle E native projection contract mismatch: {name}")
    out_features, in_features = rec.shape
    if x.shape[-1] != in_features:
        raise RuntimeError(
            f"Oracle E input K mismatch for {name}: {x.shape[-1]} != {in_features}")
    x_surface = x.to(torch.float16).reshape(-1, in_features).to(torch.float32)
    out = torch.empty(
        (x_surface.shape[0], out_features), device=x.device, dtype=torch.float32)
    for row0 in range(0, out_features, OUTPUT_ROW_CHUNK):
        row1 = min(out_features, row0 + OUTPUT_ROW_CHUNK)
        q_np, scale_np, bias_np = pack.native_rows(name, row0, row1)
        q = torch.from_numpy(q_np).to(device=x.device, dtype=torch.float32)
        scale = torch.from_numpy(scale_np).to(device=x.device, dtype=torch.float32)
        bias = torch.from_numpy(bias_np).to(device=x.device, dtype=torch.float32)
        weight = q.mul_(scale[:, None]).add_(bias[:, None])
        out[:, row0:row1] = torch.mm(x_surface, weight.t())
    # Do NOT cast back to BF16. The actual private-runtime output surface is
    # FP16; downstream Metal consumes those FP16 values.
    return out.to(torch.float16).reshape(*x.shape[:-1], out_features)


def _group64_matvec_torch(pack: AnimapkReference, name: str, x: torch.Tensor) -> torch.Tensor:
    """Exact group64 W8/FP16-metadata modulation matvec, FP32 input/output."""
    rec = pack.record(name)
    if rec.quantization_format != GROUP64_TENSOR_FORMAT or rec.storage_dtype != "w8":
        raise RuntimeError(f"Oracle E Metal modulation contract mismatch: {name}")
    if len(rec.shape) != 2:
        raise RuntimeError(f"Oracle E modulation tensor is not rank-2: {name}")
    out_features, in_features = rec.shape
    if x.shape[-1] != in_features:
        raise RuntimeError(
            f"Oracle E modulation K mismatch for {name}: {x.shape[-1]} != {in_features}")
    x_f32 = x.to(torch.float32).reshape(-1, in_features)
    out = torch.empty(
        (x_f32.shape[0], out_features), device=x.device, dtype=torch.float32)
    for row0 in range(0, out_features, OUTPUT_ROW_CHUNK):
        row1 = min(out_features, row0 + OUTPUT_ROW_CHUNK)
        weight_np = pack.group64_reconstructed_rows(name, row0, row1)
        weight = torch.from_numpy(weight_np).to(device=x.device, dtype=torch.float32)
        out[:, row0:row1] = torch.mm(x_f32, weight.t())
    return out.reshape(*x.shape[:-1], out_features)


def _install_exact_fp16_norm(pack: AnimapkReference, name: str, module: Any) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"Oracle E norm has no weight parameter: {name}")
    exact_np = pack.fp16_tensor(name)
    if tuple(exact_np.shape) != tuple(weight.shape):
        raise RuntimeError(
            f"Oracle E norm shape mismatch for {name}: pack={exact_np.shape} module={tuple(weight.shape)}")
    original = weight.detach().clone()
    exact = torch.from_numpy(exact_np.copy()).to(device=weight.device, dtype=torch.float16)
    # Keep the original Parameter object so Comfy's parameter/offload tracking
    # remains intact; replace only its data storage/dtype on the cloned model.
    weight.data = exact
    return original


def _fp16_rt(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16)


def _device_gelu(x: torch.Tensor) -> torch.Tensor:
    # Current baseline ANE path: FP16 surface -> FP32 exact-erf GELU -> FP16.
    y = F.gelu(x.to(torch.float16).to(torch.float32), approximate="none")
    return y.to(torch.float16)


def _device_legacy_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    transformer_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Emulate the visible FP16 storage boundaries of A12 legacy MPS attention.

    The physical path materializes QK scores as FP16, evaluates softmax from
    those scores in FP32 but stores probabilities back to FP16, then runs PV
    from FP16 probability/value inputs and materializes an FP16 attended result.
    FP32 matmul accumulation here is a stable software reference; MPS reduction
    order/private accumulator details are deliberately not claimed identical.
    """
    del transformer_options
    q = q_B_S_H_D.to(torch.float16).permute(0, 2, 1, 3)
    k = k_B_S_H_D.to(torch.float16).permute(0, 2, 1, 3)
    v = v_B_S_H_D.to(torch.float16).permute(0, 2, 1, 3)
    scale = q.shape[-1] ** -0.5
    scores = (torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * scale).to(torch.float16)
    probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(torch.float16)
    attended = torch.matmul(probabilities.to(torch.float32), v.to(torch.float32)).to(torch.float16)
    return attended.permute(0, 2, 1, 3).reshape(
        attended.shape[0], attended.shape[2], attended.shape[1] * attended.shape[3]
    )


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
        self.stage_records: list[dict[str, Any]] = []
        self.checkpoint_dir = RUN_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = RUN_DIR / "stages"
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def begin_traversal(self) -> None:
        self.traversal += 1

    def capture(self, block: int, branch: str, tensor: torch.Tensor) -> None:
        # CFG is pinned to 1.0, so traversal 0 is diffusion step 0.
        if self.traversal != 0:
            return
        key = (block, branch)
        if any((r["block"], r["branch"]) == key for r in self.records):
            raise RuntimeError(f"Oracle E duplicate checkpoint {key}")
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

    def capture_stage(self, stage: str, tensor: torch.Tensor, dtype: str) -> None:
        if not CAPTURE_BLOCK0_STAGES or self.traversal != 0:
            return
        expected = {
            "self.modulation": (6144, "float32-le", "f32"),
            "self.projection_input": (1024 * 2048, "float16-le", "f16"),
            "self.q_raw": (1024 * 2048, "float16-le", "f16"),
            "self.k_raw": (1024 * 2048, "float16-le", "f16"),
            "self.v_raw": (1024 * 2048, "float16-le", "f16"),
            "self.q_post_rope": (1024 * 2048, "float16-le", "f16"),
            "self.k_post_rope": (1024 * 2048, "float16-le", "f16"),
            "self.attended": (1024 * 2048, "float16-le", "f16"),
            "self.branch": (1024 * 2048, "float16-le", "f16"),
        }
        if stage not in expected:
            raise RuntimeError(f"Oracle E unknown CUDA block0 stage: {stage}")
        expected_numel, expected_dtype, extension = expected[stage]
        flat = tensor.detach().reshape(-1)
        if flat.numel() != expected_numel:
            raise RuntimeError(
                f"Oracle E CUDA stage {stage} has {flat.numel()} elements; expected {expected_numel}")
        if any(r["stage"] == stage for r in self.stage_records):
            raise RuntimeError(f"Oracle E duplicate CUDA block0 stage: {stage}")
        canonical = "stage_b00_" + stage.replace(".", "_")
        filename = f"{canonical}.{extension}"
        if dtype != expected_dtype:
            raise RuntimeError(f"Oracle E CUDA stage dtype contract mismatch for {stage}: {dtype}")
        if expected_dtype == "float16-le":
            work = flat.to(torch.float16).cpu().contiguous()
            np.asarray(work.numpy(), dtype="<f2").tofile(self.stage_dir / filename)
        else:
            work = flat.to(torch.float32).cpu().contiguous()
            np.asarray(work.numpy(), dtype="<f4").tofile(self.stage_dir / filename)
        record = {
            "schema": 1,
            "step": 0,
            "block": 0,
            "stage": stage,
            "name": canonical,
            "shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype),
            "dtype": expected_dtype,
            "element_count": int(flat.numel()),
            "file": f"stages/{filename}",
            "stats": _checkpoint_stats(tensor),
        }
        with _LOCK:
            self.stage_records.append(record)
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
            "stop_after_step0": STOP_AFTER_STEP0,
            "expected_block0_stage_payloads": 9 if CAPTURE_BLOCK0_STAGES else 0,
            "completed_block0_stage_payloads": len(self.stage_records),
            "block0_stage_records": self.stage_records,
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
            # Physical A12 capture stores [B,C,H,W]; Comfy Anima uses [B,C,1,H,W].
            # This is reshape-only and preserves every captured value exactly.
            if noise.ndim == 4 and latent_image.ndim == 5 and latent_image.shape[2] == 1:
                expected4 = (latent_image.shape[0], latent_image.shape[1], latent_image.shape[3], latent_image.shape[4])
                if tuple(noise.shape) == expected4:
                    noise = noise.unsqueeze(2)
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
        block0_stage_context: dict[str, torch.Tensor | None] = {"adaln": None}
        handles = []
        originals = []
        group64_count = 0
        fp16_norm_count = 0

        for block_index, block in enumerate(blocks):
            # Ten large projections use exact ANE-native row-U8 weights.
            for suffix, module in _native_modules(block):
                name = _tensor_name(block_index, suffix)
                rec = pack.record(name)
                if rec.quantization_format != ANE_TENSOR_FORMAT:
                    pack.close()
                    raise RuntimeError(f"Oracle E projection is not native in pack: {name}")
                if getattr(module, "bias", None) is not None:
                    pack.close()
                    raise RuntimeError(f"Oracle E expected bias-free native projection: {name}")
                original_forward = module.forward
                originals.append((module, "forward", original_forward))

                def native_forward(x, _name=name, _pack=pack):
                    return _ane_linear_torch(_pack, _name, x)

                module.forward = native_forward

            # Six compact Metal-only modulation matrices stay group64 W8 in the
            # hybrid pack. V2's reconstructed checkpoint rounded them to BF16,
            # so replace those forwards with exact pack arithmetic here.
            for suffix, module in _modulation_modules(block):
                name = _tensor_name(block_index, suffix)
                rec = pack.record(name)
                if rec.quantization_format != GROUP64_TENSOR_FORMAT or rec.storage_dtype != "w8":
                    pack.close()
                    raise RuntimeError(f"Oracle E modulation is not group64 W8: {name}")
                if getattr(module, "bias", None) is not None:
                    pack.close()
                    raise RuntimeError(f"Oracle E expected bias-free modulation linear: {name}")
                original_forward = module.forward
                originals.append((module, "forward", original_forward))

                def modulation_forward(x, _name=name, _pack=pack, _orig=original_forward,
                                       _block=block_index, _suffix=suffix):
                    packed_out = _group64_matvec_torch(_pack, _name, x)
                    if _block in (0, 1) and _suffix.startswith("adaln_modulation_mlp"):
                        with torch.no_grad():
                            original_out = _orig(x)
                            po = packed_out.detach().to(torch.float32).reshape(-1)
                            oo = original_out.detach().to(torch.float32).reshape(-1)
                            delta = po - oo
                            denom = torch.linalg.vector_norm(po) * torch.linalg.vector_norm(oo)
                            cosine = float(torch.dot(po, oo) / denom) if float(denom) > 0 else 1.0
                            print(
                                f"ORACLE_E_MOD_COMPARE b{_block:02d} {_suffix} "
                                f"packed_std={float(po.std(unbiased=False)):.9g} packed_maxAbs={float(po.abs().max()):.9g} "
                                f"orig_std={float(oo.std(unbiased=False)):.9g} orig_maxAbs={float(oo.abs().max()):.9g} "
                                f"delta_rms={float(torch.sqrt(torch.mean(delta*delta))):.9g} "
                                f"delta_max={float(delta.abs().max()):.9g} cosine={cosine:.12g}",
                                flush=True)
                    return packed_out

                module.forward = modulation_forward
                group64_count += 1

            # Four learned RMSNorm vectors are raw FP16 in the pack. Preserve
            # the Parameter objects but restore those exact FP16 payload values
            # so self fused RMS+RoPE and cross RMSNorm do not inherit V2's BF16
            # reconstruction loss.
            for suffix, module in _norm_modules(block):
                name = _tensor_name(block_index, suffix)
                original = _install_exact_fp16_norm(pack, name, module)
                originals.append((module, "weight_data", original))
                fp16_norm_count += 1

            # Q/K/V after the source norm/RoPE seam live in FP16 device buffers.
            for attention_label, attention in (("self", block.self_attn), ("cross", block.cross_attn)):
                original_compute_qkv = attention.compute_qkv
                originals.append((attention, "compute_qkv", original_compute_qkv))

                def compute_qkv(*args, _orig=original_compute_qkv, _b=block_index,
                                _label=attention_label, **kwargs):
                    q, k, v = _orig(*args, **kwargs)
                    q, k, v = _fp16_rt(q), _fp16_rt(k), _fp16_rt(v)
                    if CAPTURE_BLOCK0_STAGES and _b == 0 and _label == "self":
                        state.capture_stage("self.q_post_rope", q, "float16-le")
                        state.capture_stage("self.k_post_rope", k, "float16-le")
                    return q, k, v

                attention.compute_qkv = compute_qkv
                if DEVICE_LEGACY_ATTENTION or (CAPTURE_BLOCK0_STAGES and block_index == 0 and attention_label == "self"):
                    original_attn_op = attention.attn_op
                    originals.append((attention, "attn_op", original_attn_op))
                    selected_attn_op = _device_legacy_attention_op if DEVICE_LEGACY_ATTENTION else original_attn_op

                    def attention_op(q, k, v, transformer_options=None, _impl=selected_attn_op,
                                     _b=block_index, _label=attention_label):
                        result = _impl(q, k, v, transformer_options=transformer_options)
                        if CAPTURE_BLOCK0_STAGES and _b == 0 and _label == "self":
                            state.capture_stage("self.attended", result.to(torch.float16), "float16-le")
                        return result

                    attention.attn_op = attention_op

            # Current ANE baseline has fused MLP activation OFF:
            # half -> float GELU -> half.
            activation = block.mlp.activation
            original_activation = activation.forward
            originals.append((activation, "forward", original_activation))
            activation.forward = lambda x: _device_gelu(x)

            if CAPTURE_BLOCK0_STAGES and block_index == 0:
                def _block0_stage_pre(_module, args, kwargs):
                    adaln = kwargs.get("adaln_lora_B_T_3D")
                    if adaln is None and len(args) > 4:
                        adaln = args[4]
                    block0_stage_context["adaln"] = adaln.detach() if adaln is not None else None

                def _self_mod_stage(_module, _args, output):
                    combined = output
                    adaln = block0_stage_context.get("adaln")
                    if adaln is not None:
                        combined = combined + adaln
                    state.capture_stage("self.modulation", combined, "float32-le")

                def _self_input_stage(_module, args):
                    state.capture_stage("self.projection_input", args[0].to(torch.float16), "float16-le")

                def _projection_stage(stage):
                    def hook(_module, _args, output):
                        state.capture_stage(stage, output, "float16-le")
                    return hook

                handles.append(block.register_forward_pre_hook(_block0_stage_pre, with_kwargs=True))
                handles.append(block.adaln_modulation_self_attn.register_forward_hook(_self_mod_stage))
                handles.append(block.self_attn.register_forward_pre_hook(_self_input_stage))
                handles.append(block.self_attn.q_proj.register_forward_hook(_projection_stage("self.q_raw")))
                handles.append(block.self_attn.k_proj.register_forward_hook(_projection_stage("self.k_raw")))
                handles.append(block.self_attn.v_proj.register_forward_hook(_projection_stage("self.v_raw")))
                handles.append(block.self_attn.output_proj.register_forward_hook(_projection_stage("self.branch")))

            # Exact post-branch seams in pinned MiniTrainDIT:
            # cross-LN input == post-self; MLP-LN input == post-cross; block
            # output == post-MLP.
            def make_pre_capture(b: int, branch: str):
                def hook(_module, args):
                    state.capture(b, branch, args[0])
                return hook

            def make_post_capture(b: int):
                def hook(_module, _args, output):
                    state.capture(b, "mlp", output)
                    if (STOP_AFTER_STEP0 and state.traversal == 0 and b == 27
                            and len(state.records) == 84):
                        (RUN_DIR / "STEP0_COMPLETE").write_text(STEP0_SENTINEL + "\n")
                        raise RuntimeError(STEP0_SENTINEL)
                return hook

            handles.append(block.layer_norm_cross_attn.register_forward_pre_hook(
                make_pre_capture(block_index, "self")))
            handles.append(block.layer_norm_mlp.register_forward_pre_hook(
                make_pre_capture(block_index, "cross")))
            handles.append(block.register_forward_hook(make_post_capture(block_index)))

            # Temporary root-cause instrumentation for the reproduced block-1 MLP blow-up.
            # Keep it bounded to blocks 0/1 and print aggregate stats only.
            if block_index in (0, 1):
                debug_state = {"adaln": None}

                def _dbg_stats(label, value, _b=block_index):
                    t = value.detach().to(torch.float32)
                    finite = torch.isfinite(t)
                    if bool(finite.all()):
                        print(
                            f"ORACLE_E_MLP_DEBUG b{_b:02d} {label} "
                            f"shape={tuple(value.shape)} dtype={value.dtype} "
                            f"mean={float(t.mean()):.9g} std={float(t.std(unbiased=False)):.9g} "
                            f"min={float(t.min()):.9g} max={float(t.max()):.9g} "
                            f"maxAbs={float(t.abs().max()):.9g} l2={float(torch.linalg.vector_norm(t)):.9g}",
                            flush=True)
                    else:
                        print(f"ORACLE_E_MLP_DEBUG b{_b:02d} {label} NONFINITE", flush=True)

                def _block_dbg_pre(_module, args, kwargs, _st=debug_state, _b=block_index):
                    adaln = kwargs.get("adaln_lora_B_T_3D")
                    if adaln is None and len(args) > 4:
                        adaln = args[4]
                    _st["adaln"] = adaln.detach() if adaln is not None else None
                    if adaln is not None:
                        _dbg_stats("adaln_lora", adaln, _b)
                    _dbg_stats("block_input", args[0], _b)

                def _mod_dbg(_module, _args, output, _st=debug_state, _b=block_index):
                    _dbg_stats("mlp_mod_raw", output, _b)
                    combined = output
                    if _st.get("adaln") is not None:
                        combined = combined + _st["adaln"]
                    shift, scale, gate = combined.chunk(3, dim=-1)
                    _dbg_stats("mlp_shift", shift, _b)
                    _dbg_stats("mlp_scale", scale, _b)
                    _dbg_stats("mlp_gate", gate, _b)

                def _pre_dbg(label, _b=block_index):
                    def hook(_module, args):
                        _dbg_stats(label, args[0], _b)
                    return hook

                def _post_dbg(label, _b=block_index):
                    def hook(_module, _args, output):
                        _dbg_stats(label, output, _b)
                    return hook

                handles.append(block.register_forward_pre_hook(_block_dbg_pre, with_kwargs=True))
                handles.append(block.adaln_modulation_mlp.register_forward_hook(_mod_dbg))
                handles.append(block.mlp.register_forward_pre_hook(_pre_dbg("mlp_normalized_input")))
                handles.append(block.mlp.layer1.register_forward_hook(_post_dbg("mlp_layer1_out")))
                handles.append(block.mlp.activation.register_forward_hook(_post_dbg("mlp_gelu_out")))
                handles.append(block.mlp.layer2.register_forward_hook(_post_dbg("mlp_layer2_out")))

        if group64_count != 28 * 6:
            pack.close()
            raise RuntimeError(f"Oracle E expected 168 group64 modulation matrices, got {group64_count}")
        if fp16_norm_count != 28 * 4:
            pack.close()
            raise RuntimeError(f"Oracle E expected 112 FP16 learned norm vectors, got {fp16_norm_count}")

        # Block 0 is the stable traversal seam used by V2. E2 keeps the
        # residual stream FP32 like AnimaXS; E1 preserves V2's residual dtype so
        # D -> E1 isolates hybrid packed projection/aux-weight semantics.
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
            "metal_group64_modulation_count": group64_count,
            "metal_fp16_norm_count": fp16_norm_count,
            "pack_path": str(PACK_PATH),
            "pack_bytes": PACK_PATH.stat().st_size,
            "pack_sha256": digest,
            "native_format": ANE_TENSOR_FORMAT,
            "metal_group64_format": GROUP64_TENSOR_FORMAT,
            "sample_count": SAMPLE_COUNT,
            "output_row_chunk": OUTPUT_ROW_CHUNK,
            "tf32_disabled": bool(torch.cuda.is_available()),
            "noise_mode": NOISE_MODE,
            "noise_file": str(NOISE_FILE),
            "stop_after_step0": STOP_AFTER_STEP0,
            "capture_block0_stages": CAPTURE_BLOCK0_STAGES,
            "device_legacy_attention_emulation": DEVICE_LEGACY_ATTENTION,
            "native_reference_accumulation": "FP32 torch.mm; not claimed instruction-identical to private ANE",
            "modulation_reference_accumulation": "FP32 torch.mm over exact group64 W8/FP16 metadata; reduction order not claimed identical to Metal",
            "gelu_reference": "FP16 input -> CUDA exact-erf FP32 GELU -> FP16 output",
            "residual_policy": "FP32 in E2; inherited V2 dtype in E1",
        }
        (RUN_DIR / "oracle_e_arm.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n")
        state._write_manifest()
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaOracleEArm": AnimaOracleEArm}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaOracleEArm": "Anima Oracle E Arm"}
