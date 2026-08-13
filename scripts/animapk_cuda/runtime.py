"""Ladder runtime for AnimaXS parity investigation.

Two explicit CUDA/Python modes over the same validated source-oracle graph
(scripts/dit_source_oracle.py):

  decoded_reference  — decode all pack tensors into a normal weights dict
                       (CPU fp32), run the full graph.  Byte-faithful
                       correctness oracle.
  streaming_animapk  — mmap the pack, copy the current execution range into a
                       reusable device byte ring, decode W4/W8/FP16 into fp32
                       scratch, run the stage, overwrite the ring for the next
                       range.  Mirrors the production AnimaXS weight lifecycle
                       (D047: serial one-slot loop — decode block N, run block
                       N, ring overwritten by block N+1).

Both modes run the IDENTICAL graph math and capture the same per-block and
final-path tensors in one forward.  Weights are held CPU-side and moved to the
device per block at forward time, so peak GPU memory is one block's weights +
activations (bounded), matching the production ring budget.
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dit_source_oracle as dso  # noqa: E402

from .reader import build_execution_manifest  # noqa: E402

GROUP = 64


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in list(vars(obj).items()):
            setattr(obj, k, _to_device(v, device))
        return obj
    return obj


def _weight_from_ring(ring, item, base_local):
    """Decode one tensor from a raw-byte ring (device uint8 tensor) at
    base_local + data_offset.  Returns a float32 tensor on the ring device."""
    storage = item["storage_dtype"]
    shape = [int(x) for x in item["shape"]]
    ds = base_local + int(item["data_offset"])
    de = ds + int(item["data_size"])
    if storage == "fp16":
        return ring[ds:de].view(torch.float16).float().reshape(shape)
    if storage == "fp32":
        return ring[ds:de].view(torch.float32).reshape(shape)
    if storage in ("w4", "w8"):
        ss = base_local + int(item["scale_offset"])
        se = ss + int(item["scale_size"])
        zs = base_local + int(item["zero_offset"])
        ze = zs + int(item["zero_size"])
        scale = ring[ss:se].view(torch.float16).float()
        zero = ring[zs:ze].view(torch.float16).float()
        rows, cols = shape
        if storage == "w4":
            bpr = (cols + 1) // 2
            b = ring[ds:de].reshape(rows, bpr)
            lo = (b & 0x0F).float()
            hi = (b >> 4).float()
            q = torch.stack([lo, hi], dim=-1).reshape(rows, 2 * bpr)[:, :cols]
        else:
            q = ring[ds:de].float().reshape(rows, cols)
        gpr = (cols + GROUP - 1) // GROUP
        col_groups = torch.arange(cols, dtype=torch.long, device=ring.device) // GROUP
        g = col_groups.unsqueeze(0).expand(rows, cols) + torch.arange(rows, dtype=torch.long, device=ring.device).unsqueeze(1) * gpr
        sg = scale.reshape(-1)[g.clamp(max=scale.numel() - 1)]
        zg = zero.reshape(-1)[g.clamp(max=zero.numel() - 1)]
        return q * sg + zg
    raise ValueError(f"unsupported storage {storage}")


class LadderDiT:
    """Source-oracle DiT graph with per-block capture; weights come from a
    stage provider so decoded_reference and streaming share one forward.

    Provider contract: callable(keys) -> {stripped_name: fp32 CPU tensor}.
    Blocks are built lazily per forward call (decode/run interleave).
    """

    def __init__(self, provider, dtype, device):
        self.dtype = dtype
        self.device = device
        self._provider = provider
        w = self._provider(["x_embedder", "t_embedder", "t_embedding_norm"])
        self.x_proj = w["x_embedder.proj.1.weight"]
        self.t_emb = dso.DitTimestepEmbedding(w)
        self.t_norm = w["t_embedding_norm.weight"]
        self.final_w = self._provider(["final_layer"])
        self.rope = dso.dit_rope().to(dtype)

    def _block(self, i):
        wb = self._provider([f"blocks.{i}"])
        blk = dso.DitBlock(wb, i)
        return _to_device(blk, self.device)

    def forward(self, x, sigma, context, capture=True):
        dt, dev = self.dtype, self.device
        caps = {}
        x = x.to(dev).to(dt)
        pm = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4], dtype=dt, device=dev)
        x17 = torch.cat([x, pm], dim=1)
        tokens = dso.DitModel.patchify(self, x17)
        caps["pre_x_embedder_tokens"] = tokens
        emb_tok = dso.linear(tokens.reshape(1, dso.TOKENS, 68), self.x_proj.to(dev))
        emb_tok = emb_tok.reshape(1, 1, 32, 32, dso.DIM)
        caps["x_embedder_out"] = emb_tok
        t_emb_raw = dso.DitTimesteps()(torch.tensor([sigma], device=dev), dt)
        self.t_emb.w1 = self.t_emb.w1.to(dev)
        self.t_emb.w2 = self.t_emb.w2.to(dev)
        emb, adaln = self.t_emb(t_emb_raw)
        emb = emb.unsqueeze(1)
        adaln = adaln.unsqueeze(1)
        emb = dso.rms_norm(emb, self.t_norm.to(dev))
        caps["timestep_emb"] = emb
        caps["adaln_lora"] = adaln
        context = context.to(dev).to(dt)
        caps["context"] = context
        caps["rope_sha256"] = hashlib.sha256(self.rope.cpu().float().numpy().tobytes()).hexdigest()
        for i in range(dso.NUM_BLOCKS):
            blk = self._block(i)
            if capture:
                caps[f"block{i:02d}_in"] = emb_tok.detach().float().cpu()
            emb_tok = blk.forward(emb_tok, emb, adaln, context, self.rope.to(dev))
            if capture:
                caps[f"block{i:02d}_out"] = emb_tok.detach().float().cpu()
            del blk
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        emb_tok = emb_tok.to(context.dtype)
        caps["pre_final_norm"] = emb_tok.detach().float().cpu()
        final = _to_device(dso.DitFinalLayer(self.final_w), dev)
        out = final.forward(emb_tok, emb, adaln)
        caps["post_final_projection_patched"] = out.detach().float().cpu()
        vel = self.model_unpatchify(out)
        caps["post_unpatchify_velocity"] = vel.detach().float().cpu()
        return vel.float(), caps

    def model_unpatchify(self, x):
        # dso.DitModel.unpatchify — [1,1,32,32,64] -> [1,16,1,64,64]
        b, t, h, w, m = x.shape
        x = x.reshape(b, t, h, w, 2, 2, 16)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(b, 16, t, h * 2, w * 2)
        return x


class DecodedReference(LadderDiT):
    """Mode 1: full decode into a plain weights dict (CPU fp32)."""

    def __init__(self, pack, dtype, device):
        self.pack = pack
        self.full = {}
        from .quant import decode_tensor_from_pack
        for item in pack.tensor_meta:
            name = str(item["name"])
            if name.startswith("model.diffusion_model."):
                name = name[len("model.diffusion_model."):]
            self.full[name] = decode_tensor_from_pack(pack, item, device="cpu")
        super().__init__(lambda keys: {k: self.full[k] for k in keys}, dtype, device)


class StreamingAnimapk(LadderDiT):
    """Mode 2: lifecycle-faithful ring runtime (serial one-slot loop, D047).

    One reusable device ring holds the raw bytes of the current execution
    range; each provider call overwrites the ring with the requested stage and
    decodes into fresh fp32 scratch.  Bounded memory: max(raw range) ring +
    per-stage scratch.  No stage results persist between calls.
    """

    def __init__(self, pack, dtype, device):
        self.pack = pack
        self.manifest = build_execution_manifest(pack, "block")
        self.ranges = {r["key"]: r for r in self.manifest["ranges"]}
        self.max_range = max(r["end"] - r["start"] for r in self.manifest["ranges"])
        self.ring = torch.zeros(self.max_range, dtype=torch.uint8, device=device)

        def provider(keys):
            out = {}
            for key in keys:
                r = self.ranges[key]
                raw = bytes(pack.blob[r["start"]:r["end"]])
                n = len(raw)
                src = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                self.ring[:n].copy_(src, non_blocking=True)
                ring_view = self.ring[:n]
                for t in r["tensors"]:
                    item = pack.meta_by_offset(t["global_offset"])
                    local = t["global_offset"] - r["start"]
                    out[t["name"]] = _weight_from_ring(ring_view, item, local).cpu()
            return out

        super().__init__(provider, dtype, device)

    def ring_stats(self) -> dict:
        return {
            "ring_bytes": self.max_range,
            "stage_count": len(self.ranges),
            "stages": list(self.ranges.keys()),
        }
