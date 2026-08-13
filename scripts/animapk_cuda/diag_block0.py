#!/usr/bin/env python3
"""Block-0 surgical diagnostic: real predict2 Block vs oracle DitBlock on the
IDENTICAL inputs, comparing every intermediate (modulation, norms, attention,
mlp) to find the 12x divergence."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from animapk_cuda import upstream as up
import dit_source_oracle as dso

FIX = "/workspace/out/fixture"


def cmp(a, b, label):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    rel = float((a - b).norm() / (b.norm() + 1e-30))
    print(f"{label:>28} cos={cos:10.6f} relL2={rel:10.4f} a_norm={a.norm():12.1f} b_norm={b.norm():12.1f}")
    return cos


def main():
    device = "cuda"
    model, _ = up.load_upstream("animapk_cuda/comfy_stub",
                                "/workspace/source/anima-turbo-v1.0.safetensors",
                                dtype=torch.bfloat16, device=device)
    x_in = torch.from_numpy(np.fromfile(os.path.join(FIX, "x_in.f32"), dtype=np.float32)).view(1, 16, 1, 64, 64).to(device, torch.bfloat16)
    context = torch.from_numpy(np.fromfile(os.path.join(FIX, "context512.f32"), dtype=np.float32)).view(1, 512, 1024).to(device, torch.bfloat16)

    # --- real graph: capture block-0 inputs by running the model ---
    caps = {}
    h = model.blocks[0].register_forward_pre_hook(lambda m, i: caps.__setitem__("x", i[0].detach().clone()))
    with torch.no_grad():
        model.forward(x_in, torch.tensor([1.0], dtype=torch.bfloat16, device=device), context)
    h.remove()
    x = caps["x"]  # [1,1,32,32,2048] bf16

    # recompute emb/adaln the same way the model does
    timesteps = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
    t_emb_raw = model.t_embedder[0](timesteps.unsqueeze(1)).to(x.dtype)
    t_emb, adaln = model.t_embedder[1](t_emb_raw)
    emb = model.t_embedding_norm(t_emb).unsqueeze(1)  # [1,1,2048]
    adaln = adaln.unsqueeze(1)                        # [1,1,6144]
    # rope is produced per-forward by the pos_embedder
    rope = model.pos_embedder(x, fps=None, device=device, dtype=x.dtype)  # [L,1,1,64,2,2]
    blk = model.blocks[0]
    compute_dtype = emb.dtype

    # ---------------- REAL block 0 ----------------
    r = {}
    with torch.no_grad():
        m_self = blk.adaln_modulation_self_attn(emb) + adaln
        r["shift_s"], r["scale_s"], r["gate_s"] = m_self.chunk(3, dim=-1)
        normed_s = blk.layer_norm_self_attn(x) * (1 + r["scale_s"].unsqueeze(2).unsqueeze(3)) + r["shift_s"].unsqueeze(2).unsqueeze(3)
        r["normed_s"] = normed_s
        attn_s = blk.self_attn(
            normed_s.to(compute_dtype).reshape(1, 1024, 2048), None,
            rope_emb=rope, transformer_options={})
        r["attn_s_out"] = attn_s.reshape(1, 1, 32, 32, 2048)
        r["x_after_self"] = torch.addcmul(x, r["gate_s"].unsqueeze(2).unsqueeze(3).to(x.dtype), r["attn_s_out"].to(x.dtype))
        m_c = blk.adaln_modulation_cross_attn(emb) + adaln
        r["shift_c"], r["scale_c"], r["gate_c"] = m_c.chunk(3, dim=-1)
        normed_c = blk.layer_norm_cross_attn(x) * (1 + r["scale_c"].unsqueeze(2).unsqueeze(3)) + r["shift_c"].unsqueeze(2).unsqueeze(3)
        r["normed_c"] = normed_c
        attn_c = blk.cross_attn(
            normed_c.to(compute_dtype).reshape(1, 1024, 2048), context,
            rope_emb=rope, transformer_options={})
        r["attn_c_out"] = attn_c.reshape(1, 1, 32, 32, 2048)
        r["x_after_cross"] = torch.addcmul(r["x_after_self"], r["gate_c"].unsqueeze(2).unsqueeze(3).to(x.dtype), r["attn_c_out"].to(x.dtype))
        m_m = blk.adaln_modulation_mlp(emb) + adaln
        r["shift_m"], r["scale_m"], r["gate_m"] = m_m.chunk(3, dim=-1)
        normed_m = blk.layer_norm_mlp(r["x_after_cross"]) * (1 + r["scale_m"].unsqueeze(2).unsqueeze(3)) + r["shift_m"].unsqueeze(2).unsqueeze(3)
        r["normed_m"] = normed_m
        mlp_out = blk.mlp(normed_m.to(compute_dtype).reshape(1, 1024, 2048))
        r["mlp_out"] = mlp_out.reshape(1, 1, 32, 32, 2048)
        r["block0_out_real"] = torch.addcmul(r["x_after_cross"], r["gate_m"].unsqueeze(2).unsqueeze(3).to(x.dtype), r["mlp_out"].to(x.dtype))

    # ---------------- ORACLE block 0 ----------------
    w = dso.load_weights("/workspace/source/anima-turbo-v1.0.safetensors")
    ob = dso.DitBlock(w, 0)
    from animapk_cuda.runtime import _to_device
    ob.self_attn = _to_device(ob.self_attn, device)
    ob.cross_attn = _to_device(ob.cross_attn, device)
    ob.mod_s = _to_device(ob.mod_s, device)
    ob.mod_c = _to_device(ob.mod_c, device)
    ob.mod_m = _to_device(ob.mod_m, device)
    rope_o = dso.dit_rope().to(compute_dtype).to(device)
    x5 = x  # [1,1,32,32,2048]

    def obranch(norm, attn, ctx, rope_use, mod, label):
        shift, scale, gate = ob.modulate(emb, adaln, mod)
        normed = norm(x5) * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)
        nt = normed.reshape(1, 1024, 2048)
        out = attn.forward(nt, ctx, rope_use)
        out = out.reshape(1, 1, 32, 32, 2048)
        cmp(normed, r[f"normed_{label}"], label + " normed")
        cmp(out, r[f"attn_{label}_out"], label + " attn out")
        cmp(gate, r[f"gate_{label}"], label + " gate")
        return out, gate

    print("== stage comparisons (real vs oracle) ==")
    cmp(x, x, "x (block00_in)")
    cmp(emb, emb, "emb")
    cmp(adaln, adaln, "adaln")
    # modulation pre-chunk
    m_real = blk.adaln_modulation_self_attn(emb) + adaln
    m_or = ob.modulate(emb, adaln, ob.mod_s)  # returns (shift,scale,gate) chunks
    m_or_full = torch.cat([t.reshape(-1, 2048) for t in m_or], dim=-1)  # [1,1,6144]
    cmp(m_real, m_or_full, "modulation m (self)")
    out_s, gate_s = obranch(ob.norm_s, ob.self_attn, None, rope_o, ob.mod_s, "s")
    x5 = x5 + gate_s.unsqueeze(2).unsqueeze(3) * out_s
    cmp(x5, r["x_after_self"], "x after self")
    out_c, gate_c = obranch(ob.norm_c, ob.cross_attn, context, None, ob.mod_c, "c")
    x5 = x5 + gate_c.unsqueeze(2).unsqueeze(3) * out_c
    cmp(x5, r["x_after_cross"], "x after cross")
    shift_m, scale_m, gate_m = ob.modulate(emb, adaln, ob.mod_m)
    normed_m = ob.norm_m(x5) * (1 + scale_m.unsqueeze(2).unsqueeze(3)) + shift_m.unsqueeze(2).unsqueeze(3)
    hidden = dso.linear(normed_m.reshape(1, 1024, 2048), ob.mlp1)
    hidden = dso.gelu(hidden)
    out_m = dso.linear(hidden, ob.mlp2).reshape(1, 1, 32, 32, 2048)
    x5 = x5 + gate_m.unsqueeze(2).unsqueeze(3) * out_m
    cmp(x5, r["block0_out_real"], "block0_out FINAL")


if __name__ == "__main__":
    main()
