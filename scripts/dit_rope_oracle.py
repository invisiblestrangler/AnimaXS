#!/usr/bin/env python3
"""
dit_rope_oracle.py — pinned-ComfyUI structural oracle for the DiT 3-D RoPE (H004).

Purpose (AnimaXS runbook §31, TODO H004):
  Compute the 3-D rope embedding tensor [T*H*W=1024, 64 freqs, 2, 2] used by the DiT
  self-attention, using the EXACT equations of the pinned ComfyUI
  VideoRopePosition3DEmb, and compare it against the Swift CPU output (dumped as .f32
  by the harness). RoPE has NO learnable weights — the harness just dumps its computed
  tensor; this oracle independently re-transcribes generate_embeddings in numpy.

Pinned source (commit cbbc9da), comfy/ldm/cosmos/position_embedding.py:
  VideoRopePosition3DEmb.__init__              :57-98   (dim split, ntk factors, ranges)
  VideoRopePosition3DEmb.generate_embeddings   :100-163 (thetas, freqs, half_emb, stack,
                                                          concat t,h,w, rearrange)

Config (from the W4 DiT pack / MODEL_ARCHITECTURE.json "rope", runbook §31):
  head_dim=128 -> dim_h=42, dim_w=42, dim_t=44
  h_extrapolation_ratio=w_extrapolation_ratio=4.0, t_extrapolation_ratio=1.0
  enable_fps_modulation=False (image case: uses seq directly, no /fps scaling)
  Sequence grid: T=1, H=W=32  (512x512 first working resolution, 2x2 patch)

IMPORTANT precision note (see THETA PRECISION in NEXT_TASK_HANDOFF):
  h_theta = 10000.0 * (4.0 ** (42.0/40.0)) = 42870.938501451725 — do NOT use the rounded
  prose values 42871.1/42871.4. Computed as Float32 to match the Swift/NN implementation.

Usage:
  python dit_rope_oracle.py [--out DIR] [--in DIR] [--dump] [--hand-values]
"""
import argparse, os, sys
import numpy as np

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"
OUT = "/root/AnimaXS/scripts/oracle_out"

HEAD_DIM = 128
DIM_H = HEAD_DIM // 6 * 2   # 42
DIM_W = DIM_H               # 42
DIM_T = HEAD_DIM - 2 * DIM_H  # 44
# Number of 2x2 rotation blocks per head-dim 128 = 64 freqs (each block covers 2 head-dims).
# timing: temporal has dim_t//2=22 freqs, height 21, width 21 -> 64 total.
NUM_FREQS = (DIM_T // 2) + (DIM_H // 2) + (DIM_W // 2)  # 22+21+21 = 64
H = 32
W = 32
T = 1


# ---------------------------------------------------------------------------
# Pinned ComfyUI equations — VERBATIM (position_embedding.py)
# ---------------------------------------------------------------------------
def vec2(data):
    """np.float32 vector (torch .float())."""
    return np.asarray(data, dtype=np.float32)


# variable holding the computed value of range so that each of h/w/t is derivable:
# dim_spatial_range  = arange(0, dim_h, 2)[: dim_h//2].float() / dim_h
# dim_temporal_range = arange(0, dim_t, 2)[: dim_t//2].float() / dim_t
def _range(dim):
    return vec2(np.arange(0, dim, 2)[: dim // 2]) / dim


def generate_embeddings(B_T_H_W_C, fps=None):
    """Transcribes VideoRopePosition3DEmb.generate_embeddings (position_embedding.py:100-163)
    for the given (B,T,H,W,C) grid. Returns [T*H*W, 64, 2, 2] float32."""
    # __init__ computed values (position_embedding.py:57-98)
    dim_h, dim_w, dim_t = DIM_H, DIM_W, DIM_T
    dim_spatial_range = _range(dim_h)          # [0,2,...,40]/42 (21)
    dim_temporal_range = _range(dim_t)         # [0,2,...,42]/44 (22)

    h_ntk_factor = 4.0 ** (dim_h / (dim_h - 2))        # 4.0^(42/40)  (position_embedding.py:96)
    w_ntk_factor = 4.0 ** (dim_w / (dim_w - 2))
    t_ntk_factor = 1.0 ** (dim_t / (dim_t - 2))         # 1.0

    # (position_embedding.py:127-129)  <-- computed as Float32 scalar
    h_theta = np.float32(10000.0 * h_ntk_factor)
    w_theta = np.float32(10000.0 * w_ntk_factor)
    t_theta = np.float32(10000.0 * t_ntk_factor)

    # freqs (position_embedding.py:131-133)
    h_spatial_freqs = vec2(1.0 / (h_theta ** dim_spatial_range))   # (21,)
    w_spatial_freqs = vec2(1.0 / (w_theta ** dim_spatial_range))   # (21,)
    temporal_freqs = vec2(1.0 / (t_theta ** dim_temporal_range))   # (22,)

    B, TT, HH, WW, _ = B_T_H_W_C
    seq = vec2(np.arange(max(HH, WW, TT)))           # (position_embedding.py:136)
    # image case: fps is None or enable_fps_modulation=False -> seq directly (pos:145-146)
    half_emb_h = np.outer(seq[:HH], h_spatial_freqs)                # (HH,21)
    half_emb_w = np.outer(seq[:WW], w_spatial_freqs)                # (WW,21)
    half_emb_t = np.outer(seq[:TT], temporal_freqs)                 # (TT,22)

    # stack [cos,-sin,sin,cos] (position_embedding.py:150-152)
    def stack4(x):
        return np.stack([np.cos(x), -np.sin(x), np.sin(x), np.cos(x)], axis=-1)
    half_emb_h = stack4(half_emb_h)                  # (HH,21,4)
    half_emb_w = stack4(half_emb_w)                  # (WW,21,4)
    half_emb_t = stack4(half_emb_t)                  # (TT,22,4)

    # concat in t,h,w order on the freq (d) axis (position_embedding.py:154-161)
    # em[T,H,W,d,4]: d in [0,22)=temporal, [22,43)=height, [43,64)=width
    # (position_embedding.py:154-161) the einops `repeat(h,d,x -> t h w d x)` keeps each
    # half's own freq length: temporal 22, height 21, width 21 (not the 42/44 dims).
    em = np.concatenate([
        np.broadcast_to(half_emb_t[:, None, None, :, :], (TT, HH, WW) + half_emb_t.shape[1:]),   # (.,,22,4)
        np.broadcast_to(half_emb_h[None, :, None, :, :], (TT, HH, WW) + half_emb_h.shape[1:]),   # (.,,21,4)
        np.broadcast_to(half_emb_w[None, None, :, :, :], (TT, HH, WW) + half_emb_w.shape[1:]),   # (.,,21,4)
    ], axis=3)                                        # (TT,HH,WW,64,4)

    # rearrange 't h w d (i j) -> (t h w) d i j' with i=2,j=2 (position_embedding.py:163)
    out = em.reshape(TT * HH * WW, NUM_FREQS, 2, 2)   # k order [cos,-sin,sin,cos]->i,j
    return out.astype(np.float32)


def load_f32(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=OUT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dump", action="store_true", help="write the oracle rope as .f32 + npz")
    ar = ap.parse_args()

    rope = generate_embeddings((1, T, H, W, HEAD_DIM))

    def cosine(a, b):
        a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def maxabs(a, b): return np.abs(a - b).max()
    def rmse(a, b): return np.sqrt(((a - b).astype(np.float32) ** 2).mean())

    print(f"Pinned ComfyUI commit: {PINNED_COMMIT}")
    print(f"config: head_dim={HEAD_DIM} dim_h={DIM_H} dim_w={DIM_W} dim_t={DIM_T} freqs={NUM_FREQS}")
    print(f"thetas: h={np.float32(10000.0*4.0**(42.0/40.0))!r} w={np.float32(10000.0*4.0**(42.0/40.0))!r} t={np.float32(10000.0)!r}")
    print(f"rope shape={rope.shape} (expect ({T*H*W}, {NUM_FREQS}, 2, 2)) dtypes range")
    print(f"allFinite={np.isfinite(rope).all()}  min={float(rope.min()):.6f} max={float(rope.max()):.6f}")
    print(f"block[0,0,:,:] (token0,temporal-freq0):\n{rope[0,0]}")
    print(f"anchor token0 d0 (temporal) block[0,0] -> {rope[0,0].tolist()}")
    print(f"anchor token0 d22 (height)  block[0,22] -> {rope[0,22].tolist()}")
    print(f"anchor token0 d43 (width)   block[0,43] -> {rope[0,43].tolist()}")
    # token17 = (h=0,w=17) single width step: block should be rotation by 16*aw
    print(f"anchor token=17 (row0,col17) d43 block -> {rope[17,43].tolist()}")

    # Compare vs Swift dump if present
    sw_path = os.path.join(ar.indir, "dit_rope_swift.f32")
    if os.path.exists(sw_path):
        sw = load_f32(sw_path, rope.shape)
        print("=== Swift vs oracle ===")
        print(f"shape oracle={rope.shape} swift={sw.shape} match={rope.shape==sw.shape}")
        print(f"cosine={cosine(rope, sw):.9f} maxAbs={maxabs(rope, sw):.2e} rmse={rmse(rope, sw):.2e}")
        print(f"swift allFinite={np.isfinite(sw).all()}")
        exact = np.array_equal(rope, sw)
        print(f"bit-exact={exact}")
    else:
        print(f"(no {sw_path} dump yet — run harness to generate Swift output)")

    if ar.dump:
        os.makedirs(ar.out, exist_ok=True)
        rope.astype(np.float32).tofile(os.path.join(ar.out, "dit_rope_oracle.f32"))
        np.savez_compressed(os.path.join(ar.out, "dit_rope_oracle.npz"),
                            rope=rope, dim_h=DIM_H, dim_w=DIM_W, dim_t=DIM_T,
                            h_theta=np.float32(10000.0*4.0**(42.0/40.0)),
                            commit=PINNED_COMMIT)
        print(f"wrote {os.path.join(ar.out,'dit_rope_oracle.f32')} + .npz")


if __name__ == "__main__":
    main()
