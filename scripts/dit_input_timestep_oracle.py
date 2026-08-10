#!/usr/bin/env python3
"""
dit_input_timestep_oracle.py — pinned-ComfyUI structural oracle for DiT input (H001)
and timestep embedder (H002).

Purpose (AnimaXS runbook §29–§30, TODO H001/H002):
  Compute (a) the DiT input embedding [1024, 2048] from a 16-ch latent via patchify 2×2
  + x_embedder Linear(68→2048), and (b) the timestep embedding (t_embedding_B_T_D [2048])
  + adaln_lora_B_T_3D [6144] from sigma, using the EXACT equations of the pinned ComfyUI
  MiniTrainDIT, on the SAME W4-dequantized weights the Swift harness consumes. Validates
  model-implementation correctness (Swift vs ComfyUI math) independently.

Pinned source (commit cbbc9da), comfy/ldm/cosmos/predict2.py:
  Timesteps.forward                       :219-238
  TimestepEmbedding.forward               :241-269
  PatchEmbed (Rearrange + Linear)         :272-334
  MiniTrainDIT.prepare_embedded_sequence  :776-827  (padding-mask concat)
  MiniTrainDIT._forward                   :853-931  (t_embedder + t_embedding_norm)

Weights are read from .f32 files dumped by the Swift harness (same W4 dequant, verified
byte-exact D011). This oracle independently re-implements the FORWARD MATH in numpy.

Usage:
  python dit_input_timestep_oracle.py [--out DIR] [--in DIR]
"""
import argparse, os, numpy as np

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"
OUT = "/root/AnimaXS/scripts/oracle_out"
HIDDEN = 2048
HALF = 1024
TOKENDIM = 68
ADALN = 6144


# ---------------------------------------------------------------------------
# Pinned ComfyUI equations — VERBATIM (predict2.py)
# ---------------------------------------------------------------------------
# Timesteps.forward (predict2.py:219-238)
def timesteps_sinusoidal(sigma: float):
    """Returns [1,1,2048] = cat([cos(emb), sin(emb)]) with emb = sigma*exp(-log(10000)*arange(1024)/1024)."""
    half = HALF
    exponent = -np.log(10000.0) * np.arange(half, dtype=np.float32)
    exponent = exponent / (half - 0.0)
    emb = np.exp(exponent)
    emb = (np.float32(sigma) * emb).astype(np.float32)          # [1024]
    sin_emb = np.sin(emb)
    cos_emb = np.cos(emb)
    out = np.concatenate([cos_emb, sin_emb], axis=-1).astype(np.float32)  # [2048]
    return out.reshape(1, 1, 2048)


# TimestepEmbedding.forward (predict2.py:257-269)
def timestep_embedding(raw, w1, w2):
    """raw [1,1,2048]. Returns (emb_B_T_D, adaln_lora_B_T_3D) per use_adaln_lora=True."""
    # emb = linear_1(sample); SiLU; linear_2
    emb = raw @ w1.T                       # Linear1 [1,1,2048]×[2048,2048]ᵀ (bias=False)
    emb = emb * (1.0 / (1.0 + np.exp(-emb)))  # SiLU
    adaln_lora = emb @ w2.T                # Linear2 [1,1,6144] (bias=False)
    emb_B_T_D = raw                        # use_adaln_lora=True → emb_B_T_D = sample
    return emb_B_T_D, adaln_lora


# RMSNorm (comfy/rmsnorm.py + operations)
def rms_norm(x, w, eps=1e-6):
    xf = x.astype(np.float32)
    mean_sq = (xf ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + eps)
    return (xf * inv) * w.astype(np.float32)


# PatchEmbed (predict2.py:299-309): Rearrange "b c (t r) (h m) (w n) -> b t h w (c r m n)"
# with r=1, m=2, n=2, then Linear(68→2048, bias=False).
def patchify(x, in_channels, H, W, p=2):
    """x [C=17, H=64, W=64] -> tokens [H/p * W/p, 68].
    Out token (h,w) feature index = c*4 + m*2 + n (r=0)."""
    C = x.shape[0]
    ph, pw = H // p, W // p
    tokens = []
    for h in range(ph):
        for w in range(pw):
            feat = []
            for c in range(C):
                for m in range(p):
                    for n in range(p):
                        feat.append(x[c, h * p + m, w * p + n])
            tokens.append(feat)
    return np.array(tokens, dtype=np.float32)  # [ph*pw, 68]


def x_embedder(tokens, w):
    """tokens [N,68] @ w.T -> [N,2048] (Linear bias=False)."""
    return tokens @ w.T


# MiniTrainDIT.prepare_embedded_sequence (predict2.py:806-816): padding-mask concat
def prepare_embedded_sequence(latent):
    """latent [16,64,64] -> [17,64,64] (channel 16 = zeros padding mask), then patchify+embed."""
    C, H, W = latent.shape
    pad = np.zeros((1, H, W), dtype=np.float32)
    x17 = np.concatenate([latent, pad], axis=0)   # [17,64,64]
    return x17


# ---------------------------------------------------------------------------
# Loaders (read .f32 dumped by the Swift harness — same W4 dequant)
# ---------------------------------------------------------------------------
def load_f32(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=OUT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    w_x = load_f32(os.path.join(args.indir, "dit_x_embedder_weight.f32"), (HIDDEN, TOKENDIM))
    w_l1 = load_f32(os.path.join(args.indir, "dit_ts_linear1_weight.f32"), (HIDDEN, HIDDEN))
    w_l2 = load_f32(os.path.join(args.indir, "dit_ts_linear2_weight.f32"), (ADALN, HIDDEN))
    w_norm = load_f32(os.path.join(args.indir, "dit_ts_norm_weight.f32"), (HIDDEN,))
    latent = load_f32(os.path.join(args.indir, "dit_input_latent.f32"), (16, 64, 64))

    # ---- Timestep (sigma = 1.0) ----
    sigma = 1.0
    raw = timesteps_sinusoidal(sigma)                       # [1,1,2048]
    emb_bt_d, adaln_lora = timestep_embedding(raw, w_l1, w_l2)  # raw = emb_B_T_D
    t_embedding = rms_norm(emb_bt_d[0, 0], w_norm)          # RMSNorm on raw sinusoidal → [2048]

    # ---- DiT input ----
    x17 = prepare_embedded_sequence(latent)                 # [17,64,64]
    tokens = patchify(x17, 17, 64, 64)                      # [1024, 68]
    x_emb = x_embedder(tokens, w_x)                         # [1024, 2048]

    # ---- Compare vs Swift dumped outputs ----
    sw_emb = load_f32(os.path.join(args.indir, "dit_input_embedding.f32"), (1024, HIDDEN))
    sw_ts_emb = load_f32(os.path.join(args.indir, "dit_ts_embedding.f32"), (HIDDEN,))
    sw_adaln = load_f32(os.path.join(args.indir, "dit_ts_adaln_lora.f32"), (ADALN,))
    sw_raw = load_f32(os.path.join(args.indir, "dit_ts_raw_sinusoid.f32"), (HIDDEN,))

    def cosine(a, b):
        a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def maxabs(a, b): return np.abs(a - b).max()
    def rmse(a, b): return np.sqrt(((a - b).astype(np.float32) ** 2).mean())

    print(f"Pinned ComfyUI commit: {PINNED_COMMIT}")
    print("=== Timestep (sigma=1.0) ===")
    print(f"  raw sinusoid [2048]: shape={raw.shape} cosine={cosine(raw[0,0], sw_raw):.8f} maxAbs={maxabs(raw[0,0], sw_raw):.2e} rmse={rmse(raw[0,0], sw_raw):.2e}")
    print(f"  embedding [2048]:    shape={t_embedding.shape} cosine={cosine(t_embedding, sw_ts_emb):.8f} maxAbs={maxabs(t_embedding, sw_ts_emb):.2e} rmse={rmse(t_embedding, sw_ts_emb):.2e}")
    print(f"  adaln_lora [6144]:   shape={adaln_lora.shape} cosine={cosine(adaln_lora[0,0], sw_adaln):.8f} maxAbs={maxabs(adaln_lora[0,0], sw_adaln):.2e} rmse={rmse(adaln_lora[0,0], sw_adaln):.2e}")
    print(f"  adaln_lora allFinite={np.isfinite(adaln_lora).all()}  embedding allFinite={np.isfinite(t_embedding).all()}")
    print(f"  embedding[0..3]={[round(float(v),6) for v in t_embedding[:4]]}")
    print(f"  adaln_lora min/max={float(adaln_lora.min()):.4f}/{float(adaln_lora.max()):.4f}")
    print("=== DiT input ===")
    print(f"  tokens shape={tokens.shape} (expect (1024,68))  tokens[0,:8]={[round(float(v),4) for v in tokens[0,:8]]}")
    print(f"  x_embedder shape={x_emb.shape} (expect (1024,2048))")
    print(f"  x_emb cosine={cosine(x_emb, sw_emb):.8f} maxAbs={maxabs(x_emb, sw_emb):.2e} rmse={rmse(x_emb, sw_emb):.2e}")
    print(f"  x_emb allFinite={np.isfinite(x_emb).all()}  x_emb min/max={float(x_emb.min()):.4f}/{float(x_emb.max()):.4f}")
    print(f"  x_emb token0 d0..3={[round(float(v),6) for v in x_emb[0,:4]]}")

    # -----------------------------------------------------------------------
    # H003: AdaLN-LoRA modulation — Block.forward (predict2.py:486-516, 520-521)
    # -----------------------------------------------------------------------
    print("=== AdaLN-LoRA modulation (block 0, sigma=1.0) ===")
    emb_b = sw_ts_emb                      # t_embedding_B_T_D [2048]
    adaln_b = sw_adaln                     # adaln_lora_B_T_3D [6144]
    for branch in ["self_attn", "cross_attn", "mlp"]:
        w1 = load_f32(os.path.join(args.indir, f"dit_mod_{branch}_w1.f32"), (256, 2048))
        w2 = load_f32(os.path.join(args.indir, f"dit_mod_{branch}_w2.f32"), (6144, 256))
        # adaln_modulation_* = nn.Sequential(SiLU, Linear(2048→256), Linear(256→6144))
        # (predict2.py:451-465). Order: mod = Linear2( Linear1( SiLU(emb) ) ) + adaln_lora.
        silu = emb_b * (1.0 / (1.0 + np.exp(-emb_b)))          # SiLU first
        h1 = (silu @ w1.T).astype(np.float32)                   # Linear1 [256]
        mod = (h1 @ w2.T).astype(np.float32) + adaln_b          # Linear2 [6144] + lora
        shift, scale, gate = np.split(mod, 3)                   # chunk(3, dim=-1)
        for name, arr, sw in [("shift", shift, f"dit_mod_{branch}_shift.f32"),
                              ("scale", scale, f"dit_mod_{branch}_scale.f32"),
                              ("gate",  gate,  f"dit_mod_{branch}_gate.f32")]:
            s = load_f32(os.path.join(args.indir, sw), (2048,))
            print(f"  {branch}.{name}: cosine={cosine(arr, s):.8f} maxAbs={maxabs(arr, s):.2e} allFinite={np.isfinite(arr).all()}")

    # Write oracle outputs for the record
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "dit_input_timestep_oracle_case1.npz"),
                        x_embedding=x_emb, t_embedding=t_embedding, adaln_lora=adaln_lora[0,0],
                        raw_sinusoid=raw[0,0], tokens=tokens, x17=x17,
                        commit=PINNED_COMMIT)
    print("saved dit_input_timestep_oracle_case1.npz")


if __name__ == "__main__":
    main()
