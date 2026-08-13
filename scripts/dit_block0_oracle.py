#!/usr/bin/env python3
"""
dit_block0_oracle.py — pinned-ComfyUI structural oracle for the DiT transformer Block (H005).

Purpose (AnimaXS runbook §32, TODO H005, DECISIONS D030–D035):
  Compute block 0 of the MiniTrainDIT end-to-end in numpy, using the EXACT equations of the
  pinned ComfyUI `Block.forward` (predict2.py:471-591) + `Attention` (103-216) +
  `GPT2FeedForward` (22-38), on the SAME W4-dequantized weights the Swift harness dumps.
  Validates Swift-vs-ComfyUI block math independently (Lane A). It also reports the
  W4-vs-original-checkpoint `block_00_out` comparison (Lane B), which is intentionally kept
  separate from structural parity.

KEY math locked by DECISIONS:
  D026  SiLU is applied to emb BEFORE Linear1 (predict2.py:451-465).
  D027  Block LayerNorm is elementwise_affine=False, mean-CENTERING; residual = x + gate*branch.
  D030  rms_rope_split_half pairs the two HALVES of the 128-d head: (p, p+64), NOT (2p,2p+1).
  D031  No DiT GQA (16/16/16 heads); no attention mask; cross context = full zero-padded
        [512,1024] adapter output (all 512 participate); extra_per_block_pos_emb = None;
        fp32 residual / fp16 compute in the real model, but H005 Lane A is all-fp32.

Weights are read from .f32 files dumped by the Swift harness (same row-aware W4 dequant per
D034). numpy uses memmap for large weights and query-tiled attention to stay well below
the 1.2 GB pack OOM threshold (it NEVER reads the .animapk).

Usage:
  python dit_block0_oracle.py [--in DIR] [--out DIR] [--golden PATH] [--checkpoints]
      [--golden-step 7 | --latent PATH] [--sigma VALUE] [--skip-swift]
      [--weight-domain LABEL] [--boundary-dtype {none,fp16,bf16}]
      [--residual-dtype {none,fp16,bf16}]

When --latent is supplied, the oracle rebuilds H001/H002's block input and timestep
vectors from the already-dumped component weights in the parent oracle directory. This
is useful for comparing a sampler invocation whose input is not the one used by the last
Swift dump; it still never opens the .animapk pack.
"""
import argparse, os, json
import numpy as np

PINNED_COMMIT = "cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44"
OUT = "/root/AnimaXS/scripts/oracle_out/block0"
GOLDEN = "/root/anima-xsmax/results/goldens/case1_danbooru_seed1337.npz"

DIM = 2048
HEADS = 16
HEAD_DIM = 128
CTX = 1024
MLP_HID = 8192
LORA = 256
EPS = 1e-6
TOKENS = 1024
CTX_TOKENS = 512
LEGACY_ATTENTION = False


# ---------------------------------------------------------------------------
# Loaders (memmap for large weights; np.load for small vectors)
# ---------------------------------------------------------------------------
def mmap_f32(path, shape):
    return np.memmap(path, mode="r", dtype="<f4", shape=tuple(shape))


def load_f32(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if a.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, got {a.size}")
    return a.reshape(shape)


def load_vec(path, n):
    return load_f32(path, (n,))


# ---------------------------------------------------------------------------
# Pinned ComfyUI equations (predict2.py) — verbatim, fp32
# ---------------------------------------------------------------------------
def rms_norm(x, w, eps=EPS):
    """RMSNorm over last dim (comfy RMSNorm). x[...,D], w[D]."""
    xf = x.astype(np.float32)
    mean_sq = (xf ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + eps)
    return (xf * inv) * w.astype(np.float32)


def layer_norm(x, eps=EPS):
    """LayerNorm elementwise_affine=False, mean-CENTERING (predict2.py:520-521)."""
    xf = x.astype(np.float32)
    mu = xf.mean(-1, keepdims=True)
    var = ((xf - mu) ** 2).mean(-1, keepdims=True)
    return (xf - mu) / np.sqrt(var + eps)


def layer_norm_modulated(x, scale, shift):
    """_fn (predict2.py:520-521): LayerNorm(x)*(1+scale)+shift."""
    return layer_norm(x) * (1 + scale) + shift


def rms_rope_split_half(q, k, rope, qw, kw, eps=EPS):
    """Fused per-head RMSNorm + split-half 3-D RoPE (D030) — explicit loop, primary oracle.
    q,k: [N, heads, head_dim]; rope: [N, head_dim//2, 2, 2]; qw/kw: [head_dim] shared across
    all heads. Pairs (p, p+half) where half = head_dim//2 (NOT adjacent (2p,2p+1)).
    out[p] = c*a - s*b ; out[p+half] = s*a + c*b   (a=x[p], b=x[p+half])."""
    qn = rms_norm(q, qw, eps)
    kn = rms_norm(k, kw, eps)
    half = qn.shape[-1] // 2
    nfreq = rope.shape[-3]
    qo = np.empty_like(qn)
    ko = np.empty_like(kn)
    N = qn.shape[0]
    for n in range(N):
        for h in range(HEADS):
            for p in range(nfreq):
                c = rope[n, p, 0, 0]
                s = rope[n, p, 1, 0]
                a = qn[n, h, p]; b = qn[n, h, p + half]
                qo[n, h, p] = c * a - s * b
                qo[n, h, p + half] = s * a + c * b
                a = kn[n, h, p]; b = kn[n, h, p + half]
                ko[n, h, p] = c * a - s * b
                ko[n, h, p + half] = s * a + c * b
    return qo, ko


def attention(q, k, v):
    """Scaled dot-product attention, query-tiled. q [Nq,heads,head_dim], k/v [Nkv,heads,head_dim].
    Returns [Nq,heads,head_dim]."""
    scale = HEAD_DIM ** -0.5
    Nq, Nkv = q.shape[0], k.shape[0]
    out = np.zeros_like(q)
    tile = 128
    for s in range(0, Nq, tile):
        qt = q[s:s + tile]                        # [T, heads, hd]
        # scores = einsum('qhd,khd->qhk', qt, k) * scale
        scores = np.einsum("qhd,khd->qhk", qt, k) * scale
        if LEGACY_ATTENTION:
            scores = scores.astype(np.float16).astype(np.float32)
        m = scores.max(-1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(-1, keepdims=True)
        if LEGACY_ATTENTION:
            p = p.astype(np.float16).astype(np.float32)
        out[s:s + tile] = np.einsum("qhk,khd->qhd", p, v)
    return out


def split_heads(x, N, H, D):
    """x [N, H*D] -> [N, H, D]."""
    return x.reshape(N, H, D)


def concat_heads(x):
    """x [N, H, D] -> [N, H*D] (head-major: head0 dims first)."""
    N, H, D = x.shape
    return x.reshape(N, H * D)


def gelu_fast(x):
    """Exact GELU using numpy's erf-free form via math.erf vectorized once."""
    from math import erf
    vf = np.vectorize(erf, otypes=[np.float32])
    return (0.5 * x * (1 + vf(x / np.sqrt(np.float32(2.0))))).astype(np.float32)


def boundary_quantize(x, dtype):
    """Optional reference-boundary emulation; residual accumulation stays fp32.

    The canonical Lane A path is dtype=none. The frozen reference commonly runs with
    reduced compute dtype, so this diagnostic can quantize branch inputs and projection
    results at the same broad boundaries without changing the oracle's residual
    representation. bfloat16 uses torch only when explicitly requested.
    """
    if dtype == "none":
        return x
    if dtype == "fp16":
        return np.asarray(x, dtype=np.float16).astype(np.float32)
    if dtype == "bf16":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("--boundary-dtype bf16 requires torch in the active environment") from exc
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).bfloat16().float().numpy()
    raise ValueError(f"unknown boundary dtype: {dtype}")


def timesteps_sinusoidal(sigma):
    """H002 Timesteps.forward: [cos(sigma*f), sin(sigma*f)], dim 2048."""
    half = 1024
    exponent = -np.log(np.float32(10000.0)) * np.arange(half, dtype=np.float32)
    exponent = exponent / np.float32(half)
    freq = np.exp(exponent).astype(np.float32)
    phase = (np.float32(sigma) * freq).astype(np.float32)
    return np.concatenate([np.cos(phase), np.sin(phase)]).astype(np.float32)


def patchify_input(latent):
    """H001 PatchEmbed ordering for latent [16,64,64] plus zero padding channel."""
    x17 = np.concatenate([latent.astype(np.float32), np.zeros((1, 64, 64), dtype=np.float32)], axis=0)
    tokens = np.empty((TOKENS, 68), dtype=np.float32)
    t = 0
    for h in range(32):
        for w in range(32):
            d = 0
            for c in range(17):
                for m in range(2):
                    for n in range(2):
                        tokens[t, d] = x17[c, h * 2 + m, w * 2 + n]
                        d += 1
            t += 1
    return tokens


def build_component_inputs(component_dir, latent, sigma):
    """Rebuild H001/H002 inputs from prior validated .f32 component dumps."""
    w_x = load_f32(os.path.join(component_dir, "dit_x_embedder_weight.f32"), (DIM, 68))
    w_l1 = load_f32(os.path.join(component_dir, "dit_ts_linear1_weight.f32"), (DIM, DIM))
    w_l2 = load_f32(os.path.join(component_dir, "dit_ts_linear2_weight.f32"), (3 * DIM, DIM))
    w_norm = load_vec(os.path.join(component_dir, "dit_ts_norm_weight.f32"), DIM)
    if isinstance(latent, (str, os.PathLike)):
        latent = load_f32(latent, (16, 64, 64))
    latent = np.asarray(latent, dtype=np.float32).reshape(16, 64, 64)

    tokens = patchify_input(latent)
    x = (tokens @ w_x.T).astype(np.float32)

    raw = timesteps_sinusoidal(sigma)
    hidden = (raw @ w_l1.T).astype(np.float32)
    hidden = (hidden * (1.0 / (1.0 + np.exp(-hidden)))).astype(np.float32)
    adaln = (hidden @ w_l2.T).astype(np.float32)
    emb = rms_norm(raw, w_norm).astype(np.float32)
    return x, emb, adaln


def reconstruct_sampler_input(golden_path, step):
    """Recover the model input for an Euler invocation from trace_anima.py's capture.

    The trace callback stores `denoised`, despite naming that argument `x`. Block hooks are
    overwritten on every model call, so block_00_out corresponds to invocation 7. Euler's
    state before invocation `step` is recovered from raw noise and denoised[0..<step].
    """
    with np.load(golden_path) as g:
        sigmas = g["sigmas_comfy"].astype(np.float32)
        denoised = g["step_latents"].astype(np.float32)
        state = g["init_noise_randn"].astype(np.float32).reshape(16, 64, 64).copy()
    if not 0 <= step < len(sigmas) - 1:
        raise ValueError(f"golden step must be in [0,{len(sigmas)-2}], got {step}")
    for i in range(step):
        d = (state - denoised[i].reshape(16, 64, 64)) / sigmas[i]
        state = (state + d * (sigmas[i + 1] - sigmas[i])).astype(np.float32)
    return state, float(sigmas[step])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=OUT)
    ap.add_argument("--out", default=OUT, help="output directory for fixture/checkpoint files")
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--checkpoints", action="store_true", help="dump intermediate stage tensors")
    ap.add_argument("--latent", help="optional [16,64,64] model-input latent; rebuild H001/H002 inputs")
    ap.add_argument("--golden-step", type=int,
                    help="reconstruct this sampler invocation directly from golden noise/denoised states")
    ap.add_argument("--sigma", type=float, help="sigma for --latent (defaults to golden sampler step 7)")
    ap.add_argument("--components", help="directory containing validated H001/H002 .f32 dumps")
    ap.add_argument("--cross-context", help="override adapter context [512,1024] .f32")
    ap.add_argument("--input-x", help="override block input x .f32")
    ap.add_argument("--emb", help="override timestep embedding .f32")
    ap.add_argument("--adaln", help="override adaln_lora .f32")
    ap.add_argument("--skip-swift", action="store_true", help="do not compare a possibly stale Swift output dump")
    ap.add_argument("--weight-domain", default="W4", help="label used in Lane B reports")
    ap.add_argument("--boundary-dtype", choices=("none", "fp16", "bf16"), default="none",
                    help="optional reduced-dtype branch-boundary diagnostic; default is all-fp32 Lane A")
    ap.add_argument("--residual-dtype", choices=("none", "fp16", "bf16"), default="none",
                    help="optional residual-stream quantization diagnostic; default is fp32")
    ap.add_argument("--legacy-attention", action="store_true",
                    help="round attention scores and probabilities through fp16 like production")
    args = ap.parse_args()
    global LEGACY_ATTENTION
    LEGACY_ATTENTION = args.legacy_attention
    if args.latent and args.golden_step is not None:
        ap.error("--latent and --golden-step are mutually exclusive")
    D = os.path.join(args.indir, "")

    # ---- inputs (from the Swift harness / already-validated H001-H004 components) ----
    input_provenance = "Swift dump: " + (D + "block0_input_x.f32")
    if args.golden_step is not None:
        latent, sigma = reconstruct_sampler_input(args.golden, args.golden_step)
        if args.sigma is not None and not np.isclose(np.float32(args.sigma), np.float32(sigma)):
            ap.error(f"--sigma {args.sigma} does not match golden step {args.golden_step} sigma {sigma}")
        component_dir = args.components or os.path.dirname(os.path.abspath(args.indir))
        x, emb, adaln = build_component_inputs(component_dir, latent, sigma)
        input_provenance = f"reconstructed from golden sampler step={args.golden_step}, sigma={np.float32(sigma).item()}"
    elif args.latent:
        sigma = args.sigma if args.sigma is not None else 0.3050089478492737
        component_dir = args.components or os.path.dirname(os.path.abspath(args.indir))
        x, emb, adaln = build_component_inputs(component_dir, args.latent, sigma)
        input_provenance = f"reconstructed from latent={args.latent}, sigma={np.float32(sigma).item()}"
    else:
        x = load_f32(args.input_x or (D + "block0_input_x.f32"), (TOKENS, DIM))
        emb = load_vec(args.emb or (D + "block0_emb.f32"), DIM)
        adaln = load_vec(args.adaln or (D + "block0_adaln_lora.f32"), 3 * DIM)
    ctx = load_f32(args.cross_context or (D + "block0_cross_ctx.f32"), (CTX_TOKENS, CTX))
    rope = load_f32(D + "block0_rope.f32", (TOKENS, HEAD_DIM // 2, 2, 2))  # H004 rope

    # ---- block-0 dequantized weights ----
    def w(name, shape): return mmap_f32(D + f"block0_{name}.f32", shape)
    def vec(name, n): return load_vec(D + f"block0_{name}.f32", n)
    def boundary(a): return boundary_quantize(a, args.boundary_dtype)
    def residual(a): return boundary_quantize(a, args.residual_dtype)

    # AdaLN
    mod_self_w1 = w("mod_self_w1", (LORA, DIM)); mod_self_w2 = w("mod_self_w2", (3 * DIM, LORA))
    mod_cross_w1 = w("mod_cross_w1", (LORA, DIM)); mod_cross_w2 = w("mod_cross_w2", (3 * DIM, LORA))
    mod_mlp_w1 = w("mod_mlp_w1", (LORA, DIM)); mod_mlp_w2 = w("mod_mlp_w2", (3 * DIM, LORA))
    # Self attn
    self_q = w("self_q", (DIM, DIM)); self_k = w("self_k", (DIM, DIM))
    self_v = w("self_v", (DIM, DIM)); self_o = w("self_o", (DIM, DIM))
    self_qn = vec("self_q_norm", HEAD_DIM); self_kn = vec("self_k_norm", HEAD_DIM)
    # Cross attn
    cross_q = w("cross_q", (DIM, DIM)); cross_k = w("cross_k", (DIM, CTX))
    cross_v = w("cross_v", (DIM, CTX)); cross_o = w("cross_o", (DIM, DIM))
    cross_qn = vec("cross_q_norm", HEAD_DIM); cross_kn = vec("cross_k_norm", HEAD_DIM)
    # MLP
    mlp_w1 = w("mlp_w1", (MLP_HID, DIM)); mlp_w2 = w("mlp_w2", (DIM, MLP_HID))

    print(f"Pinned ComfyUI commit: {PINNED_COMMIT}")
    print(f"Input provenance: {input_provenance}")

    # ======================================================================
    # 1. AdaLN modulation (predict2.py:486-495): SiLU -> Linear1 -> Linear2, + adaln_lora
    # ======================================================================
    def modulate(w1, w2):
        silu = emb * (1.0 / (1.0 + np.exp(-emb)))          # SiLU
        h1 = silu @ w1.T                                    # [LORA]
        mod = (h1 @ w2.T) + adaln                          # [3*DIM]
        sh, sc, ga = np.split(mod, 3)                       # chunk(3) -> shift,scale,gate
        return sh, sc, ga
    self_shift, self_scale, self_gate = modulate(mod_self_w1, mod_self_w2)
    cross_shift, cross_scale, cross_gate = modulate(mod_cross_w1, mod_cross_w2)
    mlp_shift, mlp_scale, mlp_gate = modulate(mod_mlp_w1, mod_mlp_w2)

    print(f"self_shift shape={self_shift.shape} (expect (2048,))")

    # ======================================================================
    # 2. SELF-ATTENTION (predict2.py:523-542)
    # ======================================================================
    x = residual(x)
    self_norm = boundary(layer_norm_modulated(x, self_scale, self_shift))  # [1024,2048]
    q = boundary(self_norm @ self_q.T)                                 # [1024,2048]
    k = boundary(self_norm @ self_k.T)
    v = boundary(self_norm @ self_v.T)
    q = split_heads(q, TOKENS, HEADS, HEAD_DIM)
    k = split_heads(k, TOKENS, HEADS, HEAD_DIM)
    v = split_heads(v, TOKENS, HEADS, HEAD_DIM)
    q, k = rms_rope_split_half(q, k, rope, self_qn, self_kn)  # D030 split-half
    q = boundary(q)
    k = boundary(k)
    self_q_rope = q.copy()
    self_k_rope = k.copy()
    self_attn = attention(q, k, v)                                   # [1024,16,128]
    self_out = boundary(boundary(concat_heads(self_attn)) @ self_o.T) # [1024,2048]
    x1 = residual(x + self_gate * self_out)
    print(f"x1 (after self) shape={x1.shape} (expect (1024,2048))")

    # ======================================================================
    # 3. CROSS-ATTENTION (predict2.py:568-575): context = adapter [512,1024], no RoPE
    # ======================================================================
    cross_norm = boundary(layer_norm_modulated(x1, cross_scale, cross_shift))
    q = boundary(cross_norm @ cross_q.T)                              # [1024,2048]
    k = boundary(ctx @ cross_k.T)                                    # [512,2048]
    v = boundary(ctx @ cross_v.T)
    q = split_heads(q, TOKENS, HEADS, HEAD_DIM)
    k = split_heads(k, CTX_TOKENS, HEADS, HEAD_DIM)
    v = split_heads(v, CTX_TOKENS, HEADS, HEAD_DIM)
    q = rms_norm(q, cross_qn)                                        # per-head RMSNorm, NO RoPE
    k = rms_norm(k, cross_kn)
    q = boundary(q)
    k = boundary(k)
    cross_q_norm = q.copy()
    cross_k_norm = k.copy()
    cross_attn = attention(q, k, v)                                  # [1024,16,128]
    cross_out = boundary(boundary(concat_heads(cross_attn)) @ cross_o.T)
    x2 = residual(x1 + cross_gate * cross_out)
    print(f"x2 (after cross) shape={x2.shape} (expect (1024,2048))")

    # ======================================================================
    # 4. MLP (predict2.py:577-590): Linear1 -> exact GELU -> Linear2
    # ======================================================================
    mlp_norm = boundary(layer_norm_modulated(x2, mlp_scale, mlp_shift))
    h = boundary(mlp_norm @ mlp_w1.T)                                 # [1024,8192]
    h = boundary(gelu_fast(h))
    y = boundary(h @ mlp_w2.T)                                        # [1024,2048]
    x3 = residual(x2 + mlp_gate * y)
    print(f"x3 (block0 output) shape={x3.shape} (expect (1024,2048))")

    # ======================================================================
    # Compare vs Swift dump
    # ======================================================================
    def cosine(a, b):
        a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def maxabs(a, b): return np.abs(a - b).max()
    def rmse(a, b): return np.sqrt(((a - b).astype(np.float32) ** 2).mean())

    sw_path = os.path.join(args.indir, "block0_swift_output.f32")
    sw = None
    if os.path.exists(sw_path) and not args.skip_swift:
        sw = load_f32(sw_path, (TOKENS, DIM))
        print("=== Swift vs NumPy oracle (Lane A) ===")
        print(f"shape oracle={x3.shape} swift={sw.shape} match={x3.shape==sw.shape}")
        print(f"cosine={cosine(x3, sw):.9f} maxAbs={maxabs(x3, sw):.2e} rmse={rmse(x3, sw):.2e}")
        print(f"swift allFinite={np.isfinite(sw).all()}  oracle allFinite={np.isfinite(x3).all()}")
    else:
        print(f"(no {sw_path} — run harness to generate Swift output)")

    # ======================================================================
    # Lane B: vs golden block_00_out
    # ======================================================================
    if os.path.exists(args.golden):
        g = np.load(args.golden)["block_00_out"]                     # (1,1,32,32,2048)
        g = g.reshape(TOKENS, DIM)
        print(f"=== NumPy-{args.weight_domain} vs golden block_00_out (Lane B) ===")
        print(f"shape oracle={x3.shape} golden={g.shape} match={x3.shape==g.shape}")
        print(f"cosine={cosine(x3, g):.9f} maxAbs={maxabs(x3, g):.2e} rmse={rmse(x3, g):.2e}")
        print(f"golden allFinite={np.isfinite(g).all()}  min={float(g.min()):.4f} max={float(g.max()):.4f}")
        rel_l2 = np.linalg.norm((x3 - g).reshape(-1)) / np.linalg.norm(g.reshape(-1))
        print(f"relative L2 = {rel_l2:.6e}")
        if sw is not None:
            print(f"=== Swift-{args.weight_domain} vs golden block_00_out (Lane B) ===")
            print(f"cosine={cosine(sw, g):.9f} maxAbs={maxabs(sw, g):.2e} rmse={rmse(sw, g):.2e} relL2={np.linalg.norm((sw-g).reshape(-1))/np.linalg.norm(g.reshape(-1)):.6e}")

    # Checkpoints (diagnostic) + fixture
    os.makedirs(args.out, exist_ok=True)
    if args.checkpoints:
        for name, arr in [("self_norm", self_norm), ("self_q_rope", self_q_rope),
                          ("self_k_rope", self_k_rope), ("x1", x1),
                          ("cross_norm", cross_norm), ("cross_q_norm", cross_q_norm),
                          ("cross_k_norm", cross_k_norm), ("x2", x2),
                          ("mlp_norm", mlp_norm), ("x3", x3)]:
            arr.astype(np.float32).tofile(os.path.join(args.out, f"block0_{name}.f32"))
    np.savez_compressed(os.path.join(args.out, "dit_block0_oracle_case1.npz"),
                        block0_output=x3, self_gate=self_gate, cross_gate=cross_gate,
                        mlp_gate=mlp_gate, commit=PINNED_COMMIT,
                        weight_domain=args.weight_domain, input_provenance=input_provenance)
    print(f"saved {os.path.join(args.out,'dit_block0_oracle_case1.npz')}")


if __name__ == "__main__":
    main()
