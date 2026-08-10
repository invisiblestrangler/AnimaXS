import Foundation

/// CPU-reference DiT 3-D RoPE embedding (H004) — transcribed VERBATIM from the pinned
/// ComfyUI source `comfy/ldm/cosmos/position_embedding.py` (commit cbbc9da):
///   - `VideoRopePosition3DEmb.__init__`             (position_embedding.py:57-98)
///   - `VideoRopePosition3DEmb.generate_embeddings`  (position_embedding.py:100-163)
///
/// At 512×512 (T=1, H=W=32, 2×2 patch → 1024 tokens) this produces the self-attention
/// RoPE basis tensor of shape [1024 tokens, 64 freqs, 2, 2] — each of the 128 head-dims
/// is covered by 64 frequency entries (22 temporal + 21 height + 21 width), each a 2×2
/// complex rotation block `[[cos, -sin], [sin, cos]]`.
///
/// Axis split / thetas (MODEL_ARCHITECTURE.json "rope", runbook §31):
///   head_dim=128 → dim_h=42, dim_w=42, dim_t=44
///   h/w extrapolation ratio 4.0, t ratio 1.0 → thetas computed (NOT rounded prose):
///     h_theta = w_theta = 10000.0 · 4.0^(42/40) ≈ 42870.938501451725
///     t_theta = 10000.0
///   (Model JSON prose "42871.1"/"42871.4" are rounded; use the computed value per D026.)
///
/// Frequency ordering in the fused 64-block row (position_embedding.py:154-161) is
/// **t, h, w**: freqs [0..22)=temporal, [22..43)=height, [43..64)=width. Do not reorder.
///
/// RoPE has NO learnable weights — the tensor is purely computed from the grid dims.
/// This is applied to self-attention only (predict2.py:184-192, is_selfattn gate).
struct DitRoPE {

    static let headDim = 128
    static let dimH = headDim / 6 * 2          // 42
    static let dimW = dimH                     // 42
    static let dimT = headDim - 2 * dimH       // 44
    /// Number of 2×2 rotation blocks per head: (dim_t//2)+(dim_h//2)+(dim_w//2) = 22+21+21 = 64.
    static let numFreqs = (dimT / 2) + (dimH / 2) + (dimW / 2)

    /// Exact computed thetas (position_embedding.py:127-129). h/w extrapolation ratio 4.0,
    /// t ratio 1.0. Computed in Double then narrowed to Float32 to match the numpy/torch
    /// `(float32 theta) ** (float32 range)` math.
    static let hTheta: Float = Float(10000.0 * pow(4.0, 42.0 / 40.0))  // ≈ 42870.938
    static let wTheta: Float = Float(10000.0 * pow(4.0, 42.0 / 40.0))
    static let tTheta: Float = Float(10000.0 * pow(1.0, 44.0 / 42.0))  // = 10000.0

    /// dim_spatial_range = arange(0, dim, 2)[: dim//2] / dim   (position_embedding.py:87-89)
    static func dimRange(_ dim: Int) -> [Float] {
        let count = dim / 2
        return (0..<count).map { Float($0 * 2) / Float(dim) }
    }

    /// 1.0 / theta**range  (position_embedding.py:131-133), computed in Float32.
    static func freqs(theta: Float, range: [Float]) -> [Float] {
        range.map { Float(1.0) / pow(theta, $0) }
    }

    /// Generate the [T*H*W, numFreqs, 2, 2] rope basis, flattened row-major to
    /// `[token][freq][i][j]` where each block is [cos, -sin, sin, cos] at (i,j) =
    /// (0,0),(0,1),(1,0),(1,1). Returns `flat` count = tokens · numFreqs · 4.
    ///
    /// - Parameters:
    ///   - T,H,W: grid dims (default 512×512 → T=1, H=W=32).
    static func generate(T: Int = 1, H: Int = 32, W: Int = 32) -> [Float] {
        let seq = (0..<max(max(H, W), T)).map { Float($0) }     // position_embedding.py:136
        let freqH = freqs(theta: hTheta, range: dimRange(dimH)) // 21
        let freqW = freqs(theta: wTheta, range: dimRange(dimW)) // 21
        let freqT = freqs(theta: tTheta, range: dimRange(dimT)) // 22

        let tokens = T * H * W
        var out = [Float](repeating: 0, count: tokens * Self.numFreqs * 4)

        var tok = 0
        for t in 0..<T {
            let argT = seq[t]                     // image case: use seq directly (T=1 →0)
            for h in 0..<H {
                let argH = seq[h]
                for w in 0..<W {
                    let argW = seq[w]
                    var f = 0
                    // temporal freqs [0..22)  -> block index 0..21
                    for d in 0..<(dimT / 2) {
                        writeBlock(&out, at: tok, freq: f, angle: argT * freqT[d])
                        f += 1
                    }
                    // height freqs [22..43)    -> block index 22..42
                    for d in 0..<(dimH / 2) {
                        writeBlock(&out, at: tok, freq: f, angle: argH * freqH[d])
                        f += 1
                    }
                    // width freqs [43..64)     -> block index 43..63
                    for d in 0..<(dimW / 2) {
                        writeBlock(&out, at: tok, freq: f, angle: argW * freqW[d])
                        f += 1
                    }
                    tok += 1
                }
            }
        }
        return out
    }

    /// Write a 2×2 block `[cos,-sin,sin,cos]` (position_embedding.py:150-152) at
    /// `(token, freq)` in the flat row-major tensor (block i,j laid out as k = i*2+j).
    private static func writeBlock(_ out: inout [Float], at tok: Int, freq f: Int, angle: Float) {
        let c = cos(angle)
        let s = sin(angle)
        let base = (tok * Self.numFreqs + f) * 4
        out[base + 0] = c      // (0,0)
        out[base + 1] = -s     // (0,1)
        out[base + 2] = s      // (1,0)
        out[base + 3] = c      // (1,1)
    }
}
