import Foundation

/// CPU-reference DiT transformer `Block` (H005) — transcribed VERBATIM from the pinned
/// ComfyUI source `comfy/ldm/cosmos/predict2.py` (commit cbbc9da), `Block.forward`
/// (471-591) + `Attention` (103-216) + `GPT2FeedForward` (22-38).
///
/// This is the H005 structural correctness gate: a single MiniTrainDIT block, end-to-end,
/// with Swift-vs-NumPy parity on the same W4-dequantized inputs and weights. It reuses the
/// already-validated H001–H004 pieces (DiTInput → block input x, TimestepEmbedder →
/// emb/adaln_lora, DitRoPE → rope, LLMAdapter → cross context) and only adds the block-level
/// primitives. The original golden is a separate Lane B comparison because it was captured
/// from the source BF16 checkpoint; the source-proven W4 tolerance is recorded in D035.
///
/// Reference dataflow (fp32 transcription for the structural oracle — see D031, Lane A):
///   1. AdaLN modulation (3 branches, SiLU first per D026): shift/scale/gate each [2048].
///   2. SELF-ATTN: LayerNorm(x)*(1+scale)+shift → q/k/v proj [2048] → reshape [16,128] →
///      per-head RMSNorm(128) + split-half 3-D RoPE (D030: pairs (p,p+64)) → MHA →
///      concat heads → output_proj → x += gate*branch.
///   3. CROSS-ATTN: LayerNorm+AdaLN → q proj(norm_x), k/v proj(context[512,1024]) →
///      per-head RMSNorm (NO RoPE) → MHA → output_proj → x += gate*branch.
///   4. MLP: LayerNorm+AdaLN → Linear(2048→8192) → exact GELU → Linear(8192→2048) →
///      x += gate*branch.
///
/// All block-level math is Float32 (D031). Weight tensors (all NO bias, W4 in pack):
///   blocks.N.adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}.weight
///   blocks.N.self_attn.{q,k,v}_proj.weight [2048,2048], output_proj.weight, {q,k}_norm.weight [128]
///   blocks.N.cross_attn.q_proj.weight [2048,2048], {k,v}_proj.weight [2048,1024],
///        output_proj.weight, {q,k}_norm.weight [128]
///   blocks.N.mlp.layer{1,2}.weight [8192,2048]/[2048,8192]
enum DiTBlockCPU {

    static let dim = 2048
    static let heads = 16
    static let headDim = 128
    static let ctxDim = 1024
    static let mlpHidden = 8192
    static let eps: Float = 1e-6

    /// Dequantized weights for one block (block 0 for H005). All NO bias.
    struct Weights {
        // AdaLN
        var modSelfW1: [[Float]]   // [256,2048]
        var modSelfW2: [[Float]]   // [6144,256]
        var modCrossW1: [[Float]]
        var modCrossW2: [[Float]]
        var modMLPW1: [[Float]]
        var modMLPW2: [[Float]]
        // Self-attn
        var selfQ: [[Float]]       // [2048,2048]
        var selfK: [[Float]]
        var selfV: [[Float]]
        var selfO: [[Float]]
        var selfQNorm: [Float]     // [128] shared across heads
        var selfKNorm: [Float]
        // Cross-attn
        var crossQ: [[Float]]      // [2048,2048]
        var crossK: [[Float]]      // [2048,1024]
        var crossV: [[Float]]      // [2048,1024]
        var crossO: [[Float]]      // [2048,2048]
        var crossQNorm: [Float]    // [128]
        var crossKNorm: [Float]
        // MLP
        var mlpW1: [[Float]]       // [8192,2048]
        var mlpW2: [[Float]]       // [2048,8192]
    }

    // MARK: - Primitives (independently testable)

    /// LayerNorm over the last dim, elementwise_affine=False, mean-CENTERING (D027).
    ///   out = (x - mean) / sqrt(var + eps)
    static func layerNorm(_ x: [Float]) -> [Float] {
        let n = x.count
        var mean: Float = 0
        for v in x { mean += v }
        mean /= Float(n)
        var varSum: Float = 0
        for v in x { let d = v - mean; varSum += d * d }
        let variance = varSum / Float(n)
        let inv = 1.0 / sqrt(variance + Self.eps)
        return x.map { ($0 - mean) * inv }
    }

    /// LayerNorm + AdaLN (predict2.py:520-521 `_fn`): norm(x) * (1+scale) + shift.
    /// `x` is a FLAT [tokens*dim] buffer; LayerNorm is per-row (last dim), scale/shift are
    /// [dim] and broadcast across tokens. Returns flat [tokens*dim].
    static func layerNormModulated(_ x: [Float], scale: [Float], shift: [Float]) -> [Float] {
        let dim = scale.count
        precondition(x.count % dim == 0, "x.count \(x.count) not a multiple of dim \(dim)")
        let tokens = x.count / dim
        var out = [Float](repeating: 0, count: x.count)
        for t in 0..<tokens {
            let base = t * dim
            let row = Array(x[base..<(base + dim)])
            let ln = layerNorm(row)
            for i in 0..<dim { out[base + i] = ln[i] * (1 + scale[i]) + shift[i] }
        }
        return out
    }

    /// Per-token-per-head RMSNorm(128) with a SHARED [128] weight (D030/D031). Applied to a
    /// flat [tokens, heads, headDim] buffer (row-major). eps 1e-6, no mean-subtraction.
    ///   out[i,h,:] = x[i,h,:] / sqrt(mean(x[i,h,:]^2) + eps) * weight
    static func rmsNormHeads(_ x: [Float], weight: [Float], tokens: Int, heads: Int, headDim: Int) -> [Float] {
        precondition(x.count == tokens * heads * headDim)
        var out = [Float](repeating: 0, count: x.count)
        for t in 0..<tokens {
            for h in 0..<heads {
                let base = (t * heads + h) * headDim
                var sum: Float = 0
                for d in 0..<headDim { let v = x[base + d]; sum += v * v }
                let inv = 1.0 / sqrt(sum / Float(headDim) + Self.eps)
                for d in 0..<headDim { out[base + d] = x[base + d] * inv * weight[d] }
            }
        }
        return out
    }

    /// Apply the 3-D RoPE to a flat [tokens, heads, headDim] buffer using the SPLIT-HALF
    /// pairing (D030): pair p = (p, p+headDim/2), NOT adjacent (2p, 2p+1).
    ///   rope: flat [tokens, numFreqs, 4] where numFreqs = headDim/2 and the 4 = 2×2 block
    ///         [m00,m01,m10,m11] = [cos, -sin, sin, cos].
    ///   out[p]     = c*a - s*b
    ///   out[p+half] = s*a + c*b      (a=x[p], b=x[p+half])
    /// For T=1,H=W=32 this rope is DitRoPE.generate's [1024,64,4] flat output.
    static func applySplitHalfRoPE(_ x: [Float], rope: [Float], tokens: Int, heads: Int, headDim: Int) -> [Float] {
        precondition(x.count == tokens * heads * headDim)
        let half = headDim / 2
        let numFreqs = headDim / 2
        precondition(rope.count == tokens * numFreqs * 4)
        var out = x
        for t in 0..<tokens {
            for h in 0..<heads {
                let base = (t * heads + h) * headDim
                let rb = t * numFreqs * 4
                for p in 0..<half {
                    let a = out[base + p]
                    let b = out[base + half + p]
                    let c = rope[rb + p * 4 + 0]   // m00 = cos
                    let s = rope[rb + p * 4 + 2]   // m10 = sin
                    out[base + p]         = c * a - s * b
                    out[base + half + p]  = s * a + c * b
                }
            }
        }
        return out
    }

    /// Scaled dot-product attention for one head (attention.py:166-219).
    ///   q [Sq], k [Sk], v [Sk], each length = headDim (fp32). scale = headDim^-0.5.
    ///   score[i,j] = dot(q_i, k_j) * scale; stable softmax over j; out[i] = sum_j p_ij v_j.
    static func attentionHead(_ q: [Float], _ k: [Float], _ v: [Float], headDim: Int) -> [Float] {
        let sq = q.count / headDim
        let sk = k.count / headDim
        let scale = 1.0 / sqrt(Float(headDim))
        var out = [Float](repeating: 0, count: sq * headDim)
        var scores = [Float](repeating: 0, count: sq * sk)
        // score[i,j]
        for i in 0..<sq {
            let qi = i * headDim
            var rowMax: Float = -.greatestFiniteMagnitude
            for j in 0..<sk {
                let kj = j * headDim
                var dot: Float = 0
                for d in 0..<headDim { dot += q[qi + d] * k[kj + d] }
                let s = dot * scale
                scores[i * sk + j] = s
                if s > rowMax { rowMax = s }
            }
            // stable softmax
            var sum: Float = 0
            for j in 0..<sk { sum += exp(scores[i * sk + j] - rowMax) }
            // weighted sum of v
            for d in 0..<headDim {
                var acc: Float = 0
                for j in 0..<sk {
                    let p = exp(scores[i * sk + j] - rowMax) / sum
                    acc += p * v[j * headDim + d]
                }
                out[qi + d] = acc
            }
        }
        return out
    }

    /// Concatenate [tokens, heads, headDim] back to [tokens, heads*headDim] (head-major:
    /// head0 dims 0..127, head1 128..255, ...). This is the attention output layout before
    /// output_proj (predict2.py torch_attention_op returns normal concat head repr).
    static func concatHeads(_ headsFlat: [Float], tokens: Int, heads: Int, headDim: Int) -> [Float] {
        // headsFlat is row-major [token][head][dim]; output [token][head*headDim+dim]
        // i.e. just flatten each token's [heads, headDim] block.
        var out = [Float](repeating: 0, count: tokens * heads * headDim)
        for t in 0..<tokens {
            let srcBase = t * heads * headDim
            let dstBase = t * heads * headDim
            for i in 0..<(heads * headDim) { out[dstBase + i] = headsFlat[srcBase + i] }
        }
        return out
    }

    /// Exact GELU (torch nn.GELU, approximate='none') — shared via DiTWeights.gelu.
    static func gelu(_ x: Float) -> Float { DiTWeights.gelu(x) }

    /// Full block forward (fp32). x is the residual stream [1024, 2048] (token-major).
    /// Returns [1024, 2048].
    static func forward(
        x: [[Float]],
        emb: [Float],            // t_embedding_B_T_D [2048]
        adalnLora: [Float],      // adaln_lora_B_T_3D [6144]
        crossCtx: [[Float]],     // [512, 1024] adapter output
        rope: [Float],           // [1024*64*4] DitRoPE.generate flat
        w: Weights
    ) -> [[Float]] {
        let tokens = x.count
        precondition(x[0].count == Self.dim)
        var out = x

        // 1. AdaLN modulation per branch (SiLU first, D026). chunk → shift/scale/gate [2048]
        func modulate(_ w1: [[Float]], _ w2: [[Float]]) -> (shift: [Float], scale: [Float], gate: [Float]) {
            let silu = emb.map { $0 / (1 + exp(-$0)) }
            let h1 = DiTWeights.matmul([silu], w1, m: 1, k: Self.dim, n: 256)[0]
            var mod = DiTWeights.matmul([h1], w2, m: 1, k: 256, n: 3 * Self.dim)[0]
            for i in 0..<mod.count { mod[i] += adalnLora[i] }
            return (Array(mod[0..<Self.dim]), Array(mod[Self.dim..<(2 * Self.dim)]), Array(mod[(2 * Self.dim)..<(3 * Self.dim)]))
        }
        let selfM = modulate(w.modSelfW1, w.modSelfW2)
        let crossM = modulate(w.modCrossW1, w.modCrossW2)
        let mlpM = modulate(w.modMLPW1, w.modMLPW2)

        // 2. SELF-ATTN
        let flatX = out.flatMap { $0 }
        let selfNorm = layerNormModulated(flatX, scale: selfM.scale, shift: selfM.shift)  // [1024,2048]
        var q = DiTWeights.matmul(asRows(selfNorm, tokens, Self.dim), w.selfQ, m: tokens, k: Self.dim, n: Self.dim).flatMap { $0 }
        var k = DiTWeights.matmul(asRows(selfNorm, tokens, Self.dim), w.selfK, m: tokens, k: Self.dim, n: Self.dim).flatMap { $0 }
        var v = DiTWeights.matmul(asRows(selfNorm, tokens, Self.dim), w.selfV, m: tokens, k: Self.dim, n: Self.dim).flatMap { $0 }
        q = rmsNormHeads(q, weight: w.selfQNorm, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        k = rmsNormHeads(k, weight: w.selfKNorm, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        q = applySplitHalfRoPE(q, rope: rope, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        k = applySplitHalfRoPE(k, rope: rope, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        let attnFlat = mha(q, k, v, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        let attnConcat = concatHeads(attnFlat, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        var result = DiTWeights.matmul(asRows(attnConcat, tokens, Self.dim), w.selfO, m: tokens, k: Self.dim, n: Self.dim).flatMap { $0 }
        for i in 0..<out.count { for d in 0..<Self.dim { out[i][d] = out[i][d] + selfM.gate[d] * result[i * Self.dim + d] } }

        // 3. CROSS-ATTN
        let flatX1 = out.flatMap { $0 }
        let crossNorm = layerNormModulated(flatX1, scale: crossM.scale, shift: crossM.shift)
        let crossTokens = crossCtx.count
        q = DiTWeights.matmul(asRows(crossNorm, tokens, Self.dim), w.crossQ, m: tokens, k: Self.dim, n: Self.dim).flatMap { $0 }
        k = DiTWeights.matmul(crossCtx, w.crossK, m: crossTokens, k: Self.ctxDim, n: Self.dim).flatMap { $0 }
        v = DiTWeights.matmul(crossCtx, w.crossV, m: crossTokens, k: Self.ctxDim, n: Self.dim).flatMap { $0 }
        q = rmsNormHeads(q, weight: w.crossQNorm, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        k = rmsNormHeads(k, weight: w.crossKNorm, tokens: crossTokens, heads: Self.heads, headDim: Self.headDim)
        let crossAttnFlat = mha(q, k, v, tokens: tokens, kvTokens: crossTokens, heads: Self.heads, headDim: Self.headDim)
        let crossConcat = concatHeads(crossAttnFlat, tokens: tokens, heads: Self.heads, headDim: Self.headDim)
        result = DiTWeights.matmul(asRows(crossConcat, tokens, Self.dim), w.crossO, m: tokens, k: Self.dim, n: Self.dim).flatMap { $0 }
        for i in 0..<out.count { for d in 0..<Self.dim { out[i][d] = out[i][d] + crossM.gate[d] * result[i * Self.dim + d] } }

        // 4. MLP
        let flatX2 = out.flatMap { $0 }
        let mlpNorm = layerNormModulated(flatX2, scale: mlpM.scale, shift: mlpM.shift)
        var h = DiTWeights.matmul(asRows(mlpNorm, tokens, Self.dim), w.mlpW1, m: tokens, k: Self.dim, n: Self.mlpHidden).flatMap { $0 }
        for i in 0..<h.count { h[i] = gelu(h[i]) }
        let y = DiTWeights.matmul(asRows(h, tokens, Self.mlpHidden), w.mlpW2, m: tokens, k: Self.mlpHidden, n: Self.dim).flatMap { $0 }
        for i in 0..<out.count { for d in 0..<Self.dim { out[i][d] = out[i][d] + mlpM.gate[d] * y[i * Self.dim + d] } }

        return out
    }

    /// MHA over a flat [tokens*q*kv... ] q/k/v each [tokens(,kv), heads*headDim].
    /// q/k/v are flat [qLen*heads*headDim] / [kvLen*heads*headDim]; returns flat
    /// [qLen*heads*headDim] in [token][head][dim] order.
    private static func mha(_ q: [Float], _ k: [Float], _ v: [Float], tokens: Int, kvTokens: Int? = nil,
                            heads: Int, headDim: Int) -> [Float] {
        let kvLen = kvTokens ?? tokens
        var out = [Float](repeating: 0, count: tokens * heads * headDim)
        for h in 0..<heads {
            // gather per-head slices
            var qh = [Float](repeating: 0, count: tokens * headDim)
            var kh = [Float](repeating: 0, count: kvLen * headDim)
            var vh = [Float](repeating: 0, count: kvLen * headDim)
            for t in 0..<tokens { for d in 0..<headDim { qh[t * headDim + d] = q[(t * heads + h) * headDim + d] } }
            for j in 0..<kvLen { for d in 0..<headDim { kh[j * headDim + d] = k[(j * heads + h) * headDim + d]; vh[j * headDim + d] = v[(j * heads + h) * headDim + d] } }
            let oh = attentionHead(qh, kh, vh, headDim: headDim)
            for t in 0..<tokens { for d in 0..<headDim { out[(t * heads + h) * headDim + d] = oh[t * headDim + d] } }
        }
        return out
    }

    /// Interpret a flat [N*D] buffer as [[Float]] of N rows length D (no copy).
    private static func asRows(_ flat: [Float], _ n: Int, _ d: Int) -> [[Float]] {
        precondition(flat.count == n * d)
        var rows = [[Float]](repeating: [], count: n)
        for i in 0..<n { rows[i] = Array(flat[(i * d)..<((i + 1) * d)]) }
        return rows
    }
}
