import Foundation

/// CPU reference implementations of the Qwen3-0.6B numerical primitives.
/// These are used for (a) unit-testing the layer math against the golden `cond_context`
/// and (b) as the reference for the Metal/MPS kernels (runbook §27). All reductions use
/// Float32 accumulation; the residual stream is Float32.
enum QwenNumerics {

    /// RMSNorm over the last dimension. eps from model metadata (1e-6).
    static func rmsNorm(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count)
        var sum: Float = 0
        for v in x { sum += v * v }
        let inv = 1.0 / sqrt(sum / Float(x.count) + eps)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = x[i] * inv * weight[i] }
        return out
    }

    /// Row-wise RMSNorm over a [rows, cols] matrix (each row normalized independently).
    static func rmsNormRows(_ x: [[Float]], weight: [Float], eps: Float) -> [[Float]] {
        x.map { rmsNorm($0, weight: weight, eps: eps) }
    }

    /// RoPE (theta 1e6, head_dim 128), **half-split rotate** matching the reference
    /// `comfy/text_encoders/llama.py apply_rope` + `precompute_freqs_cis` (Qwen3_06B).
    /// Applies to each `headDim`-sized chunk using the chunk's position; `positions` has
    /// one entry per chunk (for multi-head tensors, repeat each sequence position per head).
    ///
    /// Formula (half_dim = headDim/2):
    ///   freqs[j] = pos / theta^(2j/headDim)        for j in 0..<half_dim
    ///   cos = cos(freqs), sin = sin(freqs)  (broadcast over both halves)
    ///   out[:half]  = x[:half]*cos - x[half:]*sin
    ///   out[half:]  = x[half:]*cos + x[:half]*sin
    static func ropeNeoX(_ x: [Float], positions: [Int], theta: Float = 1_000_000.0, headDim: Int = 128) -> [Float] {
        var out = x
        let chunks = x.count / headDim
        precondition(positions.count == chunks, "positions (\(positions.count)) must match chunks (\(chunks))")
        let half = headDim / 2
        for c in 0..<chunks {
            let pos = Float(positions[c])
            // precompute cos/sin for this chunk
            var cosArr = [Float](repeating: 0, count: half)
            var sinArr = [Float](repeating: 0, count: half)
            for j in 0..<half {
                let invFreq = 1.0 / pow(theta, Float(2 * j) / Float(headDim))
                let angle = pos * invFreq
                cosArr[j] = cosf(angle)
                sinArr[j] = sinf(angle)
            }
            let base = c * headDim
            for j in 0..<half {
                let a = x[base + j]           // first half
                let b = x[base + j + half]    // second half
                out[base + j] = a * cosArr[j] - b * sinArr[j]
                out[base + j + half] = b * cosArr[j] + a * sinArr[j]
            }
        }
        return out
    }

    /// Gemma3 per-head RMSNorm (used for Q/K before RoPE). Normalizes each head's
    /// head_dim vector, applies a per-head weight [head_dim], eps 1e-6.
    static func gemma3HeadNorm(_ x: [Float], weight: [Float], headDim: Int, eps: Float = 1e-6) -> [Float] {
        let heads = x.count / headDim
        precondition(weight.count == headDim)
        var out = x
        for h in 0..<heads {
            var sum: Float = 0
            for d in 0..<headDim { let v = x[h * headDim + d]; sum += v * v }
            let inv = 1.0 / sqrt(sum / Float(headDim) + eps)
            for d in 0..<headDim { out[h * headDim + d] = x[h * headDim + d] * inv * weight[d] }
        }
        return out
    }

    /// Gated SiLU: silu(gate) * up  → down. gate/up are [intermediate], output [intermediate].
    static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

    static func gatedSiLU(_ gate: [Float], _ up: [Float]) -> [Float] {
        precondition(gate.count == up.count)
        var out = [Float](repeating: 0, count: gate.count)
        for i in 0..<gate.count { out[i] = silu(gate[i]) * up[i] }
        return out
    }

    /// Scaled dot-product attention for one head: [qLen, headDim] × [kvLen, headDim]ᵀ
    /// with softmax over kvLen, then × V. Returns [qLen, headDim]. Float32 accum.
    static func scaledDotAttention(q: [Float], k: [Float], v: [Float],
                                   qLen: Int, kvLen: Int, headDim: Int,
                                   scale: Float, mask: [Float]? = nil) -> [Float] {
        var out = [Float](repeating: 0, count: qLen * headDim)
        for i in 0..<qLen {
            var scores = [Float](repeating: 0, count: kvLen)
            var maxScore: Float = -Float.greatestFiniteMagnitude
            for j in 0..<kvLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += q[i * headDim + d] * k[j * headDim + d]
                }
                scores[j] = dot * scale
                if let m = mask, m[j] <= 0 { scores[j] = -Float.greatestFiniteMagnitude }
                if scores[j] > maxScore { maxScore = scores[j] }
            }
            var sum: Float = 0
            for j in 0..<kvLen { scores[j] = exp(scores[j] - maxScore); sum += scores[j] }
            for j in 0..<kvLen { scores[j] /= sum }
            for d in 0..<headDim {
                var acc: Float = 0
                for j in 0..<kvLen { acc += scores[j] * v[j * headDim + d] }
                out[i * headDim + d] = acc
            }
        }
        return out
    }

    /// Build a causal attention mask [kvLen] for position i (1 = attend, 0 = masked).
    static func causalMask(position: Int, kvLen: Int) -> [Float] {
        (0..<kvLen).map { $0 <= position ? 1.0 : 0.0 }
    }
}
