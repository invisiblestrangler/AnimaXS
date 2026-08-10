import Foundation

/// CPU-reference Qwen3-0.6B text encoder. Reads the W8 TE pack via AnimapkFile, gathers
/// embedding rows on demand, and runs the 28 layers with pure-CPU linear algebra.
///
/// This is the REFERENCE for the Metal/MPS implementation (runbook §27) and is used to
/// validate the encoder against the golden `cond_context` on any platform (including CI
/// via the bundled fixtures and the real-pack path). Produces the last-layer hidden state
/// WITHOUT the final norm (layer_norm_hidden_state=False).
struct QwenEncoderCPU {

    let pack: AnimapkFile
    let embedding: AnimapkTensor
    let layerTensors: [[String: AnimapkTensor]]  // 28 layers, each 11 tensors by suffix
    let finalNorm: AnimapkTensor?               // model.norm.weight (final RMSNorm)

    private static let layerCount = 28
    private static let hidden = 1024
    private static let heads = 16
    private static let kvHeads = 8
    private static let headDim = 128
    private static let intermediate = 3072
    private static let eps: Float = 1e-6

    init?(pack: AnimapkFile) {
        self.pack = pack
        guard let emb = pack.tensor(named: "model.embed_tokens.weight") else { return nil }
        self.embedding = emb
        self.finalNorm = pack.tensor(named: "model.norm.weight")

        var layers: [[String: AnimapkTensor]] = []
        for l in 0..<Self.layerCount {
            let prefix = "model.layers.\(l)."
            var dict: [String: AnimapkTensor] = [:]
            let suffixes = ["input_layernorm.weight", "post_attention_layernorm.weight",
                            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            for sfx in suffixes {
                guard let t = pack.tensor(named: prefix + sfx) else { return nil }
                dict[sfx] = t
            }
            layers.append(dict)
        }
        self.layerTensors = layers
    }

    // MARK: - Weight loading

    /// Dequantize a tensor to a [rows, cols] Float32 matrix. Handles both W8 and fp16 storage.
    private func dequantMatrix(_ t: AnimapkTensor, rows: Int, cols: Int) -> [[Float]] {
        let flat: [Float]
        if t.storage == .fp16 {
            flat = QuantDecoders.fp16ToFloat32(pack.dataBytes(t).data, count: rows * cols)
        } else {
            let scale = pack.scaleBytes(t).map { $0.data } ?? Data()
            let zero = pack.zeroBytes(t).map { $0.data } ?? Data()
            flat = QuantDecoders.dequantW8(data: pack.dataBytes(t).data, scale: scale, zero: zero, k: rows * cols)
        }
        return (0..<rows).map { r in Array(flat[(r * cols)..<((r + 1) * cols)]) }
    }

    private func dequantVector(_ t: AnimapkTensor, n: Int) -> [Float] {
        if t.storage == .fp16 {
            return QuantDecoders.fp16ToFloat32(pack.dataBytes(t).data, count: n)
        }
        let scale = pack.scaleBytes(t).map { $0.data } ?? Data()
        let zero = pack.zeroBytes(t).map { $0.data } ?? Data()
        return QuantDecoders.dequantW8(data: pack.dataBytes(t).data, scale: scale, zero: zero, k: n)
    }

    // MARK: - Matmul

    /// out[m,n] = a[m,k] × w[n,k]ᵀ   (weight stored rows=output, cols=input — pack layout)
    private func matmul(_ a: [[Float]], _ w: [[Float]], m: Int, k: Int, n: Int) -> [[Float]] {
        var out = [[Float]](repeating: [Float](repeating: 0, count: n), count: m)
        for i in 0..<m {
            for j in 0..<n {
                var acc: Float = 0
                let wRow = w[j]
                let aRow = a[i]
                for p in 0..<k { acc += aRow[p] * wRow[p] }
                out[i][j] = acc
            }
        }
        return out
    }

    // MARK: - Forward

    /// Encode a token sequence → [seq, hidden] cond_context.
    /// Pinned ComfyUI: cond_context = BaseLlama.forward last-layer output AFTER the final
    /// RMSNorm. Qwen3_06BConfig.final_norm=True (llama.py:130) AND anima.py Qwen3_06BModel
    /// layer_norm_hidden_state=False only suppresses final norm on the *intermediate* path;
    /// the main output z (layer=="last") is post-final-norm. Verified: golden cond_context
    /// matches oracle POST-final-norm (cosine 0.992) vs pre-final-norm (0.62).
    func encode(tokenIDs: [Int]) -> [[Float]] {
        var x = gatherEmbedding(rows: tokenIDs)   // [seq, 1024]
        for l in 0..<Self.layerCount {
            x = runLayer(x, layerIndex: l)
        }
        if let norm = finalNorm {
            let w = dequantVector(norm, n: Self.hidden)
            x = QwenNumerics.rmsNormRows(x, weight: w, eps: Self.eps)
        }
        return x
    }

    /// Gather [seq, 1024] from the W8 embedding table (rows on demand).
    private func gatherEmbedding(rows: [Int]) -> [[Float]] {
        let dim = 1024
        let data = pack.dataBytes(embedding).data
        let scale = pack.scaleBytes(embedding).map { $0.data } ?? Data()
        let zero = pack.zeroBytes(embedding).map { $0.data } ?? Data()
        var out = [[Float]](repeating: [Float](repeating: 0, count: dim), count: rows.count)
        for (r, row) in rows.enumerated() {
            // slice row r of [151936, 1024]: data = row*dim uint8 bytes; scale/zero = 16 fp16 per row = 32 bytes
            let rowData = data.subdata(in: (row * dim)..<((row + 1) * dim))
            let rowScale = scale.subdata(in: (row * 32)..<((row + 1) * 32))
            let rowZero = zero.subdata(in: (row * 32)..<((row + 1) * 32))
            out[r] = QuantDecoders.dequantW8(data: rowData, scale: rowScale, zero: rowZero, k: dim)
        }
        return out
    }

    /// One Qwen layer: RMSNorm → self-attn → residual → RMSNorm → gated SiLU MLP → residual.
    private func runLayer(_ x: [[Float]], layerIndex l: Int) -> [[Float]] {
        let t = layerTensors[l]
        let hidden = Self.hidden
        let seq = x.count

        // input_layernorm (RMSNorm)
        let inNormW = dequantVector(t["input_layernorm.weight"]!, n: hidden)
        let h1 = QwenNumerics.rmsNormRows(x, weight: inNormW, eps: Self.eps)

        // self-attention (GQA: 16 Q heads ×128, 8 KV heads ×128; q_proj out=2048, k/v out=1024)
        let qW = dequantMatrix(t["self_attn.q_proj.weight"]!, rows: 16 * 128, cols: hidden)
        let kW = dequantMatrix(t["self_attn.k_proj.weight"]!, rows: 8 * 128, cols: hidden)
        let vW = dequantMatrix(t["self_attn.v_proj.weight"]!, rows: 8 * 128, cols: hidden)
        let oW = dequantMatrix(t["self_attn.o_proj.weight"]!, rows: hidden, cols: 16 * 128)
        let q = matmul(h1, qW, m: seq, k: hidden, n: 16 * 128)    // [seq, 2048] = 16 heads × 128
        let kk = matmul(h1, kW, m: seq, k: hidden, n: 8 * 128)    // [seq, 1024] = 8 KV heads × 128
        let v = matmul(h1, vW, m: seq, k: hidden, n: 8 * 128)     // [seq, 1024]

        // gemma3 per-head Q/K norm (each head's 128-dim vector)
        let qNormW = dequantVector(t["self_attn.q_norm.weight"]!, n: Self.headDim)
        let kNormW = dequantVector(t["self_attn.k_norm.weight"]!, n: Self.headDim)
        let qFlat = q.flatMap { QwenNumerics.gemma3HeadNorm($0, weight: qNormW, headDim: Self.headDim) }
        let kFlat = kk.flatMap { QwenNumerics.gemma3HeadNorm($0, weight: kNormW, headDim: Self.headDim) }

        // RoPE (theta 1e6) on Q (16 heads) and K (8 heads), GPT-NeoX pairs.
        // Each head-chunk at sequence position s uses position s → repeat positions per head.
        let positions = (0..<seq).map { $0 }
        let qPositions = positions.flatMap { p in (0..<16).map { _ in p } }      // seq × 16
        let kPositions = positions.flatMap { p in (0..<8).map { _ in p } }       // seq × 8
        let qRope = QwenNumerics.ropeNeoX(qFlat, positions: qPositions, theta: 1_000_000, headDim: Self.headDim)
        let kRope = QwenNumerics.ropeNeoX(kFlat, positions: kPositions, theta: 1_000_000, headDim: Self.headDim)

        // GQA attention: Q head h reads KV head (h / repeatFactor), repeatFactor = heads/kvHeads = 2.
        // Pinned ComfyUI comfy/ops.repeat_kv_for_gqa uses repeat_interleave(n_rep) → contiguous
        // grouped KV heads [K0,K0,K1,K1,...]. See docs/QWEN_ENCODER_DEBUG.md + GqaHeadMappingTests.
        // Do NOT use h % kvHeads (round-robin) — that is the wrong grouping.
        var attnOut = [[Float]](repeating: [Float](repeating: 0, count: 16 * 128), count: seq)
        let qHidden = 16 * 128
        let kvHidden = 8 * 128
        let repeatFactor = Self.heads / Self.kvHeads
        precondition(Self.heads % Self.kvHeads == 0, "query heads must be divisible by kv heads")
        for i in 0..<seq {
            for h in 0..<Self.heads {
                let kvHead = h / repeatFactor  // GQA (grouped); NOT h % kvHeads
                // q head h of token i
                var qHead = [Float](repeating: 0, count: Self.headDim)
                let qBase = i * qHidden + h * Self.headDim
                for d in 0..<Self.headDim { qHead[d] = qRope[qBase + d] }
                // k/v for KV head across all seq positions
                var kAll = [Float](repeating: 0, count: seq * Self.headDim)
                var vAll = [Float](repeating: 0, count: seq * Self.headDim)
                let kvBase = kvHead * Self.headDim
                for s in 0..<seq {
                    let kSrc = s * kvHidden + kvBase   // kRope is flat [seq, kvHidden]
                    let dst = s * Self.headDim
                    for d in 0..<Self.headDim {
                        kAll[dst + d] = kRope[kSrc + d]
                        vAll[dst + d] = v[s][kvBase + d]  // v is [seq, kvHidden], row s
                    }
                }
                let mask = QwenNumerics.causalMask(position: i, kvLen: seq)
                let scale: Float = 1.0 / sqrt(Float(Self.headDim))
                let headOut = QwenNumerics.scaledDotAttention(q: qHead, k: kAll, v: vAll,
                                                              qLen: 1, kvLen: seq, headDim: Self.headDim,
                                                              scale: scale, mask: mask)
                for d in 0..<Self.headDim { attnOut[i][h * Self.headDim + d] = headOut[d] }
            }
        }
        // o_proj: [seq, 2048] → [seq, 1024]
        let attnProj = matmul(attnOut, oW, m: seq, k: 16 * 128, n: hidden)

        // residual 1
        var x2 = [[Float]](repeating: [Float](repeating: 0, count: hidden), count: seq)
        for i in 0..<seq { for d in 0..<hidden { x2[i][d] = x[i][d] + attnProj[i][d] } }

        // post_attention_layernorm + gated SiLU MLP
        let postW = dequantVector(t["post_attention_layernorm.weight"]!, n: hidden)
        let h2 = QwenNumerics.rmsNormRows(x2, weight: postW, eps: Self.eps)
        let gateW = dequantMatrix(t["mlp.gate_proj.weight"]!, rows: Self.intermediate, cols: hidden)
        let upW = dequantMatrix(t["mlp.up_proj.weight"]!, rows: Self.intermediate, cols: hidden)
        let downW = dequantMatrix(t["mlp.down_proj.weight"]!, rows: hidden, cols: Self.intermediate)
        let gate = matmul(h2, gateW, m: seq, k: hidden, n: Self.intermediate)
        let up = matmul(h2, upW, m: seq, k: hidden, n: Self.intermediate)
        var gated = [[Float]](repeating: [Float](repeating: 0, count: Self.intermediate), count: seq)
        for i in 0..<seq { gated[i] = QwenNumerics.gatedSiLU(gate[i], up[i]) }
        let down = matmul(gated, downW, m: seq, k: Self.intermediate, n: hidden)

        // residual 2
        var xOut = [[Float]](repeating: [Float](repeating: 0, count: hidden), count: seq)
        for i in 0..<seq { for d in 0..<hidden { xOut[i][d] = x2[i][d] + down[i][d] } }
        return xOut
    }
}
