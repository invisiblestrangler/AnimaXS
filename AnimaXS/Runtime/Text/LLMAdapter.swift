import Foundation

/// CPU-reference LLMAdapter (lllite) — Anima-Turbo text→image conditioning adapter.
///
/// Transcribed VERBATIM from the pinned ComfyUI source `comfy/ldm/anima/model.py`
/// (commit cbbc9da): `LLMAdapter`, `TransformerBlock`, `Attention`, `RotaryEmbedding`,
/// `rotate_half`, `apply_rotary_pos_emb`, and `Anima.preprocess_text_embeds`.
///
/// Pipeline (runbook §28, G001):
///   T5 ids → W4 embed gather [T,1024] → 6 TransformerBlocks (self-attn + cross-attn
///   over Qwen context) → out_proj → RMSNorm → × t5 weights → zero-pad to 512 → [1,512,1024].
///
/// Config (RUNTIME_CONSTANTS adapter_lllite + MODEL_ARCHITECTURE adapter):
///   model_dim 1024, layers 6, heads 16, head_dim 64, mlp_size 4096, rope_theta 10000,
///   in_proj Identity, use_self_attn True, final RMSNorm eps 1e-6.
///   out_proj and MLP carry biases (default nn.Linear behavior in the pinned source).
///
/// IMPORTANT DIFFERENCES from Qwen encoder (do not reuse Qwen code blindly):
///   - RoPE is INTERLEAVED (HF rotate_half), NOT half-split (Qwen3 uses half-split).
///   - MLP activation is exact GELU (not SiLU), and MLP has biases.
///   - Attention is MHA 16/16 (no GQA), bidirectional (no causal mask).
///   - Self-attn context == query (target tokens); cross-attn context == Qwen hidden.
struct LLMAdapter {

    private let pack: AnimapkFile

    private static let modelDim = 1024
    private static let numHeads = 16
    private static let headDim = 64          // modelDim / numHeads
    private static let numLayers = 6
    private static let mlpSize = 4096        // modelDim * 4.0
    private static let vocab = 32128
    private static let eps: Float = 1e-6
    private static let ropeTheta: Float = 10000.0
    private static let maxCondLen = 512

    /// Lookup tensors by full pack name (prefix `model.diffusion_model.llm_adapter.`).
    private func T(_ name: String) -> AnimapkTensor? {
        pack.tensor(named: "model.diffusion_model.llm_adapter." + name)
    }

    init?(pack: AnimapkFile) {
        self.pack = pack
        // Validate all required tensors are present.
        let required = ["embed.weight", "out_proj.weight", "out_proj.bias", "norm.weight"]
        for name in required where T(name) == nil { return nil }
        for i in 0..<Self.numLayers {
            let p = "blocks.\(i)."
            let blockReq = ["norm_self_attn.weight", "norm_cross_attn.weight", "norm_mlp.weight",
                            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                            "cross_attn.q_proj.weight", "cross_attn.k_proj.weight",
                            "cross_attn.v_proj.weight", "cross_attn.o_proj.weight",
                            "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
                            "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"]
            for b in blockReq where T(p + b) == nil { return nil }
        }
    }

    // MARK: - Weight loading (W4 / fp16)

    private func dequantMatrix(_ t: AnimapkTensor, rows: Int, cols: Int) -> [[Float]] {
        if t.storage == .fp16 {
            let flat = QuantDecoders.fp16ToFloat32(pack.dataBytes(t).data, count: rows * cols)
            return (0..<rows).map { r in Array(flat[(r * cols)..<((r + 1) * cols)]) }
        }
        // W4 storage (group 64, even K → low nibble)
        let scale = pack.scaleBytes(t).map { $0.data } ?? Data()
        let zero = pack.zeroBytes(t).map { $0.data } ?? Data()
        let flat = QuantDecoders.dequantW4Matrix(data: pack.dataBytes(t).data, scale: scale, zero: zero,
                                                 rows: rows, cols: cols)
        return (0..<rows).map { r in Array(flat[(r * cols)..<((r + 1) * cols)]) }
    }

    private func dequantVector(_ t: AnimapkTensor, n: Int) -> [Float] {
        if t.storage == .fp16 {
            return QuantDecoders.fp16ToFloat32(pack.dataBytes(t).data, count: n)
        }
        let scale = pack.scaleBytes(t).map { $0.data } ?? Data()
        let zero = pack.zeroBytes(t).map { $0.data } ?? Data()
        return QuantDecoders.dequantW4(data: pack.dataBytes(t).data, scale: scale, zero: zero, k: n)
    }

    // MARK: - Matmul

    /// out[m,n] = a[m,k] × w[n,k]ᵀ  (weight stored rows=output, cols=input)
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

    /// Linear with bias: out[m,n] = a[m,k] × w[n,k]ᵀ + b[n]
    private func linear(_ a: [[Float]], _ w: [[Float]], _ b: [Float]?, m: Int, k: Int, n: Int) -> [[Float]] {
        var out = matmul(a, w, m: m, k: k, n: n)
        if let b {
            for i in 0..<m { for j in 0..<n { out[i][j] += b[j] } }
        }
        return out
    }

    // MARK: - Embedding gather (W4)

    /// Gather rows of the [vocab, 1024] W4 embedding tensor.
    private func gatherEmbedding(_ ids: [Int]) -> [[Float]] {
        let dim = Self.modelDim
        guard let emb = T("embed.weight") else { return [] }
        var out = [[Float]](repeating: [Float](repeating: 0, count: dim), count: ids.count)
        // For a W4 tensor of [vocab, dim], the packed data is stored as [vocab*dim/2] bytes
        // (2 values per byte, even K → low nibble), and scale/zero are per (vocab, dim/64)
        // groups. We dequantize each row slice in place using the Data-based decoder.
        // Dequantize row-by-row: data offset = row*(dim/2) bytes, scale/zero offset =
        // row*(dim/64)*2 bytes (fp16 pairs) for W4.
        let dataAll = pack.dataBytes(emb).data
        let scaleAll = pack.scaleBytes(emb).map { $0.data } ?? Data()
        let zeroAll = pack.zeroBytes(emb).map { $0.data } ?? Data()
        let half = dim / 2
        let groupsPerRow = dim / 64
        for (r, row) in ids.enumerated() {
            let dStart = row * half
            let sStart = row * groupsPerRow * 2   // fp16 = 2 bytes
            let zStart = row * groupsPerRow * 2
            let rowData = dataAll.subdata(in: dStart..<(dStart + half))
            let rowScale = scaleAll.subdata(in: sStart..<(sStart + groupsPerRow * 2))
            let rowZero = zeroAll.subdata(in: zStart..<(zStart + groupsPerRow * 2))
            out[r] = QuantDecoders.dequantW4(data: rowData, scale: rowScale, zero: rowZero, k: dim)
        }
        return out
    }

    // MARK: - RoPE (INTERLEAVED, rotate_half — pinned model.py:7-39)

    /// Interleaved rotary position embedding applied to a per-head flat array.
    /// x is [heads*headDim] (one token); positions[c] gives the position for head-chunk c.
    /// Formula (model.py rotate_half + apply_rotary_pos_emb):
    ///   rotate_half(x) = cat((-x[half:], x[:half]))
    ///   out = x*cos + rotate_half(x)*sin
    ///   inv_freq[j] = 1/theta^(2j/headDim); freqs = pos*inv_freq; cos/sin over cat(freqs,freqs)
    static func ropeInterleaved(_ x: [Float], positions: [Int], theta: Float = LLMAdapter.ropeTheta, headDim: Int = LLMAdapter.headDim) -> [Float] {
        var out = x
        let chunks = x.count / headDim
        precondition(positions.count == chunks)
        let half = headDim / 2
        var invFreq = [Float](repeating: 0, count: half)
        for j in 0..<half { invFreq[j] = 1.0 / pow(theta, Float(2 * j) / Float(headDim)) }
        for c in 0..<chunks {
            let pos = Float(positions[c])
            // cos/sin over cat((freqs),(freqs)) → headDim entries
            var cosArr = [Float](repeating: 0, count: headDim)
            var sinArr = [Float](repeating: 0, count: headDim)
            for j in 0..<half {
                let angle = pos * invFreq[j]
                cosArr[j] = cosf(angle); cosArr[j + half] = cosf(angle)
                sinArr[j] = sinf(angle); sinArr[j + half] = sinf(angle)
            }
            let base = c * headDim
            // rotate_half(x)[d] = (d < half) ? -x[d+half] : x[d-half]
            // out = x*cos + rotate_half(x)*sin  (pinned model.py:13-17)
            for d in 0..<headDim {
                let rotVal: Float = (d < half) ? -x[base + d + half] : x[base + d - half]
                out[base + d] = x[base + d] * cosArr[d] + rotVal * sinArr[d]
            }
        }
        return out
    }

    // MARK: - Attention (MHA, bidirectional)

    /// Multi-head attention over q [qLen, modelDim] and ctx [kvLen, modelDim].
    /// qProj/kProj/vProj [modelDim, modelDim]; qNorm/kNorm [headDim]. head_dim 64.
    /// position for q head-chunks: qPos; for kv: kvPos (each length = seqLen*heads).
    private func mha(_ qIn: [[Float]], _ ctx: [[Float]],
                     qProj: [[Float]], kProj: [[Float]], vProj: [[Float]], oProj: [[Float]],
                     qNormW: [Float], kNormW: [Float],
                     qPos: [Int], kvPos: [Int]) -> [[Float]] {
        let qLen = qIn.count, kvLen = ctx.count, hd = Self.headDim, H = Self.numHeads
        let q = matmul(qIn, qProj, m: qLen, k: Self.modelDim, n: H * hd)      // [qLen, 1024]
        let k = matmul(ctx, kProj, m: kvLen, k: Self.modelDim, n: H * hd)
        let v = matmul(ctx, vProj, m: kvLen, k: Self.modelDim, n: H * hd)
        // per-head Q/K norm
        let qn = q.flatMap { QwenNumerics.gemma3HeadNorm($0, weight: qNormW, headDim: hd) }
        let kn = k.flatMap { QwenNumerics.gemma3HeadNorm($0, weight: kNormW, headDim: hd) }
        // RoPE (interleaved) on Q and K
        let qR = Self.ropeInterleaved(qn, positions: qPos, headDim: hd)
        let kR = Self.ropeInterleaved(kn, positions: kvPos, headDim: hd)
        // attention per head, bidirectional (no mask)
        let qHidden = H * hd, kvHidden = H * hd
        var out = [[Float]](repeating: [Float](repeating: 0, count: qHidden), count: qLen)
        let scale: Float = 1.0 / sqrt(Float(hd))
        for i in 0..<qLen {
            for h in 0..<H {
                var qHead = [Float](repeating: 0, count: hd)
                let qb = i * qHidden + h * hd
                for d in 0..<hd { qHead[d] = qR[qb + d] }
                var kAll = [Float](repeating: 0, count: kvLen * hd)
                var vAll = [Float](repeating: 0, count: kvLen * hd)
                for s in 0..<kvLen {
                    for d in 0..<hd {
                        kAll[s * hd + d] = kR[s * kvHidden + h * hd + d]
                        vAll[s * hd + d] = v[s][h * hd + d]
                    }
                }
                let headOut = QwenNumerics.scaledDotAttention(
                    q: qHead, k: kAll, v: vAll, qLen: 1, kvLen: kvLen, headDim: hd, scale: scale, mask: nil)
                for d in 0..<hd { out[i][h * hd + d] = headOut[d] }
            }
        }
        // o_proj
        return matmul(out, oProj, m: qLen, k: H * hd, n: Self.modelDim)
    }

    // MARK: - Full forward

    /// Run the LLMAdapter. Inputs:
    ///   - context: Qwen last hidden [seq, 1024] (the source_hidden_states)
    ///   - t5IDs: T5 token IDs (target_input_ids)
    ///   - t5Weights: t5xxl_weights per target token
    /// Returns the padded conditioning [512, 1024] (already × weights).
    func encode(context: [[Float]], t5IDs: [Int], t5Weights: [Float]) -> [[Float]] {
        let tCount = t5IDs.count, C = context.count, hd = Self.headDim, H = Self.numHeads

        // x = in_proj(embed(target)) — in_proj is Identity (model_dim == target_dim)
        var x = gatherEmbedding(t5IDs)   // [tCount, 1024]

        // position_ids for target (T) and context (C)
        let targetPos = (0..<tCount).flatMap { p in (0..<H).map { _ in p } }
        let contextPos = (0..<C).flatMap { p in (0..<H).map { _ in p } }

        for l in 0..<Self.numLayers {
            let p = "blocks.\(l)."
            // self-attn
            let nsW = dequantVector(T(p + "norm_self_attn.weight")!, n: Self.modelDim)
            let normed = QwenNumerics.rmsNormRows(x, weight: nsW, eps: Self.eps)
            let sa = mha(normed, normed,
                         qProj: dequantMatrix(T(p + "self_attn.q_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         kProj: dequantMatrix(T(p + "self_attn.k_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         vProj: dequantMatrix(T(p + "self_attn.v_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         oProj: dequantMatrix(T(p + "self_attn.o_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         qNormW: dequantVector(T(p + "self_attn.q_norm.weight")!, n: hd),
                         kNormW: dequantVector(T(p + "self_attn.k_norm.weight")!, n: hd),
                         qPos: targetPos, kvPos: targetPos)
            for i in 0..<tCount { for d in 0..<Self.modelDim { x[i][d] += sa[i][d] } }

            // cross-attn (context = Qwen hidden)
            let ncW = dequantVector(T(p + "norm_cross_attn.weight")!, n: Self.modelDim)
            let normed2 = QwenNumerics.rmsNormRows(x, weight: ncW, eps: Self.eps)
            let ca = mha(normed2, context,
                         qProj: dequantMatrix(T(p + "cross_attn.q_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         kProj: dequantMatrix(T(p + "cross_attn.k_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         vProj: dequantMatrix(T(p + "cross_attn.v_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         oProj: dequantMatrix(T(p + "cross_attn.o_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim),
                         qNormW: dequantVector(T(p + "cross_attn.q_norm.weight")!, n: hd),
                         kNormW: dequantVector(T(p + "cross_attn.k_norm.weight")!, n: hd),
                         qPos: targetPos, kvPos: contextPos)
            for i in 0..<tCount { for d in 0..<Self.modelDim { x[i][d] += ca[i][d] } }

            // MLP: RMSNorm → Linear(1024→4096)+bias → GELU → Linear(4096→1024)+bias
            let nmW = dequantVector(T(p + "norm_mlp.weight")!, n: Self.modelDim)
            let normed3 = QwenNumerics.rmsNormRows(x, weight: nmW, eps: Self.eps)
            let w0 = dequantMatrix(T(p + "mlp.0.weight")!, rows: Self.mlpSize, cols: Self.modelDim)
            let b0 = dequantVector(T(p + "mlp.0.bias")!, n: Self.mlpSize)
            let w2 = dequantMatrix(T(p + "mlp.2.weight")!, rows: Self.modelDim, cols: Self.mlpSize)
            let b2 = dequantVector(T(p + "mlp.2.bias")!, n: Self.modelDim)
            let h = linear(normed3, w0, b0, m: tCount, k: Self.modelDim, n: Self.mlpSize)
            let g = h.map { row in row.map(Self.gelu) }
            let down = linear(g, w2, b2, m: tCount, k: Self.mlpSize, n: Self.modelDim)
            for i in 0..<tCount { for d in 0..<Self.modelDim { x[i][d] += down[i][d] } }
        }

        // out_proj + final RMSNorm
        let ow = dequantMatrix(T("out_proj.weight")!, rows: Self.modelDim, cols: Self.modelDim)
        let ob = dequantVector(T("out_proj.bias")!, n: Self.modelDim)
        var out = linear(x, ow, ob, m: tCount, k: Self.modelDim, n: Self.modelDim)
        let nw = dequantVector(T("norm.weight")!, n: Self.modelDim)
        out = QwenNumerics.rmsNormRows(out, weight: nw, eps: Self.eps)

        // × t5 weights
        for i in 0..<tCount { for d in 0..<Self.modelDim { out[i][d] *= t5Weights[i] } }

        // zero-pad to 512
        if tCount < Self.maxCondLen {
            out += [[Float]](repeating: [Float](repeating: 0, count: Self.modelDim), count: Self.maxCondLen - tCount)
        }
        return out   // [512, 1024]
    }

    /// Exact GELU (torch.nn.GELU default, approximate='none').
    static func gelu(_ x: Float) -> Float {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5 * x * (1 + erf(x / sqrt(2.0)))
    }
}
