import Foundation

/// CPU-reference DiT timestep embedder (H002) — transcribed VERBATIM from the pinned
/// ComfyUI source `comfy/ldm/cosmos/predict2.py` (commit cbbc9da):
///   - `Timesteps.forward`                     (predict2.py:219-238)  → sinusoidal, dim 2048
///   - `TimestepEmbedding.forward`             (predict2.py:241-269)  → Linear1→SiLU→Linear2
///   - `MiniTrainDIT._forward` t_embedding_norm (predict2.py:737, 881-882) → RMSNorm on the
///     raw sinusoidal, producing `t_embedding_B_T_D` for the blocks.
///
/// The DiT timestep input is **sigma** (not an arbitrary integer step). Runbook §30.
///
/// Config (RUNTIME_CONSTANTS dit + MODEL_ARCHITECTURE dit):
///   timestep_dim 2048, base 10000, half_dim 1024, use_adaln_lora True →
///   linear_1(2048→2048, no bias), SiLU, linear_2(2048→6144, no bias).
///
/// Outputs (both Float32):
///   - `embedding` (2048):  t_embedding_B_T_D = RMSNorm(raw sinusoidal)  [fed to blocks]
///   - `adalnLora` (6144):  adaln_lora_B_T_3D = Linear2(SiLU(Linear1(raw sinusoidal)))
///
/// Weight tensors (DiT pack):
///   model.diffusion_model.t_embedder.1.linear_1.weight   [2048, 2048]  W4
///   model.diffusion_model.t_embedder.1.linear_2.weight   [6144, 2048]  W4
///   model.diffusion_model.t_embedding_norm.weight        [2048]        fp16
/// (`t_embedder` is a `nn.Sequential(Timesteps, TimestepEmbedding)`, so index 0 = Timesteps,
/// index 1 = TimestepEmbedding.)
struct TimestepEmbedder {

    private let pack: AnimapkFile

    static let dim = 2048
    static let halfDim = 1024
    static let base: Float = 10_000.0
    static let adalnDim = 3 * dim  // 6144
    static let eps: Float = 1e-6

    init?(pack: AnimapkFile) {
        self.pack = pack
        let req = ["t_embedder.1.linear_1.weight",
                   "t_embedder.1.linear_2.weight",
                   "t_embedding_norm.weight"]
        for n in req where T(n) == nil { return nil }
    }

    /// Lookup tensor by full pack name (prefix `model.diffusion_model.`).
    private func T(_ name: String) -> AnimapkTensor? {
        pack.tensor(named: "model.diffusion_model." + name)
    }

    /// Sinusoidal timestep embedding (predict2.py:219-238), Float32.
    ///   exponent = −log(10000)·arange(halfDim)/halfDim;  emb = sigma·exp(exponent);
    ///   out = cat([cos(emb), sin(emb)], dim=−1)  → [1, 1, 2048]  (cos in [0:1024], sin in [1024:2048]).
    static func sinusoidal(sigma: Float) -> [Float] {
        var out = [Float](repeating: 0, count: Self.dim)
        let logBase = log(Self.base)
        for i in 0..<Self.halfDim {
            let exponent = -logBase * Float(i) / Float(Self.halfDim)
            let angle = sigma * exp(exponent)
            out[i] = cosf(angle)
            out[Self.halfDim + i] = sinf(angle)
        }
        return out
    }

    /// Run the full timestep path for a given sigma. Returns:
    ///   (embedding: t_embedding_B_T_D [2048], adalnLora: adaln_lora_B_T_3D [6144]).
    func embed(sigma: Float) -> (embedding: [Float], adalnLora: [Float]) {
        // 1. raw sinusoidal (Timesteps) → [2048]
        let raw = Self.sinusoidal(sigma: sigma)

        // 2. TimestepEmbedding (predict2.py:257-269):
        //    emb = linear_1(raw); emb = SiLU(emb); emb = linear_2(emb)
        //    use_adaln_lora=True → adaln_lora_B_T_3D = emb; emb_B_T_D = raw (sample)
        let w1 = DiTWeights.dequantMatrix(T("t_embedder.1.linear_1.weight")!, pack: pack, rows: Self.dim, cols: Self.dim)
        let w2 = DiTWeights.dequantMatrix(T("t_embedder.1.linear_2.weight")!, pack: pack, rows: Self.adalnDim, cols: Self.dim)
        let h1 = DiTWeights.matmul([raw], w1, m: 1, k: Self.dim, n: Self.dim)[0]   // linear_1
        let silu = h1.map { $0 / (1 + exp(-$0)) }                                  // SiLU
        let adalnLora = DiTWeights.matmul([silu], w2, m: 1, k: Self.dim, n: Self.adalnDim)[0]  // linear_2 → [6144]

        // 3. t_embedding_norm (predict2.py:737, 881-882): RMSNorm(raw sinusoidal)
        let normW = DiTWeights.dequantVector(T("t_embedding_norm.weight")!, pack: pack, n: Self.dim)
        let embedding = DiTWeights.rmsNorm(raw, weight: normW, eps: Self.eps)

        return (embedding, adalnLora)
    }
}
