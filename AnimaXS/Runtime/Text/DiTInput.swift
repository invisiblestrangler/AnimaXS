import Foundation

/// CPU-reference DiT input projection (H001) — transcribed VERBATIM from the pinned
/// ComfyUI source `comfy/ldm/cosmos/predict2.py` (commit cbbc9da):
///   - `PatchEmbed` (predict2.py:272-334) with `Rearrange` `"b c (t r) (h m) (w n) -> b t h w (c r m n)"`
///     and a Linear(in_channels·r·m·n → out_channels, bias=False).
///   - padding-mask concat in `MiniTrainDIT.prepare_embedded_sequence` (predict2.py:806-816):
///     `x = cat([x, zeros(B,1,T,H,W)])` when `concat_padding_mask=True`.
///
/// At 512×512 (first working resolution, runbook §29):
///   latent [16, 64, 64]  →  +1 padding-mask channel (zeros)  →  17 effective channels
///   patchify 2×2, T=1    →  32×32 = 1024 tokens × 68 (c·4 + m·2 + n)
///   x_embedder Linear(68→2048, no bias)  →  [1024, 2048] Float32 residual stream
///
/// The residual stream MUST be Float32 (activations exceed fp16's finite range; runbook §7).
///
/// Weight tensor:
///   model.diffusion_model.x_embedder.proj.1.weight   [2048, 68]  W4
/// (`x_embedder.proj` is `nn.Sequential(Rearrange, Linear)`, so index 1 = the Linear.)
struct DiTInput {

    private let pack: AnimapkFile

    static let latentChannels = 16
    static let inChannels = 17          // 16 latent + 1 padding mask
    static let patchSpatial = 2
    static let tokenDim = 68            // inChannels × 2 × 2 × 1
    static let hidden = 2048

    init?(pack: AnimapkFile) {
        self.pack = pack
        guard T("x_embedder.proj.1.weight") != nil else { return nil }
    }

    /// Lookup tensor by full pack name (prefix `model.diffusion_model.`).
    private func T(_ name: String) -> AnimapkTensor? {
        pack.tensor(named: "model.diffusion_model." + name)
    }

    /// Patchify a 17-channel latent [C=17, H=64, W=64] (16 latent + 1 padding-mask channel)
    /// into [1024, 68] patch tokens, then project to [1024, 2048] via x_embedder (fp32).
    ///
    /// - Parameters:
    ///   - latent: 16-channel latent, flat row-major [16, 64, 64].
    ///   - paddingMask: optional 1-channel [64, 64] padding mask (zeros for full-frame 512).
    func embed(latent: [Float], paddingMask: [Float]? = nil) -> [[Float]] {
        let H = 64, W = 64
        let patchRows = H / Self.patchSpatial       // 32
        let patchCols = W / Self.patchSpatial       // 32
        let tokens = patchRows * patchCols          // 1024

        // Build 17-channel input: channel 16 = padding mask (zeros when nil).
        // mask[i] at spatial (y,x).
        var inCh = [[Float]](repeating: [Float](repeating: 0, count: H * W), count: Self.inChannels)
        for c in 0..<Self.latentChannels {
            let base = c * H * W
            for i in 0..<(H * W) { inCh[c][i] = latent[base + i] }
        }
        if let pm = paddingMask {
            for i in 0..<(H * W) { inCh[Self.latentChannels][i] = pm[i] }
        }
        // (channel 16 stays zeros if paddingMask is nil — matches predict2.py:808)

        // Patchify: out token (h,w) feature index = c·4 + m·2 + n  (r=0).
        // Source: inCh[c][ (h·2+m)·W + (w·2+n) ]
        var tokens2d = [[Float]](repeating: [Float](repeating: 0, count: Self.tokenDim), count: tokens)
        for h in 0..<patchRows {
            for w in 0..<patchCols {
                let tok = h * patchCols + w
                for c in 0..<Self.inChannels {
                    let ch = inCh[c]
                    for m in 0..<Self.patchSpatial {
                        for n in 0..<Self.patchSpatial {
                            let y = h * Self.patchSpatial + m
                            let x = w * Self.patchSpatial + n
                            tokens2d[tok][c * 4 + m * 2 + n] = ch[y * W + x]
                        }
                    }
                }
            }
        }

        // x_embedder Linear(68→2048, no bias): out[token][d] = Σ_p w[d][p]·patch[token][p]
        let w = DiTWeights.dequantMatrix(T("x_embedder.proj.1.weight")!, pack: pack, rows: Self.hidden, cols: Self.tokenDim)
        return DiTWeights.matmul(tokens2d, w, m: tokens, k: Self.tokenDim, n: Self.hidden)
    }
}
