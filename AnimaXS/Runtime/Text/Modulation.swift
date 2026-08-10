import Foundation

/// CPU-reference DiT AdaLN-LoRA modulation (H003) — transcribed VERBATIM from the pinned
/// ComfyUI source `comfy/ldm/cosmos/predict2.py` (commit cbbc9da), `Block.forward` (471-591).
///
/// For each of the three block branches (self_attn, cross_attn, mlp), the modulation is:
///   mod_branch = Linear2( SiLU( Linear1( emb_B_T_D ) ) )        # Linear1 2048→256, Linear2 256→6144
///   mod        = mod_branch + adaln_lora_B_T_3D                  # add the shared LoRA term
///   shift, scale, gate = mod.chunk(3, dim=-1)                    # each [2048], in order
/// (predict2.py:486-516; adaln_modulation_* = Sequential(SiLU, Linear, Linear) at 451-465).
///
/// Weights per block (W4): blocks.N.adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}.weight
///   .1.weight [256,2048], .2.weight [6144,256]   (index = position in the Sequential).
///
/// The modulation is applied (predict2.py:520-521, 542, 575, 590) with a **LayerNorm that
/// is elementwise_affine=False and mean-CENTERS** (NOT RMSNorm):
///   normalized = LayerNorm(x, eps 1e-6) * (1 + scale) + shift     # fp32
///   x = x + gate * branch(normalized)
///
/// The entire modulation path stays in Float32 (runbook §30).
struct Modulation {

    private let pack: AnimapkFile
    private let blockIndex: Int

    static let dim = 2048
    static let loraDim = 256
    static let branchOut = 3 * dim       // 6144
    static let eps: Float = 1e-6

    init?(pack: AnimapkFile, blockIndex: Int = 0) {
        self.pack = pack
        self.blockIndex = blockIndex
        guard T("blocks.\(blockIndex).adaln_modulation_self_attn.1.weight") != nil else { return nil }
    }

    /// Lookup tensor by full pack name (prefix `model.diffusion_model.`).
    private func T(_ name: String) -> AnimapkTensor? {
        pack.tensor(named: "model.diffusion_model." + name)
    }

    /// Compute one branch's (shift, scale, gate), each [dim], from emb_B_T_D [dim] +
    /// adaln_lora_B_T_3D [branchOut]. Pure math (predict2.py:486-516), fp32.
    func branch(branch: Branch, emb: [Float], adalnLora: [Float]) -> ModulationTriplet {
        let w1 = DiTWeights.dequantMatrix(T("blocks.\(blockIndex)." + branch.rawValue + ".1.weight")!, pack: pack, rows: Self.loraDim, cols: Self.dim)
        let w2 = DiTWeights.dequantMatrix(T("blocks.\(blockIndex)." + branch.rawValue + ".2.weight")!, pack: pack, rows: Self.branchOut, cols: Self.loraDim)

        // SiLU → Linear1(2048→256) → Linear2(256→6144)
        let silu = emb.map { $0 / (1 + exp(-$0)) }
        let h1 = DiTWeights.matmul([silu], w1, m: 1, k: Self.dim, n: Self.loraDim)[0]      // [256]
        let mod = DiTWeights.matmul([h1], w2, m: 1, k: Self.loraDim, n: Self.branchOut)[0]  // [6144]

        // mod = branch_out + adaln_lora_B_T_3D
        var modSum = [Float](repeating: 0, count: Self.branchOut)
        for i in 0..<Self.branchOut { modSum[i] = mod[i] + adalnLora[i] }

        // chunk(3, dim=-1) → shift [0:2048], scale [2048:4096], gate [4096:6144]
        return ModulationTriplet(
            shift: Array(modSum[0..<Self.dim]),
            scale: Array(modSum[Self.dim..<(2 * Self.dim)]),
            gate: Array(modSum[(2 * Self.dim)..<(3 * Self.dim)])
        )
    }

    /// Compute all three branches from one emb_B_T_D + adaln_lora_B_T_3D.
    func allBranches(emb: [Float], adalnLora: [Float]) -> BranchOutput {
        BranchOutput(
            selfAttn: branch(branch: .selfAttn, emb: emb, adalnLora: adalnLora),
            crossAttn: branch(branch: .crossAttn, emb: emb, adalnLora: adalnLora),
            mlp: branch(branch: .mlp, emb: emb, adalnLora: adalnLora)
        )
    }

    /// LayerNorm with elementwise_affine=False + AdaLN shift/scale application
    /// (predict2.py:520-521 `_fn`). `x` is [dim]; returns [dim] fp32.
    ///   normalized = (x - mean) / sqrt(var + eps) * (1 + scale) + shift
    static func applyLayerNormModulation(_ x: [Float], scale: [Float], shift: [Float]) -> [Float] {
        precondition(x.count == scale.count && x.count == shift.count)
        let n = x.count
        var mean: Float = 0
        for v in x { mean += v }
        mean /= Float(n)
        var varSum: Float = 0
        for v in x { let d = v - mean; varSum += d * d }
        let variance = varSum / Float(n)
        let inv = 1.0 / sqrt(variance + Self.eps)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = (x[i] - mean) * inv * (1 + scale[i]) + shift[i]
        }
        return out
    }

    enum Branch: String {
        case selfAttn = "adaln_modulation_self_attn"
        case crossAttn = "adaln_modulation_cross_attn"
        case mlp = "adaln_modulation_mlp"
    }

    struct ModulationTriplet {
        let shift: [Float]
        let scale: [Float]
        let gate: [Float]
    }

    struct BranchOutput {
        let selfAttn: ModulationTriplet
        let crossAttn: ModulationTriplet
        let mlp: ModulationTriplet
    }
}
