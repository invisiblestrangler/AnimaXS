import Foundation

/// Canonical model constants from PHASE0_2_HANDOFF/RUNTIME_CONSTANTS.json + model_info.json.
/// These are the source of truth for the numerical pipeline.
enum ModelConstants {
    // DiT
    static let ditBlocks = 28
    static let ditHidden = 2048
    static let ditHeads = 16
    static let ditHeadDim = 128
    static let ditMLP = 8192
    static let ditLatentChannels = 16
    static let ditInChannels = 17 // 16 latent + 1 padding mask
    static let ditPatchSpatial = 2
    static let ditTokensAt512 = 1024
    static let ditCrossSeq = 512
    static let ditCrossEmb = 1024
    static let adalnLoRADim = 256
    static let w4GroupSize = 64
    static let ditBlockStoredBytes = 38_993_920
    static let largestMatrixFP16Bytes = 33_554_432
    static let ditPaddingMaskChannel = 1

    // RoPE 3D (DiT self-attention)
    static let ropeHeadDim = 128
    static let ropeAxesH = 42
    static let ropeAxesW = 42
    static let ropeAxesT = 44
    static let ropeSpatialTheta: Float = 42_871.1
    static let ropeTemporalTheta: Float = 10_000.0

    // Timestep
    static let timestepDim = 2048
    static let timestepBase: Float = 10_000.0

    // Text encoder (Qwen3-0.6B)
    static let teLayers = 28
    static let teHidden = 1024
    static let teHeads = 16
    static let teKVHeads = 8
    static let teHeadDim = 128
    static let teMLP = 3072
    static let teVocab = 151_936
    static let teRopeTheta: Float = 1_000_000.0
    static let teRmsNormEps: Float = 1e-6
    static let teLayerStoredBytes = 16_777_216
    static let w8GroupSize = 64

    // LLM adapter
    static let adapterLayers = 6
    static let adapterEmbedVocab = 32_128
    static let adapterDim = 1024
    static let adapterHeads = 16
    static let adapterHeadDim = 64
    static let adapterMLP = 4096
    static let adapterRopeTheta: Float = 10_000.0
    static let adapterRmsNormEps: Float = 1e-6

    // Sampler
    static let sigma8Step: [Float] = [
        1.0, 0.95469, 0.90036, 0.834, 0.75112,
        0.64469, 0.50299, 0.30501, 0.0
    ]
    static let samplerSteps = 8
    static let cfg: Float = 1.0
    static let shift: Float = 3.0
    static let sigmaFn: (Float) -> Float = { 3 * $0 / (1 + 2 * $0) }

    // Wan21 latent normalization
    static let latentsMean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]
    static let latentsStd: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
    ]

    // VAE
    static let vaeSigmaData: Float = 0.5
    static let vaeLatentChannels = 16
    static let vaeGroupNormGroups = 32

    // Resolution
    static let latentSize = 64
    static let imageSize = 512
}
