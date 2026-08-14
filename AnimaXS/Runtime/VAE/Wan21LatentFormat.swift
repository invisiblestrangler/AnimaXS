import Foundation
import Metal

/// Anima/Wan21 model latent-format boundary (sampler space <-> VAE space).
///
/// Root cause (proven 2026-08-14, `scripts/animapk_cuda/wan21_boundary_proof.py`):
/// ComfyUI applies `latent_format.process_out` to the sampler return BEFORE the
/// workflow/VAE receives the latent (comfy/samplers.py `CFGGuider.inner_sample`:
/// `return self.inner_model.process_latent_out(samples.to(torch.float32))`).
/// Anima is registered with `latent_format = Wan21` (comfy/supported_models.py).
///
///   sampler_latent (raw DiT/Euler output) --Wan21.process_out--> vae_latent
///   vae_latent     (VAE decoder input)     --Wan21.process_in --> sampler_latent
///
/// The custom Metal runtime previously decoded RAW sampler-space latents as if
/// they were VAE-space, producing the visible 8px grid. Apply
/// `applyProcessOutInPlace` EXACTLY ONCE at the sampler->VAE boundary (see
/// GenerationEngine), never inside the VAE decoder itself.
///
/// Constants are the pinned project comfy-ref snapshot (MiniTrainDIT/comfy
/// cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44) — DO NOT replace with current
/// ComfyUI master values. They match `scripts/animapk_cuda/wan21_latent_format.py`.
///
/// Latent buffer layout: channel-major fp32 [16, 64, 64] (65536 floats =
/// DiffusionSampler.latentElements). Channel c occupies the contiguous block
/// [c*4096, (c+1)*4096).
enum Wan21LatentFormat {
    static let scaleFactor: Float = 1.0

    /// Wan21.latents_mean (pinned comfy-ref).
    static let latentsMean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]

    /// Wan21.latents_std (pinned comfy-ref).
    static let latentsStd: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]

    static let channelCount = 16
    static let elementsPerChannel = 64 * 64

    /// Wan21.process_out: sampler-space -> VAE-space, channel-major [C, H*W].
    ///
    /// vae[c,i] = sampler[c,i] * latentsStd[c] / scaleFactor + latentsMean[c]
    static func processOut(_ x: [Float]) -> [Float] {
        precondition(x.count == channelCount * elementsPerChannel,
                     "expected \(channelCount) x \(elementsPerChannel) channel-major latent, got \(x.count)")
        var out = [Float](repeating: 0, count: x.count)
        for c in 0..<channelCount {
            let std = latentsStd[c] / scaleFactor
            let base = c * elementsPerChannel
            for i in 0..<elementsPerChannel {
                out[base + i] = x[base + i] * std + latentsMean[c]
            }
        }
        return out
    }

    /// Wan21.process_in (exact inverse): VAE-space -> sampler-space.
    static func processIn(_ x: [Float]) -> [Float] {
        precondition(x.count == channelCount * elementsPerChannel)
        var out = [Float](repeating: 0, count: x.count)
        for c in 0..<channelCount {
            let invStd = scaleFactor / latentsStd[c]
            let base = c * elementsPerChannel
            for i in 0..<elementsPerChannel {
                out[base + i] = (x[base + i] - latentsMean[c]) * invStd
            }
        }
        return out
    }

    /// In-place Wan21.process_out on a shared fp32 channel-major latent buffer.
    ///
    /// Exactly the same math as `processOut`. Used at the sampler->VAE boundary
    /// in GenerationEngine so the diffusion output is converted to VAE decode
    /// space EXACTLY ONCE before it reaches VAEDecoder.
    static func applyProcessOutInPlace(_ buffer: MTLBuffer) {
        let count = buffer.length / MemoryLayout<Float>.stride
        precondition(count == channelCount * elementsPerChannel,
                     "latent buffer has \(count) floats, expected \(channelCount * elementsPerChannel)")
        guard let p = buffer.contents().assumingMemoryBound(to: Float.self) else { return }
        for c in 0..<channelCount {
            let std = latentsStd[c] / scaleFactor
            let base = c * elementsPerChannel
            for i in 0..<elementsPerChannel {
                p[base + i] = p[base + i] * std + latentsMean[c]
            }
        }
    }
}