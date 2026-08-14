import XCTest
import Metal
@testable import AnimaXS

/// Regression tests for the Anima/Wan21 sampler<->VAE latent-format boundary
/// (Wan21LatentFormat.swift). Mirrors scripts/test_wan21_latent_format.py and
/// the pinned comfy-ref semantics (MiniTrainDIT/comfy cbbc9dab…).
///
/// Catches: wrong channel order, missing mean, inverse-by-mistake, duplicated
/// transform, and dtype/broadcast errors — the failure modes of the 8px grid
/// root cause (decoding raw sampler-space latents as VAE-space).
final class Wan21LatentFormatTests: XCTestCase {

    /// D3 known-vector fixture: golden crop with provenance.
    /// From the canonical fixture case1_danbooru_seed1337.npz (SHA
    /// 44d35d4f…a8dc): channel 0, position (0,0) of step_latents[7] (raw
    /// SAMPLER-space callback denoised) and final_latent (ComfyUI's
    /// POST-process_out VAE-space latent). Verified bit-exact on CUDA:
    /// process_out(step_latents[7]) == final_latent (wan21_boundary_proof A2).
    private let rawStep7Channel0: [Float] = [
        0.5178982, 0.34929597, 0.99847376, 0.92103589, -0.3197512, -0.57249177,
        -0.45125312, -0.62969398, -0.66076744, 0.1608859, 0.38026184,
        0.76742512, 0.27528617, -0.82789308, 0.63779151, 1.3282219,
    ]
    private let finalChannel0: [Float] = [
        0.70254421, -0.20098871, 1.412648, 2.5535872, -0.56446856, -0.048468411,
        -1.3273047, 0.24462569, -1.7464504, 0.27482298, 1.6412262,
        0.83237153, 0.2587738, -1.8813281, 2.0521247, 2.2527733,
    ]

    private func makeLatent(values: [Float]) -> [Float] {
        // Repeat the 16 channel-0 values across each channel's 4096 elements.
        var x = [Float](repeating: 0, count: 16 * 64 * 64)
        for c in 0..<16 {
            let base = c * 64 * 64
            for i in 0..<(64 * 64) {
                x[base + i] = values[c]
            }
        }
        return x
    }

    /// D3 / 18.4: processOut(raw golden callback crop) must match the golden
    /// final_latent crop (the bit-exact relationship proven on CUDA).
    func testKnownVectorMatchesPinnedGoldenCrop() {
        let raw = makeLatent(values: rawStep7Channel0)
        let converted = Wan21LatentFormat.processOut(raw)
        for c in 0..<16 {
            let expected = finalChannel0[c]
            let got = converted[c * 64 * 64]
            XCTAssertEqual(got, expected, accuracy: 2e-4,
                           "channel \(c) process_out mismatch vs golden final_latent")
        }
    }

    /// 18.1: process_out(0) == latents_mean exactly (catches a missing mean).
    func testZeroInputIsMean() {
        let x = [Float](repeating: 0, count: 16 * 64 * 64)
        let out = Wan21LatentFormat.processOut(x)
        for c in 0..<16 {
            XCTAssertEqual(out[c * 64 * 64], Wan21LatentFormat.latentsMean[c],
                           accuracy: 1e-6, "process_out(0) must equal latents_mean[\(c)]")
        }
    }

    /// 18.3: deliberately distinct per-channel constants — a transpose/channel-
    /// order or broadcast bug cannot pass.
    func testChannelMappingDistinct() {
        let x = makeLatent(values: (0..<16).map { Float($0 + 1) })
        let out = Wan21LatentFormat.processOut(x)
        var distinct: Set<Float> = []
        for c in 0..<16 {
            let expected = Float(c + 1) * Wan21LatentFormat.latentsStd[c]
                / Wan21LatentFormat.scaleFactor + Wan21LatentFormat.latentsMean[c]
            XCTAssertEqual(out[c * 64 * 64], expected, accuracy: 1e-5,
                           "channel \(c) mapping")
            distinct.insert(out[c * 64 * 64])
        }
        XCTAssertEqual(distinct.count, 16, "all channels must be distinguishable")
    }

    /// 18.2: process_in(process_out(x)) ~= x within dtype tolerance.
    func testInverseRoundTrip() {
        var x = [Float](repeating: 0, count: 16 * 64 * 64)
        var seed: UInt64 = 42
        for i in 0..<x.count {
            // deterministic LCG in [-3, 3]
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            x[i] = Float(seed >> 33) / Float(1 << 24) * 6 - 3
        }
        let roundtrip = Wan21LatentFormat.processIn(Wan21LatentFormat.processOut(x))
        for i in 0..<x.count {
            XCTAssertEqual(roundtrip[i], x[i], accuracy: 2e-4,
                           "inverse mismatch at \(i)")
        }
    }

    /// Buffer path must be identical to the CPU array path (in-place is the
    /// production path used by GenerationEngine).
    func testBufferInPlaceMatchesArrayPath() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("SKIPPED_NO_METAL: MTLCreateSystemDefaultDevice unavailable")
        }
        let count = 16 * 64 * 64
        var x = [Float](repeating: 0, count: count)
        var seed: UInt64 = 1337
        for i in 0..<count {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            x[i] = Float(seed >> 33) / Float(1 << 24) * 6 - 3
        }
        guard let buffer = device.makeBuffer(bytes: &x, length: count * 4,
                                             options: .storageModeShared) else {
            throw XCTSkip("SKIPPED_NO_METAL: buffer allocation failed")
        }
        let expected = Wan21LatentFormat.processOut(x)
        Wan21LatentFormat.applyProcessOutInPlace(buffer)
        let got = Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count),
            count: count))
        for i in 0..<count {
            XCTAssertEqual(got[i], expected[i], accuracy: 1e-7,
                           "buffer in-place mismatch at \(i)")
        }
    }
}