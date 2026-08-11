import Foundation

/// App-stable normal-noise generator for production seeds.
///
/// SplitMix64 supplies uniform bits and Box-Muller supplies normal samples.
/// This deliberately does not claim bit parity with PyTorch/ComfyUI; golden
/// validation injects the recorded `init_noise_randn` instead.
struct SeededRNG {
    private var state: UInt64
    private var spare: Float?

    init(seed: UInt64) {
        state = seed
    }

    mutating func nextNormal() -> Float {
        if let value = spare {
            spare = nil
            return value
        }
        // 53 random bits mapped to the open interval (0, 1), avoiding log(0).
        let scale = 1.0 / 9_007_199_254_740_993.0
        let u1 = (Double(nextUInt64() >> 11) + 1.0) * scale
        let u2 = (Double(nextUInt64() >> 11) + 1.0) * scale
        let radius = sqrt(-2.0 * log(u1))
        let angle = 2.0 * Double.pi * u2
        spare = Float(radius * sin(angle))
        return Float(radius * cos(angle))
    }

    mutating func normal(count: Int) throws -> [Float] {
        guard count >= 0 else {
            throw AnimapkError.validation("normal noise count must be nonnegative")
        }
        return (0..<count).map { _ in nextNormal() }
    }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }
}
