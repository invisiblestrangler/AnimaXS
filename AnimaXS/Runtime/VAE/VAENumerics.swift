import Foundation

/// Small pinned-source CPU references for Wan VAE primitives. Production VAE
/// kernels must match these equations but must not use these array loops.
enum VAENumerics {
    static func decodeLatent(_ latent: [Float], channels: Int = 16) throws -> [Float] {
        guard channels == ModelConstants.latentsMean.count,
              channels == ModelConstants.latentsStd.count,
              !latent.isEmpty, latent.count.isMultiple(of: channels) else {
            throw AnimapkError.validation("invalid Wan latent normalization input")
        }
        let spatial = latent.count / channels
        var output = latent
        for channel in 0..<channels {
            let mean = ModelConstants.latentsMean[channel]
            let std = ModelConstants.latentsStd[channel]
            for index in 0..<spatial {
                let offset = channel * spatial + index
                output[offset] = latent[offset] / ModelConstants.vaeSigmaData * std + mean
            }
        }
        return output
    }

    /// Wan `F.normalize(x, dim=1) * sqrt(C) * gamma`, channel-first CHW.
    static func channelRMSNorm(
        _ input: [Float], gamma: [Float], channels: Int, spatial: Int
    ) throws -> [Float] {
        guard channels > 0, spatial > 0, gamma.count == channels,
              input.count == channels * spatial else {
            throw AnimapkError.validation("invalid Wan RMS normalization input")
        }
        var output = [Float](repeating: 0, count: input.count)
        let scale = sqrt(Float(channels))
        for position in 0..<spatial {
            var sum: Float = 0
            for channel in 0..<channels {
                let value = input[channel * spatial + position]
                sum += value * value
            }
            let inverse = 1 / max(sqrt(sum), 1e-12)
            for channel in 0..<channels {
                let offset = channel * spatial + position
                output[offset] = input[offset] * inverse * scale * gamma[channel]
            }
        }
        return output
    }

    static func silu(_ input: [Float]) -> [Float] {
        input.map { $0 / (1 + exp(-$0)) }
    }

    /// Integer 2x nearest-exact is an exact source-pixel replication.
    static func nearestExact2x(
        _ input: [Float], channels: Int, height: Int, width: Int
    ) throws -> [Float] {
        guard channels > 0, height > 0, width > 0,
              input.count == channels * height * width else {
            throw AnimapkError.validation("invalid nearest-exact input")
        }
        let outputHeight = height * 2, outputWidth = width * 2
        var output = [Float](repeating: 0, count: channels * outputHeight * outputWidth)
        for channel in 0..<channels {
            for y in 0..<outputHeight {
                for x in 0..<outputWidth {
                    output[(channel * outputHeight + y) * outputWidth + x] =
                        input[(channel * height + y / 2) * width + x / 2]
                }
            }
        }
        return output
    }
}
