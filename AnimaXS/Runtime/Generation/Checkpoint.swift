import Foundation

struct ModelHashes: Codable, Equatable {
    let dit: String
    let textEncoder: String
    let vae: String
}

/// Versioned, atomically persisted sampler checkpoint.
struct GenerationCheckpoint: Codable, Equatable {
    static let formatVersion = 1

    let version: Int
    let latent: Data
    let step: Int
    let prompt: String
    let seed: UInt64
    let width: Int
    let height: Int
    let modelHashes: ModelHashes

    init(
        latent: [Float], step: Int, prompt: String, seed: UInt64,
        width: Int, height: Int, modelHashes: ModelHashes
    ) throws {
        guard step >= 0, step <= ModelConstants.samplerSteps,
              width > 0, height > 0, width.isMultiple(of: 8), height.isMultiple(of: 8),
              latent.count == ModelConstants.ditLatentChannels * (width / 8) * (height / 8),
              latent.allSatisfy(\.isFinite),
              !modelHashes.dit.isEmpty, !modelHashes.textEncoder.isEmpty,
              !modelHashes.vae.isEmpty else {
            throw AnimapkError.validation("invalid generation checkpoint")
        }
        version = Self.formatVersion
        self.latent = latent.withUnsafeBytes { Data($0) }
        self.step = step
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.modelHashes = modelHashes
    }

    func latentValues() throws -> [Float] {
        let expected = ModelConstants.ditLatentChannels * (width / 8) * (height / 8)
        guard version == Self.formatVersion, latent.count == expected * 4 else {
            throw AnimapkError.validation("unsupported or corrupt generation checkpoint")
        }
        return latent.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
    }

    func writeAtomically(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: url, options: [.atomic])
    }

    static func load(from url: URL) throws -> GenerationCheckpoint {
        let checkpoint = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        let values = try checkpoint.latentValues()
        guard checkpoint.step >= 0, checkpoint.step <= ModelConstants.samplerSteps,
              checkpoint.width > 0, checkpoint.height > 0,
              checkpoint.width.isMultiple(of: 8), checkpoint.height.isMultiple(of: 8),
              values.allSatisfy(\.isFinite),
              !checkpoint.modelHashes.dit.isEmpty,
              !checkpoint.modelHashes.textEncoder.isEmpty,
              !checkpoint.modelHashes.vae.isEmpty else {
            throw AnimapkError.validation("invalid generation checkpoint metadata")
        }
        return checkpoint
    }
}
