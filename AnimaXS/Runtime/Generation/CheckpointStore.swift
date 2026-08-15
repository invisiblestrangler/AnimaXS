import Foundation

/// File-backed checkpoint persistence for resumable generation (I004/K003).
///
/// The checkpoint is written atomically after every completed diffusion step,
/// so a cancel, background transition, or memory warning can always resume
/// from the last fully completed step. The store also validates that a
/// checkpoint is compatible with the current generation inputs (prompt, seed,
/// resolution, model hashes) before it is offered to the UI.
final class CheckpointStore {
    private let fileURL: URL

    init(directory: URL? = nil) throws {
        if let directory {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            self.fileURL = directory.appendingPathComponent("generation-checkpoint.json")
        } else {
            let base = try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true)
            let directory = base.appendingPathComponent("AnimaXS", isDirectory: true)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            self.fileURL = directory.appendingPathComponent("generation-checkpoint.json")
        }
    }

    var hasCheckpoint: Bool {
        FileManager.default.fileExists(atPath: fileURL.path)
    }

    func load() -> GenerationCheckpoint? {
        guard hasCheckpoint else { return nil }
        do {
            return try GenerationCheckpoint.load(from: fileURL)
        } catch {
            // A corrupt checkpoint must not trap the app: remove it and treat
            // as absent so the user can start fresh (K003 repair path).
            try? FileManager.default.removeItem(at: fileURL)
            return nil
        }
    }

    func save(_ checkpoint: GenerationCheckpoint) throws {
        try checkpoint.writeAtomically(to: fileURL)
    }

    func remove() {
        try? FileManager.default.removeItem(at: fileURL)
    }

    /// Checks that a checkpoint can resume the requested generation.
    ///
    /// - Returns: The checkpoint's next step (number of completed steps) on
    ///   success, or an error describing the first incompatible field.
    func validate(
        _ checkpoint: GenerationCheckpoint,
        prompt: String,
        seed: UInt64,
        resolution: (width: Int, height: Int),
        modelHashes: ModelHashes
    ) throws -> Int {
        guard checkpoint.step >= 1, checkpoint.step < ModelConstants.samplerSteps else {
            throw GenerationError.sampler(
                "checkpoint step \(checkpoint.step) out of range 1...\(ModelConstants.samplerSteps - 1) (a fully completed run is not resumable)")
        }
        guard checkpoint.prompt == prompt else {
            throw GenerationError.sampler(
                "checkpoint prompt does not match the current prompt")
        }
        guard checkpoint.seed == seed else {
            throw GenerationError.sampler(
                "checkpoint seed does not match the current seed")
        }
        guard checkpoint.width == resolution.width,
              checkpoint.height == resolution.height else {
            throw GenerationError.sampler(
                "checkpoint resolution \(checkpoint.width)x\(checkpoint.height) "
                + "does not match \(resolution.width)x\(resolution.height)")
        }
        guard checkpoint.modelHashes == modelHashes else {
            throw GenerationError.sampler(
                "checkpoint model hashes do not match the current model packs")
        }
        // Version, shape, and finiteness were already validated by load().
        return checkpoint.step
    }
}
