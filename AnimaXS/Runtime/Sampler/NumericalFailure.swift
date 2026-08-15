import Foundation

/// A non-finite numerical failure detected during diffusion, formatted for
/// human consumption with 1-based (human-visible) step/block indexing.
///
/// The sampler throws this when the post-Euler latent is non-finite. The
/// diagnostic monitor may attach `block`/`stage`/`condition` attribution when
/// the first unsafe boundary is known; otherwise the message falls back to a
/// precise but unattributed description of where the finite check fired.
struct NumericalFailure: Error, LocalizedError, CustomStringConvertible, Equatable {
    /// Diffusion step, 1-based, in `1...totalSteps`.
    let step: Int
    /// Total Euler steps (production: `ModelConstants.samplerSteps` = 8).
    let totalSteps: Int
    /// DiT block, 1-based, when the failure is attributed to a block.
    let block: Int?
    /// Total DiT blocks (production: 28).
    let totalBlocks: Int
    /// Stage/substage attribution, e.g. "cross-attention output".
    let stage: String?
    /// Numerical condition observed, e.g. "Inf detected".
    let condition: String?
    /// Fallback description used when no stage/block attribution is available.
    let fallback: String

    /// Human-readable 1-based message.
    ///
    /// Attributed form:
    ///   "Numerical failure at diffusion step 5/8, block 17/28, cross-attention output: Inf detected."
    /// Unattributed form:
    ///   "Numerical failure at diffusion step 5/8 (Euler output is non-finite)."
    var message: String {
        let stepText = "diffusion step \(step)/\(totalSteps)"
        let prefix = "Numerical failure at \(stepText)"
        if let block, let stage, let condition {
            return "\(prefix), block \(block)/\(totalBlocks), \(stage): \(condition)."
        }
        return "\(prefix) (\(fallback))."
    }

    var description: String { message }

    var errorDescription: String? { message }

    /// Build an unattributed failure for the post-Euler finite check.
    static func eulerOutput(
        step: Int, totalSteps: Int
    ) -> NumericalFailure {
        NumericalFailure(
            step: step, totalSteps: totalSteps, block: nil, totalBlocks: 0,
            stage: nil, condition: nil, fallback: "Euler output is non-finite")
    }

    /// Build a stage-attributed failure.
    static func attributed(
        step: Int, totalSteps: Int, block: Int, totalBlocks: Int,
        stage: String, condition: String
    ) -> NumericalFailure {
        NumericalFailure(
            step: step, totalSteps: totalSteps, block: block, totalBlocks: totalBlocks,
            stage: stage, condition: condition, fallback: "Euler output is non-finite")
    }
}
