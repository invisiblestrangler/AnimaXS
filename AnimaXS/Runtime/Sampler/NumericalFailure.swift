import Foundation

/// Where a numerical failure occurred in the diffusion pipeline. This makes
/// "after all blocks, in the final layer" explicit so the message can never
/// fabricate a block index (e.g. "block 1/28") for a final-layer failure.
enum NumericalLocation: Equatable {
    /// A failure attributed to a specific DiT block (1-based index).
    case block(Int)
    /// A failure in the final layer, after all DiT blocks.
    case finalLayer
    /// A failure of the post-Euler finite guard (no block / final-layer stage).
    case eulerUpdate
}

/// A non-finite numerical failure detected during diffusion, formatted for
/// human consumption with 1-based (human-visible) step/block indexing.
///
/// The sampler throws this when the post-Euler latent is non-finite. The
/// diagnostic monitor may attach `location`/`stage`/`condition` attribution
/// when the first unsafe boundary is known; otherwise the message falls back
/// to a precise but unattributed description of where the finite check fired.
struct NumericalFailure: Error, LocalizedError, CustomStringConvertible, Equatable {
    /// Diffusion step, 1-based, in `1...totalSteps`.
    let step: Int
    /// Total Euler steps (production: `ModelConstants.samplerSteps` = 8).
    let totalSteps: Int
    /// Where the failure occurred (block / final layer / Euler update).
    let location: NumericalLocation
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
    /// Attributed forms:
    ///   "Numerical failure at diffusion step 5/8, block 17/28, cross-attention output: Inf detected."
    ///   "Numerical failure at diffusion step 1/8, final layer (after block 28/28): Inf detected."
    /// Unattributed form:
    ///   "Numerical failure at diffusion step 5/8 (Euler output is non-finite)."
    var message: String {
        let stepText = "diffusion step \(step)/\(totalSteps)"
        let prefix = "Numerical failure at \(stepText)"
        switch location {
        case .block(let block):
            if let stage, let condition {
                return "\(prefix), block \(block)/\(totalBlocks), \(stage): \(condition)."
            }
        case .finalLayer:
            if let stage, let condition {
                return "\(prefix), final layer (after block \(totalBlocks)/\(totalBlocks)), \(stage): \(condition)."
            }
        case .eulerUpdate:
            break
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
            step: step, totalSteps: totalSteps,
            location: .eulerUpdate, totalBlocks: 0,
            stage: nil, condition: nil, fallback: "Euler output is non-finite")
    }

    /// Build a block-attributed failure.
    static func attributed(
        step: Int, totalSteps: Int, block: Int, totalBlocks: Int,
        stage: String, condition: String
    ) -> NumericalFailure {
        NumericalFailure(
            step: step, totalSteps: totalSteps,
            location: .block(block), totalBlocks: totalBlocks,
            stage: stage, condition: condition, fallback: "Euler output is non-finite")
    }

    /// Build a final-layer-attributed failure. Used when the numerical monitor
    /// reports an issue at a final-layer probe, which has no block index.
    static func finalLayer(
        step: Int, totalSteps: Int, totalBlocks: Int,
        stage: String, condition: String
    ) -> NumericalFailure {
        NumericalFailure(
            step: step, totalSteps: totalSteps,
            location: .finalLayer, totalBlocks: totalBlocks,
            stage: stage, condition: condition, fallback: "Euler output is non-finite")
    }
}
