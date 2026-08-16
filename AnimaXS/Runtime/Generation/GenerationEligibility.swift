import Foundation

/// Single source of truth for whether a user-triggered generation can start.
///
/// The same value drives the Generate button's enabled state, the visible
/// explanatory text under it, and the actual start guard — so there is no way
/// for the button to be tappable while the start guard silently returns.
enum GenerationEligibility: Equatable {
    case ready
    case blocked(String)

    /// Evaluates every condition exactly once, in a stable priority order.
    /// The first blocking condition wins; the returned reason is user-visible.
    static func evaluate(
        modelsResolved: Bool,
        isGenerating: Bool,
        prompt: String,
        seedText: String,
        metalAvailable: Bool
    ) -> GenerationEligibility {
        if isGenerating {
            return .blocked("A generation is already running.")
        }
        if !modelsResolved {
            return .blocked("Models are not ready yet. Download or import all three model packs first.")
        }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .blocked("Prompt cannot be empty.")
        }
        if UInt64(seedText.trimmingCharacters(in: .whitespaces)) == nil {
            return .blocked("Seed must be an unsigned 64-bit integer.")
        }
        if !metalAvailable {
            return .blocked("Metal is unavailable on this device.")
        }
        return .ready
    }

    /// True when generation may start.
    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    /// The user-visible reason when generation is blocked.
    var blockedReason: String? {
        if case .blocked(let reason) = self { return reason }
        return nil
    }
}
