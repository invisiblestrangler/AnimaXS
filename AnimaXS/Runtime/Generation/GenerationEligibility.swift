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
    ///
    /// - Parameters:
    ///   - w8Selected: True when the run config selects the experimental W8
    ///     pack. W8 must be verified/ready before Generate is offered.
    ///   - w8Ready: True when the experimental W8 pack is verified and
    ///     installed. Ignored when `w8Selected` is false.
    static func evaluate(
        modelsResolved: Bool,
        isGenerating: Bool,
        canResume: Bool,
        prompt: String,
        seedText: String,
        metalAvailable: Bool,
        w8Selected: Bool = false,
        w8Ready: Bool = false
    ) -> GenerationEligibility {
        if isGenerating {
            return .blocked("A generation is already running.")
        }
        if canResume {
            return .blocked("A checkpoint is available — use Resume instead.")
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
        if w8Selected && !w8Ready {
            return .blocked("Experimental W8 is selected but not verified. Import it in Diagnostics or switch back to Production W4.")
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
