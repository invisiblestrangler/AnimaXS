import XCTest
@testable import AnimaXS

/// Fix G regression tests: every generation attempt resolves to either
/// `.ready` or a `.blocked(reason)` with a user-visible explanation — there is
/// no path where a tapped Generate silently does nothing.
final class GenerationEligibilityTests: XCTestCase {

    private func evaluate(
        modelsResolved: Bool = true,
        isGenerating: Bool = false,
        prompt: String = "masterpiece",
        seedText: String = "1337",
        metalAvailable: Bool = true,
        optimizationBlockingReason: String? = nil
    ) -> GenerationEligibility {
        GenerationEligibility.evaluate(
            modelsResolved: modelsResolved,
            isGenerating: isGenerating,
            prompt: prompt,
            seedText: seedText,
            metalAvailable: metalAvailable,
            optimizationBlockingReason: optimizationBlockingReason)
    }

    func testReadyWhenAllConditionsMet() {
        XCTAssertEqual(evaluate(), .ready)
    }

    func testReadyIgnoresWhitespacePadding() {
        XCTAssertEqual(evaluate(prompt: "  masterpiece\n", seedText: "  1337  "), .ready)
    }

    func testBlockedWhenAlreadyGenerating() {
        // Generation in flight takes precedence over everything else.
        let result = evaluate(
            modelsResolved: false, isGenerating: true, prompt: "",
            seedText: "not-a-seed")
        guard case .blocked(let reason) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertTrue(reason.contains("already running"))
    }

    func testBlockedWhenModelsMissing() {
        let result = evaluate(modelsResolved: false)
        guard case .blocked(let reason) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertTrue(reason.contains("Models"))
    }

    func testBlockedWhenPromptEmpty() {
        for empty in ["", "   ", "\n\n"] {
            let result = evaluate(prompt: empty)
            guard case .blocked(let reason) = result else {
                return XCTFail("expected blocked for prompt '\(empty)', got \(result)")
            }
            XCTAssertTrue(reason.contains("Prompt cannot be empty"), reason)
        }
    }

    func testBlockedWhenSeedInvalid() {
        for bad in ["", "abc", "12.5", "-1", "18446744073709551616"] {  // > UInt64.max
            let result = evaluate(seedText: bad)
            guard case .blocked(let reason) = result else {
                return XCTFail("expected blocked for seed '\(bad)', got \(result)")
            }
            XCTAssertTrue(reason.contains("Seed"), reason)
        }
    }

    func testBlockedWhenMetalUnavailable() {
        let result = evaluate(metalAvailable: false)
        guard case .blocked(let reason) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertTrue(reason.contains("Metal"))
    }

    func testBlockedReasonIsNeverEmpty() {
        // Property-style sweep: for every single-condition miss the reason must
        // be non-empty and informative — the UI can always explain the block.
        let cases: [GenerationEligibility] = [
            evaluate(modelsResolved: false),
            evaluate(isGenerating: true),
            evaluate(prompt: ""),
            evaluate(seedText: "x"),
            evaluate(metalAvailable: false),
        ]
        for result in cases {
            if case .blocked(let reason) = result {
                XCTAssertFalse(reason.isEmpty)
            } else {
                XCTFail("expected a blocked case")
            }
        }
    }

    func testValidPathTransitionsAwayFromIdle() {
        // The ready case is the only one that allows startGeneration to call
        // the coordinator, which synchronously leaves .idle. (The synchronous
        // transition itself is covered by GenerationCoordinatorTests.)
        XCTAssertTrue(evaluate().isReady)
        XCTAssertNil(evaluate().blockedReason)
        XCTAssertNotNil(evaluate(prompt: "").blockedReason)
    }

    // MARK: - Central compatibility validator threading (Task 9)

    // An incompatible optimization configuration blocks Generate with the
    // validator's exact reason, even when every other condition is fine.
    func testBlockedByOptimizationConfigReason() {
        let reason = InferenceOptimizationConfig.noCopyBlockingReason
        let result = evaluate(optimizationBlockingReason: reason)
        guard case .blocked(let blocked) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertEqual(blocked, reason)
        XCTAssertFalse(result.isReady)
    }

    // The validator reason takes priority over the input conditions (models/
    // prompt/seed/Metal), so the incompatible configuration is never masked
    // by an environment issue.
    func testOptimizationBlockingReasonTakesPriorityOverInputConditions() {
        let reason = InferenceOptimizationConfig.linearBackendBlockingReason
        let result = evaluate(
            modelsResolved: false,
            prompt: "",
            seedText: "bad",
            metalAvailable: false,
            optimizationBlockingReason: reason)
        guard case .blocked(let blocked) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertEqual(blocked, reason)
    }

    // An in-flight generation still wins over the config reason: the running
    // generation guard is the very first check.
    func testAlreadyGeneratingWinsOverOptimizationBlockingReason() {
        let result = evaluate(
            isGenerating: true,
            optimizationBlockingReason: InferenceOptimizationConfig.noCopyBlockingReason)
        guard case .blocked(let reason) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertTrue(reason.contains("already running"))
    }

    // No config reason (compatible config) leaves the normal conditions
    // untouched.
    func testNilOptimizationBlockingReasonKeepsNormalConditions() {
        XCTAssertEqual(evaluate(optimizationBlockingReason: nil), .ready)
        XCTAssertEqual(evaluate(prompt: "", optimizationBlockingReason: nil)
            .blockedReason, "Prompt cannot be empty.")
    }
}
