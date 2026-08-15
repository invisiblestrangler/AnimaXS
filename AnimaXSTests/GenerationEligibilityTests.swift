import XCTest
@testable import AnimaXS

/// Fix G regression tests: every generation attempt resolves to either
/// `.ready` or a `.blocked(reason)` with a user-visible explanation — there is
/// no path where a tapped Generate silently does nothing.
final class GenerationEligibilityTests: XCTestCase {

    private func evaluate(
        modelsResolved: Bool = true,
        isGenerating: Bool = false,
        canResume: Bool = false,
        prompt: String = "masterpiece",
        seedText: String = "1337",
        metalAvailable: Bool = true
    ) -> GenerationEligibility {
        GenerationEligibility.evaluate(
            modelsResolved: modelsResolved,
            isGenerating: isGenerating,
            canResume: canResume,
            prompt: prompt,
            seedText: seedText,
            metalAvailable: metalAvailable)
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

    func testBlockedWhenCheckpointAvailable() {
        let result = evaluate(canResume: true)
        guard case .blocked(let reason) = result else {
            return XCTFail("expected blocked, got \(result)")
        }
        XCTAssertTrue(reason.contains("Resume"))
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
            evaluate(canResume: true),
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
}
