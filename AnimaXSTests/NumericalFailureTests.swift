import XCTest
@testable import AnimaXS

/// Phase 1 — numerical failure reporting must use real interpolation and
/// human-friendly 1-based step/block indexing (runbook Phase 1).
final class NumericalFailureTests: XCTestCase {

    func testAttributedMessageUsesOneBasedIndexing() {
        let failure = NumericalFailure.attributed(
            step: 5, totalSteps: 8, block: 17, totalBlocks: 28,
            stage: "cross-attention output", condition: "Inf detected")
        XCTAssertEqual(
            failure.message,
            "Numerical failure at diffusion step 5/8, block 17/28, cross-attention output: Inf detected.")
        XCTAssertEqual(failure.localizedDescription, failure.message)
        XCTAssertEqual(String(describing: failure), failure.message)
    }

    func testEulerOutputFallbackMessage() {
        let failure = NumericalFailure.eulerOutput(step: 3, totalSteps: 8)
        XCTAssertEqual(
            failure.message,
            "Numerical failure at diffusion step 3/8 (Euler output is non-finite).")
    }

    func testFirstStepAndLastBlockIndexing() {
        // Internal 0-based step 0 must surface as human 1-based step 1.
        let first = NumericalFailure.eulerOutput(step: 1, totalSteps: 8)
        XCTAssertTrue(first.message.hasPrefix("Numerical failure at diffusion step 1/8"))
        let last = NumericalFailure.attributed(
            step: 8, totalSteps: 8, block: 28, totalBlocks: 28,
            stage: "final layer", condition: "NaN detected")
        XCTAssertTrue(last.message.contains("diffusion step 8/8"))
        XCTAssertTrue(last.message.contains("block 28/28"))
        XCTAssertTrue(last.message.contains("final layer: NaN detected"))
    }

    func testAllFailureConditionsAreRepresented() {
        for condition in ["NaN detected", "positive Inf detected", "negative Inf detected"] {
            let failure = NumericalFailure.attributed(
                step: 1, totalSteps: 8, block: 1, totalBlocks: 28,
                stage: "self-attention output", condition: condition)
            XCTAssertTrue(failure.message.contains(condition))
        }
    }

    // P1-E: a final-layer failure must never fabricate "block 1/28". It is
    // attributed explicitly as the final layer after all blocks.
    func testFinalLayerFailureNeverFabricatesBlockOne() {
        let failure = NumericalFailure.finalLayer(
            step: 1, totalSteps: 8, totalBlocks: 28,
            stage: "final-layer residual conversion", condition: "Inf detected")
        XCTAssertEqual(
            failure.message,
            "Numerical failure at diffusion step 1/8, final layer (after block 28/28), final-layer residual conversion: Inf detected.")
        XCTAssertFalse(failure.message.contains("block 1/28"))
        XCTAssertFalse(failure.message.contains("block 1/"))
    }

    func testFinalLayerFailureMessageShape() {
        let failure = NumericalFailure.finalLayer(
            step: 5, totalSteps: 8, totalBlocks: 28,
            stage: "final-layer projection input", condition: "NaN detected")
        XCTAssertTrue(failure.message.contains("diffusion step 5/8"))
        XCTAssertTrue(failure.message.contains("final layer (after block 28/28)"))
        XCTAssertTrue(failure.message.contains("final-layer projection input"))
    }
}
