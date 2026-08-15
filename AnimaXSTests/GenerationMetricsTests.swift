import XCTest
@testable import AnimaXS

/// Phase 7/8 — generation telemetry accumulator and summary formatting.
final class GenerationMetricsTests: XCTestCase {

    func testStagesAndOtherReconcileWithTotalWall() {
        let collector = MetricsCollector()
        collector.beginStage(.textEncode)
        collector.beginStage(.adapter)
        collector.beginStage(.diffusion)
        collector.beginStage(.vae)
        // All stages share the same instant — no real sleeping in unit tests.
        collector.endStage(.textEncode)
        collector.endStage(.adapter)
        collector.endStage(.diffusion)
        collector.endStage(.vae)
        collector.finalize(totalWall: 100)
        let metrics = collector.snapshot()
        // Nothing measurable elapsed, so "other" absorbs the whole wall.
        XCTAssertEqual(metrics.other, 100, accuracy: 0.001)
    }

    func testStepsBlocksAndMetalAccounting() {
        let collector = MetricsCollector()
        collector.beginStep(0)
        for block in 0..<28 {
            collector.beginBlock(block)
            collector.endBlock()
        }
        collector.endStep()
        collector.recordWeightCopy(bytes: 38_993_920, seconds: 0.35)
        collector.recordGPUCommand(seconds: 1.5)
        collector.recordHostWait(seconds: 0.1)
        collector.recordEncode(seconds: 0.02)
        collector.recordMemory(allocated: 1_400_000_000, available: 600_000_000)
        collector.recordMemory(allocated: 1_500_000_000, available: 590_000_000)
        collector.setNumericalWarnings(2)

        let metrics = collector.snapshot()
        XCTAssertEqual(metrics.stepTimes.count, 1)
        XCTAssertEqual(metrics.blockTimes.count, 28)
        XCTAssertEqual(metrics.blockCount, 28)
        XCTAssertEqual(metrics.weightCopyBytes, 38_993_920)
        XCTAssertEqual(metrics.weightCopyTime, 0.35, accuracy: 0.001)
        XCTAssertEqual(metrics.gpuCommandTime, 1.5, accuracy: 0.001)
        XCTAssertEqual(metrics.peakMetalAllocation, 1_500_000_000)
        XCTAssertEqual(metrics.minAvailableMemory, 590_000_000)
        XCTAssertEqual(metrics.numericalWarnings, 2)
    }

    func testSummaryTextShape() {
        let collector = MetricsCollector()
        collector.beginStep(0)
        collector.endStep()
        collector.recordWeightCopy(bytes: 38_993_920, seconds: 31.5)
        collector.recordGPUCommand(seconds: 181.3)
        collector.recordHostWait(seconds: 35.0)
        collector.recordMemory(allocated: 1_500_000_000, available: 600_000_000)
        collector.finalize(totalWall: 247.8)

        let text = collector.snapshot().summaryText
        XCTAssertTrue(text.contains("Generation: 247.8 s"))
        XCTAssertTrue(text.contains("Measured GPU command time: 181.3 s"))
        XCTAssertTrue(text.contains("Weight copy/load time: 31.5 s"))
        XCTAssertTrue(text.contains("Host/other measured time: 35.0 s"))
        XCTAssertTrue(text.contains("Peak Metal allocation: 1.40 GB"))
        XCTAssertTrue(text.contains("Minimum available process memory: 572 MB"))
        XCTAssertTrue(text.contains("Numerical warnings: 0"))
    }

    func testSummarySaysNAWhenNoMemorySampled() {
        let collector = MetricsCollector()
        collector.finalize(totalWall: 10)
        let text = collector.snapshot().summaryText
        XCTAssertFalse(text.contains("Minimum available process memory: 0"))
    }
}
