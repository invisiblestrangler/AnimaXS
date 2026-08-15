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
        XCTAssertTrue(text.contains("Weight copy/load CPU work: 31.5 s, 37 MB"))
        XCTAssertTrue(text.contains("may overlap GPU time when ping-pong is on"))
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

    /// The summary must embed the immutable per-run config snapshot, the
    /// executor tile counters, the environment start/end, and the overlap
    /// caveat for weight-copy time (§14, §18.6).
    func testSummaryIncludesConfigCountersAndEnvironment() {
        let collector = MetricsCollector()
        var config = InferenceOptimizationConfig.currentBaseline
        config.linearTileRows = 1024
        config.attentionTileRows = 512
        config.directLinearMPSIO = true
        config.pingPongWeightStreaming = false
        config.numericalMonitoring = false
        collector.recordOptimizationConfig(config)
        collector.recordLinearGEMMTile(directInput: true, directOutput: true)
        collector.recordLinearGEMMTile(directInput: true, directOutput: false)
        collector.recordLinearGEMMTile(directInput: false, directOutput: true)
        collector.recordAttentionQueryTile()
        collector.recordEnvironmentStart(EnvironmentSnapshot(
            powerState: "unplugged", batteryLevel: 78, thermalState: "nominal", lowPowerMode: false))
        collector.recordEnvironmentEnd(EnvironmentSnapshot(
            powerState: "unplugged", batteryLevel: 51, thermalState: "fair", lowPowerMode: false))
        collector.finalize(totalWall: 100)

        let text = collector.snapshot().summaryText
        // Config snapshot. (No DiT pack file was recorded for this run, so the
        // summary reports "unknown".)
        XCTAssertTrue(text.contains("Inference configuration"))
        XCTAssertTrue(text.contains("DiT pack: unknown"))
        XCTAssertTrue(text.contains("Linear tile rows: 1024"))
        XCTAssertTrue(text.contains("Attention tile rows: 512"))
        XCTAssertTrue(text.contains("Direct MPS linear I/O: on"))
        XCTAssertTrue(text.contains("Ping-pong weight streaming: off"))
        XCTAssertTrue(text.contains("Numerical monitor: off"))
        // Checkpointing on for production W4.
        XCTAssertTrue(text.contains("Checkpointing: on"))
        // Counters: 3 GEMM tiles → 2 direct + 1 copied input, 2 direct + 1 copied output.
        XCTAssertTrue(text.contains("Linear GEMM tiles: 3"))
        XCTAssertTrue(text.contains("Linear input tiles: 2 direct / 1 copied"))
        XCTAssertTrue(text.contains("Linear output tiles: 2 direct / 1 copied"))
        XCTAssertTrue(text.contains("Attention query tiles: 1"))
        // Monitor OFF wording.
        XCTAssertTrue(text.contains("Numerical monitor: off (Euler finite guard on)"))
        XCTAssertTrue(text.contains("Numerical warnings: not collected"))
        XCTAssertFalse(text.contains("Numerical warnings: 0"))
        // Environment.
        XCTAssertTrue(text.contains("Power: unplugged -> unplugged"))
        XCTAssertTrue(text.contains("Battery: 78% -> 51%"))
        XCTAssertTrue(text.contains("Thermal: nominal -> fair"))
        XCTAssertTrue(text.contains("Low Power Mode: off -> off"))
    }

    /// The summary reports which DiT pack file ran and that checkpointing is
    /// always on (the DiT slot holds one verified pack).
    func testSummaryReportsDiTPackFilenameAndCheckpointing() {
        let collector = MetricsCollector()
        collector.recordDiTPackFilename("anima-turbo-v1.0-xsmax-w8-v2.animapk")
        collector.finalize(totalWall: 100)
        let text = collector.snapshot().summaryText
        XCTAssertTrue(text.contains("DiT pack: anima-turbo-v1.0-xsmax-w8-v2.animapk"))
        XCTAssertTrue(text.contains("Checkpointing: on"))
    }
}
