import XCTest
import Metal
@testable import AnimaXS

/// Phase 3 — numerical-health instrumentation (probe kernels + monitor).
/// GPU tests run on the standard simulator Metal device; they skip cleanly
/// where Metal is unavailable (same convention as MetalExecutionTests).
final class NumericalMonitorTests: XCTestCase {

    private func requireContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: MTLCreateSystemDefaultDevice/default library unavailable")
        }
        return context
    }

    private func makeBuffer<T>(_ values: [T], on device: MTLDevice) -> MTLBuffer {
        values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                              options: .storageModeShared)!
        }
    }

    private func makeFloatBuffer(_ values: [Float], on device: MTLDevice) -> MTLBuffer {
        makeBuffer(values, on: device)
    }

    private func makeHalfBuffer(_ values: [Float], on device: MTLDevice) -> MTLBuffer {
        makeBuffer(values.map { Float16($0) }, on: device)
    }

    // MARK: - CPU-side stats decoding

    func testStatsConditionDecoding() {
        var stats = NumericalMonitor.Stats()
        XCTAssertFalse(stats.hasIssue)
        XCTAssertEqual(stats.condition, "non-finite value detected")

        stats.flags = NumericalMonitor.Flag.nan.rawValue
        XCTAssertEqual(stats.condition, "NaN detected")
        stats.flags = NumericalMonitor.Flag.posInf.rawValue
        XCTAssertEqual(stats.condition, "positive Inf detected")
        stats.flags = NumericalMonitor.Flag.negInf.rawValue
        XCTAssertEqual(stats.condition, "negative Inf detected")
        stats.flags = NumericalMonitor.Flag.resultNaN.rawValue
        XCTAssertEqual(stats.condition, "residual became NaN")
        stats.flags = NumericalMonitor.Flag.resultInf.rawValue
        XCTAssertEqual(stats.condition, "residual became non-finite")
        stats.flags = NumericalMonitor.Flag.halfOverflow.rawValue
        XCTAssertEqual(stats.condition, "value exceeded FP16 finite range")
    }

    func testEarliestIssueDrivesAttributedFailure() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        // Simulate a GPU-observed issue at self-attention scores in block 4.
        var raw = [UInt32](repeating: 0, count: NumericalMonitor.Probe.allCases.count * 4)
        let slot = NumericalMonitor.Probe.selfScores.rawValue
        raw[slot * 4 + 0] = NumericalMonitor.Flag.posInf.rawValue
        raw[slot * 4 + 2] = 42
        // Overwrite the stats buffer with the crafted values (shared memory).
        raw.withUnsafeBytes { bytes in
            memcpy(monitor.statsBufferForTesting, bytes.baseAddress!, bytes.count)
        }
        monitor.noteBlockCompleted(step: 3, block: 4) // 0-based step 3, block 4
        let issue = try XCTUnwrap(monitor.earliestIssue)
        XCTAssertEqual(issue.step, 4)   // 1-based
        XCTAssertEqual(issue.block, 5)  // 1-based
        XCTAssertEqual(issue.probe, .selfScores)
        XCTAssertEqual(issue.stats.condition, "positive Inf detected")

        let failure = NumericalFailure.attributed(
            step: issue.step, totalSteps: 8, block: issue.block ?? 1, totalBlocks: 28,
            stage: issue.probe.stageLabel, condition: issue.stats.condition)
        XCTAssertEqual(
            failure.message,
            "Numerical failure at diffusion step 4/8, block 5/28, self-attention scores: positive Inf detected.")
    }

    // MARK: - float_to_half_probe

    func testFloatToHalfProbeFlagsNaNInfAndOverflow() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        let input = makeFloatBuffer(
            [1.0, -1.0, Float.nan, Float.infinity, -Float.infinity, 70_000.0, 65504.0, 0.5],
            on: context.device)
        let output = try XCTUnwrap(
            context.device.makeBuffer(length: 8 * 2, options: .storageModeShared))
        let pipeline = try context.pipeline(named: "float_to_half_probe")
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var count: UInt32 = 8
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        monitor.bindProbe(encoder, probe: .selfProjectionInput, statsIndex: 3, slotIndex: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)

        monitor.noteStepCompleted(step: 0)
        let stats = monitor.report()[.selfProjectionInput] ?? .noIssue
        XCTAssertTrue(stats.hasNaN)
        XCTAssertTrue(stats.hasPosInf)
        XCTAssertTrue(stats.hasNegInf)
        XCTAssertTrue(stats.hasHalfOverflow)
        XCTAssertEqual(stats.firstIndex, 2) // NaN at index 2 (execution order)
        XCTAssertEqual(stats.maxAbs, 70_000.0, accuracy: 0.01)

        // Conversion semantics must be unchanged: 70_000 → Inf in fp16.
        let halves = output.contents().assumingMemoryBound(to: Float16.self)
        XCTAssertEqual(Float(halves[0]), 1.0)
        XCTAssertEqual(Float(halves[4]), -.infinity)
        XCTAssertEqual(Float(halves[5]), .infinity) // 70_000 overflows fp16
        XCTAssertTrue(halves[2].isNaN)
    }

    // MARK: - gate_add_half_f32_probe

    func testGateAddProbeDetectsBranchInfPoisoningResidual() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        // Branch carries +Inf at index 3; residual starts finite.
        let residual = makeFloatBuffer([1.0, 2.0, 3.0, 4.0, 5.0], on: context.device)
        let branch = makeHalfBuffer([0.1, 0.2, 0.3, Float.infinity, 0.5], on: context.device)
        let gate = makeFloatBuffer([1.0, 1.0, 1.0, 1.0, 1.0], on: context.device)
        let pipeline = try context.pipeline(named: "gate_add_half_f32_probe")
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var n: UInt32 = 5, count: UInt32 = 5
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBuffer(gate, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&count, length: 4, index: 4)
        monitor.bindProbe(encoder, probe: .selfGateAdd, statsIndex: 5, slotIndex: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

        monitor.noteStepCompleted(step: 0)
        let stats = monitor.report()[.selfGateAdd] ?? .noIssue
        XCTAssertTrue(stats.hasPosInf)
        XCTAssertTrue(stats.resultInf) // 4 + Inf*1 = Inf
        XCTAssertEqual(stats.firstIndex, 3)

        // Residual must have been poisoned at index 3.
        let floats = residual.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual(floats[3], .infinity)
    }

    // MARK: - standalone probe passes

    func testProbeF16AndF32Passes() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        let halfValues = makeHalfBuffer([1.0, Float.nan, 3.0, Float.infinity], on: context.device)
        let floatValues = makeFloatBuffer([-2.0, 4.0, 80_000.0, -Float.infinity], on: context.device)

        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        try monitor.encodeProbe(command, values: halfValues, count: 4, probe: .selfQToken)
        try monitor.encodeProbeF32(command, values: floatValues, count: 4, probe: .velocity)
        command.commit()
        command.waitUntilCompleted()

        monitor.noteStepCompleted(step: 0)
        let report = monitor.report()
        let qStats = report[.selfQToken] ?? .noIssue
        XCTAssertTrue(qStats.hasNaN)
        XCTAssertTrue(qStats.hasPosInf)
        XCTAssertEqual(qStats.firstIndex, 1)
        let velocityStats = report[.velocity] ?? .noIssue
        XCTAssertTrue(velocityStats.hasNegInf)
        XCTAssertTrue(velocityStats.hasHalfOverflow) // 80_000 exceeds fp16 range
        XCTAssertEqual(velocityStats.maxAbs, 80_000.0, accuracy: 1.0)
    }

    // MARK: - attribution ordering across probes

    func testFirstIssueUsesExecutionOrder() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        // Flag BOTH a late probe (euler) and an early probe (projection input):
        // the earliest in execution order must win.
        var raw = [UInt32](repeating: 0, count: NumericalMonitor.Probe.allCases.count * 4)
        let earlySlot = NumericalMonitor.Probe.mlpProjectionInput.rawValue
        let lateSlot = NumericalMonitor.Probe.eulerOutput.rawValue
        raw[earlySlot * 4 + 0] = NumericalMonitor.Flag.halfOverflow.rawValue
        raw[earlySlot * 4 + 2] = 10
        raw[lateSlot * 4 + 0] = NumericalMonitor.Flag.nan.rawValue
        raw[lateSlot * 4 + 2] = 5
        raw.withUnsafeBytes { bytes in
            memcpy(monitor.statsBufferForTesting, bytes.baseAddress!, bytes.count)
        }
        monitor.noteStepCompleted(step: 0)
        let issue = try XCTUnwrap(monitor.earliestIssue)
        XCTAssertEqual(issue.probe, .mlpProjectionInput)
        XCTAssertEqual(issue.stats.condition, "value exceeded FP16 finite range")
    }

    // MARK: - warningDetails magnitude evidence

    /// `warningDetails()` must include the slot's recorded maxAbs magnitude
    /// (not just the condition) for a probe with a serious flag, and must stay
    /// silent for healthy probes.
    func testWarningDetailsIncludesRecordedMaxAbs() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        var raw = [UInt32](repeating: 0, count: NumericalMonitor.Probe.allCases.count * 4)
        // Serious halfOverflow on finalResidualToHalf with a real magnitude,
        // and a magnitude recorded on a healthy slot that must NOT appear.
        let seriousSlot = NumericalMonitor.Probe.finalResidualToHalf.rawValue
        let healthySlot = NumericalMonitor.Probe.patchesToHalf.rawValue
        raw[seriousSlot * 4 + 0] = NumericalMonitor.Flag.halfOverflow.rawValue
        raw[seriousSlot * 4 + 1] = Float(103_424.0).bitPattern
        raw[seriousSlot * 4 + 2] = 7
        raw[healthySlot * 4 + 1] = Float(300_000.0).bitPattern
        raw.withUnsafeBytes { bytes in
            memcpy(monitor.statsBufferForTesting, bytes.baseAddress!, bytes.count)
        }
        let details = monitor.warningDetails()
        XCTAssertTrue(details.contains("final-layer residual conversion"),
                      "condition's stage label must be present")
        XCTAssertTrue(details.contains("value exceeded FP16 finite range"),
                      "condition string must be present")
        XCTAssertTrue(details.contains("maxAbs=103424.0"),
                      "recorded maxAbs magnitude must be present in the detail")
        XCTAssertFalse(details.contains("patchesToHalf"),
                       "healthy probes must not appear in warning details")
        XCTAssertFalse(details.contains("300000"),
                       "magnitudes of healthy probes must not be reported")
        XCTAssertEqual(monitor.warningCount(), 1)
    }

    /// A serious probe whose slot has no usable magnitude (zero) must still
    /// report its condition, without an invented maxAbs.
    func testWarningDetailsOmitsZeroMaxAbs() throws {
        let context = try requireContext()
        let monitor = try NumericalMonitor(context: context)
        monitor.beginRun()
        var raw = [UInt32](repeating: 0, count: NumericalMonitor.Probe.allCases.count * 4)
        let slot = NumericalMonitor.Probe.selfScores.rawValue
        raw[slot * 4 + 0] = NumericalMonitor.Flag.posInf.rawValue
        raw[slot * 4 + 2] = 42
        raw.withUnsafeBytes { bytes in
            memcpy(monitor.statsBufferForTesting, bytes.baseAddress!, bytes.count)
        }
        let details = monitor.warningDetails()
        XCTAssertTrue(details.contains("self-attention scores: positive Inf detected"))
        XCTAssertFalse(details.contains("maxAbs"),
                       "no maxAbs suffix when the recorded magnitude is zero")
    }

    // MARK: - Monitoring OFF (§18.5)

    /// When the numerical monitor is disabled, the sampler has no monitor; a
    /// non-finite final latent must still fail safely through the retained
    /// Euler finite guard (falling back to an un-attributed Euler failure).
    /// This proves the guard is independent of monitoring.
    func testNumericalFailureWithoutMonitorStillFailsSafely() {
        let failure = DiffusionSampler.numericalFailureForTesting(step: 5, monitor: nil)
        // A nil monitor means no attribution — the guard surfaces the
        // 1-based step via the Euler-output failure, never a crash or a
        // silently-accepted non-finite latent.
        XCTAssertEqual(failure.step, 6)
        XCTAssertTrue(failure.errorDescription?.contains("step 6") == true
                      || failure.errorDescription?.contains("6") == true)
    }

    /// The summary must NEVER print "Numerical warnings: 0" for a run where
    /// monitoring was off — it reports "not collected" instead.
    func testMetricsMonitorOffNeverReportsZeroWarnings() {
        let metrics = MetricsCollector()
        metrics.recordOptimizationConfig(Self.monitorOffConfig)
        let summary = metrics.snapshot().summaryText
        XCTAssertTrue(summary.contains("Numerical monitor: off (Euler finite guard on)"))
        XCTAssertTrue(summary.contains("Numerical warnings: not collected"))
        XCTAssertFalse(summary.contains("Numerical warnings: 0"),
                       "must not report a collected count of 0 when monitoring was off")
    }

    /// Monitor ON still records warnings and reports them as a count.
    func testMetricsMonitorOnReportsCount() {
        let metrics = MetricsCollector()
        metrics.recordOptimizationConfig(InferenceOptimizationConfig.currentBaseline)
        metrics.setNumericalWarnings(3)
        metrics.setNumericalDetails("self-attention scores: Inf detected")
        let summary = metrics.snapshot().summaryText
        XCTAssertTrue(summary.contains("Numerical monitor: on"))
        XCTAssertTrue(summary.contains("Numerical warnings: 3"))
        XCTAssertFalse(summary.contains("not collected"))
    }

    private static var monitorOffConfig: InferenceOptimizationConfig {
        var config = InferenceOptimizationConfig.currentBaseline
        config.numericalMonitoring = false
        return config
    }
}
