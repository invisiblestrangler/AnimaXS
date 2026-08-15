import Foundation

/// Timing + memory telemetry for one generation run (optimization runbook
/// Phase 7). The unplugged device is the authoritative benchmark; this type
/// exists so the app itself can produce it without a cable.
///
/// All wall times are `ProcessInfo.systemUptime` seconds (monotonic). GPU
/// command time comes from `MTLCommandBuffer.gpuStartTime/gpuEndTime` where
/// populated; "host wait" is the residual of block wall time after copy +
/// encode + GPU command time, so the counters reconcile with wall time.
struct GenerationMetrics: Equatable {
    // End-to-end / stages (seconds)
    var totalWall: Double = 0
    var textEncode: Double = 0
    var adapter: Double = 0
    var diffusion: Double = 0
    var vae: Double = 0
    var other: Double = 0

    // Diffusion (seconds)
    var stepTimes: [Double] = []
    var blockTimes: [Double] = []
    var blockCount: Int = 0

    // Weight streaming + Metal accounting (seconds)
    var weightCopyTime: Double = 0
    var weightCopyBytes: Int64 = 0
    var encodeTime: Double = 0
    var gpuCommandTime: Double = 0
    var hostWaitTime: Double = 0

    // Memory / thermal
    var peakMetalAllocation: UInt64 = 0
    var currentMetalAllocation: UInt64 = 0
    var minAvailableMemory: UInt64 = UInt64.max
    var thermalState: String = "n/a"

    // Numerical health
    var numericalWarnings: Int = 0

    var averageBlockWall: Double {
        guard !blockTimes.isEmpty else { return 0 }
        return blockTimes.reduce(0, +) / Double(blockTimes.count)
    }

    /// Compact text report (also the in-app post-generation summary shape).
    var summaryText: String {
        var lines: [String] = []
        lines.append(String(format: "Generation: %.1f s", totalWall))
        lines.append(String(format: "Text encode: %.1f s", textEncode))
        lines.append(String(format: "DiT: %.1f s", diffusion))
        lines.append(String(format: "VAE: %.1f s", vae))
        lines.append(String(format: "Other: %.1f s", other))
        lines.append("")
        lines.append(String(format: "Average block wall time: %.2f s", averageBlockWall))
        lines.append(String(format: "Measured GPU command time: %.1f s", gpuCommandTime))
        lines.append(String(format: "Weight copy/load time: %.1f s", weightCopyTime))
        lines.append(String(format: "Host/other measured time: %.1f s", hostWaitTime))
        lines.append("")
        lines.append(String(format: "Peak Metal allocation: %.2f GB",
                            Double(peakMetalAllocation) / 1_073_741_824))
        if minAvailableMemory != UInt64.max {
            lines.append(String(format: "Minimum available process memory: %.0f MB",
                                Double(minAvailableMemory) / 1_048_576))
        }
        lines.append("Numerical warnings: \(numericalWarnings)")
        return lines.joined(separator: "\n")
    }
}

/// Reference-type accumulator for one run's metrics. Thread-confined to the
/// engine's serial executor (created per generation in `generate`).
final class MetricsCollector {
    enum Stage {
        case textEncode, adapter, diffusion, vae, tokenizing
    }

    private(set) var metrics = GenerationMetrics()
    private var stageStart: [Stage: Double] = [:]
    private var stepStartTime: Double?
    private var blockStartTime: Double?
    private(set) var currentStep = -1
    private(set) var currentBlock = -1

    private func now() -> Double { ProcessInfo.processInfo.systemUptime }

    // MARK: - Stages

    func beginStage(_ stage: Stage) {
        stageStart[stage] = now()
    }

    func endStage(_ stage: Stage) {
        guard let start = stageStart.removeValue(forKey: stage) else { return }
        let elapsed = now() - start
        switch stage {
        case .textEncode: metrics.textEncode += elapsed
        case .adapter: metrics.adapter += elapsed
        case .diffusion: metrics.diffusion += elapsed
        case .vae: metrics.vae += elapsed
        case .tokenizing: break // folded into "other"
        }
    }

    // MARK: - Diffusion steps / blocks

    func beginStep(_ step: Int) {
        currentStep = step
        stepStartTime = now()
    }

    func endStep() {
        guard let start = stepStartTime else { return }
        metrics.stepTimes.append(now() - start)
        stepStartTime = nil
    }

    func beginBlock(_ block: Int) {
        currentBlock = block
        blockStartTime = now()
    }

    func endBlock() {
        guard let start = blockStartTime else { return }
        metrics.blockTimes.append(now() - start)
        metrics.blockCount += 1
        blockStartTime = nil
    }

    // MARK: - Metal accounting

    func recordWeightCopy(bytes: Int, seconds: Double) {
        metrics.weightCopyTime += seconds
        metrics.weightCopyBytes += Int64(bytes)
    }

    func recordEncode(seconds: Double) {
        metrics.encodeTime += seconds
    }

    func recordGPUCommand(seconds: Double) {
        metrics.gpuCommandTime += seconds
    }

    func recordHostWait(seconds: Double) {
        metrics.hostWaitTime += max(0, seconds)
    }

    // MARK: - Memory / thermal

    func recordMemory(allocated: UInt64, available: UInt64) {
        metrics.peakMetalAllocation = max(metrics.peakMetalAllocation, allocated)
        metrics.currentMetalAllocation = allocated
        if available < metrics.minAvailableMemory { metrics.minAvailableMemory = available }
    }

    func recordThermal(_ state: ProcessInfo.ThermalState) {
        metrics.thermalState = String(describing: state)
    }

    func setNumericalWarnings(_ count: Int) {
        metrics.numericalWarnings = count
    }

    func finalize(totalWall: Double) {
        metrics.totalWall = totalWall
        let accounted = metrics.textEncode + metrics.adapter + metrics.diffusion + metrics.vae
        metrics.other = max(0, totalWall - accounted)
    }

    func snapshot() -> GenerationMetrics { metrics }
}
