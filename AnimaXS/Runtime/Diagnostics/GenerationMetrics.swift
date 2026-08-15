import Foundation

/// A start/end environment fact recorded around a generation (unplugged is
/// authoritative for benchmarking). Values are observational telemetry — they
/// never gate or throttle generation.
struct EnvironmentSnapshot: Equatable {
    let powerState: String
    let batteryLevel: Int // 0...100 percent, -1 when unknown
    let thermalState: String
    let lowPowerMode: Bool

    static let unknown = EnvironmentSnapshot(
        powerState: "n/a", batteryLevel: -1, thermalState: "n/a", lowPowerMode: false)

    var batteryText: String {
        batteryLevel < 0 ? "n/a" : "\(batteryLevel)%"
    }
}

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
    /// Per-probe detail, e.g. "self-attention scores: Inf detected; MLP output: NaN detected".
    var numericalDetails: String = ""
    /// True when the monitor was DISABLED for this run (warnings were not
    /// collected). The final finite guard stays on regardless.
    var numericalMonitoringDisabled: Bool = false

    // Immutable per-run configuration snapshot (captured at Generate time).
    var optimizationConfig: InferenceOptimizationConfig?

    // Cheap executor counters (simple integer increments; no GPU readbacks).
    var linearGEMMTiles: Int = 0
    var linearDirectInputTiles: Int = 0
    var linearCopiedInputTiles: Int = 0
    var linearDirectOutputTiles: Int = 0
    var linearCopiedOutputTiles: Int = 0
    var attentionQueryTiles: Int = 0

    // Environment start/end (observational telemetry).
    var environmentStart: EnvironmentSnapshot = .unknown
    var environmentEnd: EnvironmentSnapshot = .unknown

    var averageBlockWall: Double {
        guard !blockTimes.isEmpty else { return 0 }
        return blockTimes.reduce(0, +) / Double(blockTimes.count)
    }

    /// The DiT pack variant actually used by this run.
    var ditPackVariant: DiTPackVariant {
        optimizationConfig?.ditPackVariant ?? .productionW4
    }

    /// Whether checkpointing was enabled for this run (W8 disables it).
    var checkpointingEnabled: Bool {
        optimizationConfig?.checkpointingEnabled ?? true
    }

    /// Compact text report (also the in-app post-generation summary shape).
    var summaryText: String {
        var lines: [String] = []
        lines.append(String(format: "Generation: %.1f s", totalWall))
        lines.append(String(format: "Text encode: %.1f s", textEncode))
        lines.append(String(format: "Adapter: %.1f s", adapter))
        lines.append(String(format: "DiT: %.1f s", diffusion))
        lines.append(String(format: "VAE: %.1f s", vae))
        lines.append(String(format: "Other: %.1f s", other))
        lines.append("")
        lines.append(String(format: "Average block wall time: %.2f s", averageBlockWall))
        lines.append(String(format: "Measured GPU command time: %.1f s", gpuCommandTime))
        lines.append(String(format: "Metal encode time: %.1f s", encodeTime))
        lines.append(String(format: "Weight copy/load CPU work: %.1f s, %.0f MB",
                            weightCopyTime, Double(weightCopyBytes) / 1_048_576))
        lines.append("  (may overlap GPU time when ping-pong is on)")
        lines.append(String(format: "Host/other measured time: %.1f s", hostWaitTime))
        lines.append("")
        if linearGEMMTiles > 0 || attentionQueryTiles > 0 {
            lines.append("Linear GEMM tiles: \(linearGEMMTiles)")
            lines.append("Linear input tiles: \(linearDirectInputTiles) direct / \(linearCopiedInputTiles) copied")
            lines.append("Linear output tiles: \(linearDirectOutputTiles) direct / \(linearCopiedOutputTiles) copied")
            lines.append("Attention query tiles: \(attentionQueryTiles)")
            lines.append("")
        }
        lines.append(String(format: "Peak Metal allocation: %.2f GB",
                            Double(peakMetalAllocation) / 1_073_741_824))
        if minAvailableMemory != UInt64.max {
            lines.append(String(format: "Minimum available process memory: %.0f MB (telemetry, not a budget)",
                                Double(minAvailableMemory) / 1_048_576))
        }
        if numericalMonitoringDisabled {
            lines.append("Numerical monitor: off (Euler finite guard on)")
            lines.append("Numerical warnings: not collected")
        } else {
            lines.append("Numerical monitor: on")
            lines.append("Numerical warnings: \(numericalWarnings)"
                + (numericalDetails.isEmpty ? "" : " (\(numericalDetails))"))
        }
        lines.append("")
        lines.append("Inference configuration")
        lines.append("DiT pack: \(ditPackVariant == .productionW4 ? "W4 production" : "W8 v2 experimental")")
        if let config = optimizationConfig {
            lines.append("Linear tile rows: \(config.linearTileRows)")
            lines.append("Attention tile rows: \(config.attentionTileRows)")
            lines.append("Direct MPS linear I/O: \(config.directLinearMPSIO ? "on" : "off")")
            lines.append("Ping-pong weight streaming: \(config.pingPongWeightStreaming ? "on" : "off")")
            lines.append("Numerical monitor: \(config.numericalMonitoring ? "on" : "off")")
        }
        lines.append("Checkpointing: \(checkpointingEnabled ? "on" : "off (experimental W8)")")
        lines.append("")
        lines.append("Environment")
        lines.append("Power: \(environmentStart.powerState) -> \(environmentEnd.powerState)")
        lines.append("Battery: \(environmentStart.batteryText) -> \(environmentEnd.batteryText)")
        lines.append("Thermal: \(environmentStart.thermalState) -> \(environmentEnd.thermalState)")
        lines.append("Low Power Mode: \(environmentStart.lowPowerMode ? "on" : "off") -> \(environmentEnd.lowPowerMode ? "on" : "off")")
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

    func setNumericalDetails(_ text: String) {
        metrics.numericalDetails = text
    }

    func setNumericalMonitoringDisabled(_ disabled: Bool) {
        metrics.numericalMonitoringDisabled = disabled
    }

    /// Records the immutable config snapshot actually used by the run.
    func recordOptimizationConfig(_ config: InferenceOptimizationConfig) {
        metrics.optimizationConfig = config
        metrics.numericalMonitoringDisabled = !config.numericalMonitoring
    }

    func recordEnvironmentStart(_ snapshot: EnvironmentSnapshot) {
        metrics.environmentStart = snapshot
    }

    func recordEnvironmentEnd(_ snapshot: EnvironmentSnapshot) {
        metrics.environmentEnd = snapshot
    }

    // MARK: - Executor tile counters

    func recordLinearGEMMTile(directInput: Bool, directOutput: Bool) {
        metrics.linearGEMMTiles += 1
        if directInput { metrics.linearDirectInputTiles += 1 } else { metrics.linearCopiedInputTiles += 1 }
        if directOutput { metrics.linearDirectOutputTiles += 1 } else { metrics.linearCopiedOutputTiles += 1 }
    }

    func recordAttentionQueryTile() {
        metrics.attentionQueryTiles += 1
    }

    func finalize(totalWall: Double) {
        metrics.totalWall = totalWall
        let accounted = metrics.textEncode + metrics.adapter + metrics.diffusion + metrics.vae
        metrics.other = max(0, totalWall - accounted)
    }

    func snapshot() -> GenerationMetrics { metrics }
}
