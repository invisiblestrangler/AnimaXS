import SwiftUI
import UniformTypeIdentifiers
import UIKit
import Dispatch
#if canImport(Darwin)
import Darwin
#endif

/// K005 — diagnostics screen. Runs the shared `DiagnosticsEngine` self-tests
/// and exports the structured report as JSON.
///
/// Design (real-device stabilization):
/// - Opening the screen runs ONLY the cheap snapshot (device facts + model
///   presence/size/receipt state). No auto-running of heavy tests.
/// - Explicit buttons run: basic self-tests, hardware (Metal/MPS) tests, and
///   deep model SHA-256 — each level independently, with visible per-test
///   progress.
/// - A persistent `DiagnosticRunMarker` localizes a native crash to the exact
///   test that was running ("Previous diagnostic run ended unexpectedly
///   while: …").
/// - Export/share serializes the already-built report — zero test execution.
struct DiagnosticsView: View {
    @State private var snapshot: DiagnosticsReport?
    @State private var basicItems: [DiagnosticItem] = []
    @State private var hardwareItems: [DiagnosticItem] = []
    @State private var deepItems: [DiagnosticItem] = []
    @State private var isRunning = false
    @State private var currentTest: String?
    @State private var previousRunWarning: String?
    @State private var exportError: String?
    @State private var exportPresented = false
    @State private var aneProbeText: String?
    @State private var jsonURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("anima-xs-diagnostics.json")

    /// Phase 8 — telemetry summary of the most recent generation, passed in
    /// from the coordinator so it is visible without a cable.
    var lastMetricsText: String?

    /// Runtime inference-optimization settings (persistent, Diagnostics-only).
    @ObservedObject var optimizationSettings: InferenceOptimizationSettings

    /// Task 9: the resolved DiT pack's numerics policy (defaults to the W4
    /// policy when the pack is not resolved yet). Used to evaluate the
    /// central compatibility validator for the current configuration.
    var ditNumericsPolicy: DiTNumericsPolicy
    /// Resolved DiT variant id, when available, so pack/backend compatibility
    /// uses the same central validator as the Generate screen.
    var ditVariantID: String?

    /// Whether a generation is currently active (controls are disabled while
    /// it runs so a toggle can never mutate an in-flight run).
    var isGenerating: Bool

    private let marker = DiagnosticRunMarker()

    init(
        lastMetricsText: String? = nil,
        optimizationSettings: InferenceOptimizationSettings,
        ditNumericsPolicy: DiTNumericsPolicy = .w4Legacy,
        ditVariantID: String? = nil,
        isGenerating: Bool = false
    ) {
        self.lastMetricsText = lastMetricsText
        self.optimizationSettings = optimizationSettings
        self.ditNumericsPolicy = ditNumericsPolicy
        self.ditVariantID = ditVariantID
        self.isGenerating = isGenerating
    }

    var body: some View {
        Form {
            previousRunSection
            deviceSection
            metricsSection
            performanceSection
            modelSection
            basicTestsSection
            hardwareTestsSection
            aneResidencyProbeSection
            deepIntegritySection
            actionsSection
        }
        .navigationTitle("Diagnostics")
        .task {
            await loadSnapshot()
        }
        .fileExporter(
            isPresented: $exportPresented,
            document: JSONFile(url: jsonURL),
            contentType: .json,
            defaultFilename: "anima-xs-diagnostics.json"
        ) { result in
            if case .failure(let error) = result {
                exportError = error.localizedDescription
            }
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var previousRunSection: some View {
        if let warning = previousRunWarning {
            Section("Previous run") {
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                Text("The app may have been killed by a native assertion or memory pressure. Re-run the tests to confirm.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var deviceSection: some View {
        Section("Device & app") {
            if let snapshot {
                LabeledContent("App version", value: snapshot.appVersion)
                LabeledContent("OS", value: snapshot.osVersion)
                LabeledContent("Device", value: snapshot.deviceModel)
                LabeledContent("Memory", value: "\(snapshot.physicalMemoryBytes / 1_048_576) MB")
                LabeledContent("Process memory available",
                               value: snapshot.availableProcessMemoryBytes.map {
                                   String(format: "%.1f MB", Double($0) / 1_048_576)
                               } ?? "n/a")
                LabeledContent("Metal", value: snapshot.metalAvailable ? snapshot.metalDeviceName : "unavailable")
                LabeledContent("Disk free", value: "\(snapshot.availableDiskBytes / 1_048_576) MB")
                LabeledContent("Thermal", value: snapshot.thermalState)
            } else {
                ProgressView("Loading snapshot…")
            }
        }
    }

    @ViewBuilder
    private var metricsSection: some View {
        if let text = lastMetricsText {
            Section("Last generation metrics") {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                ShareLink(item: text) {
                    Label("Copy / share metrics", systemImage: "square.and.arrow.up")
                }
                .font(.caption)
            }
        }
    }

    /// Runtime inference-performance experiments (one build, runtime choice).
    /// Controls are disabled during an active generation so a toggle can never
    /// mutate the in-flight run; the captured run configuration is shown in
    /// the post-run metrics summary instead.
    @ViewBuilder
    private var performanceSection: some View {
        Section("Inference performance experiments") {
            Text("Applies to the next fresh generation. Each Generate starts from step 0 for a clean performance comparison.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("Preset (P9)", selection: presetBinding) {
                // Task 9: when the user has manually adjusted any individual
                // control the active-preset marker is cleared, and the picker
                // shows a "Custom" row instead of pretending a named preset
                // (e.g. Baseline) is active. Picking a named preset applies
                // it and re-establishes the marker.
                Text("Custom").tag(nil as InferencePreset?)
                ForEach(InferencePreset.allCases, id: \.self) { preset in
                    // QUARANTINED (Task 4) + DISABLED (Task 5): presets whose
                    // QGEMM part is quarantined, or whose P6 mmap no-copy part
                    // is disabled, remain selectable but are shown as disabled
                    // with a visible explanation. Their makeConfig() forces
                    // the affected components back to the known-good values,
                    // so applying one can never run the 10x-slower direct
                    // path or the no-copy weight path — and the disabled look
                    // makes it impossible to miss.
                    Text(preset.label).tag(preset as InferencePreset?)
                        .disabled(preset.containsQuarantinedLinearBackend
                                  || preset.containsDisabledNoCopy)
                }
            }
            .disabled(isGenerating)
            // Task 9: the central validator's verdict — the SAME reason text
            // that blocks Generate. Shown whenever the current configuration
            // is incompatible (defense-in-depth; the config is never silently
            // mutated here).
            if let reason = optimizationBlockingReason {
                Label {
                    Text(reason)
                        .font(.caption2)
                } icon: {
                    Image(systemName: "xmark.octagon.fill")
                        .foregroundStyle(.red)
                }
                .foregroundStyle(.secondary)
            }
            if optimizationSettings.activePreset?.containsQuarantinedLinearBackend == true {
                quarantineNote
            }
            if optimizationSettings.activePreset?.containsDisabledNoCopy == true {
                p6NoCopyNote
            }
            Text("A preset sets every control below at once. Adjusting any individual control below switches the preset section to Custom — the preset combination no longer exactly matches. Nothing here is claimed fastest until the physical XS Max is measured.")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Picker("Linear tile rows", selection: linearTileSelection) {
                ForEach(InferenceOptimizationConfig.allowedTileRows, id: \.self) { rows in
                    Text("\(rows)").tag(rows)
                }
            }
            .disabled(isGenerating)
            Picker("Attention tile rows", selection: attentionTileSelection) {
                ForEach(InferenceOptimizationConfig.allowedTileRows, id: \.self) { rows in
                    Text("\(rows)").tag(rows)
                }
            }
            .disabled(isGenerating)
            Toggle("Direct MPS linear I/O", isOn: directLinearBinding)
                .disabled(isGenerating)
            Toggle("Ping-pong weight streaming", isOn: pingPongBinding)
                .disabled(isGenerating)
            Toggle("Numerical monitor", isOn: numericalMonitorBinding)
                .disabled(isGenerating)
            Toggle("Fused LayerNorm+AdaLN+to-half", isOn: fusedNormModulationBinding)
                .disabled(isGenerating)
            Toggle("Fused MLP in-place GELU", isOn: fusedMLPActivationBinding)
                .disabled(isGenerating)
            Toggle("Strided token-major attention (P4)", isOn: stridedTokenMajorAttentionBinding)
                .disabled(isGenerating)
            Toggle("Cross-attention K/V cache (P5)", isOn: crossKVCacheBinding)
                .disabled(isGenerating)
            Toggle("Mmap no-copy weight source (P6, experimental)", isOn: noCopyWeightSourceBinding)
                // DISABLED (Task 5): permanently locked OFF for normal
                // device settings after a physical A12 run hit a real GPU
                // page fault (kIOGPUCommandBufferCallbackErrorPageFault)
                // while no-copy bytes were being served. The setting is
                // normalized to false and never persisted; the research
                // implementation stays intact for an isolated hardware test.
                .disabled(true)
            p6NoCopyNote
            Picker("DiT attention backend (P7)", selection: attentionBackendBinding) {
                ForEach(DiTAttentionBackend.allCases, id: \.self) { backend in
                    Text(backend.rawValue).tag(backend)
                }
            }
            .disabled(isGenerating)
            Picker("DiT linear backend", selection: linearBackendBinding) {
                // QUARANTINED (Task 4): `.directQuantized` and `.hybrid` are
                // filtered out of the normal device picker entirely — they
                // measured ~10x slower than dequantized MPS on the A12 device
                // and must never be selectable in production/device settings.
                // The P8 research kernel stays intact and testable via
                // LinearExecutor for research/diagnostic tests.
                ForEach(DiTLinearBackend.allCases.filter { !$0.isQuarantined }, id: \.self) { backend in
                    Text(backend.rawValue).tag(backend)
                }
            }
            .disabled(isGenerating)
            if optimizationSettings.linearBackend == .aneHybridW8 {
                aneHybridNote
            }
            quarantineNote
            Button("Reset to current baseline") {
                optimizationSettings.resetToBaseline()
            }
            .disabled(isGenerating)
            if isGenerating {
                Text("Controls are disabled while a generation is running.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    /// A12/H11 backend note. This path intentionally remains opt-in: it uses
    /// the device-proven private ANE runtime, accepts W8 DiT packs only, and
    /// offloads the large projection GEMMs while keeping nonlinear/attention
    /// math on Metal.
    private var aneHybridNote: some View {
        Label {
            Text("ANE hybrid (A12/H11): W8-only experimental device backend. Self/cross-attention projection GEMMs and MLP1/MLP2 run on ANE; AdaLN, RMSNorm, RoPE, attention, GELU and residual math stay on Metal. Use for sideload/device testing, not App Store distribution.")
                .font(.caption2)
        } icon: {
            Image(systemName: "cpu")
                .foregroundStyle(.secondary)
        }
        .foregroundStyle(.secondary)
    }

    /// QUARANTINED (Task 4): visible explanation shown whenever the P8 direct
    /// QGEMM backends (or the presets containing them) are hidden/disabled in
    /// the UI. Wording states it is a measured A12 PERFORMANCE regression
    /// (~10x slower than dequantized MPS), NOT a proven correctness failure —
    /// the research kernel stays intact and testable via `LinearExecutor`.
    private var quarantineNote: some View {
        Label {
            Text(InferenceOptimizationSettings.quarantineReason)
                .font(.caption2)
        } icon: {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
        }
        .foregroundStyle(.secondary)
    }

    /// DISABLED (Task 5): visible explanation shown whenever the P6 mmap
    /// no-copy weight source (or the presets containing it) is disabled in
    /// the UI. Wording states it is blocked after a real A12 GPU page fault
    /// (kIOGPUCommandBufferCallbackErrorPageFault) while no-copy bytes were
    /// being served — correctness/safety hardening, NOT a proof of the
    /// historical root cause; the research implementation stays intact.
    private var p6NoCopyNote: some View {
        Label {
            Text(InferenceOptimizationSettings.p6NoCopyDisabledReason)
                .font(.caption2)
        } icon: {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
        }
        .foregroundStyle(.secondary)
    }

    /// P9: binding for the preset picker. `activePreset` is `nil` when the
    /// user refines an individual control (or after reset), and the picker
    /// shows the "Custom" row (nil tag) — it never pretends a named preset is
    /// active. Picking always applies the chosen preset via `setPreset`.
    private var presetBinding: Binding<InferencePreset?> {
        Binding(
            get: { optimizationSettings.activePreset },
            set: { newValue in
                if let newValue {
                    optimizationSettings.setPreset(newValue)
                }
                // Selecting "Custom" (nil) keeps the current controls exactly
                // as they are — it only clears the marker.
            })
    }

    /// Task 9: the central compatibility validator's verdict for the CURRENT
    /// configuration. Visible whenever the current config is blocked (even
    /// while a named preset is active), so the user sees the exact reason
    /// Generate is disabled — with the same wording as the Generate screen.
    private var optimizationBlockingReason: String? {
        InferenceOptimizationConfig.blockingReason(
            for: optimizationSettings.snapshot,
            numerics: ditNumericsPolicy,
            ditVariantID: ditVariantID)
    }

    private var linearTileSelection: Binding<Int> {
        Binding(
            get: { optimizationSettings.linearTileRows },
            set: { optimizationSettings.setLinearTileRows($0) })
    }

    private var attentionTileSelection: Binding<Int> {
        Binding(
            get: { optimizationSettings.attentionTileRows },
            set: { optimizationSettings.setAttentionTileRows($0) })
    }

    private var directLinearBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.directLinearMPSIO },
            set: { optimizationSettings.setDirectLinearMPSIO($0) })
    }

    private var pingPongBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.pingPongWeightStreaming },
            set: { optimizationSettings.setPingPongWeightStreaming($0) })
    }

    private var numericalMonitorBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.numericalMonitoring },
            set: { optimizationSettings.setNumericalMonitoring($0) })
    }

    private var fusedNormModulationBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.fusedNormModulation },
            set: { optimizationSettings.setFusedNormModulation($0) })
    }

    private var fusedMLPActivationBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.fusedMLPActivation },
            set: { optimizationSettings.setFusedMLPActivation($0) })
    }

    private var stridedTokenMajorAttentionBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.stridedTokenMajorAttention },
            set: { optimizationSettings.setStridedTokenMajorAttention($0) })
    }

    private var crossKVCacheBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.crossKVCache },
            set: { optimizationSettings.setCrossKVCache($0) })
    }

    private var noCopyWeightSourceBinding: Binding<Bool> {
        Binding(
            get: { optimizationSettings.noCopyWeightSource },
            set: { optimizationSettings.setNoCopyWeightSource($0) })
    }

    private var attentionBackendBinding: Binding<DiTAttentionBackend> {
        Binding(
            get: { optimizationSettings.attentionBackend },
            set: { optimizationSettings.setAttentionBackend($0) })
    }

    private var linearBackendBinding: Binding<DiTLinearBackend> {
        Binding(
            get: { optimizationSettings.linearBackend },
            set: { optimizationSettings.setLinearBackend($0) })
    }

    @ViewBuilder
    private var modelSection: some View {
        Section("Model packs") {
            if let snapshot {
                ForEach(snapshot.modelPacks, id: \.filename) { pack in
                    VStack(alignment: .leading, spacing: 2) {
                        LabeledContent(
                            pack.filename,
                            value: modelStateLabel(pack))
                        Text(modelStateDetail(pack))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var basicTestsSection: some View {
        Section("Self-tests") {
            Text("Deterministic CPU tests (W4/W8 decode, RNG, small file read, Metal capability facts). No model hashing.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(isRunning && currentTest != nil ? "Running \(currentTest ?? "tests")…" : "Run basic self-tests") {
                Task { await runBasic() }
            }
            .disabled(isRunning)
            ForEach(basicItems, id: \.name) { item in
                testRow(item)
            }
        }
    }

    private var hardwareTestsSection: some View {
        Section("Hardware tests") {
            Text("Real Metal/MPS command-buffer probes. Runs sequentially. A native failure here can crash the app — the current test is recorded so the next launch can say which one.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(isRunning && currentTest != nil ? "Running \(currentTest ?? "tests")…" : "Run hardware tests") {
                Task { await runHardware() }
            }
            .disabled(isRunning)
            ForEach(hardwareItems, id: \.name) { item in
                testRow(item)
            }
        }
    }

    @ViewBuilder
    private var aneResidencyProbeSection: some View {
        Section("ANE residency / load probe") {
            Text("Device-only ANE streaming-ring probe v3. Tests the production candidate with zero pinned blocks and only two block slices resident at once (8 current + 8 prefetched programs), including a full 28-block warm rotation and sustained load/evaluate overlap. It never runs automatically or modifies model weights.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(isRunning && currentTest == "ANE residency/load probe"
                   ? "Running ANE ring probe…" : "Run ANE streaming-ring probe") {
                Task { await runANEProbe() }
            }
            .disabled(isRunning || isGenerating)
            if isGenerating {
                Text("Finish or cancel the active generation before running the ANE probe.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if let aneProbeText {
                Text(aneProbeText)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                ShareLink(item: aneProbeText) {
                    Label("Copy / share ANE probe", systemImage: "square.and.arrow.up")
                }
                .font(.caption)
            }
        }
    }

    private var deepIntegritySection: some View {
        Section("Deep model integrity") {
            Text("Verifies every installed pack with a full SHA-256 pass — reads about 2.07 GB and may take a while / warm the device. Never runs automatically.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(isRunning && currentTest != nil ? "Running \(currentTest ?? "tests")…" : "Verify all model SHA-256") {
                Task { await runDeep() }
            }
            .disabled(isRunning)
            ForEach(deepItems, id: \.name) { item in
                testRow(item)
            }
        }
    }

    private var actionsSection: some View {
        Section {
            Button(isRunning ? "Running diagnostics…" : "Run diagnostics") {
                Task { await runAll() }
            }
            .disabled(isRunning)
            if isRunning, let currentTest {
                ProgressView("Running \(currentTest)…")
            }
            if snapshot != nil {
                Button("Export JSON") {
                    exportJSON()
                }
                ShareLink(item: jsonURL) {
                    Label("Share JSON report", systemImage: "square.and.arrow.up")
                }
            }
            if let exportError {
                Text(exportError).foregroundStyle(.red).font(.caption)
            }
        }
    }

    private func testRow(_ item: DiagnosticItem) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Image(systemName: statusSymbol(item.status))
                .foregroundStyle(statusColor(item.status))
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name).font(.subheadline)
                Text(item.detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Actions

    private func loadSnapshot() async {
        let engine = DiagnosticsEngine()
        snapshot = await engine.snapshot()
        previousRunWarning = marker.unfinishedTest().map {
            "Previous diagnostic run ended unexpectedly while: \($0)."
        }
    }

    private func runBasic() async {
        isRunning = true
        currentTest = "basic self-tests"
        basicItems = DiagnosticsEngine().basicSelfTests()
        isRunning = false
        currentTest = nil
    }

    private func runHardware() async {
        isRunning = true
        marker.beginSession()
        let engine = DiagnosticsEngine()
        hardwareItems = await engine.hardwareTests(
            marker: marker,
            progress: { test in
                Task { @MainActor in self.currentTest = test }
            })
        marker.markSessionClean()
        isRunning = false
        currentTest = nil
    }

    private func runANEProbe() async {
        guard !isRunning, !isGenerating else { return }
        isRunning = true
        currentTest = "ANE residency/load probe"
        aneProbeText = "Running…"
        marker.beginSession()
        marker.markStarted("ANE residency/load probe")
        let result = await Task.detached(priority: .userInitiated) {
            await ANERingProbe.run()
        }.value
        aneProbeText = result
        marker.markCompleted("ANE residency/load probe")
        marker.markSessionClean()
        isRunning = false
        currentTest = nil
    }

    private func runDeep() async {
        isRunning = true
        marker.beginSession()
        let engine = DiagnosticsEngine()
        deepItems = await engine.deepIntegrity(
            marker: marker,
            progress: { test in
                Task { @MainActor in self.currentTest = test }
            })
        marker.markSessionClean()
        isRunning = false
        currentTest = nil
    }

    private func runAll() async {
        isRunning = true
        marker.beginSession()
        let engine = DiagnosticsEngine()
        basicItems = engine.basicSelfTests()
        hardwareItems = await engine.hardwareTests(
            marker: marker,
            progress: { test in
                Task { @MainActor in self.currentTest = test }
            })
        marker.markSessionClean()
        isRunning = false
        currentTest = nil
    }

    /// Serializes the current report WITHOUT running any test.
    private func exportJSON() {
        guard let report = exportableReport else { return }
        do {
            try DiagnosticsEngine().writeJSON(report, to: jsonURL)
            exportPresented = true
        } catch {
            exportError = error.localizedDescription
        }
    }

    /// Snapshot + all results collected so far. Deep-integrity passes are
    /// folded back into the per-pack `sha256Verified` flags.
    private var exportableReport: DiagnosticsReport? {
        guard var report = snapshot else { return nil }
        report.selfTests = basicItems + hardwareItems + deepItems
        report.modelPacks = report.modelPacks.map { pack in
            var updated = pack
            updated.sha256Verified = deepItems.contains {
                $0.name == "Deep SHA: \(pack.filename)" && $0.status == .pass
            }
            return updated
        }
        return report
    }

    // MARK: - Label helpers

    private func modelStateLabel(_ pack: DiagnosticsReport.ModelPackInfo) -> String {
        guard pack.installed else { return "missing" }
        if pack.sha256Verified { return "verified (deep SHA)" }
        return pack.verified ? "verified (receipt)" : "installed, unverified"
    }

    private func modelStateDetail(_ pack: DiagnosticsReport.ModelPackInfo) -> String {
        guard pack.installed else {
            return "expected \(pack.expectedSize) bytes — not present"
        }
        return "\(pack.sizeBytes) bytes (expected \(pack.expectedSize))"
    }

    private func statusSymbol(_ status: DiagnosticStatus) -> String {
        switch status {
        case .pass: return "checkmark.circle.fill"
        case .fail: return "xmark.circle.fill"
        case .skipped: return "minus.circle"
        }
    }

    private func statusColor(_ status: DiagnosticStatus) -> Color {
        switch status {
        case .pass: return .green
        case .fail: return .red
        case .skipped: return .gray
        }
    }
}

/// One-build physical-device probe for private ANE model loading and
/// residency. It intentionally lives in DiagnosticsView.swift so adding the
/// research probe does not alter project/source membership or production
/// generation behavior.
private enum ANEResidencyProbe {
    private static let mib = 1_048_576.0

    private struct MemorySnapshot {
        let available: UInt64
        let footprint: UInt64?
        let metal: UInt64
        let thermal: String

        static func capture(context: MetalContext) -> MemorySnapshot {
            context.refreshDiagnostics()
            return MemorySnapshot(
                available: UInt64(os_proc_available_memory()),
                footprint: processPhysicalFootprint(),
                metal: context.currentAllocatedSize,
                thermal: String(describing: ProcessInfo.processInfo.thermalState))
        }

        var compact: String {
            let footprintText = footprint.map { mb($0) } ?? "n/a"
            return "avail=\(mb(available)) footprint=\(footprintText) Metal=\(mb(metal)) thermal=\(thermal)"
        }
    }

    private final class PressureRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private let source: DispatchSourceMemoryPressure
        private var observer: NSObjectProtocol?
        private var dispatchEvents: [String] = []
        private var uiWarningCount = 0
        private var issueSignals = 0
        private var didSeeCritical = false
        private var stopped = false

        init() {
            source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.normal, .warning, .critical],
                queue: DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-pressure-probe"))
            source.setEventHandler { [weak self] in
                guard let self else { return }
                let event = self.source.data
                var labels: [String] = []
                if event.contains(.normal) { labels.append("normal") }
                if event.contains(.warning) { labels.append("warning") }
                if event.contains(.critical) { labels.append("critical") }
                let stamp = String(format: "%.3f", ProcessInfo.processInfo.systemUptime)
                self.lock.lock()
                self.dispatchEvents.append("\(stamp):\(labels.joined(separator: "+"))")
                if event.contains(.warning) || event.contains(.critical) { self.issueSignals += 1 }
                if event.contains(.critical) { self.didSeeCritical = true }
                self.lock.unlock()
            }
            observer = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil, queue: nil
            ) { [weak self] _ in
                guard let self else { return }
                self.lock.lock()
                self.uiWarningCount += 1
                self.issueSignals += 1
                self.lock.unlock()
            }
            source.resume()
        }

        var issueCount: Int {
            lock.lock(); defer { lock.unlock() }
            return issueSignals
        }

        var criticalSeen: Bool {
            lock.lock(); defer { lock.unlock() }
            return didSeeCritical
        }

        var compact: String {
            lock.lock(); defer { lock.unlock() }
            let events = dispatchEvents.isEmpty ? "none" : dispatchEvents.joined(separator: ",")
            return "UIKitWarnings=\(uiWarningCount) issueSignals=\(issueSignals) dispatch=\(events)"
        }

        func stop() {
            lock.lock()
            if stopped { lock.unlock(); return }
            stopped = true
            let token = observer
            observer = nil
            lock.unlock()
            source.cancel()
            if let token { NotificationCenter.default.removeObserver(token) }
        }
    }

    private enum ProgramKind {
        case qkv(cacheKey: String)
        case projection(
            cacheKey: String, suffix: String, tag: String,
            inputChannels: Int, outputChannels: Int, spatial: Int)
    }

    private struct ProgramSpec {
        let block: Int
        let label: String
        let logicalBytes: UInt64
        let kind: ProgramKind
    }

    private enum LoadedProgram {
        case qkv(A12ANEQKVModel)
        case projection(A12ANEProjectionModel)

        var loadMilliseconds: Double {
            switch self {
            case .qkv(let model): return model.loadMilliseconds
            case .projection(let model): return model.loadMilliseconds
            }
        }
    }

    private struct SequenceOutcome {
        let programs: Int
        let logicalBytes: UInt64
        let onsetDetected: Bool
        let baselineMedianMS: Double
        let finalLoadMS: Double
    }

    private final class ModelsBox: @unchecked Sendable {
        let models: ANEW8DiTModels
        init(_ models: ANEW8DiTModels) { self.models = models }
    }

    private final class OverlapResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Result<(aneMS: Double, wallMS: Double), Error>?

        func set(_ result: Result<(aneMS: Double, wallMS: Double), Error>) {
            lock.lock(); stored = result; lock.unlock()
        }

        func get() -> Result<(aneMS: Double, wallMS: Double), Error>? {
            lock.lock(); defer { lock.unlock() }
            return stored
        }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE residency/load probe v2",
            "Safety: high-pressure probes stop on the first pathological single-load or pressure signal; they do not intentionally march through the old helper-failure zone.",
            "Questions: count-vs-bytes? same-object reload? reconstructed warm reload? unload reclamation? load/eval overlap?"
        ]
        let pressure = PressureRecorder()
        defer { pressure.stop() }

        do {
            guard A12ANEIsAvailable() else {
                throw AnimapkError.validation("ANE unavailable: \(A12ANERuntimeStatus())")
            }
            guard let context = MetalContext() else {
                throw AnimapkError.validation("Metal unavailable")
            }
            guard let ditEntry = ModelManifest.entries.first(where: { $0.component == .dit }) else {
                throw AnimapkError.validation("DiT manifest entry missing")
            }
            let store = try ModelStore()
            let ditURL = await store.localURL(for: ditEntry)
            guard FileManager.default.fileExists(atPath: ditURL.path) else {
                throw AnimapkError.validation("installed DiT pack missing")
            }
            let file = try AnimapkFile(url: ditURL)
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                throw AnimapkError.validation("ANE prepared-model cache is incomplete")
            }

            let surfaces = try ANEW8DiTSurfaces(device: context.device)
            let specs = try makeProgramSpecs(file: file)

            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a") programs=\(specs.count)")
            lines.append("baseline: \(MemorySnapshot.capture(context: context).compact)")
            lines.append("")

            lines.append("EXPERIMENT 1 — same-object reload vs reconstructed warm reload")
            let reloadBaseline = try reloadMicrobenchmark(
                file: file, surfaces: surfaces, context: context, pressure: pressure, lines: &lines)
            lines.append("")

            lines.append("EXPERIMENT 2 — load/evaluate overlap (safe 16-program working set)")
            try overlapExperiment(
                file: file, surfaces: surfaces, context: context, pressure: pressure, lines: &lines)
            lines.append("")
            try? await Task.sleep(nanoseconds: 750_000_000)

            let preCountSentinel = try freshProjectionSentinel(file: file)
            lines.append(String(
                format: "pre-count recovery sentinel fresh selfO load=%6.1fms (reload-micro fresh median=%6.1fms) %@ | %@",
                preCountSentinel, reloadBaseline, MemorySnapshot.capture(context: context).compact, pressure.compact))
            if preCountSentinel > max(300.0, reloadBaseline * 4.0) || pressure.criticalSeen {
                lines.append("STOP: runtime did not recover cleanly enough for high-residency experiments.")
                return lines.joined(separator: "\n")
            }
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("EXPERIMENT 3A — count-heavy residency")
            lines.append("Order: smallest prepared programs first. Caps: 104 programs OR 600MB logical. Stop immediately on first pathological single load / new pressure signal.")
            let countOutcome = try residencySequence(
                specs: specs.sorted {
                    if $0.logicalBytes == $1.logicalBytes { return $0.label < $1.label }
                    return $0.logicalBytes < $1.logicalBytes
                },
                maxPrograms: 104,
                maxLogicalBytes: 600 * 1_048_576,
                surfaces: surfaces, context: context, pressure: pressure,
                lines: &lines)
            lines.append(String(
                format: "count-heavy result: programs=%d logical=%.0fMB onset=%@ baselineSingleLoad=%.1fms finalLoad=%.1fms",
                countOutcome.programs, Double(countOutcome.logicalBytes) / mib,
                countOutcome.onsetDetected ? "YES" : "no",
                countOutcome.baselineMedianMS, countOutcome.finalLoadMS))
            try? await Task.sleep(nanoseconds: 1_000_000_000)

            let postCountSentinel = try freshProjectionSentinel(file: file)
            lines.append(String(
                format: "post-count recovery sentinel fresh selfO load=%6.1fms %@ | %@",
                postCountSentinel, MemorySnapshot.capture(context: context).compact, pressure.compact))
            if postCountSentinel > max(300.0, reloadBaseline * 4.0) || pressure.criticalSeen {
                lines.append("STOP: count-heavy probe did not reclaim cleanly; byte-heavy probe skipped to protect ANE helper.")
                return lines.joined(separator: "\n")
            }
            try? await Task.sleep(nanoseconds: 750_000_000)

            lines.append("")
            lines.append("EXPERIMENT 3B — byte-heavy residency")
            lines.append("Order: largest prepared programs first. Caps: 48 programs OR 740MB logical. Stop immediately on first pathological single load / new pressure signal.")
            let byteOutcome = try residencySequence(
                specs: specs.sorted {
                    if $0.logicalBytes == $1.logicalBytes { return $0.label < $1.label }
                    return $0.logicalBytes > $1.logicalBytes
                },
                maxPrograms: 48,
                maxLogicalBytes: 740 * 1_048_576,
                surfaces: surfaces, context: context, pressure: pressure,
                lines: &lines)
            lines.append(String(
                format: "byte-heavy result: programs=%d logical=%.0fMB onset=%@ baselineSingleLoad=%.1fms finalLoad=%.1fms",
                byteOutcome.programs, Double(byteOutcome.logicalBytes) / mib,
                byteOutcome.onsetDetected ? "YES" : "no",
                byteOutcome.baselineMedianMS, byteOutcome.finalLoadMS))
            try? await Task.sleep(nanoseconds: 1_000_000_000)

            let finalSentinel = try freshProjectionSentinel(file: file)
            lines.append(String(
                format: "final recovery sentinel fresh selfO load=%6.1fms %@ | %@",
                finalSentinel, MemorySnapshot.capture(context: context).compact, pressure.compact))
            lines.append("")
            lines.append("INTERPRETATION GUIDE")
            lines.append("- 3A onset at high count but well below ~662MB => count/descriptor/VA-entry pressure is favored.")
            lines.append("- 3B onset near old logical-byte boundary with far fewer programs => byte/mapping pressure is favored.")
            lines.append("- Both => combined resource/fragmentation/scratch pressure; multi-procedure fusion still deserves a POC.")
            lines.append("- Neither => old 8-program/block shape/order or delayed reclamation matters; use the exact onset records + sentinels.")
            lines.append("- Experiment 1 near-equal same-object/reconstructed reload => cost lives mainly inside _ANEClient loadModel, not filesystem/modelAtURL reconstruction.")
            lines.append("- Experiment 2 overlapWall materially below sequentialWall => pinned + streaming can hide some loader latency; otherwise assume serialization.")
            lines.append("")
            lines.append("final pressure: \(pressure.compact)")
        } catch {
            lines.append("ERROR: \(error.localizedDescription)")
            lines.append("pressure at error: \(pressure.compact)")
        }
        return lines.joined(separator: "\n")
    }

    private static func reloadMicrobenchmark(
        file: AnimapkFile, surfaces: ANEW8DiTSurfaces, context: MetalContext,
        pressure: PressureRecorder, lines: inout [String]
    ) throws -> Double {
        let spec = try projectionSpec(
            file: file, block: 0, suffix: "self_attn.output_proj.weight")
        var model: A12ANEProjectionModel? = try makeProjection(spec)
        guard let first = model else {
            throw AnimapkError.validation("reload microbenchmark failed to construct first selfO")
        }
        var firstEval = 0.0
        _ = try first.evaluateInput(
            surfaces.tokenInput, output: surfaces.tokenOutput, milliseconds: &firstEval)
        lines.append(String(
            format: "fresh#0 loadANE=%6.1fms eval=%5.1fms %@ | %@",
            first.loadMilliseconds, firstEval,
            MemorySnapshot.capture(context: context).compact, pressure.compact))

        var sameObjectLoads: [Double] = []
        for cycle in 1...3 {
            guard first.diagnosticUnloadKeepingModel() else { throw AnimapkError.validation("same-object diagnostic unload failed") }
            let wallStart = ProcessInfo.processInfo.systemUptime
            let reloadMS = first.diagnosticReloadMilliseconds()
            guard reloadMS >= 0 else {
                throw AnimapkError.validation("same-object diagnostic reload failed")
            }
            let wallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
            sameObjectLoads.append(reloadMS)
            lines.append(String(
                format: "same-object#%d loadANE=%6.1fms loadWall=%6.1fms",
                cycle, reloadMS, wallMS))
        }
        first.invalidate()
        model = nil

        var reconstructedLoads: [Double] = []
        for cycle in 1...3 {
            let wallStart = ProcessInfo.processInfo.systemUptime
            var rebuilt: A12ANEProjectionModel? = try makeProjection(spec)
            let wallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
            guard let value = rebuilt else {
                throw AnimapkError.validation("reconstructed reload returned nil")
            }
            reconstructedLoads.append(value.loadMilliseconds)
            lines.append(String(
                format: "reconstructed#%d loadANE=%6.1fms loadWall=%6.1fms",
                cycle, value.loadMilliseconds, wallMS))
            value.invalidate()
            rebuilt = nil
        }

        let sameMedian = median(sameObjectLoads)
        let reconstructedMedian = median(reconstructedLoads)
        lines.append(String(
            format: "reload medians: same-object=%.1fms reconstructed=%.1fms delta=%.1fms",
            sameMedian, reconstructedMedian, reconstructedMedian - sameMedian))
        return reconstructedMedian
    }

    private static func overlapExperiment(
        file: AnimapkFile, surfaces: ANEW8DiTSurfaces, context: MetalContext,
        pressure: PressureRecorder, lines: inout [String]
    ) throws {
        var cacheA: ANEW8DiTModelCache? = try ANEW8DiTModelCache(file: file)
        var cacheB: ANEW8DiTModelCache? = try ANEW8DiTModelCache(file: file)
        guard let cacheA, let cacheB else {
            throw AnimapkError.validation("overlap experiment cache construction failed")
        }
        let a = try cacheA.models(for: 0).models
        let b = try cacheB.models(for: 1).models
        let boxB = ModelsBox(b)
        try unloadKeepingModels(b)

        let evalRepeats = 4
        let evalStart = ProcessInfo.processInfo.systemUptime
        var evalANE = 0.0
        for _ in 0..<evalRepeats { evalANE += try evaluateAll(a, surfaces: surfaces) }
        let evalWall = (ProcessInfo.processInfo.systemUptime - evalStart) * 1_000

        let seqLoadStart = ProcessInfo.processInfo.systemUptime
        let seqLoadANE = try reloadModels(b)
        let seqLoadWall = (ProcessInfo.processInfo.systemUptime - seqLoadStart) * 1_000
        let sequentialWall = evalWall + seqLoadWall
        try unloadKeepingModels(b)

        let resultBox = OverlapResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        let overlapStart = ProcessInfo.processInfo.systemUptime
        DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-overlap-loader", qos: .userInitiated).async {
            do {
                let start = ProcessInfo.processInfo.systemUptime
                let ane = try reloadModels(boxB.models)
                let wall = (ProcessInfo.processInfo.systemUptime - start) * 1_000
                resultBox.set(.success((aneMS: ane, wallMS: wall)))
            } catch {
                resultBox.set(.failure(error))
            }
            semaphore.signal()
        }

        var overlapEvalANE = 0.0
        for _ in 0..<evalRepeats { overlapEvalANE += try evaluateAll(a, surfaces: surfaces) }
        semaphore.wait()
        let overlapWall = (ProcessInfo.processInfo.systemUptime - overlapStart) * 1_000
        guard let result = resultBox.get() else {
            throw AnimapkError.validation("overlap loader produced no result")
        }
        let overlapLoad = try result.get()

        lines.append(String(
            format: "sequential: eval4 ANE=%.1fms wall=%.1fms + reload8 ANE=%.1fms wall=%.1fms => %.1fms",
            evalANE, evalWall, seqLoadANE, seqLoadWall, sequentialWall))
        lines.append(String(
            format: "overlap: eval4 ANE=%.1fms || reload8 ANE=%.1fms wall=%.1fms => overlapWall=%.1fms gain=%.1fms (%.1f%%)",
            overlapEvalANE, overlapLoad.aneMS, overlapLoad.wallMS, overlapWall,
            sequentialWall - overlapWall,
            sequentialWall > 0 ? (sequentialWall - overlapWall) / sequentialWall * 100 : 0))
        lines.append("\(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")

        try unloadKeepingModels(b)
        _ = cacheB
        _ = cacheA
    }

    private static func residencySequence(
        specs: [ProgramSpec], maxPrograms: Int, maxLogicalBytes: UInt64,
        surfaces: ANEW8DiTSurfaces, context: MetalContext,
        pressure: PressureRecorder, lines: inout [String]
    ) throws -> SequenceOutcome {
        var loaded: [LoadedProgram] = []
        loaded.reserveCapacity(maxPrograms)
        var logicalBytes: UInt64 = 0
        var baselineSamples: [Double] = []
        var onset = false
        var lastLoadMS = 0.0
        let startIssues = pressure.issueCount

        for spec in specs {
            if loaded.count >= maxPrograms { break }
            if logicalBytes + spec.logicalBytes > maxLogicalBytes { break }

            let wallStart = ProcessInfo.processInfo.systemUptime
            let program = try loadProgram(spec)
            let wallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
            lastLoadMS = program.loadMilliseconds
            let evalMS = try evaluate(program, spec: spec, surfaces: surfaces)
            loaded.append(program)
            logicalBytes += spec.logicalBytes

            if baselineSamples.count < 8 { baselineSamples.append(lastLoadMS) }
            let baseline = median(baselineSamples)
            let slowThreshold = max(250.0, baseline * 6.0)
            let pressureChanged = pressure.issueCount > startIssues
            let pathological = baselineSamples.count >= 8 && lastLoadMS > slowThreshold

            if loaded.count <= 8 || loaded.count % 8 == 0 || pathological || pressureChanged {
                lines.append(String(
                    format: "#%03d logical=%6.0fMB last=%@ loadANE=%7.1fms loadWall=%7.1fms eval=%6.1fms threshold=%6.1fms %@ | %@",
                    loaded.count, Double(logicalBytes) / mib, spec.label,
                    lastLoadMS, wallMS, evalMS, slowThreshold,
                    MemorySnapshot.capture(context: context).compact, pressure.compact))
            }

            if pathological || pressureChanged {
                onset = true
                lines.append(
                    "STOP-ONSET: \(pathological ? "pathological load" : "new pressure signal") at programs=\(loaded.count) logical=\(String(format: "%.0f", Double(logicalBytes) / mib))MB.")
                break
            }
        }

        let baseline = median(baselineSamples)
        let outcome = SequenceOutcome(
            programs: loaded.count, logicalBytes: logicalBytes,
            onsetDetected: onset, baselineMedianMS: baseline, finalLoadMS: lastLoadMS)
        loaded.removeAll()
        return outcome
    }

    private static func makeProgramSpecs(file: AnimapkFile) throws -> [ProgramSpec] {
        var out: [ProgramSpec] = []
        out.reserveCapacity(ANEW8NativePack.expectedPreparedModelCount)

        for block in 0..<ModelConstants.ditBlocks {
            func tensor(_ suffix: String) throws -> AnimapkTensor {
                try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
            }
            func digest(_ suffix: String) throws -> String {
                guard let value = try tensor(suffix).blobSHA256 else {
                    throw AnimapkError.validation("ANE probe tensor hash missing")
                }
                return value
            }

            let q = try tensor("self_attn.q_proj.weight")
            let k = try tensor("self_attn.k_proj.weight")
            let v = try tensor("self_attn.v_proj.weight")
            let qkvBytes = try payloadBytes(q) + payloadBytes(k) + payloadBytes(v)
            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block,
                q: try digest("self_attn.q_proj.weight"),
                k: try digest("self_attn.k_proj.weight"),
                v: try digest("self_attn.v_proj.weight"))
            out.append(ProgramSpec(
                block: block, label: "b\(block).selfQKV", logicalBytes: qkvBytes,
                kind: .qkv(cacheKey: qkvKey)))

            for spec in ANEW8NativePack.projectionSpecs where
                spec.suffix != "self_attn.q_proj.weight" &&
                spec.suffix != "self_attn.k_proj.weight" &&
                spec.suffix != "self_attn.v_proj.weight" {
                let t = try tensor(spec.suffix)
                let key = ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: try digest(spec.suffix))
                out.append(ProgramSpec(
                    block: block, label: "b\(block).\(spec.tag)",
                    logicalBytes: try payloadBytes(t),
                    kind: .projection(
                        cacheKey: key, suffix: spec.suffix, tag: spec.tag,
                        inputChannels: spec.columns, outputChannels: spec.rows,
                        spatial: spec.spatial)))
            }
        }

        guard out.count == ANEW8NativePack.expectedPreparedModelCount else {
            throw AnimapkError.validation(
                "ANE probe program inventory \(out.count) != \(ANEW8NativePack.expectedPreparedModelCount)")
        }
        return out
    }

    private static func projectionSpec(
        file: AnimapkFile, block: Int, suffix: String
    ) throws -> ProgramSpec {
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            throw AnimapkError.validation("unknown projection suffix \(suffix)")
        }
        let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
        guard let digest = tensor.blobSHA256 else {
            throw AnimapkError.validation("projection digest missing")
        }
        return ProgramSpec(
            block: block, label: "b\(block).\(spec.tag)",
            logicalBytes: try payloadBytes(tensor),
            kind: .projection(
                cacheKey: ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: digest),
                suffix: suffix, tag: spec.tag,
                inputChannels: spec.columns, outputChannels: spec.rows,
                spatial: spec.spatial))
    }

    private static func makeProjection(_ spec: ProgramSpec) throws -> A12ANEProjectionModel {
        guard case let .projection(
            cacheKey, _, tag, inputChannels, outputChannels, spatial) = spec.kind else {
            throw AnimapkError.validation("expected projection program spec")
        }
        return try A12ANEProjectionModel(
            preparedInputChannels: UInt(inputChannels),
            outputChannels: UInt(outputChannels),
            spatial: UInt(spatial),
            label: "probe_\(spec.block)_\(tag)",
            cacheKey: cacheKey)
    }

    private static func loadProgram(_ spec: ProgramSpec) throws -> LoadedProgram {
        switch spec.kind {
        case .qkv(let cacheKey):
            return .qkv(try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(DiTBlockExecutor.tokens),
                label: "probe_\(spec.block)_self_qkv",
                cacheKey: cacheKey))
        case .projection:
            return .projection(try makeProjection(spec))
        }
    }

    private static func evaluate(
        _ program: LoadedProgram, spec: ProgramSpec, surfaces: ANEW8DiTSurfaces
    ) throws -> Double {
        var ms = 0.0
        switch (program, spec.kind) {
        case (.qkv(let model), .qkv):
            _ = try model.evaluateInput(
                surfaces.tokenInput,
                qOutput: surfaces.q, kOutput: surfaces.k, vOutput: surfaces.v,
                milliseconds: &ms)
        case (.projection(let model), .projection(_, let suffix, _, _, _, _)):
            switch suffix {
            case "cross_attn.k_proj.weight":
                _ = try model.evaluateInput(
                    surfaces.contextInput, output: surfaces.contextK, milliseconds: &ms)
            case "cross_attn.v_proj.weight":
                _ = try model.evaluateInput(
                    surfaces.contextInput, output: surfaces.contextV, milliseconds: &ms)
            case "mlp.layer1.weight":
                _ = try model.evaluateInput(
                    surfaces.tokenInput, output: surfaces.hidden, milliseconds: &ms)
            case "mlp.layer2.weight":
                _ = try model.evaluateInput(
                    surfaces.hidden, output: surfaces.tokenOutput, milliseconds: &ms)
            case "cross_attn.q_proj.weight":
                _ = try model.evaluateInput(
                    surfaces.tokenInput, output: surfaces.q, milliseconds: &ms)
            default:
                _ = try model.evaluateInput(
                    surfaces.tokenInput, output: surfaces.tokenOutput, milliseconds: &ms)
            }
        default:
            throw AnimapkError.validation("ANE probe program/spec kind mismatch")
        }
        return ms
    }

    private static func freshProjectionSentinel(file: AnimapkFile) throws -> Double {
        let spec = try projectionSpec(
            file: file, block: 0, suffix: "self_attn.output_proj.weight")
        var model: A12ANEProjectionModel? = try makeProjection(spec)
        guard let value = model?.loadMilliseconds else {
            throw AnimapkError.validation("fresh projection sentinel returned nil")
        }
        model?.invalidate()
        model = nil
        return value
    }

    private static func unloadKeepingModels(_ models: ANEW8DiTModels) throws {
        let ok = models.selfQKV.diagnosticUnloadKeepingModel()
            && models.selfO.diagnosticUnloadKeepingModel()
            && models.crossQ.diagnosticUnloadKeepingModel()
            && models.crossK.diagnosticUnloadKeepingModel()
            && models.crossV.diagnosticUnloadKeepingModel()
            && models.crossO.diagnosticUnloadKeepingModel()
            && models.mlpUp.diagnosticUnloadKeepingModel()
            && models.mlpDown.diagnosticUnloadKeepingModel()
        if !ok { throw AnimapkError.validation("diagnostic block unload failed") }
    }

    private static func reloadModels(_ models: ANEW8DiTModels) throws -> Double {
        var total = 0.0
        func reload(_ model: A12ANEProjectionModel) throws {
            let ms = model.diagnosticReloadMilliseconds()
            guard ms >= 0 else { throw AnimapkError.validation("diagnostic projection reload failed") }
            total += ms
        }
        let qkvMS = models.selfQKV.diagnosticReloadMilliseconds()
        guard qkvMS >= 0 else { throw AnimapkError.validation("diagnostic QKV reload failed") }
        total += qkvMS
        try reload(models.selfO)
        try reload(models.crossQ)
        try reload(models.crossK)
        try reload(models.crossV)
        try reload(models.crossO)
        try reload(models.mlpUp)
        try reload(models.mlpDown)
        return total
    }

    private static func evaluateAll(
        _ models: ANEW8DiTModels, surfaces: ANEW8DiTSurfaces
    ) throws -> Double {
        var totalMS = 0.0
        var qkvMS = 0.0
        _ = try models.selfQKV.evaluateInput(
            surfaces.tokenInput,
            qOutput: surfaces.q, kOutput: surfaces.k, vOutput: surfaces.v,
            milliseconds: &qkvMS)
        totalMS += qkvMS

        func projection(
            _ model: A12ANEProjectionModel,
            _ input: A12ANESurface,
            _ output: A12ANESurface
        ) throws {
            var ms = 0.0
            _ = try model.evaluateInput(input, output: output, milliseconds: &ms)
            totalMS += ms
        }

        try projection(models.selfO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.crossQ, surfaces.tokenInput, surfaces.q)
        try projection(models.crossK, surfaces.contextInput, surfaces.contextK)
        try projection(models.crossV, surfaces.contextInput, surfaces.contextV)
        try projection(models.crossO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.mlpUp, surfaces.tokenInput, surfaces.hidden)
        try projection(models.mlpDown, surfaces.hidden, surfaces.tokenOutput)
        return totalMS
    }

    private static func payloadBytes(_ tensor: AnimapkTensor) throws -> UInt64 {
        guard let scale = tensor.scaleSize, let zero = tensor.zeroSize else {
            throw AnimapkError.validation("ANE probe expected W8 scale/zero payload sizes")
        }
        return tensor.dataSize + scale + zero
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 { return (sorted[mid - 1] + sorted[mid]) / 2 }
        return sorted[mid]
    }

    private static func mb(_ bytes: UInt64) -> String {
        String(format: "%.0fMB", Double(bytes) / mib)
    }

    private static func processPhysicalFootprint() -> UInt64? {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
        #else
        return nil
        #endif
    }
}

/// Minimal file wrapper so the JSON can be shared via SwiftUI fileExporter.
struct JSONFile: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    var url: URL

    init(url: URL) { self.url = url }

    init(configuration: ReadConfiguration) throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("anima-xs-diagnostics.json")
        if let data = configuration.file.regularFileContents {
            try data.write(to: url)
        }
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let data = try Data(contentsOf: url)
        return FileWrapper(regularFileWithContents: data)
    }
}
