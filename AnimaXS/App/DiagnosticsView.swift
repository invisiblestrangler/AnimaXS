import SwiftUI
import UniformTypeIdentifiers

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
