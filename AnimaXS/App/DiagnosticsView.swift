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

    /// Whether a generation is currently active (controls are disabled while
    /// it runs so a toggle can never mutate an in-flight run).
    var isGenerating: Bool

    private let marker = DiagnosticRunMarker()

    init(
        lastMetricsText: String? = nil,
        optimizationSettings: InferenceOptimizationSettings,
        isGenerating: Bool = false
    ) {
        self.lastMetricsText = lastMetricsText
        self.optimizationSettings = optimizationSettings
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
            Text("Applies to the next fresh generation. Use Generate, not Resume, for performance comparisons.")
                .font(.caption)
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
                .disabled(isGenerating)
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
