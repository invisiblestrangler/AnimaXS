import SwiftUI
import UniformTypeIdentifiers
import os
#if canImport(Darwin)
import Darwin
#endif

/// K001 — minimal useful app. Model section (3 packs: download/import/repair),
/// prompt, seed + Randomize, Generate, Cancel, progress (stage/step/block +
/// elapsed), image, Share, Diagnostics, and user-recoverable errors.
///
/// Real-device stabilization changes:
/// - model discovery is local-only and side-effect free (no auto-download);
/// - launch avoids full SHA-256 re-hashing via verification receipts;
/// - Files imports hold security-scoped access for the whole async operation;
/// - Generate can never silently no-op: eligibility is computed once and
///   drives the button, the visible explanation, and the start guard;
/// - the prompt/number keyboards have an explicit dismissal path;
/// - Generate taps log thermal/memory/state facts for the next device run.
struct ContentView: View {
    @StateObject private var coordinator = GenerationCoordinator()
    @StateObject private var catalog = ModelCatalog()
    // Runtime inference-optimization settings (Diagnostics). Captured into an
    // immutable snapshot at Generate time; never mutated mid-run.
    @StateObject private var optimizationSettings = InferenceOptimizationSettings()
    @State private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
    @State private var seedText = "1337"
    @State private var generationStart = Date()
    @State private var elapsedText = ""
    @State private var elapsedTimer: Timer?
    @State private var shareImage: UIImage?
    @State private var showDiagnostics = false
    @State private var showingImporter = false
    @State private var importComponent: ModelComponent?
    @FocusState private var focusedField: FocusField?

    private enum FocusField: Hashable {
        case prompt
        case seed
    }

    private static let generationLog = Logger(
        subsystem: "com.invisiblestrangler.AnimaXS", category: "Generation")

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                promptSection
                seedSection
                generationSection
                errorSection
                imageSection
                metricsSection
            }
            .navigationTitle("AnimaXS")
            .navigationDestination(isPresented: $showDiagnostics) {
                DiagnosticsView(
                    lastMetricsText: coordinator.lastMetricsText,
                    optimizationSettings: optimizationSettings,
                    isGenerating: coordinator.isGenerating)
            }
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Diagnostics") { showDiagnostics = true }
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { focusedField = nil }
                }
            }
            .task {
                await catalog.refresh()
            }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppDidEnterBackground)
            ) { _ in coordinator.appDidEnterBackground() }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppWillEnterForeground)
            ) { _ in coordinator.appWillEnterForeground() }
            .onReceive(
                NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            ) { _ in coordinator.handleMemoryWarning() }
            .onChange(of: coordinator.state) { _, newState in
                // Elapsed-timer lifecycle: any terminal state must stop the
                // repeating timer so it cannot keep counting after the
                // generation is done.
                switch newState {
                case .completed, .cancelled, .failed:
                    stopElapsedTimer(updateFinalElapsed: true)
                default:
                    break
                }
            }
            .onDisappear {
                // Never leave the repeating timer attached to the run loop
                // after this view is gone.
                stopElapsedTimer(updateFinalElapsed: false)
            }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                guard let component = importComponent,
                      let url = try? result.get().first else { return }
                // Files-provided URLs are security-scoped: access must be held
                // for the ENTIRE async import (size check + SHA-256 + copy),
                // not just for obtaining the URL. Stopping access immediately
                // after creating the Task would break the read.
                let didStart = url.startAccessingSecurityScopedResource()
                Task {
                    defer {
                        if didStart {
                            url.stopAccessingSecurityScopedResource()
                        }
                    }
                    await catalog.importPack(component, from: url)
                }
            }
        }
    }

    // MARK: - Model section

    private var modelSection: some View {
        Section("Models") {
            ForEach(ModelComponent.allCases, id: \.self) { component in
                modelRow(component)
            }
        }
    }

    @ViewBuilder
    private func modelRow(_ component: ModelComponent) -> some View {
        let state = catalog.state(for: component)
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(component.displayName).font(.subheadline)
                Text(stateLabel(state)).font(.caption).foregroundStyle(stateColor(state))
                if case .failed(let message) = state {
                    Text(message).font(.caption2).foregroundStyle(.red).lineLimit(2)
                }
            }
            Spacer()
            if case .ready = state {
                Button("Repair") { Task { await catalog.repair(component) } }
                    .font(.caption)
            } else if case .failed = state {
                Button("Retry") { Task { await catalog.retry(component) } }
                    .font(.caption)
                Button("Import") { importComponent = component; showingImporter = true }
                    .font(.caption)
            } else if case .missing = state {
                Button("Download") { Task { await catalog.download(component) } }
                    .font(.caption)
                Button("Import") { importComponent = component; showingImporter = true }
                    .font(.caption)
            }
        }
    }

    // MARK: - Prompt

    private var promptSection: some View {
        Section("Prompt") {
            TextEditor(text: $prompt)
                .frame(minHeight: 100)
                .focused($focusedField, equals: .prompt)
        }
    }

    // MARK: - Seed

    private var seedSection: some View {
        Section("Seed") {
            HStack {
                TextField("Seed", text: $seedText)
                    .keyboardType(.numberPad)
                    .focused($focusedField, equals: .seed)
                Button("Randomize") {
                    seedText = String(UInt64.random(in: 0..<UInt64.max))
                }
                .disabled(isGenerating)
            }
        }
    }

    // MARK: - Generation controls

    private var generationSection: some View {
        Section {
            if isGenerating {
                // While a generation is active the section shows progress and
                // Cancel — never the normal Generate control, and never the
                // internal single-generation guard reason as a live warning.
                Button("Cancel", role: .destructive) {
                    // Freeze the visible elapsed time immediately, then request
                    // cooperative cancellation at the engine's next safe
                    // boundary.
                    stopElapsedTimer(updateFinalElapsed: true)
                    coordinator.cancel()
                }
            } else {
                Button("Generate") {
                    startGeneration()
                }
                .disabled(!eligibility.isReady)
                if let reason = eligibility.blockedReason {
                    Text(reason).font(.caption).foregroundStyle(.orange)
                }
            }
            progressView
        }
    }

    private var errorSection: some View {
        Group {
            if case .failed(let message) = coordinator.state {
                Section("Error") {
                    Text(message).foregroundStyle(.red)
                }
            }
        }
    }

    private var imageSection: some View {
        Section {
            if let image = coordinator.image {
                Image(uiImage: image)
                    .resizable().scaledToFit().frame(maxHeight: 300)
                if let url = shareURL(for: image) {
                    ShareLink(item: url, preview: SharePreview("AnimaXS image", image: Image(uiImage: image)))
                }
            }
        }
    }

    /// Phase 8 — post-generation telemetry summary (readable unplugged).
    @ViewBuilder
    private var metricsSection: some View {
        if let text = coordinator.lastMetricsText {
            Section("Run metrics") {
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

    /// Writes the generated image to a temporary PNG for sharing; returns nil
    /// if the file could not be created (share is then omitted gracefully).
    private func shareURL(for image: UIImage) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("anima-xs-\(UUID().uuidString).png")
        guard let data = image.pngData() else { return nil }
        do {
            try data.write(to: url)
            return url
        } catch {
            return nil
        }
    }

    // MARK: - Helpers

    /// Single eligibility source for the Generate button, the visible blocked
    /// reason, and the start guard — a tapped Generate can never silently
    /// return while the UI claims it is enabled.
    private var eligibility: GenerationEligibility {
        GenerationEligibility.evaluate(
            modelsResolved: catalog.resolved != nil,
            isGenerating: isGenerating,
            prompt: prompt,
            seedText: seedText,
            metalAvailable: coordinator.isMetalAvailable)
    }

    /// Stops the repeating elapsed timer. When `updateFinalElapsed` is true the
    /// visible text is frozen once at the value at stop time; a subsequent call
    /// (e.g. the coordinator reaching `.cancelled` after a Cancel tap) does not
    /// overwrite the frozen value because the timer is already nil.
    private func stopElapsedTimer(updateFinalElapsed: Bool = true) {
        guard elapsedTimer != nil else { return }
        elapsedTimer?.invalidate()
        elapsedTimer = nil
        if updateFinalElapsed {
            elapsedText = String(format: "%.1f s", Date().timeIntervalSince(generationStart))
        }
    }

    private func startGeneration() {
        logGenerationAttempt()
        guard case .ready = eligibility else {
            Self.generationLog.warning(
                "generation blocked: \(eligibility.blockedReason ?? "unknown", privacy: .public)")
            return
        }
        guard let seed = UInt64(seedText) else {
            Self.generationLog.error("generation guard: seed did not parse despite eligibility")
            return
        }
        // Re-read the resolved set rather than force-unwrapping: the catalog
        // can change between eligibility evaluation and this guard.
        guard let models = catalog.resolved else {
            Self.generationLog.error("generation guard: models disappeared after eligibility")
            return
        }
        Self.generationLog.info("generation accepted")
        generationStart = Date()
        elapsedText = ""
        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            elapsedText = String(format: "%.1f s", Date().timeIntervalSince(generationStart))
        }
        // Capture the immutable config snapshot at Generate time so a
        // mid-run toggle change can never affect this run. The DiT pack is
        // whatever variant (W4 or W8-v2) was imported into the .dit slot;
        // `models.dit` already points at it.
        let config = optimizationSettings.snapshot
        coordinator.generate(
            prompt: prompt, seed: seed, models: models,
            optimization: config)
        Self.generationLog.info(
            "generation state after start: \(String(describing: coordinator.state), privacy: .public)")
    }

    /// Physical-device instrumentation: records the facts needed to tell a
    /// UI-guard rejection from an inference crash on the next device run.
    private func logGenerationAttempt() {
        let thermal = ProcessInfo.processInfo.thermalState
        let availableMemory = availableProcessMemory().map { "\($0)" } ?? "n/a"
        // Break the values out of the interpolation so the compiler does not
        // have to type-check a long chain of nested conversions.
        let promptValid = !prompt.trimmingCharacters(in: .whitespaces).isEmpty
        let seedParses = UInt64(seedText.trimmingCharacters(in: .whitespaces)) != nil
        let modelsResolved = catalog.resolved != nil
        let thermalDescription = String(describing: thermal)
        let stateDescription = String(describing: coordinator.state)
        // Logger's OSLogMessage overload requires a single interpolation
        // literal — runtime String concatenation does not compile.
        Self.generationLog.info(
            "Generate tapped: promptValid=\(promptValid) seedParses=\(seedParses) modelsResolved=\(modelsResolved) thermal=\(thermalDescription) availableMemoryBytes=\(availableMemory) coordinatorState=\(stateDescription)")
    }

    private func availableProcessMemory() -> UInt64? {
        #if canImport(Darwin)
        return UInt64(os_proc_available_memory())
        #else
        return nil
        #endif
    }

    private var isGenerating: Bool { coordinator.isGenerating }

    @ViewBuilder
    private var progressView: some View {
        switch coordinator.state {
        case .idle:
            Text("Ready.").foregroundStyle(.secondary)
        case .tokenizing:
            Text("Tokenizing…")
        case .encodingPrompt:
            Text("Encoding prompt…")
        case .adapting:
            Text("Adapting…")
        case .diffusing(let step, let block, let totalSteps, let totalBlocks):
            VStack(alignment: .leading, spacing: 4) {
                Text("Diffusing step \(step)/\(totalSteps) · block \(block)/\(totalBlocks) · \(elapsedText)")
                ProgressView(
                    value: Double(step - 1) + Double(block) / Double(totalBlocks),
                    total: Double(totalSteps))
            }
        case .decoding:
            Text("Decoding VAE… · \(elapsedText)")
        case .completed:
            Text("Done. · \(elapsedText)").foregroundStyle(.green)
        case .cancelled:
            Text("Cancelled. · \(elapsedText)").foregroundStyle(.orange)
        case .failed:
            EmptyView()
        }
    }

    private func stateLabel(_ state: ModelStore.State) -> String {
        switch state {
        case .missing: return "missing"
        case .downloading: return "downloading…"
        case .verifying: return "verifying…"
        case .ready: return "ready"
        case .failed: return "failed"
        }
    }

    private func stateColor(_ state: ModelStore.State) -> Color {
        switch state {
        case .ready: return .green
        case .missing, .downloading, .verifying: return .secondary
        case .failed: return .red
        }
    }
}

// MARK: - Model catalog (K001 model section)

/// Observable model-state catalog bound to `ModelStore`. Exposes the three
/// production packs' states, download/import/repair actions, and the resolved
/// three-URL set once all are ready.
///
/// Discovery is local-only: `refresh()` never downloads and never hashes on
/// the main actor. Any full verification (only when a verification receipt is
/// absent/stale) runs on the `ModelStore` actor, off the UI thread; the UI
/// sees per-row `verifying…` states while that happens.
@MainActor
final class ModelCatalog: ObservableObject {
    @Published private var states: [ModelComponent: ModelStore.State] = [:]
    @Published private(set) var resolved: ResolvedModels?

    private let store: ModelStore?

    init() {
        // A store that fails to create its directory is surfaced via resolved=nil.
        store = try? ModelStore()
    }

    func state(for component: ModelComponent) -> ModelStore.State {
        states[component] ?? .missing
    }

    func refresh() async {
        guard let store else {
            states = ModelComponent.allCases.reduce(into: [:]) { $0[$1] = .failed("ModelStore unavailable") }
            resolved = nil
            return
        }
        // Local-only discovery: a valid verification receipt makes each row a
        // stat check; a missing/stale receipt triggers one full verification on
        // the store actor (off the main actor). Never downloads.
        for entry in ModelManifest.entries {
            states[entry.component] = await store.discover(entry)
        }
        resolved = (try? await store.resolveInstalledModels())
    }

    func download(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .downloading
            let url = try await store.download(entry)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    /// Explicit re-verification of an existing local file (full size + SHA),
    /// used by "Retry" on a failed row. Never downloads.
    func retry(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .verifying
            let url = try await store.verifyExisting(entry)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    func repair(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .downloading
            let url = try await store.repair(entry)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    func importPack(_ component: ModelComponent, from source: URL) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .verifying
            let url = try await store.importPack(entry, from: source)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    private func updateResolved() async throws {
        guard let store else { resolved = nil; return }
        resolved = try await store.resolveInstalledModels()
    }
}

extension ModelComponent {
    var displayName: String {
        switch self {
        case .dit: return "Anima Turbo DiT (adapter + diffusion)"
        case .textEncoder: return "Qwen3 text encoder"
        case .vae: return "Qwen-Image VAE"
        }
    }

    static var allCases: [ModelComponent] {
        [.textEncoder, .dit, .vae]
    }
}
