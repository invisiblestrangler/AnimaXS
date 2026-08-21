import SwiftUI
import UniformTypeIdentifiers
import os
#if canImport(Darwin)
import Darwin
#endif

/// K001 — minimal useful app. Model section (3 packs: download/import/repair),
/// optional user LoRA, prompt, seed + Randomize, Generate, Cancel, progress,
/// image, Share, Diagnostics, and user-recoverable errors.
struct ContentView: View {
    @StateObject private var coordinator = GenerationCoordinator()
    @StateObject private var catalog = ModelCatalog()
    @StateObject private var loraCatalog = LoRACatalog()
    @StateObject private var optimizationSettings = InferenceOptimizationSettings()
    @AppStorage("generation.lastPrompt")
    private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
    @AppStorage("generation.lastSeed")
    private var seedText = "1337"
    @AppStorage("generation.loraStrength")
    private var loraStrength = 1.0
    @State private var generationStart = Date()
    @State private var elapsedText = ""
    @State private var elapsedTimer: Timer?
    @State private var shareImage: UIImage?
    @State private var showDiagnostics = false
    @State private var showingImporter = false
    @State private var importComponent: ModelComponent?
    @State private var showingLoRAImporter = false
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
                loraSection
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
                    ditNumericsPolicy: resolvedDitNumericsPolicy,
                    ditVariantID: catalog.resolved?.dit.variant.id,
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
                await loraCatalog.refresh()
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
                switch newState {
                case .completed, .cancelled, .failed:
                    stopElapsedTimer(updateFinalElapsed: true)
                default:
                    break
                }
            }
            .onDisappear {
                stopElapsedTimer(updateFinalElapsed: false)
            }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                let component = importComponent
                importComponent = nil
                guard let component,
                      let url = try? result.get().first else { return }
                let didStart = url.startAccessingSecurityScopedResource()
                Task {
                    defer {
                        if didStart { url.stopAccessingSecurityScopedResource() }
                    }
                    await catalog.importPack(component, from: url)
                }
            }
            .fileImporter(
                isPresented: $showingLoRAImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                guard !isGenerating,
                      let url = try? result.get().first else { return }
                // Keep Files access alive for validation AND the complete copy
                // into app-owned storage. Inference never depends on a transient
                // security-scoped picker URL.
                let didStart = url.startAccessingSecurityScopedResource()
                Task {
                    defer {
                        if didStart { url.stopAccessingSecurityScopedResource() }
                    }
                    await loraCatalog.importAdapter(from: url)
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
                    .buttonStyle(.borderless)
            } else if case .failed = state {
                Button("Retry") { Task { await catalog.retry(component) } }
                    .font(.caption)
                    .buttonStyle(.borderless)
                Button("Import") { importComponent = component; showingImporter = true }
                    .font(.caption)
                    .buttonStyle(.borderless)
            } else if case .missing = state {
                Button("Download") { Task { await catalog.download(component) } }
                    .font(.caption)
                    .buttonStyle(.borderless)
                Button("Import") { importComponent = component; showingImporter = true }
                    .font(.caption)
                    .buttonStyle(.borderless)
            }
        }
    }

    // MARK: - External DiT LoRA

    private var loraSection: some View {
        Section("LoRA") {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(loraCatalog.selected?.displayName ?? "None")
                        .font(.subheadline)
                        .lineLimit(1)
                    if let selected = loraCatalog.selected {
                        Text("\(selected.moduleCount) adapted projection\(selected.moduleCount == 1 ? "" : "s")")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if loraCatalog.isImporting {
                        Text("Importing and validating…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Optional DiT adapter")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if loraCatalog.selected != nil {
                    Button("Remove", role: .destructive) {
                        loraCatalog.remove()
                    }
                    .font(.caption)
                    .buttonStyle(.borderless)
                    .disabled(isGenerating || loraCatalog.isImporting)
                }
                Button("Import") { showingLoRAImporter = true }
                    .font(.caption)
                    .buttonStyle(.borderless)
                    .disabled(isGenerating || loraCatalog.isImporting)
            }

            if loraCatalog.selected != nil {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Strength")
                        Spacer()
                        Text(loraStrength, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $loraStrength, in: 0...2, step: 0.05)
                        .disabled(isGenerating || loraCatalog.isImporting)
                    if loraStrength == 0 {
                        Text("0.00 disables the adapter and uses the baseline path.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let error = loraCatalog.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
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
                Button("Cancel", role: .destructive) {
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

    private var resolvedDitNumericsPolicy: DiTNumericsPolicy {
        if let resolved = catalog.resolved {
            return DiTNumericsPolicy.fromVariantID(resolved.dit.variant.id)
        }
        return .w4Legacy
    }

    private var eligibility: GenerationEligibility {
        let optimizationBlockingReason: String?
        if loraCatalog.isImporting {
            optimizationBlockingReason = "LoRA import is still in progress."
        } else {
            optimizationBlockingReason = InferenceOptimizationConfig.blockingReason(
                for: optimizationSettings.snapshot,
                numerics: resolvedDitNumericsPolicy,
                ditVariantID: catalog.resolved?.dit.variant.id)
        }
        return GenerationEligibility.evaluate(
            modelsResolved: catalog.resolved != nil,
            isGenerating: isGenerating,
            prompt: prompt,
            seedText: seedText,
            metalAvailable: coordinator.isMetalAvailable,
            optimizationBlockingReason: optimizationBlockingReason)
    }

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
        guard let baseModels = catalog.resolved else {
            Self.generationLog.error("generation guard: models disappeared after eligibility")
            return
        }

        // Freeze adapter identity + strength into the same immutable generation
        // snapshot as the three model packs. Strength zero intentionally becomes
        // nil, so the runtime never parses/allocates LoRA buffers for baseline.
        let adapterSnapshot: ResolvedLoRA?
        if let selected = loraCatalog.selected, loraStrength != 0 {
            adapterSnapshot = ResolvedLoRA(
                url: selected.url,
                displayName: selected.displayName,
                strength: Float(loraStrength))
        } else {
            adapterSnapshot = nil
        }
        let models = baseModels.withLoRA(adapterSnapshot)

        Self.generationLog.info("generation accepted")
        generationStart = Date()
        elapsedText = ""
        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            elapsedText = String(format: "%.1f s", Date().timeIntervalSince(generationStart))
        }
        let config = optimizationSettings.snapshot
        coordinator.generate(
            prompt: prompt, seed: seed, models: models,
            optimization: config)
        Self.generationLog.info(
            "generation state after start: \(String(describing: coordinator.state), privacy: .public)")
    }

    private func logGenerationAttempt() {
        let thermal = ProcessInfo.processInfo.thermalState
        let availableMemory = availableProcessMemory().map { "\($0)" } ?? "n/a"
        let promptValid = !prompt.trimmingCharacters(in: .whitespaces).isEmpty
        let seedParses = UInt64(seedText.trimmingCharacters(in: .whitespaces)) != nil
        let modelsResolved = catalog.resolved != nil
        let thermalDescription = String(describing: thermal)
        let stateDescription = String(describing: coordinator.state)
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

@MainActor
final class ModelCatalog: ObservableObject {
    @Published private var states: [ModelComponent: ModelStore.State] = [:]
    @Published private(set) var resolved: ResolvedModels?

    private let store: ModelStore?

    init() {
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

// MARK: - User LoRA catalog

struct ImportedLoRA: Equatable, Sendable {
    let url: URL
    let displayName: String
    let moduleCount: Int
}

/// Owns one v1 external DiT LoRA. The imported file is copied into Application
/// Support so a generation never depends on a Files picker URL or bookmark.
/// Parsing/copying runs off the MainActor; only the small published state is
/// returned to the UI.
@MainActor
final class LoRACatalog: ObservableObject {
    @Published private(set) var selected: ImportedLoRA?
    @Published private(set) var isImporting = false
    @Published private(set) var errorMessage: String?

    private static let displayNameKey = "generation.activeLoRADisplayName"
    private let directory: URL?

    init() {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
        directory = base?.appendingPathComponent("LoRA", isDirectory: true)
    }

    func refresh() async {
        guard let destination = activeURL else {
            selected = nil
            return
        }
        let storedName = UserDefaults.standard.string(forKey: Self.displayNameKey)
            ?? "Imported LoRA"
        do {
            let moduleCount = try await Task.detached(priority: .utility) {
                guard FileManager.default.fileExists(atPath: destination.path) else { return nil as Int? }
                return try DiTLoRAFile(url: destination).modules.count
            }.value
            if let moduleCount {
                selected = ImportedLoRA(
                    url: destination, displayName: storedName,
                    moduleCount: moduleCount)
                errorMessage = nil
            } else {
                selected = nil
            }
        } catch {
            selected = nil
            errorMessage = "Stored LoRA is invalid: \(error.localizedDescription)"
        }
    }

    func importAdapter(from source: URL) async {
        guard !isImporting, let destination = activeURL,
              let directory else { return }
        isImporting = true
        errorMessage = nil
        defer { isImporting = false }
        let displayName = source.deletingPathExtension().lastPathComponent
        do {
            let moduleCount = try await Task.detached(priority: .userInitiated) {
                // Validate the security-scoped source before copying it.
                _ = try DiTLoRAFile(url: source)
                let fm = FileManager.default
                try fm.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                let temporary = directory.appendingPathComponent(
                    "import-\(UUID().uuidString).safetensors")
                defer { try? fm.removeItem(at: temporary) }
                try fm.copyItem(at: source, to: temporary)
                // Validate the exact app-owned bytes before publishing them.
                let copied = try DiTLoRAFile(url: temporary)
                let count = copied.modules.count
                withExtendedLifetime(copied) {}
                if fm.fileExists(atPath: destination.path) {
                    _ = try fm.replaceItemAt(destination, withItemAt: temporary)
                } else {
                    try fm.moveItem(at: temporary, to: destination)
                }
                return count
            }.value
            UserDefaults.standard.set(displayName, forKey: Self.displayNameKey)
            selected = ImportedLoRA(
                url: destination, displayName: displayName,
                moduleCount: moduleCount)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func remove() {
        guard !isImporting else { return }
        if let destination = activeURL {
            do {
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                selected = nil
                errorMessage = nil
                UserDefaults.standard.removeObject(forKey: Self.displayNameKey)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private var activeURL: URL? {
        directory?.appendingPathComponent("active.safetensors")
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
