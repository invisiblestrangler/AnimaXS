import SwiftUI
import UniformTypeIdentifiers

/// K001 — minimal useful app. Model section (3 packs: download/import/repair),
/// prompt, seed + Randomize, Generate, Cancel, progress (stage/step/block +
/// elapsed), Resume, image, Share, Diagnostics, and user-recoverable errors.
struct ContentView: View {
    @StateObject private var coordinator = GenerationCoordinator()
    @StateObject private var catalog = ModelCatalog()
    @State private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
    @State private var seedText = "1337"
    @State private var generationStart = Date()
    @State private var elapsedText = ""
    @State private var elapsedTimer: Timer?
    @State private var shareImage: UIImage?
    @State private var showDiagnostics = false
    @State private var showingImporter = false
    @State private var importComponent: ModelComponent?

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                promptSection
                seedSection
                generationSection
                errorSection
                imageSection
            }
            .navigationTitle("AnimaXS")
            .navigationDestination(isPresented: $showDiagnostics) {
                DiagnosticsView()
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Diagnostics") { showDiagnostics = true }
                }
            }
            .task { await catalog.refresh() }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppDidEnterBackground)
            ) { _ in coordinator.appDidEnterBackground() }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppWillEnterForeground)
            ) { _ in coordinator.appWillEnterForeground() }
            .onReceive(
                NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            ) { _ in coordinator.handleMemoryWarning() }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                guard let component = importComponent,
                      let url = try? result.get().first else { return }
                Task { await catalog.importPack(component, from: url) }
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
                Button("Retry") { Task { await catalog.download(component) } }
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
        }
    }

    // MARK: - Seed

    private var seedSection: some View {
        Section("Seed") {
            HStack {
                TextField("Seed", text: $seedText)
                    .keyboardType(.numberPad)
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
            if canGenerate {
                Button("Generate") {
                    startGeneration(resume: false)
                }
            } else if coordinator.canResume {
                Button("Resume") {
                    startGeneration(resume: true)
                }
            }
            if isGenerating {
                Button("Cancel", role: .destructive) {
                    coordinator.cancel()
                }
            }
            if coordinator.canResume {
                Text("Checkpoint: \(coordinator.completedSteps ?? 0)/8 steps — Resume available.")
                    .font(.caption).foregroundStyle(.orange)
            }
            progressView
        }
    }

    private var errorSection: some View {
        Group {
            if case .failed(let message) = coordinator.state {
                Section("Error") {
                    Text(message).foregroundStyle(.red)
                    if coordinator.completedSteps ?? 0 > 0 {
                        Button("Discard checkpoint") { coordinator.discardCheckpoint() }
                    }
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

    private func startGeneration(resume: Bool) {
        guard let models = catalog.resolved, !prompt.trimmingCharacters(in: .whitespaces).isEmpty else {
            return
        }
        guard let seed = UInt64(seedText) else {
            return
        }
        generationStart = Date()
        elapsedText = ""
        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            elapsedText = String(format: "%.1f s", Date().timeIntervalSince(generationStart))
        }
        if resume {
            coordinator.resume(prompt: prompt, seed: seed, models: models)
        } else {
            coordinator.generate(prompt: prompt, seed: seed, models: models)
        }
    }

    private var isGenerating: Bool { coordinator.isGenerating }

    private var canGenerate: Bool {
        catalog.resolved != nil && !isGenerating && !coordinator.canResume
    }

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
            return
        }
        // Discover already-installed valid packs (cold-launch ready).
        for entry in ModelManifest.entries {
            let url = await store.localURL(for: entry)
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    try ModelManifest.verify(url, against: entry)
                    states[entry.component] = .ready(url)
                } catch {
                    states[entry.component] = .failed(error.localizedDescription)
                }
            } else {
                states[entry.component] = .missing
            }
        }
        try? await updateResolved()
    }

    func download(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .downloading
            let url = try await store.prepare(entry)
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
        resolved = try await store.resolvedModels()
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
