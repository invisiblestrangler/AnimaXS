import SwiftUI

/// Main screen (K002 wiring; full K001 UI built in a later phase).
///
/// Model URLs are resolved through `ModelStore` from the three production
/// packs (Qwen text encoder, DiT+adapter, VAE). The visible seed feeds the
/// production RNG. No `/dev/null`, no fixture packs, no hidden randomization.
struct ContentView: View {
    @StateObject private var coordinator = GenerationCoordinator()
    @State private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
    @State private var seedText = "1337"
    @State private var resolvedModels: ResolvedModels?
    @State private var modelStatus = "Resolving models…"
    @State private var generationStart = Date()
    @State private var elapsedText = ""

    private let store = try? ModelStore()

    var body: some View {
        NavigationStack {
            Form {
                Section("Models") {
                    Text(modelStatus)
                        .foregroundStyle(resolvedModels == nil ? .orange : .green)
                }
                Section("Prompt") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 100)
                }
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
                Section {
                    if canGenerate {
                        Button("Generate") {
                            guard let seed = UInt64(seedText) else { return }
                            guard let models = resolvedModels else { return }
                            generationStart = Date()
                            elapsedText = ""
                            coordinator.generate(prompt: prompt, seed: seed, models: models)
                        }
                    } else if coordinator.canResume {
                        Button("Resume") {
                            guard let seed = UInt64(seedText) else { return }
                            guard let models = resolvedModels else { return }
                            generationStart = Date()
                            elapsedText = ""
                            coordinator.resume(prompt: prompt, seed: seed, models: models)
                        }
                    }

                    if isGenerating {
                        Button("Cancel", role: .destructive) {
                            coordinator.cancel()
                        }
                    }

                    progressView

                    if case .failed(let message) = coordinator.state {
                        Text("Error: \(message)")
                            .foregroundStyle(.red)
                    }
                    if coordinator.canResume {
                        Text("Checkpoint: \(coordinator.completedSteps ?? 0)/8 steps completed — Resume available.")
                            .foregroundStyle(.orange)
                    }

                    if let image = coordinator.image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 300)
                    }
                }
            }
            .navigationTitle("AnimaXS")
            .task {
                await resolveModels()
            }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppDidEnterBackground)
            ) { _ in
                coordinator.appDidEnterBackground()
            }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppWillEnterForeground)
            ) { _ in
                coordinator.appWillEnterForeground()
            }
            .onReceive(
                NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            ) { _ in
                coordinator.handleMemoryWarning()
            }
            .onReceive(
                Timer.publish(every: 5, on: .main, in: .common).autoconnect()
            ) { _ in
                coordinator.handleThermalState(ProcessInfo.processInfo.thermalState)
            }
        }
    }

    // MARK: - Model resolution

    private func resolveModels() async {
        guard let store else {
            modelStatus = "ModelStore unavailable"
            return
        }
        do {
            // Three production packs; the DiT pack serves both adapter and sampler.
            let byComponent = Dictionary(
                uniqueKeysWithValues: try await withThrowingTaskGroup(of: (ModelComponent, URL).self) { group in
                    for entry in ModelManifest.entries {
                        group.addTask {
                            let url = try await store.prepare(entry)
                            return (entry.component, url)
                        }
                    }
                    var result: [(ModelComponent, URL)] = []
                    for try await pair in group { result.append(pair) }
                    return result
                })
            guard let textEncoder = byComponent[.textEncoder],
                  let dit = byComponent[.dit],
                  let vae = byComponent[.vae] else {
                modelStatus = "Incomplete model set"
                return
            }
            resolvedModels = ResolvedModels(textEncoder: textEncoder, dit: dit, vae: vae)
            modelStatus = "Models ready"
        } catch {
            modelStatus = "Model error: \(error.localizedDescription)"
            resolvedModels = nil
        }
    }

    // MARK: - Derived state

    private var isGenerating: Bool {
        coordinator.isGenerating
    }

    private var canGenerate: Bool {
        !isGenerating && !coordinator.canResume && resolvedModels != nil
    }

    @ViewBuilder
    private var progressView: some View {
        switch coordinator.state {
        case .idle:
            Text("Ready.")
                .foregroundStyle(.secondary)
        case .tokenizing:
            Text("Tokenizing…")
        case .encodingPrompt:
            Text("Encoding prompt…")
        case .adapting:
            Text("Adapting…")
        case .diffusing(let step, let block, let totalSteps, let totalBlocks):
            VStack(alignment: .leading, spacing: 4) {
                Text("Diffusing step \(step)/\(totalSteps) · block \(block)/\(totalBlocks)")
                ProgressView(
                    value: Double(step - 1) + Double(block) / Double(totalBlocks),
                    total: Double(totalSteps))
            }
        case .decoding:
            Text("Decoding VAE…")
        case .completed:
            Text("Done.")
                .foregroundStyle(.green)
        case .cancelled:
            Text("Cancelled.")
                .foregroundStyle(.orange)
        case .failed:
            EmptyView()
        }
    }
}
