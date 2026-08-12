import SwiftUI

/// Main screen. Connected to GenerationCoordinator (K002).
struct ContentView: View {
    @StateObject private var coordinator = GenerationCoordinator()
    @State private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
    @State private var seedText = "1337"

    var body: some View {
        NavigationStack {
            Form {
                Section("Prompt") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 120)
                }
                Section("Seed") {
                    TextField("Seed", text: $seedText)
                        .keyboardType(.numberPad)
                }
                Section {
                    Button("Generate") {
                        // K002 wires the coordinator. Model URLs are resolved
                        // by ModelStore when model-assets-v1 is available (A005).
                        // Until then, generation will fail at the pack-loading
                        // stage with a clear error in `state`.
                        Task {
                            await coordinator.generate(
                                prompt: prompt,
                                modelURLs: ModelURLs(
                                    qwen: URL(fileURLWithPath: "/dev/null"),
                                    adapter: URL(fileURLWithPath: "/dev/null"),
                                    dit: URL(fileURLWithPath: "/dev/null"),
                                    vae: URL(fileURLWithPath: "/dev/null")))
                        }
                    }
                    .disabled(isGenerating)

                    if case .failed(let message) = coordinator.state {
                        Text("Error: \(message)")
                            .foregroundStyle(.red)
                    }

                    if let image = coordinator.image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 300)
                    }

                    switch coordinator.state {
                    case .idle:
                        Text("Ready to generate.")
                            .foregroundStyle(.secondary)
                    case .tokenizing:
                        Text("Tokenizing…")
                    case .encodingPrompt:
                        Text("Encoding prompt…")
                    case .adapting:
                        Text("Adapting…")
                    case .diffusing(let step, let total):
                        Text("Diffusing step \(step)/\(total)…")
                    case .decoding:
                        Text("Decoding VAE…")
                    case .completed:
                        Text("Done.")
                            .foregroundStyle(.green)
                    case .failed:
                        EmptyView()
                    }
                }
            }
            .navigationTitle("AnimaXS")
        }
    }

    private var isGenerating: Bool {
        switch coordinator.state {
        case .tokenizing, .encodingPrompt, .adapting,
             .diffusing, .decoding:
            return true
        default:
            return false
        }
    }
}
