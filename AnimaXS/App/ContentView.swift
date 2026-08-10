import SwiftUI

/// Minimal placeholder main screen. UI wiring arrives with the generation pipeline (K001).
struct ContentView: View {
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
                    Text("Inference core not yet connected.")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("AnimaXS")
        }
    }
}
