import SwiftUI

#if DEBUG
/// Debug-only launcher for the ANE Oracle E physical-device parity capture.
/// It is intentionally separate from production Generate/Diagnostics state so
/// the experiment can be removed cleanly after A12 localization is complete.
struct OracleEDeviceCaptureLauncher: View {
    @State private var presented = false

    var body: some View {
        Button {
            presented = true
        } label: {
            Label("Oracle E", systemImage: "waveform.path.ecg")
                .font(.caption.bold())
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Open Oracle E device capture")
        .sheet(isPresented: $presented) {
            NavigationStack {
                OracleEDeviceCaptureSheet()
            }
        }
    }
}

@MainActor
private struct OracleEDeviceCaptureSheet: View {
    /// Exact positive prompt from the pinned Oracle-V2 control configuration.
    private static let oraclePrompt = "masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, blue eyes, black witch hat, dark blue robe, holding a glowing lantern, moonlit forest, fireflies, detailed anime illustration"
    private static let oracleSeed: UInt64 = 123_456_789

    @Environment(\.dismiss) private var dismiss
    @State private var models: ResolvedModels?
    @State private var modelStatus = "Checking installed packs…"
    @State private var captureStatus = "Idle"
    @State private var captureURL: URL?
    @State private var isRunning = false

    var body: some View {
        Form {
            Section("Pinned control") {
                LabeledContent("Seed", value: String(Self.oracleSeed))
                Text(Self.oraclePrompt)
                    .font(.caption)
                    .textSelection(.enabled)
                Text("Runs exactly one DiT traversal (step 0) and stops after block 27 MLP. No VAE decode and no later diffusion steps.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Pack gate") {
                Text(modelStatus)
                    .font(.caption)
                if let models {
                    LabeledContent("DiT variant", value: models.dit.variant.id)
                    LabeledContent("DiT pack", value: models.dit.variant.displayFilename)
                }
                Button("Refresh local packs") {
                    Task { await refreshModels() }
                }
                .disabled(isRunning)
            }

            Section("A12 capture") {
                Button(isRunning ? "Capturing…" : "Capture Oracle E step 0") {
                    Task { await runCapture() }
                }
                .disabled(isRunning || models?.dit.variant.id != ModelManifest.ditW8ANEV1.id)

                Text(captureStatus)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)

                if let captureURL {
                    ShareLink(item: captureURL) {
                        Label("Share .oraclee capture", systemImage: "square.and.arrow.up")
                    }
                    .font(.caption)
                }
            }

            Section("Safety") {
                Text("Do not start this while a normal generation is running. The capture uses the same ANE/Metal resources and is deliberately a standalone diagnostic lane.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                Text("The output contains only model-independent activations/conditioning/noise samples and metadata; it does not copy model weights.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Oracle E · A12")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Close") { dismiss() }
                    .disabled(isRunning)
            }
        }
        .interactiveDismissDisabled(isRunning)
        .task { await refreshModels() }
    }

    private func refreshModels() async {
        guard !isRunning else { return }
        do {
            let store = try ModelStore()
            let resolved = try await store.resolveInstalledModels()
            models = resolved
            if resolved.dit.variant.id == ModelManifest.ditW8ANEV1.id {
                modelStatus = "Ready: exact ANE-native W8 pack is installed."
            } else {
                modelStatus = "Blocked: import the w8-ane-v1 DiT pack before running Oracle E."
            }
        } catch {
            models = nil
            modelStatus = "Blocked: \(error.localizedDescription)"
        }
    }

    private func runCapture() async {
        guard !isRunning else { return }
        guard let models else {
            captureStatus = "Blocked: model set is not resolved."
            return
        }
        guard models.dit.variant.id == ModelManifest.ditW8ANEV1.id else {
            captureStatus = "Blocked: DiT variant must be w8-ane-v1."
            return
        }
        guard let context = MetalContext() else {
            captureStatus = "Blocked: Metal is unavailable."
            return
        }

        isRunning = true
        captureURL = nil
        captureStatus = "Running pinned step-0 ANE capture…"
        defer { isRunning = false }

        do {
            let documents = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true)
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd-HHmmss"
            let destination = documents.appendingPathComponent(
                "AnimaXS-OracleE-A12-\(formatter.string(from: Date())).oraclee",
                isDirectory: false)
            let runner = OracleEDeviceCaptureRunner(context: context)
            let url = try await runner.run(
                prompt: Self.oraclePrompt,
                seed: Self.oracleSeed,
                models: models,
                outputURL: destination)
            captureURL = url
            captureStatus = "PASS: 84/84 step-0 branch checkpoints captured. Share the .oraclee file for comparison."
        } catch {
            captureStatus = "FAILED: \(error.localizedDescription)"
        }
    }
}
#endif
