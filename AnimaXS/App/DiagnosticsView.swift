import SwiftUI
import UniformTypeIdentifiers

/// K005 — diagnostics screen. Runs the shared `DiagnosticsEngine` self-tests
/// and exports the structured report as JSON.
struct DiagnosticsView: View {
    @State private var report: DiagnosticsReport?
    @State private var isRunning = false
    @State private var exportError: String?
    @State private var exportPresented = false
    @State private var jsonURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("anima-xs-diagnostics.json")

    var body: some View {
        Form {
            Section("Device & app") {
                if let report {
                    LabeledContent("App version", value: report.appVersion)
                    LabeledContent("OS", value: report.osVersion)
                    LabeledContent("Device", value: report.deviceModel)
                    LabeledContent("Memory", value: "\(report.physicalMemoryBytes / 1_048_576) MB")
                    LabeledContent("Metal", value: report.metalAvailable ? report.metalDeviceName : "unavailable")
                    LabeledContent("Disk free", value: "\(report.availableDiskBytes / 1_048_576) MB")
                    LabeledContent("Thermal", value: report.thermalState)
                }
            }

            Section("Model packs") {
                if let report {
                    ForEach(report.modelPacks, id: \.filename) { pack in
                        LabeledContent(
                            pack.filename,
                            value: pack.installed
                                ? (pack.verified ? "verified" : "corrupt")
                                : "missing")
                    }
                }
            }

            Section("Self-tests") {
                if let report {
                    ForEach(report.selfTests, id: \.name) { item in
                        HStack(alignment: .firstTextBaseline) {
                            Image(systemName: statusSymbol(item.status))
                                .foregroundStyle(statusColor(item.status))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.name).font(.subheadline)
                                Text(item.detail).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                } else if isRunning {
                    ProgressView("Running self-tests…")
                }
            }

            Section {
                Button(isRunning ? "Running…" : "Run diagnostics") {
                    Task { await run() }
                }
                .disabled(isRunning)
                if report != nil {
                    Button("Export JSON") {
                        exportPresented = true
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
        .navigationTitle("Diagnostics")
        .task { await run() }
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

    private func run() async {
        isRunning = true
        let engine = DiagnosticsEngine()
        let report = await engine.report()
        self.report = report
        // Persist the JSON so ShareLink/fileExporter can read it.
        try? await engine.writeJSON(to: jsonURL)
        isRunning = false
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
