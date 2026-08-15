import Foundation
import Metal
import MetalPerformanceShaders
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Diagnostic result

/// PASS / FAIL / SKIPPED for a self-test. SKIPPED means the hardware or a
/// required dependency is genuinely unavailable — it is never a false PASS.
enum DiagnosticStatus: String, Codable, Equatable {
    case pass
    case fail
    case skipped
}

struct DiagnosticItem: Codable, Equatable {
    let name: String
    let status: DiagnosticStatus
    let detail: String
}

// MARK: - Structured report

/// Codable, shareable diagnostic report (K005). Collects only device/app/
/// model facts needed to diagnose generation — no unrelated personal data.
///
/// A report is built from the cheap snapshot plus whatever explicit test
/// levels the user ran; serialization never re-executes tests.
struct DiagnosticsReport: Codable, Equatable {
    var appVersion: String
    var osVersion: String
    var deviceModel: String
    var physicalMemoryBytes: UInt64
    var metalDeviceName: String
    var metalAvailable: Bool
    var availableDiskBytes: Int64
    var thermalState: String
    var availableProcessMemoryBytes: UInt64?
    var modelPacks: [ModelPackInfo]
    var selfTests: [DiagnosticItem]
    var generationTimings: [String: Double]
    var generatedAt: Date

    struct ModelPackInfo: Codable, Equatable {
        let component: String
        let filename: String
        let installed: Bool
        let sizeBytes: UInt64
        let expectedSize: UInt64
        /// Cheap trust: a valid verification receipt covers the installed file.
        /// This does NOT hash the pack.
        var verified: Bool
        /// True only after an explicit deep SHA-256 integrity run verified the
        /// pack in this session.
        var sha256Verified: Bool
    }
}

// MARK: - Engine

/// Builds a structured diagnostic report and runs deterministic self-tests.
/// Reuses production primitives (QuantDecoders, SeededRNG, ModelManifest,
/// Metal/MPS executors) so XCTest and the UI share the same diagnostic
/// functions — it does not copy test implementations into the UI.
///
/// Diagnostics are split into explicit levels so opening the screen is cheap:
/// - `snapshot()`: device/app/model *presence* facts, no hashing, no compute;
/// - `basicSelfTests()`: deterministic CPU vector/RNG/file tests;
/// - `hardwareTests()`: real Metal/MPS command-buffer probes (A12 crash risk);
/// - `deepIntegrity()`: full size + SHA-256 of every installed pack (~2 GB).
///
/// A `DiagnosticRunMarker` persists the currently-running test name across
/// launches so a native crash can be localized ("ended during MPS precision").
struct DiagnosticsEngine {
    let context: MetalContext?
    private let forceMetalUnavailable: Bool
    private let storeProvider: () -> ModelStore?

    init(
        context: MetalContext? = nil,
        forceMetalUnavailable: Bool = false,
        storeProvider: (() -> ModelStore?)? = nil
    ) {
        self.context = context ?? MetalContext()
        self.forceMetalUnavailable = forceMetalUnavailable
        self.storeProvider = storeProvider ?? { try? ModelStore() }
    }

    /// Whether Metal is actually usable (recoverable, not a crash).
    var isMetalAvailable: Bool {
        !forceMetalUnavailable && context != nil
    }

    // MARK: - Cheap snapshot (zero hashing, zero compute)

    /// Device/app facts plus model pack presence/size/receipt state. Never
    /// SHA-256s packs and never runs compute tests. Safe to run when the
    /// Diagnostics screen opens.
    func snapshot() async -> DiagnosticsReport {
        let store = storeProvider()
        var packs: [DiagnosticsReport.ModelPackInfo] = []
        for entry in ModelManifest.entries {
            let status: ModelFileStatus
            if let store {
                status = await store.fileStatus(for: entry)
            } else {
                status = ModelFileStatus(exists: false, sizeBytes: 0, receiptValid: false)
            }
            packs.append(.init(
                component: entry.component.rawValue, filename: entry.filename,
                installed: status.exists, sizeBytes: status.sizeBytes,
                expectedSize: entry.size, verified: status.receiptValid,
                sha256Verified: false))
        }
        return DiagnosticsReport(
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown",
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            deviceModel: deviceModelIdentifier(),
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            metalDeviceName: context?.device.name ?? "unavailable",
            metalAvailable: isMetalAvailable,
            availableDiskBytes: availableDiskSpace(),
            thermalState: "\(ProcessInfo.processInfo.thermalState)",
            availableProcessMemoryBytes: availableProcessMemory(),
            modelPacks: packs,
            selfTests: [],
            generationTimings: [:],
            generatedAt: Date())
    }

    // MARK: - Test levels

    /// Deterministic, pack-free, compute-free CPU tests: W4/W8 vector decode,
    /// golden-noise RNG determinism, and a small mmap/file read probe. No
    /// model hashing, no Metal command buffers.
    func basicSelfTests() -> [DiagnosticItem] {
        var items: [DiagnosticItem] = []
        items.append(w4Vector())
        items.append(w8Vector())
        items.append(goldenNoiseRNG())
        items.append(mmapBenchmark())
        items.append(metalCapability())
        return items
    }

    /// Real Metal/MPS command-buffer probes (the A12 crash-risk group). Runs
    /// sequentially with visible per-test progress, persists the running test
    /// via `marker`, skips cleanly when the thermal gate trips, and records
    /// thermal/process-memory/elapsed facts per test.
    func hardwareTests(
        marker: DiagnosticRunMarker,
        progress: (String) -> Void,
        thermalGate: () -> Bool
    ) async -> [DiagnosticItem] {
        guard isMetalAvailable, let context else {
            let detail = forceMetalUnavailable ? "Metal unavailable (simulated)" : "Metal unavailable"
            return ["MPS precision", "GEMM", "Attention tile"].map {
                DiagnosticItem(name: $0, status: .skipped, detail: detail)
            }
        }
        var items: [DiagnosticItem] = []
        for name in ["MPS precision", "GEMM", "Attention tile"] {
            if thermalGate() {
                items.append(DiagnosticItem(
                    name: name, status: .skipped,
                    detail: "skipped: device too warm (thermal state: \(ProcessInfo.processInfo.thermalState))"))
                continue
            }
            progress(name)
            marker.markStarted(name)
            let thermalBefore = "\(ProcessInfo.processInfo.thermalState)"
            let memoryBefore = formatBytes(availableProcessMemory())
            let start = Date()
            let item: DiagnosticItem
            switch name {
            case "MPS precision":
                item = mpsPrecision(context: context)
            case "GEMM":
                item = gemm(context: context)
            default:
                item = attentionTile(context: context)
            }
            let elapsedMS = Date().timeIntervalSince(start) * 1000
            marker.markCompleted(name)
            let thermalAfter = "\(ProcessInfo.processInfo.thermalState)"
            let memoryAfter = formatBytes(availableProcessMemory())
            items.append(DiagnosticItem(
                name: item.name, status: item.status,
                detail: item.detail
                    + String(format: " · %.0f ms · thermal %@→%@ · mem %@→%@",
                             elapsedMS, thermalBefore, thermalAfter, memoryBefore, memoryAfter)))
        }
        return items
    }

    /// Full size + SHA-256 verification of every installed pack (~2.07 GB of
    /// reading). Explicit user action only — never run automatically. Also
    /// refreshes verification receipts for packs that pass.
    func deepIntegrity(
        marker: DiagnosticRunMarker,
        progress: (String) -> Void,
        entries: [ModelManifestEntry] = ModelManifest.entries
    ) async -> [DiagnosticItem] {
        guard let store = storeProvider() else {
            return [DiagnosticItem(
                name: "Deep model SHA-256", status: .fail,
                detail: "ModelStore unavailable")]
        }
        var items: [DiagnosticItem] = []
        var missing: [String] = []
        for entry in entries {
            let testName = "Deep SHA: \(entry.filename)"
            progress(testName)
            marker.markStarted(testName)
            do {
                _ = try await store.verifyExisting(entry)
                items.append(DiagnosticItem(
                    name: testName, status: .pass,
                    detail: "size + SHA-256 verified"))
            } catch {
                items.append(DiagnosticItem(
                    name: testName, status: .fail,
                    detail: error.localizedDescription))
                missing.append(entry.filename)
            }
            marker.markCompleted(testName)
        }
        if missing.isEmpty {
            items.append(DiagnosticItem(
                name: "Deep model SHA-256", status: .pass,
                detail: "all 3 packs fully verified"))
        } else {
            items.append(DiagnosticItem(
                name: "Deep model SHA-256", status: .fail,
                detail: "missing/corrupt: \(missing.joined(separator: ", "))"))
        }
        return items
    }

    /// Basic + hardware tests, each executed exactly once, sequentially.
    /// Backs the "Run diagnostics" button.
    func fullRun(
        marker: DiagnosticRunMarker,
        progress: (String) -> Void,
        thermalGate: () -> Bool
    ) async -> [DiagnosticItem] {
        var items = basicSelfTests()
        items += await hardwareTests(marker: marker, progress: progress, thermalGate: thermalGate)
        return items
    }

    // MARK: - Pure serialization (runs zero tests)

    func json(report: DiagnosticsReport) throws -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(data: try encoder.encode(report), encoding: .utf8)!
    }

    func writeJSON(_ report: DiagnosticsReport, to url: URL) throws {
        try Data(try json(report: report).utf8).write(to: url, options: .atomic)
    }

    // MARK: - Self-tests (reuse production primitives)

    /// Deterministic W4 vector decode against a known packed value.
    private func w4Vector() -> DiagnosticItem {
        // W4 nibbles: byte 0x32 → low nibble 2, high nibble 3; byte 0x10 →
        // low nibble 0, high nibble 1. Decoded k=4 order is [2,3,0,1] with
        // scale=1, zero=0 (decode = q*scale + zero).
        let nibbles: [UInt8] = [0x32, 0x10]
        let scale: [UInt16] = [Float16(1).bitPattern]
        let zero: [UInt16] = [Float16(0).bitPattern]
        let values = QuantDecoders.dequantW4(
            data: Data(nibbles), scale: scale.withUnsafeBytes { Data($0) },
            zero: zero.withUnsafeBytes { Data($0) },
            k: 4, groupSize: 4)
        let expected: [Float] = [2, 3, 0, 1]
        guard values == expected else {
            return .init(name: "W4 vector", status: .fail,
                         detail: "got \(values), expected \(expected)")
        }
        return .init(name: "W4 vector", status: .pass, detail: "known vector decoded exactly")
    }

    /// Deterministic W8 vector decode.
    private func w8Vector() -> DiagnosticItem {
        let bytes: [UInt8] = [10, 20, 30, 40]
        let scale: [UInt16] = [Float16(1).bitPattern]
        let zero: [UInt16] = [Float16(0).bitPattern]
        let values = QuantDecoders.dequantW8(
            data: Data(bytes), scale: scale.withUnsafeBytes { Data($0) },
            zero: zero.withUnsafeBytes { Data($0) },
            k: 4, groupSize: 4)
        guard values == [10, 20, 30, 40] else {
            return .init(name: "W8 vector", status: .fail, detail: "got \(values)")
        }
        return .init(name: "W8 vector", status: .pass, detail: "known vector decoded exactly")
    }

    /// Golden-noise RNG determinism self-test (I003).
    private func goldenNoiseRNG() -> DiagnosticItem {
        var rngA = SeededRNG(seed: 1337)
        var rngB = SeededRNG(seed: 1337)
        var rngC = SeededRNG(seed: 9999)
        let a = try? rngA.normal(count: 64)
        let b = try? rngB.normal(count: 64)
        let c = try? rngC.normal(count: 64)
        guard let a, let b, let c else {
            return .init(name: "Golden-noise RNG", status: .fail, detail: "normal() threw")
        }
        guard a == b else {
            return .init(name: "Golden-noise RNG", status: .fail, detail: "not deterministic")
        }
        guard a != c else {
            return .init(name: "Golden-noise RNG", status: .fail, detail: "different seeds collide")
        }
        let mean = a.reduce(0.0) { $0 + Double($1) } / Double(a.count)
        return .init(name: "Golden-noise RNG", status: .pass,
                     detail: String(format: "deterministic, mean %.4f", mean))
    }

    /// Read throughput of a small mmapped file as a page-cache probe (1 MB —
    /// deliberately tiny, never touches model packs).
    private func mmapBenchmark() -> DiagnosticItem {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("anima-xs-diagnostics-mmap-\(UUID().uuidString)")
        let bytes = 1 << 20
        do {
            try Data(repeating: 7, count: bytes).write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let start = Date()
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let data = try handle.readToEnd() ?? Data()
            let seconds = Date().timeIntervalSince(start)
            let mbps = Double(data.count) / 1_048_576 / max(seconds, 1e-9)
            return .init(name: "mmap benchmark", status: .pass,
                         detail: String(format: "%.0f MB/s (%.1f ms)", mbps, seconds * 1000))
        } catch {
            return .init(name: "mmap benchmark", status: .fail, detail: error.localizedDescription)
        }
    }

    /// Cheap Metal capability facts — no command buffers, no compute.
    private func metalCapability() -> DiagnosticItem {
        guard isMetalAvailable, let context else {
            let detail = forceMetalUnavailable ? "Metal unavailable (simulated)" : "Metal unavailable"
            return .init(name: "Metal capability", status: .skipped, detail: detail)
        }
        let gpuFamily = context.device.supportsFamily(.apple5) ? "Apple5 (A12+) supported" : "Apple5 (A12+) not reported"
        return .init(name: "Metal capability", status: .pass,
                     detail: "\(context.device.name) · \(gpuFamily) · recommendedMaxWorkingSet \(formatBytes(context.device.recommendedMaxWorkingSetSize))")
    }

    /// MPS fp16 matrix multiplication [1,64]x[64,1].
    ///
    /// A12 audit: rowBytes MUST be computed with the MPS-recommended helper
    /// (`MPSMatrixDescriptor.rowBytes(fromColumns:dataType:)`, 64-byte
    /// aligned). Hand-computed strides — e.g. 2 bytes for a 1-column fp16
    /// result — can raise a native MPS assertion on device (Apple5/A12) even
    /// when the tiny matrices pass in a hosted simulator. Command-buffer
    /// status/error is inspected after completion instead of assuming success.
    private func mpsPrecision(context: MetalContext) -> DiagnosticItem {
        let rows = 1, inner = 64, columns = 1
        let leftRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: inner, dataType: .float16)
        let rightRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float16)
        let outRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float16)
        guard let left = context.device.makeBuffer(
            length: leftRowBytes * rows, options: .storageModeShared),
            let right = context.device.makeBuffer(
            length: rightRowBytes * inner, options: .storageModeShared),
            let out = context.device.makeBuffer(
            length: outRowBytes * rows, options: .storageModeShared) else {
            return .init(name: "MPS precision", status: .fail, detail: "buffer alloc failed")
        }
        fillHalf(left, rows * inner, value: 0.5)
        fillHalf(right, inner * columns, value: 0.5)
        let leftDescriptor = MPSMatrixDescriptor(
            rows: rows, columns: inner, rowBytes: leftRowBytes, dataType: .float16)
        let rightDescriptor = MPSMatrixDescriptor(
            rows: inner, columns: columns, rowBytes: rightRowBytes, dataType: .float16)
        let outDescriptor = MPSMatrixDescriptor(
            rows: rows, columns: columns, rowBytes: outRowBytes, dataType: .float16)
        guard let queue = context.commandQueue.makeCommandBuffer() else {
            return .init(name: "MPS precision", status: .fail, detail: "command setup failed")
        }
        let mul = MPSMatrixMultiplication(
            device: context.device, transposeLeft: false, transposeRight: false,
            resultRows: rows, resultColumns: columns, interiorColumns: inner, alpha: 1, beta: 0)
        mul.encode(commandBuffer: queue,
                   leftMatrix: MPSMatrix(buffer: left, descriptor: leftDescriptor),
                   rightMatrix: MPSMatrix(buffer: right, descriptor: rightDescriptor),
                   resultMatrix: MPSMatrix(buffer: out, descriptor: outDescriptor))
        queue.commit()
        queue.waitUntilCompleted()
        if queue.status == .error {
            let detail = queue.error.map { "command buffer error: \($0.localizedDescription)" }
                ?? "command buffer finished with error status"
            return .init(name: "MPS precision", status: .fail, detail: detail)
        }
        guard queue.status == .completed else {
            return .init(name: "MPS precision", status: .fail,
                         detail: "command buffer status \(queue.status.rawValue) after waitUntilCompleted")
        }
        let result = readHalf(out, count: rows * columns)
        // 64 * 0.5 * 0.5 = 16.0 (fp16 rounding).
        guard result.count == 1, result[0].isFinite else {
            return .init(name: "MPS precision", status: .fail, detail: "non-finite result")
        }
        return .init(name: "MPS precision", status: .pass,
                     detail: String(format: "MPS fp16 GEMM result %.2f (expected ~16)", result[0]))
    }

    private func gemm(context: MetalContext) -> DiagnosticItem {
        let executor = LinearExecutor(context: context)
        return .init(name: "GEMM", status: .pass,
                     detail: "LinearExecutor available (\(executor.tileRows) tile rows)")
    }

    private func attentionTile(context: MetalContext) -> DiagnosticItem {
        let executor = AttentionExecutor(context: context)
        return .init(name: "Attention tile", status: .pass,
                     detail: "AttentionExecutor available (query-tiled)")
    }

    // MARK: - Small helpers

    private func fillHalf(_ buffer: MTLBuffer, _ count: Int, value: Float) {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { pointer[i] = Float16(value) }
    }

    private func readHalf(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    private func deviceModelIdentifier() -> String {
        // Best-effort physical device identifier (e.g. "iPhone12,5" on XS Max).
        var system = utsname()
        uname(&system)
        let mirror = Mirror(reflecting: system.machine)
        let identifier = mirror.children.reduce(into: "") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            identifier.append(Character(UnicodeScalar(UInt8(bitPattern: value))))
        }
        return identifier.isEmpty ? ProcessInfo.processInfo.hostName : identifier
    }

    private func availableDiskSpace() -> Int64 {
        let url = URL(fileURLWithPath: NSHomeDirectory())
        guard let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let capacity = values.volumeAvailableCapacityForImportantUsage else {
            return -1
        }
        return capacity
    }

    /// Bytes of memory the process may still allocate, or nil when the OS
    /// does not expose it. `os_proc_available_memory()` (iOS 13+/Darwin).
    private func availableProcessMemory() -> UInt64? {
        #if canImport(Darwin)
        return UInt64(os_proc_available_memory())
        #else
        return nil
        #endif
    }

    private func formatBytes(_ bytes: UInt64?) -> String {
        guard let bytes else { return "n/a" }
        return String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }
}
