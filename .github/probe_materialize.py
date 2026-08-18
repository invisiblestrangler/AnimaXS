from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing patch anchor in {path}: {old[:120]!r}")
    if text.count(old) != 1:
        raise SystemExit(f"non-unique patch anchor in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


# Objective-C bridge: diagnostic-only private-runtime capability/procedure facts.
replace_once(
    "AnimaXS/Runtime/ANE/A12ANEBridge.h",
    'FOUNDATION_EXPORT NSString *A12ANERuntimeStatus(void);\n',
    'FOUNDATION_EXPORT NSString *A12ANERuntimeStatus(void);\n'
    '/// Diagnostic-only selector inventory for the private client.\n'
    'FOUNDATION_EXPORT NSString *A12ANEClientCapabilitySummary(void);\n')

replace_once(
    "AnimaXS/Runtime/ANE/A12ANEBridge.h",
    '@property(nonatomic, readonly) NSString *label;\n\n/// qBytes is row-major U8 [outputChannels,inputChannels].',
    '@property(nonatomic, readonly) NSString *label;\n'
    '/// Diagnostic metadata reported by the loaded ANE program.\n'
    '@property(nonatomic, readonly) NSUInteger procedureCount;\n'
    '@property(nonatomic, readonly) NSString *procedureSummary;\n\n'
    '/// qBytes is row-major U8 [outputChannels,inputChannels].')

replace_once(
    "AnimaXS/Runtime/ANE/A12ANEBridge.m",
    '''NSString *A12ANERuntimeStatus(void) {
    if (!A12LoadANE()) return @"AppleNeuralEngine private framework unavailable";
    if (!A12IOSurface().ok) return @"IOSurface runtime unavailable";
    NSArray<NSString *> *names = @[@"_ANEModel", @"_ANEClient", @"_ANERequest", @"_ANEIOSurfaceObject"];
    for (NSString *name in names) if (!NSClassFromString(name)) return [@"Missing runtime class " stringByAppendingString:name];
    return @"available";
}
''',
    '''NSString *A12ANERuntimeStatus(void) {
    if (!A12LoadANE()) return @"AppleNeuralEngine private framework unavailable";
    if (!A12IOSurface().ok) return @"IOSurface runtime unavailable";
    NSArray<NSString *> *names = @[@"_ANEModel", @"_ANEClient", @"_ANERequest", @"_ANEIOSurfaceObject"];
    for (NSString *name in names) if (!NSClassFromString(name)) return [@"Missing runtime class " stringByAppendingString:name];
    return @"available";
}

NSString *A12ANEClientCapabilitySummary(void) {
    if (!A12LoadANE()) return @"ANE framework unavailable";
    Class clientClass = NSClassFromString(@"_ANEClient");
    if (!clientClass) return @"_ANEClient unavailable";
    BOOL newInstance = class_getInstanceMethod(
        clientClass, NSSelectorFromString(@"loadModelNewInstance:options:modelInstParams:qos:error:")) != NULL;
    BOOL chaining = class_getInstanceMethod(
        clientClass, NSSelectorFromString(@"prepareChainingWithModel:options:chainingReq:qos:error:")) != NULL;
    BOOL unload = class_getInstanceMethod(
        clientClass, NSSelectorFromString(@"unloadModel:options:qos:error:")) != NULL;
    return [NSString stringWithFormat:@"loadModelNewInstance=%@ prepareChaining=%@ unloadModel=%@",
        newInstance ? @"yes" : @"no",
        chaining ? @"yes" : @"no",
        unload ? @"yes" : @"no"];
}
''')

projection_eval_anchor = '- (BOOL)evaluateInput:(A12ANESurface *)input output:(A12ANESurface *)output milliseconds:(double *)milliseconds error:(NSError **)error {\n'
procedure_getters = '''- (NSUInteger)procedureCount {
    NSDictionary *attrs = A12ModelAttributes(_model);
    id descValue = attrs[@"ANEFModelDescription"];
    NSDictionary *desc = [descValue isKindOfClass:NSDictionary.class] ? descValue : @{};
    id proceduresValue = desc[@"ANEFModelProcedures"];
    NSArray *procedures = [proceduresValue isKindOfClass:NSArray.class] ? proceduresValue : @[];
    return procedures.count;
}

- (NSString *)procedureSummary {
    NSDictionary *attrs = A12ModelAttributes(_model);
    id descValue = attrs[@"ANEFModelDescription"];
    NSDictionary *desc = [descValue isKindOfClass:NSDictionary.class] ? descValue : @{};
    id namesValue = desc[@"kANEFModelProcedureNameToIDMapKey"];
    NSDictionary *names = [namesValue isKindOfClass:NSDictionary.class] ? namesValue : @{};
    return [NSString stringWithFormat:@"count=%lu names=%@",
        (unsigned long)self.procedureCount, A12String(names)];
}

'''
replace_once(
    "AnimaXS/Runtime/ANE/A12ANEBridge.m",
    projection_eval_anchor,
    procedure_getters + projection_eval_anchor)

# UI: one explicit one-build device probe, kept in an existing source file so
# XcodeGen/project membership remains unchanged.
replace_once(
    "AnimaXS/App/DiagnosticsView.swift",
    'import SwiftUI\nimport UniformTypeIdentifiers\n',
    'import SwiftUI\nimport UniformTypeIdentifiers\nimport UIKit\nimport Dispatch\n#if canImport(Darwin)\nimport Darwin\n#endif\n')

replace_once(
    "AnimaXS/App/DiagnosticsView.swift",
    '    @State private var exportPresented = false\n',
    '    @State private var exportPresented = false\n'
    '    @State private var aneProbeText: String?\n')

replace_once(
    "AnimaXS/App/DiagnosticsView.swift",
    '            hardwareTestsSection\n            deepIntegritySection\n',
    '            hardwareTestsSection\n            aneResidencyProbeSection\n            deepIntegritySection\n')

ane_section = '''    @ViewBuilder
    private var aneResidencyProbeSection: some View {
        Section("ANE residency / load probe") {
            Text("Device-only research probe for the ANE-native W8 backend. It progressively loads the same 8 ANE programs per block used by production, evaluates them once, records app/Metal memory plus system pressure events, releases everything, then reloads one complete block repeatedly. It never runs automatically and does not modify model weights.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(isRunning && currentTest == "ANE residency/load probe"
                   ? "Running ANE probe…" : "Run ANE residency/load probe") {
                Task { await runANEProbe() }
            }
            .disabled(isRunning || isGenerating)
            if isGenerating {
                Text("Finish or cancel the active generation before running the ANE probe.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if let aneProbeText {
                Text(aneProbeText)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                ShareLink(item: aneProbeText) {
                    Label("Copy / share ANE probe", systemImage: "square.and.arrow.up")
                }
                .font(.caption)
            }
        }
    }

'''
replace_once(
    "AnimaXS/App/DiagnosticsView.swift",
    '    private var deepIntegritySection: some View {\n',
    ane_section + '    private var deepIntegritySection: some View {\n')

run_probe = '''    private func runANEProbe() async {
        guard !isRunning, !isGenerating else { return }
        isRunning = true
        currentTest = "ANE residency/load probe"
        aneProbeText = "Running…"
        marker.beginSession()
        marker.markStarted("ANE residency/load probe")
        let result = await Task.detached(priority: .userInitiated) {
            await ANEResidencyProbe.run()
        }.value
        aneProbeText = result
        marker.markCompleted("ANE residency/load probe")
        marker.markSessionClean()
        isRunning = false
        currentTest = nil
    }

'''
replace_once(
    "AnimaXS/App/DiagnosticsView.swift",
    '    private func runDeep() async {\n',
    run_probe + '    private func runDeep() async {\n')

helper = r'''/// One-build physical-device probe for private ANE model loading and
/// residency. It intentionally lives in DiagnosticsView.swift so adding the
/// research probe does not alter project/source membership or production
/// generation behavior.
private enum ANEResidencyProbe {
    private struct MemorySnapshot {
        let available: UInt64
        let footprint: UInt64?
        let metal: UInt64
        let thermal: String

        static func capture(context: MetalContext) -> MemorySnapshot {
            context.refreshDiagnostics()
            return MemorySnapshot(
                available: UInt64(os_proc_available_memory()),
                footprint: processPhysicalFootprint(),
                metal: context.currentAllocatedSize,
                thermal: String(describing: ProcessInfo.processInfo.thermalState))
        }

        var compact: String {
            let footprintText = footprint.map { mb($0) } ?? "n/a"
            return "avail=\(mb(available)) footprint=\(footprintText) Metal=\(mb(metal)) thermal=\(thermal)"
        }
    }

    private final class PressureRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private let source: DispatchSourceMemoryPressure
        private var observer: NSObjectProtocol?
        private var dispatchEvents: [String] = []
        private var uiWarningCount = 0
        private var didSeeCritical = false
        private var stopped = false

        init() {
            source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.normal, .warning, .critical],
                queue: DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-pressure-probe"))
            source.setEventHandler { [weak self] in
                guard let self else { return }
                let event = self.source.data
                var labels: [String] = []
                if event.contains(.normal) { labels.append("normal") }
                if event.contains(.warning) { labels.append("warning") }
                if event.contains(.critical) { labels.append("critical") }
                let stamp = String(format: "%.3f", ProcessInfo.processInfo.systemUptime)
                self.lock.lock()
                self.dispatchEvents.append("\(stamp):\(labels.joined(separator: "+"))")
                if event.contains(.critical) { self.didSeeCritical = true }
                self.lock.unlock()
            }
            observer = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil, queue: nil
            ) { [weak self] _ in
                guard let self else { return }
                self.lock.lock()
                self.uiWarningCount += 1
                self.lock.unlock()
            }
            source.resume()
        }

        var criticalSeen: Bool {
            lock.lock(); defer { lock.unlock() }
            return didSeeCritical
        }

        var compact: String {
            lock.lock(); defer { lock.unlock() }
            let events = dispatchEvents.isEmpty ? "none" : dispatchEvents.joined(separator: ",")
            return "UIKitWarnings=\(uiWarningCount) dispatch=\(events)"
        }

        func stop() {
            lock.lock()
            if stopped { lock.unlock(); return }
            stopped = true
            let token = observer
            observer = nil
            lock.unlock()
            source.cancel()
            if let token { NotificationCenter.default.removeObserver(token) }
        }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE residency/load probe v1",
            "Policy: continue through ordinary memory warnings; stop progressive loading on dispatch critical pressure."
        ]
        let pressure = PressureRecorder()
        defer { pressure.stop() }

        do {
            guard A12ANEIsAvailable() else {
                throw AnimapkError.validation("ANE unavailable: \(A12ANERuntimeStatus())")
            }
            guard let context = MetalContext() else {
                throw AnimapkError.validation("Metal unavailable")
            }
            guard let ditEntry = ModelManifest.entries.first(where: { $0.component == .dit }) else {
                throw AnimapkError.validation("DiT manifest entry missing")
            }
            let store = try ModelStore()
            let ditURL = await store.localURL(for: ditEntry)
            guard FileManager.default.fileExists(atPath: ditURL.path) else {
                throw AnimapkError.validation("installed DiT pack missing")
            }
            let file = try AnimapkFile(url: ditURL)
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                throw AnimapkError.validation("ANE prepared-model cache is incomplete")
            }

            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("baseline: \(MemorySnapshot.capture(context: context).compact)")
            lines.append("")
            lines.append("Progressive production residency (8 programs/block, one evaluation/program)")

            var cache: ANEW8DiTModelCache? = try ANEW8DiTModelCache(file: file)
            let surfaces = try ANEW8DiTSurfaces(device: context.device)
            var logicalBytes: UInt64 = 0
            var loadedBlocks = 0

            for block in 0..<ModelConstants.ditBlocks {
                guard let cache else { break }
                let wallStart = ProcessInfo.processInfo.systemUptime
                let result = try cache.models(for: block)
                let loadWallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
                let evalMS = try evaluateAll(result.models, surfaces: surfaces)
                loadedBlocks = block + 1
                logicalBytes += try logicalNativeBytes(file: file, block: block)
                let snapshot = MemorySnapshot.capture(context: context)
                lines.append(String(
                    format: "b%02d programs=%3d logical=%7.0fMB loadANE=%7.1fms loadWall=%7.1fms eval=%6.1fms %@ | %@",
                    block, loadedBlocks * 8,
                    Double(logicalBytes) / 1_048_576,
                    result.newlyLoadedMilliseconds, loadWallMS, evalMS,
                    snapshot.compact, pressure.compact))

                if block == 0 {
                    lines.append("block0 selfO procedure metadata: \(result.models.selfO.procedureSummary)")
                }
                if pressure.criticalSeen {
                    lines.append("STOP: dispatch critical memory pressure observed after block \(block).")
                    break
                }
            }

            lines.append("")
            lines.append("Release/recovery")
            cache = nil
            lines.append("release+0ms: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            try? await Task.sleep(nanoseconds: 250_000_000)
            lines.append("release+250ms: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            try? await Task.sleep(nanoseconds: 750_000_000)
            lines.append("release+1000ms: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            lines.append("release+3000ms: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")

            if !pressure.criticalSeen {
                lines.append("")
                lines.append("Warm full-block reload (new cache object each cycle)")
                for cycle in 1...3 {
                    var cycleCache: ANEW8DiTModelCache? = try ANEW8DiTModelCache(file: file)
                    var cycleResult: (models: ANEW8DiTModels, newlyLoadedMilliseconds: Double)? = try cycleCache!.models(for: 0)
                    let wallStart = ProcessInfo.processInfo.systemUptime
                    let loadANE = cycleResult!.newlyLoadedMilliseconds
                    let evalMS = try evaluateAll(cycleResult!.models, surfaces: surfaces)
                    let wallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
                    lines.append(String(
                        format: "cycle%d loadANE=%7.1fms postLoadEvalWall=%7.1fms eval=%6.1fms %@ | %@",
                        cycle, loadANE, wallMS, evalMS,
                        MemorySnapshot.capture(context: context).compact, pressure.compact))
                    cycleResult = nil
                    cycleCache = nil
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
            } else {
                lines.append("Warm reload skipped because critical pressure was observed.")
            }

            lines.append("")
            lines.append("final pressure: \(pressure.compact)")
        } catch {
            lines.append("ERROR: \(error.localizedDescription)")
            lines.append("pressure at error: \(pressure.compact)")
        }
        return lines.joined(separator: "\n")
    }

    private static func evaluateAll(
        _ models: ANEW8DiTModels, surfaces: ANEW8DiTSurfaces
    ) throws -> Double {
        var totalMS = 0.0
        var qkvMS = 0.0
        _ = try models.selfQKV.evaluateInput(
            surfaces.tokenInput,
            qOutput: surfaces.q, kOutput: surfaces.k, vOutput: surfaces.v,
            milliseconds: &qkvMS)
        totalMS += qkvMS

        func projection(
            _ model: A12ANEProjectionModel,
            _ input: A12ANESurface,
            _ output: A12ANESurface
        ) throws {
            var ms = 0.0
            _ = try model.evaluateInput(input, output: output, milliseconds: &ms)
            totalMS += ms
        }

        try projection(models.selfO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.crossQ, surfaces.tokenInput, surfaces.q)
        try projection(models.crossK, surfaces.contextInput, surfaces.contextK)
        try projection(models.crossV, surfaces.contextInput, surfaces.contextV)
        try projection(models.crossO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.mlpUp, surfaces.tokenInput, surfaces.hidden)
        try projection(models.mlpDown, surfaces.hidden, surfaces.tokenOutput)
        return totalMS
    }

    private static func logicalNativeBytes(file: AnimapkFile, block: Int) throws -> UInt64 {
        var total: UInt64 = 0
        for spec in ANEW8NativePack.projectionSpecs {
            let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: spec.suffix)
            total += tensor.dataSize + tensor.scaleSize + tensor.zeroSize
        }
        return total
    }

    private static func mb(_ bytes: UInt64) -> String {
        String(format: "%.0fMB", Double(bytes) / 1_048_576)
    }

    private static func processPhysicalFootprint() -> UInt64? {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
        #else
        return nil
        #endif
    }
}

'''
replace_once(
    "AnimaXS/App/DiagnosticsView.swift",
    '/// Minimal file wrapper so the JSON can be shared via SwiftUI fileExporter.\n',
    helper + '/// Minimal file wrapper so the JSON can be shared via SwiftUI fileExporter.\n')

# Remove both transport mechanisms from the materialized source commit.
for transient in [
    Path('.github/workflows/materialize-ane-residency-probe.yml'),
    Path('.github/probe_materialize.py'),
]:
    if transient.exists():
        transient.unlink()
