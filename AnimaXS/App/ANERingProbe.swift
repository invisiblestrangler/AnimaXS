import Foundation
import Dispatch
import Metal
import UIKit
#if canImport(Darwin)
import Darwin
#endif

/// V6 device-only recipe search for the A12/H11 native W8 backend.
///
/// V5 established that a depth-3 stream can hide post-destroy loader waits and
/// that 2 pinned + 3 streaming (40 resident programs) beat the 0+2/0+3 rings.
/// V6 removes V5's fixed-order bias, measures bounded asynchronous retirement,
/// and models the exact P5 cross-K/V-warm state where crossK/crossV are no
/// longer needed after the first diffusion step.
///
/// No production generation path is changed by this probe.
enum ANERingProbe {
    private static let blockCount = ModelConstants.ditBlocks
    private static let depth = 3
    private static let programCeiling = 80

    private enum Profile: String, Sendable {
        case full8
        case kvWarm6

        var programs: Int { self == .full8 ? 8 : 6 }
        var includesCrossKV: Bool { self == .full8 }
        var label: String { self == .full8 ? "full8" : "kvWarm6" }
    }

    private struct PressureMark { let issues: Int }

    private final class PressureRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private let source: DispatchSourceMemoryPressure
        private var observer: NSObjectProtocol?
        private var dispatchEvents: [String] = []
        private var uiWarnings = 0
        private var stopped = false

        init() {
            source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.normal, .warning, .critical],
                queue: DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-v6-pressure"))
            source.setEventHandler { [weak self] in
                guard let self else { return }
                let data = self.source.data
                var labels: [String] = []
                if data.contains(.normal) { labels.append("normal") }
                if data.contains(.warning) { labels.append("warning") }
                if data.contains(.critical) { labels.append("critical") }
                self.lock.lock()
                self.dispatchEvents.append(labels.joined(separator: "+"))
                self.lock.unlock()
            }
            observer = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil, queue: nil
            ) { [weak self] _ in
                guard let self else { return }
                self.lock.lock()
                self.uiWarnings += 1
                self.lock.unlock()
            }
            source.resume()
        }

        var compact: String {
            lock.lock(); defer { lock.unlock() }
            let events = dispatchEvents.isEmpty ? "none" : dispatchEvents.joined(separator: ",")
            return "UIKitWarnings=\(uiWarnings) dispatch=\(events)"
        }

        var criticalSeen: Bool {
            lock.lock(); defer { lock.unlock() }
            return dispatchEvents.contains { $0.contains("critical") }
        }

        var issueCount: Int {
            lock.lock(); defer { lock.unlock() }
            let dispatchIssues = dispatchEvents.filter {
                $0.contains("warning") || $0.contains("critical")
            }.count
            return uiWarnings + dispatchIssues
        }

        func mark() -> PressureMark { PressureMark(issues: issueCount) }
        func newIssues(since mark: PressureMark) -> Int { max(0, issueCount - mark.issues) }

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
            let fp = footprint.map { mb($0) } ?? "n/a"
            return "avail=\(mb(available)) footprint=\(fp) Metal=\(mb(metal)) thermal=\(thermal)"
        }
    }

    /// Tracks actual loaded ANE program count, not blockCount*8. This matters
    /// for the KV-warm six-program profile.
    private final class ResidencyTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var programsByBlock: [Int: Int] = [:]
        private var peakProgramsValue = 0
        private var peakBlocksValue = 0

        func add(block: Int, programs: Int) throws {
            lock.lock()
            if programsByBlock[block] != nil {
                lock.unlock()
                throw AnimapkError.validation("V6 duplicate resident block \(block)")
            }
            programsByBlock[block] = programs
            let total = programsByBlock.values.reduce(0, +)
            peakProgramsValue = max(peakProgramsValue, total)
            peakBlocksValue = max(peakBlocksValue, programsByBlock.count)
            if total > programCeiling {
                programsByBlock.removeValue(forKey: block)
                lock.unlock()
                throw AnimapkError.validation(
                    "V6 residency exceeded \(programCeiling)-program ceiling: \(total)")
            }
            lock.unlock()
        }

        func remove(_ block: Int) {
            lock.lock(); programsByBlock.removeValue(forKey: block); lock.unlock()
        }

        func resetPeak() {
            lock.lock()
            peakProgramsValue = programsByBlock.values.reduce(0, +)
            peakBlocksValue = programsByBlock.count
            lock.unlock()
        }

        var currentPrograms: Int {
            lock.lock(); defer { lock.unlock() }
            return programsByBlock.values.reduce(0, +)
        }

        var currentBlocks: Int {
            lock.lock(); defer { lock.unlock() }
            return programsByBlock.count
        }

        var peakPrograms: Int {
            lock.lock(); defer { lock.unlock() }
            return peakProgramsValue
        }

        var peakBlocks: Int {
            lock.lock(); defer { lock.unlock() }
            return peakBlocksValue
        }

        var compact: String {
            lock.lock(); defer { lock.unlock() }
            let programs = programsByBlock.values.reduce(0, +)
            return "resident=\(programsByBlock.count) blocks/\(programs) programs peak=\(peakBlocksValue) blocks/\(peakProgramsValue) programs"
        }
    }

    private final class ModelSet: @unchecked Sendable {
        let selfQKV: A12ANEQKVModel
        let selfO: A12ANEProjectionModel
        let crossQ: A12ANEProjectionModel
        let crossK: A12ANEProjectionModel?
        let crossV: A12ANEProjectionModel?
        let crossO: A12ANEProjectionModel
        let mlpUp: A12ANEProjectionModel
        let mlpDown: A12ANEProjectionModel
        let profile: Profile

        init(
            selfQKV: A12ANEQKVModel,
            selfO: A12ANEProjectionModel,
            crossQ: A12ANEProjectionModel,
            crossK: A12ANEProjectionModel?,
            crossV: A12ANEProjectionModel?,
            crossO: A12ANEProjectionModel,
            mlpUp: A12ANEProjectionModel,
            mlpDown: A12ANEProjectionModel,
            profile: Profile
        ) {
            self.selfQKV = selfQKV
            self.selfO = selfO
            self.crossQ = crossQ
            self.crossK = crossK
            self.crossV = crossV
            self.crossO = crossO
            self.mlpUp = mlpUp
            self.mlpDown = mlpDown
            self.profile = profile
        }

        var programCount: Int { profile.programs }

        var loadMilliseconds: Double {
            selfQKV.loadMilliseconds + selfO.loadMilliseconds + crossQ.loadMilliseconds
                + (crossK?.loadMilliseconds ?? 0) + (crossV?.loadMilliseconds ?? 0)
                + crossO.loadMilliseconds + mlpUp.loadMilliseconds + mlpDown.loadMilliseconds
        }

        func invalidateTimed() -> (callSum: Double, wall: Double) {
            let wallStart = ProcessInfo.processInfo.systemUptime
            var sum = 0.0
            func destroy(_ body: () -> Void) {
                let start = ProcessInfo.processInfo.systemUptime
                body()
                sum += elapsedMS(since: start)
            }
            destroy { selfQKV.invalidate() }
            destroy { selfO.invalidate() }
            destroy { crossQ.invalidate() }
            if let crossK { destroy { crossK.invalidate() } }
            if let crossV { destroy { crossV.invalidate() } }
            destroy { crossO.invalidate() }
            destroy { mlpUp.invalidate() }
            destroy { mlpDown.invalidate() }
            return (sum, elapsedMS(since: wallStart))
        }
    }

    private final class LoadedBlock: @unchecked Sendable {
        let block: Int
        let models: ModelSet
        let loadANE: Double
        let loadWall: Double
        private let tracker: ResidencyTracker
        private let lock = NSLock()
        private var destroyed = false

        init(
            block: Int, models: ModelSet, loadANE: Double, loadWall: Double,
            tracker: ResidencyTracker
        ) throws {
            self.block = block
            self.models = models
            self.loadANE = loadANE
            self.loadWall = loadWall
            self.tracker = tracker
            try tracker.add(block: block, programs: models.programCount)
        }

        var hostLoadWall: Double { max(0, loadWall - loadANE) }

        func destroyTimed() -> (callSum: Double, wall: Double) {
            lock.lock()
            if destroyed { lock.unlock(); return (0, 0) }
            destroyed = true
            lock.unlock()
            let result = models.invalidateTimed()
            tracker.remove(block)
            return result
        }

        func destroyUntimed() {
            lock.lock()
            if destroyed { lock.unlock(); return }
            destroyed = true
            lock.unlock()
            models.selfQKV.invalidate()
            models.selfO.invalidate()
            models.crossQ.invalidate()
            models.crossK?.invalidate()
            models.crossV?.invalidate()
            models.crossO.invalidate()
            models.mlpUp.invalidate()
            models.mlpDown.invalidate()
            tracker.remove(block)
        }

        deinit { destroyUntimed() }
    }

    private final class SharedLoader: @unchecked Sendable {
        private let file: AnimapkFile
        let tracker = ResidencyTracker()

        init(file: AnimapkFile) throws {
            self.file = file
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                throw AnimapkError.validation("ANE prepared-model cache is incomplete")
            }
            guard A12ANEIsAvailable() else {
                throw AnimapkError.validation(
                    "A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
            }
        }

        func load(block: Int, profile: Profile) throws -> LoadedBlock {
            guard (0..<blockCount).contains(block) else {
                throw AnimapkError.validation("V6 block index out of range: \(block)")
            }
            let started = ProcessInfo.processInfo.systemUptime

            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(
                    file: file, block: block, suffix: suffix)
                guard let digest = tensor.blobSHA256 else {
                    throw AnimapkError.validation("ANE-native tensor hash missing")
                }
                return digest
            }

            func projection(_ suffix: String) throws -> A12ANEProjectionModel {
                guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                    throw AnimapkError.validation(
                        "unknown ANE projection suffix: \(suffix)")
                }
                let key = ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: try digest(suffix))
                return try A12ANEProjectionModel(
                    preparedInputChannels: UInt(spec.columns),
                    outputChannels: UInt(spec.rows),
                    spatial: UInt(spec.spatial),
                    label: "v6_b\(block)_\(spec.tag)",
                    cacheKey: key)
            }

            let qHash = try digest("self_attn.q_proj.weight")
            let kHash = try digest("self_attn.k_proj.weight")
            let vHash = try digest("self_attn.v_proj.weight")
            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block, q: qHash, k: kHash, v: vHash)
            let selfQKV = try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(DiTBlockExecutor.tokens),
                label: "v6_b\(block)_self_qkv",
                cacheKey: qkvKey)

            let crossK = profile.includesCrossKV
                ? try projection("cross_attn.k_proj.weight") : nil
            let crossV = profile.includesCrossKV
                ? try projection("cross_attn.v_proj.weight") : nil

            let models = ModelSet(
                selfQKV: selfQKV,
                selfO: try projection("self_attn.output_proj.weight"),
                crossQ: try projection("cross_attn.q_proj.weight"),
                crossK: crossK,
                crossV: crossV,
                crossO: try projection("cross_attn.output_proj.weight"),
                mlpUp: try projection("mlp.layer1.weight"),
                mlpDown: try projection("mlp.layer2.weight"),
                profile: profile)
            let wall = elapsedMS(since: started)
            return try LoadedBlock(
                block: block, models: models,
                loadANE: models.loadMilliseconds, loadWall: wall,
                tracker: tracker)
        }
    }

    private final class LoadResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<LoadedBlock, Error>?
        func set(_ value: Result<LoadedBlock, Error>) {
            lock.lock(); result = value; lock.unlock()
        }
        func take() -> Result<LoadedBlock, Error>? {
            lock.lock(); defer { lock.unlock() }
            let value = result
            result = nil
            return value
        }
    }

    private final class LoadFuture: @unchecked Sendable {
        let block: Int
        private let semaphore = DispatchSemaphore(value: 0)
        private let box = LoadResultBox()

        init(
            block: Int, profile: Profile, loader: SharedLoader,
            queue: DispatchQueue
        ) {
            self.block = block
            queue.async {
                do { self.box.set(.success(try loader.load(block: block, profile: profile))) }
                catch { self.box.set(.failure(error)) }
                self.semaphore.signal()
            }
        }

        func wait() throws -> LoadedBlock {
            semaphore.wait()
            guard let result = box.take() else {
                throw AnimapkError.validation(
                    "V6 loader returned no result for block \(block)")
            }
            return try result.get()
        }
    }

    private final class RetireResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: (callSum: Double, wall: Double)?
        func set(_ newValue: (callSum: Double, wall: Double)) {
            lock.lock(); value = newValue; lock.unlock()
        }
        func take() -> (callSum: Double, wall: Double)? {
            lock.lock(); defer { lock.unlock() }
            let result = value
            value = nil
            return result
        }
    }

    private final class RetireFuture: @unchecked Sendable {
        private let semaphore = DispatchSemaphore(value: 0)
        private let box = RetireResultBox()

        init(block: LoadedBlock, queue: DispatchQueue) {
            queue.async {
                self.box.set(block.destroyTimed())
                self.semaphore.signal()
            }
        }

        func wait() throws -> (callSum: Double, wall: Double) {
            semaphore.wait()
            guard let result = box.take() else {
                throw AnimapkError.validation("V6 retire queue returned no result")
            }
            return result
        }
    }

    private struct PassStats {
        var wall = 0.0
        var loadANE = 0.0
        var loadWall = 0.0
        var loadHost = 0.0
        var evalANE = 0.0
        var evalWall = 0.0
        var unloadCallSum = 0.0
        var unloadWall = 0.0
        var nextWait = 0.0
        var retireBackpressure = 0.0
        var retireDrain = 0.0
        var maxLoad = 0.0
        var maxUnload = 0.0
        var maxNextWait = 0.0
        var maxRetireWait = 0.0
        var peakPrograms = 0
        var peakBlocks = 0
        var loads = 0
        var unloads = 0

        var avgLoad: Double { loads > 0 ? loadWall / Double(loads) : 0 }
        var avgUnload: Double { unloads > 0 ? unloadWall / Double(unloads) : 0 }
    }

    private struct VariantResult {
        let profile: Profile
        let pinned: Int
        let asyncRetire: Bool
        let pinSetupWall: Double
        let stats: PassStats
        let newPressure: Int
        let memory: MemorySnapshot

        var tag: String {
            "\(profile.label) \(pinned)+3\(asyncRetire ? "+retire1" : "")"
        }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE Michelin probe v6",
            "Goal: remove V5 order bias, test bounded async retirement, and model exact P5 KV-warm six-program slices.",
            "Safety ceiling: <=80 simultaneously resident private ANE programs.",
            "full8 = selfQKV,selfO,crossQ,crossK,crossV,crossO,mlpUp,mlpDown.",
            "kvWarm6 = exact post-P5-hit dynamic set: selfQKV,selfO,crossQ,crossO,mlpUp,mlpDown.",
            "Important: projection-only timings are architecture probes, not full DiT/generation latency."
        ]
        let pressure = PressureRecorder()
        defer { pressure.stop() }

        do {
            guard A12ANEIsAvailable() else {
                throw AnimapkError.validation(
                    "ANE unavailable: \(A12ANERuntimeStatus())")
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
            let loaderStart = ProcessInfo.processInfo.systemUptime
            let loader = try SharedLoader(file: file)
            let surfaces = try ANEW8DiTSurfaces(device: context.device)
            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("shared-loader init=\(ms(elapsedMS(since: loaderStart)))ms")
            lines.append("baseline: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            lines.append("")

            // A complete first-touch is required so every prepared program in
            // both profiles is in the same translated/warm runtime state.
            lines.append("PRECONDITION — one full8 first-touch traversal")
            let cold = try precondition(
                loader: loader, surfaces: surfaces, pressure: pressure,
                context: context, lines: &lines)
            lines.append(summary("precondition", cold))
            lines.append("after precondition: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            guard !pressure.criticalSeen else {
                throw AnimapkError.validation("critical pressure during V6 precondition")
            }
            try? await Task.sleep(nanoseconds: 500_000_000)

            // Mirrored ordering controls a monotonic warm/runtime trend:
            // each pin depth appears once early and once equally late.
            lines.append("")
            lines.append("EXPERIMENT A — full8 mirrored pin-depth sweep, depth3, synchronous retirement")
            let fullOrder = [0, 2, 4, 6, 6, 4, 2, 0]
            var fullResults: [VariantResult] = []
            for (index, pinned) in fullOrder.enumerated() {
                let result = try runScheduler(
                    profile: .full8, pinned: pinned, asyncRetire: false,
                    loader: loader, surfaces: surfaces, context: context,
                    pressure: pressure)
                fullResults.append(result)
                lines.append("A\(index + 1) \(variantSummary(result))")
                guard !pressure.criticalSeen else {
                    throw AnimapkError.validation(
                        "critical pressure in full8 pinned sweep at \(pinned)+3")
                }
                try? await Task.sleep(nanoseconds: 175_000_000)
            }
            appendMirroredScorecard(
                title: "FULL8 ORDER-CONTROLLED SCORECARD",
                results: fullResults, lines: &lines)

            lines.append("")
            lines.append("EXPERIMENT B — full8 6+3 synchronous vs bounded retire1 (ABBA)")
            let fullRetireOrder = [false, true, true, false]
            var fullRetireResults: [VariantResult] = []
            for (index, asyncRetire) in fullRetireOrder.enumerated() {
                let result = try runScheduler(
                    profile: .full8, pinned: 6, asyncRetire: asyncRetire,
                    loader: loader, surfaces: surfaces, context: context,
                    pressure: pressure)
                fullRetireResults.append(result)
                lines.append("B\(index + 1) \(variantSummary(result))")
                guard !pressure.criticalSeen else {
                    throw AnimapkError.validation(
                        "critical pressure in full8 retirement experiment")
                }
                try? await Task.sleep(nanoseconds: 175_000_000)
            }
            appendRetireScorecard(
                title: "FULL8 RETIRE SCORECARD",
                results: fullRetireResults, lines: &lines)

            // Keep the real ~112 MiB P5 cache allocation alive while measuring
            // six-program residency. We do not fake its contents here; V5/P5
            // production code already owns correctness. This experiment models
            // the private-ANE load/eval/unload cost after the cache is warm.
            lines.append("")
            lines.append("EXPERIMENT C — allocate real CrossKVCache, then mirrored kvWarm6 sweep")
            guard let kvMemoryModel = CrossKVCache(device: context.device) else {
                throw AnimapkError.validation(
                    "V6 could not allocate the production CrossKVCache memory model")
            }
            defer { withExtendedLifetime(kvMemoryModel) {} }
            lines.append("KV memory model allocated: \(MemorySnapshot.capture(context: context).compact)")
            let warmOrder = [4, 6, 8, 10, 10, 8, 6, 4]
            var warmResults: [VariantResult] = []
            for (index, pinned) in warmOrder.enumerated() {
                let result = try runScheduler(
                    profile: .kvWarm6, pinned: pinned, asyncRetire: false,
                    loader: loader, surfaces: surfaces, context: context,
                    pressure: pressure)
                warmResults.append(result)
                lines.append("C\(index + 1) \(variantSummary(result))")
                guard !pressure.criticalSeen else {
                    throw AnimapkError.validation(
                        "critical pressure in kvWarm6 pinned sweep at \(pinned)+3")
                }
                try? await Task.sleep(nanoseconds: 175_000_000)
            }
            appendMirroredScorecard(
                title: "KV-WARM6 ORDER-CONTROLLED SCORECARD",
                results: warmResults, lines: &lines)

            lines.append("")
            lines.append("EXPERIMENT D — kvWarm6 8+3 synchronous vs bounded retire1 (ABBA, <=72 programs)")
            let warmRetireOrder = [false, true, true, false]
            var warmRetireResults: [VariantResult] = []
            for (index, asyncRetire) in warmRetireOrder.enumerated() {
                let result = try runScheduler(
                    profile: .kvWarm6, pinned: 8, asyncRetire: asyncRetire,
                    loader: loader, surfaces: surfaces, context: context,
                    pressure: pressure)
                warmRetireResults.append(result)
                lines.append("D\(index + 1) \(variantSummary(result))")
                guard !pressure.criticalSeen else {
                    throw AnimapkError.validation(
                        "critical pressure in kvWarm6 retirement experiment")
                }
                try? await Task.sleep(nanoseconds: 175_000_000)
            }
            appendRetireScorecard(
                title: "KV-WARM6 RETIRE SCORECARD",
                results: warmRetireResults, lines: &lines)

            let bestFull = bestMirrored(results: fullResults)
            let bestWarm = bestMirrored(results: warmResults)
            lines.append("")
            lines.append("V6 RECIPE ESTIMATE — projection traversal only")
            if let bestFull, let bestWarm {
                let first = median(bestFull.map(\.stats.wall))
                let later = median(bestWarm.map(\.stats.wall))
                let projection8 = first + 7.0 * later
                lines.append(
                    "best full8 pin=\(bestFull[0].pinned): median=\(ms(first))ms/traversal")
                lines.append(
                    "best kvWarm6 pin=\(bestWarm[0].pinned): median=\(ms(later))ms/traversal")
                lines.append(
                    "first-step full8 + seven KV-warm6 traversals = \(seconds(projection8))s projection-only.")
                lines.append(
                    "Do NOT compare this directly with full generation time; Metal attention/AdaLN/GELU/final-layer work is outside this probe.")
            }

            lines.append("")
            lines.append("MULTI-PROCEDURE STATUS")
            lines.append(
                "V5 public MLProgram had two functions, but the current private constructor is Espresso-specific and failed looking for model.espresso.net.")
            lines.append(
                "V6 intentionally does not repeat that invalid MLProgram->Espresso load. Next procedure work must target a true Espresso/ANEF multi-procedure container or a different private constructor.")
            lines.append("")
            lines.append(
                "final: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact) | \(loader.tracker.compact)")
        } catch {
            lines.append("ERROR: \(error.localizedDescription)")
            lines.append("pressure at error: \(pressure.compact)")
        }
        return lines.joined(separator: "\n")
    }

    private static func precondition(
        loader: SharedLoader,
        surfaces: ANEW8DiTSurfaces,
        pressure: PressureRecorder,
        context: MetalContext,
        lines: inout [String]
    ) throws -> PassStats {
        guard loader.tracker.currentPrograms == 0 else {
            throw AnimapkError.validation("V6 precondition started with residual residency")
        }
        loader.tracker.resetPeak()
        var stats = PassStats()
        let started = ProcessInfo.processInfo.systemUptime
        for block in 0..<blockCount {
            let loaded = try loader.load(block: block, profile: .full8)
            recordLoad(loaded, into: &stats)
            let eval = try evaluateBlock(loaded.models, surfaces: surfaces)
            stats.evalANE += eval.ane
            stats.evalWall += eval.wall
            let unload = loaded.destroyTimed()
            recordUnload(unload, into: &stats)
            stats.peakPrograms = max(stats.peakPrograms, loader.tracker.peakPrograms)
            stats.peakBlocks = max(stats.peakBlocks, loader.tracker.peakBlocks)
            if block < 2 || block % 7 == 6 || block == blockCount - 1 {
                lines.append(
                    "pre b\(two(block)) load=\(ms(loaded.loadWall)) eval=\(ms(eval.wall)) unload=\(ms(unload.wall)) \(loader.tracker.compact) | \(MemorySnapshot.capture(context: context).compact)")
            }
            if pressure.criticalSeen {
                throw AnimapkError.validation(
                    "critical pressure while preconditioning b\(block)")
            }
        }
        stats.wall = elapsedMS(since: started)
        return stats
    }

    /// Generic bounded scheduler used for every V6 comparison. Pinned blocks
    /// survive the measured pass; streamed blocks use one serial load queue.
    /// With retire1, exactly one streamed block may be retiring asynchronously,
    /// so peak program count is (pinned + depth + 1) * profile.programs.
    private static func runScheduler(
        profile: Profile,
        pinned: Int,
        asyncRetire: Bool,
        loader: SharedLoader,
        surfaces: ANEW8DiTSurfaces,
        context: MetalContext,
        pressure: PressureRecorder
    ) throws -> VariantResult {
        guard (0..<blockCount).contains(pinned) else {
            throw AnimapkError.validation("V6 invalid pinned count \(pinned)")
        }
        let requestedPeakBlocks = pinned + depth + (asyncRetire ? 1 : 0)
        let requestedPeakPrograms = requestedPeakBlocks * profile.programs
        guard requestedPeakPrograms <= programCeiling else {
            throw AnimapkError.validation(
                "V6 recipe \(profile.label) \(pinned)+3\(asyncRetire ? "+retire1" : "") requests \(requestedPeakPrograms) programs")
        }
        guard loader.tracker.currentPrograms == 0 else {
            throw AnimapkError.validation(
                "V6 scheduler started with residual residency: \(loader.tracker.compact)")
        }
        loader.tracker.resetPeak()
        let mark = pressure.mark()

        var pinnedBlocks: [LoadedBlock] = []
        let pinStart = ProcessInfo.processInfo.systemUptime
        for block in 0..<pinned {
            pinnedBlocks.append(try loader.load(block: block, profile: profile))
        }
        let pinSetupWall = elapsedMS(since: pinStart)

        let loadQueue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.v6-load-\(profile.rawValue)-\(pinned)-\(asyncRetire)",
            qos: .userInitiated)
        let retireQueue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.v6-retire-\(profile.rawValue)-\(pinned)",
            qos: .userInitiated)

        var stats = PassStats()
        var futures: [Int: LoadFuture] = [:]
        if pinned < blockCount {
            for block in pinned..<min(blockCount, pinned + depth) {
                futures[block] = LoadFuture(
                    block: block, profile: profile, loader: loader, queue: loadQueue)
            }
        }

        let passStart = ProcessInfo.processInfo.systemUptime

        // Pinned evaluation overlaps the prologue loader.
        for loaded in pinnedBlocks {
            let eval = try evaluateBlock(loaded.models, surfaces: surfaces)
            stats.evalANE += eval.ane
            stats.evalWall += eval.wall
        }

        var retiring: RetireFuture?
        for block in pinned..<blockCount {
            guard let future = futures.removeValue(forKey: block) else {
                throw AnimapkError.validation(
                    "V6 missing load future for streamed block \(block)")
            }
            let waitStart = ProcessInfo.processInfo.systemUptime
            let active = try future.wait()
            let wait = elapsedMS(since: waitStart)
            stats.nextWait += wait
            stats.maxNextWait = max(stats.maxNextWait, wait)
            recordLoad(active, into: &stats)

            let eval = try evaluateBlock(active.models, surfaces: surfaces)
            stats.evalANE += eval.ane
            stats.evalWall += eval.wall

            if asyncRetire {
                if let previous = retiring {
                    let retireWaitStart = ProcessInfo.processInfo.systemUptime
                    let retired = try previous.wait()
                    let retireWait = elapsedMS(since: retireWaitStart)
                    stats.retireBackpressure += retireWait
                    stats.maxRetireWait = max(stats.maxRetireWait, retireWait)
                    recordUnload(retired, into: &stats)
                }
                retiring = RetireFuture(block: active, queue: retireQueue)
            } else {
                recordUnload(active.destroyTimed(), into: &stats)
            }

            let newBlock = block + depth
            if newBlock < blockCount {
                futures[newBlock] = LoadFuture(
                    block: newBlock, profile: profile,
                    loader: loader, queue: loadQueue)
            }

            stats.peakPrograms = max(stats.peakPrograms, loader.tracker.peakPrograms)
            stats.peakBlocks = max(stats.peakBlocks, loader.tracker.peakBlocks)
            if pressure.criticalSeen {
                throw AnimapkError.validation(
                    "critical pressure in \(profile.label) \(pinned)+3")
            }
        }

        if let retiring {
            let drainStart = ProcessInfo.processInfo.systemUptime
            let retired = try retiring.wait()
            stats.retireDrain += elapsedMS(since: drainStart)
            recordUnload(retired, into: &stats)
        }

        stats.wall = elapsedMS(since: passStart)
        stats.peakPrograms = max(stats.peakPrograms, loader.tracker.peakPrograms)
        stats.peakBlocks = max(stats.peakBlocks, loader.tracker.peakBlocks)

        let expectedPinnedPrograms = pinned * profile.programs
        guard loader.tracker.currentPrograms == expectedPinnedPrograms else {
            throw AnimapkError.validation(
                "V6 streamed pass leaked residency: expected \(expectedPinnedPrograms), got \(loader.tracker.compact)")
        }

        for block in pinnedBlocks { block.destroyUntimed() }
        pinnedBlocks.removeAll()
        guard loader.tracker.currentPrograms == 0 else {
            throw AnimapkError.validation(
                "V6 scheduler cleanup leaked \(loader.tracker.compact)")
        }

        return VariantResult(
            profile: profile, pinned: pinned, asyncRetire: asyncRetire,
            pinSetupWall: pinSetupWall, stats: stats,
            newPressure: pressure.newIssues(since: mark),
            memory: MemorySnapshot.capture(context: context))
    }

    private static func evaluateBlock(
        _ models: ModelSet, surfaces: ANEW8DiTSurfaces
    ) throws -> (ane: Double, wall: Double) {
        let started = ProcessInfo.processInfo.systemUptime
        var totalANE = 0.0
        var value = 0.0
        _ = try models.selfQKV.evaluateInput(
            surfaces.tokenInput,
            qOutput: surfaces.q, kOutput: surfaces.k, vOutput: surfaces.v,
            milliseconds: &value)
        totalANE += value

        func projection(
            _ model: A12ANEProjectionModel,
            _ input: A12ANESurface,
            _ output: A12ANESurface
        ) throws {
            var milliseconds = 0.0
            _ = try model.evaluateInput(
                input, output: output, milliseconds: &milliseconds)
            totalANE += milliseconds
        }

        try projection(models.selfO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.crossQ, surfaces.tokenInput, surfaces.q)
        if let crossK = models.crossK {
            try projection(crossK, surfaces.contextInput, surfaces.contextK)
        }
        if let crossV = models.crossV {
            try projection(crossV, surfaces.contextInput, surfaces.contextV)
        }
        try projection(models.crossO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.mlpUp, surfaces.tokenInput, surfaces.hidden)
        try projection(models.mlpDown, surfaces.hidden, surfaces.tokenOutput)
        return (totalANE, elapsedMS(since: started))
    }

    private static func recordLoad(_ loaded: LoadedBlock, into stats: inout PassStats) {
        stats.loadANE += loaded.loadANE
        stats.loadWall += loaded.loadWall
        stats.loadHost += loaded.hostLoadWall
        stats.loads += 1
        stats.maxLoad = max(stats.maxLoad, loaded.loadWall)
    }

    private static func recordUnload(
        _ unload: (callSum: Double, wall: Double), into stats: inout PassStats
    ) {
        stats.unloadCallSum += unload.callSum
        stats.unloadWall += unload.wall
        stats.unloads += 1
        stats.maxUnload = max(stats.maxUnload, unload.wall)
    }

    private static func variantSummary(_ result: VariantResult) -> String {
        let s = result.stats
        return "\(result.tag): wall=\(ms(s.wall))ms avg=\(ms(s.wall / Double(blockCount)))ms/block "
            + "pinSetup=\(ms(result.pinSetupWall)) load=\(ms(s.loadWall)) unload=\(ms(s.unloadWall)) "
            + "eval=\(ms(s.evalWall)) nextWait=\(ms(s.nextWait)) retireWait=\(ms(s.retireBackpressure)) "
            + "retireDrain=\(ms(s.retireDrain)) avgLoad=\(ms(s.avgLoad)) avgUnload=\(ms(s.avgUnload)) "
            + "peak=\(s.peakBlocks) blocks/\(s.peakPrograms) programs pressure+\(result.newPressure) | \(result.memory.compact)"
    }

    private static func summary(_ name: String, _ s: PassStats) -> String {
        "\(name): wall=\(ms(s.wall))ms loads=\(s.loads) loadANE=\(ms(s.loadANE)) loadWall=\(ms(s.loadWall)) "
            + "evalANE=\(ms(s.evalANE)) evalWall=\(ms(s.evalWall)) unloads=\(s.unloads) "
            + "unloadCallSum=\(ms(s.unloadCallSum)) unloadWall=\(ms(s.unloadWall)) "
            + "avgLoad=\(ms(s.avgLoad)) avgUnload=\(ms(s.avgUnload)) peak=\(s.peakBlocks) blocks/\(s.peakPrograms) programs"
    }

    private static func appendMirroredScorecard(
        title: String, results: [VariantResult], lines: inout [String]
    ) {
        lines.append(title)
        let pins = Array(Set(results.map(\.pinned))).sorted()
        var ranked: [(Int, Double, Double, Int)] = []
        for pinned in pins {
            let group = results.filter { $0.pinned == pinned }
            let wall = median(group.map(\.stats.wall))
            let pin = median(group.map(\.pinSetupWall))
            let pressure = group.map(\.newPressure).max() ?? 0
            ranked.append((pinned, wall, pin, pressure))
        }
        ranked.sort { $0.1 < $1.1 }
        for (rank, item) in ranked.enumerated() {
            lines.append(
                "#\(rank + 1) pin=\(item.0): medianWall=\(ms(item.1))ms (\(ms(item.1 / Double(blockCount)))ms/block) medianPinSetup=\(ms(item.2))ms maxPressure+\(item.3)")
        }
    }

    private static func appendRetireScorecard(
        title: String, results: [VariantResult], lines: inout [String]
    ) {
        lines.append(title)
        for mode in [false, true] {
            let group = results.filter { $0.asyncRetire == mode }
            guard !group.isEmpty else { continue }
            lines.append(
                "\(mode ? "retire1" : "sync"): medianWall=\(ms(median(group.map(\.stats.wall))))ms "
                + "medianRetireBackpressure=\(ms(median(group.map(\.stats.retireBackpressure))))ms "
                + "medianNextWait=\(ms(median(group.map(\.stats.nextWait))))ms "
                + "peakPrograms=\(group.map(\.stats.peakPrograms).max() ?? 0) maxPressure+\(group.map(\.newPressure).max() ?? 0)")
        }
    }

    private static func bestMirrored(results: [VariantResult]) -> [VariantResult]? {
        let pins = Array(Set(results.map(\.pinned)))
        let groups = pins.map { pin in results.filter { $0.pinned == pin } }
        return groups.min {
            median($0.map(\.stats.wall)) < median($1.map(\.stats.wall))
        }
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    }

    private static func elapsedMS(since start: TimeInterval) -> Double {
        (ProcessInfo.processInfo.systemUptime - start) * 1_000
    }

    private static func ms(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private static func seconds(_ value: Double) -> String {
        String(format: "%.2f", value / 1_000.0)
    }

    private static func two(_ value: Int) -> String {
        String(format: "%02d", value)
    }

    private static func mb(_ bytes: UInt64) -> String {
        String(format: "%.0fMB", Double(bytes) / 1_048_576.0)
    }

    private static func processPhysicalFootprint() -> UInt64? {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
        #else
        return nil
        #endif
    }
}
