import Foundation
import Dispatch
import Metal
import UIKit
import CoreML
#if canImport(Darwin)
import Darwin
#endif

/// V5 Michelin experiment for the A12/H11 native W8 backend.
///
/// V4 proved three things we now treat as facts for this probe:
/// - one shared loader removes nearly all host-side pack/cache overhead;
/// - warm block load is still much slower than ANE evaluation;
/// - synchronous block destruction/unload is substantial and can overlap the
///   background loader. V5 measures that unload directly and compares small
///   rings that exploit it without approaching the old high-residency cliff.
///
/// Scheduler recipes:
///   A) 0 pinned + 2 streaming = 16 programs max
///   B) 0 pinned + 3 streaming = 24 programs max
///   C) 2 pinned + 3 streaming = 40 programs max
///
/// The second, independent track inspects a CI-generated Core ML multifunction
/// asset through public MLModelAsset APIs and the private _ANEModel procedure
/// metadata. Failure there does not abort the scheduler results.
enum ANERingProbe {
    private static let blockCount = ModelConstants.ditBlocks
    private static let programsPerBlock = 8
    private static let measuredPasses = 3

    private struct PressureMark {
        let issues: Int
    }

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
                queue: DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-v5-pressure"))
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

    private final class ResidencyTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var blocks: Set<Int> = []
        private var peak = 0
        private let ceiling: Int

        init(ceiling: Int = 5) { self.ceiling = ceiling }

        func add(_ block: Int) throws {
            lock.lock()
            let inserted = blocks.insert(block).inserted
            peak = max(peak, blocks.count)
            let count = blocks.count
            lock.unlock()
            guard inserted else {
                throw AnimapkError.validation("V5 duplicate resident block \(block)")
            }
            guard count <= ceiling else {
                throw AnimapkError.validation("V5 residency exceeded \(ceiling)-block ceiling: \(count)")
            }
        }

        func remove(_ block: Int) {
            lock.lock(); blocks.remove(block); lock.unlock()
        }

        func resetPeak() {
            lock.lock(); peak = blocks.count; lock.unlock()
        }

        var currentCount: Int {
            lock.lock(); defer { lock.unlock() }
            return blocks.count
        }

        var peakCount: Int {
            lock.lock(); defer { lock.unlock() }
            return peak
        }

        var compact: String {
            lock.lock(); defer { lock.unlock() }
            return "resident=\(blocks.count) blocks/\(blocks.count * programsPerBlock) programs peak=\(peak) blocks/\(peak * programsPerBlock) programs"
        }
    }

    private final class LoadedBlock: @unchecked Sendable {
        let block: Int
        let models: ANEW8DiTModels
        let loadANE: Double
        let loadWall: Double
        private let tracker: ResidencyTracker
        private let lock = NSLock()
        private var destroyed = false

        init(
            block: Int,
            models: ANEW8DiTModels,
            loadANE: Double,
            loadWall: Double,
            tracker: ResidencyTracker
        ) throws {
            self.block = block
            self.models = models
            self.loadANE = loadANE
            self.loadWall = loadWall
            self.tracker = tracker
            try tracker.add(block)
        }

        var hostLoadWall: Double { max(0, loadWall - loadANE) }

        /// Timed diagnostic destruction. The Obj-C bridge unloads each private
        /// ANE model and destroys its retained _ANEModel object in the same call.
        func destroyTimed() throws -> (ane: Double, wall: Double) {
            lock.lock()
            if destroyed { lock.unlock(); return (0, 0) }
            destroyed = true
            lock.unlock()

            let started = ProcessInfo.processInfo.systemUptime
            var ane = 0.0
            func destroy(_ model: A12ANEProjectionModel) throws {
                let value = model.diagnosticDestroyMilliseconds()
                guard value >= 0 else {
                    throw AnimapkError.validation("V5 timed projection destroy failed for block \(block)")
                }
                ane += value
            }

            let qkv = models.selfQKV.diagnosticDestroyMilliseconds()
            guard qkv >= 0 else {
                throw AnimapkError.validation("V5 timed QKV destroy failed for block \(block)")
            }
            ane += qkv
            try destroy(models.selfO)
            try destroy(models.crossQ)
            try destroy(models.crossK)
            try destroy(models.crossV)
            try destroy(models.crossO)
            try destroy(models.mlpUp)
            try destroy(models.mlpDown)
            let wall = elapsedMS(since: started)
            tracker.remove(block)
            return (ane, wall)
        }

        func destroyUntimed() {
            lock.lock()
            if destroyed { lock.unlock(); return }
            destroyed = true
            lock.unlock()
            models.selfQKV.invalidate()
            models.selfO.invalidate()
            models.crossQ.invalidate()
            models.crossK.invalidate()
            models.crossV.invalidate()
            models.crossO.invalidate()
            models.mlpUp.invalidate()
            models.mlpDown.invalidate()
            tracker.remove(block)
        }

        deinit { destroyUntimed() }
    }

    /// One loader for the entire V5 run. Native-pack validation and the 224-file
    /// prepared-cache completeness check happen once, never in the hot ring.
    private final class SharedLoader: @unchecked Sendable {
        private let file: AnimapkFile
        let tracker = ResidencyTracker(ceiling: 5)

        init(file: AnimapkFile) throws {
            self.file = file
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                throw AnimapkError.validation("ANE prepared-model cache is incomplete")
            }
            guard A12ANEIsAvailable() else {
                throw AnimapkError.validation("A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
            }
        }

        func load(block: Int) throws -> LoadedBlock {
            guard (0..<blockCount).contains(block) else {
                throw AnimapkError.validation("V5 block index out of range: \(block)")
            }
            let started = ProcessInfo.processInfo.systemUptime

            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
                guard let digest = tensor.blobSHA256 else {
                    throw AnimapkError.validation("ANE-native tensor hash missing")
                }
                return digest
            }

            func projection(_ suffix: String) throws -> A12ANEProjectionModel {
                guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                    throw AnimapkError.validation("unknown ANE projection suffix: \(suffix)")
                }
                let key = ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: try digest(suffix))
                return try A12ANEProjectionModel(
                    preparedInputChannels: UInt(spec.columns),
                    outputChannels: UInt(spec.rows), spatial: UInt(spec.spatial),
                    label: "v5_b\(block)_\(spec.tag)", cacheKey: key)
            }

            let qHash = try digest("self_attn.q_proj.weight")
            let kHash = try digest("self_attn.k_proj.weight")
            let vHash = try digest("self_attn.v_proj.weight")
            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block, q: qHash, k: kHash, v: vHash)
            let selfQKV = try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(DiTBlockExecutor.tokens),
                label: "v5_b\(block)_self_qkv", cacheKey: qkvKey)

            let models = ANEW8DiTModels(
                selfQKV: selfQKV,
                selfO: try projection("self_attn.output_proj.weight"),
                crossQ: try projection("cross_attn.q_proj.weight"),
                crossK: try projection("cross_attn.k_proj.weight"),
                crossV: try projection("cross_attn.v_proj.weight"),
                crossO: try projection("cross_attn.output_proj.weight"),
                mlpUp: try projection("mlp.layer1.weight"),
                mlpDown: try projection("mlp.layer2.weight"))
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
        let position: Int
        let block: Int
        private let semaphore = DispatchSemaphore(value: 0)
        private let box = LoadResultBox()

        init(position: Int, block: Int, loader: SharedLoader, queue: DispatchQueue) {
            self.position = position
            self.block = block
            queue.async {
                do { self.box.set(.success(try loader.load(block: block))) }
                catch { self.box.set(.failure(error)) }
                self.semaphore.signal()
            }
        }

        func wait() throws -> LoadedBlock {
            semaphore.wait()
            guard let result = box.take() else {
                throw AnimapkError.validation("V5 loader returned no result for block \(block)")
            }
            return try result.get()
        }
    }

    private struct PassStats {
        var wall = 0.0
        var loadANE = 0.0
        var loadWall = 0.0
        var loadHost = 0.0
        var evalANE = 0.0
        var evalWall = 0.0
        var unloadANE = 0.0
        var unloadWall = 0.0
        var waitWall = 0.0
        var maxLoad = 0.0
        var maxUnload = 0.0
        var maxWait = 0.0
        var maxResident = 0
        var loads = 0
        var unloads = 0

        var avgLoad: Double { loads > 0 ? loadWall / Double(loads) : 0 }
        var avgUnload: Double { unloads > 0 ? unloadWall / Double(unloads) : 0 }
        var avgEval: Double { evalWall / Double(blockCount) }
    }

    private struct VariantResult {
        let name: String
        let maxPrograms: Int
        let pinSetupWall: Double
        let passes: [PassStats]
        let newPressureIssues: Int
        var steady: PassStats { passes[min(1, passes.count - 1)] }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE Michelin probe v5",
            "Tracks: scheduler/unload microscope + independent multifunction/procedure POC.",
            "Schedulers: 0+2 (16 programs), 0+3 (24 programs), 2+3 (40 programs).",
            "Accounting: stage wall includes eval + timed destroy/unload + post-destroy wait; background load is measured separately.",
            "Each scheduler runs 3 full passes; pass #1 is the steady comparison."
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

            let loaderStart = ProcessInfo.processInfo.systemUptime
            let loader = try SharedLoader(file: file)
            let loaderInit = elapsedMS(since: loaderStart)
            let surfaces = try ANEW8DiTSurfaces(device: context.device)

            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("shared-loader init=\(ms(loaderInit))ms")
            lines.append("baseline: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            lines.append("")

            lines.append("PRECONDITION — one cold/touch traversal with direct unload timing")
            let cold = try preconditionRuntime(
                loader: loader, surfaces: surfaces, context: context,
                pressure: pressure, lines: &lines)
            lines.append(passSummary(name: "precondition", pass: 0, stats: cold))
            lines.append("after precondition: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            if pressure.criticalSeen {
                throw AnimapkError.validation("critical pressure during V5 precondition")
            }
            try? await Task.sleep(nanoseconds: 750_000_000)

            lines.append("")
            lines.append("EXPERIMENT A1 — corrected 0+2: eval || next-load, then unload || next-load, then wait")
            let zeroTwo = try runUnpinned(
                depth: 2, loader: loader, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            appendVariantSummary(zeroTwo, context: context, pressure: pressure, lines: &lines)
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("EXPERIMENT A2 — 0+3: current + next + next2, serial loader two blocks ahead")
            let zeroThree = try runUnpinned(
                depth: 3, loader: loader, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            appendVariantSummary(zeroThree, context: context, pressure: pressure, lines: &lines)
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("EXPERIMENT A3 — 2+3: pin b00-b01, stream remaining 26 with depth 3")
            let twoThree = try runTwoPinnedThreeStream(
                loader: loader, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            appendVariantSummary(twoThree, context: context, pressure: pressure, lines: &lines)

            let variants = [zeroTwo, zeroThree, twoThree]
            let ranked = variants.sorted { $0.steady.wall < $1.steady.wall }
            lines.append("")
            lines.append("MICHELIN SCHEDULER SCORECARD — steady pass #1")
            for (rank, result) in ranked.enumerated() {
                let s = result.steady
                lines.append(
                    "#\(rank + 1) \(result.name): wall=\(ms(s.wall))ms (\(ms(s.wall / Double(blockCount)))ms/block) " +
                    "load=\(ms(s.loadWall))ms unload=\(ms(s.unloadWall))ms eval=\(ms(s.evalWall))ms wait=\(ms(s.waitWall))ms " +
                    "avgLoad=\(ms(s.avgLoad)) avgUnload=\(ms(s.avgUnload)) maxResident=\(s.maxResident) blocks/\(s.maxResident * programsPerBlock) programs newPressure=\(result.newPressureIssues)")
            }
            if let winner = ranked.first {
                lines.append("SCHEDULER WINNER: \(winner.name) — projected 8-step projection traversal \(seconds(winner.steady.wall * 8))s.")
            }

            lines.append("")
            lines.append("EXPERIMENT B — multifunction / private ANE procedure POC")
            await runMultiProcedurePOC(lines: &lines)

            lines.append("")
            lines.append("DECISION GUIDE")
            lines.append("- 0+3 wins with ~zero post-destroy wait and no pressure => prefer the 24-program ring; pinning is unnecessary.")
            lines.append("- 2+3 materially wins with no new pressure => a 40-program conservative pinned-prefix fallback is justified.")
            lines.append("- If unload is ~load-eval and 0+2 becomes competitive, two slots may be enough once unload is ordered correctly.")
            lines.append("- Public functions=2 AND private procedures>=2 => immediately scale POC to one 8-procedure transformer block.")
            lines.append("- Public functions=2 but private procedures<2/fail => public Core ML multifunction does not map directly to one private ANEF load; keep scheduler work and investigate Espresso/ANEF procedure packaging separately.")
            lines.append("final: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact) | \(loader.tracker.compact)")
        } catch {
            lines.append("ERROR: \(error.localizedDescription)")
            lines.append("pressure at error: \(pressure.compact)")
        }

        return lines.joined(separator: "\n")
    }

    private static func preconditionRuntime(
        loader: SharedLoader,
        surfaces: ANEW8DiTSurfaces,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) throws -> PassStats {
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("V5 precondition started with residual residency")
        }
        loader.tracker.resetPeak()
        var stats = PassStats()
        let started = ProcessInfo.processInfo.systemUptime

        for block in 0..<blockCount {
            let loaded = try loader.load(block: block)
            recordLoad(loaded, into: &stats)
            let eval = try evaluateBlock(loaded.models, surfaces: surfaces)
            stats.evalANE += eval.ane
            stats.evalWall += eval.wall
            let unload = try loaded.destroyTimed()
            recordUnload(unload, into: &stats)
            stats.maxResident = max(stats.maxResident, loader.tracker.peakCount)

            if block < 2 || block % 4 == 3 || block == blockCount - 1 {
                lines.append(
                    "pre b\(two(block)) load=\(ms(loaded.loadWall))ms ANE=\(ms(loaded.loadANE)) host=\(ms(loaded.hostLoadWall)) " +
                    "eval=\(ms(eval.wall))ms unload=\(ms(unload.wall))ms ANE=\(ms(unload.ane)) \(loader.tracker.compact) | \(MemorySnapshot.capture(context: context).compact)")
            }
            if pressure.criticalSeen {
                throw AnimapkError.validation("critical pressure while preconditioning b\(block)")
            }
        }
        stats.wall = elapsedMS(since: started)
        stats.maxResident = max(stats.maxResident, loader.tracker.peakCount)
        return stats
    }

    /// Continuous unpinned ring. `depth=2` keeps current+next; `depth=3` keeps
    /// current+next+next2. A single serial loader preserves the private-runtime
    /// safety property: model loads themselves are never concurrent.
    private static func runUnpinned(
        depth: Int,
        loader: SharedLoader,
        surfaces: ANEW8DiTSurfaces,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) throws -> VariantResult {
        guard depth == 2 || depth == 3 else {
            throw AnimapkError.validation("V5 unpinned depth must be 2 or 3")
        }
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("0+\(depth) started with residual residency")
        }
        loader.tracker.resetPeak()
        let mark = pressure.mark()
        let queue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-v5-zero-\(depth)", qos: .userInitiated)
        var stats = Array(repeating: PassStats(), count: measuredPasses)
        let total = measuredPasses * blockCount

        let entryStart = ProcessInfo.processInfo.systemUptime
        var current: LoadedBlock? = try loader.load(block: 0)
        let entryWall = elapsedMS(since: entryStart)
        lines.append("0+\(depth) entry b00 load=\(ms(entryWall))ms ANE=\(ms(current?.loadANE ?? 0))")

        var futures: [Int: LoadFuture] = [:]
        if total > 1 {
            for position in 1..<min(depth, total) {
                futures[position] = LoadFuture(
                    position: position, block: position % blockCount,
                    loader: loader, queue: queue)
            }
        }

        for position in 0..<total {
            let pass = position / blockCount
            let block = position % blockCount
            guard let active = current, active.block == block else {
                throw AnimapkError.validation("0+\(depth) current mismatch at position \(position)")
            }
            let stageStart = ProcessInfo.processInfo.systemUptime
            let eval = try evaluateBlock(active.models, surfaces: surfaces)
            stats[pass].evalANE += eval.ane
            stats[pass].evalWall += eval.wall

            let unload = try active.destroyTimed()
            recordUnload(unload, into: &stats[pass])

            var next: LoadedBlock?
            var wait = 0.0
            let nextPosition = position + 1
            if nextPosition < total {
                guard let future = futures.removeValue(forKey: nextPosition) else {
                    throw AnimapkError.validation("0+\(depth) missing future for position \(nextPosition)")
                }
                let waitStart = ProcessInfo.processInfo.systemUptime
                next = try future.wait()
                wait = elapsedMS(since: waitStart)
                let targetPass = nextPosition / blockCount
                if let next { recordLoad(next, into: &stats[targetPass]) }

                let newPosition = position + depth
                if newPosition < total {
                    futures[newPosition] = LoadFuture(
                        position: newPosition, block: newPosition % blockCount,
                        loader: loader, queue: queue)
                }
            }

            stats[pass].waitWall += wait
            stats[pass].maxWait = max(stats[pass].maxWait, wait)
            stats[pass].wall += elapsedMS(since: stageStart)
            stats[pass].maxResident = max(stats[pass].maxResident, loader.tracker.peakCount)

            if pass == 1 {
                lines.append(
                    "0+\(depth) p1 b\(two(block)) eval=\(ms(eval.wall)) unload=\(ms(unload.wall)) unloadANE=\(ms(unload.ane)) " +
                    "wait=\(ms(wait)) stage=\(ms(elapsedMS(since: stageStart))) \(loader.tracker.compact)")
            }
            current = next
            if pressure.criticalSeen {
                throw AnimapkError.validation("critical pressure in 0+\(depth), pass \(pass), b\(block)")
            }
        }

        current?.destroyUntimed()
        for future in futures.values {
            let loaded = try future.wait()
            loaded.destroyUntimed()
        }
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("0+\(depth) leaked \(loader.tracker.currentCount) blocks")
        }
        for index in stats.indices {
            stats[index].maxResident = max(stats[index].maxResident, min(loader.tracker.peakCount, depth))
        }
        return VariantResult(
            name: "0+\(depth)", maxPrograms: depth * programsPerBlock,
            pinSetupWall: 0, passes: stats,
            newPressureIssues: pressure.newIssues(since: mark))
    }

    private static func runTwoPinnedThreeStream(
        loader: SharedLoader,
        surfaces: ANEW8DiTSurfaces,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) throws -> VariantResult {
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("2+3 started with residual residency")
        }
        loader.tracker.resetPeak()
        let mark = pressure.mark()

        var pinned: [LoadedBlock] = []
        let pinStart = ProcessInfo.processInfo.systemUptime
        for block in 0..<2 {
            let loaded = try loader.load(block: block)
            pinned.append(loaded)
            lines.append("2+3 pin b\(two(block)) load=\(ms(loaded.loadWall))ms ANE=\(ms(loaded.loadANE))")
        }
        let pinSetupWall = elapsedMS(since: pinStart)
        let queue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-v5-two-three", qos: .userInitiated)
        var passStats: [PassStats] = []

        for pass in 0..<measuredPasses {
            var stats = PassStats()
            let passStart = ProcessInfo.processInfo.systemUptime
            var futures: [Int: LoadFuture] = [:]

            futures[2] = LoadFuture(position: 2, block: 2, loader: loader, queue: queue)
            futures[3] = LoadFuture(position: 3, block: 3, loader: loader, queue: queue)
            futures[4] = LoadFuture(position: 4, block: 4, loader: loader, queue: queue)

            for block in 0..<2 {
                let eval = try evaluateBlock(pinned[block].models, surfaces: surfaces)
                stats.evalANE += eval.ane
                stats.evalWall += eval.wall
                stats.maxResident = max(stats.maxResident, loader.tracker.peakCount)
                if pass == 1 {
                    lines.append("2+3 p1 pinned b\(two(block)) eval=\(ms(eval.wall))ms \(loader.tracker.compact)")
                }
            }

            var current: LoadedBlock?
            for block in 2..<blockCount {
                if current == nil {
                    guard let future = futures.removeValue(forKey: block) else {
                        throw AnimapkError.validation("2+3 missing future for b\(block)")
                    }
                    let waitStart = ProcessInfo.processInfo.systemUptime
                    current = try future.wait()
                    let wait = elapsedMS(since: waitStart)
                    stats.waitWall += wait
                    stats.maxWait = max(stats.maxWait, wait)
                    if let current { recordLoad(current, into: &stats) }
                }
                guard let active = current, active.block == block else {
                    throw AnimapkError.validation("2+3 current mismatch at b\(block)")
                }

                let stageStart = ProcessInfo.processInfo.systemUptime
                let eval = try evaluateBlock(active.models, surfaces: surfaces)
                stats.evalANE += eval.ane
                stats.evalWall += eval.wall
                let unload = try active.destroyTimed()
                recordUnload(unload, into: &stats)

                var next: LoadedBlock?
                var wait = 0.0
                let nextBlock = block + 1
                if nextBlock < blockCount {
                    guard let future = futures.removeValue(forKey: nextBlock) else {
                        throw AnimapkError.validation("2+3 missing future for next b\(nextBlock)")
                    }
                    let waitStart = ProcessInfo.processInfo.systemUptime
                    next = try future.wait()
                    wait = elapsedMS(since: waitStart)
                    if let next { recordLoad(next, into: &stats) }

                    let newBlock = block + 3
                    if newBlock < blockCount {
                        futures[newBlock] = LoadFuture(
                            position: newBlock, block: newBlock,
                            loader: loader, queue: queue)
                    }
                }

                stats.waitWall += wait
                stats.maxWait = max(stats.maxWait, wait)
                stats.maxResident = max(stats.maxResident, loader.tracker.peakCount)
                if pass == 1 {
                    lines.append(
                        "2+3 p1 stream b\(two(block)) eval=\(ms(eval.wall)) unload=\(ms(unload.wall)) unloadANE=\(ms(unload.ane)) " +
                        "wait=\(ms(wait)) stage=\(ms(elapsedMS(since: stageStart))) \(loader.tracker.compact)")
                }
                current = next
                if pressure.criticalSeen {
                    throw AnimapkError.validation("critical pressure in 2+3 pass \(pass), b\(block)")
                }
            }

            current?.destroyUntimed()
            for future in futures.values {
                let loaded = try future.wait()
                loaded.destroyUntimed()
            }
            stats.wall = elapsedMS(since: passStart)
            stats.maxResident = max(stats.maxResident, loader.tracker.peakCount)
            passStats.append(stats)
        }

        for block in pinned { block.destroyUntimed() }
        pinned.removeAll()
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("2+3 leaked \(loader.tracker.currentCount) blocks")
        }
        return VariantResult(
            name: "2+3", maxPrograms: 40,
            pinSetupWall: pinSetupWall, passes: passStats,
            newPressureIssues: pressure.newIssues(since: mark))
    }

    private static func runMultiProcedurePOC(lines: inout [String]) async {
        guard let templateURL = Bundle.main.url(
            forResource: "Conv2048W8Template", withExtension: "bundle") else {
            lines.append("multifunction: SKIP — Conv2048W8Template.bundle missing")
            return
        }
        let assetURL = templateURL.appendingPathComponent("V5TwoProcedure.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            lines.append("multifunction: SKIP — CI-generated V5TwoProcedure.mlmodelc missing from bundle")
            return
        }

        do {
            let asset = try MLModelAsset(url: assetURL)
            let names = try await asset.functionNames
            lines.append("public MLModelAsset functions=\(names.count) names=\(names.sorted())")
        } catch {
            lines.append("public MLModelAsset ERROR: \(error.localizedDescription)")
        }

        lines.append(A12ANEMultiProcedureProbe(assetURL, 64, 64))
    }

    private static func appendVariantSummary(
        _ result: VariantResult,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) {
        if result.pinSetupWall > 0 {
            lines.append("\(result.name) one-time pin setup wall=\(ms(result.pinSetupWall))ms")
        }
        for (pass, stats) in result.passes.enumerated() {
            lines.append(passSummary(name: result.name, pass: pass, stats: stats))
        }
        let s = result.steady
        lines.append(
            "\(result.name) STEADY: wall=\(ms(s.wall))ms avg=\(ms(s.wall / Double(blockCount)))ms/block " +
            "load=\(ms(s.loadWall)) unload=\(ms(s.unloadWall)) eval=\(ms(s.evalWall)) wait=\(ms(s.waitWall)) " +
            "maxResident=\(s.maxResident) blocks/\(s.maxResident * programsPerBlock) programs newPressure=\(result.newPressureIssues) | " +
            "\(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
    }

    private static func passSummary(name: String, pass: Int, stats: PassStats) -> String {
        "\(name) pass#\(pass): wall=\(ms(stats.wall))ms loads=\(stats.loads) loadANE=\(ms(stats.loadANE)) loadWall=\(ms(stats.loadWall)) host=\(ms(stats.loadHost)) " +
        "evalANE=\(ms(stats.evalANE)) evalWall=\(ms(stats.evalWall)) unloads=\(stats.unloads) unloadANE=\(ms(stats.unloadANE)) unloadWall=\(ms(stats.unloadWall)) " +
        "wait=\(ms(stats.waitWall)) avgLoad=\(ms(stats.avgLoad)) avgUnload=\(ms(stats.avgUnload)) maxLoad=\(ms(stats.maxLoad)) maxUnload=\(ms(stats.maxUnload)) maxWait=\(ms(stats.maxWait)) maxResident=\(stats.maxResident)"
    }

    private static func recordLoad(_ loaded: LoadedBlock, into stats: inout PassStats) {
        stats.loadANE += loaded.loadANE
        stats.loadWall += loaded.loadWall
        stats.loadHost += loaded.hostLoadWall
        stats.loads += 1
        stats.maxLoad = max(stats.maxLoad, loaded.loadWall)
    }

    private static func recordUnload(_ unload: (ane: Double, wall: Double), into stats: inout PassStats) {
        stats.unloadANE += unload.ane
        stats.unloadWall += unload.wall
        stats.unloads += 1
        stats.maxUnload = max(stats.maxUnload, unload.wall)
    }

    private static func evaluateBlock(
        _ models: ANEW8DiTModels,
        surfaces: ANEW8DiTSurfaces
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
            var ms = 0.0
            _ = try model.evaluateInput(input, output: output, milliseconds: &ms)
            totalANE += ms
        }

        try projection(models.selfO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.crossQ, surfaces.tokenInput, surfaces.q)
        try projection(models.crossK, surfaces.contextInput, surfaces.contextK)
        try projection(models.crossV, surfaces.contextInput, surfaces.contextV)
        try projection(models.crossO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.mlpUp, surfaces.tokenInput, surfaces.hidden)
        try projection(models.mlpDown, surfaces.hidden, surfaces.tokenOutput)
        return (totalANE, elapsedMS(since: started))
    }

    private static func elapsedMS(since start: TimeInterval) -> Double {
        (ProcessInfo.processInfo.systemUptime - start) * 1_000
    }

    private static func ms(_ value: Double) -> String { String(format: "%.1f", value) }
    private static func seconds(_ value: Double) -> String { String(format: "%.2f", value / 1_000) }
    private static func two(_ value: Int) -> String { String(format: "%02d", value) }
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
