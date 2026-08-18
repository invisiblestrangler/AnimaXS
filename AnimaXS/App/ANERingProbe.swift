import Foundation
import Dispatch
import Metal
import UIKit
#if canImport(Darwin)
import Darwin
#endif

/// V4 production-architecture bakeoff for the A12/H11 ANE W8 path.
///
/// V3 proved that a two-block ring is safe and useful, but its `BlockSlot`
/// constructed a fresh `ANEW8DiTModelCache` for every streamed block. That
/// repeated whole-pack validation/cache-completeness work and inflated host
/// wall time. V4 keeps all experiment-only state in this already-compiled file:
/// one shared loader validates the pack/prepared cache exactly once, then only
/// constructs and evicts the eight ANE programs belonging to each requested
/// transformer block.
///
/// Recipes compared under the same preconditioned runtime:
///   A) 0 pinned + 2 streaming = 16 programs max
///   B) 4 pinned + 2 streaming = 48 programs max
///   C) 6 pinned + 2 streaming = 64 programs max
///
/// Diagnostic only. Production inference behavior is unchanged.
enum ANERingProbe {
    private static let blockCount = ModelConstants.ditBlocks
    private static let programsPerBlock = 8

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
                queue: DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-v4-pressure"))
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

    /// Independent residency accounting because normal process-memory counters
    /// do not expose private ANE service residency. V4 never intentionally
    /// exceeds eight resident transformer blocks / 64 ANE programs.
    private final class ResidencyTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var blocks: Set<Int> = []
        private var peak = 0

        func add(_ block: Int) throws {
            lock.lock()
            let inserted = blocks.insert(block).inserted
            peak = max(peak, blocks.count)
            let count = blocks.count
            lock.unlock()
            guard inserted else {
                throw AnimapkError.validation("V4 attempted duplicate residency for block \(block)")
            }
            guard count <= 8 else {
                throw AnimapkError.validation("V4 residency exceeded safety ceiling: \(count) blocks")
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
        private var invalidated = false

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

        var hostWall: Double { max(0, loadWall - loadANE) }

        func invalidate() {
            lock.lock()
            if invalidated { lock.unlock(); return }
            invalidated = true
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

        deinit { invalidate() }
    }

    /// Shared diagnostic loader. Expensive pack validation and the 224-entry
    /// prepared-model completeness check happen once in init, never per block.
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
                throw AnimapkError.validation("A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
            }
        }

        func load(block: Int) throws -> LoadedBlock {
            guard (0..<blockCount).contains(block) else {
                throw AnimapkError.validation("V4 block index out of range: \(block)")
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
                    label: "v4_b\(block)_\(spec.tag)", cacheKey: key)
            }

            let qHash = try digest("self_attn.q_proj.weight")
            let kHash = try digest("self_attn.k_proj.weight")
            let vHash = try digest("self_attn.v_proj.weight")
            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block, q: qHash, k: kHash, v: vHash)
            let selfQKV = try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(DiTBlockExecutor.tokens),
                label: "v4_b\(block)_self_qkv", cacheKey: qkvKey)

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
                block: block,
                models: models,
                loadANE: models.loadMilliseconds,
                loadWall: wall,
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

        init(block: Int, loader: SharedLoader, queue: DispatchQueue) {
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
                throw AnimapkError.validation("ANE V4 loader returned no result for block \(block)")
            }
            return try result.get()
        }
    }

    private struct PassStats {
        var wall = 0.0
        var loadANE = 0.0
        var loadWall = 0.0
        var hostWall = 0.0
        var evalANE = 0.0
        var evalWall = 0.0
        var waitWall = 0.0
        var maxLoadWall = 0.0
        var maxWaitWall = 0.0
        var maxResidentBlocks = 0
        var streamLoads = 0

        var averageLoadWall: Double { streamLoads > 0 ? loadWall / Double(streamLoads) : 0 }
        var averageLoadANE: Double { streamLoads > 0 ? loadANE / Double(streamLoads) : 0 }
        var averageHostWall: Double { streamLoads > 0 ? hostWall / Double(streamLoads) : 0 }
        var averageEvalWall: Double { evalWall / Double(blockCount) }
        var averageWaitWall: Double { waitWall / Double(blockCount) }
    }

    private struct VariantResult {
        let name: String
        let pinned: Int
        let pinSetupANE: Double
        let pinSetupWall: Double
        let passes: [PassStats]

        var steady: PassStats { passes[min(1, passes.count - 1)] }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE recipe bakeoff v4",
            "Goal: remove V3's repeated cache-construction tax and choose the best safe residency recipe.",
            "Variants: 0+2 (16 programs), 4+2 (48 programs), 6+2 (64 programs).",
            "Shared-loader policy: validate pack + prepared cache exactly once; rotate only block model wrappers.",
            "Each variant runs 3 full 28-block passes; pass #1 is the steady-state comparison."
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

            // Equalize private-runtime first-touch state so recipe A does not pay
            // all 224 cold loads while recipes B/C inherit a warm service.
            lines.append("PRECONDITION — touch every block once, then fully evict")
            let precondition = try preconditionRuntime(
                loader: loader, surfaces: surfaces, context: context,
                pressure: pressure, lines: &lines)
            lines.append("precondition: wall=\(ms(precondition.wall))ms loadANE=\(ms(precondition.loadANE))ms loadWall=\(ms(precondition.loadWall))ms host=\(ms(precondition.hostWall))ms evalWall=\(ms(precondition.evalWall))ms peak=\(precondition.maxResidentBlocks) blocks/\(precondition.maxResidentBlocks * programsPerBlock) programs")
            lines.append("after precondition: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            if pressure.criticalSeen {
                throw AnimapkError.validation("critical memory pressure during V4 precondition")
            }
            try? await Task.sleep(nanoseconds: 750_000_000)

            lines.append("")
            lines.append("VARIANT A — 0 pinned + 2 streaming (16-program ceiling)")
            let zero = try runZeroPinned(
                loader: loader, passes: 3, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            appendVariantSummary(zero, loader: loader, context: context, pressure: pressure, lines: &lines)
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("VARIANT B — 4 pinned + 2 streaming (48-program ceiling)")
            let four = try runPinnedPrefix(
                pinned: 4, loader: loader, passes: 3, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            appendVariantSummary(four, loader: loader, context: context, pressure: pressure, lines: &lines)
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("VARIANT C — 6 pinned + 2 streaming (64-program ceiling)")
            let six = try runPinnedPrefix(
                pinned: 6, loader: loader, passes: 3, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            appendVariantSummary(six, loader: loader, context: context, pressure: pressure, lines: &lines)

            let ranked = [zero, four, six].sorted { $0.steady.wall < $1.steady.wall }
            guard let winner = ranked.first else {
                throw AnimapkError.validation("V4 produced no variant result")
            }
            lines.append("")
            lines.append("MICHELIN SCORECARD — steady pass #1")
            for (rank, result) in ranked.enumerated() {
                let s = result.steady
                lines.append("#\(rank + 1) \(result.name): wall=\(ms(s.wall))ms (\(ms(s.wall / Double(blockCount)))ms/block) streamLoadANE=\(ms(s.loadANE))ms streamLoadWall=\(ms(s.loadWall))ms host=\(ms(s.hostWall))ms wait=\(ms(s.waitWall))ms evalWall=\(ms(s.evalWall))ms maxResident=\(s.maxResidentBlocks) blocks/\(s.maxResidentBlocks * programsPerBlock) programs")
            }
            let zeroWall = zero.steady.wall
            let gain = zeroWall - winner.steady.wall
            let gainPercent = zeroWall > 0 ? gain / zeroWall * 100 : 0
            lines.append("WINNER: \(winner.name) — \(ms(gain))ms faster than corrected 0+2 steady pass (\(pct(gainPercent))%).")
            lines.append("Probe-equivalent 8-step projection traversal at winner steady rate: \(seconds(winner.steady.wall * 8))s (excludes Metal attention/AdaLN/residual work, text encoder, and VAE).")
            lines.append("")
            lines.append("DECISION GUIDE")
            lines.append("- If 6+2 wins cleanly with maxResident=8 blocks and no pressure signals, it is the preferred production residency recipe before model-format changes.")
            lines.append("- If 4+2 is effectively tied with 6+2, prefer 4+2 for extra residency headroom.")
            lines.append("- If corrected 0+2 is close to pinned variants, keep the simpler 16-program ring.")
            lines.append("- If all variants remain loader-bound, V5 should attack program count/cold load directly with the multi-procedure proof-of-concept.")
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
        loader.tracker.resetPeak()
        var stats = PassStats()
        let started = ProcessInfo.processInfo.systemUptime

        for block in 0..<blockCount {
            let loaded = try loader.load(block: block)
            let eval = try evaluateBlock(loaded.models, surfaces: surfaces)
            recordLoad(loaded, into: &stats)
            stats.evalANE += eval.ane
            stats.evalWall += eval.wall
            stats.maxResidentBlocks = max(stats.maxResidentBlocks, loader.tracker.currentCount)

            if block < 2 || block % 4 == 3 || block == blockCount - 1 {
                lines.append("pre b\(two(block)) loadANE=\(ms(loaded.loadANE))ms loadWall=\(ms(loaded.loadWall))ms host=\(ms(loaded.hostWall))ms eval=\(ms(eval.wall))ms \(loader.tracker.compact) | \(MemorySnapshot.capture(context: context).compact)")
            }
            loaded.invalidate()
            if pressure.criticalSeen {
                throw AnimapkError.validation("critical memory pressure while preconditioning block \(block)")
            }
        }

        stats.wall = elapsedMS(since: started)
        stats.maxResidentBlocks = max(stats.maxResidentBlocks, loader.tracker.peakCount)
        return stats
    }

    private static func runZeroPinned(
        loader: SharedLoader,
        passes: Int,
        surfaces: ANEW8DiTSurfaces,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) throws -> VariantResult {
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("0+2 started with residual ANE blocks")
        }
        loader.tracker.resetPeak()
        let queue = DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-v4-zero-loader", qos: .userInitiated)
        var stats = Array(repeating: PassStats(), count: passes)

        let entry = try loader.load(block: 0)
        lines.append("0+2 entry b00 loadANE=\(ms(entry.loadANE))ms loadWall=\(ms(entry.loadWall))ms host=\(ms(entry.hostWall))ms")
        var current: LoadedBlock? = entry

        let total = passes * blockCount
        for position in 0..<total {
            let pass = position / blockCount
            let block = position % blockCount
            guard let active = current, active.block == block else {
                throw AnimapkError.validation("0+2 current block mismatch at position \(position)")
            }

            let hasNext = position + 1 < total
            let nextBlock = (block + 1) % blockCount
            let future = hasNext ? LoadFuture(block: nextBlock, loader: loader, queue: queue) : nil
            let stageStart = ProcessInfo.processInfo.systemUptime
            let eval = try evaluateBlock(active.models, surfaces: surfaces)

            var next: LoadedBlock?
            var waitMS = 0.0
            if let future {
                let waitStart = ProcessInfo.processInfo.systemUptime
                next = try future.wait()
                waitMS = elapsedMS(since: waitStart)
                if let loaded = next { recordLoad(loaded, into: &stats[pass]) }
            }

            let stageWall = elapsedMS(since: stageStart)
            stats[pass].wall += stageWall
            stats[pass].evalANE += eval.ane
            stats[pass].evalWall += eval.wall
            stats[pass].waitWall += waitMS
            stats[pass].maxWaitWall = max(stats[pass].maxWaitWall, waitMS)
            stats[pass].maxResidentBlocks = max(stats[pass].maxResidentBlocks, loader.tracker.currentCount)

            if pass == 1 {
                let nextText = hasNext ? "b\(two(nextBlock))" : "none"
                let loadText = next.map { "load=\(ms($0.loadWall))ms ANE=\(ms($0.loadANE)) host=\(ms($0.hostWall))" } ?? "load=0.0ms"
                lines.append("0+2 p1 b\(two(block)) eval=\(ms(eval.wall))ms -> \(nextText) \(loadText) wait=\(ms(waitMS))ms stage=\(ms(stageWall))ms \(loader.tracker.compact)")
            }

            active.invalidate()
            current = next
            if pressure.criticalSeen {
                throw AnimapkError.validation("critical memory pressure in 0+2 pass \(pass), block \(block)")
            }
        }

        current?.invalidate()
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("0+2 leaked \(loader.tracker.currentCount) resident blocks")
        }
        for index in stats.indices {
            stats[index].maxResidentBlocks = max(stats[index].maxResidentBlocks, min(loader.tracker.peakCount, 2))
        }
        return VariantResult(name: "0+2", pinned: 0, pinSetupANE: 0, pinSetupWall: 0, passes: stats)
    }

    private static func runPinnedPrefix(
        pinned: Int,
        loader: SharedLoader,
        passes: Int,
        surfaces: ANEW8DiTSurfaces,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) throws -> VariantResult {
        guard pinned > 0, pinned + 2 <= 8, pinned < blockCount else {
            throw AnimapkError.validation("invalid V4 pinned-prefix size \(pinned)")
        }
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("\(pinned)+2 started with residual ANE blocks")
        }
        loader.tracker.resetPeak()

        var pinnedBlocks: [Int: LoadedBlock] = [:]
        var pinSetupANE = 0.0
        let pinStart = ProcessInfo.processInfo.systemUptime
        for block in 0..<pinned {
            let loaded = try loader.load(block: block)
            pinSetupANE += loaded.loadANE
            pinnedBlocks[block] = loaded
            lines.append("\(pinned)+2 pin b\(two(block)) loadANE=\(ms(loaded.loadANE))ms loadWall=\(ms(loaded.loadWall))ms host=\(ms(loaded.hostWall))ms")
        }
        let pinSetupWall = elapsedMS(since: pinStart)
        lines.append("\(pinned)+2 pin setup total: ANE=\(ms(pinSetupANE))ms wall=\(ms(pinSetupWall))ms \(loader.tracker.compact)")

        let queue = DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-v4-p\(pinned)-loader", qos: .userInitiated)
        var passStats: [PassStats] = []
        passStats.reserveCapacity(passes)

        for pass in 0..<passes {
            var stats = PassStats()
            let passStart = ProcessInfo.processInfo.systemUptime

            // Fill the two stream slots while the pinned prefix executes.
            var currentFuture: LoadFuture? = LoadFuture(block: pinned, loader: loader, queue: queue)
            var nextFuture: LoadFuture? = pinned + 1 < blockCount
                ? LoadFuture(block: pinned + 1, loader: loader, queue: queue)
                : nil

            for block in 0..<pinned {
                guard let active = pinnedBlocks[block] else {
                    throw AnimapkError.validation("missing pinned block \(block)")
                }
                let eval = try evaluateBlock(active.models, surfaces: surfaces)
                stats.evalANE += eval.ane
                stats.evalWall += eval.wall
                stats.maxResidentBlocks = max(stats.maxResidentBlocks, loader.tracker.currentCount)
                if pass == 1 {
                    lines.append("\(pinned)+2 p1 pinned b\(two(block)) eval=\(ms(eval.wall))ms \(loader.tracker.compact)")
                }
                if pressure.criticalSeen {
                    throw AnimapkError.validation("critical memory pressure in \(pinned)+2 pinned block \(block)")
                }
            }

            for block in pinned..<blockCount {
                guard let future = currentFuture, future.block == block else {
                    throw AnimapkError.validation("\(pinned)+2 future mismatch for block \(block)")
                }
                let waitStart = ProcessInfo.processInfo.systemUptime
                let active = try future.wait()
                let waitMS = elapsedMS(since: waitStart)
                recordLoad(active, into: &stats)
                stats.waitWall += waitMS
                stats.maxWaitWall = max(stats.maxWaitWall, waitMS)
                stats.maxResidentBlocks = max(stats.maxResidentBlocks, loader.tracker.currentCount)

                let evalStart = ProcessInfo.processInfo.systemUptime
                let eval = try evaluateBlock(active.models, surfaces: surfaces)
                let evalStageWall = elapsedMS(since: evalStart)
                stats.evalANE += eval.ane
                stats.evalWall += eval.wall

                // Strict pinned+2 ceiling: free this stream slot before block+2
                // is allowed to enter residency.
                active.invalidate()
                let following = block + 2
                let followingFuture: LoadFuture? = following < blockCount
                    ? LoadFuture(block: following, loader: loader, queue: queue)
                    : nil
                currentFuture = nextFuture
                nextFuture = followingFuture

                stats.maxResidentBlocks = max(stats.maxResidentBlocks, loader.tracker.currentCount)
                if pass == 1 {
                    lines.append("\(pinned)+2 p1 stream b\(two(block)) load=\(ms(active.loadWall))ms ANE=\(ms(active.loadANE)) host=\(ms(active.hostWall)) wait=\(ms(waitMS))ms eval=\(ms(eval.wall))ms stage≈\(ms(waitMS + evalStageWall))ms \(loader.tracker.compact)")
                }
                if pressure.criticalSeen {
                    throw AnimapkError.validation("critical memory pressure in \(pinned)+2 stream block \(block)")
                }
            }

            stats.wall = elapsedMS(since: passStart)
            stats.maxResidentBlocks = max(stats.maxResidentBlocks, loader.tracker.currentCount)
            passStats.append(stats)
            lines.append(passSummary(name: "\(pinned)+2", pass: pass, stats: stats))
        }

        for loaded in pinnedBlocks.values { loaded.invalidate() }
        pinnedBlocks.removeAll()
        guard loader.tracker.currentCount == 0 else {
            throw AnimapkError.validation("\(pinned)+2 leaked \(loader.tracker.currentCount) resident blocks")
        }
        return VariantResult(
            name: "\(pinned)+2", pinned: pinned,
            pinSetupANE: pinSetupANE, pinSetupWall: pinSetupWall,
            passes: passStats)
    }

    private static func appendVariantSummary(
        _ result: VariantResult,
        loader: SharedLoader,
        context: MetalContext,
        pressure: PressureRecorder,
        lines: inout [String]
    ) {
        if result.pinned > 0 {
            lines.append("\(result.name) one-time pin setup: ANE=\(ms(result.pinSetupANE))ms wall=\(ms(result.pinSetupWall))ms")
        }
        for (pass, stats) in result.passes.enumerated() {
            lines.append(passSummary(name: result.name, pass: pass, stats: stats))
        }
        let s = result.steady
        lines.append("\(result.name) STEADY: wall=\(ms(s.wall))ms avg=\(ms(s.wall / Double(blockCount)))ms/block loadANE=\(ms(s.loadANE))ms loadWall=\(ms(s.loadWall))ms host=\(ms(s.hostWall))ms wait=\(ms(s.waitWall))ms eval=\(ms(s.evalWall))ms maxResident=\(s.maxResidentBlocks) blocks/\(s.maxResidentBlocks * programsPerBlock) programs | \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact) | now=\(loader.tracker.currentCount)")
    }

    private static func passSummary(name: String, pass: Int, stats: PassStats) -> String {
        "\(name) pass#\(pass): wall=\(ms(stats.wall))ms loads=\(stats.streamLoads) loadANE=\(ms(stats.loadANE))ms loadWall=\(ms(stats.loadWall))ms host=\(ms(stats.hostWall))ms avgLoad=\(ms(stats.averageLoadWall))ms (ANE=\(ms(stats.averageLoadANE)) host=\(ms(stats.averageHostWall))) wait=\(ms(stats.waitWall))ms avgWait=\(ms(stats.averageWaitWall))ms evalWall=\(ms(stats.evalWall))ms avgEval=\(ms(stats.averageEvalWall))ms maxLoad=\(ms(stats.maxLoadWall))ms maxWait=\(ms(stats.maxWaitWall))ms maxResident=\(stats.maxResidentBlocks)"
    }

    private static func recordLoad(_ loaded: LoadedBlock, into stats: inout PassStats) {
        stats.loadANE += loaded.loadANE
        stats.loadWall += loaded.loadWall
        stats.hostWall += loaded.hostWall
        stats.streamLoads += 1
        stats.maxLoadWall = max(stats.maxLoadWall, loaded.loadWall)
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
    private static func seconds(_ milliseconds: Double) -> String { String(format: "%.2f", milliseconds / 1_000) }
    private static func pct(_ value: Double) -> String { String(format: "%.1f", value) }
    private static func two(_ value: Int) -> String { String(format: "%02d", value) }
    private static func mb(_ bytes: UInt64) -> String { String(format: "%.0fMB", Double(bytes) / 1_048_576.0) }

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
