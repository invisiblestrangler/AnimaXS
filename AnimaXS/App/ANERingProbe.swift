import Foundation
import Dispatch
import Metal
import UIKit
#if canImport(Darwin)
import Darwin
#endif

/// Physical-device experiment for the architecture suggested by the v2 results:
/// never keep a large pinned ANE set. Keep only the current DiT block and the
/// next prefetched block resident (8 + 8 programs), destroy/reconstruct model
/// wrappers on every rotation, and test whether warm `_ANEClient loadModel`
/// latency can be hidden behind the current block's ANE evaluation.
///
/// This is diagnostic-only. It does not alter production inference behavior.
enum ANERingProbe {
    private static let mib = 1_048_576.0
    private static let blockCount = ModelConstants.ditBlocks

    private final class PressureRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private let source: DispatchSourceMemoryPressure
        private var observer: NSObjectProtocol?
        private var events: [String] = []
        private var uiWarnings = 0
        private var stopped = false

        init() {
            source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.normal, .warning, .critical],
                queue: DispatchQueue(label: "com.invisiblestrangler.AnimaXS.ane-ring-pressure"))
            source.setEventHandler { [weak self] in
                guard let self else { return }
                let data = self.source.data
                var labels: [String] = []
                if data.contains(.normal) { labels.append("normal") }
                if data.contains(.warning) { labels.append("warning") }
                if data.contains(.critical) { labels.append("critical") }
                self.lock.lock()
                self.events.append(labels.joined(separator: "+"))
                self.lock.unlock()
            }
            observer = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil, queue: nil
            ) { [weak self] _ in
                self?.lock.lock()
                self?.uiWarnings += 1
                self?.lock.unlock()
            }
            source.resume()
        }

        var compact: String {
            lock.lock(); defer { lock.unlock() }
            return "UIKitWarnings=\(uiWarnings) dispatch=\(events.isEmpty ? "none" : events.joined(separator: ","))"
        }

        var hasCritical: Bool {
            lock.lock(); defer { lock.unlock() }
            return events.contains { $0.contains("critical") }
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

    private final class BlockSlot: @unchecked Sendable {
        let block: Int
        let cache: ANEW8DiTModelCache
        let models: ANEW8DiTModels
        let loadANE: Double
        let loadWall: Double
        private var invalidated = false

        init(file: AnimapkFile, block: Int) throws {
            let start = ProcessInfo.processInfo.systemUptime
            let cache = try ANEW8DiTModelCache(file: file)
            let result = try cache.models(for: block)
            self.block = block
            self.cache = cache
            self.models = result.models
            self.loadANE = result.newlyLoadedMilliseconds
            self.loadWall = (ProcessInfo.processInfo.systemUptime - start) * 1_000
        }

        func invalidate() {
            guard !invalidated else { return }
            invalidated = true
            models.selfQKV.invalidate()
            models.selfO.invalidate()
            models.crossQ.invalidate()
            models.crossK.invalidate()
            models.crossV.invalidate()
            models.crossO.invalidate()
            models.mlpUp.invalidate()
            models.mlpDown.invalidate()
        }

        deinit { invalidate() }
    }

    private final class FileBox: @unchecked Sendable {
        let file: AnimapkFile
        init(_ file: AnimapkFile) { self.file = file }
    }

    private final class SlotResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<BlockSlot, Error>?

        func set(_ value: Result<BlockSlot, Error>) {
            lock.lock(); result = value; lock.unlock()
        }

        func take() -> Result<BlockSlot, Error>? {
            lock.lock(); defer { lock.unlock() }
            let value = result
            result = nil
            return value
        }
    }

    private struct PassStats {
        var wall = 0.0
        var loadANE = 0.0
        var loadWall = 0.0
        var evalANE = 0.0
        var evalWall = 0.0
        var postEvalStall = 0.0
        var maxBlockLoadWall = 0.0
        var maxStageWall = 0.0

        var avgLoadWall: Double { loadWall / Double(blockCount) }
        var avgEvalWall: Double { evalWall / Double(blockCount) }
        var avgStageWall: Double { wall / Double(blockCount) }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE streaming-ring probe v3",
            "Candidate: zero pinned blocks; two-block ping-pong residency (8 current + 8 next = 16 programs max).",
            "Lifecycle: every rotation destroys/reconstructs wrappers from prepared .mlmodelc; no retained _ANEModel host cache.",
            "Sequence: process-first sequential pass -> warm sequential pass -> three sustained overlapped ring passes."
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
            let surfaces = try ANEW8DiTSurfaces(device: context.device)

            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("baseline: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
            lines.append("")

            lines.append("EXPERIMENT 1 — sequential ring, pass 0 (process-first / may be cold or partially warm)")
            let first = try sequentialPass(
                index: 0, file: file, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            lines.append(summary("sequential#0", first))
            if pressure.hasCritical { throw AnimapkError.validation("critical pressure after sequential pass 0") }
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("EXPERIMENT 2 — sequential ring, pass 1 (all 28 blocks have now been touched once)")
            let warmSequential = try sequentialPass(
                index: 1, file: file, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            lines.append(summary("sequential#1-warm", warmSequential))
            if pressure.hasCritical { throw AnimapkError.validation("critical pressure after warm sequential pass") }
            try? await Task.sleep(nanoseconds: 500_000_000)

            lines.append("")
            lines.append("EXPERIMENT 3 — sustained two-block overlap ring (3 passes / 84 evaluations)")
            let overlap = try overlapRing(
                passes: 3, file: file, surfaces: surfaces,
                context: context, pressure: pressure, lines: &lines)
            for (index, stats) in overlap.enumerated() {
                lines.append(summary("overlap#\(index)", stats))
            }

            if overlap.count >= 2 {
                let steady = overlap[1]
                let delta = warmSequential.wall - steady.wall
                let percent = warmSequential.wall > 0 ? delta / warmSequential.wall * 100 : 0
                lines.append("")
                lines.append(String(
                    format: "STEADY-STATE COMPARISON: warmSequential=%.1fms overlapMiddle=%.1fms gain=%.1fms (%.1f%%)",
                    warmSequential.wall, steady.wall, delta, percent))
                lines.append(String(
                    format: "overlapMiddle per-block: stage=%.1fms eval=%.1fms postEvalStall=%.1fms; maxStage=%.1fms",
                    steady.avgStageWall, steady.avgEvalWall,
                    steady.postEvalStall / Double(blockCount), steady.maxStageWall))
                lines.append(String(
                    format: "warm reconstructed load per block: %.1fms average, %.1fms max",
                    warmSequential.avgLoadWall, warmSequential.maxBlockLoadWall))
            }

            lines.append("")
            lines.append("INTERPRETATION")
            lines.append("- If overlap pass 1/2 stays near eval wall with little post-eval stall, a 16-program production ring is viable and pinning is unnecessary.")
            lines.append("- If warm sequential loads remain low but overlap stalls, loader/evaluator contention is the issue; test deeper prefetch or a tiny pinned set, not 8 pinned blocks.")
            lines.append("- If sequential pass 1 becomes cold again for early blocks, the private runtime cannot retain warm state across a full 28-block rotation; then multi-procedure/weight-rebinding becomes higher priority.")
            lines.append("- Any memory-pressure signal at only 16 resident programs invalidates the ring assumption and must be investigated before production use.")
            lines.append("final: \(MemorySnapshot.capture(context: context).compact) | \(pressure.compact)")
        } catch {
            lines.append("ERROR: \(error.localizedDescription)")
            lines.append("pressure at error: \(pressure.compact)")
        }

        return lines.joined(separator: "\n")
    }

    private static func sequentialPass(
        index: Int, file: AnimapkFile, surfaces: ANEW8DiTSurfaces,
        context: MetalContext, pressure: PressureRecorder, lines: inout [String]
    ) throws -> PassStats {
        var stats = PassStats()
        let passStart = ProcessInfo.processInfo.systemUptime

        for block in 0..<blockCount {
            var slot: BlockSlot? = try BlockSlot(file: file, block: block)
            guard let value = slot else {
                throw AnimapkError.validation("sequential slot unexpectedly nil")
            }
            let evalStart = ProcessInfo.processInfo.systemUptime
            let evalANE = try evaluateAll(value.models, surfaces: surfaces)
            let evalWall = (ProcessInfo.processInfo.systemUptime - evalStart) * 1_000

            stats.loadANE += value.loadANE
            stats.loadWall += value.loadWall
            stats.evalANE += evalANE
            stats.evalWall += evalWall
            stats.maxBlockLoadWall = max(stats.maxBlockLoadWall, value.loadWall)

            if block < 2 || block % 4 == 3 || block == blockCount - 1 || value.loadWall > 250 {
                lines.append(String(
                    format: "seq%d b%02d loadANE=%7.1fms loadWall=%7.1fms evalANE=%6.1fms evalWall=%6.1fms %@ | %@",
                    index, block, value.loadANE, value.loadWall, evalANE, evalWall,
                    MemorySnapshot.capture(context: context).compact, pressure.compact))
            }

            value.invalidate()
            slot = nil
            if pressure.hasCritical {
                throw AnimapkError.validation("critical memory pressure during sequential pass")
            }
        }

        stats.wall = (ProcessInfo.processInfo.systemUptime - passStart) * 1_000
        return stats
    }

    private static func overlapRing(
        passes: Int, file: AnimapkFile, surfaces: ANEW8DiTSurfaces,
        context: MetalContext, pressure: PressureRecorder, lines: inout [String]
    ) throws -> [PassStats] {
        guard passes > 0 else { return [] }
        let fileBox = FileBox(file)
        let queue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-ring-loader",
            qos: .userInitiated)

        var stats = Array(repeating: PassStats(), count: passes)
        var current: BlockSlot? = try BlockSlot(file: file, block: 0)
        guard let entry = current else {
            throw AnimapkError.validation("ring entry slot unexpectedly nil")
        }
        lines.append(String(
            format: "ring entry b00 loadANE=%.1fms loadWall=%.1fms",
            entry.loadANE, entry.loadWall))

        let total = passes * blockCount
        for position in 0..<total {
            let pass = position / blockCount
            let block = position % blockCount
            guard let active = current, active.block == block else {
                throw AnimapkError.validation("ring current block mismatch at position \(position)")
            }

            let hasNext = position + 1 < total
            let nextBlock = (block + 1) % blockCount
            let resultBox = SlotResultBox()
            let semaphore = DispatchSemaphore(value: 0)
            let stageStart = ProcessInfo.processInfo.systemUptime

            if hasNext {
                queue.async {
                    do {
                        resultBox.set(.success(try BlockSlot(file: fileBox.file, block: nextBlock)))
                    } catch {
                        resultBox.set(.failure(error))
                    }
                    semaphore.signal()
                }
            }

            let evalStart = ProcessInfo.processInfo.systemUptime
            let evalANE = try evaluateAll(active.models, surfaces: surfaces)
            let evalWall = (ProcessInfo.processInfo.systemUptime - evalStart) * 1_000

            var next: BlockSlot?
            var nextLoadANE = 0.0
            var nextLoadWall = 0.0
            if hasNext {
                semaphore.wait()
                guard let result = resultBox.take() else {
                    throw AnimapkError.validation("ring loader returned no result")
                }
                next = try result.get()
                nextLoadANE = next?.loadANE ?? 0
                nextLoadWall = next?.loadWall ?? 0
            }

            let stageWall = (ProcessInfo.processInfo.systemUptime - stageStart) * 1_000
            let stall = max(0, stageWall - evalWall)
            stats[pass].wall += stageWall
            stats[pass].loadANE += nextLoadANE
            stats[pass].loadWall += nextLoadWall
            stats[pass].evalANE += evalANE
            stats[pass].evalWall += evalWall
            stats[pass].postEvalStall += stall
            stats[pass].maxBlockLoadWall = max(stats[pass].maxBlockLoadWall, nextLoadWall)
            stats[pass].maxStageWall = max(stats[pass].maxStageWall, stageWall)

            if block < 2 || block % 4 == 3 || block == blockCount - 1 || stall > 25 || nextLoadWall > 150 {
                lines.append(String(
                    format: "ring%d b%02d evalWall=%6.1fms -> prefetch b%02d loadWall=%7.1fms stage=%7.1fms stall=%6.1fms %@ | %@",
                    pass, block, nextBlock, evalWall, nextLoadWall, stageWall, stall,
                    MemorySnapshot.capture(context: context).compact, pressure.compact))
            }

            active.invalidate()
            current = next
            if pressure.hasCritical {
                throw AnimapkError.validation("critical memory pressure during overlap ring")
            }
        }

        current?.invalidate()
        current = nil
        return stats
    }

    private static func evaluateAll(
        _ models: ANEW8DiTModels, surfaces: ANEW8DiTSurfaces
    ) throws -> Double {
        var total = 0.0
        var ms = 0.0
        _ = try models.selfQKV.evaluateInput(
            surfaces.tokenInput,
            qOutput: surfaces.q, kOutput: surfaces.k, vOutput: surfaces.v,
            milliseconds: &ms)
        total += ms

        func projection(
            _ model: A12ANEProjectionModel,
            _ input: A12ANESurface,
            _ output: A12ANESurface
        ) throws {
            var value = 0.0
            _ = try model.evaluateInput(input, output: output, milliseconds: &value)
            total += value
        }

        try projection(models.selfO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.crossQ, surfaces.tokenInput, surfaces.q)
        try projection(models.crossK, surfaces.contextInput, surfaces.contextK)
        try projection(models.crossV, surfaces.contextInput, surfaces.contextV)
        try projection(models.crossO, surfaces.tokenInput, surfaces.tokenOutput)
        try projection(models.mlpUp, surfaces.tokenInput, surfaces.hidden)
        try projection(models.mlpDown, surfaces.hidden, surfaces.tokenOutput)
        return total
    }

    private static func summary(_ name: String, _ stats: PassStats) -> String {
        String(
            format: "%@: wall=%.1fms loadANE=%.1fms loadWall=%.1fms (avg %.1f/block max %.1f) evalANE=%.1fms evalWall=%.1fms (avg %.1f/block) postEvalStall=%.1fms avgStage=%.1fms",
            name, stats.wall, stats.loadANE, stats.loadWall,
            stats.avgLoadWall, stats.maxBlockLoadWall,
            stats.evalANE, stats.evalWall, stats.avgEvalWall,
            stats.postEvalStall, stats.avgStageWall)
    }

    private static func mb(_ bytes: UInt64) -> String {
        String(format: "%.0fMB", Double(bytes) / mib)
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
