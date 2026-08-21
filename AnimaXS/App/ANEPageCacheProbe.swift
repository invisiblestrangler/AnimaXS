import Foundation
import Metal
import UIKit
#if canImport(Darwin)
import Darwin
#endif

/// Device-only diagnostic for the legacy W8 ANE loader bottleneck.
///
/// It deliberately separates ordinary file-backed page residency from private
/// ANE residency:
///   1. mmap the real prepared `.mlmodelc` bundle files without faulting them;
///   2. measure current residency with `mincore`;
///   3. run an untouched 2-pinned + 2-streaming warm6 baseline;
///   4. fault the complete steady-state warm6 prepared working set into normal,
///      reclaimable VM pages; if that is healthy, also fault the two first-pass
///      cross-K/V models per block to reach all 224 prepared legacy models;
///   5. run real six-program legacy block sets with 6+2 residency, automatically
///      stepping down to 4+2 then 2+2 after a memory warning/load failure;
///   6. always run a prefaulted 2+2 comparison when it is safe enough to do so.
///
/// Only one lightweight self_o dispatch per block is evaluated. The quantity
/// under test is `_ANEClient loadModel` wall time, not transformer correctness.
/// Production inference/scheduler state is never mutated.
enum ANEPageCacheProbe {
    private static let streamingSlots = 2
    private static let warmProjectionSuffixes = [
        "self_attn.output_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.output_proj.weight",
        "mlp.layer1.weight",
        "mlp.layer2.weight"
    ]
    private static let crossKVProjectionSuffixes = [
        "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight"
    ]

    private final class MemoryWarningCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        private var token: NSObjectProtocol?

        init() {
            token = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                self?.lock.lock()
                self?.value += 1
                self?.lock.unlock()
            }
        }

        func snapshot() -> Int {
            lock.lock(); defer { lock.unlock() }
            return value
        }

        deinit {
            if let token { NotificationCenter.default.removeObserver(token) }
        }
    }

    private final class LiveReport {
        private(set) var lines: [String] = []
        let url: URL

        init() {
            let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
                .first!
                .appendingPathComponent("AnimaXS-ANE", isDirectory: true)
            try? FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            url = root.appendingPathComponent(
                "pagecache-saturation-live-latest.log", isDirectory: false)
            try? "".write(to: url, atomically: true, encoding: .utf8)
        }

        func add(_ line: String = "") {
            lines.append(line)
            // Atomic rewrite is intentional: if jetsam wins, the latest complete
            // line remains recoverable on the next Diagnostics open.
            try? (lines.joined(separator: "\n") + "\n")
                .write(to: url, atomically: true, encoding: .utf8)
        }

        var text: String { lines.joined(separator: "\n") }
    }

    private final class FileMapping {
        let url: URL
        let pointer: UnsafeMutableRawPointer
        let length: Int
        let pageSize: Int

        init(url: URL) throws {
            self.url = url
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else {
                throw AnimapkError.validation(
                    "page-cache probe open failed: \(url.lastPathComponent) errno=\(errno)")
            }
            defer { close(fd) }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_size > 0 else {
                throw AnimapkError.validation(
                    "page-cache probe stat failed: \(url.lastPathComponent) errno=\(errno)")
            }
            let byteCount = Int(info.st_size)
            let mapped = mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, fd, 0)
            guard mapped != MAP_FAILED, let mapped else {
                throw AnimapkError.validation(
                    "page-cache probe mmap failed: \(url.lastPathComponent) errno=\(errno)")
            }
            pointer = mapped
            length = byteCount
            pageSize = Int(getpagesize())
        }

        deinit { munmap(pointer, length) }

        func touch() -> UInt8 {
            _ = madvise(pointer, length, MADV_SEQUENTIAL)
            _ = madvise(pointer, length, MADV_WILLNEED)
            var checksum: UInt8 = 0
            var offset = 0
            while offset < length {
                checksum ^= pointer.load(fromByteOffset: offset, as: UInt8.self)
                offset += pageSize
            }
            if length > 1 {
                checksum ^= pointer.load(fromByteOffset: length - 1, as: UInt8.self)
            }
            return checksum
        }

        func discardAdvice() {
            _ = madvise(pointer, length, MADV_DONTNEED)
        }

        func residency() -> (residentPages: Int, totalPages: Int) {
            let pageCount = (length + pageSize - 1) / pageSize
            var vector = [Int8](repeating: 0, count: pageCount)
            let rc = vector.withUnsafeMutableBufferPointer { buffer in
                mincore(pointer, length, buffer.baseAddress)
            }
            guard rc == 0 else { return (0, pageCount) }
            let resident = vector.reduce(into: 0) { count, byte in
                if (Int(byte) & 1) != 0 { count += 1 }
            }
            return (resident, pageCount)
        }
    }

    private struct MappingGroup {
        let warm: [[FileMapping]]
        let crossKVExtras: [[FileMapping]]

        var allWarm: [FileMapping] { warm.flatMap { $0 } }
        var allExtras: [FileMapping] { crossKVExtras.flatMap { $0 } }
        var all: [FileMapping] { allWarm + allExtras }
    }

    private final class WarmBlockModels {
        let qkv: A12ANEQKVModel
        let selfO: A12ANEProjectionModel
        let crossQ: A12ANEProjectionModel
        let crossO: A12ANEProjectionModel
        let mlpUp: A12ANEProjectionModel
        let mlpDown: A12ANEProjectionModel
        let loadMilliseconds: Double

        init(file: AnimapkFile, block: Int) throws {
            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(
                    file: file, block: block, suffix: suffix)
                guard let digest = tensor.blobSHA256 else {
                    throw AnimapkError.validation(
                        "page-cache probe hash missing b\(block) \(suffix)")
                }
                return digest
            }
            func projection(_ suffix: String) throws -> A12ANEProjectionModel {
                guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                    throw AnimapkError.validation(
                        "page-cache probe projection spec missing: \(suffix)")
                }
                let key = ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: try digest(suffix))
                return try A12ANEProjectionModel(
                    preparedInputChannels: UInt(spec.columns),
                    outputChannels: UInt(spec.rows),
                    spatial: UInt(spec.spatial),
                    label: "pagecache_b\(block)_\(spec.tag)",
                    cacheKey: key)
            }

            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block,
                q: try digest("self_attn.q_proj.weight"),
                k: try digest("self_attn.k_proj.weight"),
                v: try digest("self_attn.v_proj.weight"))
            qkv = try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(ModelConstants.ditTokensAt512),
                label: "pagecache_b\(block)_self_qkv",
                cacheKey: qkvKey)
            selfO = try projection("self_attn.output_proj.weight")
            crossQ = try projection("cross_attn.q_proj.weight")
            crossO = try projection("cross_attn.output_proj.weight")
            mlpUp = try projection("mlp.layer1.weight")
            mlpDown = try projection("mlp.layer2.weight")
            loadMilliseconds = qkv.loadMilliseconds + selfO.loadMilliseconds
                + crossQ.loadMilliseconds + crossO.loadMilliseconds
                + mlpUp.loadMilliseconds + mlpDown.loadMilliseconds
        }

        func evaluateSelfO(surfaces: ANEW8DiTSurfaces) throws -> Double {
            var milliseconds = 0.0
            _ = try selfO.evaluateInput(
                surfaces.tokenInput,
                output: surfaces.tokenOutput,
                milliseconds: &milliseconds)
            return milliseconds
        }

        func invalidateAll() {
            qkv.invalidate()
            selfO.invalidate()
            crossQ.invalidate()
            crossO.invalidate()
            mlpUp.invalidate()
            mlpDown.invalidate()
        }

        deinit { invalidateAll() }
    }

    private struct TraversalResult {
        let label: String
        let pinned: Int
        let completedBlocks: Int
        let blockSetsLoaded: Int
        let loadMilliseconds: Double
        let evalMilliseconds: Double
        let wallMilliseconds: Double
        let warningDelta: Int
        let failure: String?

        var completed: Bool {
            completedBlocks == ModelConstants.ditBlocks
                && warningDelta == 0
                && failure == nil
        }

        var compact: String {
            let avgBlockLoad = blockSetsLoaded > 0
                ? loadMilliseconds / Double(blockSetsLoaded) : 0
            let avgModelLoad = blockSetsLoaded > 0
                ? loadMilliseconds / Double(blockSetsLoaded * 6) : 0
            let state = completed ? "PASS" : "PARTIAL"
            return String(
                format: "%@ %@ p%d+s2 blocks=%d/28 sets=%d load=%.1fms avgBlock=%.1fms avgModel=%.1fms eval=%.1fms wall=%.1fms warnings=%d%@",
                label, state, pinned, completedBlocks, blockSetsLoaded,
                loadMilliseconds, avgBlockLoad, avgModelLoad,
                evalMilliseconds, wallMilliseconds, warningDelta,
                failure.map { " error=\($0)" } ?? "")
        }
    }

    static func previousLiveLog() -> String? {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("AnimaXS-ANE", isDirectory: true)
        guard let url = root?.appendingPathComponent(
            "pagecache-saturation-live-latest.log", isDirectory: false),
              let text = try? String(contentsOf: url, encoding: .utf8),
              !text.isEmpty else { return nil }
        return text
    }

    static func run() async -> String {
        let report = LiveReport()
        let warnings = MemoryWarningCounter()
        report.add("ANE legacy page-cache saturation probe v1")
        report.add("diagnostic only; real prepared legacy W8 bundles; warm6=QKV+selfO+crossQ+crossO+MLP1+MLP2")
        report.add("baseline=2 pinned + 2 streaming without explicit prefault; page cache then warm6 saturation -> full224 extras when safe")
        report.add("candidate=6+2; automatic 4+2 then 2+2 after warning/failure; prefaulted 2+2 always attempted when viable")
        report.add("live=\(report.url.path)")

        do {
            guard A12ANEIsAvailable() else {
                report.add("SKIP ANE unavailable: \(A12ANERuntimeStatus())")
                return report.text
            }
            guard let context = MetalContext() else {
                report.add("FAIL Metal unavailable")
                return report.text
            }
            guard let ditEntry = ModelManifest.entries.first(where: { $0.component == .dit }) else {
                report.add("FAIL DiT manifest entry missing")
                return report.text
            }
            let store = try ModelStore()
            let ditURL = await store.localURL(for: ditEntry)
            guard FileManager.default.fileExists(atPath: ditURL.path) else {
                report.add("FAIL installed DiT pack missing")
                return report.text
            }
            let file = try AnimapkFile(url: ditURL)
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                report.add("FAIL prepared ANE cache incomplete")
                return report.text
            }

            report.add("pack=\(ditURL.lastPathComponent) pageSize=\(getpagesize())B avail=\(availableMB())MB")
            let groups = try makeMappings(file: file)
            report.add(String(
                format: "mapped warm6=%d files %.1fMB; crossKV-extra=%d files %.1fMB; all224 bundles=%.1fMB virtual",
                groups.allWarm.count, bytesMB(groups.allWarm),
                groups.allExtras.count, bytesMB(groups.allExtras),
                bytesMB(groups.all)))

            reportResidency("RESIDENCY initial-warm6", mappings: groups.allWarm, report: report)
            reportResidency("RESIDENCY initial-full224", mappings: groups.all, report: report)

            let surfaces = try ANEW8DiTSurfaces(device: context.device)
            let baseline = runTraversal(
                label: "BASELINE-UNTOUCHED", file: file, surfaces: surfaces,
                pinned: 2, warnings: warnings, report: report)
            report.add(baseline.compact)
            reportResidency("RESIDENCY after-baseline-warm6", mappings: groups.allWarm, report: report)

            let warmFill = touchByBlock(
                label: "PREFILL warm6", blocks: groups.warm,
                warnings: warnings, report: report)
            reportResidency("RESIDENCY after-warm6-prefill", mappings: groups.allWarm, report: report)

            var saturationWarning = warmFill.warningDelta > 0
            if !saturationWarning {
                let fullFill = touchByBlock(
                    label: "PREFILL full224-crossKV-extra", blocks: groups.crossKVExtras,
                    warnings: warnings, report: report)
                saturationWarning = fullFill.warningDelta > 0
                reportResidency("RESIDENCY after-full224-prefill", mappings: groups.all, report: report)
                if fullFill.warningDelta > 0 {
                    report.add("FULL224 pressure: discarding crossKV-extra mapping advice before fallback")
                    for mapping in groups.allExtras { mapping.discardAdvice() }
                }
            }

            var prefaultTwo: TraversalResult?
            if saturationWarning {
                report.add("AUTO-STEP: page saturation itself warned; skip 6+2 and start at 2+2")
                _ = touchByBlock(
                    label: "REWARM warm6 for p2", blocks: groups.warm,
                    warnings: warnings, report: report)
                prefaultTwo = runTraversal(
                    label: "PREFAULT", file: file, surfaces: surfaces,
                    pinned: 2, warnings: warnings, report: report)
                report.add(prefaultTwo!.compact)
            } else {
                let six = runTraversal(
                    label: "PREFAULT", file: file, surfaces: surfaces,
                    pinned: 6, warnings: warnings, report: report)
                report.add(six.compact)
                reportResidency("RESIDENCY after-p6", mappings: groups.allWarm, report: report)

                if !six.completed {
                    report.add("AUTO-STEP: 6+2 warned/failed -> 4+2")
                    _ = touchByBlock(
                        label: "REWARM warm6 for p4", blocks: groups.warm,
                        warnings: warnings, report: report)
                    let four = runTraversal(
                        label: "PREFAULT", file: file, surfaces: surfaces,
                        pinned: 4, warnings: warnings, report: report)
                    report.add(four.compact)
                    reportResidency("RESIDENCY after-p4", mappings: groups.allWarm, report: report)
                }

                report.add("COMPARE: rewarm and run minimal 2+2")
                _ = touchByBlock(
                    label: "REWARM warm6 for p2", blocks: groups.warm,
                    warnings: warnings, report: report)
                prefaultTwo = runTraversal(
                    label: "PREFAULT", file: file, surfaces: surfaces,
                    pinned: 2, warnings: warnings, report: report)
                report.add(prefaultTwo!.compact)
            }

            reportResidency("RESIDENCY final-warm6", mappings: groups.allWarm, report: report)
            if let two = prefaultTwo, baseline.loadMilliseconds > 0, two.loadMilliseconds > 0 {
                report.add(String(
                    format: "SPEEDUP p2+s2 legacy-load baseline=%.1fms prefault=%.1fms ratio=%.2fx delta=%.1fms",
                    baseline.loadMilliseconds, two.loadMilliseconds,
                    baseline.loadMilliseconds / two.loadMilliseconds,
                    baseline.loadMilliseconds - two.loadMilliseconds))
            }
            report.add("RESULT COMPLETE warningsTotal=\(warnings.snapshot()) avail=\(availableMB())MB")
        } catch {
            report.add("RESULT FAIL setup/error=\(error.localizedDescription)")
        }
        return report.text
    }

    private static func makeMappings(file: AnimapkFile) throws -> MappingGroup {
        var warmBlocks: [[FileMapping]] = []
        var extraBlocks: [[FileMapping]] = []
        warmBlocks.reserveCapacity(ModelConstants.ditBlocks)
        extraBlocks.reserveCapacity(ModelConstants.ditBlocks)
        for block in 0..<ModelConstants.ditBlocks {
            let warmKeys = try cacheKeys(file: file, block: block, includeCrossKV: false)
            let fullKeys = try cacheKeys(file: file, block: block, includeCrossKV: true)
            let extraKeys = fullKeys.filter { !warmKeys.contains($0) }
            warmBlocks.append(try bundleFiles(keys: warmKeys).map(FileMapping.init(url:)))
            extraBlocks.append(try bundleFiles(keys: extraKeys).map(FileMapping.init(url:)))
        }
        return MappingGroup(warm: warmBlocks, crossKVExtras: extraBlocks)
    }

    private static func cacheKeys(
        file: AnimapkFile, block: Int, includeCrossKV: Bool
    ) throws -> [String] {
        func digest(_ suffix: String) throws -> String {
            let tensor = try ANEW8NativePack.tensor(
                file: file, block: block, suffix: suffix)
            guard let digest = tensor.blobSHA256 else {
                throw AnimapkError.validation("page-cache key hash missing: \(suffix)")
            }
            return digest
        }
        var keys = [ANEW8NativePack.qkvCacheKey(
            block: block,
            q: try digest("self_attn.q_proj.weight"),
            k: try digest("self_attn.k_proj.weight"),
            v: try digest("self_attn.v_proj.weight"))]
        var suffixes = warmProjectionSuffixes
        if includeCrossKV { suffixes += crossKVProjectionSuffixes }
        for suffix in suffixes {
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                throw AnimapkError.validation("page-cache key spec missing: \(suffix)")
            }
            keys.append(ANEW8NativePack.projectionCacheKey(
                block: block, tag: spec.tag, hash: try digest(suffix)))
        }
        return keys
    }

    private static func bundleFiles(keys: [String]) throws -> [URL] {
        guard let cacheRoot = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AnimaXS-ANE", isDirectory: true) else {
            throw AnimapkError.validation("page-cache root unavailable")
        }
        var urls: [URL] = []
        for key in keys {
            let bundle = cacheRoot.appendingPathComponent(
                key + ".mlmodelc", isDirectory: true)
            guard FileManager.default.fileExists(atPath: bundle.path) else {
                throw AnimapkError.validation(
                    "prepared bundle missing for page-cache key \(key.prefix(48))")
            }
            guard let enumerator = FileManager.default.enumerator(
                at: bundle,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]) else {
                throw AnimapkError.validation(
                    "cannot enumerate prepared bundle \(bundle.lastPathComponent)")
            }
            for case let url as URL in enumerator {
                let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                if values.isRegularFile == true, (values.fileSize ?? 0) > 0 {
                    urls.append(url)
                }
            }
        }
        return urls
    }

    @discardableResult
    private static func touchByBlock(
        label: String,
        blocks: [[FileMapping]],
        warnings: MemoryWarningCounter,
        report: LiveReport
    ) -> (bytes: Int, wallMilliseconds: Double, warningDelta: Int, checksum: UInt8) {
        let warningStart = warnings.snapshot()
        let wallStart = ProcessInfo.processInfo.systemUptime
        var bytes = 0
        var checksum: UInt8 = 0
        for (block, mappings) in blocks.enumerated() {
            for mapping in mappings {
                checksum ^= mapping.touch()
                bytes += mapping.length
            }
            if block % 4 == 3 || block == blocks.count - 1 {
                report.add(String(
                    format: "%@ progress block=%d/%d touched=%.1fMB avail=%.0fMB warnings=%d",
                    label, block + 1, blocks.count,
                    Double(bytes) / 1_048_576.0,
                    Double(os_proc_available_memory()) / 1_048_576.0,
                    warnings.snapshot() - warningStart))
            }
            if warnings.snapshot() > warningStart {
                report.add("\(label) STOP after memory warning at block \(block)")
                break
            }
        }
        let wallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
        let delta = warnings.snapshot() - warningStart
        let mbps = wallMS > 0 ? (Double(bytes) / 1_048_576.0) / (wallMS / 1_000.0) : 0
        report.add(String(
            format: "%@ DONE touched=%.1fMB wall=%.1fms throughput=%.0fMB/s warnings=%d checksum=%u",
            label, Double(bytes) / 1_048_576.0, wallMS, mbps, delta, checksum))
        return (bytes, wallMS, delta, checksum)
    }

    private static func runTraversal(
        label: String,
        file: AnimapkFile,
        surfaces: ANEW8DiTSurfaces,
        pinned: Int,
        warnings: MemoryWarningCounter,
        report: LiveReport
    ) -> TraversalResult {
        precondition((0...8).contains(pinned))
        let warningStart = warnings.snapshot()
        let wallStart = ProcessInfo.processInfo.systemUptime
        var pinnedModels: [Int: WarmBlockModels] = [:]
        var slots: [WarmBlockModels?] = [nil, nil]
        var loadMS = 0.0
        var evalMS = 0.0
        var loadedSets = 0
        var completedBlocks = 0
        var failure: String?

        func load(_ block: Int) throws -> WarmBlockModels {
            let models = try WarmBlockModels(file: file, block: block)
            loadMS += models.loadMilliseconds
            loadedSets += 1
            return models
        }
        func warned() -> Bool { warnings.snapshot() > warningStart }
        defer {
            for model in pinnedModels.values { model.invalidateAll() }
            for model in slots.compactMap({ $0 }) { model.invalidateAll() }
        }

        do {
            for block in 0..<pinned {
                pinnedModels[block] = try load(block)
                if warned() {
                    failure = "memory warning while creating pinned prefix at b\(block)"
                    throw ProbeStop.stop
                }
            }
            for block in pinned..<min(ModelConstants.ditBlocks, pinned + streamingSlots) {
                let slot = (block - pinned) % streamingSlots
                slots[slot] = try load(block)
                if warned() {
                    failure = "memory warning while priming stream slot at b\(block)"
                    throw ProbeStop.stop
                }
            }

            for block in 0..<ModelConstants.ditBlocks {
                let model: WarmBlockModels
                if block < pinned {
                    guard let value = pinnedModels[block] else {
                        throw AnimapkError.validation("page-cache probe missing pinned b\(block)")
                    }
                    model = value
                } else {
                    let slot = (block - pinned) % streamingSlots
                    guard let value = slots[slot] else {
                        throw AnimapkError.validation("page-cache probe missing stream slot b\(block)")
                    }
                    model = value
                }
                evalMS += try model.evaluateSelfO(surfaces: surfaces)
                completedBlocks += 1
                if warned() {
                    failure = "memory warning after eval b\(block)"
                    throw ProbeStop.stop
                }

                if block >= pinned {
                    let slot = (block - pinned) % streamingSlots
                    slots[slot]?.invalidateAll()
                    slots[slot] = nil
                    let next = block + streamingSlots
                    if next < ModelConstants.ditBlocks {
                        slots[slot] = try load(next)
                        if warned() {
                            failure = "memory warning loading streamed b\(next)"
                            throw ProbeStop.stop
                        }
                    }
                }
            }
        } catch ProbeStop.stop {
            // Expected auto-step boundary; partial metrics are still useful.
        } catch {
            failure = error.localizedDescription
        }

        let wallMS = (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000
        let delta = warnings.snapshot() - warningStart
        report.add(String(
            format: "%@ p%d+s2 progress blocks=%d sets=%d avail=%.0fMB",
            label, pinned, completedBlocks, loadedSets,
            Double(os_proc_available_memory()) / 1_048_576.0))
        return TraversalResult(
            label: label,
            pinned: pinned,
            completedBlocks: completedBlocks,
            blockSetsLoaded: loadedSets,
            loadMilliseconds: loadMS,
            evalMilliseconds: evalMS,
            wallMilliseconds: wallMS,
            warningDelta: delta,
            failure: failure)
    }

    private enum ProbeStop: Error { case stop }

    private static func reportResidency(
        _ label: String, mappings: [FileMapping], report: LiveReport
    ) {
        var residentBytes = 0
        var totalBytes = 0
        var residentPages = 0
        var totalPages = 0
        for mapping in mappings {
            let pages = mapping.residency()
            residentPages += pages.residentPages
            totalPages += pages.totalPages
            totalBytes += mapping.length
            residentBytes += min(mapping.length, pages.residentPages * mapping.pageSize)
        }
        let ratio = totalPages > 0 ? 100.0 * Double(residentPages) / Double(totalPages) : 0
        report.add(String(
            format: "%@ resident=%.1f/%.1fMB pages=%d/%d %.1f%% avail=%.0fMB",
            label,
            Double(residentBytes) / 1_048_576.0,
            Double(totalBytes) / 1_048_576.0,
            residentPages, totalPages, ratio,
            Double(os_proc_available_memory()) / 1_048_576.0))
    }

    private static func bytesMB(_ mappings: [FileMapping]) -> Double {
        Double(mappings.reduce(0) { $0 + $1.length }) / 1_048_576.0
    }

    private static func availableMB() -> String {
        String(format: "%.0f", Double(os_proc_available_memory()) / 1_048_576.0)
    }
}
