import Foundation
import Metal
import UIKit
#if canImport(Darwin)
import Darwin
#endif

/// Very short device-only experiment for the legacy W8 ANE loading bottleneck.
/// Production inference is untouched. The existing Diagnostics ANE-probe button
/// runs this experiment on the `experiment/ane-pagecache-saturation` branch.
///
/// The test separates ordinary file-backed VM residency from ANE residency:
///  - map the real 224 prepared `.mlmodelc` bundles and query `mincore`;
///  - measure an untouched 2-pinned + 2-streaming legacy warm6 traversal;
///  - fault all prepared bundle files into ordinary reclaimable pages;
///  - try 6+2, then step down after pressure/failure;
///  - always compare a prefaulted 2+2 traversal with the untouched 2+2 baseline.
///
/// A warm6 block is the exact post-cross-KV production working set:
/// fused self-QKV + self-O + cross-Q + cross-O + MLP-up + MLP-down.
enum ANERingProbe {
    private static let streamSlots = 2

    private final class WarningCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        private var token: NSObjectProtocol?

        init() {
            token = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                self?.lock.lock()
                self?.count += 1
                self?.lock.unlock()
            }
        }

        func value() -> Int {
            lock.lock(); defer { lock.unlock() }
            return count
        }

        deinit {
            if let token { NotificationCenter.default.removeObserver(token) }
        }
    }

    private final class Report {
        private(set) var lines: [String] = []
        private let liveURL: URL

        init() {
            let docs = FileManager.default.urls(
                for: .documentDirectory, in: .userDomainMask).first!
            liveURL = docs.appendingPathComponent("ANE_PAGECACHE_LIVE.txt")
            try? "".write(to: liveURL, atomically: true, encoding: .utf8)
        }

        func add(_ text: String = "") {
            lines.append(text)
            let current = lines.joined(separator: "\n") + "\n"
            try? current.write(to: liveURL, atomically: true, encoding: .utf8)
            print("[ANE-PAGECACHE] \(text)")
        }

        var text: String { lines.joined(separator: "\n") }
    }

    private final class Mapping {
        let url: URL
        let pointer: UnsafeMutableRawPointer
        let length: Int
        let pageSize: Int

        init(url: URL) throws {
            self.url = url
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else {
                throw AnimapkError.validation("open failed \(url.lastPathComponent), errno=\(errno)")
            }
            defer { close(fd) }

            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size > 0 else {
                throw AnimapkError.validation("stat failed \(url.lastPathComponent), errno=\(errno)")
            }
            length = Int(st.st_size)
            pageSize = Int(getpagesize())
            let raw = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0)
            guard raw != MAP_FAILED, let raw else {
                throw AnimapkError.validation("mmap failed \(url.lastPathComponent), errno=\(errno)")
            }
            pointer = raw
        }

        deinit { _ = munmap(pointer, length) }

        func touch() -> UInt8 {
            _ = madvise(pointer, length, MADV_SEQUENTIAL)
            _ = madvise(pointer, length, MADV_WILLNEED)
            var checksum: UInt8 = 0
            var offset = 0
            while offset < length {
                checksum ^= pointer.load(fromByteOffset: offset, as: UInt8.self)
                offset += pageSize
            }
            checksum ^= pointer.load(fromByteOffset: length - 1, as: UInt8.self)
            return checksum
        }

        func residency() -> (resident: Int, total: Int) {
            let pages = (length + pageSize - 1) / pageSize
            var vector = [Int8](repeating: 0, count: pages)
            let rc = vector.withUnsafeMutableBufferPointer {
                mincore(pointer, length, $0.baseAddress)
            }
            guard rc == 0 else { return (0, pages) }
            let resident = vector.reduce(into: 0) { result, byte in
                if (Int(byte) & 1) != 0 { result += 1 }
            }
            return (resident, pages)
        }
    }

    private final class Warm6Models {
        let qkv: A12ANEQKVModel
        let selfO: A12ANEProjectionModel
        let crossQ: A12ANEProjectionModel
        let crossO: A12ANEProjectionModel
        let mlpUp: A12ANEProjectionModel
        let mlpDown: A12ANEProjectionModel
        let loadMS: Double

        init(file: AnimapkFile, block: Int) throws {
            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(
                    file: file, block: block, suffix: suffix)
                guard let value = tensor.blobSHA256 else {
                    throw AnimapkError.validation("missing ANE hash b\(block) \(suffix)")
                }
                return value
            }

            func projection(_ suffix: String) throws -> A12ANEProjectionModel {
                guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                    throw AnimapkError.validation("missing ANE spec \(suffix)")
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
            loadMS = qkv.loadMilliseconds + selfO.loadMilliseconds
                + crossQ.loadMilliseconds + crossO.loadMilliseconds
                + mlpUp.loadMilliseconds + mlpDown.loadMilliseconds
        }

        func sentinel(_ surfaces: ANEW8DiTSurfaces) throws -> Double {
            var ms = 0.0
            _ = try selfO.evaluateInput(
                surfaces.tokenInput,
                output: surfaces.tokenOutput,
                milliseconds: &ms)
            return ms
        }

        func unload() {
            qkv.invalidate()
            selfO.invalidate()
            crossQ.invalidate()
            crossO.invalidate()
            mlpUp.invalidate()
            mlpDown.invalidate()
        }

        deinit { unload() }
    }

    private struct Traversal {
        let pinned: Int
        let completedBlocks: Int
        let loadedSets: Int
        let loadMS: Double
        let evalMS: Double
        let wallMS: Double
        let warnings: Int
        let error: String?

        var succeeded: Bool {
            completedBlocks == ModelConstants.ditBlocks && warnings == 0 && error == nil
        }

        var line: String {
            let blockAverage = loadedSets > 0 ? loadMS / Double(loadedSets) : 0
            let modelAverage = loadedSets > 0 ? loadMS / Double(loadedSets * 6) : 0
            return "p\(pinned)+s2 \(succeeded ? "PASS" : "PARTIAL") "
                + "blocks=\(completedBlocks)/28 sets=\(loadedSets) "
                + "load=\(f1(loadMS))ms avgBlock=\(f1(blockAverage))ms "
                + "avgModel=\(f1(modelAverage))ms eval=\(f1(evalMS))ms "
                + "wall=\(f1(wallMS))ms warnings=\(warnings)"
                + (error.map { " error=\($0)" } ?? "")
        }
    }

    static func run() async -> String {
        let report = Report()
        let warningCounter = WarningCounter()
        report.add("ANE legacy page-cache saturation probe v1")
        report.add("real prepared W8 cache; ordinary MAP_PRIVATE file pages; no mlock")
        report.add("baseline untouched p2+s2 -> prefault all224 -> p6+s2 with automatic step-down -> prefault p2+s2 comparison")
        report.add("live copy: Documents/ANE_PAGECACHE_LIVE.txt")

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

            let mappings = try mapPreparedCache(file: file)
            report.add("pack=\(ditURL.lastPathComponent) pageSize=\(getpagesize())B")
            report.add("mapped files=\(mappings.count) bytes=\(f1(megabytes(mappings)))MB avail=\(availableMB())MB")
            addResidency("INITIAL", mappings: mappings, report: report)

            let surfaces = try ANEW8DiTSurfaces(device: context.device)
            report.add("BASELINE — no explicit prefault")
            let baseline = traverse(
                file: file, surfaces: surfaces, pinned: 2,
                warnings: warningCounter)
            report.add(baseline.line)
            addResidency("AFTER-BASELINE", mappings: mappings, report: report)

            report.add("PREFILL — touch every page in all 224 prepared bundles")
            let warningBeforeFill = warningCounter.value()
            let fillStart = ProcessInfo.processInfo.systemUptime
            var touched = 0
            var checksum: UInt8 = 0
            for (index, mapping) in mappings.enumerated() {
                checksum ^= mapping.touch()
                touched += mapping.length
                if index % 32 == 31 || index == mappings.count - 1 {
                    report.add("PREFILL \(index + 1)/\(mappings.count) touched=\(f1(Double(touched) / 1_048_576.0))MB avail=\(availableMB())MB warnings=\(warningCounter.value() - warningBeforeFill)")
                }
                if warningCounter.value() > warningBeforeFill {
                    report.add("PREFILL STOP memory warning at file \(index + 1)/\(mappings.count)")
                    break
                }
            }
            let fillMS = (ProcessInfo.processInfo.systemUptime - fillStart) * 1_000
            let fillMB = Double(touched) / 1_048_576.0
            let fillRate = fillMS > 0 ? fillMB / (fillMS / 1_000.0) : 0
            report.add("PREFILL DONE bytes=\(f1(fillMB))MB wall=\(f1(fillMS))ms rate=\(f1(fillRate))MB/s checksum=\(checksum)")
            addResidency("AFTER-PREFILL", mappings: mappings, report: report)

            let fillWarned = warningCounter.value() > warningBeforeFill
            var candidate6: Traversal?
            var candidate4: Traversal?
            if fillWarned {
                report.add("AUTO-STEP page saturation warned -> skip p6/p4 and use p2+s2")
            } else {
                report.add("PREFAULT CANDIDATE — p6+s2")
                candidate6 = traverse(
                    file: file, surfaces: surfaces, pinned: 6,
                    warnings: warningCounter)
                report.add(candidate6!.line)
                addResidency("AFTER-p6", mappings: mappings, report: report)

                if candidate6?.succeeded != true {
                    report.add("AUTO-STEP p6+s2 warned/failed -> p4+s2")
                    touchAll(mappings)
                    candidate4 = traverse(
                        file: file, surfaces: surfaces, pinned: 4,
                        warnings: warningCounter)
                    report.add(candidate4!.line)
                    addResidency("AFTER-p4", mappings: mappings, report: report)
                }
            }

            report.add("COMPARE — re-touch all pages then p2+s2")
            let warningsBeforeRetouch = warningCounter.value()
            touchAll(mappings)
            let retouchWarned = warningCounter.value() > warningsBeforeRetouch
            if retouchWarned {
                report.add("COMPARE retouch generated memory warning; still attempting bounded p2+s2")
            }
            let prefault2 = traverse(
                file: file, surfaces: surfaces, pinned: 2,
                warnings: warningCounter)
            report.add(prefault2.line)
            addResidency("FINAL", mappings: mappings, report: report)

            if baseline.loadMS > 0, prefault2.loadMS > 0 {
                let ratio = baseline.loadMS / prefault2.loadMS
                report.add("SPEEDUP SAME-POLICY p2+s2 baseline=\(f1(baseline.loadMS))ms prefault=\(f1(prefault2.loadMS))ms ratio=\(f2(ratio))x delta=\(f1(baseline.loadMS - prefault2.loadMS))ms")
            }
            report.add("RESULT COMPLETE warningsTotal=\(warningCounter.value()) avail=\(availableMB())MB")
        } catch {
            report.add("RESULT FAIL \(error.localizedDescription)")
        }
        return report.text
    }

    private static func traverse(
        file: AnimapkFile,
        surfaces: ANEW8DiTSurfaces,
        pinned: Int,
        warnings: WarningCounter
    ) -> Traversal {
        let warningStart = warnings.value()
        let wallStart = ProcessInfo.processInfo.systemUptime
        var pinnedModels: [Int: Warm6Models] = [:]
        var slots: [Warm6Models?] = Array(repeating: nil, count: streamSlots)
        var completed = 0
        var loadedSets = 0
        var loadMS = 0.0
        var evalMS = 0.0
        var failure: String?

        func load(_ block: Int) throws -> Warm6Models {
            let value = try Warm6Models(file: file, block: block)
            loadMS += value.loadMS
            loadedSets += 1
            return value
        }
        func warned() -> Bool { warnings.value() > warningStart }

        defer {
            for value in pinnedModels.values { value.unload() }
            for value in slots.compactMap({ $0 }) { value.unload() }
        }

        do {
            for block in 0..<pinned {
                pinnedModels[block] = try load(block)
                if warned() { throw ProbeStop.memoryWarning }
            }
            for block in pinned..<min(ModelConstants.ditBlocks, pinned + streamSlots) {
                let slot = (block - pinned) % streamSlots
                slots[slot] = try load(block)
                if warned() { throw ProbeStop.memoryWarning }
            }

            for block in 0..<ModelConstants.ditBlocks {
                let model: Warm6Models
                if block < pinned {
                    guard let value = pinnedModels[block] else {
                        throw AnimapkError.validation("probe missing pinned block \(block)")
                    }
                    model = value
                } else {
                    let slot = (block - pinned) % streamSlots
                    guard let value = slots[slot] else {
                        throw AnimapkError.validation("probe missing streaming block \(block)")
                    }
                    model = value
                }
                evalMS += try model.sentinel(surfaces)
                completed += 1
                if warned() { throw ProbeStop.memoryWarning }

                if block >= pinned {
                    let slot = (block - pinned) % streamSlots
                    slots[slot]?.unload()
                    slots[slot] = nil
                    let next = block + streamSlots
                    if next < ModelConstants.ditBlocks {
                        slots[slot] = try load(next)
                        if warned() { throw ProbeStop.memoryWarning }
                    }
                }
            }
        } catch ProbeStop.memoryWarning {
            failure = "memory warning"
        } catch {
            failure = error.localizedDescription
        }

        return Traversal(
            pinned: pinned,
            completedBlocks: completed,
            loadedSets: loadedSets,
            loadMS: loadMS,
            evalMS: evalMS,
            wallMS: (ProcessInfo.processInfo.systemUptime - wallStart) * 1_000,
            warnings: warnings.value() - warningStart,
            error: failure)
    }

    private enum ProbeStop: Error { case memoryWarning }

    private static func mapPreparedCache(file: AnimapkFile) throws -> [Mapping] {
        let keys = try ANEW8ModelPreparer.expectedCacheKeys(file: file)
        guard keys.count == ANEW8NativePack.expectedPreparedModelCount else {
            throw AnimapkError.validation("expected 224 prepared cache keys, got \(keys.count)")
        }
        guard let root = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AnimaXS-ANE", isDirectory: true) else {
            throw AnimapkError.validation("ANE cache root unavailable")
        }

        var result: [Mapping] = []
        for key in keys {
            let bundle = root.appendingPathComponent(key + ".mlmodelc", isDirectory: true)
            guard let enumerator = FileManager.default.enumerator(
                at: bundle,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]) else {
                throw AnimapkError.validation("cannot enumerate \(bundle.lastPathComponent)")
            }
            for case let url as URL in enumerator {
                let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                if values.isRegularFile == true, (values.fileSize ?? 0) > 0 {
                    result.append(try Mapping(url: url))
                }
            }
        }
        return result
    }

    private static func touchAll(_ mappings: [Mapping]) {
        var checksum: UInt8 = 0
        for mapping in mappings { checksum ^= mapping.touch() }
        if checksum == 255 { print("[ANE-PAGECACHE] checksum=255") }
    }

    private static func addResidency(
        _ label: String,
        mappings: [Mapping],
        report: Report
    ) {
        var residentPages = 0
        var totalPages = 0
        var residentBytes = 0
        var totalBytes = 0
        for mapping in mappings {
            let state = mapping.residency()
            residentPages += state.resident
            totalPages += state.total
            residentBytes += min(mapping.length, state.resident * mapping.pageSize)
            totalBytes += mapping.length
        }
        let percent = totalPages > 0
            ? 100.0 * Double(residentPages) / Double(totalPages) : 0
        report.add("RESIDENCY \(label) resident=\(f1(Double(residentBytes) / 1_048_576.0))/\(f1(Double(totalBytes) / 1_048_576.0))MB pages=\(residentPages)/\(totalPages) \(f1(percent))% avail=\(availableMB())MB")
    }

    private static func megabytes(_ mappings: [Mapping]) -> Double {
        Double(mappings.reduce(0) { $0 + $1.length }) / 1_048_576.0
    }

    private static func availableMB() -> String {
        f1(Double(os_proc_available_memory()) / 1_048_576.0)
    }

    private static func f1(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private static func f2(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
