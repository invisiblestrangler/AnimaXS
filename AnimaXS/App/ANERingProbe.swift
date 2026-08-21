import Foundation
import Metal
import UIKit

/// Device-only probe for one remaining ANE scheduling question:
/// can `_ANEClient loadModel` overlap real `evaluateWithModel` work on A12?
///
/// Four top-level cases are reported:
///  1. shared-connection load alone;
///  2. shared-connection evaluation alone;
///  3. shared-connection load + evaluation started concurrently;
///  4. load + evaluation started concurrently on two independently initialized
///     `_ANEClient` objects, with independent-client isolated controls.
///
/// The measured load unit is one exact production warm6 block admission:
/// fused self-QKV + self-O + cross-Q + cross-O + MLP-up + MLP-down. Each target
/// is unloaded while retaining its `_ANEModel`, other model identities are
/// churned to evict immediate runtime-hot state, and its prepared bundles are
/// prefaulted before timing. This removes cold filesystem and Swift object
/// construction from the concurrency measurement.
///
/// Production inference and scheduler code are not modified.
enum ANERingProbe {
    private static let projectionSuffixes = [
        "self_attn.output_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.output_proj.weight",
        "mlp.layer1.weight",
        "mlp.layer2.weight"
    ]
    private static let evalSuffix = "mlp.layer1.weight"
    private static let initialChurnCount = 12
    private static let extendedChurnCount = 24
    private static let coldEnoughWarm6MS = 100.0

    private struct LoadResult: Sendable {
        let apiMS: Double
        let wallMS: Double
        let error: String?

        var ok: Bool { error == nil && apiMS >= 0 }
    }

    private struct EvalResult: Sendable {
        let calls: Int
        let apiMS: Double
        let wallMS: Double
        let error: String?

        var ok: Bool { error == nil }
    }

    private struct JointResult: Sendable {
        let load: LoadResult
        let eval: EvalResult
        let wallMS: Double
    }

    private struct PrefaultResult: Sendable {
        let bytes: UInt64
        let wallMS: Double
        let checksum: UInt8
    }

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

    private final class Box<T>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: T?

        func set(_ newValue: T) {
            lock.lock(); value = newValue; lock.unlock()
        }

        func get() -> T? {
            lock.lock(); defer { lock.unlock() }
            return value
        }
    }

    /// Type-erased retained model with exactly the diagnostic lifecycle methods
    /// needed by this probe. The closures retain the Objective-C model object.
    private final class ReloadableModel: @unchecked Sendable {
        let object: NSObject
        private let unloadImpl: () -> Bool
        private let reloadImpl: () -> Double
        private let invalidateImpl: () -> Void

        init(_ model: A12ANEProjectionModel) {
            object = model
            unloadImpl = { model.diagnosticUnloadKeepingModel() }
            reloadImpl = { model.diagnosticReloadMilliseconds() }
            invalidateImpl = { model.invalidate() }
        }

        init(_ model: A12ANEQKVModel) {
            object = model
            unloadImpl = { model.diagnosticUnloadKeepingModel() }
            reloadImpl = { model.diagnosticReloadMilliseconds() }
            invalidateImpl = { model.invalidate() }
        }

        func unload() -> Bool { unloadImpl() }
        func reload() -> Double { reloadImpl() }
        func invalidate() { invalidateImpl() }

        func setClient(_ client: NSObject) throws {
            let selector = NSSelectorFromString("setClient:")
            guard object.responds(to: selector) else {
                throw AnimapkError.validation(
                    "ANE wrapper does not expose diagnostic private client setter")
            }
            object.setValue(client, forKey: "client")
        }
    }

    /// Six resident legacy programs matching the post-cross-KV production path.
    private final class Warm6Target: @unchecked Sendable {
        let block: Int
        let cacheKeys: [String]
        private let models: [ReloadableModel]

        init(file: AnimapkFile, block: Int) throws {
            self.block = block

            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(
                    file: file, block: block, suffix: suffix)
                guard let digest = tensor.blobSHA256 else {
                    throw AnimapkError.validation(
                        "concurrency probe missing digest b\(block) \(suffix)")
                }
                return digest
            }

            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block,
                q: try digest("self_attn.q_proj.weight"),
                k: try digest("self_attn.k_proj.weight"),
                v: try digest("self_attn.v_proj.weight"))
            let qkv = try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(ModelConstants.ditTokensAt512),
                label: "conc_b\(block)_self_qkv",
                cacheKey: qkvKey)

            var retained: [ReloadableModel] = [ReloadableModel(qkv)]
            var keys: [String] = [qkvKey]
            for suffix in ANERingProbe.projectionSuffixes {
                guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                    throw AnimapkError.validation(
                        "concurrency probe missing projection spec \(suffix)")
                }
                let key = ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: try digest(suffix))
                let model = try A12ANEProjectionModel(
                    preparedInputChannels: UInt(spec.columns),
                    outputChannels: UInt(spec.rows),
                    spatial: UInt(spec.spatial),
                    label: "conc_b\(block)_\(spec.tag)",
                    cacheKey: key)
                retained.append(ReloadableModel(model))
                keys.append(key)
            }
            models = retained
            cacheKeys = keys
        }

        func unloadAll() -> Bool {
            models.reduce(true) { partial, model in model.unload() && partial }
        }

        func reloadAll() -> LoadResult {
            let start = nowMS()
            var api = 0.0
            for model in models {
                let value = model.reload()
                if value < 0 {
                    return LoadResult(
                        apiMS: api,
                        wallMS: nowMS() - start,
                        error: "reload failed after \(api) ms")
                }
                api += value
            }
            return LoadResult(
                apiMS: api,
                wallMS: nowMS() - start,
                error: nil)
        }

        func setClient(_ client: NSObject) throws {
            for model in models { try model.setClient(client) }
        }

        func invalidateAll() {
            for model in models { model.invalidate() }
        }

        deinit { invalidateAll() }
    }

    private final class EvalContext: @unchecked Sendable {
        let model: A12ANEProjectionModel
        let surfaces: ANEW8DiTSurfaces

        init(model: A12ANEProjectionModel, surfaces: ANEW8DiTSurfaces) {
            self.model = model
            self.surfaces = surfaces
        }
    }

    static func run() async -> String {
        let warnings = WarningCounter()
        var lines = [
            "ANE load/eval concurrency probe v1",
            "cases=1 load-alone(shared), 2 eval-alone(shared), 3 concurrent(shared), 4 concurrent(two independent clients)",
            "load-unit=real warm6 block; measured reload excludes wrapper construction; backing bundles prefaulted after runtime-hot churn",
            "classification: overlapScore 0≈serialized, 1≈ideal full overlap",
            ""
        ]

        do {
            guard A12ANEIsAvailable() else {
                lines.append("SKIP ANE unavailable: \(A12ANERuntimeStatus())")
                return lines.joined(separator: "\n")
            }
            guard let context = MetalContext() else {
                lines.append("FAIL Metal unavailable")
                return lines.joined(separator: "\n")
            }
            guard let ditEntry = ModelManifest.entries.first(where: { $0.component == .dit }) else {
                lines.append("FAIL DiT manifest entry missing")
                return lines.joined(separator: "\n")
            }
            let store = try ModelStore()
            let ditURL = await store.localURL(for: ditEntry)
            guard FileManager.default.fileExists(atPath: ditURL.path) else {
                lines.append("FAIL installed DiT pack missing")
                return lines.joined(separator: "\n")
            }
            let file = try AnimapkFile(url: ditURL)
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                lines.append("FAIL prepared ANE cache incomplete")
                return lines.joined(separator: "\n")
            }

            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("")

            let surfaces = try ANEW8DiTSurfaces(device: context.device)

            // CASE 1 — shared-connection load alone.
            lines.append("CASE 1 — LOAD ALONE / sharedConnection")
            let loadTarget = try Warm6Target(file: file, block: 12)
            guard loadTarget.unloadAll() else {
                throw AnimapkError.validation("case1 initial warm6 unload failed")
            }
            var churnCount = initialChurnCount
            let pf1 = try prepareColdPageHot(
                target: loadTarget, file: file, churnCount: churnCount)
            var sharedLoadAlone = loadTarget.reloadAll()
            if sharedLoadAlone.ok && sharedLoadAlone.apiMS < coldEnoughWarm6MS {
                lines.append(String(
                    format: "case1 first reload %.1fms still runtime-hot; extending churn %d→%d",
                    sharedLoadAlone.apiMS, churnCount, extendedChurnCount))
                guard loadTarget.unloadAll() else {
                    throw AnimapkError.validation("case1 retry unload failed")
                }
                churnCount = extendedChurnCount
                _ = try prepareColdPageHot(
                    target: loadTarget, file: file, churnCount: churnCount)
                sharedLoadAlone = loadTarget.reloadAll()
            }
            lines.append(formatLoad(
                prefix: "case1", result: sharedLoadAlone,
                prefault: pf1, churnCount: churnCount))
            guard loadTarget.unloadAll() else {
                throw AnimapkError.validation("case1 final unload failed")
            }
            loadTarget.invalidateAll()

            // Shared evaluation model + dynamic loop length (~>=600ms and at
            // least 3x the isolated load) so overlap/no-overlap is obvious.
            let sharedEvalModel = try makeProjection(
                file: file, block: 0, suffix: evalSuffix,
                label: "conc_shared_eval")
            let sharedEval = EvalContext(model: sharedEvalModel, surfaces: surfaces)
            _ = try prefault(cacheKeys: [try projectionKey(
                file: file, block: 0, suffix: evalSuffix)])
            _ = measureEval(sharedEval, calls: 2) // warm-up outside all cases
            let oneEval = measureEval(sharedEval, calls: 1)
            guard oneEval.ok, oneEval.apiMS > 0 else {
                throw AnimapkError.validation(
                    "unable to calibrate ANE evaluation loop: \(oneEval.error ?? "unknown")")
            }
            let targetEvalMS = max(600.0, sharedLoadAlone.wallMS * 3.0)
            let evalCalls = max(16, min(256,
                Int(ceil(targetEvalMS / max(oneEval.apiMS, 0.5)))))
            lines.append(String(
                format: "calibration oneEval=%.2fms targetEval=%.1fms calls=%d",
                oneEval.apiMS, targetEvalMS, evalCalls))
            lines.append("")

            // CASE 2 — shared evaluation alone.
            lines.append("CASE 2 — EVAL ALONE / sharedConnection")
            let sharedEvalAlone = measureEval(sharedEval, calls: evalCalls)
            lines.append(formatEval(prefix: "case2", result: sharedEvalAlone))
            lines.append("")

            // CASE 3 — same shared client object underneath both wrappers.
            lines.append("CASE 3 — CONCURRENT LOAD+EVAL / sharedConnection")
            let sharedConcurrentTarget = try Warm6Target(file: file, block: 13)
            guard sharedConcurrentTarget.unloadAll() else {
                throw AnimapkError.validation("case3 initial warm6 unload failed")
            }
            let pf3 = try prepareColdPageHot(
                target: sharedConcurrentTarget, file: file, churnCount: churnCount)
            let sharedJoint = runConcurrent(
                eval: sharedEval,
                target: sharedConcurrentTarget,
                calls: evalCalls)
            lines.append(formatJoint(prefix: "case3", joint: sharedJoint, prefault: pf3))
            lines.append(overlapLine(
                prefix: "case3",
                evalBaseline: sharedEvalAlone.wallMS,
                loadBaseline: sharedLoadAlone.wallMS,
                joint: sharedJoint.wallMS))
            _ = sharedConcurrentTarget.unloadAll()
            sharedConcurrentTarget.invalidateAll()
            sharedEvalModel.invalidate()
            lines.append("")

            // CASE 4 — two separately initialized _ANEClient objects.
            lines.append("CASE 4 — CONCURRENT LOAD+EVAL / two independent _ANEClient objects")
            do {
                let evalClient = try makeIndependentClient()
                let loadClient = try makeIndependentClient()
                let distinct = evalClient !== loadClient
                lines.append(
                    "independent clients: eval=\(ObjectIdentifier(evalClient)) load=\(ObjectIdentifier(loadClient)) distinct=\(distinct ? "yes" : "NO")")
                guard distinct else {
                    throw AnimapkError.validation(
                        "_ANEClient init returned the same object twice")
                }

                let independentEvalModel = try makeProjection(
                    file: file, block: 1, suffix: evalSuffix,
                    label: "conc_independent_eval")
                guard independentEvalModel.diagnosticUnloadKeepingModel() else {
                    throw AnimapkError.validation(
                        "case4 eval model initial unload failed")
                }
                try setClient(independentEvalModel, client: evalClient)
                let evalReload = independentEvalModel.diagnosticReloadMilliseconds()
                guard evalReload >= 0 else {
                    throw AnimapkError.validation(
                        "case4 eval model failed to load on independent client")
                }
                let independentEval = EvalContext(
                    model: independentEvalModel, surfaces: surfaces)
                _ = measureEval(independentEval, calls: 2)
                let independentEvalAlone = measureEval(
                    independentEval, calls: evalCalls)
                lines.append(formatEval(
                    prefix: "case4-control-eval", result: independentEvalAlone))

                let independentTarget = try Warm6Target(file: file, block: 14)
                guard independentTarget.unloadAll() else {
                    throw AnimapkError.validation(
                        "case4 target initial unload failed")
                }
                try independentTarget.setClient(loadClient)

                let pf4a = try prepareColdPageHot(
                    target: independentTarget, file: file, churnCount: churnCount)
                let independentLoadAlone = independentTarget.reloadAll()
                lines.append(formatLoad(
                    prefix: "case4-control-load",
                    result: independentLoadAlone,
                    prefault: pf4a,
                    churnCount: churnCount))
                guard independentTarget.unloadAll() else {
                    throw AnimapkError.validation(
                        "case4 target control unload failed")
                }

                let pf4 = try prepareColdPageHot(
                    target: independentTarget, file: file, churnCount: churnCount)
                let independentJoint = runConcurrent(
                    eval: independentEval,
                    target: independentTarget,
                    calls: evalCalls)
                lines.append(formatJoint(
                    prefix: "case4", joint: independentJoint, prefault: pf4))
                lines.append(overlapLine(
                    prefix: "case4",
                    evalBaseline: independentEvalAlone.wallMS,
                    loadBaseline: independentLoadAlone.wallMS,
                    joint: independentJoint.wallMS))
                _ = independentTarget.unloadAll()
                independentTarget.invalidateAll()
                independentEvalModel.invalidate()
            } catch {
                lines.append("case4 SKIP/FAIL: \(error.localizedDescription)")
                lines.append("Interpretation: cases 1–3 remain valid; independent-client path was unavailable or rejected on this runtime.")
            }

            lines.append("")
            lines.append("RESULT COMPLETE memoryWarnings=\(warnings.value())")
            lines.append("Interpretation guide:")
            lines.append("  overlapScore≈0: API calls overlap on CPU threads but ANE load/eval serialize in practice")
            lines.append("  overlapScore≈1: load is substantially hidden under ANE evaluation")
            lines.append("  case3≈0 but case4≫0: sharedConnection is the bottleneck; separate clients are actionable")
            lines.append("  case3≈0 and case4≈0: serialization is likely below the client object (aned/driver/hardware path)")
        } catch {
            lines.append("FAIL setup/runtime: \(error.localizedDescription)")
            lines.append("memoryWarnings=\(warnings.value())")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Measured operations

    private static func measureEval(
        _ context: EvalContext,
        calls: Int
    ) -> EvalResult {
        let start = nowMS()
        var api = 0.0
        do {
            for _ in 0..<calls {
                var ms = 0.0
                _ = try context.model.evaluateInput(
                    context.surfaces.tokenInput,
                    output: context.surfaces.hidden,
                    milliseconds: &ms)
                api += ms
            }
            return EvalResult(
                calls: calls,
                apiMS: api,
                wallMS: nowMS() - start,
                error: nil)
        } catch {
            return EvalResult(
                calls: calls,
                apiMS: api,
                wallMS: nowMS() - start,
                error: error.localizedDescription)
        }
    }

    private static func runConcurrent(
        eval: EvalContext,
        target: Warm6Target,
        calls: Int
    ) -> JointResult {
        let ready = DispatchSemaphore(value: 0)
        let startGate = DispatchSemaphore(value: 0)
        let group = DispatchGroup()
        let evalBox = Box<EvalResult>()
        let loadBox = Box<LoadResult>()
        let evalQueue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-concurrency-eval",
            qos: .userInitiated)
        let loadQueue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-concurrency-load",
            qos: .userInitiated)

        group.enter()
        evalQueue.async {
            ready.signal()
            startGate.wait()
            evalBox.set(measureEval(eval, calls: calls))
            group.leave()
        }

        group.enter()
        loadQueue.async {
            ready.signal()
            startGate.wait()
            loadBox.set(target.reloadAll())
            group.leave()
        }

        ready.wait()
        ready.wait()
        let jointStart = nowMS()
        startGate.signal()
        startGate.signal()
        group.wait()
        let wall = nowMS() - jointStart

        return JointResult(
            load: loadBox.get() ?? LoadResult(
                apiMS: 0, wallMS: wall,
                error: "loader thread returned no result"),
            eval: evalBox.get() ?? EvalResult(
                calls: calls, apiMS: 0, wallMS: wall,
                error: "eval thread returned no result"),
            wallMS: wall)
    }

    // MARK: - Runtime-hot eviction + page-hot control

    private static func prepareColdPageHot(
        target: Warm6Target,
        file: AnimapkFile,
        churnCount: Int
    ) throws -> PrefaultResult {
        guard target.unloadAll() else {
            throw AnimapkError.validation(
                "warm6 target b\(target.block) unload before churn failed")
        }
        try churn(file: file, count: churnCount)
        return try prefault(cacheKeys: target.cacheKeys)
    }

    private static func churn(file: AnimapkFile, count: Int) throws {
        var made = 0
        var cursor = 0
        while made < count {
            let block = 16 + ((cursor / projectionSuffixes.count) % 12)
            let suffix = projectionSuffixes[cursor % projectionSuffixes.count]
            cursor += 1
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else { continue }
            let key = try projectionKey(file: file, block: block, suffix: suffix)
            _ = try prefault(cacheKeys: [key])
            let model = try A12ANEProjectionModel(
                preparedInputChannels: UInt(spec.columns),
                outputChannels: UInt(spec.rows),
                spatial: UInt(spec.spatial),
                label: "conc_churn_b\(block)_\(spec.tag)_\(made)",
                cacheKey: key)
            model.invalidate()
            made += 1
        }
    }

    private static func prefault(cacheKeys: [String]) throws -> PrefaultResult {
        let start = nowMS()
        var total: UInt64 = 0
        var checksum: UInt8 = 0
        let fm = FileManager.default
        guard let root = fm.urls(
            for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AnimaXS-ANE", isDirectory: true) else {
            throw AnimapkError.validation("unable to resolve ANE cache root")
        }

        for key in cacheKeys {
            let bundle = root.appendingPathComponent(
                "\(key).mlmodelc", isDirectory: true)
            guard fm.fileExists(atPath: bundle.path) else {
                throw AnimapkError.validation(
                    "prepared bundle missing for concurrency prefault")
            }
            guard let enumerator = fm.enumerator(
                at: bundle,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]) else {
                throw AnimapkError.validation(
                    "unable to enumerate prepared bundle")
            }
            for case let url as URL in enumerator {
                let values = try url.resourceValues(forKeys: [.isRegularFileKey])
                guard values.isRegularFile == true else { continue }
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                while true {
                    let data = try handle.read(upToCount: 1 << 20) ?? Data()
                    if data.isEmpty { break }
                    total += UInt64(data.count)
                    checksum ^= data.first ?? 0
                    checksum ^= data.last ?? 0
                }
            }
        }
        return PrefaultResult(
            bytes: total,
            wallMS: nowMS() - start,
            checksum: checksum)
    }

    // MARK: - Model / client construction

    private static func projectionKey(
        file: AnimapkFile,
        block: Int,
        suffix: String
    ) throws -> String {
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            throw AnimapkError.validation(
                "unknown projection suffix \(suffix)")
        }
        let tensor = try ANEW8NativePack.tensor(
            file: file, block: block, suffix: suffix)
        guard let digest = tensor.blobSHA256 else {
            throw AnimapkError.validation(
                "missing projection digest b\(block) \(suffix)")
        }
        return ANEW8NativePack.projectionCacheKey(
            block: block, tag: spec.tag, hash: digest)
    }

    private static func makeProjection(
        file: AnimapkFile,
        block: Int,
        suffix: String,
        label: String
    ) throws -> A12ANEProjectionModel {
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            throw AnimapkError.validation("unknown projection suffix")
        }
        return try A12ANEProjectionModel(
            preparedInputChannels: UInt(spec.columns),
            outputChannels: UInt(spec.rows),
            spatial: UInt(spec.spatial),
            label: label,
            cacheKey: try projectionKey(
                file: file, block: block, suffix: suffix))
    }

    private static func makeIndependentClient() throws -> NSObject {
        guard let clientType = NSClassFromString("_ANEClient") as? NSObject.Type else {
            throw AnimapkError.validation("_ANEClient class unavailable")
        }
        let client = clientType.init()
        guard client.responds(to: NSSelectorFromString(
            "loadModel:options:qos:error:")),
              client.responds(to: NSSelectorFromString(
                "evaluateWithModel:options:request:qos:error:")) else {
            throw AnimapkError.validation(
                "independently initialized _ANEClient lacks load/evaluate selectors")
        }
        return client
    }

    private static func setClient(
        _ model: A12ANEProjectionModel,
        client: NSObject
    ) throws {
        let object = model as NSObject
        guard object.responds(to: NSSelectorFromString("setClient:")) else {
            throw AnimapkError.validation(
                "projection wrapper private client setter unavailable")
        }
        object.setValue(client, forKey: "client")
    }

    // MARK: - Reporting

    private static func overlapLine(
        prefix: String,
        evalBaseline: Double,
        loadBaseline: Double,
        joint: Double
    ) -> String {
        let serial = evalBaseline + loadBaseline
        let ideal = max(evalBaseline, loadBaseline)
        let hideable = max(1.0, serial - ideal)
        let rawScore = (serial - joint) / hideable
        let score = max(-2.0, min(2.0, rawScore))
        let classification: String
        if score >= 0.75 {
            classification = "OVERLAP"
        } else if score <= 0.25 {
            classification = "SERIALIZED"
        } else {
            classification = "PARTIAL"
        }
        return String(
            format: "%@ COMPARISON serialBaseline=%.1fms idealOverlap=%.1fms joint=%.1fms overlapScore=%.2f => %@",
            prefix, serial, ideal, joint, score, classification)
    }

    private static func formatLoad(
        prefix: String,
        result: LoadResult,
        prefault: PrefaultResult,
        churnCount: Int
    ) -> String {
        String(
            format: "%@ load ok=%@ api=%.1fms wall=%.1fms churn=%d prefault=%.1fMB/%.1fms checksum=%u%@",
            prefix, result.ok ? "yes" : "NO",
            result.apiMS, result.wallMS, churnCount,
            Double(prefault.bytes) / 1_048_576.0,
            prefault.wallMS, prefault.checksum,
            result.error.map { " error=\($0)" } ?? "")
    }

    private static func formatEval(
        prefix: String,
        result: EvalResult
    ) -> String {
        String(
            format: "%@ eval ok=%@ calls=%d apiSum=%.1fms wall=%.1fms avgApi=%.2fms%@",
            prefix, result.ok ? "yes" : "NO",
            result.calls, result.apiMS, result.wallMS,
            result.calls > 0 ? result.apiMS / Double(result.calls) : 0,
            result.error.map { " error=\($0)" } ?? "")
    }

    private static func formatJoint(
        prefix: String,
        joint: JointResult,
        prefault: PrefaultResult
    ) -> String {
        String(
            format: "%@ jointWall=%.1fms | load(api=%.1f wall=%.1f ok=%@) | eval(calls=%d api=%.1f wall=%.1f ok=%@) | prefault=%.1fMB/%.1fms",
            prefix, joint.wallMS,
            joint.load.apiMS, joint.load.wallMS, joint.load.ok ? "yes" : "NO",
            joint.eval.calls, joint.eval.apiMS, joint.eval.wallMS,
            joint.eval.ok ? "yes" : "NO",
            Double(prefault.bytes) / 1_048_576.0, prefault.wallMS)
    }

    private static func nowMS() -> Double {
        ProcessInfo.processInfo.systemUptime * 1_000.0
    }
}
