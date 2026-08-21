import CryptoKit
import Foundation
import Metal

/// Canonical metadata/runtime contract for the ANMA-v1 hybrid ANE W8 pack.
enum ANEW8NativePack {
    static let quantScheme = "w8-ane-hybrid-v1"
    static let tensorFormat = "ane_u8_per_row_fp32_v1"
    static let cacheFormat = "ane-u8-row-v1"
    static let expectedNativeTensorCount = 280
    static let expectedPreparedModelCount = 224

    struct ProjectionSpec: Equatable {
        let suffix: String
        let rows: Int
        let columns: Int
        let spatial: Int
        let tag: String
    }

    static let projectionSpecs: [ProjectionSpec] = [
        .init(suffix: "self_attn.q_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "self_q"),
        .init(suffix: "self_attn.k_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "self_k"),
        .init(suffix: "self_attn.v_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "self_v"),
        .init(suffix: "self_attn.output_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "self_o"),
        .init(suffix: "cross_attn.q_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "cross_q"),
        .init(suffix: "cross_attn.k_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.contextDim, spatial: DiTBlockExecutor.contextTokens, tag: "cross_k"),
        .init(suffix: "cross_attn.v_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.contextDim, spatial: DiTBlockExecutor.contextTokens, tag: "cross_v"),
        .init(suffix: "cross_attn.output_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "cross_o"),
        .init(suffix: "mlp.layer1.weight", rows: DiTBlockExecutor.hidden,
              columns: DiTBlockExecutor.dim, spatial: ModelConstants.ditTokensAt512, tag: "mlp1"),
        .init(suffix: "mlp.layer2.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.hidden, spatial: ModelConstants.ditTokensAt512, tag: "mlp2"),
    ]

    static func spec(suffix: String) -> ProjectionSpec? {
        projectionSpecs.first { $0.suffix == suffix }
    }

    static func validate(file: AnimapkFile) throws {
        guard file.component == "dit", file.quantScheme == quantScheme, file.quantGroup == 64 else {
            throw AnimapkError.validation(
                "ANE Hybrid W8 requires a w8-ane-hybrid-v1 DiT pack with group=64 metadata")
        }
        var nativeCount = 0
        for block in 0..<ModelConstants.ditBlocks {
            let prefix = "model.diffusion_model.blocks.\(block)."
            for spec in projectionSpecs {
                let name = prefix + spec.suffix
                guard let tensor = file.tensor(named: name) else {
                    throw AnimapkError.validation("ANE-native pack missing tensor: \(name)")
                }
                try validateNativeTensor(
                    tensor, expectedRows: spec.rows, expectedColumns: spec.columns)
                nativeCount += 1
            }
            let blockNative = file.tensors.filter {
                $0.name.hasPrefix(prefix) && $0.quantizationFormat == tensorFormat
            }
            guard blockNative.count == projectionSpecs.count else {
                throw AnimapkError.validation(
                    "ANE-native block \(block) has \(blockNative.count) native tensors; expected 10")
            }
        }
        let allNative = file.tensors.filter { $0.quantizationFormat == tensorFormat }
        guard nativeCount == expectedNativeTensorCount,
              allNative.count == expectedNativeTensorCount else {
            throw AnimapkError.validation(
                "ANE-native pack must contain exactly \(expectedNativeTensorCount) projection tensors")
        }
    }

    static func validateNativeTensor(
        _ tensor: AnimapkTensor, expectedRows: Int, expectedColumns: Int
    ) throws {
        guard tensor.storage == .w8,
              tensor.quantizationFormat == tensorFormat,
              tensor.shape == [expectedRows, expectedColumns] else {
            throw AnimapkError.validation(
                "invalid ANE-native tensor metadata for \(tensor.name)")
        }
        guard tensor.dataSize == UInt64(expectedRows * expectedColumns),
              tensor.scaleSize == UInt64(expectedRows * MemoryLayout<Float>.stride),
              tensor.zeroSize == UInt64(expectedRows * MemoryLayout<Float>.stride) else {
            throw AnimapkError.validation(
                "invalid ANE-native payload sizes for \(tensor.name)")
        }
        guard let digest = tensor.blobSHA256, isSHA256Hex(digest) else {
            throw AnimapkError.validation(
                "ANE-native tensor \(tensor.name) is missing a valid blob_sha256")
        }
    }

    static func nativeWeight(
        file: AnimapkFile, tensor: AnimapkTensor,
        expectedRows: Int, expectedColumns: Int
    ) throws -> ANEW8NativeWeight {
        try validateNativeTensor(
            tensor, expectedRows: expectedRows, expectedColumns: expectedColumns)
        guard let scale = file.scaleBytes(tensor), let bias = file.zeroBytes(tensor),
              let digest = tensor.blobSHA256 else {
            throw AnimapkError.validation("ANE-native tensor payload is incomplete: \(tensor.name)")
        }
        return ANEW8NativeWeight(
            q: try noCopyData(file.dataBytes(tensor), label: tensor.name + " data"),
            biasF32: try noCopyData(bias, label: tensor.name + " bias"),
            scaleF32: try noCopyData(scale, label: tensor.name + " scale"),
            blobSHA256: digest.lowercased())
    }

    static func tensor(file: AnimapkFile, block: Int, suffix: String) throws -> AnimapkTensor {
        let name = "model.diffusion_model.blocks.\(block).\(suffix)"
        guard let tensor = file.tensor(named: name), let spec = spec(suffix: suffix) else {
            throw AnimapkError.validation("ANE-native weight missing: \(name)")
        }
        try validateNativeTensor(tensor, expectedRows: spec.rows, expectedColumns: spec.columns)
        return tensor
    }

    static func projectionCacheKey(block: Int, tag: String, hash: String) -> String {
        "\(cacheFormat)-b\(block)-\(tag)-\(hash.lowercased())"
    }

    static func qkvCacheKey(block: Int, q: String, k: String, v: String) -> String {
        "\(cacheFormat)-b\(block)-selfqkv-\(q.lowercased())-\(k.lowercased())-\(v.lowercased())"
    }

    static func namespace(file: AnimapkFile) throws -> String {
        try validate(file: file)
        var hasher = SHA256()
        let native = file.tensors
            .filter { $0.quantizationFormat == tensorFormat }
            .sorted { $0.name < $1.name }
        for tensor in native {
            guard let digest = tensor.blobSHA256 else {
                throw AnimapkError.validation("ANE-native tensor hash missing: \(tensor.name)")
            }
            hasher.update(data: Data(tensor.name.utf8))
            hasher.update(data: Data([0]))
            hasher.update(data: Data(digest.lowercased().utf8))
            hasher.update(data: Data([10]))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func noCopyData(
        _ bytes: UnsafeRawBufferPointer, label: String
    ) throws -> Data {
        guard bytes.count > 0, let base = bytes.baseAddress else {
            throw AnimapkError.validation("empty ANE-native mmap region: \(label)")
        }
        return Data(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: base),
            count: bytes.count,
            deallocator: .none)
    }

    private static func isSHA256Hex(_ value: String) -> Bool {
        guard value.count == 64 else { return false }
        return value.unicodeScalars.allSatisfy {
            (48...57).contains($0.value) || (65...70).contains($0.value) || (97...102).contains($0.value)
        }
    }
}

struct ANEW8NativeWeight {
    /// mmap-backed views; `ANEW8DiTModelCache` retains the owning AnimapkFile.
    let q: Data
    let biasF32: Data
    let scaleF32: Data
    let blobSHA256: String

    var payloadBytes: Int { q.count + biasF32.count + scaleF32.count }
}

struct ANEW8PreparationResult: Equatable {
    let seconds: Double
    let payloadBytesWritten: UInt64
    let cacheHits: Int
    let cacheMisses: Int
    let preparedModels: Int
}

/// Materializes the 224 Espresso model directories before diffusion. It performs
/// disk/template work only; it never constructs `_ANEModel`, loads `_ANEClient`,
/// or evaluates ANE. The operation is idempotent and keyed by per-tensor SHA256.
enum ANEW8ModelPreparer {
    private struct Marker: Codable {
        let format: String
        let quantScheme: String
        let expectedModels: Int
        let sourceTensorCount: Int
        let namespace: String
        let complete: Bool
    }

    static func isPrepared(file: AnimapkFile) -> Bool {
        guard (try? ANEW8NativePack.validate(file: file)) != nil,
              let namespace = try? ANEW8NativePack.namespace(file: file),
              let marker = readMarker(namespace: namespace),
              marker.complete,
              marker.format == ANEW8NativePack.cacheFormat,
              marker.quantScheme == ANEW8NativePack.quantScheme,
              marker.expectedModels == ANEW8NativePack.expectedPreparedModelCount,
              marker.sourceTensorCount == ANEW8NativePack.expectedNativeTensorCount,
              marker.namespace == namespace,
              let keys = try? expectedCacheKeys(file: file),
              keys.count == ANEW8NativePack.expectedPreparedModelCount else {
            return false
        }
        return keys.allSatisfy { A12ANEPreparedModelExists($0) }
    }

    static func ensurePrepared(file: AnimapkFile) throws -> ANEW8PreparationResult {
        try ANEW8NativePack.validate(file: file)
        let started = ProcessInfo.processInfo.systemUptime
        let namespace = try ANEW8NativePack.namespace(file: file)
        if isPrepared(file: file) {
            return ANEW8PreparationResult(
                seconds: ProcessInfo.processInfo.systemUptime - started,
                payloadBytesWritten: 0,
                cacheHits: ANEW8NativePack.expectedPreparedModelCount,
                cacheMisses: 0,
                preparedModels: ANEW8NativePack.expectedPreparedModelCount)
        }

        var bytesWritten: UInt64 = 0
        var hits = 0
        var misses = 0
        for block in 0..<ModelConstants.ditBlocks {
            let prefix = "model.diffusion_model.blocks.\(block)."
            func weight(_ suffix: String) throws -> ANEW8NativeWeight {
                guard let spec = ANEW8NativePack.spec(suffix: suffix),
                      let tensor = file.tensor(named: prefix + suffix) else {
                    throw AnimapkError.validation("ANE-native preparation tensor missing: \(prefix)\(suffix)")
                }
                return try ANEW8NativePack.nativeWeight(
                    file: file, tensor: tensor,
                    expectedRows: spec.rows, expectedColumns: spec.columns)
            }

            let selfQ = try weight("self_attn.q_proj.weight")
            let selfK = try weight("self_attn.k_proj.weight")
            let selfV = try weight("self_attn.v_proj.weight")
            let qkvKey = ANEW8NativePack.qkvCacheKey(
                block: block, q: selfQ.blobSHA256, k: selfK.blobSHA256, v: selfV.blobSHA256)
            if A12ANEPreparedModelExists(qkvKey) {
                hits += 1
            } else {
                var prepareError: NSError?
                guard A12ANEPrepareQKVModel(
                    selfQ.q, selfQ.biasF32, selfQ.scaleF32,
                    selfK.q, selfK.biasF32, selfK.scaleF32,
                    selfV.q, selfV.biasF32, selfV.scaleF32,
                    UInt(DiTBlockExecutor.dim), UInt(DiTBlockExecutor.tokens),
                    "dit_b\(block)_self_qkv", qkvKey, &prepareError) else {
                    throw prepareError ?? AnimapkError.validation(
                        "failed to prepare ANE fused QKV model for block \(block)")
                }
                misses += 1
                bytesWritten += UInt64(selfQ.payloadBytes + selfK.payloadBytes + selfV.payloadBytes)
            }

            for spec in ANEW8NativePack.projectionSpecs where
                spec.suffix != "self_attn.q_proj.weight" &&
                spec.suffix != "self_attn.k_proj.weight" &&
                spec.suffix != "self_attn.v_proj.weight" {
                let native = try weight(spec.suffix)
                let key = ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: native.blobSHA256)
                if A12ANEPreparedModelExists(key) {
                    hits += 1
                    continue
                }
                var prepareError: NSError?
                guard A12ANEPrepareProjectionModel(
                    native.q, native.biasF32, native.scaleF32,
                    UInt(spec.columns), UInt(spec.rows), UInt(spec.spatial),
                    "dit_b\(block)_\(spec.tag)", key, &prepareError) else {
                    throw prepareError ?? AnimapkError.validation(
                        "failed to prepare ANE projection \(spec.tag) for block \(block)")
                }
                misses += 1
                bytesWritten += UInt64(native.payloadBytes)
            }
        }
        guard hits + misses == ANEW8NativePack.expectedPreparedModelCount else {
            throw AnimapkError.validation(
                "ANE prepared model count \(hits + misses) != \(ANEW8NativePack.expectedPreparedModelCount)")
        }
        let marker = Marker(
            format: ANEW8NativePack.cacheFormat,
            quantScheme: ANEW8NativePack.quantScheme,
            expectedModels: ANEW8NativePack.expectedPreparedModelCount,
            sourceTensorCount: ANEW8NativePack.expectedNativeTensorCount,
            namespace: namespace,
            complete: true)
        try writeMarker(marker, namespace: namespace)
        guard isPrepared(file: file) else {
            throw AnimapkError.validation("ANE model cache failed post-preparation completeness check")
        }
        return ANEW8PreparationResult(
            seconds: ProcessInfo.processInfo.systemUptime - started,
            payloadBytesWritten: bytesWritten,
            cacheHits: hits,
            cacheMisses: misses,
            preparedModels: hits + misses)
    }

    static func expectedCacheKeys(file: AnimapkFile) throws -> [String] {
        try ANEW8NativePack.validate(file: file)
        var keys: [String] = []
        keys.reserveCapacity(ANEW8NativePack.expectedPreparedModelCount)
        for block in 0..<ModelConstants.ditBlocks {
            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
                guard let digest = tensor.blobSHA256 else {
                    throw AnimapkError.validation("ANE-native tensor hash missing")
                }
                return digest
            }
            keys.append(ANEW8NativePack.qkvCacheKey(
                block: block,
                q: try digest("self_attn.q_proj.weight"),
                k: try digest("self_attn.k_proj.weight"),
                v: try digest("self_attn.v_proj.weight")))
            for spec in ANEW8NativePack.projectionSpecs where
                spec.suffix != "self_attn.q_proj.weight" &&
                spec.suffix != "self_attn.k_proj.weight" &&
                spec.suffix != "self_attn.v_proj.weight" {
                keys.append(ANEW8NativePack.projectionCacheKey(
                    block: block, tag: spec.tag, hash: try digest(spec.suffix)))
            }
        }
        return keys
    }

    private static func cacheRoot() -> URL? {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AnimaXS-ANE", isDirectory: true)
    }

    private static func markerURL(namespace: String) -> URL? {
        cacheRoot()?.appendingPathComponent("prepared-\(namespace).json", isDirectory: false)
    }

    private static func readMarker(namespace: String) -> Marker? {
        guard let url = markerURL(namespace: namespace),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Marker.self, from: data)
    }

    private static func writeMarker(_ marker: Marker, namespace: String) throws {
        guard let root = cacheRoot(), let url = markerURL(namespace: namespace) else {
            throw AnimapkError.validation("unable to resolve ANE model cache directory")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(marker)
        try data.write(to: url, options: .atomic)
    }
}

enum ANEW8DiTModelProfile: String, Sendable {
    case full8
    case kvWarm6

    var programCount: Int { self == .full8 ? 8 : 6 }
    var includesCrossKV: Bool { self == .full8 }
}

/// Device-measured A12 scheduler policy. These are production constants, not
/// user-facing tuning knobs: V6 measured depth-3 prefetch and bounded retire1.
/// Production keeps 4 pinned full8 blocks for the first/KV-miss traversal and
/// conservatively 6 pinned six-program blocks once exact P5 K/V is warm after
/// a real generation hit memory pressure at the previous 8-pin setting. The
/// theoretical peaks are 64 full8 and 60 kvWarm6 programs.
enum ANEW8DiTSchedulerPolicy {
    static let prefetchDepth = 3
    static let retireDepth = 1
    static let fullPinnedBlocks = 4
    static let warmPinnedBlocks = 6
    static let measuredSafetyCeilingPrograms = 80

    static func pinnedBlocks(for profile: ANEW8DiTModelProfile) -> Int {
        profile == .full8 ? fullPinnedBlocks : warmPinnedBlocks
    }

    static func theoreticalPeakPrograms(for profile: ANEW8DiTModelProfile) -> Int {
        (pinnedBlocks(for: profile) + prefetchDepth + retireDepth) * profile.programCount
    }
}

struct ANEW8ScheduledModels {
    let models: ANEW8DiTModels
    /// Sum of private `_ANEClient loadModel` time for models that entered the
    /// scheduler while servicing this block. Background-prefetched work is
    /// charged once, when its block first consumes it.
    let newlyLoadedMilliseconds: Double
    /// Foreground wall time spent waiting for scheduler setup / a load future.
    /// Background load/unload work that was fully hidden is intentionally 0.
    let waitMilliseconds: Double
}

/// One block's private-ANE projection set. `crossK`/`crossV` stay non-optional
/// for source compatibility with old diagnostics, but a kvWarm6 set aliases
/// them to `crossQ` and marks `hasCrossKVModels == false`; production guards
/// that such a set is only used on an exact P5 cache hit. Aliased entries are
/// never evaluated or invalidated separately.
final class ANEW8DiTModels: @unchecked Sendable {
    let selfQKV: A12ANEQKVModel
    let selfO: A12ANEProjectionModel
    let crossQ: A12ANEProjectionModel
    let crossK: A12ANEProjectionModel
    let crossV: A12ANEProjectionModel
    let crossO: A12ANEProjectionModel
    let mlpUp: A12ANEProjectionModel
    let mlpDown: A12ANEProjectionModel
    let loadMilliseconds: Double

    private let lock = NSLock()
    private var ownsCrossKV: Bool
    private var invalidated = false

    var hasCrossKVModels: Bool {
        lock.lock(); defer { lock.unlock() }
        return ownsCrossKV && !invalidated
    }

    init(
        selfQKV: A12ANEQKVModel,
        selfO: A12ANEProjectionModel,
        crossQ: A12ANEProjectionModel,
        crossK: A12ANEProjectionModel?,
        crossV: A12ANEProjectionModel?,
        crossO: A12ANEProjectionModel,
        mlpUp: A12ANEProjectionModel,
        mlpDown: A12ANEProjectionModel
    ) {
        self.selfQKV = selfQKV
        self.selfO = selfO
        self.crossQ = crossQ
        self.crossK = crossK ?? crossQ
        self.crossV = crossV ?? crossQ
        self.crossO = crossO
        self.mlpUp = mlpUp
        self.mlpDown = mlpDown
        ownsCrossKV = crossK != nil && crossV != nil
        loadMilliseconds = selfQKV.loadMilliseconds + selfO.loadMilliseconds
            + crossQ.loadMilliseconds
            + (crossK?.loadMilliseconds ?? 0)
            + (crossV?.loadMilliseconds ?? 0)
            + crossO.loadMilliseconds + mlpUp.loadMilliseconds + mlpDown.loadMilliseconds
    }

    /// After a block has populated its exact generation-local P5 K/V cache,
    /// the two invariant projection programs are dead weight. Drop only those
    /// two while retaining the six dynamic programs for the warm pinned prefix.
    func dropCrossKV() {
        lock.lock()
        guard ownsCrossKV, !invalidated else { lock.unlock(); return }
        ownsCrossKV = false
        lock.unlock()
        crossK.invalidate()
        crossV.invalidate()
    }

    func invalidateAll() {
        lock.lock()
        guard !invalidated else { lock.unlock(); return }
        invalidated = true
        let invalidateCrossKV = ownsCrossKV
        ownsCrossKV = false
        lock.unlock()

        selfQKV.invalidate()
        selfO.invalidate()
        crossQ.invalidate()
        if invalidateCrossKV {
            crossK.invalidate()
            crossV.invalidate()
        }
        crossO.invalidate()
        mlpUp.invalidate()
        mlpDown.invalidate()
    }

    deinit { invalidateAll() }
}

/// Stateless loader for already-prepared Espresso programs. Pack validation and
/// prepared-cache completeness are paid exactly once at scheduler construction;
/// every hot-path load below is only model construction + `_ANEClient loadModel`.
private final class ANEW8DiTModelLoader: @unchecked Sendable {
    private let file: AnimapkFile

    init(file: AnimapkFile) throws {
        self.file = file
        try ANEW8NativePack.validate(file: file)
        guard ANEW8ModelPreparer.isPrepared(file: file) else {
            throw AnimapkError.validation(
                "ANE-native model cache is not prepared; prepare the pack before diffusion")
        }
        guard A12ANEIsAvailable() else {
            throw AnimapkError.validation("A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
        }
    }

    func load(block: Int, profile: ANEW8DiTModelProfile) throws -> ANEW8DiTModels {
        guard (0..<ModelConstants.ditBlocks).contains(block) else {
            throw AnimapkError.validation("ANE block index out of range: \(block)")
        }
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
                label: "dit_b\(block)_\(spec.tag)", cacheKey: key)
        }

        let qHash = try digest("self_attn.q_proj.weight")
        let kHash = try digest("self_attn.k_proj.weight")
        let vHash = try digest("self_attn.v_proj.weight")
        let qkvKey = ANEW8NativePack.qkvCacheKey(block: block, q: qHash, k: kHash, v: vHash)
        let selfQKV = try A12ANEQKVModel(
            preparedChannels: UInt(DiTBlockExecutor.dim),
            spatial: UInt(ModelConstants.ditTokensAt512),
            label: "dit_b\(block)_self_qkv", cacheKey: qkvKey)
        let crossQ = try projection("cross_attn.q_proj.weight")
        let crossK = profile.includesCrossKV ? try projection("cross_attn.k_proj.weight") : nil
        let crossV = profile.includesCrossKV ? try projection("cross_attn.v_proj.weight") : nil
        return ANEW8DiTModels(
            selfQKV: selfQKV,
            selfO: try projection("self_attn.output_proj.weight"),
            crossQ: crossQ,
            crossK: crossK,
            crossV: crossV,
            crossO: try projection("cross_attn.output_proj.weight"),
            mlpUp: try projection("mlp.layer1.weight"),
            mlpDown: try projection("mlp.layer2.weight"))
    }
}

private final class ANEW8LoadResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<ANEW8DiTModels, Error>?
    func set(_ result: Result<ANEW8DiTModels, Error>) {
        lock.lock(); value = result; lock.unlock()
    }
    func take() -> Result<ANEW8DiTModels, Error>? {
        lock.lock(); defer { lock.unlock() }
        let result = value
        value = nil
        return result
    }
}

private final class ANEW8LoadFuture: @unchecked Sendable {
    let block: Int
    private let semaphore = DispatchSemaphore(value: 0)
    private let box = ANEW8LoadResultBox()

    init(block: Int, profile: ANEW8DiTModelProfile,
         loader: ANEW8DiTModelLoader, queue: DispatchQueue) {
        self.block = block
        queue.async {
            do { self.box.set(.success(try loader.load(block: block, profile: profile))) }
            catch { self.box.set(.failure(error)) }
            self.semaphore.signal()
        }
    }

    func wait() throws -> ANEW8DiTModels {
        semaphore.wait()
        guard let result = box.take() else {
            throw AnimapkError.validation("ANE scheduler load future returned no result")
        }
        return try result.get()
    }
}

private final class ANEW8RetireFuture: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)

    init(models: ANEW8DiTModels, queue: DispatchQueue) {
        queue.async {
            models.invalidateAll()
            self.semaphore.signal()
        }
    }

    func wait() { semaphore.wait() }
}

/// Generation-lifetime bounded private-ANE scheduler. It recognizes each
/// ordered 0...27 DiT traversal itself, so `DitForward` does not need an ANE-
/// specific loop. First/KV-miss traversals use full8 4+3+retire1. Once the
/// generation-local P5 cache is warm, blocks use six dynamic programs with a
/// conservative 6+3+retire1 policy. The pinned prefix survives across steps.
final class ANEW8DiTModelCache: @unchecked Sendable {
    private let loader: ANEW8DiTModelLoader
    private let loadQueue = DispatchQueue(
        label: "com.invisiblestrangler.AnimaXS.ane-production-loader", qos: .userInitiated)
    private let retireQueue = DispatchQueue(
        label: "com.invisiblestrangler.AnimaXS.ane-production-retire", qos: .userInitiated)

    private var pinned: [Int: ANEW8DiTModels] = [:]
    private var futures: [Int: ANEW8LoadFuture] = [:]
    private var activeStreamed: [Int: ANEW8DiTModels] = [:]
    private var retiring: ANEW8RetireFuture?
    private var traversalProfile: ANEW8DiTModelProfile?
    private var expectedBlock = 0
    private var pendingLoadMilliseconds = 0.0
    private var pendingForegroundWaitMilliseconds = 0.0

    /// Old diagnostic callers still use the historical unbounded API. Keep it
    /// isolated from production scheduler state so V2/V3 research code compiles.
    private var legacyModelsByBlock: [Int: ANEW8DiTModels] = [:]

    init(file: AnimapkFile) throws {
        loader = try ANEW8DiTModelLoader(file: file)
        precondition(
            ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .full8)
                <= ANEW8DiTSchedulerPolicy.measuredSafetyCeilingPrograms)
        precondition(
            ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .kvWarm6)
                <= ANEW8DiTSchedulerPolicy.measuredSafetyCeilingPrograms)
    }

    /// Production API. `kvWarm` must be the actual per-block P5 readiness, not
    /// a diffusion-step number; resumed/partial runs therefore remain correct.
    func scheduledModels(for block: Int, kvWarm: Bool) throws -> ANEW8ScheduledModels {
        let profile: ANEW8DiTModelProfile = kvWarm ? .kvWarm6 : .full8
        guard block == expectedBlock else {
            throw AnimapkError.validation(
                "ANE scheduler expected block \(expectedBlock), got \(block)")
        }
        if block == 0 {
            try beginTraversal(profile: profile)
        } else if traversalProfile != profile {
            throw AnimapkError.validation(
                "ANE scheduler profile changed inside traversal at block \(block)")
        }

        let models: ANEW8DiTModels
        var waitMilliseconds = takePendingForegroundWait()
        var loadedMilliseconds = takePendingLoadMilliseconds()
        let pinnedCount = ANEW8DiTSchedulerPolicy.pinnedBlocks(for: profile)
        if block < pinnedCount {
            guard let value = pinned[block] else {
                throw AnimapkError.validation("ANE scheduler missing pinned block \(block)")
            }
            models = value
        } else {
            guard let future = futures.removeValue(forKey: block) else {
                throw AnimapkError.validation("ANE scheduler missing future for block \(block)")
            }
            let waitStart = ProcessInfo.processInfo.systemUptime
            models = try future.wait()
            waitMilliseconds += elapsedMS(since: waitStart)
            loadedMilliseconds += models.loadMilliseconds
            activeStreamed[block] = models
        }
        expectedBlock += 1
        return ANEW8ScheduledModels(
            models: models,
            newlyLoadedMilliseconds: loadedMilliseconds,
            waitMilliseconds: waitMilliseconds)
    }

    /// Called only after all ANE/Metal work for this block has completed.
    /// Pinned cache-miss blocks immediately shed crossK/crossV once the exact
    /// generation-local P5 entry exists. Streamed blocks use bounded retire1.
    func complete(block: Int, crossKVReady: Bool) throws {
        guard let profile = traversalProfile else {
            throw AnimapkError.validation("ANE scheduler completion outside traversal")
        }
        let pinnedCount = ANEW8DiTSchedulerPolicy.pinnedBlocks(for: profile)
        if block < pinnedCount {
            if crossKVReady { pinned[block]?.dropCrossKV() }
        } else {
            guard let models = activeStreamed.removeValue(forKey: block) else {
                throw AnimapkError.validation("ANE scheduler missing active streamed block \(block)")
            }
            if let previous = retiring { previous.wait() }
            retiring = ANEW8RetireFuture(models: models, queue: retireQueue)

            let newBlock = block + ANEW8DiTSchedulerPolicy.prefetchDepth
            if newBlock < ModelConstants.ditBlocks {
                futures[newBlock] = ANEW8LoadFuture(
                    block: newBlock, profile: profile, loader: loader, queue: loadQueue)
            }
        }

        if block == ModelConstants.ditBlocks - 1 {
            if let finalRetire = retiring { finalRetire.wait() }
            retiring = nil
            guard futures.isEmpty, activeStreamed.isEmpty else {
                throw AnimapkError.validation("ANE scheduler leaked streamed state at traversal end")
            }
            traversalProfile = nil
            expectedBlock = 0
        }
    }

    /// Failure/cancellation cleanup. This is intentionally stronger than the
    /// happy-path traversal end: all pinned state is released because the
    /// generation is no longer useful and must not leave private residency.
    func abortTraversal() {
        if let retiring { retiring.wait() }
        retiring = nil
        for (_, future) in futures {
            if let models = try? future.wait() { models.invalidateAll() }
        }
        futures.removeAll()
        for (_, models) in activeStreamed { models.invalidateAll() }
        activeStreamed.removeAll()
        for (_, models) in pinned { models.invalidateAll() }
        pinned.removeAll()
        traversalProfile = nil
        expectedBlock = 0
        pendingLoadMilliseconds = 0
        pendingForegroundWaitMilliseconds = 0
    }

    /// Historical diagnostic API: unbounded, no scheduler. Production must use
    /// `scheduledModels(for:kvWarm:)` so residency stays bounded.
    func models(for block: Int) throws -> (models: ANEW8DiTModels, newlyLoadedMilliseconds: Double) {
        if let cached = legacyModelsByBlock[block] { return (cached, 0) }
        let models = try loader.load(block: block, profile: .full8)
        legacyModelsByBlock[block] = models
        return (models, models.loadMilliseconds)
    }

    deinit {
        abortTraversal()
        for (_, models) in legacyModelsByBlock { models.invalidateAll() }
        legacyModelsByBlock.removeAll()
    }

    private func beginTraversal(profile: ANEW8DiTModelProfile) throws {
        guard traversalProfile == nil, futures.isEmpty, activeStreamed.isEmpty else {
            throw AnimapkError.validation("ANE scheduler began traversal with residual streamed state")
        }
        traversalProfile = profile
        expectedBlock = 0

        let setupStart = ProcessInfo.processInfo.systemUptime
        var setupLoadMilliseconds = 0.0
        let targetPinned = ANEW8DiTSchedulerPolicy.pinnedBlocks(for: profile)

        // Exact P5 transition: preserve the six useful programs in the first
        // four pins instead of throwing them away and reloading them.
        if profile == .kvWarm6 {
            for block in 0..<min(ANEW8DiTSchedulerPolicy.fullPinnedBlocks, targetPinned) {
                pinned[block]?.dropCrossKV()
            }
        }

        // A cache-allocation failure keeps every traversal in full8. If the
        // profile is warm, expand the retained prefix from 4 -> 6 using six-
        // program loads. Any incompatible stale pin is rebuilt defensively.
        for block in 0..<targetPinned {
            if let existing = pinned[block] {
                if profile == .full8 && !existing.hasCrossKVModels {
                    existing.invalidateAll()
                    pinned.removeValue(forKey: block)
                } else {
                    continue
                }
            }
            let models = try loader.load(block: block, profile: profile)
            pinned[block] = models
            setupLoadMilliseconds += models.loadMilliseconds
        }

        // If a future policy ever reduces the pinned prefix, retire surplus
        // pins now. Current production transition grows 4 -> 6.
        let surplus = pinned.keys.filter { $0 >= targetPinned }
        for block in surplus {
            pinned.removeValue(forKey: block)?.invalidateAll()
        }

        if targetPinned < ModelConstants.ditBlocks {
            for block in targetPinned..<min(
                ModelConstants.ditBlocks,
                targetPinned + ANEW8DiTSchedulerPolicy.prefetchDepth) {
                futures[block] = ANEW8LoadFuture(
                    block: block, profile: profile, loader: loader, queue: loadQueue)
            }
        }
        pendingLoadMilliseconds += setupLoadMilliseconds
        pendingForegroundWaitMilliseconds += elapsedMS(since: setupStart)
    }

    private func takePendingLoadMilliseconds() -> Double {
        let value = pendingLoadMilliseconds
        pendingLoadMilliseconds = 0
        return value
    }

    private func takePendingForegroundWait() -> Double {
        let value = pendingForegroundWaitMilliseconds
        pendingForegroundWaitMilliseconds = 0
        return value
    }

    private func elapsedMS(since start: Double) -> Double {
        (ProcessInfo.processInfo.systemUptime - start) * 1_000
    }
}

/// Reused shared IOSurfaces for one block execution. The large hidden surface
/// is also the Metal GELU buffer. At spatial=1024 its H11 plane stride is tight,
/// so elementwise activation can run in-place without a layout conversion.
final class ANEW8DiTSurfaces {
    let tokenInput: A12ANESurface
    let q: A12ANESurface
    let k: A12ANESurface
    let v: A12ANESurface
    let tokenOutput: A12ANESurface
    let contextInput: A12ANESurface
    let contextK: A12ANESurface
    let contextV: A12ANESurface
    let hidden: A12ANESurface

    init(device: MTLDevice) throws {
        tokenInput = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(ModelConstants.ditTokensAt512))
        q = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(ModelConstants.ditTokensAt512))
        k = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(ModelConstants.ditTokensAt512))
        v = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(ModelConstants.ditTokensAt512))
        tokenOutput = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(ModelConstants.ditTokensAt512))
        contextInput = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.contextDim), spatial: UInt(DiTBlockExecutor.contextTokens))
        contextK = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.contextTokens))
        contextV = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.contextTokens))
        hidden = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.hidden), spatial: UInt(ModelConstants.ditTokensAt512))
    }
}
