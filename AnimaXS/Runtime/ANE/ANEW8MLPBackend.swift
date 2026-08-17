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
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "self_q"),
        .init(suffix: "self_attn.k_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "self_k"),
        .init(suffix: "self_attn.v_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "self_v"),
        .init(suffix: "self_attn.output_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "self_o"),
        .init(suffix: "cross_attn.q_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "cross_q"),
        .init(suffix: "cross_attn.k_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.contextDim, spatial: DiTBlockExecutor.contextTokens, tag: "cross_k"),
        .init(suffix: "cross_attn.v_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.contextDim, spatial: DiTBlockExecutor.contextTokens, tag: "cross_v"),
        .init(suffix: "cross_attn.output_proj.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "cross_o"),
        .init(suffix: "mlp.layer1.weight", rows: DiTBlockExecutor.hidden,
              columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "mlp1"),
        .init(suffix: "mlp.layer2.weight", rows: DiTBlockExecutor.dim,
              columns: DiTBlockExecutor.hidden, spatial: DiTBlockExecutor.tokens, tag: "mlp2"),
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
                _ = try A12ANEPrepareProjectionModel(
                    native.q, native.biasF32, native.scaleF32,
                    UInt(spec.columns), UInt(spec.rows), UInt(spec.spatial),
                    "dit_b\(block)_\(spec.tag)", key)
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

final class ANEW8DiTModels {
    let selfQKV: A12ANEQKVModel
    let selfO: A12ANEProjectionModel
    let crossQ: A12ANEProjectionModel
    let crossK: A12ANEProjectionModel
    let crossV: A12ANEProjectionModel
    let crossO: A12ANEProjectionModel
    let mlpUp: A12ANEProjectionModel
    let mlpDown: A12ANEProjectionModel
    let loadMilliseconds: Double

    init(
        selfQKV: A12ANEQKVModel,
        selfO: A12ANEProjectionModel,
        crossQ: A12ANEProjectionModel,
        crossK: A12ANEProjectionModel,
        crossV: A12ANEProjectionModel,
        crossO: A12ANEProjectionModel,
        mlpUp: A12ANEProjectionModel,
        mlpDown: A12ANEProjectionModel
    ) {
        self.selfQKV = selfQKV
        self.selfO = selfO
        self.crossQ = crossQ
        self.crossK = crossK
        self.crossV = crossV
        self.crossO = crossO
        self.mlpUp = mlpUp
        self.mlpDown = mlpDown
        loadMilliseconds = selfQKV.loadMilliseconds + selfO.loadMilliseconds
            + crossQ.loadMilliseconds + crossK.loadMilliseconds
            + crossV.loadMilliseconds + crossO.loadMilliseconds
            + mlpUp.loadMilliseconds + mlpDown.loadMilliseconds
    }
}

/// Generation-lifetime loaded-model cache. Prepared Espresso files must already
/// exist; first use performs only `_ANEModel` construction + `_ANEClient` load.
final class ANEW8DiTModelCache {
    private let file: AnimapkFile
    private var modelsByBlock: [Int: ANEW8DiTModels] = [:]

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

    func models(for block: Int) throws -> (models: ANEW8DiTModels, newlyLoadedMilliseconds: Double) {
        if let cached = modelsByBlock[block] { return (cached, 0) }
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
            spatial: UInt(DiTBlockExecutor.tokens),
            label: "dit_b\(block)_self_qkv", cacheKey: qkvKey)

        let models = ANEW8DiTModels(
            selfQKV: selfQKV,
            selfO: try projection("self_attn.output_proj.weight"),
            crossQ: try projection("cross_attn.q_proj.weight"),
            crossK: try projection("cross_attn.k_proj.weight"),
            crossV: try projection("cross_attn.v_proj.weight"),
            crossO: try projection("cross_attn.output_proj.weight"),
            mlpUp: try projection("mlp.layer1.weight"),
            mlpDown: try projection("mlp.layer2.weight"))
        modelsByBlock[block] = models
        return (models, models.loadMilliseconds)
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
        tokenInput = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.tokens))
        q = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.tokens))
        k = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.tokens))
        v = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.tokens))
        tokenOutput = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.tokens))
        contextInput = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.contextDim), spatial: UInt(DiTBlockExecutor.contextTokens))
        contextK = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.contextTokens))
        contextV = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.contextTokens))
        hidden = try A12ANESurface(device: device, channels: UInt(DiTBlockExecutor.hidden), spatial: UInt(DiTBlockExecutor.tokens))
    }
}
