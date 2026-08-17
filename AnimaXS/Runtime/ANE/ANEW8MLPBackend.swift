import Foundation
import Metal

/// A12/H11 ANE W8 execution is deliberately isolated from the legacy Metal
/// linear executor. The hybrid backend moves the large DiT projection GEMMs to
/// ANE while keeping AdaLN, learned RMSNorm, RoPE, attention, GELU and residual
/// math on the existing Metal implementation.
///
/// ANE consumes channel-major H11 surfaces while AnimaXS uses tight token-major
/// activations. `DiTBlockExecutor` bridges those layouts with small Metal
/// kernels over IOSurface-backed buffers shared by both processors; activations
/// are never copied through CPU memory.
struct ANEW8RepackedWeight {
    let q: Data
    let biasF32: Data
    let scaleF32: Data
    let fingerprint: String
}

enum ANEW8Repacker {
    /// Converts AnimaXS groupwise U8 affine W8
    ///     w = q * fp16(groupScale) + fp16(groupZero)
    /// into the H11 Espresso form proven by AnimaANEProbe v14:
    ///     w ~= qRow * fp32(rowScale) + fp32(rowBias)
    /// with one scale+bias per output row.
    static func repack(
        file: AnimapkFile,
        tensor: AnimapkTensor,
        expectedRows: Int,
        expectedColumns: Int
    ) throws -> ANEW8RepackedWeight {
        guard tensor.storage == .w8 else {
            throw AnimapkError.validation("ANE backend requires W8 tensor: \(tensor.name) is \(tensor.storageDtype)")
        }
        guard tensor.shape == [expectedRows, expectedColumns] else {
            throw AnimapkError.validation(
                "ANE tensor shape \(tensor.name) \(tensor.shape) != [\(expectedRows), \(expectedColumns)]")
        }
        guard let group = file.quantGroup, group > 0 else {
            throw AnimapkError.validation("ANE W8 backend requires a valid quantization group")
        }
        let packed = file.dataBytes(tensor)
        guard let scales = file.scaleBytes(tensor), let zeros = file.zeroBytes(tensor) else {
            throw AnimapkError.validation("ANE W8 tensor is missing scale/zero payload: \(tensor.name)")
        }
        let groupsPerRow = (expectedColumns + group - 1) / group
        guard packed.count >= expectedRows * expectedColumns,
              scales.count >= expectedRows * groupsPerRow * MemoryLayout<UInt16>.stride,
              zeros.count >= expectedRows * groupsPerRow * MemoryLayout<UInt16>.stride else {
            throw AnimapkError.validation("ANE W8 tensor payload is truncated: \(tensor.name)")
        }

        var q = Data(count: expectedRows * expectedColumns)
        var bias = Data(count: expectedRows * MemoryLayout<Float>.stride)
        var scale = Data(count: expectedRows * MemoryLayout<Float>.stride)

        let sourceQ = packed.bindMemory(to: UInt8.self)
        let sourceScale = scales.bindMemory(to: UInt16.self)
        let sourceZero = zeros.bindMemory(to: UInt16.self)
        q.withUnsafeMutableBytes { qRaw in
            bias.withUnsafeMutableBytes { biasRaw in
                scale.withUnsafeMutableBytes { scaleRaw in
                    let outputQ = qRaw.bindMemory(to: UInt8.self).baseAddress!
                    let outputBias = biasRaw.bindMemory(to: Float.self).baseAddress!
                    let outputScale = scaleRaw.bindMemory(to: Float.self).baseAddress!

                    for row in 0..<expectedRows {
                        let rowBase = row * expectedColumns
                        let groupBase = row * groupsPerRow
                        var low = Float.greatestFiniteMagnitude
                        var high = -Float.greatestFiniteMagnitude

                        for column in 0..<expectedColumns {
                            let groupIndex = groupBase + column / group
                            let s = Float(Float16(bitPattern: sourceScale[groupIndex]))
                            let z = Float(Float16(bitPattern: sourceZero[groupIndex]))
                            let value = Float(sourceQ[rowBase + column]) * s + z
                            guard value.isFinite else { continue }
                            low = min(low, value)
                            high = max(high, value)
                        }

                        guard low.isFinite, high.isFinite else {
                            throwRepackSentinel(outputBias: outputBias, outputScale: outputScale, row: row)
                            continue
                        }
                        let rowScale = high > low ? (high - low) / 255.0 : 1.0
                        let rowBias = low
                        outputBias[row] = rowBias
                        outputScale[row] = rowScale

                        for column in 0..<expectedColumns {
                            let groupIndex = groupBase + column / group
                            let s = Float(Float16(bitPattern: sourceScale[groupIndex]))
                            let z = Float(Float16(bitPattern: sourceZero[groupIndex]))
                            let value = Float(sourceQ[rowBase + column]) * s + z
                            if !value.isFinite {
                                outputQ[rowBase + column] = 0
                                continue
                            }
                            let quantized = rowScale > 0 ? ((value - rowBias) / rowScale).rounded() : 0
                            let clamped = min(255, max(0, Int(quantized.isFinite ? quantized : 0)))
                            outputQ[rowBase + column] = UInt8(clamped)
                        }
                    }
                }
            }
        }

        var hash = FNV1a64()
        hash.update(tensor.name.utf8)
        q.withUnsafeBytes { hash.update($0) }
        bias.withUnsafeBytes { hash.update($0) }
        scale.withUnsafeBytes { hash.update($0) }
        return ANEW8RepackedWeight(
            q: q, biasF32: bias, scaleF32: scale,
            fingerprint: String(format: "%016llx", hash.value))
    }

    /// Keeps the nested raw-buffer closure non-throwing while still making an
    /// impossible all-nonfinite source row deterministic. The source W8 packs
    /// are validated elsewhere; this path should never be hit for a real pack.
    @inline(__always)
    private static func throwRepackSentinel(
        outputBias: UnsafeMutablePointer<Float>,
        outputScale: UnsafeMutablePointer<Float>,
        row: Int
    ) {
        outputBias[row] = 0
        outputScale[row] = 1
    }
}

private struct FNV1a64 {
    private(set) var value: UInt64 = 0xcbf29ce484222325
    mutating func update<S: Sequence>(_ bytes: S) where S.Element == UInt8 {
        for byte in bytes {
            value ^= UInt64(byte)
            value &*= 0x100000001b3
        }
    }
    mutating func update(_ bytes: UnsafeRawBufferPointer) {
        for byte in bytes { update(CollectionOfOne(byte)) }
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

/// Generation-lifetime model cache. The device probe established that many H11
/// programs can stay loaded concurrently without retaining their source W8
/// payload in this process, so every block model is loaded once on first use and
/// reused by all subsequent diffusion steps.
final class ANEW8DiTModelCache {
    private let file: AnimapkFile
    private var modelsByBlock: [Int: ANEW8DiTModels] = [:]

    init(file: AnimapkFile) throws {
        self.file = file
        self.modelsByBlock = [:]
        guard file.component == "dit" else {
            throw AnimapkError.validation("ANE hybrid backend requires a DiT pack")
        }
        guard file.quantGroup != nil else {
            throw AnimapkError.validation("ANE hybrid backend requires a quantized DiT pack")
        }
        let required: [(String, [Int])] = [
            ("self_attn.q_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.dim]),
            ("self_attn.k_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.dim]),
            ("self_attn.v_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.dim]),
            ("self_attn.output_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.dim]),
            ("cross_attn.q_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.dim]),
            ("cross_attn.k_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.contextDim]),
            ("cross_attn.v_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.contextDim]),
            ("cross_attn.output_proj.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.dim]),
            ("mlp.layer1.weight", [DiTBlockExecutor.hidden, DiTBlockExecutor.dim]),
            ("mlp.layer2.weight", [DiTBlockExecutor.dim, DiTBlockExecutor.hidden]),
        ]
        // Reject W4/mixed/malformed packs before the first diffusion block.
        for block in 0..<ModelConstants.ditBlocks {
            let prefix = "model.diffusion_model.blocks.\(block)."
            for (suffix, expectedShape) in required {
                let name = prefix + suffix
                guard let tensor = file.tensor(named: name) else {
                    throw AnimapkError.validation("ANE hybrid backend missing tensor: \(name)")
                }
                guard tensor.storage == .w8 else {
                    throw AnimapkError.validation(
                        "ANE hybrid backend is W8-only; \(name) uses \(tensor.storageDtype)")
                }
                guard tensor.shape == expectedShape else {
                    throw AnimapkError.validation(
                        "ANE hybrid backend shape mismatch for \(name): \(tensor.shape) != \(expectedShape)")
                }
            }
        }
        guard A12ANEIsAvailable() else {
            throw AnimapkError.validation("A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
        }
    }

    func models(for block: Int) throws -> (models: ANEW8DiTModels, newlyLoadedMilliseconds: Double) {
        if let cached = modelsByBlock[block] { return (cached, 0) }
        let prefix = "model.diffusion_model.blocks.\(block)."
        func repack(_ suffix: String, rows: Int, columns: Int) throws -> ANEW8RepackedWeight {
            let name = prefix + suffix
            guard let tensor = file.tensor(named: name) else {
                throw AnimapkError.validation("ANE hybrid weight missing: \(name)")
            }
            return try ANEW8Repacker.repack(
                file: file, tensor: tensor, expectedRows: rows, expectedColumns: columns)
        }
        func projection(
            _ suffix: String, rows: Int, columns: Int, spatial: Int, tag: String
        ) throws -> A12ANEProjectionModel {
            let weight = try repack(suffix, rows: rows, columns: columns)
            return try A12ANEProjectionModel(
                qBytes: weight.q, biasF32: weight.biasF32, scaleF32: weight.scaleF32,
                inputChannels: UInt(columns), outputChannels: UInt(rows), spatial: UInt(spatial),
                label: "dit_b\(block)_\(tag)",
                cacheKey: "b\(block)-\(tag)-\(weight.fingerprint)")
        }

        let selfQ = try repack("self_attn.q_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.dim)
        let selfK = try repack("self_attn.k_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.dim)
        let selfV = try repack("self_attn.v_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.dim)
        let qkvFingerprint = [selfQ.fingerprint, selfK.fingerprint, selfV.fingerprint].joined(separator: "-")
        let selfQKV = try A12ANEQKVModel(
            qBytes: selfQ.q, qBiasF32: selfQ.biasF32, qScaleF32: selfQ.scaleF32,
            kBytes: selfK.q, kBiasF32: selfK.biasF32, kScaleF32: selfK.scaleF32,
            vBytes: selfV.q, vBiasF32: selfV.biasF32, vScaleF32: selfV.scaleF32,
            channels: UInt(DiTBlockExecutor.dim), spatial: UInt(DiTBlockExecutor.tokens),
            label: "dit_b\(block)_self_qkv", cacheKey: "b\(block)-selfqkv-\(qkvFingerprint)")

        let models = ANEW8DiTModels(
            selfQKV: selfQKV,
            selfO: try projection("self_attn.output_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "self_o"),
            crossQ: try projection("cross_attn.q_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "cross_q"),
            crossK: try projection("cross_attn.k_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.contextDim, spatial: DiTBlockExecutor.contextTokens, tag: "cross_k"),
            crossV: try projection("cross_attn.v_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.contextDim, spatial: DiTBlockExecutor.contextTokens, tag: "cross_v"),
            crossO: try projection("cross_attn.output_proj.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "cross_o"),
            mlpUp: try projection("mlp.layer1.weight", rows: DiTBlockExecutor.hidden, columns: DiTBlockExecutor.dim, spatial: DiTBlockExecutor.tokens, tag: "mlp1"),
            mlpDown: try projection("mlp.layer2.weight", rows: DiTBlockExecutor.dim, columns: DiTBlockExecutor.hidden, spatial: DiTBlockExecutor.tokens, tag: "mlp2"))
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
