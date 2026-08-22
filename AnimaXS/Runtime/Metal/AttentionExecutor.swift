import Foundation
import Metal
import MetalPerformanceShaders

/// Immutable NegPiP V signs scoped to one diffusion task. Task-local state is
/// important here: Diagnostics or tests running concurrently must never observe
/// the creative generation's negative prompt.
struct NegPiPValueMaskSnapshot: Sendable {
    let lease: UInt64
    let signs: [Float]
}

enum NegPiPGenerationContext {
    @TaskLocal static var active: NegPiPValueMaskSnapshot?

    static func make(signs: [Float]?) throws -> NegPiPValueMaskSnapshot? {
        guard let signs else { return nil }
        guard signs.count == LLMAdapterMetal.maximumTokens,
              signs.allSatisfy({ $0 == 1 || $0 == -1 }) else {
            throw AnimapkError.validation("NegPiP sign mask must contain exactly 512 values of +1/-1")
        }
        // A compiler result containing only +1 is equivalent to baseline and
        // must not allocate/encode any NegPiP GPU work.
        guard signs.contains(-1) else { return nil }
        return NegPiPValueMaskSnapshot(
            lease: UInt64.random(in: 1...UInt64.max), signs: signs)
    }

    static func snapshot() -> NegPiPValueMaskSnapshot? { active }
}

enum AttentionNumerics: String, CaseIterable {
    case legacy
    case fp32ScoresAndSoftmax
    /// Emulate a BF16 scaled-dot-product attention on Apple5, which has no
    /// native BF16 storage. Q/K/V, scores, probabilities, and the PV result
    /// are rounded to BF16 while the buffers remain fp16 for MPS.
    case bf16Compute = "bf16_compute"
}

/// Diagnostic activation-boundary policy. `bf16Boundaries` stores BF16-rounded
/// values in fp32 buffers because Apple5 GPUs do not expose native bfloat16.
enum ActivationNumerics: String, CaseIterable {
    case legacy
    case bf16Boundaries = "bf16_boundaries"
    /// Full BF16 model arithmetic emulation. Unlike `bf16Boundaries`, this
    /// rounds every compute boundary while retaining the model residual in
    /// its selected dtype (BF16), rather than rounding only residual adds.
    case bf16Compute = "bf16_compute"
}

/// P7: runtime-selectable DiT attention backends (runbook §12). All of them
/// are exact (no approximation) implementations of scaled dot-product
/// attention; they differ in HOW the scores/softmax/PV are computed. The
/// physical device selects the winner later — the default keeps the
/// known-good W4 behavior (see `InferenceOptimizationConfig`).
enum DiTAttentionBackend: String, Codable, CaseIterable {
    /// Legacy head-major MPS path (per-head transpose layout, full score
    /// tile). The P0-P6 known-good behavior for every attention user.
    case legacyHeadMajorMPS
    /// P4: strided token-major MPS path (strided per-head MPSMatrix views,
    /// no token↔head transposes, full score tile).
    case stridedTokenMajorMPS
    /// P7-A: streaming/online-softmax MPS — MPS QK/PV per KEY CHUNK with a
    /// running FP32 max/sum and FP32 output accumulator, so no full
    /// `[queryTile, keyCount]` score tile is ever live.
    case streamingMPS
    /// P7-B: DiT-specialized pure-Metal Flash-style online attention
    /// (headDim 128, heads 16, token-major, FP32 accumulation, simd_sum
    /// score dots, running-max/rescale mandatory). Rejects unsupported
    /// shapes loudly.
    case metalFlash
}

/// Input/output buffer layout contract for the attention executor.
///
/// - `.headMajor`: legacy `[heads, rows, headDim]` tightly packed buffers —
///   the ONLY layout Qwen/VAE/adapter attention uses, and the default.
/// - `.tokenMajor(tokenStride:)`: `[rows, tokenStride]` buffers where each
///   head occupies `headDim` CONTIGUOUS columns within every row (DiT:
///   modelDim 2048, headDim 128, heads 16). P4: MPS consumes strided
///   per-head matrix views of these buffers directly, so no token↔head
///   transpose kernels are needed. `tokenStride` is in HALF elements
///   (2048 for DiT).
enum AttentionInputLayout: Equatable {
    case headMajor
    case tokenMajor(tokenStride: Int)

    var isTokenMajor: Bool {
        if case .tokenMajor = self { return true }
        return false
    }
}

/// Query-tiled scaled dot-product attention over tightly packed fp16 tensors.
/// Layout is `[heads, rows, headDim]`; only one `[tileRows,keyCount]` score tile
/// is retained. The executor is intentionally non-reentrant because it reuses scratch.
///
/// P4: `layout == .tokenMajor` builds STRIDED per-head `MPSMatrix` views of
/// token-major `[rows, tokenStride]` Q/K/V/output buffers (each head reads
/// only its `headDim` contiguous columns of every row), eliminating the
/// token↔head transpose kernels. The legacy head-major path is byte-for-byte
/// unchanged and remains the default for every other attention user.
final class AttentionExecutor {
    static let defaultTileRows = 128

    private let context: MetalContext
    private let buffers: BufferPool
    private let monitor: NumericalMonitor?
    let tileRows: Int
    let numerics: AttentionNumerics
    /// P4: input/output layout contract. Defaults to the legacy head-major
    /// layout so Qwen/VAE/adapter attention is untouched.
    let layout: AttentionInputLayout
    /// P7: DiT attention backend selector. Only consulted for token-major
    /// DiT attention; Qwen/VAE/adapter (head-major) always run the legacy
    /// MPS path. `.stridedTokenMajorMPS` maps to the P4 strided path and
    /// honors the `stridedTokenMajorAttention` boolean at the caller.
    let attentionBackend: DiTAttentionBackend
    /// Run telemetry collector (nil in tests / diagnostic-only construction).
    /// Receives the cheap query-tile counter (simple integer increment).
    var metrics: MetricsCollector?

    /// Lazily created so a baseline/empty-negative generation does not even
    /// instantiate the NegPiP image kernel. The mask itself is expanded once
    /// per generation and reused; only the cheap V multiply runs on a cache miss.
    private lazy var negPiPMultiply = MPSImageMultiply(device: context.device)
    private var negPiPMaskGeneration: UInt64?
    private var negPiPMaskLayoutKey: String?

    init(context: MetalContext, tileRows: Int = defaultTileRows,
         numerics: AttentionNumerics = .legacy,
         monitor: NumericalMonitor? = nil,
         layout: AttentionInputLayout = .headMajor,
         attentionBackend: DiTAttentionBackend = .legacyHeadMajorMPS) {
        precondition(tileRows > 0)
        self.context = context
        self.buffers = BufferPool(device: context.device)
        self.monitor = monitor
        self.tileRows = tileRows
        self.numerics = numerics
        self.layout = layout
        self.attentionBackend = attentionBackend
    }

    func maximumScoreScratchBytes(keyCount: Int, queryCount: Int? = nil) throws -> Int {
        guard keyCount > 0 else { throw AnimapkError.validation("attention key count must be positive") }
        let effectiveTile = queryCount.map { min(tileRows, $0) } ?? tileRows
        let elementBytes = numerics == .legacy
            ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        return try checkedProduct(effectiveTile, keyCount, elementBytes)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int = 0,
        key: MTLBuffer, keyOffset: Int = 0,
        value: MTLBuffer, valueOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        keyValueHeads: Int? = nil,
        causal: Bool = false,
        probe: NumericalMonitor.Probe? = nil,
        layout: AttentionInputLayout? = nil,
        negPiPValueAlreadyMasked: Bool = false
    ) throws {
        let kvHeads = keyValueHeads ?? heads
        let inputLayout = layout ?? self.layout
        let tokenStride: Int? = {
            if case .tokenMajor(let stride) = inputLayout { return stride }
            return nil
        }()
        try validate(query: query, queryOffset: queryOffset, key: key, keyOffset: keyOffset,
                     value: value, valueOffset: valueOffset, output: output,
                     outputOffset: outputOffset, heads: heads, queryCount: queryCount,
                     keyCount: keyCount, headDim: headDim, keyValueHeads: kvHeads,
                     causal: causal, layout: inputLayout)

        // Direct callers/tests still get the V-only NegPiP boundary here. The
        // production DiT path applies the same primitive before its cross-K/V
        // cache store, then passes `negPiPValueAlreadyMasked` so cache hits are
        // never flipped a second time. K is never touched.
        let effectiveValue: (buffer: MTLBuffer, offset: Int)
        if negPiPValueAlreadyMasked {
            effectiveValue = (value, valueOffset)
        } else {
            effectiveValue = try negPiPValueIfNeeded(
                commandBuffer: commandBuffer,
                value: value, valueOffset: valueOffset,
                heads: heads, keyCount: keyCount, headDim: headDim,
                kvHeads: kvHeads, layout: inputLayout, probe: probe)
        }

        let halfBytes = MemoryLayout<Float16>.stride
        let scoreScratch = buffers.buffer(
            key: "attention.scores.fp16", bytes: try maximumScoreScratchBytes(keyCount: keyCount, queryCount: queryCount))
        let softmaxName = (monitor != nil && probe != nil)
            ? "attention_softmax_rows_probe" : "attention_softmax_rows"
        let softmax = try context.pipeline(named: softmaxName)
        let headRowBytes = headDim * halfBytes
        let scoreRowBytes = keyCount * halfBytes
        let scale = 1 / sqrt(Double(headDim))

        if numerics == .fp32ScoresAndSoftmax {
            if inputLayout.isTokenMajor {
                throw AnimapkError.validation(
                    "P4 strided token-major attention does not support fp32ScoresAndSoftmax; select baseline numerics")
            }
            try encodeFP32(commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                           key: key, keyOffset: keyOffset,
                           value: effectiveValue.buffer, valueOffset: effectiveValue.offset,
                           output: output, outputOffset: outputOffset, heads: heads,
                           queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                           kvHeads: kvHeads, causal: causal, probe: probe)
            return
        }

        if inputLayout.isTokenMajor {
            guard numerics != .bf16Compute else {
                throw AnimapkError.validation(
                    "P4 strided token-major attention does not support bf16Compute numerics; select baseline numerics")
            }
            guard let tokenStride else {
                throw AnimapkError.validation(
                    "P4 token-major attention requires a token stride")
            }
            switch attentionBackend {
            case .streamingMPS:
                try encodeStreamingTokenMajor(
                    commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                    key: key, keyOffset: keyOffset,
                    value: effectiveValue.buffer, valueOffset: effectiveValue.offset,
                    output: output, outputOffset: outputOffset,
                    heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                    kvHeads: kvHeads, tokenStride: tokenStride, causal: causal, probe: probe,
                    halfBytes: halfBytes, scale: scale)
                return
            case .metalFlash:
                try encodeMetalFlash(
                    commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                    key: key, keyOffset: keyOffset,
                    value: effectiveValue.buffer, valueOffset: effectiveValue.offset,
                    output: output, outputOffset: outputOffset,
                    heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                    kvHeads: kvHeads, tokenStride: tokenStride, causal: causal, probe: probe,
                    scoreScratch: scoreScratch, softmax: softmax, halfBytes: halfBytes,
                    scoreRowBytes: scoreRowBytes, scale: scale)
                return
            case .legacyHeadMajorMPS, .stridedTokenMajorMPS:
                break
            }
            try encodeTokenMajor(
                commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                key: key, keyOffset: keyOffset,
                value: effectiveValue.buffer, valueOffset: effectiveValue.offset,
                output: output, outputOffset: outputOffset,
                heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                kvHeads: kvHeads, tokenStride: tokenStride, causal: causal, probe: probe,
                scoreScratch: scoreScratch, softmax: softmax, halfBytes: halfBytes,
                scoreRowBytes: scoreRowBytes, scale: scale)
            return
        }

        for head in 0..<heads {
            let kvHead = head / (heads / kvHeads)
            let keyMatrix = MPSMatrix(
                buffer: key, offset: keyOffset + kvHead * keyCount * headRowBytes,
                descriptor: MPSMatrixDescriptor(rows: keyCount, columns: headDim,
                                                rowBytes: headRowBytes, dataType: .float16))
            let valueMatrix = MPSMatrix(
                buffer: effectiveValue.buffer,
                offset: effectiveValue.offset + kvHead * keyCount * headRowBytes,
                descriptor: MPSMatrixDescriptor(rows: keyCount, columns: headDim,
                                                rowBytes: headRowBytes, dataType: .float16))
            var queryBase = 0
            while queryBase < queryCount {
                let rows = min(tileRows, queryCount - queryBase)
                let queryMatrix = MPSMatrix(
                    buffer: query,
                    offset: queryOffset + (head * queryCount + queryBase) * headRowBytes,
                    descriptor: MPSMatrixDescriptor(rows: rows, columns: headDim,
                                                    rowBytes: headRowBytes, dataType: .float16))
                let scoreMatrix = MPSMatrix(
                    buffer: scoreScratch,
                    descriptor: MPSMatrixDescriptor(rows: rows, columns: keyCount,
                                                    rowBytes: scoreRowBytes, dataType: .float16))
                MPSMatrixMultiplication(
                    device: context.device, transposeLeft: false, transposeRight: true,
                    resultRows: rows, resultColumns: keyCount, interiorColumns: headDim,
                    alpha: scale, beta: 0).encode(
                        commandBuffer: commandBuffer, leftMatrix: queryMatrix,
                        rightMatrix: keyMatrix, resultMatrix: scoreMatrix)

                if numerics == .bf16Compute {
                    try encodeRoundHalf(commandBuffer, input: scoreScratch,
                                        output: scoreScratch, count: rows * keyCount)
                }

                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create attention softmax encoder")
                }
                var rowCount = UInt32(rows), columns = UInt32(keyCount)
                var base = UInt32(queryBase), causalFlag = causal ? UInt32(1) : UInt32(0)
                encoder.setComputePipelineState(softmax)
                encoder.setBuffer(scoreScratch, offset: 0, index: 0)
                encoder.setBytes(&rowCount, length: 4, index: 1)
                encoder.setBytes(&columns, length: 4, index: 2)
                encoder.setBytes(&base, length: 4, index: 3)
                encoder.setBytes(&causalFlag, length: 4, index: 4)
                if let monitor, let probe {
                    monitor.bindProbe(encoder, probe: probe, statsIndex: 5, slotIndex: 6)
                }
                let threads = reductionThreads(limit: softmax.maxTotalThreadsPerThreadgroup)
                encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                encoder.endEncoding()

                if numerics == .bf16Compute {
                    try encodeRoundHalf(commandBuffer, input: scoreScratch,
                                        output: scoreScratch, count: rows * keyCount)
                }

                let outputMatrix = MPSMatrix(
                    buffer: output,
                    offset: outputOffset + (head * queryCount + queryBase) * headRowBytes,
                    descriptor: MPSMatrixDescriptor(rows: rows, columns: headDim,
                                                    rowBytes: headRowBytes, dataType: .float16))
                MPSMatrixMultiplication(
                    device: context.device, transposeLeft: false, transposeRight: false,
                    resultRows: rows, resultColumns: headDim, interiorColumns: keyCount,
                    alpha: 1, beta: 0).encode(
                        commandBuffer: commandBuffer, leftMatrix: scoreMatrix,
                        rightMatrix: valueMatrix, resultMatrix: outputMatrix)
                if numerics == .bf16Compute {
                    try encodeRoundHalf(commandBuffer, input: output,
                                        output: output, count: rows * headDim,
                                        offset: outputOffset + (head * queryCount + queryBase) * headRowBytes)
                }
                metrics?.recordConversionBytes(UInt64(rows * keyCount * halfBytes))
                metrics?.recordConversionBytes(UInt64(rows * headDim * halfBytes))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// Returns the original V buffer when NegPiP is inactive. Production DiT
    /// calls this on a cross-K/V cache miss after base projection + LoRA + the
    /// existing compute boundary, then stores the returned V in the cache.
    /// Direct attention callers/tests may still use it through `encode`.
    func negPiPValueIfNeeded(
        commandBuffer: MTLCommandBuffer,
        value: MTLBuffer,
        valueOffset: Int,
        heads: Int,
        keyCount: Int,
        headDim: Int,
        kvHeads: Int,
        layout: AttentionInputLayout,
        probe: NumericalMonitor.Probe?
    ) throws -> (buffer: MTLBuffer, offset: Int) {
        guard case .crossScores? = probe,
              let snapshot = NegPiPGenerationContext.snapshot() else {
            return (value, valueOffset)
        }
        guard heads == DiTBlockExecutor.heads,
              kvHeads == DiTBlockExecutor.heads,
              keyCount == DiTBlockExecutor.contextTokens,
              headDim == DiTBlockExecutor.headDim,
              !snapshot.signs.isEmpty else {
            throw AnimapkError.validation("active NegPiP reached an unexpected cross-attention shape")
        }
        guard valueOffset == 0 else {
            throw AnimapkError.validation("NegPiP requires a zero-based DiT V buffer")
        }

        let width: Int
        let height: Int
        let layoutKey: String
        switch layout {
        case .tokenMajor(let tokenStride):
            guard tokenStride == heads * headDim else {
                throw AnimapkError.validation("NegPiP token-major stride does not match DiT model width")
            }
            width = tokenStride
            height = keyCount
            layoutKey = "token"
        case .headMajor:
            width = headDim
            height = kvHeads * keyCount
            layoutKey = "head"
        }

        let rowBytes = try checkedProduct(width, MemoryLayout<Float16>.stride)
        let bytes = try checkedProduct(rowBytes, height)
        let alignment = context.device.minimumLinearTextureAlignment(for: .r16Float)
        guard alignment > 0, rowBytes.isMultiple(of: alignment), value.length >= bytes else {
            throw AnimapkError.validation("NegPiP V layout is not compatible with an r16Float buffer texture")
        }
        // All production DiT projection/transposed V scratch and the unit-test
        // oracle use BufferPool/.storageModeShared. Linear buffer textures must
        // declare the same storage mode as their backing buffer; leaving the
        // descriptor at its default causes Metal validation to abort.
        guard value.storageMode == .shared else {
            throw AnimapkError.validation("NegPiP requires shared DiT V scratch")
        }

        let mask = buffers.buffer(key: "attention.negpip.mask.\(layoutKey)", bytes: bytes)
        if negPiPMaskGeneration != snapshot.lease || negPiPMaskLayoutKey != layoutKey {
            let pointer = mask.contents().bindMemory(
                to: Float16.self, capacity: bytes / MemoryLayout<Float16>.stride)
            switch layout {
            case .tokenMajor:
                for token in 0..<keyCount {
                    let sign = Float16(snapshot.signs[token])
                    let base = token * width
                    for column in 0..<width { pointer[base + column] = sign }
                }
            case .headMajor:
                for head in 0..<kvHeads {
                    for token in 0..<keyCount {
                        let sign = Float16(snapshot.signs[token])
                        let base = (head * keyCount + token) * headDim
                        for column in 0..<headDim { pointer[base + column] = sign }
                    }
                }
            }
            negPiPMaskGeneration = snapshot.lease
            negPiPMaskLayoutKey = layoutKey
        }

        let masked = buffers.buffer(key: "attention.negpip.value.\(layoutKey)", bytes: bytes)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let sourceTexture = value.makeTexture(
                descriptor: descriptor, offset: valueOffset, bytesPerRow: rowBytes),
              let maskTexture = mask.makeTexture(
                descriptor: descriptor, offset: 0, bytesPerRow: rowBytes),
              let destinationTexture = masked.makeTexture(
                descriptor: descriptor, offset: 0, bytesPerRow: rowBytes) else {
            throw AnimapkError.validation("failed to create NegPiP buffer-backed textures")
        }
        negPiPMultiply.encode(
            commandBuffer: commandBuffer,
            primaryTexture: sourceTexture,
            secondaryTexture: maskTexture,
            destinationTexture: destinationTexture)
        return (masked, 0)
    }

    private func encodeRoundHalf(
        _ commandBuffer: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        count: Int, offset: Int = 0
    ) throws {
        let pipeline = try context.pipeline(named: "round_half_to_bf16")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create BF16 attention boundary encoder")
        }
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: offset, index: 0)
        encoder.setBuffer(output, offset: offset, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        metrics?.recordConversionBytes(UInt64(Int(count) * MemoryLayout<Float16>.stride))
    }

    private func encodeFP32(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int, key: MTLBuffer, keyOffset: Int,
        value: MTLBuffer, valueOffset: Int, output: MTLBuffer, outputOffset: Int,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        kvHeads: Int, causal: Bool, probe: NumericalMonitor.Probe?
    ) throws {
        let halfBytes = MemoryLayout<Float16>.stride
        let headRowBytes = headDim * halfBytes
        let scoreScratch = buffers.buffer(
            key: "attention.scores.fp32", bytes: try maximumScoreScratchBytes(keyCount: keyCount, queryCount: queryCount))
        let qk = try context.pipeline(named: "attention_qk_f16_to_f32")
        let softmaxName = (monitor != nil && probe != nil)
            ? "attention_softmax_rows_f32_probe" : "attention_softmax_rows_f32"
        let softmax = try context.pipeline(named: softmaxName)
        let pv = try context.pipeline(named: "attention_pv_f32_f16_to_f16")
        let scale = Float(1 / sqrt(Double(headDim)))

        for head in 0..<heads {
            let kvHead = head / (heads / kvHeads)
            var queryBase = 0
            while queryBase < queryCount {
                let rows = min(tileRows, queryCount - queryBase)
                guard let qkEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create FP32 attention QK encoder")
                }
                var rowCount = UInt32(rows), columns = UInt32(keyCount)
                var dimension = UInt32(headDim)
                qkEncoder.setComputePipelineState(qk)
                qkEncoder.setBuffer(query, offset: queryOffset + (head * queryCount + queryBase) * headRowBytes, index: 0)
                qkEncoder.setBuffer(key, offset: keyOffset + kvHead * keyCount * headRowBytes, index: 1)
                qkEncoder.setBuffer(scoreScratch, offset: 0, index: 2)
                qkEncoder.setBytes(&rowCount, length: 4, index: 3)
                qkEncoder.setBytes(&columns, length: 4, index: 4)
                qkEncoder.setBytes(&dimension, length: 4, index: 5)
                var fpScale = scale
                qkEncoder.setBytes(&fpScale, length: 4, index: 6)
                qkEncoder.dispatchThreads(MTLSize(width: keyCount, height: rows, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                qkEncoder.endEncoding()

                guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create FP32 attention softmax encoder")
                }
                var base = UInt32(queryBase), causalFlag = causal ? UInt32(1) : UInt32(0)
                softmaxEncoder.setComputePipelineState(softmax)
                softmaxEncoder.setBuffer(scoreScratch, offset: 0, index: 0)
                softmaxEncoder.setBytes(&rowCount, length: 4, index: 1)
                softmaxEncoder.setBytes(&columns, length: 4, index: 2)
                softmaxEncoder.setBytes(&base, length: 4, index: 3)
                softmaxEncoder.setBytes(&causalFlag, length: 4, index: 4)
                if let monitor, let probe {
                    monitor.bindProbe(softmaxEncoder, probe: probe, statsIndex: 5, slotIndex: 6)
                }
                let threads = reductionThreads(limit: softmax.maxTotalThreadsPerThreadgroup)
                softmaxEncoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                softmaxEncoder.endEncoding()

                guard let pvEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create FP32 attention PV encoder")
                }
                pvEncoder.setComputePipelineState(pv)
                pvEncoder.setBuffer(scoreScratch, offset: 0, index: 0)
                pvEncoder.setBuffer(value, offset: valueOffset + kvHead * keyCount * headRowBytes, index: 1)
                pvEncoder.setBuffer(output, offset: outputOffset + (head * queryCount + queryBase) * headRowBytes, index: 2)
                pvEncoder.setBytes(&rowCount, length: 4, index: 3)
                pvEncoder.setBytes(&columns, length: 4, index: 4)
                pvEncoder.setBytes(&dimension, length: 4, index: 5)
                pvEncoder.dispatchThreads(MTLSize(width: headDim, height: rows, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                pvEncoder.endEncoding()
                metrics?.recordConversionBytes(UInt64(rows * keyCount * MemoryLayout<Float>.stride))
                metrics?.recordConversionBytes(UInt64(rows * headDim * MemoryLayout<Float16>.stride))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// P4: strided token-major attention. Q/K/V/output are `[rows, tokenStride]`
    /// half buffers; each head is an `MPSMatrix` VIEW with the token row stride.
    private func encodeTokenMajor(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int,
        key: MTLBuffer, keyOffset: Int,
        value: MTLBuffer, valueOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        kvHeads: Int, tokenStride: Int,
        causal: Bool, probe: NumericalMonitor.Probe?,
        scoreScratch: MTLBuffer, softmax: MTLComputePipelineState,
        halfBytes: Int, scoreRowBytes: Int, scale: Double
    ) throws {
        for head in 0..<heads {
            _ = kvHeads
            let keyMatrix = tokenMajorHeadMatrix(
                buffer: key, baseOffset: keyOffset, head: head,
                rowBase: 0, rows: keyCount, tokenStride: tokenStride, headDim: headDim)
            let valueMatrix = tokenMajorHeadMatrix(
                buffer: value, baseOffset: valueOffset, head: head,
                rowBase: 0, rows: keyCount, tokenStride: tokenStride, headDim: headDim)
            var queryBase = 0
            while queryBase < queryCount {
                let rows = min(tileRows, queryCount - queryBase)
                let queryMatrix = tokenMajorHeadMatrix(
                    buffer: query, baseOffset: queryOffset, head: head,
                    rowBase: queryBase, rows: rows, tokenStride: tokenStride, headDim: headDim)
                let scoreMatrix = MPSMatrix(
                    buffer: scoreScratch,
                    descriptor: MPSMatrixDescriptor(rows: rows, columns: keyCount,
                                                    rowBytes: scoreRowBytes, dataType: .float16))
                MPSMatrixMultiplication(
                    device: context.device, transposeLeft: false, transposeRight: true,
                    resultRows: rows, resultColumns: keyCount, interiorColumns: headDim,
                    alpha: scale, beta: 0).encode(
                        commandBuffer: commandBuffer, leftMatrix: queryMatrix,
                        rightMatrix: keyMatrix, resultMatrix: scoreMatrix)

                if numerics == .bf16Compute {
                    try encodeRoundHalf(commandBuffer, input: scoreScratch,
                                        output: scoreScratch, count: rows * keyCount)
                }

                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create attention softmax encoder")
                }
                var rowCount = UInt32(rows), columns = UInt32(keyCount)
                var base = UInt32(queryBase), causalFlag = causal ? UInt32(1) : UInt32(0)
                encoder.setComputePipelineState(softmax)
                encoder.setBuffer(scoreScratch, offset: 0, index: 0)
                encoder.setBytes(&rowCount, length: 4, index: 1)
                encoder.setBytes(&columns, length: 4, index: 2)
                encoder.setBytes(&base, length: 4, index: 3)
                encoder.setBytes(&causalFlag, length: 4, index: 4)
                if let monitor, let probe {
                    monitor.bindProbe(encoder, probe: probe, statsIndex: 5, slotIndex: 6)
                }
                let threads = reductionThreads(limit: softmax.maxTotalThreadsPerThreadgroup)
                encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                encoder.endEncoding()

                if numerics == .bf16Compute {
                    try encodeRoundHalf(commandBuffer, input: scoreScratch,
                                        output: scoreScratch, count: rows * keyCount)
                }

                let outputMatrix = tokenMajorHeadMatrix(
                    buffer: output, baseOffset: outputOffset, head: head,
                    rowBase: queryBase, rows: rows, tokenStride: tokenStride, headDim: headDim)
                MPSMatrixMultiplication(
                    device: context.device, transposeLeft: false, transposeRight: false,
                    resultRows: rows, resultColumns: headDim, interiorColumns: keyCount,
                    alpha: 1, beta: 0).encode(
                        commandBuffer: commandBuffer, leftMatrix: scoreMatrix,
                        rightMatrix: valueMatrix, resultMatrix: outputMatrix)
                metrics?.recordConversionBytes(UInt64(rows * keyCount * halfBytes))
                metrics?.recordConversionBytes(UInt64(rows * headDim * halfBytes))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// P7-A: streaming/online-softmax MPS attention over the strided
    /// token-major layout (DiT).
    private func encodeStreamingTokenMajor(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int,
        key: MTLBuffer, keyOffset: Int,
        value: MTLBuffer, valueOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        kvHeads: Int, tokenStride: Int,
        causal: Bool, probe: NumericalMonitor.Probe?,
        halfBytes: Int, scale: Double
    ) throws {
        guard !causal else {
            throw AnimapkError.validation("P7 streaming MPS attention requires non-causal attention")
        }
        _ = kvHeads
        let keyChunks = [64, 128, 256]
        let chunkColumns = keyChunks.first(where: { $0 <= keyCount }) ?? max(1, keyCount)
        let prepare = try context.pipeline(named: "streaming_softmax_prepare")
        let accumulate = try context.pipeline(named: "streaming_chunk_accumulate")
        let finalize = try context.pipeline(named: "streaming_softmax_finalize")
        let scoreScratch = buffers.buffer(
            key: "attention.stream.scores.fp16",
            bytes: try checkedProduct(tileRows, chunkColumns, halfBytes))
        let maxTileRows = min(tileRows, queryCount)
        let stateBytes = try checkedProduct(maxTileRows, MemoryLayout<Float>.stride)
        let accumulatorBytes = try checkedProduct(maxTileRows, headDim, MemoryLayout<Float>.stride)
        let runningMax = buffers.buffer(key: "attention.stream.max.f32", bytes: stateBytes)
        let runningSum = buffers.buffer(key: "attention.stream.sum.f32", bytes: stateBytes)
        let runningAlpha = buffers.buffer(key: "attention.stream.alpha.f32", bytes: stateBytes)
        let accumulator = buffers.buffer(key: "attention.stream.acc.f32", bytes: accumulatorBytes)
        let scaleFloat = Float(scale)
        _ = scaleFloat
        let reduction = reductionThreads(limit: prepare.maxTotalThreadsPerThreadgroup)

        for head in 0..<heads {
            let keyMatrix = tokenMajorHeadMatrix(
                buffer: key, baseOffset: keyOffset, head: head,
                rowBase: 0, rows: keyCount, tokenStride: tokenStride, headDim: headDim)
            let valueMatrix = tokenMajorHeadMatrix(
                buffer: value, baseOffset: valueOffset, head: head,
                rowBase: 0, rows: keyCount, tokenStride: tokenStride, headDim: headDim)
            var queryBase = 0
            while queryBase < queryCount {
                let rows = min(tileRows, queryCount - queryBase)
                let queryMatrix = tokenMajorHeadMatrix(
                    buffer: query, baseOffset: queryOffset, head: head,
                    rowBase: queryBase, rows: rows, tokenStride: tokenStride, headDim: headDim)
                let outputMatrix = tokenMajorHeadMatrix(
                    buffer: output, baseOffset: outputOffset, head: head,
                    rowBase: queryBase, rows: rows, tokenStride: tokenStride, headDim: headDim)
                var chunkBase = 0
                var chunkIndex = 0
                while chunkBase < keyCount {
                    let columns = min(chunkColumns, keyCount - chunkBase)
                    let scoreMatrix = MPSMatrix(
                        buffer: scoreScratch,
                        descriptor: MPSMatrixDescriptor(rows: rows, columns: columns,
                                                        rowBytes: columns * halfBytes, dataType: .float16))
                    MPSMatrixMultiplication(
                        device: context.device, transposeLeft: false, transposeRight: true,
                        resultRows: rows, resultColumns: columns, interiorColumns: headDim,
                        alpha: scale, beta: 0).encode(
                            commandBuffer: commandBuffer, leftMatrix: queryMatrix,
                            rightMatrix: keyMatrix, resultMatrix: scoreMatrix)

                    guard let prepareEncoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw AnimapkError.validation("failed to create streaming attention prepare encoder")
                    }
                    var rowCount = UInt32(rows), colCount = UInt32(columns)
                    var stride = UInt32(columns)
                    var firstChunk = chunkIndex == 0 ? UInt32(1) : UInt32(0)
                    prepareEncoder.setComputePipelineState(prepare)
                    prepareEncoder.setBuffer(scoreScratch, offset: 0, index: 0)
                    prepareEncoder.setBuffer(runningMax, offset: 0, index: 1)
                    prepareEncoder.setBuffer(runningSum, offset: 0, index: 2)
                    prepareEncoder.setBuffer(runningAlpha, offset: 0, index: 3)
                    prepareEncoder.setBytes(&rowCount, length: 4, index: 4)
                    prepareEncoder.setBytes(&colCount, length: 4, index: 5)
                    prepareEncoder.setBytes(&stride, length: 4, index: 6)
                    prepareEncoder.setBytes(&firstChunk, length: 4, index: 7)
                    prepareEncoder.dispatchThreadgroups(
                        MTLSize(width: rows, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: reduction, height: 1, depth: 1))
                    prepareEncoder.endEncoding()

                    let chunkOutMatrix = MPSMatrix(
                        buffer: buffers.buffer(
                            key: "attention.stream.chunkOut.fp16",
                            bytes: try checkedProduct(rows, headDim, halfBytes)),
                        descriptor: MPSMatrixDescriptor(rows: rows, columns: headDim,
                                                        rowBytes: headDim * halfBytes, dataType: .float16))
                    MPSMatrixMultiplication(
                        device: context.device, transposeLeft: false, transposeRight: false,
                        resultRows: rows, resultColumns: headDim, interiorColumns: columns,
                        alpha: 1, beta: 0).encode(
                            commandBuffer: commandBuffer, leftMatrix: scoreMatrix,
                            rightMatrix: valueMatrix, resultMatrix: chunkOutMatrix)

                    guard let accEncoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw AnimapkError.validation("failed to create streaming attention accumulate encoder")
                    }
                    accEncoder.setComputePipelineState(accumulate)
                    accEncoder.setBuffer(chunkOutMatrix.data, offset: chunkOutMatrix.offset, index: 0)
                    accEncoder.setBuffer(accumulator, offset: 0, index: 1)
                    accEncoder.setBuffer(runningAlpha, offset: 0, index: 2)
                    accEncoder.setBytes(&rowCount, length: 4, index: 3)
                    var dim = UInt32(headDim)
                    accEncoder.setBytes(&dim, length: 4, index: 4)
                    accEncoder.setBytes(&firstChunk, length: 4, index: 5)
                    let width = min(accumulate.threadExecutionWidth, accumulate.maxTotalThreadsPerThreadgroup)
                    accEncoder.dispatchThreads(
                        MTLSize(width: headDim, height: rows, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                    accEncoder.endEncoding()

                    chunkBase += columns
                    chunkIndex += 1
                }
                guard let finalizeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create streaming attention finalize encoder")
                }
                var rowCount = UInt32(rows), dim = UInt32(headDim)
                var stride = UInt32(tokenStride)
                finalizeEncoder.setComputePipelineState(finalize)
                finalizeEncoder.setBuffer(accumulator, offset: 0, index: 0)
                finalizeEncoder.setBuffer(runningSum, offset: 0, index: 1)
                finalizeEncoder.setBuffer(outputMatrix.data, offset: outputMatrix.offset, index: 2)
                finalizeEncoder.setBytes(&rowCount, length: 4, index: 3)
                finalizeEncoder.setBytes(&dim, length: 4, index: 4)
                finalizeEncoder.setBytes(&stride, length: 4, index: 5)
                let width = min(finalize.threadExecutionWidth, finalize.maxTotalThreadsPerThreadgroup)
                finalizeEncoder.dispatchThreads(
                    MTLSize(width: headDim, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                finalizeEncoder.endEncoding()

                metrics?.recordConversionBytes(UInt64(rows * chunkColumns * halfBytes))
                metrics?.recordConversionBytes(UInt64(rows * headDim * halfBytes))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// P7-B: DiT-specialized pure-Metal Flash-style online attention.
    private func encodeMetalFlash(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int,
        key: MTLBuffer, keyOffset: Int,
        value: MTLBuffer, valueOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        kvHeads: Int, tokenStride: Int,
        causal: Bool, probe: NumericalMonitor.Probe?,
        scoreScratch: MTLBuffer, softmax: MTLComputePipelineState,
        halfBytes: Int, scoreRowBytes: Int, scale: Double
    ) throws {
        _ = probe; _ = scoreScratch; _ = softmax; _ = scoreRowBytes; _ = scale
        guard !causal, heads == 16, headDim == 128,
              tokenStride == heads * headDim, kvHeads == heads else {
            throw AnimapkError.validation(
                "P7 Metal Flash attention is DiT-specialized: requires heads == 16, headDim == 128, non-causal, tokenStride == heads * headDim")
        }
        let pipeline = try context.pipeline(named: "dit_flash_attention_h128_q4_k32")
        guard pipeline.threadExecutionWidth == 32 else {
            throw AnimapkError.validation(
                "P7 Metal Flash attention requires a 32-lane SIMD pipeline (threadExecutionWidth == 32); backend unsupported on this device")
        }
        let k16Pipeline = try context.pipeline(named: "dit_flash_attention_h128_q4_k16")
        guard k16Pipeline.threadExecutionWidth == 32 else {
            throw AnimapkError.validation(
                "P7 Metal Flash attention requires a 32-lane SIMD pipeline (threadExecutionWidth == 32); backend unsupported on this device")
        }
        let groupsPerHead = (queryCount + 3) / 4
        var queryCountU = UInt32(queryCount), keyCountU = UInt32(keyCount)
        var tokenStrideU = UInt32(tokenStride)
        var scaleF = Float(1 / sqrt(Double(headDim)))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Metal Flash attention encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(query, offset: queryOffset, index: 0)
        encoder.setBuffer(key, offset: keyOffset, index: 1)
        encoder.setBuffer(value, offset: valueOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBytes(&queryCountU, length: 4, index: 4)
        encoder.setBytes(&keyCountU, length: 4, index: 5)
        encoder.setBytes(&tokenStrideU, length: 4, index: 6)
        encoder.setBytes(&scaleF, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: groupsPerHead, height: 1, depth: heads),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        metrics?.recordConversionBytes(UInt64(queryCount * tokenStride * halfBytes))
        metrics?.recordAttentionQueryTile()
    }

    /// P4-B: strided per-head matrix view of a token-major `[rows, tokenStride]`
    /// half buffer.
    private func tokenMajorHeadMatrix(
        buffer: MTLBuffer, baseOffset: Int, head: Int,
        rowBase: Int, rows: Int, tokenStride: Int, headDim: Int
    ) -> MPSMatrix {
        let scalarBytes = MemoryLayout<Float16>.stride
        let rowBytes = tokenStride * scalarBytes
        let offset = baseOffset + rowBase * rowBytes + head * headDim * scalarBytes
        return MPSMatrix(buffer: buffer, offset: offset,
                         descriptor: MPSMatrixDescriptor(rows: rows, columns: headDim,
                                                         rowBytes: rowBytes, dataType: .float16))
    }

    func execute(
        query: MTLBuffer, queryOffset: Int = 0,
        key: MTLBuffer, keyOffset: Int = 0,
        value: MTLBuffer, valueOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        keyValueHeads: Int? = nil,
        causal: Bool = false,
        layout: AttentionInputLayout? = nil
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create attention command buffer")
        }
        try encode(commandBuffer: command, query: query, queryOffset: queryOffset,
                   key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
                   output: output, outputOffset: outputOffset, heads: heads,
                   queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                   keyValueHeads: keyValueHeads, causal: causal, layout: layout)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }

    private func validate(
        query: MTLBuffer, queryOffset: Int, key: MTLBuffer, keyOffset: Int,
        value: MTLBuffer, valueOffset: Int, output: MTLBuffer, outputOffset: Int,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        keyValueHeads: Int, causal: Bool, layout: AttentionInputLayout
    ) throws {
        guard heads > 0, queryCount > 0, keyCount > 0, headDim > 0,
              keyValueHeads > 0, heads.isMultiple(of: keyValueHeads),
              queryOffset >= 0, keyOffset >= 0, valueOffset >= 0, outputOffset >= 0,
              !causal || queryCount == keyCount else {
            throw AnimapkError.validation("invalid attention shape, offset, or causal dimensions")
        }
        if case .tokenMajor(let tokenStride) = layout {
            guard tokenStride > 0, headDim <= tokenStride,
                  tokenStride.isMultiple(of: headDim),
                  heads * headDim == tokenStride,
                  keyValueHeads == heads else {
                throw AnimapkError.validation(
                    "P4 token-major attention requires tokenStride == heads * headDim and keyValueHeads == heads")
            }
            guard queryOffset % 16 == 0, keyOffset % 16 == 0,
                  valueOffset % 16 == 0, outputOffset % 16 == 0,
                  (tokenStride * 2).isMultiple(of: 16) else {
                throw AnimapkError.validation(
                    "P4 token-major attention requires 16-byte-aligned offsets and row stride")
            }
        }
        let queryBytes: Int
        let keyBytes: Int
        if case .tokenMajor(let tokenStride) = layout {
            let elementBytes = MemoryLayout<Float16>.stride
            queryBytes = try checkedProduct(queryCount, tokenStride, elementBytes)
            keyBytes = try checkedProduct(keyCount, tokenStride, elementBytes)
        } else {
            queryBytes = try checkedProduct(heads, queryCount, headDim, 2)
            keyBytes = try checkedProduct(keyValueHeads, keyCount, headDim, 2)
        }
        guard try checkedEnd(queryOffset, queryBytes) <= query.length,
              try checkedEnd(keyOffset, keyBytes) <= key.length,
              try checkedEnd(valueOffset, keyBytes) <= value.length,
              try checkedEnd(outputOffset, queryBytes) <= output.length else {
            throw AnimapkError.validation("attention buffer range is out of bounds")
        }
    }

    private func reductionThreads(limit: Int) -> Int {
        var threads = 1
        while threads * 2 <= min(256, limit) { threads *= 2 }
        return threads
    }

    private func checkedProduct(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else { throw AnimapkError.validation("attention byte size overflow") }
            result = next
        }
        return result
    }

    private func checkedEnd(_ offset: Int, _ bytes: Int) throws -> Int {
        let (end, overflow) = offset.addingReportingOverflow(bytes)
        guard !overflow else { throw AnimapkError.validation("attention buffer range overflow") }
        return end
    }
}
