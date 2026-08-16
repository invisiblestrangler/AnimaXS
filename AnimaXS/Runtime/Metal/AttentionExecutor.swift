import Foundation
import Metal
import MetalPerformanceShaders

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
    /// running FP32 max/sum and an FP32 output accumulator, so no full
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
        // Bound the configured tile by the real query count so scratch sizing
        // never over-allocates beyond what the row loop will actually touch.
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
        layout: AttentionInputLayout? = nil
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
                           key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
                           output: output, outputOffset: outputOffset, heads: heads,
                           queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                           kvHeads: kvHeads, causal: causal, probe: probe)
            return
        }

        if inputLayout.isTokenMajor {
            // P4-F: the BF16 boundary round is contiguous in the legacy layout
            // but would corrupt the strided token-major layout; refuse loudly
            // instead of silently producing wrong results. Select baseline
            // numerics (or disable the strided toggle) to proceed.
            guard numerics != .bf16Compute else {
                throw AnimapkError.validation(
                    "P4 strided token-major attention does not support bf16Compute numerics; select baseline numerics")
            }
            // Guaranteed non-nil here: isTokenMajor is true only for .tokenMajor.
            guard let tokenStride else {
                throw AnimapkError.validation(
                    "P4 token-major attention requires a token stride")
            }
            switch attentionBackend {
            case .streamingMPS:
                // P7-A: streaming/online-softmax MPS — same strided per-head
                // views, but keys are processed in chunks with a running
                // FP32 max/sum and FP32 output accumulator.
                try encodeStreamingTokenMajor(
                    commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                    key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
                    output: output, outputOffset: outputOffset,
                    heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                    kvHeads: kvHeads, tokenStride: tokenStride, causal: causal, probe: probe,
                    halfBytes: halfBytes, scale: scale)
                return
            case .metalFlash:
                // P7-B: DiT-specialized pure-Metal Flash attention. Strict
                // shape requirements; anything else throws (never corrupts).
                try encodeMetalFlash(
                    commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                    key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
                    output: output, outputOffset: outputOffset,
                    heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                    kvHeads: kvHeads, tokenStride: tokenStride, causal: causal,
                    scoreScratch: scoreScratch, softmax: softmax, halfBytes: halfBytes,
                    scoreRowBytes: scoreRowBytes, scale: scale)
                return
            case .legacyHeadMajorMPS, .stridedTokenMajorMPS:
                break
            }
            try encodeTokenMajor(
                commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
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
                buffer: value, offset: valueOffset + kvHead * keyCount * headRowBytes,
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
                // P2-C: fp16 score tile + PV result materialized (counted once each).
                metrics?.recordConversionBytes(UInt64(rows * keyCount * halfBytes))
                metrics?.recordConversionBytes(UInt64(rows * headDim * halfBytes))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
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
        // P2-C: fp16 elements materialized by the BF16 round-trip (counted once).
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
                // P2-C: fp32 score tile materialized (f16→f32) + fp16 PV result
                // (f32→f16), each counted once.
                metrics?.recordConversionBytes(UInt64(rows * keyCount * MemoryLayout<Float>.stride))
                metrics?.recordConversionBytes(UInt64(rows * headDim * MemoryLayout<Float16>.stride))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// P4: strided token-major attention. Q/K/V/output are `[rows, tokenStride]`
    /// half buffers; each head is an `MPSMatrix` VIEW with the token row stride
    /// (`tokenStride * 2` bytes) and a per-head column offset of
    /// `head * headDim * 2` bytes. The score tile stays TIGHT (`rows x keyCount`,
    /// rowBytes = keyCount * 2) — never a token-dim stride. No transpose kernels.
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
            // kvHeads == heads is validated for token-major (P4-D keeps the
            // GQA head mapping for the legacy head-major path only).
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
                // P2-C: fp16 score tile + PV result materialized (counted once each).
                metrics?.recordConversionBytes(UInt64(rows * keyCount * halfBytes))
                metrics?.recordConversionBytes(UInt64(rows * headDim * halfBytes))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// P7-A: streaming/online-softmax MPS attention over the strided
    /// token-major layout (DiT). MPS computes QK^T and PV per KEY CHUNK
    /// (Bk ∈ {64, 128, 256}); a tiny fp16 `[rows, Bk]` score tile is the only
    /// live score memory. The online-softmax state (running max FP32, running
    /// sum FP32, FP32 output accumulator) is carried across chunks by three
    /// compute kernels; every chunk of every query tile encodes into the
    /// SAME block command buffer — NO per-chunk wait, NO added command-buffer
    /// completion. The output accumulator is NEVER fp16.
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
        // The online-softmax kernels are non-causal (DiT self/cross are
        // non-causal); a causal streaming request would silently produce
        // wrong results, so refuse loudly.
        guard !causal else {
            throw AnimapkError.validation("P7 streaming MPS attention requires non-causal attention")
        }
        _ = kvHeads  // == heads, validated by the token-major layout rules
        let keyChunks = [64, 128, 256]
        // Bound the chunk by the key count so a small test (e.g. K=5) still
        // exercises a single chunk and never over-allocates.
        let chunkColumns = keyChunks.first(where: { $0 <= keyCount }) ?? max(1, keyCount)
        let prepare = try context.pipeline(named: "streaming_softmax_prepare")
        let accumulate = try context.pipeline(named: "streaming_chunk_accumulate")
        let finalize = try context.pipeline(named: "streaming_softmax_finalize")
        let scoreScratch = buffers.buffer(
            key: "attention.stream.scores.fp16",
            bytes: try checkedProduct(tileRows, chunkColumns, halfBytes))
        // FP32 online state: max/sum/alpha per query-tile row + output
        // accumulator [tileRows, headDim]. Bounded by the real query count.
        let maxTileRows = min(tileRows, queryCount)
        let stateBytes = try checkedProduct(maxTileRows, MemoryLayout<Float>.stride)
        let accumulatorBytes = try checkedProduct(maxTileRows, headDim, MemoryLayout<Float>.stride)
        let runningMax = buffers.buffer(key: "attention.stream.max.f32", bytes: stateBytes)
        let runningSum = buffers.buffer(key: "attention.stream.sum.f32", bytes: stateBytes)
        let runningAlpha = buffers.buffer(key: "attention.stream.alpha.f32", bytes: stateBytes)
        let accumulator = buffers.buffer(key: "attention.stream.acc.f32", bytes: accumulatorBytes)
        let scaleFloat = Float(scale)
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
                    // QK^T for this chunk only (strided per-head views).
                    MPSMatrixMultiplication(
                        device: context.device, transposeLeft: false, transposeRight: true,
                        resultRows: rows, resultColumns: columns, interiorColumns: headDim,
                        alpha: scale, beta: 0).encode(
                            commandBuffer: commandBuffer, leftMatrix: queryMatrix,
                            rightMatrix: keyMatrix, resultMatrix: scoreMatrix)

                    // Online-softmax prepare: update running max/sum, rewrite
                    // the chunk scores to rescaled probabilities, store alpha.
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

                    // PV for this chunk (MPS).
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

                    // FP32 accumulate: acc = alpha * acc + float(chunkOut).
                    guard let accEncoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw AnimapkError.validation("failed to create streaming attention accumulate encoder")
                    }
                    accEncoder.setComputePipelineState(accumulate)
                    accEncoder.setBuffer(chunkOutMatrix.buffer, offset: chunkOutMatrix.offset, index: 0)
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
                // Finalize: output = half(acc / runningSum) into the
                // token-major output view (strided write).
                guard let finalizeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw AnimapkError.validation("failed to create streaming attention finalize encoder")
                }
                var rowCount = UInt32(rows), dim = UInt32(headDim)
                var stride = UInt32(tokenStride)
                finalizeEncoder.setComputePipelineState(finalize)
                finalizeEncoder.setBuffer(accumulator, offset: 0, index: 0)
                finalizeEncoder.setBuffer(runningSum, offset: 0, index: 1)
                finalizeEncoder.setBuffer(outputMatrix.buffer, offset: outputMatrix.offset, index: 2)
                finalizeEncoder.setBytes(&rowCount, length: 4, index: 3)
                finalizeEncoder.setBytes(&dim, length: 4, index: 4)
                finalizeEncoder.setBytes(&stride, length: 4, index: 5)
                let width = min(finalize.threadExecutionWidth, finalize.maxTotalThreadsPerThreadgroup)
                finalizeEncoder.dispatchThreads(
                    MTLSize(width: headDim, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                finalizeEncoder.endEncoding()

                // P2-C: the score tile and PV result materialized by MPS for
                // this chunk (counted once each); the fp32 state traffic is
                // internal to the backend and not double-counted.
                metrics?.recordConversionBytes(UInt64(rows * chunkColumns * halfBytes))
                metrics?.recordConversionBytes(UInt64(rows * headDim * halfBytes))
                metrics?.recordAttentionQueryTile()
                queryBase += rows
            }
        }
    }

    /// P7-B: DiT-specialized pure-Metal Flash-style online attention. Strict
    /// shape gate: headDim == 128, heads == 16, non-causal, token-major with
    /// tokenStride == heads * headDim, and the selected compute pipeline must
    /// expose `threadExecutionWidth == 32` (the kernel's SIMD-group mapping
    /// assumes 32 lanes). Anything else throws loudly — the backend never
    /// runs on an unsupported shape. Qwen/VAE/adapter (head-major layout)
    /// can never reach this path.
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
        guard !causal, heads == 16, headDim == 128,
              tokenStride == heads * headDim, kvHeads == heads else {
            throw AnimapkError.validation(
                "P7 Metal Flash attention is DiT-specialized: requires heads == 16, headDim == 128, non-causal, tokenStride == heads * headDim")
        }
        // A12-safe gate: the kernel's SIMD-group mapping assumes a 32-lane
        // SIMD width. If the device/pipeline reports anything else, mark the
        // backend unsupported for this device and refuse to run.
        let pipeline = try context.pipeline(named: "dit_flash_attention_h128_q4_k32")
        guard pipeline.threadExecutionWidth == 32 else {
            throw AnimapkError.validation(
                "P7 Metal Flash attention requires a 32-lane SIMD pipeline (threadExecutionWidth == 32); backend unsupported on this device")
        }
        // The K=16 profile is the fallback for devices whose threadgroup
        // memory or occupancy prefers a smaller tile; it is selected at the
        // call site and must pass the same SIMD gate.
        let k16Pipeline = try context.pipeline(named: "dit_flash_attention_h128_q4_k16")
        guard k16Pipeline.threadExecutionWidth == 32 else {
            throw AnimapkError.validation(
                "P7 Metal Flash attention requires a 32-lane SIMD pipeline (threadExecutionWidth == 32); backend unsupported on this device")
        }
        // One threadgroup covers 4 query rows; grid = (ceil(queryCount/4),
        // 1, heads). group.z carries the head index into the kernel.
        let groupsPerHead = (queryCount + 3) / 4
        var queryCountU = UInt32(queryCount), keyCountU = UInt32(keyCount)
        var tokenStrideU = UInt32(tokenStride)
        var scaleF = Float(scale)
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
        // P2-C: every output element materialized once (the backend writes
        // the token-major attended buffer directly); no score tile is ever
        // materialized by the Flash path.
        metrics?.recordConversionBytes(UInt64(queryCount * tokenStride * halfBytes))
        metrics?.recordAttentionQueryTile()
    }

    /// P4-B: strided per-head matrix view of a token-major `[rows, tokenStride]`
    /// half buffer. Each head reads exactly its `headDim` contiguous columns of
    /// every token row; the row stride is the full token stride, so head `h`
    /// starts at column `h * headDim` of every row.
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
            // P4: each head owns headDim contiguous columns of every token row;
            // GQA is not expressible without a strided K/V head gather.
            guard tokenStride > 0, headDim <= tokenStride,
                  tokenStride.isMultiple(of: headDim),
                  heads * headDim == tokenStride,
                  keyValueHeads == heads else {
                throw AnimapkError.validation(
                    "P4 token-major attention requires tokenStride == heads * headDim and keyValueHeads == heads")
            }
            // P4-F: refuse to silently corrupt layout when MPS rejects the
            // strided descriptor — fail the experimental backend loudly.
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
