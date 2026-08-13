import Foundation
import Metal
import MetalPerformanceShaders

enum AttentionNumerics: String, CaseIterable {
    case legacy
    case fp32ScoresAndSoftmax
}

/// Query-tiled scaled dot-product attention over tightly packed fp16 tensors.
/// Layout is `[heads, rows, headDim]`; only one `[tileRows,keyCount]` score tile
/// is retained. The executor is intentionally non-reentrant because it reuses scratch.
final class AttentionExecutor {
    static let defaultTileRows = 128

    private let context: MetalContext
    private let buffers: BufferPool
    let tileRows: Int
    let numerics: AttentionNumerics

    init(context: MetalContext, tileRows: Int = defaultTileRows,
         numerics: AttentionNumerics = .legacy) {
        precondition(tileRows > 0)
        self.context = context
        self.buffers = BufferPool(device: context.device)
        self.tileRows = tileRows
        self.numerics = numerics
    }

    func maximumScoreScratchBytes(keyCount: Int) throws -> Int {
        guard keyCount > 0 else { throw AnimapkError.validation("attention key count must be positive") }
        let elementBytes = numerics == .legacy
            ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        return try checkedProduct(tileRows, keyCount, elementBytes)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int = 0,
        key: MTLBuffer, keyOffset: Int = 0,
        value: MTLBuffer, valueOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        keyValueHeads: Int? = nil,
        causal: Bool = false
    ) throws {
        let kvHeads = keyValueHeads ?? heads
        try validate(query: query, queryOffset: queryOffset, key: key, keyOffset: keyOffset,
                     value: value, valueOffset: valueOffset, output: output,
                     outputOffset: outputOffset, heads: heads, queryCount: queryCount,
                     keyCount: keyCount, headDim: headDim, keyValueHeads: kvHeads,
                     causal: causal)
        let halfBytes = MemoryLayout<Float16>.stride
        let scoreScratch = buffers.buffer(
            key: "attention.scores.fp16", bytes: try maximumScoreScratchBytes(keyCount: keyCount))
        let softmax = try context.pipeline(named: "attention_softmax_rows")
        let headRowBytes = headDim * halfBytes
        let scoreRowBytes = keyCount * halfBytes
        let scale = 1 / sqrt(Double(headDim))

        if numerics == .fp32ScoresAndSoftmax {
            try encodeFP32(commandBuffer: commandBuffer, query: query, queryOffset: queryOffset,
                           key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
                           output: output, outputOffset: outputOffset, heads: heads,
                           queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                           kvHeads: kvHeads, causal: causal)
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
                let threads = reductionThreads(limit: softmax.maxTotalThreadsPerThreadgroup)
                encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                encoder.endEncoding()

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
                queryBase += rows
            }
        }
    }

    private func encodeFP32(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer, queryOffset: Int, key: MTLBuffer, keyOffset: Int,
        value: MTLBuffer, valueOffset: Int, output: MTLBuffer, outputOffset: Int,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        kvHeads: Int, causal: Bool
    ) throws {
        let halfBytes = MemoryLayout<Float16>.stride
        let headRowBytes = headDim * halfBytes
        let scoreScratch = buffers.buffer(
            key: "attention.scores.fp32", bytes: try maximumScoreScratchBytes(keyCount: keyCount))
        let qk = try context.pipeline(named: "attention_qk_f16_to_f32")
        let softmax = try context.pipeline(named: "attention_softmax_rows_f32")
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
                queryBase += rows
            }
        }
    }

    func execute(
        query: MTLBuffer, queryOffset: Int = 0,
        key: MTLBuffer, keyOffset: Int = 0,
        value: MTLBuffer, valueOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        heads: Int, queryCount: Int, keyCount: Int, headDim: Int,
        keyValueHeads: Int? = nil,
        causal: Bool = false
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create attention command buffer")
        }
        try encode(commandBuffer: command, query: query, queryOffset: queryOffset,
                   key: key, keyOffset: keyOffset, value: value, valueOffset: valueOffset,
                   output: output, outputOffset: outputOffset, heads: heads,
                   queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                   keyValueHeads: keyValueHeads, causal: causal)
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
        keyValueHeads: Int, causal: Bool
    ) throws {
        guard heads > 0, queryCount > 0, keyCount > 0, headDim > 0,
              keyValueHeads > 0, heads.isMultiple(of: keyValueHeads),
              queryOffset >= 0, keyOffset >= 0, valueOffset >= 0, outputOffset >= 0,
              !causal || queryCount == keyCount else {
            throw AnimapkError.validation("invalid attention shape, offset, or causal dimensions")
        }
        let queryBytes = try checkedProduct(heads, queryCount, headDim, 2)
        let keyBytes = try checkedProduct(keyValueHeads, keyCount, headDim, 2)
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
