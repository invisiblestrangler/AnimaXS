import Foundation
import Metal

/// Bounded-memory production lllite adapter. The six W4 blocks are streamed
/// one at a time; residual arithmetic is fp32 and MPS projection boundaries fp16.
final class LLMAdapterMetal {
    static let maximumTokens = 512
    static let hidden = 1_024
    private static let heads = 16
    private static let headDim = 64
    private static let ropePairs = 32
    private static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: LLMAdapterLocator
    private let layer: LLMAdapterLayerExecutor
    private let finalStreamer: WeightStreamer
    private let linear: LinearExecutor
    private let buffers: BufferPool

    init(context: MetalContext, file: AnimapkFile) throws {
        let locator = try LLMAdapterLocator(file: file)
        guard file.quantGroup == 64 else {
            throw AnimapkError.validation("adapter Metal executor requires quant group 64")
        }
        self.context = context
        self.file = file
        self.locator = locator
        self.layer = try LLMAdapterLayerExecutor(context: context, file: file, locator: locator)
        self.finalStreamer = try WeightStreamer(
            device: context.device, capacity: Int(locator.final.length))
        self.linear = LinearExecutor(context: context)
        self.buffers = BufferPool(device: context.device)
    }

    /// Executes the adapter and writes fp32 `[512,1024]`; rows after `t5IDs.count`
    /// are exactly zero. Qwen context is tightly packed fp32 `[contextTokens,1024]`.
    func execute(
        qwenContext: MTLBuffer, contextTokens: Int,
        t5IDs: [Int], t5Weights: [Float], output: MTLBuffer,
        layerCompleted: ((Int, MTLBuffer) throws -> Void)? = nil
    ) async throws {
        let targetTokens = t5IDs.count
        guard contextTokens > 0, contextTokens <= Self.maximumTokens,
              targetTokens > 0, targetTokens <= Self.maximumTokens,
              t5Weights.count == targetTokens,
              qwenContext.length >= contextTokens * Self.hidden * 4,
              output.length >= Self.maximumTokens * Self.hidden * 4,
              t5IDs.allSatisfy({ $0 >= 0 }) else {
            throw AnimapkError.validation("invalid adapter input shape or buffer")
        }

        let residual = buffers.buffer(
            key: "adapter.residual.f32", bytes: targetTokens * Self.hidden * 4)
        try await gatherEmbedding(t5IDs, output: residual)
        let contextHalf = buffers.buffer(
            key: "adapter.context.f16", bytes: contextTokens * Self.hidden * 2)
        try await convertContext(qwenContext, output: contextHalf,
                                 count: contextTokens * Self.hidden)
        let targetRope = makeRope(tokens: targetTokens, key: "adapter.target.rope")
        let contextRope = makeRope(tokens: contextTokens, key: "adapter.context.rope")
        for index in 0..<LLMAdapterLocator.blockCount {
            try await layer.execute(
                layerIndex: index, residual: residual, targetRope: targetRope,
                targetTokens: targetTokens, context: contextHalf,
                contextRope: contextRope, contextTokens: contextTokens)
            try layerCompleted?(index, residual)
        }
        try await executeFinal(
            residual: residual, targetTokens: targetTokens,
            tokenWeights: t5Weights, output: output)
    }

    private func gatherEmbedding(_ ids: [Int], output: MTLBuffer) async throws {
        let rows = ids.count
        let packed = buffers.buffer(key: "adapter.embedding.packed", bytes: rows * 512)
        let parameterBytes = rows * 16 * 2
        let scale = buffers.buffer(key: "adapter.embedding.scale", bytes: parameterBytes)
        let zero = buffers.buffer(key: "adapter.embedding.zero", bytes: parameterBytes)
        for (destinationRow, id) in ids.enumerated() {
            let spans = try locator.embeddingRow(id)
            try copy(spans.data, to: packed, destinationOffset: destinationRow * 512)
            try copy(spans.scale, to: scale, destinationOffset: destinationRow * 32)
            try copy(spans.zero, to: zero, destinationOffset: destinationRow * 32)
        }
        let half = buffers.buffer(key: "adapter.embedding.f16", bytes: rows * Self.hidden * 2)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create adapter embedding command buffer")
        }
        let pipeline = try context.pipeline(named: "dequant_w4_to_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter embedding decoder")
        }
        var columns = UInt32(Self.hidden), stride = UInt32(512), rowCount = UInt32(rows)
        var outputStride = UInt32(Self.hidden)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packed, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(zero, offset: 0, index: 2)
        encoder.setBuffer(half, offset: 0, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&stride, length: 4, index: 5)
        encoder.setBytes(&rowCount, length: 4, index: 6)
        encoder.setBytes(&outputStride, length: 4, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: Self.hidden, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        try encodeUnary(command, "half_to_float", half, output, rows * Self.hidden)
        try await commit(command)
    }

    private func copy(
        _ span: AnimapkRelativeSpan, to destination: MTLBuffer, destinationOffset: Int
    ) throws {
        let start = locator.embeddingFileOffset + span.offset
        let source = try file.bytes(in: start..<(start + span.length))
        guard let base = source.baseAddress,
              destinationOffset >= 0, destinationOffset + source.count <= destination.length else {
            throw AnimapkError.validation("adapter embedding staging range is invalid")
        }
        memcpy(destination.contents().advanced(by: destinationOffset), base, source.count)
    }

    private func convertContext(_ input: MTLBuffer, output: MTLBuffer, count: Int) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create adapter context command buffer")
        }
        try encodeUnary(command, "float_to_half", input, output, count)
        try await commit(command)
    }

    private func makeRope(tokens: Int, key: String) -> MTLBuffer {
        let valueCount = tokens * Self.ropePairs * 4
        let rope = buffers.buffer(key: key, bytes: valueCount * 4)
        let pointer = rope.contents().bindMemory(to: Float.self, capacity: valueCount)
        for token in 0..<tokens {
            for pair in 0..<Self.ropePairs {
                let inverse = 1 / pow(Float(10_000), Float(2 * pair) / Float(Self.headDim))
                let angle = Float(token) * inverse
                let base = (token * Self.ropePairs + pair) * 4
                pointer[base] = cosf(angle)
                pointer[base + 1] = -sinf(angle)
                pointer[base + 2] = sinf(angle)
                pointer[base + 3] = cosf(angle)
            }
        }
        return rope
    }

    private func executeFinal(
        residual: MTLBuffer, targetTokens: Int, tokenWeights: [Float], output: MTLBuffer
    ) async throws {
        try finalStreamer.load(locator.final, from: file)
        let weights = try LLMAdapterFinalWeights(range: locator.final, ring: finalStreamer.ring)
        let inputHalf = buffers.buffer(
            key: "adapter.final.input.f16", bytes: targetTokens * Self.hidden * 2)
        let projected = buffers.buffer(
            key: "adapter.final.projected.f16", bytes: targetTokens * Self.hidden * 2)
        let weightBuffer = buffers.buffer(
            key: "adapter.final.token_weights.f32", bytes: targetTokens * 4)
        tokenWeights.withUnsafeBytes { bytes in
            memcpy(weightBuffer.contents(), bytes.baseAddress!, bytes.count)
        }
        memset(output.contents(), 0, Self.maximumTokens * Self.hidden * 4)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create adapter final command buffer")
        }
        try encodeUnary(command, "float_to_half", residual, inputHalf,
                        targetTokens * Self.hidden)
        try linear.encode(commandBuffer: command, input: inputHalf, weight: weights.projection,
                          output: projected, inputRows: targetTokens)
        try encodeBias(command, values: projected, biasOffset: weights.bias,
                       columns: Self.hidden, count: targetTokens * Self.hidden)
        try encodeFinalNorm(command, projected, normOffset: weights.norm,
                            tokenWeights: weightBuffer, output: output, rows: targetTokens)
        try await commit(command)
    }

    private func encodeBias(
        _ command: MTLCommandBuffer, values: MTLBuffer, biasOffset: Int,
        columns: Int, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_bias_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter final bias encoder")
        }
        var columns = UInt32(columns), elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(finalStreamer.ring, offset: biasOffset, index: 1)
        encoder.setBytes(&columns, length: 4, index: 2)
        encoder.setBytes(&elements, length: 4, index: 3)
        dispatch(encoder, pipeline, count)
        encoder.endEncoding()
    }

    private func encodeFinalNorm(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, normOffset: Int,
        tokenWeights: MTLBuffer, output: MTLBuffer, rows: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rmsnorm_half_to_weighted_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter final norm encoder")
        }
        var rows = UInt32(rows), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(finalStreamer.ring, offset: normOffset, index: 1)
        encoder.setBuffer(tokenWeights, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&epsilon, length: 4, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: Int(rows), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeUnary(
        _ command: MTLCommandBuffer, _ name: String, _ input: MTLBuffer,
        _ output: MTLBuffer, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: name)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(name) encoder")
        }
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        dispatch(encoder, pipeline, Int(count))
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ count: Int
    ) {
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func commit(_ command: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }
}

private final class LLMAdapterLayerExecutor {
    private static let hidden = 1_024
    private static let heads = 16
    private static let headDim = 64
    private static let mlp = 4_096
    private static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: LLMAdapterLocator
    private let streamer: WeightStreamer
    private let buffers: BufferPool
    private let linear: LinearExecutor
    private let attention: AttentionExecutor

    init(context: MetalContext, file: AnimapkFile, locator: LLMAdapterLocator) throws {
        guard let maximum = locator.blocks.map(\.length).max(), maximum <= UInt64(Int.max) else {
            throw AnimapkError.validation("invalid adapter block ranges")
        }
        self.context = context
        self.file = file
        self.locator = locator
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(maximum))
        self.buffers = BufferPool(device: context.device)
        self.linear = LinearExecutor(context: context)
        self.attention = AttentionExecutor(context: context)
    }

    func execute(
        layerIndex: Int, residual: MTLBuffer, targetRope: MTLBuffer,
        targetTokens: Int, context: MTLBuffer, contextRope: MTLBuffer,
        contextTokens: Int
    ) async throws {
        guard residual.length >= targetTokens * Self.hidden * 4,
              context.length >= contextTokens * Self.hidden * 2,
              targetRope.length >= targetTokens * 32 * 4 * 4,
              contextRope.length >= contextTokens * 32 * 4 * 4 else {
            throw AnimapkError.validation("invalid adapter block input buffers")
        }
        let range = try locator.block(layerIndex)
        try streamer.load(range, from: file)
        let weights = try LLMAdapterBlockWeights(range: range, ring: streamer.ring)
        guard let command = self.context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create adapter block command buffer")
        }

        try encodeAttention(
            command, residual: residual, normOffset: weights.selfNorm,
            qWeight: weights.selfQ, kWeight: weights.selfK, vWeight: weights.selfV,
            oWeight: weights.selfO, qNorm: weights.selfQNorm, kNorm: weights.selfKNorm,
            keyValueInput: nil, queryRope: targetRope, keyRope: targetRope,
            queryCount: targetTokens, keyCount: targetTokens)
        try encodeAttention(
            command, residual: residual, normOffset: weights.crossNorm,
            qWeight: weights.crossQ, kWeight: weights.crossK, vWeight: weights.crossV,
            oWeight: weights.crossO, qNorm: weights.crossQNorm, kNorm: weights.crossKNorm,
            keyValueInput: context, queryRope: targetRope, keyRope: contextRope,
            queryCount: targetTokens, keyCount: contextTokens)

        let normalized = buffer("adapter.norm.f32", targetTokens * Self.hidden, Float.self)
        let normalizedHalf = buffer("adapter.norm.f16", targetTokens * Self.hidden, Float16.self)
        try encodeRMSNorm(command, residual, normalized, weights.mlpNorm, targetTokens)
        try encodeUnary(command, "float_to_half", normalized, normalizedHalf,
                        targetTokens * Self.hidden)
        let hidden = buffer("adapter.mlp.hidden.f16", targetTokens * Self.mlp, Float16.self)
        try linear.encode(commandBuffer: command, input: normalizedHalf,
                          weight: weights.mlpIn, output: hidden, inputRows: targetTokens)
        try encodeBiasGELU(command, hidden, weights.mlpInBias,
                           columns: Self.mlp, count: targetTokens * Self.mlp)
        let branch = buffer("adapter.mlp.out.f16", targetTokens * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: hidden,
                          weight: weights.mlpOut, output: branch, inputRows: targetTokens)
        try encodeBiasAdd(command, residual, branch, weights.mlpOutBias,
                          columns: Self.hidden, count: targetTokens * Self.hidden)
        try await commit(command)
    }

    private func encodeAttention(
        _ command: MTLCommandBuffer, residual: MTLBuffer, normOffset: Int,
        qWeight: QuantizedLinearWeightBuffers, kWeight: QuantizedLinearWeightBuffers,
        vWeight: QuantizedLinearWeightBuffers, oWeight: QuantizedLinearWeightBuffers,
        qNorm: Int, kNorm: Int, keyValueInput: MTLBuffer?,
        queryRope: MTLBuffer, keyRope: MTLBuffer,
        queryCount: Int, keyCount: Int
    ) throws {
        let normalized = buffer("adapter.norm.f32", queryCount * Self.hidden, Float.self)
        let queryInput = buffer("adapter.norm.f16", queryCount * Self.hidden, Float16.self)
        try encodeRMSNorm(command, residual, normalized, normOffset, queryCount)
        try encodeUnary(command, "float_to_half", normalized, queryInput, queryCount * Self.hidden)
        let kvInput = keyValueInput ?? queryInput
        let q = buffer("adapter.q.token.f16", queryCount * Self.hidden, Float16.self)
        let k = buffer("adapter.k.token.f16", keyCount * Self.hidden, Float16.self)
        let v = buffer("adapter.v.token.f16", keyCount * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: queryInput, weight: qWeight,
                          output: q, inputRows: queryCount)
        try linear.encode(commandBuffer: command, input: kvInput, weight: kWeight,
                          output: k, inputRows: keyCount)
        try linear.encode(commandBuffer: command, input: kvInput, weight: vWeight,
                          output: v, inputRows: keyCount)
        let qr = buffer("adapter.q.rope.f16", queryCount * Self.hidden, Float16.self)
        let kr = buffer("adapter.k.rope.f16", keyCount * Self.hidden, Float16.self)
        try encodeRMSRoPE(command, q, qNorm, queryRope, qr, queryCount)
        try encodeRMSRoPE(command, k, kNorm, keyRope, kr, keyCount)
        let qh = buffer("adapter.q.head.f16", queryCount * Self.hidden, Float16.self)
        let kh = buffer("adapter.k.head.f16", keyCount * Self.hidden, Float16.self)
        let vh = buffer("adapter.v.head.f16", keyCount * Self.hidden, Float16.self)
        try encodeTranspose(command, qr, qh, queryCount, true)
        try encodeTranspose(command, kr, kh, keyCount, true)
        try encodeTranspose(command, v, vh, keyCount, true)
        let attended = buffer("adapter.attended.head.f16", queryCount * Self.hidden, Float16.self)
        try attention.encode(
            commandBuffer: command, query: qh, key: kh, value: vh, output: attended,
            heads: Self.heads, queryCount: queryCount, keyCount: keyCount,
            headDim: Self.headDim, causal: false)
        let attendedToken = buffer(
            "adapter.attended.token.f16", queryCount * Self.hidden, Float16.self)
        try encodeTranspose(command, attended, attendedToken, queryCount, false)
        let projected = buffer(
            "adapter.attention.out.f16", queryCount * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: attendedToken,
                          weight: oWeight, output: projected, inputRows: queryCount)
        try encodeAddHalf(command, residual, projected, queryCount * Self.hidden)
    }

    private func encodeRMSNorm(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ output: MTLBuffer,
        _ weightOffset: Int, _ rows: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rmsnorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter RMSNorm encoder")
        }
        var columns = UInt32(Self.hidden), epsilon = Self.eps
        var useWeight: UInt32 = 1, rows = UInt32(rows)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(streamer.ring, offset: weightOffset, index: 2)
        encoder.setBytes(&columns, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&useWeight, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: Int(rows), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRMSRoPE(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ normOffset: Int,
        _ rope: MTLBuffer, _ output: MTLBuffer, _ tokens: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rms_rope_adapter64")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter Q/K norm-RoPE encoder")
        }
        var tokens = UInt32(tokens), heads = UInt32(Self.heads), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(streamer.ring, offset: normOffset, index: 1)
        encoder.setBuffer(rope, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&tokens, length: 4, index: 4)
        encoder.setBytes(&heads, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: Int(tokens * heads), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeTranspose(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ output: MTLBuffer,
        _ tokens: Int, _ toHeadMajor: Bool
    ) throws {
        let pipeline = try context.pipeline(named: "transpose_token_head_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter transpose encoder")
        }
        var tokens = UInt32(tokens), heads = UInt32(Self.heads)
        var headDim = UInt32(Self.headDim), direction: UInt32 = toHeadMajor ? 1 : 0
        let count = tokens * heads * headDim
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&tokens, length: 4, index: 2)
        encoder.setBytes(&heads, length: 4, index: 3)
        encoder.setBytes(&headDim, length: 4, index: 4)
        encoder.setBytes(&direction, length: 4, index: 5)
        dispatch(encoder, pipeline, Int(count))
        encoder.endEncoding()
    }

    private func encodeBiasGELU(
        _ command: MTLCommandBuffer, _ values: MTLBuffer, _ biasOffset: Int,
        columns: Int, count: Int
    ) throws { try encodeBiasKernel(command, "bias_gelu_half", values, nil, biasOffset, columns, count) }

    private func encodeBiasAdd(
        _ command: MTLCommandBuffer, _ residual: MTLBuffer, _ branch: MTLBuffer,
        _ biasOffset: Int, columns: Int, count: Int
    ) throws { try encodeBiasKernel(command, "add_bias_half_into_float", residual, branch, biasOffset, columns, count) }

    private func encodeBiasKernel(
        _ command: MTLCommandBuffer, _ name: String, _ first: MTLBuffer,
        _ second: MTLBuffer?, _ biasOffset: Int, _ columns: Int, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: name)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter \(name) encoder")
        }
        var columns = UInt32(columns), elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(first, offset: 0, index: 0)
        if let second {
            encoder.setBuffer(second, offset: 0, index: 1)
            encoder.setBuffer(streamer.ring, offset: biasOffset, index: 2)
            encoder.setBytes(&columns, length: 4, index: 3)
            encoder.setBytes(&elements, length: 4, index: 4)
        } else {
            encoder.setBuffer(streamer.ring, offset: biasOffset, index: 1)
            encoder.setBytes(&columns, length: 4, index: 2)
            encoder.setBytes(&elements, length: 4, index: 3)
        }
        dispatch(encoder, pipeline, count)
        encoder.endEncoding()
    }

    private func encodeAddHalf(
        _ command: MTLCommandBuffer, _ residual: MTLBuffer, _ branch: MTLBuffer, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_half_into_float")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create adapter residual encoder")
        }
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        dispatch(encoder, pipeline, Int(count))
        encoder.endEncoding()
    }

    private func encodeUnary(
        _ command: MTLCommandBuffer, _ name: String, _ input: MTLBuffer,
        _ output: MTLBuffer, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: name)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(name) encoder")
        }
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        dispatch(encoder, pipeline, Int(count))
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ count: Int
    ) {
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func buffer<T>(_ key: String, _ count: Int, _: T.Type) -> MTLBuffer {
        buffers.buffer(key: key, bytes: count * MemoryLayout<T>.stride)
    }

    private func commit(_ command: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }
}

private struct LLMAdapterBlockWeights {
    let selfQ, selfK, selfV, selfO: QuantizedLinearWeightBuffers
    let crossQ, crossK, crossV, crossO: QuantizedLinearWeightBuffers
    let mlpIn, mlpOut: QuantizedLinearWeightBuffers
    let selfNorm, crossNorm, mlpNorm: Int
    let selfQNorm, selfKNorm, crossQNorm, crossKNorm: Int
    let mlpInBias, mlpOutBias: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model.llm_adapter.blocks.\(range.logicalIndex)."
        var spans: [String: AnimapkTensorSpans] = [:]
        for item in range.tensors {
            guard item.tensor.name.hasPrefix(prefix) else {
                throw AnimapkError.validation("foreign tensor in adapter block range")
            }
            spans[String(item.tensor.name.dropFirst(prefix.count))] = item
        }
        func matrix(_ name: String, _ rows: Int, _ columns: Int) throws -> QuantizedLinearWeightBuffers {
            guard let item = spans[name], item.tensor.shape == [rows, columns],
                  item.tensor.storage == .w4, let scale = item.scale, let zero = item.zero,
                  item.data.length % UInt64(rows) == 0,
                  item.data.offset <= UInt64(Int.max), scale.offset <= UInt64(Int.max),
                  zero.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid adapter matrix \(prefix)\(name)")
            }
            return QuantizedLinearWeightBuffers(
                storage: .w4, packed: ring, packedOffset: Int(item.data.offset),
                scale: ring, scaleOffset: Int(scale.offset), zero: ring,
                zeroOffset: Int(zero.offset), rows: rows, columns: columns,
                packedRowStride: Int(item.data.length / UInt64(rows)))
        }
        func vector(_ name: String, _ count: Int) throws -> Int {
            guard let item = spans[name], item.tensor.shape == [count],
                  item.tensor.storage == .fp16, item.data.length == UInt64(count * 2),
                  item.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid adapter vector \(prefix)\(name)")
            }
            return Int(item.data.offset)
        }
        selfQ = try matrix("self_attn.q_proj.weight", 1_024, 1_024)
        selfK = try matrix("self_attn.k_proj.weight", 1_024, 1_024)
        selfV = try matrix("self_attn.v_proj.weight", 1_024, 1_024)
        selfO = try matrix("self_attn.o_proj.weight", 1_024, 1_024)
        crossQ = try matrix("cross_attn.q_proj.weight", 1_024, 1_024)
        crossK = try matrix("cross_attn.k_proj.weight", 1_024, 1_024)
        crossV = try matrix("cross_attn.v_proj.weight", 1_024, 1_024)
        crossO = try matrix("cross_attn.o_proj.weight", 1_024, 1_024)
        mlpIn = try matrix("mlp.0.weight", 4_096, 1_024)
        mlpOut = try matrix("mlp.2.weight", 1_024, 4_096)
        selfNorm = try vector("norm_self_attn.weight", 1_024)
        crossNorm = try vector("norm_cross_attn.weight", 1_024)
        mlpNorm = try vector("norm_mlp.weight", 1_024)
        selfQNorm = try vector("self_attn.q_norm.weight", 64)
        selfKNorm = try vector("self_attn.k_norm.weight", 64)
        crossQNorm = try vector("cross_attn.q_norm.weight", 64)
        crossKNorm = try vector("cross_attn.k_norm.weight", 64)
        mlpInBias = try vector("mlp.0.bias", 4_096)
        mlpOutBias = try vector("mlp.2.bias", 1_024)
        guard spans.count == 19 else {
            throw AnimapkError.validation("adapter block must contain exactly 19 tensors")
        }
    }
}

private struct LLMAdapterFinalWeights {
    let projection: QuantizedLinearWeightBuffers
    let bias, norm: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model.llm_adapter."
        let spans = Dictionary(uniqueKeysWithValues: range.tensors.map {
            (String($0.tensor.name.dropFirst(prefix.count)), $0)
        })
        guard let item = spans["out_proj.weight"], item.tensor.shape == [1_024, 1_024],
              item.tensor.storage == .w4, let scale = item.scale, let zero = item.zero,
              item.data.offset <= UInt64(Int.max), scale.offset <= UInt64(Int.max),
              zero.offset <= UInt64(Int.max) else {
            throw AnimapkError.validation("invalid adapter final projection")
        }
        projection = QuantizedLinearWeightBuffers(
            storage: .w4, packed: ring, packedOffset: Int(item.data.offset),
            scale: ring, scaleOffset: Int(scale.offset), zero: ring,
            zeroOffset: Int(zero.offset), rows: 1_024, columns: 1_024,
            packedRowStride: Int(item.data.length / 1_024))
        func vector(_ name: String) throws -> Int {
            guard let value = spans[name], value.tensor.shape == [1_024],
                  value.tensor.storage == .fp16, value.data.length == 2_048,
                  value.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid adapter final \(name)")
            }
            return Int(value.data.offset)
        }
        bias = try vector("out_proj.bias")
        norm = try vector("norm.weight")
        guard spans.count == 3 else {
            throw AnimapkError.validation("adapter final range must contain exactly 3 tensors")
        }
    }
}
