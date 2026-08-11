import Foundation
import Metal

/// Streamed production Qwen3-0.6B encoder. The residual remains fp32 while
/// norms, W8 MPS projections, attention, and SwiGLU use explicit fp16 boundaries.
final class QwenEncoderMetal {
    static let maximumTokens = 512
    static let hidden = 1_024
    private static let ropePairs = 64
    private static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: QwenLayerLocator
    private let layer: QwenLayerExecutor
    private let buffers: BufferPool
    private let finalNorm: MTLBuffer

    init(context: MetalContext, file: AnimapkFile) throws {
        let locator = try QwenLayerLocator(file: file)
        guard file.quantGroup == 64 else {
            throw AnimapkError.validation("Qwen Metal encoder requires quant group 64")
        }
        guard let tensor = file.tensor(named: "model.norm.weight"),
              tensor.shape == [Self.hidden], tensor.storage == .fp16,
              tensor.dataSize == UInt64(Self.hidden * 2) else {
            throw AnimapkError.validation("Qwen final RMSNorm is missing or invalid")
        }
        guard let finalNorm = context.device.makeBuffer(
            length: Self.hidden * 2, options: .storageModeShared) else {
            throw AnimapkError.validation("failed to allocate Qwen final norm buffer")
        }
        let normBytes = try file.bytes(in:
            (tensor.blobOffset + tensor.dataOffset)..<
            (tensor.blobOffset + tensor.dataOffset + tensor.dataSize))
        guard let source = normBytes.baseAddress else {
            throw AnimapkError.validation("Qwen final norm has no mapped bytes")
        }
        memcpy(finalNorm.contents(), source, normBytes.count)

        self.context = context
        self.file = file
        self.locator = locator
        self.layer = try QwenLayerExecutor(context: context, file: file, locator: locator)
        self.buffers = BufferPool(device: context.device)
        self.finalNorm = finalNorm
    }

    /// Writes tightly packed fp32 `[tokenIDs.count,1024]` post-final-norm context.
    func execute(
        tokenIDs: [Int], output: MTLBuffer,
        layerCompleted: ((Int, MTLBuffer) throws -> Void)? = nil
    ) async throws {
        guard !tokenIDs.isEmpty, tokenIDs.count <= Self.maximumTokens,
              output.length >= tokenIDs.count * Self.hidden * 4 else {
            throw AnimapkError.validation("invalid Qwen token count or output buffer")
        }
        let count = tokenIDs.count * Self.hidden
        let residual = buffers.buffer(key: "qwen.residual.f32", bytes: count * 4)
        try await gatherEmbedding(tokenIDs, output: residual)
        let rope = makeRope(tokens: tokenIDs.count)
        for logicalIndex in 0..<QwenLayerLocator.layerCount {
            try await layer.execute(
                layerIndex: logicalIndex, residual: residual, rope: rope,
                tokens: tokenIDs.count)
            try layerCompleted?(logicalIndex, residual)
        }
        try await encodeFinalNorm(residual: residual, output: output, tokens: tokenIDs.count)
    }

    private func gatherEmbedding(_ tokenIDs: [Int], output: MTLBuffer) async throws {
        let rows = tokenIDs.count
        let packed = buffers.buffer(key: "qwen.embedding.packed", bytes: rows * Self.hidden)
        let parameters = rows * (Self.hidden / 64) * 2
        let scale = buffers.buffer(key: "qwen.embedding.scale", bytes: parameters)
        let zero = buffers.buffer(key: "qwen.embedding.zero", bytes: parameters)
        for (destinationRow, tokenID) in tokenIDs.enumerated() {
            let spans = try locator.embeddingRow(tokenID)
            try copyEmbeddingSpan(spans.data, to: packed,
                                  destinationOffset: destinationRow * Self.hidden)
            try copyEmbeddingSpan(spans.scale, to: scale,
                                  destinationOffset: destinationRow * (Self.hidden / 64) * 2)
            try copyEmbeddingSpan(spans.zero, to: zero,
                                  destinationOffset: destinationRow * (Self.hidden / 64) * 2)
        }
        let half = buffers.buffer(key: "qwen.embedding.f16", bytes: rows * Self.hidden * 2)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create Qwen embedding command buffer")
        }
        let dequant = try context.pipeline(named: "dequant_w8_to_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen embedding decoder")
        }
        var columns = UInt32(Self.hidden), stride = UInt32(Self.hidden)
        var rowCount = UInt32(rows), outputStride = UInt32(Self.hidden)
        encoder.setComputePipelineState(dequant)
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

    private func copyEmbeddingSpan(
        _ span: AnimapkRelativeSpan, to destination: MTLBuffer, destinationOffset: Int
    ) throws {
        let start = locator.embeddingFileOffset + span.offset
        let source = try file.bytes(in: start..<(start + span.length))
        guard let base = source.baseAddress,
              destinationOffset >= 0,
              destinationOffset + source.count <= destination.length else {
            throw AnimapkError.validation("Qwen embedding staging range is invalid")
        }
        memcpy(destination.contents().advanced(by: destinationOffset), base, source.count)
    }

    private func makeRope(tokens: Int) -> MTLBuffer {
        let values = tokens * Self.ropePairs * 4
        let rope = buffers.buffer(key: "qwen.rope.f32", bytes: values * 4)
        let pointer = rope.contents().bindMemory(to: Float.self, capacity: values)
        for token in 0..<tokens {
            for pair in 0..<Self.ropePairs {
                let inverse = 1 / pow(Float(1_000_000), Float(2 * pair) / 128)
                let angle = Float(token) * inverse
                let cosine = cosf(angle), sine = sinf(angle)
                let base = (token * Self.ropePairs + pair) * 4
                pointer[base] = cosine
                pointer[base + 1] = -sine
                pointer[base + 2] = sine
                pointer[base + 3] = cosine
            }
        }
        return rope
    }

    private func encodeFinalNorm(
        residual: MTLBuffer, output: MTLBuffer, tokens: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create Qwen final-norm command buffer")
        }
        let pipeline = try context.pipeline(named: "rmsnorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen final-norm encoder")
        }
        var columns = UInt32(Self.hidden), epsilon = Self.eps
        var useWeight: UInt32 = 1, rows = UInt32(tokens)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(finalNorm, offset: 0, index: 2)
        encoder.setBytes(&columns, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&useWeight, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        try await commit(command)
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
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
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

private final class QwenLayerExecutor {
    private static let hidden = 1_024
    private static let heads = 16
    private static let kvHeads = 8
    private static let headDim = 128
    private static let queryDim = 2_048
    private static let intermediate = 3_072
    private static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: QwenLayerLocator
    private let streamer: WeightStreamer
    private let buffers: BufferPool
    private let linear: LinearExecutor
    private let attention: AttentionExecutor

    init(context: MetalContext, file: AnimapkFile, locator: QwenLayerLocator) throws {
        guard let maximum = locator.layers.map(\.length).max(), maximum <= UInt64(Int.max) else {
            throw AnimapkError.validation("invalid Qwen layer ranges")
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
        layerIndex: Int, residual: MTLBuffer, rope: MTLBuffer, tokens: Int
    ) async throws {
        guard tokens > 0, tokens <= QwenEncoderMetal.maximumTokens,
              residual.length >= tokens * Self.hidden * 4,
              rope.length >= tokens * 64 * 4 * 4 else {
            throw AnimapkError.validation("invalid Qwen layer input buffers")
        }
        let range = try locator.layer(layerIndex)
        try streamer.load(range, from: file)
        let weights = try QwenLayerWeights(range: range, ring: streamer.ring)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create Qwen layer command buffer")
        }

        let normalized = buffer("qwen.norm.f32", tokens * Self.hidden, Float.self)
        let normHalf = buffer("qwen.norm.f16", tokens * Self.hidden, Float16.self)
        try encodeRMSNorm(command, residual, normalized, weights.inputNorm, tokens)
        try encodeUnary(command, "float_to_half", normalized, normHalf, tokens * Self.hidden)

        let q = buffer("qwen.q.token.f16", tokens * Self.queryDim, Float16.self)
        let k = buffer("qwen.k.token.f16", tokens * Self.hidden, Float16.self)
        let v = buffer("qwen.v.token.f16", tokens * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: normHalf, weight: weights.q,
                          output: q, inputRows: tokens)
        try linear.encode(commandBuffer: command, input: normHalf, weight: weights.k,
                          output: k, inputRows: tokens)
        try linear.encode(commandBuffer: command, input: normHalf, weight: weights.v,
                          output: v, inputRows: tokens)

        let qRope = buffer("qwen.q.rope.f16", tokens * Self.queryDim, Float16.self)
        let kRope = buffer("qwen.k.rope.f16", tokens * Self.hidden, Float16.self)
        try encodeRMSRoPE(command, q, weights.qNorm, rope, qRope, tokens, Self.heads)
        try encodeRMSRoPE(command, k, weights.kNorm, rope, kRope, tokens, Self.kvHeads)
        let qHead = buffer("qwen.q.head.f16", tokens * Self.queryDim, Float16.self)
        let kHead = buffer("qwen.k.head.f16", tokens * Self.hidden, Float16.self)
        let vHead = buffer("qwen.v.head.f16", tokens * Self.hidden, Float16.self)
        try encodeTranspose(command, qRope, qHead, tokens, Self.heads, true)
        try encodeTranspose(command, kRope, kHead, tokens, Self.kvHeads, true)
        try encodeTranspose(command, v, vHead, tokens, Self.kvHeads, true)

        let attendedHead = buffer(
            "qwen.attended.head.f16", tokens * Self.queryDim, Float16.self)
        try attention.encode(
            commandBuffer: command, query: qHead, key: kHead, value: vHead,
            output: attendedHead, heads: Self.heads, queryCount: tokens,
            keyCount: tokens, headDim: Self.headDim, keyValueHeads: Self.kvHeads,
            causal: true)
        let attendedToken = buffer(
            "qwen.attended.token.f16", tokens * Self.queryDim, Float16.self)
        try encodeTranspose(command, attendedHead, attendedToken, tokens, Self.heads, false)
        let attentionProjection = buffer(
            "qwen.attention.projection.f16", tokens * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: attendedToken, weight: weights.o,
                          output: attentionProjection, inputRows: tokens)
        try encodeAddHalf(command, residual, attentionProjection, tokens * Self.hidden)

        try encodeRMSNorm(command, residual, normalized, weights.postAttentionNorm, tokens)
        try encodeUnary(command, "float_to_half", normalized, normHalf, tokens * Self.hidden)
        let gate = buffer("qwen.mlp.gate.f16", tokens * Self.intermediate, Float16.self)
        let up = buffer("qwen.mlp.up.f16", tokens * Self.intermediate, Float16.self)
        let gated = buffer("qwen.mlp.gated.f16", tokens * Self.intermediate, Float16.self)
        try linear.encode(commandBuffer: command, input: normHalf, weight: weights.gate,
                          output: gate, inputRows: tokens)
        try linear.encode(commandBuffer: command, input: normHalf, weight: weights.up,
                          output: up, inputRows: tokens)
        try encodeGatedSiLU(command, gate, up, gated, tokens * Self.intermediate)
        let down = buffer("qwen.mlp.down.f16", tokens * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: gated, weight: weights.down,
                          output: down, inputRows: tokens)
        try encodeAddHalf(command, residual, down, tokens * Self.hidden)
        try await commit(command)
    }

    private func encodeRMSNorm(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ output: MTLBuffer,
        _ weightOffset: Int, _ rows: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rmsnorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen RMSNorm encoder")
        }
        var columns = UInt32(Self.hidden), epsilon = Self.eps
        var useWeight: UInt32 = 1, rowCount = UInt32(rows)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(streamer.ring, offset: weightOffset, index: 2)
        encoder.setBytes(&columns, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&useWeight, length: 4, index: 5)
        encoder.setBytes(&rowCount, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRMSRoPE(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ weightOffset: Int,
        _ rope: MTLBuffer, _ output: MTLBuffer, _ tokens: Int, _ heads: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rms_rope_split_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen Q/K norm-RoPE encoder")
        }
        var tokenCount = UInt32(tokens), headCount = UInt32(heads), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(streamer.ring, offset: weightOffset, index: 1)
        encoder.setBuffer(rope, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&tokenCount, length: 4, index: 4)
        encoder.setBytes(&headCount, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokens * heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeTranspose(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ output: MTLBuffer,
        _ tokens: Int, _ heads: Int, _ toHeadMajor: Bool
    ) throws {
        let pipeline = try context.pipeline(named: "transpose_token_head_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen head transpose encoder")
        }
        var tokenCount = UInt32(tokens), headCount = UInt32(heads)
        var headDim = UInt32(Self.headDim), direction: UInt32 = toHeadMajor ? 1 : 0
        let count = tokens * heads * Self.headDim
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&tokenCount, length: 4, index: 2)
        encoder.setBytes(&headCount, length: 4, index: 3)
        encoder.setBytes(&headDim, length: 4, index: 4)
        encoder.setBytes(&direction, length: 4, index: 5)
        dispatch(encoder, pipeline, count)
        encoder.endEncoding()
    }

    private func encodeGatedSiLU(
        _ command: MTLCommandBuffer, _ gate: MTLBuffer, _ up: MTLBuffer,
        _ output: MTLBuffer, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "gated_silu_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen SwiGLU encoder")
        }
        var elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&elements, length: 4, index: 3)
        dispatch(encoder, pipeline, count)
        encoder.endEncoding()
    }

    private func encodeAddHalf(
        _ command: MTLCommandBuffer, _ residual: MTLBuffer,
        _ branch: MTLBuffer, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_half_into_float")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Qwen residual encoder")
        }
        var elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBytes(&elements, length: 4, index: 2)
        dispatch(encoder, pipeline, count)
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
        var elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&elements, length: 4, index: 2)
        dispatch(encoder, pipeline, count)
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState,
        _ count: Int
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

private struct QwenLayerWeights {
    let q, k, v, o, gate, up, down: QuantizedLinearWeightBuffers
    let inputNorm, postAttentionNorm, qNorm, kNorm: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.layers.\(range.logicalIndex)."
        var spans: [String: AnimapkTensorSpans] = [:]
        for item in range.tensors {
            guard item.tensor.name.hasPrefix(prefix) else {
                throw AnimapkError.validation("foreign tensor in Qwen layer range")
            }
            let name = String(item.tensor.name.dropFirst(prefix.count))
            guard spans.updateValue(item, forKey: name) == nil else {
                throw AnimapkError.validation("duplicate Qwen layer tensor \(name)")
            }
        }
        func matrix(
            _ name: String, _ rows: Int, _ columns: Int
        ) throws -> QuantizedLinearWeightBuffers {
            guard let item = spans[name], item.tensor.shape == [rows, columns],
                  item.tensor.storage == .w8, let scale = item.scale, let zero = item.zero,
                  item.data.length % UInt64(rows) == 0,
                  item.data.offset <= UInt64(Int.max), scale.offset <= UInt64(Int.max),
                  zero.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid Qwen matrix \(prefix)\(name)")
            }
            return QuantizedLinearWeightBuffers(
                storage: .w8, packed: ring, packedOffset: Int(item.data.offset),
                scale: ring, scaleOffset: Int(scale.offset),
                zero: ring, zeroOffset: Int(zero.offset), rows: rows, columns: columns,
                packedRowStride: Int(item.data.length / UInt64(rows)))
        }
        func norm(_ name: String, _ count: Int) throws -> Int {
            guard let item = spans[name], item.tensor.shape == [count],
                  item.tensor.storage == .fp16, item.data.length == UInt64(count * 2),
                  item.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid Qwen norm \(prefix)\(name)")
            }
            return Int(item.data.offset)
        }
        inputNorm = try norm("input_layernorm.weight", 1_024)
        postAttentionNorm = try norm("post_attention_layernorm.weight", 1_024)
        qNorm = try norm("self_attn.q_norm.weight", 128)
        kNorm = try norm("self_attn.k_norm.weight", 128)
        q = try matrix("self_attn.q_proj.weight", 2_048, 1_024)
        k = try matrix("self_attn.k_proj.weight", 1_024, 1_024)
        v = try matrix("self_attn.v_proj.weight", 1_024, 1_024)
        o = try matrix("self_attn.o_proj.weight", 1_024, 2_048)
        gate = try matrix("mlp.gate_proj.weight", 3_072, 1_024)
        up = try matrix("mlp.up_proj.weight", 3_072, 1_024)
        down = try matrix("mlp.down_proj.weight", 1_024, 3_072)
        guard spans.count == 11 else {
            throw AnimapkError.validation("Qwen layer must contain exactly 11 tensors")
        }
    }
}
