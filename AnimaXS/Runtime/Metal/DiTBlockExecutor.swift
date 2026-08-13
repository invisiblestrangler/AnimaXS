import Foundation
import Metal

/// Exact fixed-shape MiniTrainDIT block executed with one streamed block range.
/// Residual/modulation/gates remain fp32; projection and attention boundaries are fp16.
/// This object is intentionally non-reentrant because every activation and weight scratch
/// allocation is reused between operations and calls.
final class DiTBlockExecutor {
    static let tokens = 1_024
    static let contextTokens = 512
    static let dim = 2_048
    static let contextDim = 1_024
    static let heads = 16
    static let headDim = 128
    static let hidden = 8_192
    static let modulationHidden = 256
    static let modulationSize = 6_144
    static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: DiTBlockLocator
    private let streamer: WeightStreamer
    private let buffers: BufferPool
    private let linear: LinearExecutor
    private let attention: AttentionExecutor

    init(context: MetalContext, file: AnimapkFile,
         attentionNumerics: AttentionNumerics = .legacy) throws {
        let locator = try DiTBlockLocator(file: file)
        guard let maximum = locator.blocks.map(\.length).max(), maximum <= UInt64(Int.max) else {
            throw AnimapkError.validation("invalid DiT execution ranges")
        }
        self.context = context
        self.file = file
        self.locator = locator
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(maximum))
        self.buffers = BufferPool(device: context.device)
        self.linear = LinearExecutor(context: context)
        self.attention = AttentionExecutor(context: context, numerics: attentionNumerics)
    }

    /// Mutates `residual` in place. All input buffers are tightly packed and use these types:
    /// residual/emb/adalnLora/rope fp32, crossContext fp16.
    func execute(
        blockIndex: Int,
        residual: MTLBuffer,
        emb: MTLBuffer,
        adalnLora: MTLBuffer,
        crossContext: MTLBuffer,
        rope: MTLBuffer
    ) async throws {
        try validateInputs(residual: residual, emb: emb, adalnLora: adalnLora,
                           crossContext: crossContext, rope: rope)
        let range = try locator.block(blockIndex)
        try streamer.load(range, from: file)
        let weights = try BlockWeights(range: range, ring: streamer.ring)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create DiT block command buffer")
        }

        let siluEmb = buffer("dit.siluEmb.f32", Self.dim, Float.self)
        try encodeUnary(command, kernel: "silu", input: emb, output: siluEmb, count: Self.dim)
        try encodeAttentionBranch(command, residual: residual, crossContext: crossContext,
                                  rope: rope, siluEmb: siluEmb, adalnLora: adalnLora,
                                  weights: weights, cross: false)
        try encodeAttentionBranch(command, residual: residual, crossContext: crossContext,
                                  rope: rope, siluEmb: siluEmb, adalnLora: adalnLora,
                                  weights: weights, cross: true)
        try encodeMLP(command, residual: residual, siluEmb: siluEmb,
                      adalnLora: adalnLora, weights: weights)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }

    private func encodeAttentionBranch(
        _ command: MTLCommandBuffer, residual: MTLBuffer, crossContext: MTLBuffer,
        rope: MTLBuffer, siluEmb: MTLBuffer, adalnLora: MTLBuffer,
        weights: BlockWeights, cross: Bool
    ) throws {
        let modulation = try encodeModulation(
            command, siluEmb: siluEmb, adalnLora: adalnLora,
            w1: cross ? weights.modCross1 : weights.modSelf1,
            w2: cross ? weights.modCross2 : weights.modSelf2)
        let norm = buffer("dit.norm.f32", Self.tokens * Self.dim, Float.self)
        let modulated = buffer("dit.modulated.f32", Self.tokens * Self.dim, Float.self)
        let projectionInput = buffer("dit.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        try encodeLayerNorm(command, input: residual, output: norm, rows: Self.tokens, columns: Self.dim)
        try encodeModulate(command, normalized: norm, modulation: modulation,
                           output: modulated, count: Self.tokens * Self.dim)
        try encodeConvert(command, kernel: "float_to_half", input: modulated,
                          output: projectionInput, count: Self.tokens * Self.dim)

        let qToken = buffer("dit.q.token.f16", Self.tokens * Self.dim, Float16.self)
        let qHead = buffer("dit.q.head.f16", Self.tokens * Self.dim, Float16.self)
        let kTokenCount = (cross ? Self.contextTokens : Self.tokens) * Self.dim
        let kToken = buffer("dit.k.token.f16", kTokenCount, Float16.self)
        let kHead = buffer("dit.k.head.f16", kTokenCount, Float16.self)
        let vToken = buffer("dit.v.token.f16", kTokenCount, Float16.self)
        let vHead = buffer("dit.v.head.f16", kTokenCount, Float16.self)
        let queryWeight = cross ? weights.crossQ : weights.selfQ
        let keyWeight = cross ? weights.crossK : weights.selfK
        let valueWeight = cross ? weights.crossV : weights.selfV
        try linear.encode(commandBuffer: command, input: projectionInput,
                          weight: queryWeight, output: qToken, inputRows: Self.tokens)
        let keyInput = cross ? crossContext : projectionInput
        let keyRows = cross ? Self.contextTokens : Self.tokens
        try linear.encode(commandBuffer: command, input: keyInput,
                          weight: keyWeight, output: kToken, inputRows: keyRows)
        try linear.encode(commandBuffer: command, input: keyInput,
                          weight: valueWeight, output: vToken, inputRows: keyRows)

        if cross {
            try encodeRMSHeads(command, input: qToken, weightOffset: weights.crossQNorm,
                               output: qToken, rows: Self.tokens * Self.heads)
            try encodeRMSHeads(command, input: kToken, weightOffset: weights.crossKNorm,
                               output: kToken, rows: Self.contextTokens * Self.heads)
        } else {
            try encodeRMSRoPE(command, input: qToken, weightOffset: weights.selfQNorm,
                              rope: rope, output: qToken)
            try encodeRMSRoPE(command, input: kToken, weightOffset: weights.selfKNorm,
                              rope: rope, output: kToken)
        }
        try encodeTranspose(command, input: qToken, output: qHead,
                            tokens: Self.tokens, toHeadMajor: true)
        try encodeTranspose(command, input: kToken, output: kHead,
                            tokens: keyRows, toHeadMajor: true)
        try encodeTranspose(command, input: vToken, output: vHead,
                            tokens: keyRows, toHeadMajor: true)
        let attendedHead = buffer("dit.attended.head.f16", Self.tokens * Self.dim, Float16.self)
        let attendedToken = buffer("dit.attended.token.f16", Self.tokens * Self.dim, Float16.self)
        try attention.encode(commandBuffer: command, query: qHead, key: kHead, value: vHead,
                             output: attendedHead, heads: Self.heads, queryCount: Self.tokens,
                             keyCount: keyRows, headDim: Self.headDim)
        try encodeTranspose(command, input: attendedHead, output: attendedToken,
                            tokens: Self.tokens, toHeadMajor: false)
        let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try linear.encode(commandBuffer: command, input: attendedToken,
                          weight: cross ? weights.crossO : weights.selfO,
                          output: branch, inputRows: Self.tokens)
        try encodeGateAdd(command, residual: residual, branch: branch, modulation: modulation,
                          count: Self.tokens * Self.dim)
    }

    private func encodeMLP(
        _ command: MTLCommandBuffer, residual: MTLBuffer, siluEmb: MTLBuffer,
        adalnLora: MTLBuffer, weights: BlockWeights
    ) throws {
        let modulation = try encodeModulation(command, siluEmb: siluEmb, adalnLora: adalnLora,
                                              w1: weights.modMLP1, w2: weights.modMLP2)
        let norm = buffer("dit.norm.f32", Self.tokens * Self.dim, Float.self)
        let modulated = buffer("dit.modulated.f32", Self.tokens * Self.dim, Float.self)
        let projectionInput = buffer("dit.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        try encodeLayerNorm(command, input: residual, output: norm, rows: Self.tokens, columns: Self.dim)
        try encodeModulate(command, normalized: norm, modulation: modulation,
                           output: modulated, count: Self.tokens * Self.dim)
        try encodeConvert(command, kernel: "float_to_half", input: modulated,
                          output: projectionInput, count: Self.tokens * Self.dim)
        let hiddenHalf = buffer("dit.hidden.f16", Self.tokens * Self.hidden, Float16.self)
        let hiddenFloat = buffer("dit.hidden.f32", Self.tokens * Self.hidden, Float.self)
        try linear.encode(commandBuffer: command, input: projectionInput,
                          weight: weights.mlp1, output: hiddenHalf, inputRows: Self.tokens)
        try encodeConvert(command, kernel: "half_to_float", input: hiddenHalf,
                          output: hiddenFloat, count: Self.tokens * Self.hidden)
        try encodeUnary(command, kernel: "gelu", input: hiddenFloat,
                        output: hiddenFloat, count: Self.tokens * Self.hidden)
        try encodeConvert(command, kernel: "float_to_half", input: hiddenFloat,
                          output: hiddenHalf, count: Self.tokens * Self.hidden)
        let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try linear.encode(commandBuffer: command, input: hiddenHalf,
                          weight: weights.mlp2, output: branch, inputRows: Self.tokens)
        try encodeGateAdd(command, residual: residual, branch: branch, modulation: modulation,
                          count: Self.tokens * Self.dim)
    }

    private func encodeModulation(
        _ command: MTLCommandBuffer, siluEmb: MTLBuffer, adalnLora: MTLBuffer,
        w1: QuantizedLinearWeightBuffers, w2: QuantizedLinearWeightBuffers
    ) throws -> MTLBuffer {
        let hidden = buffer("dit.modulation.hidden.f32", Self.modulationHidden, Float.self)
        let output = buffer("dit.modulation.f32", Self.modulationSize, Float.self)
        try encodeMatvec(command, input: siluEmb, weight: w1, output: hidden)
        try encodeMatvec(command, input: hidden, weight: w2, output: output)
        try encodeBinary(command, kernel: "add_f32", destination: output,
                         source: adalnLora, count: Self.modulationSize)
        return output
    }

    private func encodeMatvec(
        _ command: MTLCommandBuffer, input: MTLBuffer,
        weight: QuantizedLinearWeightBuffers, output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(
            named: DiTQuantizedWeightFactory.matvecKernel(for: weight.storage))
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create modulation matvec encoder")
        }
        var columns = UInt32(weight.columns), rows = UInt32(weight.rows)
        var rowStride = UInt32(weight.packedRowStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 0)
        encoder.setBuffer(weight.scale, offset: weight.scaleOffset, index: 1)
        encoder.setBuffer(weight.zero, offset: weight.zeroOffset, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&columns, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBytes(&rowStride, length: 4, index: 7)
        let threads = reductionThreads(pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(MTLSize(width: weight.rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeLayerNorm(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        rows: Int, columns: Int
    ) throws {
        let pipeline = try context.pipeline(named: "layernorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create LayerNorm encoder")
        }
        var n = UInt32(columns), epsilon = Self.eps, rowCount = UInt32(rows)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.setBytes(&epsilon, length: 4, index: 3)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeModulate(
        _ command: MTLCommandBuffer, normalized: MTLBuffer, modulation: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "modulate_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create AdaLN encoder")
        }
        var n = UInt32(Self.dim), elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(normalized, offset: 0, index: 0)
        encoder.setBuffer(modulation, offset: Self.dim * MemoryLayout<Float>.stride, index: 1)
        encoder.setBuffer(modulation, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        encoder.setBytes(&elementCount, length: 4, index: 5)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeRMSRoPE(
        _ command: MTLCommandBuffer, input: MTLBuffer, weightOffset: Int,
        rope: MTLBuffer, output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(named: "rms_rope_split_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create RMS/RoPE encoder")
        }
        var tokens = UInt32(Self.tokens), heads = UInt32(Self.heads), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(streamer.ring, offset: weightOffset, index: 1)
        encoder.setBuffer(rope, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&tokens, length: 4, index: 4)
        encoder.setBytes(&heads, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: Self.tokens * Self.heads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRMSHeads(
        _ command: MTLCommandBuffer, input: MTLBuffer, weightOffset: Int,
        output: MTLBuffer, rows: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rmsnorm_heads_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create head RMSNorm encoder")
        }
        var rowCount = UInt32(rows), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(streamer.ring, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeTranspose(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        tokens: Int, toHeadMajor: Bool
    ) throws {
        let pipeline = try context.pipeline(named: "transpose_token_head_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create head transpose encoder")
        }
        var tokenCount = UInt32(tokens), heads = UInt32(Self.heads)
        var headDim = UInt32(Self.headDim), direction: UInt32 = toHeadMajor ? 1 : 0
        let count = tokens * Self.dim
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&tokenCount, length: 4, index: 2)
        encoder.setBytes(&heads, length: 4, index: 3)
        encoder.setBytes(&headDim, length: 4, index: 4)
        encoder.setBytes(&direction, length: 4, index: 5)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeGateAdd(
        _ command: MTLCommandBuffer, residual: MTLBuffer, branch: MTLBuffer,
        modulation: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "gate_add_half_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create gated residual encoder")
        }
        var n = UInt32(Self.dim), elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBuffer(modulation, offset: 2 * Self.dim * MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&elementCount, length: 4, index: 4)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeUnary(
        _ command: MTLCommandBuffer, kernel: String, input: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(kernel) encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeConvert(
        _ command: MTLCommandBuffer, kernel: String, input: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        try encodeUnary(command, kernel: kernel, input: input, output: output, count: count)
    }

    private func encodeBinary(
        _ command: MTLCommandBuffer, kernel: String, destination: MTLBuffer,
        source: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(kernel) encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destination, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func dispatch1D(
        _ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int
    ) {
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func reductionThreads(_ maximum: Int) -> Int {
        var result = 1
        while result * 2 <= min(256, maximum) { result *= 2 }
        return result
    }

    private func buffer<T>(_ key: String, _ count: Int, _: T.Type) -> MTLBuffer {
        buffers.buffer(key: key, bytes: count * MemoryLayout<T>.stride)
    }

    private func validateInputs(
        residual: MTLBuffer, emb: MTLBuffer, adalnLora: MTLBuffer,
        crossContext: MTLBuffer, rope: MTLBuffer
    ) throws {
        let requirements = [
            (residual, Self.tokens * Self.dim * 4, "residual"),
            (emb, Self.dim * 4, "emb"),
            (adalnLora, Self.modulationSize * 4, "adalnLora"),
            (crossContext, Self.contextTokens * Self.contextDim * 2, "crossContext"),
            (rope, Self.tokens * 64 * 4 * 4, "rope"),
        ]
        for (buffer, bytes, label) in requirements where buffer.length < bytes {
            throw AnimapkError.validation("DiT \(label) buffer is too small")
        }
    }
}

private struct BlockWeights {
    let modSelf1, modSelf2, modCross1, modCross2, modMLP1, modMLP2: QuantizedLinearWeightBuffers
    let selfQ, selfK, selfV, selfO: QuantizedLinearWeightBuffers
    let crossQ, crossK, crossV, crossO: QuantizedLinearWeightBuffers
    let mlp1, mlp2: QuantizedLinearWeightBuffers
    let selfQNorm, selfKNorm, crossQNorm, crossKNorm: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model.blocks.\(range.logicalIndex)."
        var spans: [String: AnimapkTensorSpans] = [:]
        for item in range.tensors {
            guard item.tensor.name.hasPrefix(prefix) else {
                throw AnimapkError.validation("foreign tensor in DiT block range")
            }
            spans[String(item.tensor.name.dropFirst(prefix.count))] = item
        }
        func matrix(_ name: String, _ rows: Int, _ columns: Int) throws -> QuantizedLinearWeightBuffers {
            guard let item = spans[name] else {
                throw AnimapkError.validation("missing DiT matrix \(prefix)\(name)")
            }
            return try DiTQuantizedWeightFactory.makeMatrix(
                item, ring: ring, rows: rows, columns: columns,
                label: "DiT \(prefix)\(name)")
        }
        func norm(_ name: String) throws -> Int {
            guard let item = spans[name], item.tensor.shape == [128],
                  item.tensor.storage == .fp16, item.data.length == 256,
                  item.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid DiT norm \(prefix)\(name)")
            }
            return Int(item.data.offset)
        }
        modSelf1 = try matrix("adaln_modulation_self_attn.1.weight", 256, 2_048)
        modSelf2 = try matrix("adaln_modulation_self_attn.2.weight", 6_144, 256)
        modCross1 = try matrix("adaln_modulation_cross_attn.1.weight", 256, 2_048)
        modCross2 = try matrix("adaln_modulation_cross_attn.2.weight", 6_144, 256)
        modMLP1 = try matrix("adaln_modulation_mlp.1.weight", 256, 2_048)
        modMLP2 = try matrix("adaln_modulation_mlp.2.weight", 6_144, 256)
        selfQ = try matrix("self_attn.q_proj.weight", 2_048, 2_048)
        selfK = try matrix("self_attn.k_proj.weight", 2_048, 2_048)
        selfV = try matrix("self_attn.v_proj.weight", 2_048, 2_048)
        selfO = try matrix("self_attn.output_proj.weight", 2_048, 2_048)
        crossQ = try matrix("cross_attn.q_proj.weight", 2_048, 2_048)
        crossK = try matrix("cross_attn.k_proj.weight", 2_048, 1_024)
        crossV = try matrix("cross_attn.v_proj.weight", 2_048, 1_024)
        crossO = try matrix("cross_attn.output_proj.weight", 2_048, 2_048)
        mlp1 = try matrix("mlp.layer1.weight", 8_192, 2_048)
        mlp2 = try matrix("mlp.layer2.weight", 2_048, 8_192)
        selfQNorm = try norm("self_attn.q_norm.weight")
        selfKNorm = try norm("self_attn.k_norm.weight")
        crossQNorm = try norm("cross_attn.q_norm.weight")
        crossKNorm = try norm("cross_attn.k_norm.weight")
        guard spans.count == 20 else {
            throw AnimapkError.validation("DiT block must contain exactly 20 tensors")
        }
    }
}
