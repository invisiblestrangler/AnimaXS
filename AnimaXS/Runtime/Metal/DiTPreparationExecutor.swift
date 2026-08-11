import Foundation
import Metal

/// Production Metal bridge for H001/H002: latent patch embedding plus timestep
/// RMSNorm/AdaLN-LoRA base vectors, streamed from one metadata-derived range.
final class DiTPreparationExecutor {
    static let latentElements = 16 * 64 * 64
    static let tokens = 1_024
    static let hidden = 2_048
    static let adaln = 6_144
    private static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: DiTPreparationLocator
    private let streamer: WeightStreamer
    private let linear: LinearExecutor
    private let buffers: BufferPool

    init(context: MetalContext, file: AnimapkFile) throws {
        let locator = try DiTPreparationLocator(file: file)
        guard file.quantGroup == 64 else {
            throw AnimapkError.validation("DiT preparation requires quant group 64")
        }
        self.context = context
        self.file = file
        self.locator = locator
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(locator.range.length))
        self.linear = LinearExecutor(context: context)
        self.buffers = BufferPool(device: context.device)
    }

    /// Outputs fp32 residual `[1024,2048]`, timestep embedding `[2048]`, and
    /// AdaLN-LoRA base `[6144]` for one sigma.
    func execute(
        latent: MTLBuffer, sigma: Float,
        residual: MTLBuffer, embedding: MTLBuffer, adalnLora: MTLBuffer
    ) async throws {
        guard latent.length >= Self.latentElements * 4,
              residual.length >= Self.tokens * Self.hidden * 4,
              embedding.length >= Self.hidden * 4,
              adalnLora.length >= Self.adaln * 4,
              sigma.isFinite else {
            throw AnimapkError.validation("invalid DiT preparation input")
        }
        try streamer.load(locator.range, from: file)
        let weights = try DiTPreparationWeights(range: locator.range, ring: streamer.ring)
        let raw = makeTimestep(sigma)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create DiT preparation command buffer")
        }

        let modelInput = buffers.buffer(
            key: "dit.prepare.input17.f32", bytes: 17 * 64 * 64 * 4)
        guard let blit = command.makeBlitCommandEncoder() else {
            throw AnimapkError.validation("failed to create DiT preparation blit encoder")
        }
        blit.copy(from: latent, sourceOffset: 0, to: modelInput, destinationOffset: 0,
                  size: Self.latentElements * 4)
        blit.fill(buffer: modelInput,
                  range: (Self.latentElements * 4)..<(17 * 64 * 64 * 4), value: 0)
        blit.endEncoding()

        let patches = buffers.buffer(
            key: "dit.prepare.patches.f32", bytes: Self.tokens * 68 * 4)
        try encodePatchify(command, modelInput, patches)
        let patchesHalf = buffers.buffer(
            key: "dit.prepare.patches.f16", bytes: Self.tokens * 68 * 2)
        let residualHalf = buffers.buffer(
            key: "dit.prepare.residual.f16", bytes: Self.tokens * Self.hidden * 2)
        try encodeUnary(command, "float_to_half", patches, patchesHalf, Self.tokens * 68)
        try linear.encode(commandBuffer: command, input: patchesHalf,
                          weight: weights.xEmbed, output: residualHalf,
                          inputRows: Self.tokens)
        try encodeUnary(command, "half_to_float", residualHalf, residual,
                        Self.tokens * Self.hidden)

        try encodeRMSNorm(command, raw, embedding, weights.timestepNorm)
        let hidden = buffers.buffer(key: "dit.prepare.timestep.hidden.f32", bytes: Self.hidden * 4)
        let activated = buffers.buffer(
            key: "dit.prepare.timestep.silu.f32", bytes: Self.hidden * 4)
        try encodeMatvec(command, weights.timestep1, raw, hidden)
        try encodeUnary(command, "silu", hidden, activated, Self.hidden)
        try encodeMatvec(command, weights.timestep2, activated, adalnLora)
        try await commit(command)
    }

    private func makeTimestep(_ sigma: Float) -> MTLBuffer {
        let raw = buffers.buffer(key: "dit.prepare.timestep.raw.f32", bytes: Self.hidden * 4)
        let pointer = raw.contents().bindMemory(to: Float.self, capacity: Self.hidden)
        for index in 0..<1_024 {
            let frequency = exp(-log(Float(10_000)) * Float(index) / 1_024)
            let angle = sigma * frequency
            pointer[index] = cosf(angle)
            pointer[index + 1_024] = sinf(angle)
        }
        return raw
    }

    private func encodePatchify(
        _ command: MTLCommandBuffer, _ input: MTLBuffer, _ output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(named: "patchify17")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create patchify encoder")
        }
        var height: UInt32 = 64, width: UInt32 = 64
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&height, length: 4, index: 2)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: Self.tokens, height: 17, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRMSNorm(
        _ command: MTLCommandBuffer, _ input: MTLBuffer,
        _ output: MTLBuffer, _ weightOffset: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rmsnorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create timestep RMSNorm encoder")
        }
        var columns = UInt32(Self.hidden), epsilon = Self.eps
        var useWeight: UInt32 = 1, rows: UInt32 = 1
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(streamer.ring, offset: weightOffset, index: 2)
        encoder.setBytes(&columns, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&useWeight, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeMatvec(
        _ command: MTLCommandBuffer, _ weight: QuantizedLinearWeightBuffers,
        _ input: MTLBuffer, _ output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(named: "w4_matvec_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create timestep matvec encoder")
        }
        var columns = UInt32(weight.columns), rows = UInt32(weight.rows)
        var stride = UInt32(weight.packedRowStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 0)
        encoder.setBuffer(weight.scale, offset: weight.scaleOffset, index: 1)
        encoder.setBuffer(weight.zero, offset: weight.zeroOffset, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&columns, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBytes(&stride, length: 4, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: weight.rows, height: 1, depth: 1),
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

private struct DiTPreparationWeights {
    let xEmbed, timestep1, timestep2: QuantizedLinearWeightBuffers
    let timestepNorm: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model."
        let spans = Dictionary(uniqueKeysWithValues: range.tensors.map {
            (String($0.tensor.name.dropFirst(prefix.count)), $0)
        })
        func matrix(_ name: String, rows: Int, columns: Int) throws -> QuantizedLinearWeightBuffers {
            guard let item = spans[name], item.tensor.shape == [rows, columns],
                  item.tensor.storage == .w4, let scale = item.scale, let zero = item.zero,
                  item.data.length % UInt64(rows) == 0,
                  item.data.offset <= UInt64(Int.max), scale.offset <= UInt64(Int.max),
                  zero.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid DiT preparation matrix \(name)")
            }
            return QuantizedLinearWeightBuffers(
                storage: .w4, packed: ring, packedOffset: Int(item.data.offset),
                scale: ring, scaleOffset: Int(scale.offset), zero: ring,
                zeroOffset: Int(zero.offset), rows: rows, columns: columns,
                packedRowStride: Int(item.data.length / UInt64(rows)))
        }
        xEmbed = try matrix("x_embedder.proj.1.weight", rows: 2_048, columns: 68)
        timestep1 = try matrix("t_embedder.1.linear_1.weight", rows: 2_048, columns: 2_048)
        timestep2 = try matrix("t_embedder.1.linear_2.weight", rows: 6_144, columns: 2_048)
        guard let norm = spans["t_embedding_norm.weight"], norm.tensor.shape == [2_048],
              norm.tensor.storage == .fp16, norm.data.length == 4_096,
              norm.data.offset <= UInt64(Int.max), spans.count == 4 else {
            throw AnimapkError.validation("invalid DiT preparation norm/range")
        }
        timestepNorm = Int(norm.data.offset)
    }
}
