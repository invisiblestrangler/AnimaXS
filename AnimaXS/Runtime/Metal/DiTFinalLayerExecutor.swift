import Foundation
import Metal

/// Streamed MiniTrainDIT FinalLayer + unpatchify for the fixed image model shape.
final class DiTFinalLayerExecutor {
    private static let tokens = 1_024
    private static let dim = 2_048
    private static let modulationHidden = 256
    private static let modulationSize = 4_096
    private static let projected = 64
    private static let outputElements = 16 * 64 * 64
    private static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let range: AnimapkExecutionRange
    private let streamer: WeightStreamer
    private let buffers: BufferPool
    private let linear: LinearExecutor
    private let activationNumerics: ActivationNumerics
    private let monitor: NumericalMonitor?
    /// Run telemetry collector (nil in tests / diagnostic-only construction).
    var metrics: MetricsCollector?
    private var emulatesBF16: Bool { activationNumerics == .bf16Compute }

    init(context: MetalContext, file: AnimapkFile,
         activationNumerics: ActivationNumerics = .legacy,
         monitor: NumericalMonitor? = nil) throws {
        let range = try DiTFinalLayerLocator(file: file).range
        guard range.length <= UInt64(Int.max) else {
            throw AnimapkError.validation("DiT final layer range is too large")
        }
        self.context = context
        self.file = file
        self.range = range
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(range.length))
        self.buffers = BufferPool(device: context.device)
        self.linear = LinearExecutor(context: context)
        self.activationNumerics = activationNumerics
        self.monitor = monitor
    }

    /// `residual`, `emb`, and `adalnLora` are fp32. `velocity` is fp32
    /// `[1,16,1,64,64]` flattened in channel-major order.
    func execute(
        residual: MTLBuffer, emb: MTLBuffer, adalnLora: MTLBuffer,
        velocity: MTLBuffer
    ) async throws {
        try validate(residual: residual, emb: emb, adalnLora: adalnLora, velocity: velocity)
        let copyStart = ProcessInfo.processInfo.systemUptime
        try streamer.load(range, from: file)
        metrics?.recordWeightCopy(bytes: Int(range.length), seconds: ProcessInfo.processInfo.systemUptime - copyStart)
        let weights = try FinalWeights(range: range, ring: streamer.ring)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create final-layer command buffer")
        }

        let siluEmb = buffer("final.siluEmb.f32", Self.dim, Float.self)
        let modulationHidden = buffer("final.modHidden.f32", Self.modulationHidden, Float.self)
        let modulation = buffer("final.modulation.f32", Self.modulationSize, Float.self)
        try encodeUnary(command, "silu", emb, siluEmb, Self.dim)
        try encodeComputeBoundary(command, siluEmb, count: Self.dim)
        try encodeMatvec(command, siluEmb, weights.modulation1, modulationHidden)
        try encodeComputeBoundary(command, modulationHidden, count: Self.modulationHidden)
        try encodeMatvec(command, modulationHidden, weights.modulation2, modulation)
        try encodeComputeBoundary(command, modulation, count: Self.modulationSize)
        try encodeAdd(command, destination: modulation, source: adalnLora, count: Self.modulationSize)
        try encodeComputeBoundary(command, modulation, count: Self.modulationSize)

        // predict2.py:930 casts the large fp32 residual to cross-attention dtype
        // before FinalLayer. Make that fp16 boundary explicit before fp32-stat LayerNorm.
        let residualHalf = buffer("final.residual.f16", Self.tokens * Self.dim, Float16.self)
        let boundaryFloat = buffer("final.boundary.f32", Self.tokens * Self.dim, Float.self)
        let normalized = buffer("final.normalized.f32", Self.tokens * Self.dim, Float.self)
        let normalizedHalf = buffer("final.normalized.f16", Self.tokens * Self.dim, Float16.self)
        let normalizedBoundary = buffer(
            "final.normalizedBoundary.f32", Self.tokens * Self.dim, Float.self)
        let modulated = buffer("final.modulated.f32", Self.tokens * Self.dim, Float.self)
        let projectionInput = buffer("final.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        try encodeProbeConvert(command, "float_to_half", residual, residualHalf,
                               Self.tokens * Self.dim, probe: .finalResidualToHalf)
        try encodeUnary(command, "half_to_float", residualHalf, boundaryFloat, Self.tokens * Self.dim)
        try encodeLayerNorm(command, input: boundaryFloat, output: normalized)
        // torch.layer_norm preserves its fp16 input dtype. AdaLN then promotes the
        // rounded norm output when adding the fp32 shift/scale tensors.
        try encodeProbeConvert(command, "float_to_half", normalized, normalizedHalf,
                               Self.tokens * Self.dim, probe: .finalNormalizedToHalf)
        try encodeHalfComputeBoundary(command, normalizedHalf, count: Self.tokens * Self.dim)
        try encodeUnary(command, "half_to_float", normalizedHalf, normalizedBoundary,
                        Self.tokens * Self.dim)
        try encodeModulate(command, normalized: normalizedBoundary,
                           modulation: modulation, output: modulated)
        try encodeComputeBoundary(command, modulated, count: Self.tokens * Self.dim)
        try encodeProbeConvert(command, "float_to_half", modulated, projectionInput,
                               Self.tokens * Self.dim, probe: .finalProjectionInput)

        let projectedHalf = buffer("final.projected.f16", Self.tokens * Self.projected, Float16.self)
        let projectedFloat = buffer("final.projected.f32", Self.tokens * Self.projected, Float.self)
        try linear.encode(commandBuffer: command, input: projectionInput,
                          weight: weights.projection, output: projectedHalf,
                          inputRows: Self.tokens)
        try encodeHalfComputeBoundary(command, projectedHalf, count: Self.tokens * Self.projected)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: projectedHalf,
                                    count: Self.tokens * Self.projected, probe: .finalProjected)
        }
        try encodeUnary(command, "half_to_float", projectedHalf, projectedFloat,
                        Self.tokens * Self.projected)
        try encodeUnpatchify(command, input: projectedFloat, output: velocity)
        if let monitor {
            try monitor.encodeProbeF32(command, values: velocity,
                                       count: Self.outputElements, probe: .velocity)
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
            command.commit()
        }
    }

    private func encodeMatvec(
        _ command: MTLCommandBuffer, _ input: MTLBuffer,
        _ weight: QuantizedLinearWeightBuffers, _ output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(
            named: DiTQuantizedWeightFactory.matvecKernel(for: weight.storage))
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create final modulation encoder")
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
        let threads = reductionThreads(pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(MTLSize(width: weight.rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeLayerNorm(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(named: "layernorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create final LayerNorm encoder")
        }
        var columns = UInt32(Self.dim), epsilon = Self.eps, rows = UInt32(Self.tokens)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&columns, length: 4, index: 2)
        encoder.setBytes(&epsilon, length: 4, index: 3)
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: Self.tokens, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeModulate(
        _ command: MTLCommandBuffer, normalized: MTLBuffer,
        modulation: MTLBuffer, output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(named: "modulate_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create final AdaLN encoder")
        }
        let count = Self.tokens * Self.dim
        var columns = UInt32(Self.dim), elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(normalized, offset: 0, index: 0)
        encoder.setBuffer(modulation, offset: Self.dim * 4, index: 1)
        encoder.setBuffer(modulation, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&elements, length: 4, index: 5)
        dispatch(encoder, pipeline, count)
        encoder.endEncoding()
    }

    private func encodeAdd(
        _ command: MTLCommandBuffer, destination: MTLBuffer,
        source: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create final modulation add encoder")
        }
        var elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destination, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
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

    /// float_to_half with in-kernel numerical-health recording (probe kernel
    /// performs the identical conversion).
    private func encodeProbeConvert(
        _ command: MTLCommandBuffer, _ name: String, _ input: MTLBuffer,
        _ output: MTLBuffer, _ count: Int, probe: NumericalMonitor.Probe
    ) throws {
        guard let monitor else {
            try encodeUnary(command, name, input, output, count)
            return
        }
        let pipeline = try context.pipeline(named: "float_to_half_probe")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create float_to_half_probe encoder")
        }
        var elements = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&elements, length: 4, index: 2)
        monitor.bindProbe(encoder, probe: probe, statsIndex: 3, slotIndex: 4)
        let groups = (count + 255) / 256
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeComputeBoundary(
        _ command: MTLCommandBuffer, _ value: MTLBuffer, count: Int
    ) throws {
        guard emulatesBF16 else { return }
        try encodeUnary(command, "round_f32_to_bf16", value, value, count)
    }

    private func encodeHalfComputeBoundary(
        _ command: MTLCommandBuffer, _ value: MTLBuffer, count: Int
    ) throws {
        guard emulatesBF16 else { return }
        let pipeline = try context.pipeline(named: "round_half_to_bf16")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create final BF16 boundary encoder")
        }
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(value, offset: 0, index: 0)
        encoder.setBuffer(value, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeUnpatchify(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(named: "unpatchify_velocity16")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create final unpatchify encoder")
        }
        var height: UInt32 = 64, width: UInt32 = 64
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&height, length: 4, index: 2)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: Self.tokens, height: 16, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 1))
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ count: Int
    ) {
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func reductionThreads(_ maximum: Int) -> Int {
        var threads = min(256, maximum)
        var power = 1
        while power * 2 <= threads { power *= 2 }
        threads = power
        return threads
    }

    private func buffer<T>(_ key: String, _ count: Int, _: T.Type) -> MTLBuffer {
        buffers.buffer(key: key, bytes: count * MemoryLayout<T>.stride)
    }

    private func validate(
        residual: MTLBuffer, emb: MTLBuffer, adalnLora: MTLBuffer, velocity: MTLBuffer
    ) throws {
        guard residual.length >= Self.tokens * Self.dim * 4,
              emb.length >= Self.dim * 4,
              adalnLora.length >= Self.modulationSize * 4,
              velocity.length >= Self.outputElements * 4 else {
            throw AnimapkError.validation("DiT final-layer input or output buffer is too small")
        }
    }
}

private struct FinalWeights {
    let modulation1, modulation2, projection: QuantizedLinearWeightBuffers

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model.final_layer."
        var byName: [String: AnimapkTensorSpans] = [:]
        for item in range.tensors {
            guard item.tensor.name.hasPrefix(prefix) else {
                throw AnimapkError.validation("foreign tensor in DiT final-layer range")
            }
            let name = String(item.tensor.name.dropFirst(prefix.count))
            guard byName.updateValue(item, forKey: name) == nil else {
                throw AnimapkError.validation("duplicate DiT final tensor \(name)")
            }
        }
        func matrix(_ name: String, _ rows: Int, _ columns: Int) throws -> QuantizedLinearWeightBuffers {
            guard let item = byName[name] else {
                throw AnimapkError.validation("missing DiT final matrix \(name)")
            }
            return try DiTQuantizedWeightFactory.makeMatrix(
                item, ring: ring, rows: rows, columns: columns,
                label: "DiT final \(name)")
        }
        modulation1 = try matrix("adaln_modulation.1.weight", 256, 2_048)
        modulation2 = try matrix("adaln_modulation.2.weight", 4_096, 256)
        projection = try matrix("linear.weight", 64, 2_048)
    }
}
