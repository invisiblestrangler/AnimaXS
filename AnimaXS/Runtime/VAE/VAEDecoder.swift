import Foundation
import Metal
import MetalPerformanceShaders

/// Full-frame streamed Wan T=1 VAE decoder (J002).
///
/// Executes the pinned decoder graph (D052/D053/D060) with bounded memory:
///   latent (unchanged, per D060)
///     -> conv2 (1x1, 16->16)
///     -> decoder.conv1 (final-slice 3x3, 16->384)
///     -> middle: residual -> one-head spatial attention -> residual (384)
///     -> 15 upsample modules in four stages (384@64 -> 192@128 -> 384@128
///        -> 192@256 -> 96@512), resamples 3/7/11 fused nearest-exact 2x + 3x3
///     -> head: channel RMS norm -> SiLU -> 3x3 conv (96->3)
///
/// Activations are position-major `[H*W, C]` fp16 and ping-pong through two
/// reusable buffers. Each logical weight group is streamed into one ring slot
/// (loaded, consumed, then reused for the next group). The two `time_conv`
/// tensors are never executed at T=1.
final class VAEDecoder {
    private static let latentChannels = 16
    private static let latentSize = 64
    private static let outputChannels = 3
    private static let outputSize = 512

    private let context: MetalContext
    private let file: AnimapkFile
    private let locator: VAEDecoderLocator
    private let streamer: WeightStreamer
    private let buffers: BufferPool
    private let convolution: FP16ConvolutionExecutor
    private let attention: AttentionExecutor

    init(context: MetalContext, file: AnimapkFile) throws {
        let locator = try VAEDecoderLocator(file: file)
        let capacity = try locator.maximumGroupLength()
        guard capacity <= UInt64(Int.max), capacity > 0 else {
            throw AnimapkError.validation("VAE decoder has no usable weight groups")
        }
        self.context = context
        self.file = file
        self.locator = locator
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(capacity))
        self.buffers = BufferPool(device: context.device)
        self.convolution = FP16ConvolutionExecutor(context: context)
        self.attention = AttentionExecutor(context: context)
    }

    /// Decode `latent` (fp32 `[16,64,64]` channel-major, from the sampler) into
    /// `rgb` (fp32 `[3,512,512]` channel-major). The latent is consumed
    /// UNCHANGED per D060 — no mean/std denormalization.
    func execute(latent: MTLBuffer, rgb: MTLBuffer) async throws {
        guard latent.length >= Self.latentChannels * Self.latentSize * Self.latentSize * 4,
              rgb.length >= Self.outputChannels * Self.outputSize * Self.outputSize * 4 else {
            throw AnimapkError.validation("VAE decoder input or output buffer is too small")
        }
        context.refreshDiagnostics()

        // 1) latent fp32 [16,64,64] -> position-major fp16 [4096, 16]
        let inputPositions = Self.latentSize * Self.latentSize
        let positioned = activation(key: "vae.activation.a", positions: inputPositions, channels: Self.latentChannels)
        try await encodeLatentToPosition(latent: latent, output: positioned,
                                         positions: inputPositions, channels: Self.latentChannels)

        // 2) conv2: 1x1 16->16 (group 0)
        var x = positioned
        x = try await run1x1Group(groupIndex: 0, input: x, positions: inputPositions,
                                  inputChannels: Self.latentChannels,
                                  outputChannels: Self.latentChannels,
                                  key: "vae.activation.b")

        // 3) decoder.conv1: final-slice 3x3 16->384 @64 (group 1)
        x = try await run3x3Group(groupIndex: 1, input: x, height: Self.latentSize,
                                  width: Self.latentSize, inputChannels: Self.latentChannels,
                                  outputChannels: 384, key: "vae.activation.a")

        // 4) middle: residual(2) -> attention(3) -> residual(4), all 384 @64
        x = try await runResidualGroup(groupIndex: 2, input: x, height: Self.latentSize,
                                       width: Self.latentSize, inChannels: 384, outChannels: 384,
                                       key: "vae.activation.b")
        x = try await runAttentionGroup(groupIndex: 3, input: x, height: Self.latentSize,
                                        width: Self.latentSize, channels: 384,
                                        key: "vae.activation.a")
        x = try await runResidualGroup(groupIndex: 4, input: x, height: Self.latentSize,
                                       width: Self.latentSize, inChannels: 384, outChannels: 384,
                                       key: "vae.activation.b")

        // 5) upsample modules 0...14. Group index = 5 + module.
        //    Resample modules are 3, 7, 11; module 4 is the 192->384
        //    channel-change block with shortcut. x ping-pongs between the two
        //    activation buffers (never aliased as conv input and output).
        var height = Self.latentSize
        var width = Self.latentSize
        var xInA = false  // after middle residual, x lives in vae.activation.b
        for module in 0..<15 {
            let groupIndex = 5 + module
            let group = try locator.group(groupIndex)
            let nextKey = xInA ? "vae.activation.b" : "vae.activation.a"
            // Resample modules (3/7/11) carry .resample.1.weight; others don't.
            let isResample = try tensorSpan(group.range, suffix: ".resample.1.weight") != nil
            if isResample {
                // Resample module: fused nearest-exact 2x + 3x3 Cin -> Cout.
                let rs = try requireTensorSpan(group.range, suffix: ".resample.1.weight")
                let inC = rs.tensor.shape[1]
                let outC = rs.tensor.shape[0]
                let next = activation(key: nextKey, positions: height * width * 4, channels: outC)
                try await encodeResampleGroup(groupIndex: groupIndex, input: x, output: next,
                                              height: height, width: width,
                                              inputChannels: inC, outputChannels: outC)
                x = next
                xInA.toggle()
                height *= 2
                width *= 2
            } else {
                // Residual-only module.
                // Only module 4 carries a shortcut (192->384 channel change).
                let hasShortcut = try tensorSpan(group.range, suffix: ".shortcut.weight") != nil
                let w2Shape = try requireTensorSpan(group.range, suffix: ".residual.2.weight").tensor.shape
                let inC = w2Shape[1]
                let outC = w2Shape[0]
                x = try await runResidual(groupIndex: groupIndex, input: x,
                                          height: height, width: width,
                                          inChannels: inC, outChannels: outC,
                                          key: nextKey, hasShortcut: hasShortcut)
                xInA.toggle()
            }
        }

        // 6) head (group 20): RMS norm(96) -> SiLU -> 3x3 conv 96->3 @512
        let headRGB = activation(key: "vae.head.rgb", positions: Self.outputSize * Self.outputSize,
                                 channels: Self.outputChannels)
        try await encodeHead(groupIndex: 20, input: x, output: headRGB,
                             height: Self.outputSize, width: Self.outputSize)
        try await encodeRGBToChannelMajor(input: headRGB, output: rgb,
                                          positions: Self.outputSize * Self.outputSize,
                                          channels: Self.outputChannels)
        context.refreshDiagnostics()
    }
}

// MARK: - Group executors

extension VAEDecoder {
    /// Stream one weight group, encode its 1x1 convolution, await.
    private func run1x1Group(
        groupIndex: Int, input: MTLBuffer, positions: Int,
        inputChannels: Int, outputChannels: Int, key: String
    ) async throws -> MTLBuffer {
        let group = try locator.group(groupIndex)
        try streamer.load(group.range, from: file)
        let output = activation(key: key, positions: positions, channels: outputChannels)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE 1x1 group command buffer")
        }
        let weight = try requireTensorSpan(group.range, suffix: ".weight")
        let bias = try requireTensorSpan(group.range, suffix: ".bias")
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: inputChannels, dataType: .float16)
        let folded = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(weight.data.offset), shape: weight.tensor.shape,
            outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.1x1")
        try convolution.encode1x1(
            commandBuffer: command, input: input, weight: folded, weightOffset: 0,
            output: output, rows: positions,
            inputChannels: inputChannels, outputChannels: outputChannels)
        try encodeAddBias(command: command, output: output,
                          bias: streamer.ring, biasOffset: Int(bias.data.offset),
                          channels: outputChannels, count: positions * outputChannels)
        try await commit(command)
        return output
    }

    /// Stream one weight group, encode its 3x3 convolution, await.
    private func run3x3Group(
        groupIndex: Int, input: MTLBuffer, height: Int, width: Int,
        inputChannels: Int, outputChannels: Int, key: String
    ) async throws -> MTLBuffer {
        let group = try locator.group(groupIndex)
        try streamer.load(group.range, from: file)
        let output = activation(key: key, positions: height * width, channels: outputChannels)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE 3x3 group command buffer")
        }
        let weight = try requireTensorSpan(group.range, suffix: ".weight")
        let bias = try requireTensorSpan(group.range, suffix: ".bias")
        let columns = inputChannels * 9
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float16)
        let folded = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(weight.data.offset), shape: weight.tensor.shape,
            outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.3x3")
        try convolution.encode3x3(
            commandBuffer: command, input: input, weight: folded, weightOffset: 0,
            bias: streamer.ring, biasOffset: Int(bias.data.offset),
            output: output, inputHeight: height, inputWidth: width,
            outputChannels: outputChannels, inputChannels: inputChannels)
        try await commit(command)
        return output
    }
}

// MARK: - Residual / attention / resample / head

extension VAEDecoder {
    /// Wan ResidualBlock: RMS(g0)->SiLU->3x3(w2,b2)->RMS(g3)->SiLU->3x3(w6,b6)
    /// plus shortcut (identity or 1x1 conv).
    private func runResidualGroup(
        groupIndex: Int, input: MTLBuffer, height: Int, width: Int,
        inChannels: Int, outChannels: Int, key: String
    ) async throws -> MTLBuffer {
        try await runResidual(groupIndex: groupIndex, input: input, height: height, width: width,
                              inChannels: inChannels, outChannels: outChannels, key: key,
                              hasShortcut: false)
    }

    private func runResidual(
        groupIndex: Int, input: MTLBuffer, height: Int, width: Int,
        inChannels: Int, outChannels: Int, key: String, hasShortcut: Bool
    ) async throws -> MTLBuffer {
        let group = try locator.group(groupIndex)
        try streamer.load(group.range, from: file)
        let positions = height * width
        let output = activation(key: key, positions: positions, channels: outChannels)
        // Two working buffers ping-pong so no convolution aliases input/output.
        let workA = activation(key: "vae.residual.a", positions: positions, channels: outChannels)
        let workB = activation(key: "vae.residual.b", positions: positions, channels: outChannels)

        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE residual command buffer")
        }

        // Stage 1: RMS(g0) -> SiLU -> 3x3 w2/b2 (workA -> workB)
        let g0 = try requireTensorSpan(group.range, suffix: ".residual.0.gamma")
        let w2 = try requireTensorSpan(group.range, suffix: ".residual.2.weight")
        let b2 = try requireTensorSpan(group.range, suffix: ".residual.2.bias")
        let columns2 = inChannels * 9
        let rowBytes2 = MPSMatrixDescriptor.rowBytes(fromColumns: columns2, dataType: .float16)
        let folded2 = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(w2.data.offset), shape: w2.tensor.shape,
            outputRowStrideElements: rowBytes2 / 2, scratchKey: "vae.weight.residual.2")
        try encodeChannelRMS(command: command, input: input, gamma: streamer.ring,
                             gammaOffset: Int(g0.data.offset), output: workA,
                             positions: positions, channels: inChannels)
        try encodeSiluHalf(command: command, input: workA, output: workA, count: positions * inChannels)
        try convolution.encode3x3(
            commandBuffer: command, input: workA, weight: folded2, weightOffset: 0,
            bias: streamer.ring, biasOffset: Int(b2.data.offset),
            output: workB, inputHeight: height, inputWidth: width,
            outputChannels: outChannels, inputChannels: inChannels)

        // Stage 2: RMS(g3) -> SiLU -> 3x3 w6/b6 (workB -> workA)
        let g3 = try requireTensorSpan(group.range, suffix: ".residual.3.gamma")
        let w6 = try requireTensorSpan(group.range, suffix: ".residual.6.weight")
        let b6 = try requireTensorSpan(group.range, suffix: ".residual.6.bias")
        let columns6 = outChannels * 9
        let rowBytes6 = MPSMatrixDescriptor.rowBytes(fromColumns: columns6, dataType: .float16)
        let folded6 = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(w6.data.offset), shape: w6.tensor.shape,
            outputRowStrideElements: rowBytes6 / 2, scratchKey: "vae.weight.residual.6")
        try encodeChannelRMS(command: command, input: workB, gamma: streamer.ring,
                             gammaOffset: Int(g3.data.offset), output: workA,
                             positions: positions, channels: outChannels)
        try encodeSiluHalf(command: command, input: workA, output: workA, count: positions * outChannels)
        try convolution.encode3x3(
            commandBuffer: command, input: workA, weight: folded6, weightOffset: 0,
            bias: streamer.ring, biasOffset: Int(b6.data.offset),
            output: workB, inputHeight: height, inputWidth: width,
            outputChannels: outChannels, inputChannels: outChannels)

        // Shortcut (in != out): 1x1 conv from input into output.
        if hasShortcut {
            let sc = try requireTensorSpan(group.range, suffix: ".shortcut.weight")
            let scb = try requireTensorSpan(group.range, suffix: ".shortcut.bias")
            let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: inChannels, dataType: .float16)
            let foldedShortcut = try convolution.encodeFoldWeight(
                commandBuffer: command, source: streamer.ring,
                sourceOffset: Int(sc.data.offset), shape: sc.tensor.shape,
                outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.shortcut")
            try convolution.encode1x1(
                commandBuffer: command, input: input, weight: foldedShortcut, weightOffset: 0,
                output: output, rows: positions,
                inputChannels: inChannels, outputChannels: outChannels)
            try encodeAddBias(command: command, output: output,
                              bias: streamer.ring, biasOffset: Int(scb.data.offset),
                              channels: outChannels, count: positions * outChannels)
        } else {
            try encodeCopy(command: command, input: input, output: output,
                           count: positions * inChannels)
        }
        // residual add: output = workB + output
        try encodeAddHalf(command: command, lhs: workB, rhs: output, out: output,
                          count: positions * outChannels)
        try await commit(command)
        return output
    }

    /// Middle attention: RMS(norm.gamma) -> 1x1 to_qkv -> single-head spatial
    /// attention -> 1x1 proj -> + identity.
    private func runAttentionGroup(
        groupIndex: Int, input: MTLBuffer, height: Int, width: Int,
        channels: Int, key: String
    ) async throws -> MTLBuffer {
        let group = try locator.group(groupIndex)
        try streamer.load(group.range, from: file)
        let positions = height * width
        let output = activation(key: key, positions: positions, channels: channels)
        let normalized = activation(key: "vae.attention.norm", positions: positions, channels: channels)
        let qkv = activation(key: "vae.attention.qkv", positions: positions, channels: channels * 3)
        let projected = activation(key: "vae.attention.proj", positions: positions, channels: channels)

        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE attention command buffer")
        }

        let normGamma = try requireTensorSpan(group.range, suffix: ".norm.gamma")
        try encodeChannelRMS(command: command, input: input, gamma: streamer.ring,
                             gammaOffset: Int(normGamma.data.offset), output: normalized,
                             positions: positions, channels: channels)

        let toQKV = try requireTensorSpan(group.range, suffix: ".to_qkv.weight")
        let toQKVBias = try requireTensorSpan(group.range, suffix: ".to_qkv.bias")
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: channels, dataType: .float16)
        let foldedQKV = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(toQKV.data.offset), shape: toQKV.tensor.shape,
            outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.qkv")
        try convolution.encode1x1(
            commandBuffer: command, input: normalized, weight: foldedQKV, weightOffset: 0,
            output: qkv, rows: positions, inputChannels: channels,
            outputChannels: channels * 3)
        try encodeAddBias(command: command, output: qkv,
                          bias: streamer.ring, biasOffset: Int(toQKVBias.data.offset),
                          channels: channels * 3, count: positions * channels * 3)
        // Split interleaved [positions, 3C] into contiguous Q/K/V blocks.
        let splitQKV = activation(key: "vae.attention.qkv.split",
                                  positions: positions * 3, channels: channels)
        try encodeSplitQKV(command: command, qkv: qkv, split: splitQKV,
                           positions: positions, channels: channels)
        try await commit(command)

        // Attention: single head, rows=positions, keyCount=positions, headDim=channels.
        // Q/K/V are contiguous [positions, channels] blocks inside splitQKV.
        let half = MemoryLayout<Float16>.stride
        try await attention.execute(
            query: splitQKV, queryOffset: 0,
            key: splitQKV, keyOffset: positions * channels * half,
            value: splitQKV, valueOffset: positions * channels * 2 * half,
            output: projected,
            heads: 1, queryCount: positions, keyCount: positions,
            headDim: channels, causal: false)

        // proj 1x1 + residual identity.
        guard let command2 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE attention proj command buffer")
        }
        let proj = try requireTensorSpan(group.range, suffix: ".proj.weight")
        let projBias = try requireTensorSpan(group.range, suffix: ".proj.bias")
        let foldedProj = try convolution.encodeFoldWeight(
            commandBuffer: command2, source: streamer.ring,
            sourceOffset: Int(proj.data.offset), shape: proj.tensor.shape,
            outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.proj")
        try convolution.encode1x1(
            commandBuffer: command2, input: projected, weight: foldedProj, weightOffset: 0,
            output: output, rows: positions, inputChannels: channels,
            outputChannels: channels)
        try encodeAddBias(command: command2, output: output,
                          bias: streamer.ring, biasOffset: Int(projBias.data.offset),
                          channels: channels, count: positions * channels)
        try encodeAddHalf(command: command2, lhs: output, rhs: input, out: output,
                          count: positions * channels)
        try await commit(command2)
        return output
    }

    /// Resample: fused nearest-exact 2x + 3x3 convolution (Cin -> Cout).
    private func encodeResampleGroup(
        groupIndex: Int, input: MTLBuffer, output: MTLBuffer,
        height: Int, width: Int, inputChannels: Int, outputChannels: Int
    ) async throws {
        let group = try locator.group(groupIndex)
        try streamer.load(group.range, from: file)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE resample command buffer")
        }
        let weight = try requireTensorSpan(group.range, suffix: ".resample.1.weight")
        let bias = try requireTensorSpan(group.range, suffix: ".resample.1.bias")
        let columns = inputChannels * 9
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float16)
        let folded = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(weight.data.offset), shape: weight.tensor.shape,
            outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.resample")
        try convolution.encode3x3(
            commandBuffer: command, input: input, weight: folded, weightOffset: 0,
            bias: streamer.ring, biasOffset: Int(bias.data.offset),
            output: output, inputHeight: height, inputWidth: width,
            outputChannels: outputChannels, inputChannels: inputChannels,
            upsample2x: true)
        try await commit(command)
    }

    /// Head: RMS(head.0.gamma) -> SiLU -> 3x3 conv (96 -> 3).
    private func encodeHead(
        groupIndex: Int, input: MTLBuffer, output: MTLBuffer,
        height: Int, width: Int
    ) async throws {
        let group = try locator.group(groupIndex)
        try streamer.load(group.range, from: file)
        let positions = height * width
        let normalized = activation(key: "vae.head.norm", positions: positions, channels: 96)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE head command buffer")
        }
        let gamma = try requireTensorSpan(group.range, suffix: ".head.0.gamma")
        try encodeChannelRMS(command: command, input: input, gamma: streamer.ring,
                             gammaOffset: Int(gamma.data.offset), output: normalized,
                             positions: positions, channels: 96)
        try encodeSiluHalf(command: command, input: normalized, output: normalized,
                           count: positions * 96)
        let weight = try requireTensorSpan(group.range, suffix: ".head.2.weight")
        let bias = try requireTensorSpan(group.range, suffix: ".head.2.bias")
        let columns = 96 * 9
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float16)
        let folded = try convolution.encodeFoldWeight(
            commandBuffer: command, source: streamer.ring,
            sourceOffset: Int(weight.data.offset), shape: weight.tensor.shape,
            outputRowStrideElements: rowBytes / 2, scratchKey: "vae.weight.head")
        try convolution.encode3x3(
            commandBuffer: command, input: normalized, weight: folded, weightOffset: 0,
            bias: streamer.ring, biasOffset: Int(bias.data.offset),
            output: output, inputHeight: height, inputWidth: width,
            outputChannels: 3, inputChannels: 96)
        try await commit(command)
    }
}

// MARK: - Low-level encoders and helpers

extension VAEDecoder {
    /// Latent fp32 [C,H,W] -> position-major fp16 [H*W, C] (unchanged values, D060).
    private func encodeLatentToPosition(
        latent: MTLBuffer, output: MTLBuffer, positions: Int, channels: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE latent command buffer")
        }
        let pipeline = try context.pipeline(named: "vae_latent_to_position_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE latent encoder")
        }
        var height = UInt32(Self.latentSize), width = UInt32(Self.latentSize)
        var channelsU = UInt32(channels)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(latent, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&height, length: 4, index: 2)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.setBytes(&channelsU, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: positions, height: channels, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        try await commit(command)
    }

    /// Position-major fp16 [H*W, C] -> fp32 channel-major [C, H, W].
    private func encodeRGBToChannelMajor(
        input: MTLBuffer, output: MTLBuffer, positions: Int, channels: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE rgb command buffer")
        }
        let pipeline = try context.pipeline(named: "vae_position_to_rgb_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE rgb encoder")
        }
        var height = UInt32(Self.outputSize), width = UInt32(Self.outputSize)
        var channelsU = UInt32(channels)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&height, length: 4, index: 2)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.setBytes(&channelsU, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: positions, height: channels, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        try await commit(command)
    }

    /// Wan channel RMS norm (fp32 reduction) on position-major fp16.
    private func encodeChannelRMS(
        command: MTLCommandBuffer, input: MTLBuffer, gamma: MTLBuffer,
        gammaOffset: Int, output: MTLBuffer, positions: Int, channels: Int
    ) throws {
        let pipeline = try context.pipeline(named: "vae_channel_rmsnorm_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE RMS norm encoder")
        }
        var positionsU = UInt32(positions), channelsU = UInt32(channels)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gamma, offset: gammaOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&positionsU, length: 4, index: 3)
        encoder.setBytes(&channelsU, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: positions, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// SiLU on fp16 (in-place allowed: elementwise).
    private func encodeSiluHalf(
        command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "silu_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE SiLU encoder")
        }
        var countU = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&countU, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Add fp16 bias broadcast across rows (in-place on `output`).
    private func encodeAddBias(
        command: MTLCommandBuffer, output: MTLBuffer, bias: MTLBuffer,
        biasOffset: Int, channels: Int, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_bias_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE bias encoder")
        }
        var channelsU = UInt32(channels), countU = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBytes(&channelsU, length: 4, index: 2)
        encoder.setBytes(&countU, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Copy fp16 (used to seed the residual identity path).
    private func encodeCopy(
        command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "copy_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE copy encoder")
        }
        var countU = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&countU, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Elementwise fp16 add (residual).
    private func encodeAddHalf(
        command: MTLCommandBuffer, lhs: MTLBuffer, rhs: MTLBuffer,
        out: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "vae_add_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE add encoder")
        }
        var countU = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        encoder.setBytes(&countU, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Split interleaved [positions, 3*channels] to_qkv output into contiguous
    /// Q/K/V blocks for the attention executor's [heads, rows, headDim] layout.
    private func encodeSplitQKV(
        command: MTLCommandBuffer, qkv: MTLBuffer, split: MTLBuffer,
        positions: Int, channels: Int
    ) throws {
        let pipeline = try context.pipeline(named: "vae_split_qkv_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE QKV split encoder")
        }
        var positionsU = UInt32(positions), channelsU = UInt32(channels)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(split, offset: 0, index: 1)
        encoder.setBytes(&positionsU, length: 4, index: 2)
        encoder.setBytes(&channelsU, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: positions, height: channels, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
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

// MARK: - Tensor access

extension VAEDecoder {
    /// Find one tensor's spans in a group by full-name suffix.
    private func tensorSpan(_ group: AnimapkExecutionRange, suffix: String) throws -> AnimapkTensorSpans? {
        group.tensors.first(where: { $0.tensor.name.hasSuffix(suffix) })
    }

    /// Non-optional variant for tensors that must exist in a group.
    private func requireTensorSpan(
        _ group: AnimapkExecutionRange, suffix: String
    ) throws -> AnimapkTensorSpans {
        guard let item = try tensorSpan(group, suffix: suffix) else {
            throw AnimapkError.validation(
                "VAE decoder group \(group.logicalIndex) is missing tensor \(suffix)")
        }
        return item
    }
}

// MARK: - Activation buffers

extension VAEDecoder {
    /// Position-major fp16 activation buffer: `[positions, channels]`, tight rows.
    /// Buffers grow on demand and are reused across stages (bounded memory).
    private func activation(key: String, positions: Int, channels: Int) -> MTLBuffer {
        buffers.buffer(key: key, bytes: positions * channels * MemoryLayout<Float16>.stride)
    }
}
