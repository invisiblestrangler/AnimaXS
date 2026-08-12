import XCTest
import Metal
import MetalPerformanceShaders
@testable import AnimaXS

/// Pack-free Metal-vs-CPU reference tests for the Wan VAE primitives (J002).
///
/// These run in normal CI (the macos-15 arm64 simulator executes project Metal
/// kernels and MPS). Weights are deliberately non-symmetric and the shapes
/// exercise corner padding, bias, multi-channel, non-square input, and output
/// counts that are not multiples of the executor's 128-row tile so that any
/// weight-ordering, stride, or tiling bug fails loudly.
final class VAEPrimitiveTests: XCTestCase {
    private func makeBuffer<T>(_ values: [T], on device: MTLDevice) -> MTLBuffer {
        values.withUnsafeBytes { bytes in
            device.makeBuffer(
                bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)!
        }
    }

    private func requireContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: MTLCreateSystemDefaultDevice/default library unavailable")
        }
        return context
    }

    private func readHalf(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    /// CPU reference conv for rank-4 PyTorch weights [Cout,Cin,KH,KW], padding
    /// same, stride 1, HWC input.
    private func referenceConv(
        _ input: [Float], height: Int, width: Int, inputChannels: Int,
        weight: [Float], outputChannels: Int, bias: [Float]? = nil
    ) -> [Float] {
        let kernelHeight = 3, kernelWidth = 3
        let outputHeight = height, outputWidth = width
        var output = [Float](repeating: 0, count: outputHeight * outputWidth * outputChannels)
        for oy in 0..<outputHeight {
            for ox in 0..<outputWidth {
                for oc in 0..<outputChannels {
                    var sum: Float = bias?[oc] ?? 0
                    for ic in 0..<inputChannels {
                        for ky in 0..<kernelHeight {
                            for kx in 0..<kernelWidth {
                                let iy = oy + ky - 1
                                let ix = ox + kx - 1
                                guard iy >= 0, ix >= 0, iy < height, ix < width else { continue }
                                let inputValue = input[(iy * width + ix) * inputChannels + ic]
                                let weightIndex =
                                    ((oc * inputChannels + ic) * kernelHeight + ky) * kernelWidth + kx
                                sum += inputValue * weight[weightIndex]
                            }
                        }
                    }
                    output[(oy * outputWidth + ox) * outputChannels + oc] = sum
                }
            }
        }
        return output
    }

    /// Build non-symmetric rank-5 [Cout,Cin,3,3,3] weight where the final
    /// temporal slice differs from the earlier ones.
    private func rank5Weight3x3(
        outputChannels: Int, inputChannels: Int, seed: Int
    ) -> [Float16] {
        var values: [Float16] = []
        var state = UInt64(seed) &+ 0x9E3779B97F4A7C15
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max) * 2 - 1
        }
        for _ in 0..<(outputChannels * inputChannels * 27) {
            values.append(Float16(next()))
        }
        return values
    }

    private func foldFinalSlice(_ weight: [Float16], cout: Int, cin: Int) -> [Float] {
        var folded = [Float](repeating: 0, count: cout * cin * 9)
        for oc in 0..<cout {
            for ic in 0..<cin {
                for k in 0..<9 {
                    let src = ((oc * cin + ic) * 3 * 9) + 2 * 9 + k
                    folded[(oc * cin + ic) * 9 + k] = Float(weight[src])
                }
            }
        }
        return folded
    }

    // MARK: - 1x1 convolution

    func testOneByOneConvolutionMatchesReference() async throws {
        let context = try requireContext()
        let executor = FP16ConvolutionExecutor(context: context)
        let cin = 5, cout = 7, rows = 3  // non-square, odd channels
        var weight: [Float16] = []
        for oc in 0..<cout {
            for ic in 0..<cin {
                weight.append(Float16(Float(oc * 10 + ic + 1) / 8))
            }
        }
        let input: [Float16] = [1, -2, 0.5, 3, -1,
                                0, 4, -3, 2, 1,
                                2, 1, -1, 0.5, -4]
        let inputBuffer = makeBuffer(input, on: context.device)
        let weightBuffer = makeBuffer(weight, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: rows * cout * 2, options: .storageModeShared))

        // Production contract: weights are folded into row-padded MPS scratch.
        let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: cin, dataType: .float16)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        let folded = try executor.encodeFoldWeight(
            commandBuffer: command, source: weightBuffer, sourceOffset: 0,
            shape: [cout, cin, 1, 1], outputRowStrideElements: rowBytes / 2,
            scratchKey: "test.weight.1x1")
        try executor.encode1x1(
            commandBuffer: command, input: inputBuffer, weight: folded, weightOffset: 0,
            output: outputBuffer, rows: rows,
            inputChannels: cin, outputChannels: cout)
        command.commit()
        command.waitUntilCompleted()

        let actual = readHalf(outputBuffer, count: rows * cout)
        for row in 0..<rows {
            for oc in 0..<cout {
                var expected: Float = 0
                for ic in 0..<cin {
                    expected += Float(input[row * cin + ic]) * Float(weight[oc * cin + ic])
                }
                XCTAssertEqual(actual[row * cout + oc], expected, accuracy: 0.01,
                               "1x1 row \(row) out \(oc)")
            }
        }
        print("VAE_1X1_CONV=PASS")
    }

    // MARK: - 3x3 convolution (rank-4 weights)

    func testThreeByThreeConvolutionMatchesReference() async throws {
        let context = try requireContext()
        let executor = FP16ConvolutionExecutor(context: context)
        let cin = 4, cout = 6, height = 5, width = 7  // non-square
        var weight = [Float16](repeating: 0, count: cout * cin * 9)
        var state = UInt64(12345)
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max) * 2 - 1
        }
        for i in 0..<weight.count { weight[i] = Float16(next()) }
        let bias = (0..<cout).map { Float16(Float($0) / 16) }

        var input = [Float16](repeating: 0, count: height * width * cin)
        for i in 0..<input.count { input[i] = Float16(next()) }
        let inputBuffer = makeBuffer(input, on: context.device)
        let weightBuffer = makeBuffer(weight, on: context.device)
        let biasBuffer = makeBuffer(bias, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: height * width * cout * 2, options: .storageModeShared))

        let rowBytes = MPSMatrixDescriptor.rowBytes(
            fromColumns: cin * 9, dataType: .float16)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        let folded = try executor.encodeFoldWeight(
            commandBuffer: command, source: weightBuffer, sourceOffset: 0,
            shape: [cout, cin, 3, 3], outputRowStrideElements: rowBytes / 2,
            scratchKey: "test.weight.3x3")
        try executor.encode3x3(
            commandBuffer: command, input: inputBuffer, weight: folded, weightOffset: 0,
            bias: biasBuffer, biasOffset: 0,
            output: outputBuffer, inputHeight: height, inputWidth: width,
            outputChannels: cout, inputChannels: cin)
        command.commit()
        command.waitUntilCompleted()

        let actual = readHalf(outputBuffer, count: height * width * cout)
        let expected = referenceConv(
            input.map { Float($0) }, height: height, width: width, inputChannels: cin,
            weight: weight.map { Float($0) }, outputChannels: cout,
            bias: bias.map { Float($0) })
        for i in 0..<expected.count {
            XCTAssertEqual(actual[i], expected[i], accuracy: 0.02,
                           "3x3 position \(i / cout) channel \(i % cout)")
        }
        print("VAE_3X3_CONV=PASS")
    }
}

extension VAEPrimitiveTests {
    // MARK: - Rank-5 fold + 3x3 (final temporal slice, D052)

    func testRank5FoldThreeByThreeMatchesReference() async throws {
        let context = try requireContext()
        let executor = FP16ConvolutionExecutor(context: context)
        let cin = 3, cout = 5, height = 4, width = 6
        let weightHalf = rank5Weight3x3(outputChannels: cout, inputChannels: cin, seed: 777)
        let folded = foldFinalSlice(weightHalf, cout: cout, cin: cin)
        let input = (0..<(height * width * cin)).map { Float16(Float($0 % 7) - 3) }
        let inputBuffer = makeBuffer(input, on: context.device)
        let weightBuffer = makeBuffer(weightHalf, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: height * width * cout * 2, options: .storageModeShared))

        // Production contract: rank-5 causal weight folded to its final slice
        // by encodeFoldWeight, then the 3x3 GEMM path.
        let rowBytes = MPSMatrixDescriptor.rowBytes(
            fromColumns: cin * 9, dataType: .float16)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        let foldedScratch = try executor.encodeFoldWeight(
            commandBuffer: command, source: weightBuffer, sourceOffset: 0,
            shape: [cout, cin, 3, 3, 3], outputRowStrideElements: rowBytes / 2,
            scratchKey: "test.weight.rank5")
        try executor.encode3x3(
            commandBuffer: command, input: inputBuffer, weight: foldedScratch, weightOffset: 0,
            bias: nil, biasOffset: 0,
            output: outputBuffer, inputHeight: height, inputWidth: width,
            outputChannels: cout, inputChannels: cin)
        command.commit()
        command.waitUntilCompleted()

        let actual = readHalf(outputBuffer, count: height * width * cout)
        let expected = referenceConv(
            input.map { Float($0) }, height: height, width: width, inputChannels: cin,
            weight: folded, weightOffset: 0, outputChannels: cout)
        for i in 0..<expected.count {
            XCTAssertEqual(actual[i], expected[i], accuracy: 0.02,
                           "folded rank-5 position \(i / cout) channel \(i % cout)")
        }

        // Prove the fold kernel itself picks the FINAL slice: a weight whose
        // final slice is zero must produce zero output.
        var zeroTail = [Float16](repeating: 0, count: cout * cin * 27)
        for oc in 0..<cout {
            for ic in 0..<cin {
                for k in 0..<9 {
                    zeroTail[((oc * cin + ic) * 3 * 9) + 0 * 9 + k] = Float16(1)
                }
            }
        }
        let zeroTailBuffer = makeBuffer(zeroTail, on: context.device)
        let scratchRowBytes = MPSMatrixDescriptor.rowBytes(
            fromColumns: cin * 9, dataType: .float16)
        guard let tailCommand = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        let tailScratch = try executor.encodeFoldWeight(
            commandBuffer: tailCommand, source: zeroTailBuffer, sourceOffset: 0,
            shape: [cout, cin, 3, 3, 3],
            outputRowStrideElements: scratchRowBytes / 2,
            scratchKey: "vae.weight.fold.tail")
        tailCommand.commit()
        tailCommand.waitUntilCompleted()
        let foldedValues = readHalf(tailScratch, count: cout * cin * 9)
        XCTAssertTrue(foldedValues.allSatisfy { $0 == 0 },
                      "rank-5 fold must use the final temporal slice")
        print("VAE_RANK5_FOLD=PASS")
    }

    // MARK: - Fused nearest-exact 2x upsample + 3x3 conv

    func testFusedUpsampleConvolutionMatchesUnfused() async throws {
        let context = try requireContext()
        let executor = FP16ConvolutionExecutor(context: context)
        let cin = 2, cout = 3, height = 3, width = 4
        var weight = [Float16](repeating: 0, count: cout * cin * 9)
        var state = UInt64(4242)
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max) * 2 - 1
        }
        for i in 0..<weight.count { weight[i] = Float16(next()) }
        let bias = (0..<cout).map { Float16(Float($0 + 1) / 8) }
        let input = (0..<(height * width * cin)).map { Float16(Float($0 % 5) - 2) }
        let inputBuffer = makeBuffer(input, on: context.device)
        let weightBuffer = makeBuffer(weight, on: context.device)
        let biasBuffer = makeBuffer(bias, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: height * width * 4 * cout * 2, options: .storageModeShared))

        let rowBytes = MPSMatrixDescriptor.rowBytes(
            fromColumns: cin * 9, dataType: .float16)
        guard let command = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        let folded = try executor.encodeFoldWeight(
            commandBuffer: command, source: weightBuffer, sourceOffset: 0,
            shape: [cout, cin, 3, 3], outputRowStrideElements: rowBytes / 2,
            scratchKey: "test.weight.upsample")
        try executor.encode3x3(
            commandBuffer: command, input: inputBuffer, weight: folded, weightOffset: 0,
            bias: biasBuffer, biasOffset: 0,
            output: outputBuffer, inputHeight: height, inputWidth: width,
            outputChannels: cout, inputChannels: cin, upsample2x: true)
        command.commit()
        command.waitUntilCompleted()

        // Unfused CPU reference: nearest-exact 2x then ordinary 3x3 conv.
        let upHeight = height * 2, upWidth = width * 2
        var upsampled = [Float](repeating: 0, count: upHeight * upWidth * cin)
        for y in 0..<upHeight {
            for x in 0..<upWidth {
                for c in 0..<cin {
                    upsampled[(y * upWidth + x) * cin + c] =
                        Float(input[(y / 2 * width + x / 2) * cin + c])
                }
            }
        }
        let expected = referenceConv(
            upsampled, height: upHeight, width: upWidth, inputChannels: cin,
            weight: weight.map { Float($0) }, outputChannels: cout,
            bias: bias.map { Float($0) })
        let actual = readHalf(outputBuffer, count: upHeight * upWidth * cout)
        for i in 0..<expected.count {
            XCTAssertEqual(actual[i], expected[i], accuracy: 0.03,
                           "fused upsample position \(i / cout) channel \(i % cout)")
        }
        print("VAE_FUSED_UPSAMPLE_CONV=PASS")
    }

    // MARK: - Channel RMS norm (Wan F.normalize x sqrt(C) x gamma)

    func testChannelRMSNormMatchesReference() async throws {
        let context = try requireContext()
        let positions = 3, channels = 5
        let input: [Float16] = [3, 0, 4, 0, 1,   -1, 2, -2, 0, 1,   0, 0, 0, 0, 0]
        let gamma: [Float16] = [1, 2, 0.5, 1.5, 1]
        let inputBuffer = makeBuffer(input, on: context.device)
        let gammaBuffer = makeBuffer(gamma, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: positions * channels * 2, options: .storageModeShared))
        guard let command = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        var positionsU = UInt32(positions), channelsU = UInt32(channels)
        let pipeline = try context.pipeline(named: "vae_channel_rmsnorm_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            return XCTFail("no encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gammaBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&positionsU, length: 4, index: 3)
        encoder.setBytes(&channelsU, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: positions, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

        let actual = readHalf(outputBuffer, count: positions * channels)
        let scale = sqrt(Float(channels))
        for p in 0..<positions {
            var sum: Float = 0
            for c in 0..<channels { sum += Float(input[p * channels + c]) * Float(input[p * channels + c]) }
            let inverse = 1 / max(sqrt(sum), 1e-12)
            for c in 0..<channels {
                let expected = Float(input[p * channels + c]) * inverse * scale * Float(gamma[c])
                XCTAssertEqual(actual[p * channels + c], expected, accuracy: 0.01,
                               "RMS position \(p) channel \(c)")
            }
        }
        print("VAE_CHANNEL_RMS=PASS")
    }

    // MARK: - SiLU half

    func testSiluHalfMatchesReference() async throws {
        let context = try requireContext()
        let input: [Float16] = [-3, -1, 0, 0.5, 2, 5]
        let inputBuffer = makeBuffer(input, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: input.count * 2, options: .storageModeShared))
        guard let command = context.commandQueue.makeCommandBuffer() else {
            return XCTFail("no command buffer")
        }
        var count = UInt32(input.count)
        let pipeline = try context.pipeline(named: "silu_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            return XCTFail("no encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: input.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        let actual = readHalf(outputBuffer, count: input.count)
        for i in 0..<input.count {
            let x = Float(input[i])
            let expected = x / (1 + exp(-x))
            XCTAssertEqual(actual[i], expected, accuracy: 0.01, "SiLU \(i)")
        }
        print("VAE_SILU_HALF=PASS")
    }
}

extension VAEPrimitiveTests {
    // MARK: - VAE attention QKV-split wiring

    /// The VAE middle attention stores Q, K, V as contiguous [positions, C]
    /// slices inside one [positions, 3C] fp16 buffer. Prove AttentionExecutor
    /// with heads=1 reads those offsets and computes the same single-head
    /// spatial attention as a CPU reference.
    func testAttentionQKVSplitWiring() async throws {
        let context = try requireContext()
        let executor = AttentionExecutor(context: context)
        let positions = 3, channels = 4
        let half = MemoryLayout<Float16>.stride

        // qkv buffer: [positions, 3*channels]; fill with distinct values.
        var qkv = [Float16](repeating: 0, count: positions * channels * 3)
        for p in 0..<positions {
            for c in 0..<channels {
                qkv[p * channels * 3 + c] = Float16(Float(p * 100 + c + 1) / 8)      // Q
                qkv[p * channels * 3 + channels + c] = Float16(Float(p * 10 + c + 1) / 16)  // K
                qkv[p * channels * 3 + 2 * channels + c] = Float16(Float(p + c) / 4)       // V
            }
        }
        let qkvBuffer = makeBuffer(qkv, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(
            length: positions * channels * 2, options: .storageModeShared))

        try await executor.execute(
            query: qkvBuffer, queryOffset: 0,
            key: qkvBuffer, keyOffset: positions * channels * half,
            value: qkvBuffer, valueOffset: positions * channels * 2 * half,
            output: outputBuffer,
            heads: 1, queryCount: positions, keyCount: positions,
            headDim: channels, causal: false)

        // CPU reference: softmax(q @ k.T / sqrt(C)) @ v
        var q = [[Float]](repeating: [Float](repeating: 0, count: channels), count: positions)
        var k = [[Float]](repeating: [Float](repeating: 0, count: channels), count: positions)
        var v = [[Float]](repeating: [Float](repeating: 0, count: channels), count: positions)
        for p in 0..<positions {
            for c in 0..<channels {
                q[p][c] = Float(qkv[p * channels * 3 + c])
                k[p][c] = Float(qkv[p * channels * 3 + channels + c])
                v[p][c] = Float(qkv[p * channels * 3 + 2 * channels + c])
            }
        }
        let scale = 1 / sqrt(Double(channels))
        let actual = readHalf(outputBuffer, count: positions * channels)
        for i in 0..<positions {
            var weights = [Double](repeating: 0, count: positions)
            var maxScore = -Double.infinity
            for j in 0..<positions {
                var dot = 0.0
                for c in 0..<channels { dot += Double(q[i][c]) * Double(k[j][c]) }
                weights[j] = dot * scale
                maxScore = max(maxScore, weights[j])
            }
            var total = 0.0
            for j in 0..<positions { weights[j] = exp(weights[j] - maxScore); total += weights[j] }
            for c in 0..<channels {
                var expected = 0.0
                for j in 0..<positions { expected += (weights[j] / total) * Double(v[j][c]) }
                XCTAssertEqual(Float(actual[i * channels + c]), Float(expected), accuracy: 0.02,
                               "attention query \(i) channel \(c)")
            }
        }
        print("VAE_ATTENTION_SPLIT=PASS")
    }
}
