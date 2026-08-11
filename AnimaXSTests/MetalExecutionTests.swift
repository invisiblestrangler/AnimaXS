import XCTest
import Metal
import MetalPerformanceShaders
@testable import AnimaXS

/// Small, pack-free execution tests. These are intentionally part of normal CI:
/// the standard macos-15 arm64 simulator runner has been verified to execute both
/// project Metal kernels and MPS operations (final snapshot CI run 31452206651).
final class MetalExecutionTests: XCTestCase {
    private func makeBuffer<T>(_ values: [T], on device: MTLDevice) -> MTLBuffer {
        values.withUnsafeBytes { bytes in
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: bytes.count,
                options: .storageModeShared
            )!
        }
    }

    private func requireContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: MTLCreateSystemDefaultDevice/default library unavailable")
        }
        print("METAL_DEVICE=\(context.device.name) registryID=\(context.device.registryID)")
        return context
    }

    func testW4KernelExecutes() throws {
        let context = try requireContext()
        let pipeline = try context.pipeline(named: "dequant_w4_to_half")
        let packed = makeBuffer([UInt8(0x6E)], on: context.device)
        let scale = makeBuffer([Float16(0.5).bitPattern], on: context.device)
        let zero = makeBuffer([Float16(-0.25).bitPattern], on: context.device)
        let output = try XCTUnwrap(context.device.makeBuffer(length: 4, options: .storageModeShared))
        var columns: UInt32 = 2
        var packedRowStride: UInt32 = 1
        var rows: UInt32 = 1
        var outputStride = columns

        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packed, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(zero, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&packedRowStride, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: 4, height: 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 2, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, .completed)
        XCTAssertNil(commandBuffer.error)
        let values = output.contents().bindMemory(to: Float16.self, capacity: 2)
        XCTAssertEqual(Float(values[0]), 6.75, accuracy: 0.001)
        XCTAssertEqual(Float(values[1]), 2.75, accuracy: 0.001)
        print("METAL_KERNEL_SMOKE=PASS")
    }

    func testRowAwareDequantizationAndBounds() throws {
        let context = try requireContext()
        try checkW4Rows(context)
        try checkW8Rows(context)
    }

    private func checkW4Rows(_ context: MetalContext) throws {
        let rows = 2, columns = 68, stride = (columns + 1) / 2, groups = 2
        var packed = [UInt8](repeating: 0, count: rows * stride)
        for row in 0..<rows {
            for column in 0..<columns {
                let q = UInt8((row * 7 + column) & 15)
                let index = row * stride + column / 2
                packed[index] |= column.isMultiple(of: 2) ? q : q << 4
            }
        }
        let scales: [Float16] = [0.5, 0.25, 1.0, 2.0]
        let zeros: [Float16] = [-1, 1, 3, -2]
        let output = try runDequant(
            context, kernel: "dequant_w4_to_half", packed: packed,
            scales: scales, zeros: zeros, rows: rows, columns: columns,
            rowStride: stride)
        for row in 0..<rows {
            for column in 0..<columns {
                let q = Float((row * 7 + column) & 15)
                let group = row * groups + column / 64
                let expected = Float16(q * Float(scales[group]) + Float(zeros[group])).bitPattern
                XCTAssertEqual(output[row * columns + column], expected)
            }
        }
    }

    private func checkW8Rows(_ context: MetalContext) throws {
        let rows = 2, columns = 65, groups = 2
        let packed = (0..<(rows * columns)).map { UInt8(($0 * 13) & 255) }
        let scales: [Float16] = [0.125, 0.5, 0.25, 2]
        let zeros: [Float16] = [-2, 1, 4, -3]
        let output = try runDequant(
            context, kernel: "dequant_w8_to_half", packed: packed,
            scales: scales, zeros: zeros, rows: rows, columns: columns,
            rowStride: columns)
        for row in 0..<rows {
            for column in 0..<columns {
                let q = Float(packed[row * columns + column])
                let group = row * groups + column / 64
                let expected = Float16(q * Float(scales[group]) + Float(zeros[group])).bitPattern
                XCTAssertEqual(output[row * columns + column], expected)
            }
        }
    }

    private func runDequant(
        _ context: MetalContext, kernel: String, packed: [UInt8],
        scales: [Float16], zeros: [Float16], rows: Int, columns: Int,
        rowStride: Int
    ) throws -> [UInt16] {
        let pipeline = try context.pipeline(named: kernel)
        let packedBuffer = makeBuffer(packed, on: context.device)
        let scaleBuffer = makeBuffer(scales.map(\.bitPattern), on: context.device)
        let zeroBuffer = makeBuffer(zeros.map(\.bitPattern), on: context.device)
        let guardValue = Float16.nan.bitPattern
        let outputCount = rows * columns + 32
        let outputBuffer = makeBuffer([UInt16](repeating: guardValue, count: outputCount), on: context.device)
        var k = UInt32(columns), stride = UInt32(rowStride), rowCount = UInt32(rows)
        var outputStride = UInt32(columns)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(zeroBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&k, length: 4, index: 4)
        encoder.setBytes(&stride, length: 4, index: 5)
        encoder.setBytes(&rowCount, length: 4, index: 6)
        encoder.setBytes(&outputStride, length: 4, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: columns + 7, height: rows + 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: UInt16.self, capacity: outputCount)
        let result = Array(UnsafeBufferPointer(start: pointer, count: outputCount))
        XCTAssertTrue(result[(rows * columns)...].allSatisfy { $0 == guardValue }, "kernel wrote past logical output")
        return Array(result.prefix(rows * columns))
    }

    func testMetalDiagnosticsPopulate() throws {
        let context = try requireContext()
        context.refreshDiagnostics()
        XCTAssertGreaterThan(context.maxBufferLength, 0)
        XCTAssertGreaterThan(context.maxThreadgroupMemoryLength, 0)
        XCTAssertGreaterThan(context.physicalMemory, 0)
        if context.recommendedMaxWorkingSetSize > 0 {
            XCTAssertLessThanOrEqual(context.currentAllocatedSize, context.recommendedMaxWorkingSetSize)
        }
        _ = context.supportsApple5
        _ = context.thermalState
    }

    func testExactArithmeticKernels() throws {
        let context = try requireContext()
        try checkActivations(context)
        try checkNorms(context)
        try checkModulationAndResiduals(context)
    }

    private func checkActivations(_ context: MetalContext) throws {
        let input: [Float] = [-4, -1, -0.25, 0, 0.5, 1, 3]
        let geluValues = try runUnaryFloat(context, kernel: "gelu", input: input)
        let siluValues = try runUnaryFloat(context, kernel: "silu", input: input)
        for i in input.indices {
            let x = input[i]
            let exactGELU = 0.5 * x * (1 + Float(erf(Double(x) / sqrt(2.0))))
            let exactSiLU = x / (1 + exp(-x))
            XCTAssertEqual(geluValues[i], exactGELU, accuracy: 2e-6)
            XCTAssertEqual(siluValues[i], exactSiLU, accuracy: 2e-6)
        }
        // x=3 distinguishes exact erf GELU from the removed tanh approximation.
        XCTAssertGreaterThan(abs(geluValues[6] - 2.9963627), 0.0003)
    }

    private func runUnaryFloat(
        _ context: MetalContext, kernel: String, input: [Float]
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: kernel)
        let inputBuffer = makeBuffer(input, on: context.device)
        let sentinel: Float = 1_234_567
        let outputBuffer = makeBuffer(
            [Float](repeating: sentinel, count: input.count + 8), on: context.device)
        var count = UInt32(input.count)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: input.count + 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(32, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count + 8)
        let values = Array(UnsafeBufferPointer(start: pointer, count: input.count + 8))
        XCTAssertTrue(values[input.count...].allSatisfy { $0 == sentinel })
        return Array(values.prefix(input.count))
    }

    private func checkNorms(_ context: MetalContext) throws {
        let rows = 2, columns = 7
        let input: [Float] = [-3, -1, 0, 1, 2, 4, 7, 9, 5, 2, -2, -4, 0.5, 3]
        let weight: [Float16] = [0.5, 1, 1.5, 2, -0.5, 0.25, 3]
        let eps: Float = 1e-6
        let rms = try runNorm(
            context, kernel: "rmsnorm_f32_to_f32", input: input,
            weight: weight, rows: rows, columns: columns, eps: eps)
        let layer = try runNorm(
            context, kernel: "layernorm_f32_to_f32", input: input,
            weight: nil, rows: rows, columns: columns, eps: eps)
        for row in 0..<rows {
            let values = Array(input[(row * columns)..<((row + 1) * columns)])
            let squareMean = values.reduce(Float.zero) { $0 + $1 * $1 } / Float(columns)
            let mean = values.reduce(0, +) / Float(columns)
            let variance = values.reduce(Float.zero) { $0 + ($1 - mean) * ($1 - mean) } / Float(columns)
            for column in 0..<columns {
                let index = row * columns + column
                let expectedRMS = values[column] / sqrt(squareMean + eps) * Float(weight[column])
                let expectedLayer = (values[column] - mean) / sqrt(variance + eps)
                XCTAssertEqual(rms[index], expectedRMS, accuracy: 3e-5)
                XCTAssertEqual(layer[index], expectedLayer, accuracy: 3e-5)
            }
        }
    }

    private func runNorm(
        _ context: MetalContext, kernel: String, input: [Float],
        weight: [Float16]?, rows: Int, columns: Int, eps: Float
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: kernel)
        let inputBuffer = makeBuffer(input, on: context.device)
        let outputBuffer = makeBuffer(
            [Float](repeating: 1_234_567, count: input.count + columns), on: context.device)
        let weightBuffer = makeBuffer((weight ?? [0]).map(\.bitPattern), on: context.device)
        var n = UInt32(columns), epsilon = eps, rowCount = UInt32(rows)
        var useWeight: UInt32 = weight == nil ? 0 : 1
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        if kernel == "rmsnorm_f32_to_f32" {
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.setBytes(&epsilon, length: 4, index: 4)
            encoder.setBytes(&useWeight, length: 4, index: 5)
            encoder.setBytes(&rowCount, length: 4, index: 6)
        } else {
            encoder.setBytes(&n, length: 4, index: 2)
            encoder.setBytes(&epsilon, length: 4, index: 3)
            encoder.setBytes(&rowCount, length: 4, index: 4)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: rows + 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count + columns)
        let result = Array(UnsafeBufferPointer(start: pointer, count: input.count + columns))
        XCTAssertTrue(result[input.count...].allSatisfy { $0 == 1_234_567 })
        return Array(result.prefix(input.count))
    }

    private func checkModulationAndResiduals(_ context: MetalContext) throws {
        let n = 3
        let normalized: [Float] = [-1, 0, 2, 3, -2, 0.5]
        let scale: [Float] = [0.5, -0.25, 2]
        let shift: [Float] = [1, -1, 0.25]
        let modulated = try runModulate(
            context, normalized: normalized, scale: scale, shift: shift, columns: n)
        for i in normalized.indices {
            XCTAssertEqual(modulated[i], normalized[i] * (1 + scale[i % n]) + shift[i % n], accuracy: 1e-6)
        }

        let branch: [Float16] = [0.5, -2, 3, -4, 0.25, 1.5]
        let gate: [Float] = [2, -0.5, 0.25]
        let residual: [Float] = [1, 2, 3, 4, 5, 6]
        let gated = try runResidual(
            context, kernel: "gate_add_half_f32", residual: residual,
            branch: branch, gate: gate, columns: n)
        let added = try runResidual(
            context, kernel: "add_half_into_float", residual: residual,
            branch: branch, gate: nil, columns: n)
        for i in residual.indices {
            XCTAssertEqual(gated[i], residual[i] + Float(branch[i]) * gate[i % n], accuracy: 1e-6)
            XCTAssertEqual(added[i], residual[i] + Float(branch[i]), accuracy: 1e-6)
        }
    }

    private func runModulate(
        _ context: MetalContext, normalized: [Float], scale: [Float],
        shift: [Float], columns: Int
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: "modulate_f32")
        let inputs = makeBuffer(normalized, on: context.device)
        let scales = makeBuffer(scale, on: context.device)
        let shifts = makeBuffer(shift, on: context.device)
        let output = makeBuffer([Float](repeating: 1_234_567, count: normalized.count + 8), on: context.device)
        var n = UInt32(columns), count = UInt32(normalized.count)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: 0, index: 1)
        encoder.setBuffer(shifts, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        encoder.setBytes(&count, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: normalized.count + 8, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = output.contents().bindMemory(to: Float.self, capacity: normalized.count + 8)
        let values = Array(UnsafeBufferPointer(start: pointer, count: normalized.count + 8))
        XCTAssertTrue(values[normalized.count...].allSatisfy { $0 == 1_234_567 })
        return Array(values.prefix(normalized.count))
    }

    private func runResidual(
        _ context: MetalContext, kernel: String, residual: [Float],
        branch: [Float16], gate: [Float]?, columns: Int
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: kernel)
        let residualBuffer = makeBuffer(residual + [Float](repeating: 1_234_567, count: 8), on: context.device)
        let branchBuffer = makeBuffer(branch.map(\.bitPattern), on: context.device)
        let gateBuffer = makeBuffer(gate ?? [0], on: context.device)
        var n = UInt32(columns), count = UInt32(residual.count)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residualBuffer, offset: 0, index: 0)
        encoder.setBuffer(branchBuffer, offset: 0, index: 1)
        if gate != nil {
            encoder.setBuffer(gateBuffer, offset: 0, index: 2)
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.setBytes(&count, length: 4, index: 4)
        } else {
            encoder.setBytes(&count, length: 4, index: 2)
        }
        encoder.dispatchThreads(MTLSize(width: residual.count + 8, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = residualBuffer.contents().bindMemory(to: Float.self, capacity: residual.count + 8)
        let values = Array(UnsafeBufferPointer(start: pointer, count: residual.count + 8))
        XCTAssertTrue(values[residual.count...].allSatisfy { $0 == 1_234_567 })
        return Array(values.prefix(residual.count))
    }

    func testRoPEPatchAndEulerKernels() throws {
        let context = try requireContext()
        try checkSplitHalfRoPE(context)
        try checkPatchRoundTrip(context)
        try checkEuler(context)
    }

    private func checkSplitHalfRoPE(_ context: MetalContext) throws {
        let tokens = 2, heads = 2, headDimension = 128
        let input: [Float16] = (0..<(tokens * heads * headDimension)).map {
            Float16(Float(($0 % 19) - 9) / 4)
        }
        let weight: [Float16] = (0..<headDimension).map {
            Float16(Float(($0 % 7) + 1) / 7)
        }
        let rope = DitRoPE.generate(T: 1, H: 1, W: 2)
        let pipeline = try context.pipeline(named: "rms_rope_split_half")
        let inputBuffer = makeBuffer(input.map(\.bitPattern), on: context.device)
        let weightBuffer = makeBuffer(weight.map(\.bitPattern), on: context.device)
        let ropeBuffer = makeBuffer(rope, on: context.device)
        let logicalCount = input.count
        let sentinel = Float16.nan.bitPattern
        let outputBuffer = makeBuffer(
            [UInt16](repeating: sentinel, count: logicalCount + headDimension), on: context.device)
        var tokenCount = UInt32(tokens), headCount = UInt32(heads), eps: Float = 1e-6
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(ropeBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&tokenCount, length: 4, index: 4)
        encoder.setBytes(&headCount, length: 4, index: 5)
        encoder.setBytes(&eps, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokens * heads + 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: UInt16.self, capacity: logicalCount + headDimension)
        let result = Array(UnsafeBufferPointer(start: pointer, count: logicalCount + headDimension))
        XCTAssertTrue(result[logicalCount...].allSatisfy { $0 == sentinel })

        for row in 0..<(tokens * heads) {
            let base = row * headDimension
            var sum: Float = 0
            for d in 0..<headDimension {
                let value = Float(input[base + d])
                sum += value * value
            }
            let inv = 1 / sqrt(sum / Float(headDimension) + eps)
            let token = row / heads
            for p in 0..<64 {
                let a = Float(input[base + p]) * inv * Float(weight[p])
                let b = Float(input[base + p + 64]) * inv * Float(weight[p + 64])
                let ropeBase = (token * 64 + p) * 4
                let expectedFirst = Float16(rope[ropeBase] * a + rope[ropeBase + 1] * b)
                let expectedSecond = Float16(rope[ropeBase + 2] * a + rope[ropeBase + 3] * b)
                XCTAssertEqual(Float(Float16(bitPattern: result[base + p])), Float(expectedFirst), accuracy: 0.002)
                XCTAssertEqual(Float(Float16(bitPattern: result[base + p + 64])), Float(expectedSecond), accuracy: 0.002)
            }
        }
        // Token 1 has a non-identity width rotation and distinguishes split-half
        // (p,p+64) from the common but incorrect adjacent pairing.
        XCTAssertNotEqual(result[2 * headDimension + 43], input[2 * headDimension + 43].bitPattern)
    }

    private func checkPatchRoundTrip(_ context: MetalContext) throws {
        let height = 4, width = 6, channels = 17
        let input: [Float] = (0..<(channels * height * width)).map {
            Float(($0 * 17) % 101) / 10 - 5
        }
        let tokenCount = height / 2 * width / 2
        let tokenValues = try runPatchify(
            context, input: input, height: height, width: width, tokenCount: tokenCount)
        for token in 0..<tokenCount {
            let patchRow = token / (width / 2)
            let patchColumn = token % (width / 2)
            for channel in 0..<channels {
                for di in 0..<2 {
                    for dj in 0..<2 {
                        let source = channel * height * width
                            + (patchRow * 2 + di) * width + patchColumn * 2 + dj
                        let target = token * 68 + channel * 4 + di * 2 + dj
                        XCTAssertEqual(tokenValues[target], input[source])
                    }
                }
            }
        }
        let restored = try runUnpatchify(
            context, tokens: tokenValues, height: height, width: width, tokenCount: tokenCount)
        XCTAssertEqual(restored, Array(input.prefix(16 * height * width)))

        // FinalLayer has no four-channel mask tail: its token stride is 64, not 68.
        let velocityTokens = (0..<(tokenCount * 64)).map(Float.init)
        let velocity = try runVelocityUnpatchify(
            context, tokens: velocityTokens, height: height, width: width,
            tokenCount: tokenCount)
        for token in 0..<tokenCount {
            let patchRow = token / (width / 2)
            let patchColumn = token % (width / 2)
            for channel in 0..<16 {
                for di in 0..<2 {
                    for dj in 0..<2 {
                        let source = token * 64 + (di * 2 + dj) * 16 + channel
                        let target = channel * height * width
                            + (patchRow * 2 + di) * width + patchColumn * 2 + dj
                        XCTAssertEqual(velocity[target], velocityTokens[source])
                    }
                }
            }
        }
    }

    private func runPatchify(
        _ context: MetalContext, input: [Float], height: Int, width: Int, tokenCount: Int
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: "patchify17")
        let inputBuffer = makeBuffer(input, on: context.device)
        let logicalCount = tokenCount * 68
        let outputBuffer = makeBuffer([Float](repeating: 1_234_567, count: logicalCount + 68), on: context.device)
        var h = UInt32(height), w = UInt32(width)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&h, length: 4, index: 2)
        encoder.setBytes(&w, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: tokenCount + 2, height: 19, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 2, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: logicalCount + 68)
        let values = Array(UnsafeBufferPointer(start: pointer, count: logicalCount + 68))
        XCTAssertTrue(values[logicalCount...].allSatisfy { $0 == 1_234_567 })
        return Array(values.prefix(logicalCount))
    }

    private func runUnpatchify(
        _ context: MetalContext, tokens: [Float], height: Int, width: Int, tokenCount: Int
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: "unpatchify16")
        let tokenBuffer = makeBuffer(tokens, on: context.device)
        let logicalCount = 16 * height * width
        let outputBuffer = makeBuffer([Float](repeating: 1_234_567, count: logicalCount + height * width), on: context.device)
        var h = UInt32(height), w = UInt32(width)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&h, length: 4, index: 2)
        encoder.setBytes(&w, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: tokenCount + 2, height: 18, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 2, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: logicalCount + height * width)
        let values = Array(UnsafeBufferPointer(start: pointer, count: logicalCount + height * width))
        XCTAssertTrue(values[logicalCount...].allSatisfy { $0 == 1_234_567 })
        return Array(values.prefix(logicalCount))
    }

    private func runVelocityUnpatchify(
        _ context: MetalContext, tokens: [Float], height: Int, width: Int, tokenCount: Int
    ) throws -> [Float] {
        let pipeline = try context.pipeline(named: "unpatchify_velocity16")
        let tokenBuffer = makeBuffer(tokens, on: context.device)
        let logicalCount = 16 * height * width
        let outputBuffer = makeBuffer(
            [Float](repeating: 1_234_567, count: logicalCount + height * width),
            on: context.device)
        var h = UInt32(height), w = UInt32(width)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&h, length: 4, index: 2)
        encoder.setBytes(&w, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: tokenCount + 2, height: 18, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 2, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(
            to: Float.self, capacity: logicalCount + height * width)
        let values = Array(UnsafeBufferPointer(
            start: pointer, count: logicalCount + height * width))
        XCTAssertTrue(values[logicalCount...].allSatisfy { $0 == 1_234_567 })
        return Array(values.prefix(logicalCount))
    }

    private func checkEuler(_ context: MetalContext) throws {
        let x: [Float] = [-2, -0.5, 0, 1, 3]
        let denoised: [Float] = [1, -1, 2, 0.25, -4]
        let pipeline = try context.pipeline(named: "euler_step_f32")
        let xBuffer = makeBuffer(x, on: context.device)
        let denoisedBuffer = makeBuffer(denoised, on: context.device)
        let outputBuffer = makeBuffer([Float](repeating: 1_234_567, count: x.count + 8), on: context.device)
        var sigma: Float = 0.75, deltaSigma: Float = -0.125
        var count = UInt32(x.count)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(denoisedBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&sigma, length: 4, index: 3)
        encoder.setBytes(&deltaSigma, length: 4, index: 4)
        encoder.setBytes(&count, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: x.count + 8, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: x.count + 8)
        let values = Array(UnsafeBufferPointer(start: pointer, count: x.count + 8))
        for i in x.indices {
            let expected = x[i] + deltaSigma * (x[i] - denoised[i]) / sigma
            XCTAssertEqual(values[i], expected, accuracy: 1e-6)
        }
        XCTAssertTrue(values[x.count...].allSatisfy { $0 == 1_234_567 })
    }

    func testDirectW4MatvecNonAlignedK() throws {
        let context = try requireContext()
        let rows = 4, columns = 68, rowStride = (columns + 1) / 2
        var packed = [UInt8](repeating: 0, count: rows * rowStride)
        for row in 0..<rows {
            for column in 0..<columns {
                let q = UInt8((row * 11 + column * 3 + 5) & 15)
                let index = row * rowStride + column / 2
                packed[index] |= column.isMultiple(of: 2) ? q : q << 4
            }
        }
        let scale: [Float16] = [
            0.0625, 0.25,
            0.125, 0.5,
            0.03125, 0.75,
            0.2, 0.4,
        ]
        let zero: [Float16] = [
            -0.5, 1.25,
            0.75, -2,
            -1.5, 0.25,
            2, -0.75,
        ]
        let input: [Float] = (0..<columns).map { column in
            let wave = sin(Float(column) * Float(0.37)) * Float(1.5)
            let offset = Float((column % 5) - 2) * Float(0.1)
            return wave + offset
        }
        let pipeline = try context.pipeline(named: "w4_matvec_f32")
        let packedBuffer = makeBuffer(packed, on: context.device)
        let scaleBuffer = makeBuffer(scale.map(\.bitPattern), on: context.device)
        let zeroBuffer = makeBuffer(zero.map(\.bitPattern), on: context.device)
        let inputBuffer = makeBuffer(input, on: context.device)
        let outputBuffer = makeBuffer([Float](repeating: 1_234_567, count: rows + 2), on: context.device)
        var k = UInt32(columns), rowCount = UInt32(rows), stride = UInt32(rowStride)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(zeroBuffer, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        encoder.setBytes(&k, length: 4, index: 5)
        encoder.setBytes(&rowCount, length: 4, index: 6)
        encoder.setBytes(&stride, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows + 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows + 2)
        let gpu = Array(UnsafeBufferPointer(start: pointer, count: rows + 2))
        XCTAssertEqual(gpu[rows], 1_234_567)
        XCTAssertEqual(gpu[rows + 1], 1_234_567)

        var reference = [Double](repeating: 0, count: rows)
        for row in 0..<rows {
            for column in 0..<columns {
                let byte = packed[row * rowStride + column / 2]
                let q = column.isMultiple(of: 2) ? byte & 15 : byte >> 4
                let group = row * 2 + column / 64
                let weight = Double(q) * Double(Float(scale[group])) + Double(Float(zero[group]))
                reference[row] += Double(input[column]) * weight
            }
        }
        var maxAbs = 0.0
        var dot = 0.0, gpuNorm = 0.0, referenceNorm = 0.0
        for row in 0..<rows {
            let error = abs(Double(gpu[row]) - reference[row])
            maxAbs = max(maxAbs, error)
            dot += Double(gpu[row]) * reference[row]
            gpuNorm += Double(gpu[row]) * Double(gpu[row])
            referenceNorm += reference[row] * reference[row]
        }
        let cosine = dot / sqrt(gpuNorm * referenceNorm)
        print("E005_W4_MATVEC maxAbs=\(maxAbs) cosine=\(cosine)")
        XCTAssertLessThan(maxAbs, 2e-4)
        XCTAssertGreaterThan(cosine, 0.9999999)
    }

    func testMPSMatrixMultiplicationExecutes() throws {
        let context = try requireContext()
        let input: [Float16] = [1, 2, 3, 4]
        let identity: [Float16] = [1, 0, 0, 1]
        let inputBuffer = makeBuffer(input, on: context.device)
        let identityBuffer = makeBuffer(identity, on: context.device)
        let outputBuffer = try XCTUnwrap(context.device.makeBuffer(length: 8, options: .storageModeShared))
        let descriptor = MPSMatrixDescriptor(
            rows: 2,
            columns: 2,
            rowBytes: 2 * MemoryLayout<Float16>.stride,
            dataType: .float16
        )
        let inputMatrix = MPSMatrix(buffer: inputBuffer, descriptor: descriptor)
        let identityMatrix = MPSMatrix(buffer: identityBuffer, descriptor: descriptor)
        let outputMatrix = MPSMatrix(buffer: outputBuffer, descriptor: descriptor)
        let multiplication = MPSMatrixMultiplication(
            device: context.device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: 2,
            resultColumns: 2,
            interiorColumns: 2,
            alpha: 1,
            beta: 0
        )

        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        multiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: inputMatrix,
            rightMatrix: identityMatrix,
            resultMatrix: outputMatrix
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, .completed)
        XCTAssertNil(commandBuffer.error)
        let values = outputBuffer.contents().bindMemory(to: Float16.self, capacity: 4)
        for index in input.indices {
            XCTAssertEqual(Float(values[index]), Float(input[index]), accuracy: 0.001)
        }
        print("MPS_GEMM_SMOKE=PASS")
    }

    func testQwenGatedSiLUHalfExecutes() throws {
        let context = try requireContext()
        let gate: [Float16] = [-2, -0.5, 0, 1, 3]
        let up: [Float16] = [0.5, -2, 4, 1.5, -0.25]
        let gateBuffer = makeBuffer(gate, on: context.device)
        let upBuffer = makeBuffer(up, on: context.device)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: gate.count * 2, options: .storageModeShared))
        let pipeline = try context.pipeline(named: "gated_silu_half")
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var count = UInt32(gate.count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: gate.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: gate.count, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let actual = output.contents().bindMemory(to: Float16.self, capacity: gate.count)
        for index in gate.indices {
            let x = Float(gate[index])
            let expected = Float(Float16((x / (1 + exp(-x))) * Float(up[index])))
            XCTAssertEqual(Float(actual[index]), expected, accuracy: 0)
        }
    }
}
