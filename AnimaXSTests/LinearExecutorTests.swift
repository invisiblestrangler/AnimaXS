import XCTest
import Metal
@testable import AnimaXS

final class LinearExecutorTests: XCTestCase {
    private let prefixBytes = 32
    private let suffixBytes = 32

    private func requireContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        print("LINEAR_METAL_DEVICE=\(context.device.name)")
        return context
    }

    private func makeBuffer(bytes: Int, on device: MTLDevice, fill: UInt8 = 0) throws -> MTLBuffer {
        let buffer = try XCTUnwrap(device.makeBuffer(length: bytes, options: .storageModeShared))
        memset(buffer.contents(), Int32(fill), bytes)
        return buffer
    }

    private func store<T>(_ value: T, in buffer: MTLBuffer, byteOffset: Int) {
        buffer.contents().advanced(by: byteOffset).storeBytes(of: value, as: T.self)
    }

    private func loadHalf(_ buffer: MTLBuffer, byteOffset: Int) -> Float {
        Float(Float16(bitPattern: buffer.contents().advanced(by: byteOffset)
            .load(as: UInt16.self)))
    }

    func testW8TransposedLinearUsesPaddedPackedStrideAndTailTile() async throws {
        let context = try requireContext()
        let m = 133, n = 16, k = 16, packedStride = 20
        let scalarBytes = MemoryLayout<Float16>.stride

        let input = try makeBuffer(
            bytes: prefixBytes + m * k * scalarBytes + suffixBytes,
            on: context.device, fill: 0xA5)
        var inputValues = [Float](repeating: 0, count: m * k)
        for row in 0..<m {
            for column in 0..<k {
                let value = Float(((row * 3 + column * 5) % 17) - 8) / 8
                inputValues[row * k + column] = Float(Float16(value))
                store(Float16(value).bitPattern, in: input,
                      byteOffset: prefixBytes + (row * k + column) * scalarBytes)
            }
        }

        let packed = try makeBuffer(
            bytes: prefixBytes + n * packedStride + suffixBytes,
            on: context.device, fill: 0xEE)
        var dequantized = [Float](repeating: 0, count: n * k)
        for row in 0..<n {
            let scale = Float16(Float(row + 1) / 64)
            let zero = Float16(Float(row - 3) / 16)
            for column in 0..<k {
                let q = UInt8((row * 19 + column * 7) & 31)
                store(q, in: packed, byteOffset: prefixBytes + row * packedStride + column)
                dequantized[row * k + column] = Float(Float16(Float(q) * Float(scale) + Float(zero)))
            }
        }
        let scales = try makeBuffer(bytes: prefixBytes + n * scalarBytes + suffixBytes, on: context.device)
        let zeros = try makeBuffer(bytes: prefixBytes + n * scalarBytes + suffixBytes, on: context.device)
        for row in 0..<n {
            store(Float16(Float(row + 1) / 64).bitPattern, in: scales,
                  byteOffset: prefixBytes + row * scalarBytes)
            store(Float16(Float(row - 3) / 16).bitPattern, in: zeros,
                  byteOffset: prefixBytes + row * scalarBytes)
        }
        let outputBytes = m * n * scalarBytes
        let output = try makeBuffer(
            bytes: prefixBytes + outputBytes + suffixBytes,
            on: context.device, fill: 0x7B)
        let weight = QuantizedLinearWeightBuffers(
            storage: .w8, packed: packed, packedOffset: prefixBytes,
            scale: scales, scaleOffset: prefixBytes, zero: zeros, zeroOffset: prefixBytes,
            rows: n, columns: k, packedRowStride: packedStride)

        let executor = LinearExecutor(context: context)
        try await executor.execute(
            input: input, inputOffset: prefixBytes, weight: weight,
            output: output, outputOffset: prefixBytes, inputRows: m)

        for row in 0..<m {
            for outputColumn in 0..<n {
                var expected: Double = 0
                for inner in 0..<k {
                    expected += Double(inputValues[row * k + inner])
                        * Double(dequantized[outputColumn * k + inner])
                }
                let actual = loadHalf(
                    output, byteOffset: prefixBytes + (row * n + outputColumn) * scalarBytes)
                XCTAssertEqual(actual, Float(expected), accuracy: 0.025,
                               "row \(row), output \(outputColumn)")
            }
        }
        let bytes = output.contents().bindMemory(to: UInt8.self, capacity: output.length)
        XCTAssertTrue((0..<prefixBytes).allSatisfy { bytes[$0] == 0x7B })
        XCTAssertTrue(((prefixBytes + outputBytes)..<output.length).allSatisfy { bytes[$0] == 0x7B })
        print("LINEAR_W8_TILED=PASS M=\(m) N=\(n) K=\(k) tile=\(executor.tileRows)")
    }

    func testW4TransposedLinearHonorsNibbleOrderAndPaddedRows() async throws {
        let context = try requireContext()
        let m = 3, n = 16, k = 16, packedStride = 12
        let scalarBytes = MemoryLayout<Float16>.stride
        let input = try makeBuffer(bytes: m * k * scalarBytes, on: context.device)
        var inputValues = [Float](repeating: 0, count: m * k)
        for index in inputValues.indices {
            let value = Float((index * 11) % 13 - 6) / 4
            inputValues[index] = Float(Float16(value))
            store(Float16(value).bitPattern, in: input, byteOffset: index * scalarBytes)
        }
        let packed = try makeBuffer(bytes: n * packedStride, on: context.device, fill: 0xCC)
        var dequantized = [Float](repeating: 0, count: n * k)
        for row in 0..<n {
            for column in 0..<k {
                let q = UInt8((row * 5 + column * 3) & 15)
                let offset = row * packedStride + column / 2
                let pointer = packed.contents().advanced(by: offset).assumingMemoryBound(to: UInt8.self)
                pointer.pointee = column.isMultiple(of: 2)
                    ? (pointer.pointee & 0xF0) | q
                    : (pointer.pointee & 0x0F) | (q << 4)
                dequantized[row * k + column] = Float(Float16(Float(q) * 0.125 - 0.5))
            }
        }
        let scales = try makeBuffer(bytes: n * scalarBytes, on: context.device)
        let zeros = try makeBuffer(bytes: n * scalarBytes, on: context.device)
        for row in 0..<n {
            store(Float16(0.125).bitPattern, in: scales, byteOffset: row * scalarBytes)
            store(Float16(-0.5).bitPattern, in: zeros, byteOffset: row * scalarBytes)
        }
        let output = try makeBuffer(bytes: m * n * scalarBytes, on: context.device)
        let weight = QuantizedLinearWeightBuffers(
            storage: .w4, packed: packed, packedOffset: 0,
            scale: scales, scaleOffset: 0, zero: zeros, zeroOffset: 0,
            rows: n, columns: k, packedRowStride: packedStride)

        try await LinearExecutor(context: context, tileRows: 2).execute(
            input: input, weight: weight, output: output, inputRows: m)

        for row in 0..<m {
            for outputColumn in 0..<n {
                var expected: Float = 0
                for inner in 0..<k {
                    expected += inputValues[row * k + inner]
                        * dequantized[outputColumn * k + inner]
                }
                XCTAssertEqual(loadHalf(output, byteOffset: (row * n + outputColumn) * scalarBytes),
                               expected, accuracy: 0.015)
            }
        }
    }

    func testLinearRejectsUndersizedPackedStride() throws {
        let context = try requireContext()
        let buffer = try makeBuffer(bytes: 64, on: context.device)
        let weight = QuantizedLinearWeightBuffers(
            storage: .w4, packed: buffer, packedOffset: 0,
            scale: buffer, scaleOffset: 0, zero: buffer, zeroOffset: 0,
            rows: 1, columns: 16, packedRowStride: 7)
        let commandBuffer = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        XCTAssertThrowsError(try LinearExecutor(context: context).encode(
            commandBuffer: commandBuffer, input: buffer, weight: weight,
            output: buffer, inputRows: 1))
    }

    func testPrecisionAtK2048And8192() async throws {
        let context = try requireContext()
        for k in [2_048, 8_192] {
            try await characterizePrecision(context: context, k: k)
        }
    }

    private func characterizePrecision(context: MetalContext, k: Int) async throws {
        let n = 16, groups = k / 64, packedStride = k / 2
        var state = UInt32(0xA51A_0000) &+ UInt32(k)
        let normalization = 1 / sqrt(Float(k))
        var inputFloat = [Float](repeating: 0, count: k)
        for column in 0..<k {
            state = state &* 1_664_525 &+ 1_013_904_223
            let signed = Float(Int((state >> 16) % 2_001) - 1_000) / 1_000
            inputFloat[column] = signed * normalization
        }
        let inputHalf = inputFloat.map(Float16.init)

        var packed = [UInt8](repeating: 0, count: n * packedStride)
        var scales = [Float16](repeating: 0, count: n * groups)
        var zeros = [Float16](repeating: 0, count: n * groups)
        for row in 0..<n {
            for group in 0..<groups {
                scales[row * groups + group] = Float16(Float(1 + (row + group) % 3) / 64)
                zeros[row * groups + group] = Float16(Float((row + group) % 5 - 2) / 16)
            }
            for column in 0..<k {
                let q = UInt8((row * 11 + column * 7 + column / 64 * 3) & 15)
                let byte = row * packedStride + column / 2
                packed[byte] |= column.isMultiple(of: 2) ? q : q << 4
            }
        }

        var directReference64 = [Double](repeating: 0, count: n)
        var directReference32 = [Float](repeating: 0, count: n)
        var mpsReference64 = [Double](repeating: 0, count: n)
        for row in 0..<n {
            for column in 0..<k {
                let byte = packed[row * packedStride + column / 2]
                let q = column.isMultiple(of: 2) ? byte & 15 : byte >> 4
                let parameter = row * groups + column / 64
                let weightFloat = Float(q) * Float(scales[parameter]) + Float(zeros[parameter])
                directReference64[row] += Double(inputFloat[column]) * Double(weightFloat)
                directReference32[row] += inputFloat[column] * weightFloat
                let weightHalf = Float(Float16(weightFloat))
                mpsReference64[row] += Double(Float(inputHalf[column])) * Double(weightHalf)
            }
        }

        let packedBuffer = packed.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let scaleBits = scales.map(\.bitPattern)
        let zeroBits = zeros.map(\.bitPattern)
        let scaleBuffer = scaleBits.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let zeroBuffer = zeroBits.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }

        let directInput = inputFloat.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let directOutput = try makeBuffer(bytes: n * MemoryLayout<Float>.stride, on: context.device)
        let directPipeline = try context.pipeline(named: "w4_matvec_f32")
        var columns = UInt32(k), rows = UInt32(n), stride = UInt32(packedStride)
        let directCommand = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(directCommand.makeComputeCommandEncoder())
        encoder.setComputePipelineState(directPipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(zeroBuffer, offset: 0, index: 2)
        encoder.setBuffer(directInput, offset: 0, index: 3)
        encoder.setBuffer(directOutput, offset: 0, index: 4)
        encoder.setBytes(&columns, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBytes(&stride, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        directCommand.commit()
        directCommand.waitUntilCompleted()
        XCTAssertNil(directCommand.error)
        let directPointer = directOutput.contents().bindMemory(to: Float.self, capacity: n)
        let directGPU = Array(UnsafeBufferPointer(start: directPointer, count: n)).map(Double.init)

        let mpsInputBits = inputHalf.map(\.bitPattern)
        let mpsInput = mpsInputBits.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let mpsOutput = try makeBuffer(bytes: n * MemoryLayout<Float16>.stride, on: context.device)
        let weight = QuantizedLinearWeightBuffers(
            storage: .w4, packed: packedBuffer, packedOffset: 0,
            scale: scaleBuffer, scaleOffset: 0, zero: zeroBuffer, zeroOffset: 0,
            rows: n, columns: k, packedRowStride: packedStride)
        try await LinearExecutor(context: context).execute(
            input: mpsInput, weight: weight, output: mpsOutput, inputRows: 1)
        let mpsGPU = (0..<n).map {
            Double(loadHalf(mpsOutput, byteOffset: $0 * MemoryLayout<Float16>.stride))
        }

        let directMetrics = metrics(actual: directGPU, reference: directReference64)
        let cpu32Metrics = metrics(actual: directReference32.map(Double.init), reference: directReference64)
        let mpsMetrics = metrics(actual: mpsGPU, reference: mpsReference64)
        print("E007 K=\(k) direct(maxAbs=\(directMetrics.maxAbs),rmse=\(directMetrics.rmse),cos=\(directMetrics.cosine)) "
            + "cpu32(maxAbs=\(cpu32Metrics.maxAbs),rmse=\(cpu32Metrics.rmse),cos=\(cpu32Metrics.cosine)) "
            + "mps(maxAbs=\(mpsMetrics.maxAbs),rmse=\(mpsMetrics.rmse),cos=\(mpsMetrics.cosine))")
        XCTAssertLessThan(directMetrics.maxAbs, 0.001)
        XCTAssertGreaterThan(directMetrics.cosine, 0.999_999)
        XCTAssertLessThan(mpsMetrics.rmse, 0.02)
        XCTAssertGreaterThan(mpsMetrics.cosine, 0.999)
    }

    private func metrics(actual: [Double], reference: [Double])
        -> (maxAbs: Double, rmse: Double, cosine: Double) {
        var maxAbs = 0.0, squaredError = 0.0
        var dot = 0.0, actualNorm = 0.0, referenceNorm = 0.0
        for index in actual.indices {
            let error = actual[index] - reference[index]
            maxAbs = max(maxAbs, abs(error))
            squaredError += error * error
            dot += actual[index] * reference[index]
            actualNorm += actual[index] * actual[index]
            referenceNorm += reference[index] * reference[index]
        }
        return (maxAbs, sqrt(squaredError / Double(actual.count)),
                dot / sqrt(actualNorm * referenceNorm))
    }
}
