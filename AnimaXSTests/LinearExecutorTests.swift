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
        let m = 133, n = 8, k = 16, packedStride = 20
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
        let m = 3, n = 8, k = 16, packedStride = 12
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
}
