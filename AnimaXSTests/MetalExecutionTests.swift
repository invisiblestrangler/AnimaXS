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
}
