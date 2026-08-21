import XCTest
import Metal
import MetalPerformanceShaders
@testable import AnimaXS

/// Small reference/oracle tests that do not require production model packs.
final class ReferenceTestsPlaceholder: XCTestCase {
    func testPlaceholder() {
        XCTAssertTrue(ModelConstants.sigma8Step.count == 9)
    }

    /// Exact mathematical seam for the runtime overlay:
    /// y = base + scale * ((x A^T) B^T).
    /// This exercises the same two MPS GEMMs used after either a Metal/W4/W8
    /// base projection or an ANE base projection, without needing a huge pack.
    func testLoRALowRankResidualMatchesCPUReference() throws {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let m = 3, k = 8, n = 8, rank = 4
        let scale: Float = 0.75
        let scalarBytes = MemoryLayout<Float16>.stride
        XCTAssertEqual(
            MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16),
            k * scalarBytes)
        XCTAssertEqual(
            MPSMatrixDescriptor.rowBytes(fromColumns: n, dataType: .float16),
            n * scalarBytes)

        let xValues = (0..<(m * k)).map { Float(($0 % 9) - 4) / 8 }
        let baseValues = (0..<(m * n)).map { Float(($0 % 7) - 3) / 10 }
        let downValues = (0..<(rank * k)).map { Float(($0 % 5) - 2) / 8 }
        let upValues = (0..<(n * rank)).map { Float(($0 % 7) - 3) / 12 }

        let input = try makeTightHalfBuffer(xValues, device: context.device)
        let output = try makeTightHalfBuffer(baseValues, device: context.device)
        let downRowBytes = MPSMatrixDescriptor.rowBytes(
            fromColumns: k, dataType: .float16)
        let upRowBytes = MPSMatrixDescriptor.rowBytes(
            fromColumns: rank, dataType: .float16)
        let down = try makePaddedHalfMatrix(
            values: downValues, rows: rank, columns: k,
            rowBytes: downRowBytes, device: context.device)
        let up = try makePaddedHalfMatrix(
            values: upValues, rows: n, columns: rank,
            rowBytes: upRowBytes, device: context.device)

        let projection = DiTLoRAProjectionBuffers(
            key: DiTLoRAKey(block: 0, target: .selfQ),
            down: down, downRowBytes: downRowBytes,
            up: up, upRowBytes: upRowBytes,
            rank: rank, inputFeatures: k, outputFeatures: n,
            effectiveScale: scale)
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        try DiTLoRAExecutor.encodeLowRankResidual(
            context: context, scratch: BufferPool(device: context.device),
            commandBuffer: command, input: input, output: output,
            inputRows: m, projection: projection)
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)

        let x = xValues.map { Float(Float16($0)) }
        let base = baseValues.map { Float(Float16($0)) }
        let a = downValues.map { Float(Float16($0)) }
        let b = upValues.map { Float(Float16($0)) }
        for row in 0..<m {
            var low = [Float](repeating: 0, count: rank)
            for r in 0..<rank {
                for column in 0..<k {
                    low[r] += x[row * k + column] * a[r * k + column]
                }
            }
            for column in 0..<n {
                var delta: Float = 0
                for r in 0..<rank {
                    delta += low[r] * b[column * rank + r]
                }
                let expected = base[row * n + column] + scale * delta
                let actual = loadHalf(output, index: row * n + column)
                XCTAssertEqual(
                    actual, expected, accuracy: 0.015,
                    "LoRA residual mismatch row=\(row) column=\(column)")
            }
        }
    }

    private func makeTightHalfBuffer(
        _ values: [Float], device: MTLDevice
    ) throws -> MTLBuffer {
        let bits = values.map { Float16($0).bitPattern }
        return try bits.withUnsafeBytes { bytes in
            try XCTUnwrap(device.makeBuffer(
                bytes: bytes.baseAddress!, length: bytes.count,
                options: .storageModeShared))
        }
    }

    private func makePaddedHalfMatrix(
        values: [Float], rows: Int, columns: Int,
        rowBytes: Int, device: MTLDevice
    ) throws -> MTLBuffer {
        let buffer = try XCTUnwrap(device.makeBuffer(
            length: rows * rowBytes, options: .storageModeShared))
        memset(buffer.contents(), 0, buffer.length)
        for row in 0..<rows {
            let destination = buffer.contents().advanced(by: row * rowBytes)
                .bindMemory(to: UInt16.self, capacity: columns)
            for column in 0..<columns {
                destination[column] = Float16(values[row * columns + column]).bitPattern
            }
        }
        return buffer
    }

    private func loadHalf(_ buffer: MTLBuffer, index: Int) -> Float {
        let bits = buffer.contents().bindMemory(
            to: UInt16.self, capacity: buffer.length / 2)[index]
        return Float(Float16(bitPattern: bits))
    }
}
