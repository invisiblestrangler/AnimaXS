import Foundation
import Metal
import MetalPerformanceShaders

/// Ring-relative buffers and metadata for a packed W4/W8 `[N,K]` matrix.
struct QuantizedLinearWeightBuffers {
    let storage: StorageDtype
    let packed: MTLBuffer
    let packedOffset: Int
    let scale: MTLBuffer
    let scaleOffset: Int
    let zero: MTLBuffer
    let zeroOffset: Int
    let rows: Int
    let columns: Int
    let packedRowStride: Int
}

/// Common bounded-memory quantized linear path:
/// `[M,K] fp16 × [N,K]ᵀ quantized → [M,N] fp16`.
///
/// The weight matrix is dequantized once into one reusable fp16 scratch buffer.
/// Input rows are submitted to MPS in tiles so the same executor works for token
/// matrices without allocating per-tile copies.
final class LinearExecutor {
    static let defaultTileRows = 128

    private let context: MetalContext
    private let buffers: BufferPool
    let tileRows: Int

    init(context: MetalContext, tileRows: Int = defaultTileRows) {
        precondition(tileRows > 0)
        self.context = context
        self.buffers = BufferPool(device: context.device)
        self.tileRows = tileRows
    }

    /// Encode dequantization and all MPS tiles into an existing command buffer.
    /// The caller owns command-buffer commit/completion.
    func encode(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        inputOffset: Int = 0,
        weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer,
        outputOffset: Int = 0,
        inputRows: Int
    ) throws {
        try validate(input: input, inputOffset: inputOffset, weight: weight,
                     output: output, outputOffset: outputOffset, inputRows: inputRows)

        let n = weight.rows
        let k = weight.columns
        let rightRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
        let scratchBytes = try checkedProduct(n, rightRowBytes)
        let scratch = buffers.buffer(key: "linear.weight.fp16", bytes: scratchBytes)
        let kernelName: String
        switch weight.storage {
        case .w4: kernelName = "dequant_w4_to_half"
        case .w8: kernelName = "dequant_w8_to_half"
        default:
            throw AnimapkError.validation("LinearExecutor requires W4 or W8 weights")
        }
        let pipeline = try context.pipeline(named: kernelName)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create linear dequant encoder")
        }
        var columns = UInt32(k)
        var rowStride = UInt32(weight.packedRowStride)
        var rows = UInt32(n)
        var outputStride = UInt32(rightRowBytes / MemoryLayout<Float16>.stride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 0)
        encoder.setBuffer(weight.scale, offset: weight.scaleOffset, index: 1)
        encoder.setBuffer(weight.zero, offset: weight.zeroOffset, index: 2)
        encoder.setBuffer(scratch, offset: 0, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&rowStride, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBytes(&outputStride, length: 4, index: 7)
        let width = min(16, pipeline.threadExecutionWidth)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(
            MTLSize(width: k, height: n, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()

        let scalarBytes = MemoryLayout<Float16>.stride
        let rightDescriptor = MPSMatrixDescriptor(
            rows: n, columns: k, rowBytes: rightRowBytes, dataType: .float16)
        let right = MPSMatrix(buffer: scratch, descriptor: rightDescriptor)

        var rowStart = 0
        while rowStart < inputRows {
            let rowsThisTile = min(tileRows, inputRows - rowStart)
            let leftRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
            let resultRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: n, dataType: .float16)
            let leftScratch = buffers.buffer(
                key: "linear.left.fp16", bytes: try checkedProduct(rowsThisTile, leftRowBytes))
            let resultScratch = buffers.buffer(
                key: "linear.result.fp16", bytes: try checkedProduct(rowsThisTile, resultRowBytes))
            try encodeCopy(commandBuffer: commandBuffer,
                           source: input, sourceOffset: inputOffset + rowStart * k * scalarBytes,
                           destination: leftScratch, destinationOffset: 0,
                           columns: k, rows: rowsThisTile,
                           sourceStride: k, destinationStride: leftRowBytes / scalarBytes)
            let leftDescriptor = MPSMatrixDescriptor(
                rows: rowsThisTile, columns: k, rowBytes: leftRowBytes, dataType: .float16)
            let outputDescriptor = MPSMatrixDescriptor(
                rows: rowsThisTile, columns: n, rowBytes: resultRowBytes, dataType: .float16)
            let left = MPSMatrix(
                buffer: leftScratch, descriptor: leftDescriptor)
            let result = MPSMatrix(
                buffer: resultScratch, descriptor: outputDescriptor)
            let multiplication = MPSMatrixMultiplication(
                device: context.device,
                transposeLeft: false,
                transposeRight: true,
                resultRows: rowsThisTile,
                resultColumns: n,
                interiorColumns: k,
                alpha: 1,
                beta: 0)
            multiplication.encode(
                commandBuffer: commandBuffer,
                leftMatrix: left,
                rightMatrix: right,
                resultMatrix: result)
            try encodeCopy(commandBuffer: commandBuffer,
                           source: resultScratch, sourceOffset: 0,
                           destination: output,
                           destinationOffset: outputOffset + rowStart * n * scalarBytes,
                           columns: n, rows: rowsThisTile,
                           sourceStride: resultRowBytes / scalarBytes, destinationStride: n)
            rowStart += rowsThisTile
        }
    }

    private func encodeCopy(
        commandBuffer: MTLCommandBuffer, source: MTLBuffer, sourceOffset: Int,
        destination: MTLBuffer, destinationOffset: Int, columns: Int, rows: Int,
        sourceStride: Int, destinationStride: Int
    ) throws {
        let pipeline = try context.pipeline(named: "copy_half_rows")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create linear row-copy encoder")
        }
        var columnCount = UInt32(columns), rowCount = UInt32(rows)
        var sourceRowStride = UInt32(sourceStride), destinationRowStride = UInt32(destinationStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination, offset: destinationOffset, index: 1)
        encoder.setBytes(&columnCount, length: 4, index: 2)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&sourceRowStride, length: 4, index: 4)
        encoder.setBytes(&destinationRowStride, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: columns, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(16, columns), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Convenience async submission. Completion resumes off the command-buffer callback.
    func execute(
        input: MTLBuffer,
        inputOffset: Int = 0,
        weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer,
        outputOffset: Int = 0,
        inputRows: Int
    ) async throws {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create linear command buffer")
        }
        try encode(commandBuffer: commandBuffer, input: input, inputOffset: inputOffset,
                   weight: weight, output: output, outputOffset: outputOffset,
                   inputRows: inputRows)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { completed in
                if let error = completed.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            commandBuffer.commit()
        }
    }

    private func validate(
        input: MTLBuffer, inputOffset: Int, weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer, outputOffset: Int, inputRows: Int
    ) throws {
        guard inputRows > 0, weight.rows > 0, weight.columns > 0,
              inputOffset >= 0, outputOffset >= 0,
              weight.packedOffset >= 0, weight.scaleOffset >= 0, weight.zeroOffset >= 0 else {
            throw AnimapkError.validation("invalid LinearExecutor shape or offset")
        }
        let minimumStride = weight.storage == .w4
            ? (weight.columns + 1) / 2 : weight.columns
        guard weight.packedRowStride >= minimumStride else {
            throw AnimapkError.validation("packed linear row stride is too small")
        }
        let inputBytes = try checkedProduct(inputRows, weight.columns, MemoryLayout<Float16>.stride)
        let outputBytes = try checkedProduct(inputRows, weight.rows, MemoryLayout<Float16>.stride)
        let packedBytes = try checkedProduct(weight.rows, weight.packedRowStride)
        let groups = (weight.columns + 63) / 64
        let parameterBytes = try checkedProduct(weight.rows, groups, MemoryLayout<Float16>.stride)
        guard try checkedEnd(offset: inputOffset, bytes: inputBytes) <= input.length,
              try checkedEnd(offset: outputOffset, bytes: outputBytes) <= output.length,
              try checkedEnd(offset: weight.packedOffset, bytes: packedBytes) <= weight.packed.length,
              try checkedEnd(offset: weight.scaleOffset, bytes: parameterBytes) <= weight.scale.length,
              try checkedEnd(offset: weight.zeroOffset, bytes: parameterBytes) <= weight.zero.length else {
            throw AnimapkError.validation("LinearExecutor buffer range is out of bounds")
        }
    }

    private func checkedEnd(offset: Int, bytes: Int) throws -> Int {
        let (end, overflow) = offset.addingReportingOverflow(bytes)
        guard !overflow else {
            throw AnimapkError.validation("LinearExecutor buffer range overflow")
        }
        return end
    }

    private func checkedProduct(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw AnimapkError.validation("LinearExecutor byte size overflow")
            }
            result = next
        }
        return result
    }
}

/// Shared validation and construction for all rank-2 DiT/adapter matrices.
/// W4 and W8 use the same row-major group-64 span layout; only the packed row
/// stride and direct matvec kernel differ.
enum DiTQuantizedWeightFactory {
    static let groupSize = 64

    static func makeMatrix(
        _ item: AnimapkTensorSpans,
        ring: MTLBuffer,
        rows: Int,
        columns: Int,
        label: String
    ) throws -> QuantizedLinearWeightBuffers {
        let storage = item.tensor.storage
        guard storage == .w4 || storage == .w8 else {
            throw AnimapkError.validation("\(label) must be W4 or W8")
        }
        guard item.tensor.shape == [rows, columns] else {
            throw AnimapkError.validation("\(label) shape mismatch")
        }
        guard let scale = item.scale, let zero = item.zero else {
            throw AnimapkError.validation("\(label) missing scale/zero")
        }
        let rowStride = storage == .w4 ? (columns + 1) / 2 : columns
        let groupsPerRow = (columns + groupSize - 1) / groupSize
        let expectedData = rows * rowStride
        let expectedParameters = rows * groupsPerRow * MemoryLayout<Float16>.stride
        guard item.data.length == UInt64(expectedData),
              scale.length == UInt64(expectedParameters),
              zero.length == UInt64(expectedParameters),
              item.data.offset <= UInt64(Int.max),
              scale.offset <= UInt64(Int.max),
              zero.offset <= UInt64(Int.max) else {
            throw AnimapkError.validation("\(label) quantized layout mismatch")
        }
        return QuantizedLinearWeightBuffers(
            storage: storage,
            packed: ring,
            packedOffset: Int(item.data.offset),
            scale: ring,
            scaleOffset: Int(scale.offset),
            zero: ring,
            zeroOffset: Int(zero.offset),
            rows: rows,
            columns: columns,
            packedRowStride: rowStride)
    }

    static func matvecKernel(for storage: StorageDtype) throws -> String {
        switch storage {
        case .w4: return "w4_matvec_f32"
        case .w8: return "w8_matvec_f32"
        default:
            throw AnimapkError.validation("DiT direct matvec requires W4 or W8")
        }
    }
}
