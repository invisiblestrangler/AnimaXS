import Foundation
import Metal
import MetalPerformanceShaders

/// Bounded-memory fp16 VAE convolution path (J002 §21).
///
/// Supports exactly what the Wan T=1 decoder needs:
///   - 1x1 stride-1 (skip im2col, use activation rows directly)
///   - 3x3 stride-1 pad-1 (tiled im2col + MPS GEMM, ~128 output pixels per tile)
///   - fused nearest-exact 2x upsample + 3x3 (im2col reads the low-res source
///     directly; no enlarged temporary, per §25)
///
/// Activations are position-major `[height*width, channels]` fp16 (channels
/// contiguous). Weights arrive in PyTorch order and are folded into a
/// row-padded fp16 scratch matrix by `encodeFoldWeight` before the GEMM:
/// rank-5 causal weights use their FINAL temporal slice (D052); rank-4 weights
/// are used as-is. Column order is `(c*KH+ky)*KW+kx` so the im2col row and the
/// weight row line up element-for-element — verified by the tiny reference tests.
final class FP16ConvolutionExecutor {
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

    /// Fold one conv weight (rank-4 or rank-5 fp16, PyTorch layout) into a
    /// row-padded scratch matrix `[Cout, Cin*KH*KW]` with MPS rowBytes.
    /// Returns the scratch buffer. `outputRowStrideElements` is the row stride
    /// in half elements; callers use `MPSMatrixDescriptor.rowBytes` / 2.
    @discardableResult
    func encodeFoldWeight(
        commandBuffer: MTLCommandBuffer,
        source: MTLBuffer, sourceOffset: Int,
        shape: [Int],
        outputRowStrideElements: Int,
        scratchKey: String
    ) throws -> MTLBuffer {
        let rank = shape.count
        guard rank == 4 || rank == 5 else {
            throw AnimapkError.validation("VAE conv weight must be rank 4 or 5")
        }
        let outputChannels = shape[0]
        let inputChannels = shape[1]
        let kernelHeight = shape[rank - 2]
        let kernelWidth = shape[rank - 1]
        guard kernelHeight == kernelWidth, kernelHeight == 1 || kernelHeight == 3 else {
            throw AnimapkError.validation("VAE conv weight kernel must be 1x1 or 3x3")
        }
        let kernelElements = kernelHeight * kernelWidth
        let rowBytes = try checkedProduct(outputRowStrideElements, 2)
        let scratch = buffers.buffer(key: scratchKey, bytes: try checkedProduct(outputChannels, rowBytes))
        let pipeline = try context.pipeline(named: "vae_fold_weight_half")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE weight fold encoder")
        }
        var channelsIn = UInt32(inputChannels)
        var channelsOut = UInt32(outputChannels)
        var kh = UInt32(kernelHeight), kw = UInt32(kernelWidth)
        var kt = UInt32(rank == 5 ? shape[2] : 1)  // causal KT from the tensor shape
        var outStride = UInt32(outputRowStrideElements)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(scratch, offset: 0, index: 1)
        encoder.setBytes(&channelsIn, length: 4, index: 2)
        encoder.setBytes(&channelsOut, length: 4, index: 3)
        encoder.setBytes(&kh, length: 4, index: 4)
        encoder.setBytes(&kw, length: 4, index: 5)
        encoder.setBytes(&kt, length: 4, index: 6)
        encoder.setBytes(&outStride, length: 4, index: 7)
        let total = outputChannels * inputChannels * kernelElements
        encoder.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        return scratch
    }

    /// 1x1 convolution: `[M,Cin] x [Cout,Cin]^T -> [M,Cout]` via MPS GEMM.
    /// `weight` must already be a row-padded scratch (see `encodeFoldWeight`).
    func encode1x1(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, inputOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int,
        output: MTLBuffer, outputOffset: Int = 0,
        rows: Int, inputChannels: Int, outputChannels: Int
    ) throws {
        try validate1x1(input: input, inputOffset: inputOffset, weight: weight,
                        weightOffset: weightOffset, output: output,
                        outputOffset: outputOffset, rows: rows,
                        inputChannels: inputChannels, outputChannels: outputChannels)
        let half = MemoryLayout<Float16>.stride
        let inputRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: inputChannels, dataType: .float16)
        let weightRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: inputChannels, dataType: .float16)
        let outputRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: outputChannels, dataType: .float16)

        var rowStart = 0
        while rowStart < rows {
            let rowsThisTile = min(tileRows, rows - rowStart)
            let resultScratch = buffers.buffer(
                key: "vae.conv1x1.result", bytes: try checkedProduct(rowsThisTile, outputRowBytes))
            let inputMatrix = MPSMatrix(
                buffer: input, offset: inputOffset + rowStart * inputRowBytes,
                descriptor: MPSMatrixDescriptor(
                    rows: rowsThisTile, columns: inputChannels,
                    rowBytes: inputRowBytes, dataType: .float16))
            let weightMatrix = MPSMatrix(
                buffer: weight, offset: weightOffset,
                descriptor: MPSMatrixDescriptor(
                    rows: outputChannels, columns: inputChannels,
                    rowBytes: weightRowBytes, dataType: .float16))
            let resultMatrix = MPSMatrix(
                buffer: resultScratch,
                descriptor: MPSMatrixDescriptor(
                    rows: rowsThisTile, columns: outputChannels,
                    rowBytes: outputRowBytes, dataType: .float16))
            MPSMatrixMultiplication(
                device: context.device, transposeLeft: false, transposeRight: true,
                resultRows: rowsThisTile, resultColumns: outputChannels,
                interiorColumns: inputChannels, alpha: 1, beta: 0).encode(
                    commandBuffer: commandBuffer, leftMatrix: inputMatrix,
                    rightMatrix: weightMatrix, resultMatrix: resultMatrix)
            try encodeCopyRows(
                commandBuffer: commandBuffer,
                source: resultScratch, sourceOffset: 0, sourceRowStride: outputRowBytes / half,
                destination: output, destinationOffset: outputOffset + rowStart * outputRowBytes,
                destinationRowStride: outputChannels,
                columns: outputChannels, rows: rowsThisTile)
            rowStart += rowsThisTile
        }
    }

    /// 3x3 convolution via tiled im2col + MPS GEMM. `weight` must already be a
    /// row-padded scratch (see `encodeFoldWeight`). When `upsample2x` is true,
    /// the conv input is the nearest-exact 2x upsample of `input` (which is
    /// `[srcH*srcW, Cin]`) and `outputPositions = (2*srcH)*(2*srcW)`; im2col
    /// reads the source directly with zero padding in the upscaled coordinate
    /// space.
    func encode3x3(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, inputOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int,
        bias: MTLBuffer?, biasOffset: Int,
        output: MTLBuffer, outputOffset: Int = 0,
        inputHeight: Int, inputWidth: Int,
        outputChannels: Int, inputChannels: Int,
        upsample2x: Bool = false
    ) throws {
        let outputHeight = upsample2x ? inputHeight * 2 : inputHeight
        let outputWidth = upsample2x ? inputWidth * 2 : inputWidth
        let outputPositions = outputHeight * outputWidth
        try validate3x3(
            input: input, inputOffset: inputOffset, weight: weight,
            weightOffset: weightOffset, bias: bias, output: output,
            outputOffset: outputOffset, inputHeight: inputHeight,
            inputWidth: inputWidth, outputChannels: outputChannels,
            inputChannels: inputChannels, upsample2x: upsample2x)

        let half = MemoryLayout<Float16>.stride
        let im2colColumns = inputChannels * 9
        let im2colRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: im2colColumns, dataType: .float16)
        let weightRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: im2colColumns, dataType: .float16)
        let outputRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: outputChannels, dataType: .float16)

        var rowStart = 0
        while rowStart < outputPositions {
            let rowsThisTile = min(tileRows, outputPositions - rowStart)
            let im2colScratch = buffers.buffer(
                key: "vae.conv3x3.im2col", bytes: try checkedProduct(rowsThisTile, im2colRowBytes))
            try encodeIm2col(
                commandBuffer: commandBuffer,
                input: input, inputOffset: inputOffset,
                output: im2colScratch,
                inputHeight: inputHeight, inputWidth: inputWidth,
                channels: inputChannels, tileRows: rowsThisTile, tileBase: rowStart,
                upsample2x: upsample2x)

            let resultScratch = buffers.buffer(
                key: "vae.conv3x3.result", bytes: try checkedProduct(rowsThisTile, outputRowBytes))
            let im2colMatrix = MPSMatrix(
                buffer: im2colScratch,
                descriptor: MPSMatrixDescriptor(
                    rows: rowsThisTile, columns: im2colColumns,
                    rowBytes: im2colRowBytes, dataType: .float16))
            let weightMatrix = MPSMatrix(
                buffer: weight, offset: weightOffset,
                descriptor: MPSMatrixDescriptor(
                    rows: outputChannels, columns: im2colColumns,
                    rowBytes: weightRowBytes, dataType: .float16))
            let resultMatrix = MPSMatrix(
                buffer: resultScratch,
                descriptor: MPSMatrixDescriptor(
                    rows: rowsThisTile, columns: outputChannels,
                    rowBytes: outputRowBytes, dataType: .float16))
            MPSMatrixMultiplication(
                device: context.device, transposeLeft: false, transposeRight: true,
                resultRows: rowsThisTile, resultColumns: outputChannels,
                interiorColumns: im2colColumns, alpha: 1, beta: 0).encode(
                    commandBuffer: commandBuffer, leftMatrix: im2colMatrix,
                    rightMatrix: weightMatrix, resultMatrix: resultMatrix)

            if let bias {
                try encodeAddBias(commandBuffer: commandBuffer, values: resultScratch,
                                  bias: bias, biasOffset: biasOffset,
                                  columns: outputChannels,
                                  count: rowsThisTile * outputChannels)
            }

            try encodeCopyRows(
                commandBuffer: commandBuffer,
                source: resultScratch, sourceOffset: 0, sourceRowStride: outputRowBytes / half,
                destination: output, destinationOffset: outputOffset + rowStart * outputRowBytes,
                destinationRowStride: outputChannels,
                columns: outputChannels, rows: rowsThisTile)
            rowStart += rowsThisTile
        }
    }

    /// Convenience async submission for a 1x1 convolution.
    func execute1x1(
        input: MTLBuffer, inputOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int,
        output: MTLBuffer, outputOffset: Int = 0,
        rows: Int, inputChannels: Int, outputChannels: Int
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE 1x1 command buffer")
        }
        try encode1x1(commandBuffer: command, input: input, inputOffset: inputOffset,
                      weight: weight, weightOffset: weightOffset,
                      output: output, outputOffset: outputOffset,
                      rows: rows, inputChannels: inputChannels,
                      outputChannels: outputChannels)
        try await commit(command)
    }

    /// Convenience async submission for a 3x3 convolution.
    func execute3x3(
        input: MTLBuffer, inputOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int,
        bias: MTLBuffer?, biasOffset: Int,
        output: MTLBuffer, outputOffset: Int = 0,
        inputHeight: Int, inputWidth: Int,
        outputChannels: Int, inputChannels: Int,
        upsample2x: Bool = false
    ) async throws {
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create VAE 3x3 command buffer")
        }
        try encode3x3(commandBuffer: command, input: input, inputOffset: inputOffset,
                      weight: weight, weightOffset: weightOffset,
                      bias: bias, biasOffset: biasOffset,
                      output: output, outputOffset: outputOffset,
                      inputHeight: inputHeight, inputWidth: inputWidth,
                      outputChannels: outputChannels, inputChannels: inputChannels,
                      upsample2x: upsample2x)
        try await commit(command)
    }

    // MARK: - Encoders

    private func encodeIm2col(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer,
        inputHeight: Int, inputWidth: Int,
        channels: Int, tileRows: Int, tileBase: Int,
        upsample2x: Bool
    ) throws {
        let kernelName = upsample2x ? "vae_im2col_upsample3x3_half" : "vae_im2col3x3_half"
        let pipeline = try context.pipeline(named: kernelName)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE im2col encoder")
        }
        // MPS GEMM requires padded rowBytes; the im2col kernel must write rows
        // with the same stride or every row after the first misaligns.
        let outStrideElements = MPSMatrixDescriptor.rowBytes(
            fromColumns: channels * 9, dataType: .float16) / MemoryLayout<Float16>.stride
        var height = UInt32(inputHeight), width = UInt32(inputWidth)
        var channelsU = UInt32(channels), rows = UInt32(tileRows), base = UInt32(tileBase)
        var outStride = UInt32(outStrideElements)
        if upsample2x {
            var outH = UInt32(inputHeight * 2), outW = UInt32(inputWidth * 2)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: inputOffset, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&height, length: 4, index: 2)
            encoder.setBytes(&width, length: 4, index: 3)
            encoder.setBytes(&channelsU, length: 4, index: 4)
            encoder.setBytes(&outH, length: 4, index: 5)
            encoder.setBytes(&outW, length: 4, index: 6)
            encoder.setBytes(&rows, length: 4, index: 7)
            encoder.setBytes(&base, length: 4, index: 8)
            encoder.setBytes(&outStride, length: 4, index: 9)
        } else {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: inputOffset, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&height, length: 4, index: 2)
            encoder.setBytes(&width, length: 4, index: 3)
            encoder.setBytes(&channelsU, length: 4, index: 4)
            encoder.setBytes(&rows, length: 4, index: 5)
            encoder.setBytes(&base, length: 4, index: 6)
            encoder.setBytes(&outStride, length: 4, index: 7)
        }
        encoder.dispatchThreads(MTLSize(width: tileRows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeCopyRows(
        commandBuffer: MTLCommandBuffer,
        source: MTLBuffer, sourceOffset: Int, sourceRowStride: Int,
        destination: MTLBuffer, destinationOffset: Int, destinationRowStride: Int,
        columns: Int, rows: Int
    ) throws {
        let pipeline = try context.pipeline(named: "copy_half_rows")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE row-copy encoder")
        }
        var columnCount = UInt32(columns), rowCount = UInt32(rows)
        var sourceStride = UInt32(sourceRowStride), destinationStride = UInt32(destinationRowStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination, offset: destinationOffset, index: 1)
        encoder.setBytes(&columnCount, length: 4, index: 2)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&sourceStride, length: 4, index: 4)
        encoder.setBytes(&destinationStride, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: columns, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(16, columns), height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeAddBias(
        commandBuffer: MTLCommandBuffer, values: MTLBuffer,
        bias: MTLBuffer, biasOffset: Int, columns: Int, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_bias_half")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create VAE conv bias encoder")
        }
        var columnsU = UInt32(columns), countU = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBytes(&columnsU, length: 4, index: 2)
        encoder.setBytes(&countU, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1))
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

    // MARK: - Validation

    private func validate1x1(
        input: MTLBuffer, inputOffset: Int, weight: MTLBuffer, weightOffset: Int,
        output: MTLBuffer, outputOffset: Int, rows: Int,
        inputChannels: Int, outputChannels: Int
    ) throws {
        guard rows > 0, inputChannels > 0, outputChannels > 0,
              inputOffset >= 0, weightOffset >= 0, outputOffset >= 0 else {
            throw AnimapkError.validation("invalid VAE 1x1 shape or offset")
        }
        let inputBytes = try checkedProduct(rows, inputChannels, 2)
        let weightBytes = try checkedProduct(outputChannels, inputChannels, 2)
        let outputBytes = try checkedProduct(rows, outputChannels, 2)
        guard try checkedEnd(inputOffset, inputBytes) <= input.length,
              try checkedEnd(weightOffset, weightBytes) <= weight.length,
              try checkedEnd(outputOffset, outputBytes) <= output.length else {
            throw AnimapkError.validation("VAE 1x1 buffer range is out of bounds")
        }
    }

    private func validate3x3(
        input: MTLBuffer, inputOffset: Int, weight: MTLBuffer, weightOffset: Int,
        bias: MTLBuffer?, output: MTLBuffer, outputOffset: Int,
        inputHeight: Int, inputWidth: Int, outputChannels: Int,
        inputChannels: Int, upsample2x: Bool
    ) throws {
        guard inputHeight > 0, inputWidth > 0, outputChannels > 0, inputChannels > 0,
              inputOffset >= 0, weightOffset >= 0, outputOffset >= 0 else {
            throw AnimapkError.validation("invalid VAE 3x3 shape or offset")
        }
        let inputBytes = try checkedProduct(inputHeight, inputWidth, inputChannels, 2)
        let weightBytes = try checkedProduct(outputChannels, inputChannels, 9, 2)
        let outputHeight = upsample2x ? inputHeight * 2 : inputHeight
        let outputWidth = upsample2x ? inputWidth * 2 : inputWidth
        let outputBytes = try checkedProduct(outputHeight, outputWidth, outputChannels, 2)
        guard try checkedEnd(inputOffset, inputBytes) <= input.length,
              try checkedEnd(weightOffset, weightBytes) <= weight.length,
              try checkedEnd(outputOffset, outputBytes) <= output.length else {
            throw AnimapkError.validation("VAE 3x3 buffer range is out of bounds")
        }
        if let bias {
            let biasBytes = try checkedProduct(outputChannels, 2)
            guard try checkedEnd(0, biasBytes) <= bias.length else {
                throw AnimapkError.validation("VAE 3x3 bias range is out of bounds")
            }
        }
    }

    private func checkedProduct(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else { throw AnimapkError.validation("VAE byte size overflow") }
            result = next
        }
        return result
    }

    private func checkedEnd(_ offset: Int, _ bytes: Int) throws -> Int {
        let (end, overflow) = offset.addingReportingOverflow(bytes)
        guard !overflow else { throw AnimapkError.validation("VAE buffer range overflow") }
        return end
    }
}
