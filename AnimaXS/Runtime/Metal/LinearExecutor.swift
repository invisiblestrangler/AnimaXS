import Foundation
import Metal
import MetalPerformanceShaders

/// P8: matrix-family classification for the DiT linear dispatch (runbook
/// §13). Explicitly tagged at the DiT call sites — NEVER inferred from a
/// filename. Enables hybrid dispatch: MLP matrices (the largest
/// decompressed-weight scratch traffic) run the direct packed QGEMM while
/// attention projections stay on MPS until A12 data proves otherwise.
enum DiTLinearFamily {
    case attentionProjection
    case mlpUp
    case mlpDown
    case other
}

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
///
/// `directMPSIO` (runtime experiment): when a tile's tight row stride exactly
/// satisfies MPS's recommended row stride, the MPSMatrix wraps the existing
/// input/output buffers directly and the per-tile copy kernels are skipped.
/// Input and output eligibility are independent; the run metrics counters
/// report how many tiles actually hit direct wrapping on the target device.
final class LinearExecutor {
    static let defaultTileRows = 128
    /// P8: QGEMM tile-profile thresholds. The K=16-wide profile (128
    /// threads) is only chosen when N is large enough to make it worthwhile;
    /// small-N linears use the K=8-wide profile.
    static let qgemmWideProfileMinN = 16

    private let context: MetalContext
    private let buffers: BufferPool
    let tileRows: Int
    private let directMPSIO: Bool
    /// P8: direct packed QGEMM backend selector (defaults to the legacy
    /// dequantized-MPS behavior).
    let linearBackend: DiTLinearBackend
    /// P8: matrix family for hybrid dispatch (defaults to `.other` so
    /// non-DiT callers are unaffected).
    let family: DiTLinearFamily
    /// Run telemetry collector (nil in tests / diagnostic-only construction).
    /// Receives the cheap tile counters (simple integer increments).
    var metrics: MetricsCollector?

    init(context: MetalContext, tileRows: Int = defaultTileRows,
         directMPSIO: Bool = false,
         linearBackend: DiTLinearBackend = .dequantizedMPS,
         family: DiTLinearFamily = .other) {
        precondition(tileRows > 0)
        self.context = context
        self.buffers = BufferPool(device: context.device)
        self.tileRows = tileRows
        self.directMPSIO = directMPSIO
        self.linearBackend = linearBackend
        self.family = family
    }

    /// Encodes dequantization and all MPS tiles into an existing command buffer.
    /// The caller owns command-buffer commit/completion.
    func encode(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        inputOffset: Int = 0,
        weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer,
        outputOffset: Int = 0,
        inputRows: Int,
        family: DiTLinearFamily = .other
    ) throws {
        try validate(input: input, inputOffset: inputOffset, weight: weight,
                     output: output, outputOffset: outputOffset, inputRows: inputRows)

        let direct = (weight.storage == .w4 || weight.storage == .w8)
            && inputRows > 1
        if direct {
            switch linearBackend {
            case .dequantizedMPS:
                break
            case .directQuantized:
                try encodeQGEMM(commandBuffer: commandBuffer, input: input,
                                inputOffset: inputOffset, weight: weight,
                                output: output, outputOffset: outputOffset,
                                inputRows: inputRows, family: family)
                return
            case .hybrid:
                switch family {
                case .mlpUp, .mlpDown:
                    try encodeQGEMM(commandBuffer: commandBuffer, input: input,
                                    inputOffset: inputOffset, weight: weight,
                                    output: output, outputOffset: outputOffset,
                                    inputRows: inputRows, family: family)
                    return
                case .attentionProjection, .other:
                    break
                }
            case .aneHybridW8:
                break
            }
        }

        try encodeMPS(commandBuffer: commandBuffer, input: input,
                      inputOffset: inputOffset, weight: weight,
                      output: output, outputOffset: outputOffset,
                      inputRows: inputRows)
    }

    private func encodeMPS(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        inputOffset: Int,
        weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer,
        outputOffset: Int,
        inputRows: Int
    ) throws {
        let n = weight.rows
        let k = weight.columns
        let rightRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
        let scratchBytes = try checkedProduct(n, rightRowBytes)
        let scratch = buffers.buffer(key: "linear.weight.fp16", bytes: scratchBytes)
        let kernelName: String
        switch weight.storage {
        case .w4: kernelName = "dequant_w4_to_half"
        case .w8: kernelName = "dequant_w8_to_half"
        case .fp16: kernelName = "copy_half_rows"
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
        if weight.storage == .fp16 {
            encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 0)
            encoder.setBuffer(scratch, offset: 0, index: 1)
            encoder.setBytes(&columns, length: 4, index: 2)
            encoder.setBytes(&rows, length: 4, index: 3)
            encoder.setBytes(&rowStride, length: 4, index: 4)
            encoder.setBytes(&outputStride, length: 4, index: 5)
        } else {
            encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 0)
            encoder.setBuffer(weight.scale, offset: weight.scaleOffset, index: 1)
            encoder.setBuffer(weight.zero, offset: weight.zeroOffset, index: 2)
            encoder.setBuffer(scratch, offset: 0, index: 3)
            encoder.setBytes(&columns, length: 4, index: 4)
            encoder.setBytes(&rowStride, length: 4, index: 5)
            encoder.setBytes(&rows, length: 4, index: 6)
            encoder.setBytes(&outputStride, length: 4, index: 7)
        }
        let width = min(16, pipeline.threadExecutionWidth)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(
            MTLSize(width: k, height: n, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()
        metrics?.recordDequantizedWeightBytesWritten(UInt64(n * rightRowBytes))

        let scalarBytes = MemoryLayout<Float16>.stride
        let rightDescriptor = MPSMatrixDescriptor(
            rows: n, columns: k, rowBytes: rightRowBytes, dataType: .float16)
        let right = MPSMatrix(buffer: scratch, descriptor: rightDescriptor)
        let mpsLeftRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
        let mpsResultRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: n, dataType: .float16)
        let tightLeftRowBytes = k * scalarBytes
        let tightResultRowBytes = n * scalarBytes
        let inputEligible = directMPSIO && tightLeftRowBytes == mpsLeftRowBytes
        let outputEligible = directMPSIO && tightResultRowBytes == mpsResultRowBytes

        var rowStart = 0
        while rowStart < inputRows {
            let rowsThisTile = min(tileRows, inputRows - rowStart)
            let leftRowBytes = mpsLeftRowBytes
            let resultRowBytes = mpsResultRowBytes
            let left: MPSMatrix
            if inputEligible {
                left = MPSMatrix(
                    buffer: input,
                    offset: inputOffset + rowStart * k * scalarBytes,
                    descriptor: MPSMatrixDescriptor(
                        rows: rowsThisTile, columns: k,
                        rowBytes: leftRowBytes, dataType: .float16))
            } else {
                let leftScratch = buffers.buffer(
                    key: "linear.left.fp16", bytes: try checkedProduct(rowsThisTile, leftRowBytes))
                try encodeCopy(commandBuffer: commandBuffer,
                               source: input, sourceOffset: inputOffset + rowStart * k * scalarBytes,
                               destination: leftScratch, destinationOffset: 0,
                               columns: k, rows: rowsThisTile,
                               sourceStride: k, destinationStride: leftRowBytes / scalarBytes)
                left = MPSMatrix(
                    buffer: leftScratch,
                    descriptor: MPSMatrixDescriptor(
                        rows: rowsThisTile, columns: k,
                        rowBytes: leftRowBytes, dataType: .float16))
            }
            let result: MPSMatrix
            let resultBuffer: MTLBuffer
            if outputEligible {
                result = MPSMatrix(
                    buffer: output,
                    offset: outputOffset + rowStart * n * scalarBytes,
                    descriptor: MPSMatrixDescriptor(
                        rows: rowsThisTile, columns: n,
                        rowBytes: resultRowBytes, dataType: .float16))
                resultBuffer = output
            } else {
                let scratch = buffers.buffer(
                    key: "linear.result.fp16", bytes: try checkedProduct(rowsThisTile, resultRowBytes))
                result = MPSMatrix(
                    buffer: scratch,
                    descriptor: MPSMatrixDescriptor(
                        rows: rowsThisTile, columns: n,
                        rowBytes: resultRowBytes, dataType: .float16))
                resultBuffer = scratch
            }
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
            if !outputEligible {
                try encodeCopy(commandBuffer: commandBuffer,
                               source: resultBuffer, sourceOffset: 0,
                               destination: output,
                               destinationOffset: outputOffset + rowStart * n * scalarBytes,
                               columns: n, rows: rowsThisTile,
                               sourceStride: resultRowBytes / scalarBytes, destinationStride: n)
            }
            metrics?.recordLinearGEMMTile(
                directInput: inputEligible, directOutput: outputEligible)
            if !inputEligible {
                metrics?.recordConversionBytes(UInt64(rowsThisTile * k * scalarBytes))
            }
            if !outputEligible {
                metrics?.recordConversionBytes(UInt64(rowsThisTile * n * scalarBytes))
            }
            rowStart += rowsThisTile
        }
    }

    private func encodeQGEMM(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        inputOffset: Int,
        weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer,
        outputOffset: Int,
        inputRows: Int,
        family: DiTLinearFamily = .other
    ) throws {
        let m = inputRows
        let n = weight.rows
        let k = weight.columns
        guard m > 0, n > 0, k > 0 else {
            throw AnimapkError.validation("invalid QGEMM shape M=\(m) N=\(n) K=\(k)")
        }
        let isW4 = weight.storage == .w4
        let wide = n >= Self.qgemmWideProfileMinN
        let kernelName: String
        if isW4 {
            kernelName = wide ? "qgemm_8x16x64" : "qgemm_8x8x64"
        } else {
            kernelName = wide ? "qgemm_w8_8x16x64" : "qgemm_w8_8x8x64"
        }
        let pipeline = try context.pipeline(named: kernelName)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create QGEMM compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 1)
        encoder.setBuffer(weight.scale, offset: weight.scaleOffset, index: 2)
        encoder.setBuffer(weight.zero, offset: weight.zeroOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        var mM = UInt32(m), nN = UInt32(n), kK = UInt32(k)
        var rowStride = UInt32(weight.packedRowStride)
        var inputStride = UInt32(k)
        var outputStride = UInt32(n)
        encoder.setBytes(&mM, length: 4, index: 5)
        encoder.setBytes(&nN, length: 4, index: 6)
        encoder.setBytes(&kK, length: 4, index: 7)
        encoder.setBytes(&rowStride, length: 4, index: 8)
        encoder.setBytes(&inputStride, length: 4, index: 9)
        encoder.setBytes(&outputStride, length: 4, index: 10)
        let tm = 8
        let tn = wide ? 16 : 8
        let threads = wide ? 128 : 64
        let groupX = (m + tm - 1) / tm
        let groupY = (n + tn - 1) / tn
        encoder.dispatchThreadgroups(
            MTLSize(width: groupX, height: groupY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        metrics?.recordQGEMMCall(family: family)
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
        guard storage == .w4 || storage == .w8 || storage == .fp16 else {
            throw AnimapkError.validation("\(label) must be W4, W8, or FP16")
        }
        guard item.tensor.shape == [rows, columns] else {
            throw AnimapkError.validation("\(label) shape mismatch")
        }
        if storage == .fp16 {
            guard item.data.length == UInt64(rows * columns * 2),
                  item.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("\(label) FP16 layout mismatch")
            }
            return QuantizedLinearWeightBuffers(
                storage: storage, packed: ring, packedOffset: Int(item.data.offset),
                scale: ring, scaleOffset: Int(item.data.offset),
                zero: ring, zeroOffset: Int(item.data.offset), rows: rows,
                columns: columns, packedRowStride: columns)
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
        case .fp16: return "fp16_matvec_f32"
        default:
            throw AnimapkError.validation("DiT direct matvec requires W4 or W8")
        }
    }
}

// MARK: - Runtime DiT LoRA overlay

/// MPS-friendly immutable low-rank weights for one adapted projection.
struct DiTLoRAProjectionBuffers {
    let key: DiTLoRAKey
    let down: MTLBuffer
    let downRowBytes: Int
    let up: MTLBuffer
    let upRowBytes: Int
    let rank: Int
    let inputFeatures: Int
    let outputFeatures: Int
    let effectiveScale: Float
}

/// Generation-local backend-independent LoRA overlay. Base projections still
/// execute through their existing W4/W8/MPS/ANE backend. This object only
/// encodes `output += strength * alpha/rank * ((input A^T) B^T)` at the exact
/// projection-output boundary. No command buffer is committed or waited here.
final class DiTLoRAExecutor {
    private let context: MetalContext
    private let scratch: BufferPool
    private let projections: [DiTLoRAKey: DiTLoRAProjectionBuffers]

    let adapterBytes: Int
    let moduleCount: Int

    init(context: MetalContext, file: DiTLoRAFile, strength: Float) throws {
        guard strength.isFinite, strength != 0 else {
            throw AnimapkError.validation("LoRA executor requires a finite non-zero strength")
        }
        self.context = context
        self.scratch = BufferPool(device: context.device)

        var built: [DiTLoRAKey: DiTLoRAProjectionBuffers] = [:]
        built.reserveCapacity(file.modules.count)
        var totalBytes = 0
        for descriptor in file.modules.values {
            let downRowBytes = MPSMatrixDescriptor.rowBytes(
                fromColumns: descriptor.key.target.inputFeatures, dataType: .float16)
            let upRowBytes = MPSMatrixDescriptor.rowBytes(
                fromColumns: descriptor.rank, dataType: .float16)
            let downBytes = try Self.checkedProduct(descriptor.rank, downRowBytes)
            let upBytes = try Self.checkedProduct(
                descriptor.key.target.outputFeatures, upRowBytes)
            guard let down = context.device.makeBuffer(
                    length: downBytes, options: .storageModeShared),
                  let up = context.device.makeBuffer(
                    length: upBytes, options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate LoRA weight buffers")
            }
            try file.copyMatrixToFP16(
                descriptor.down, destination: down.contents(),
                destinationRowBytes: downRowBytes)
            try file.copyMatrixToFP16(
                descriptor.up, destination: up.contents(),
                destinationRowBytes: upRowBytes)
            let projection = DiTLoRAProjectionBuffers(
                key: descriptor.key,
                down: down, downRowBytes: downRowBytes,
                up: up, upRowBytes: upRowBytes,
                rank: descriptor.rank,
                inputFeatures: descriptor.key.target.inputFeatures,
                outputFeatures: descriptor.key.target.outputFeatures,
                effectiveScale: strength * descriptor.scale)
            built[descriptor.key] = projection
            totalBytes += downBytes + upBytes
        }
        self.projections = built
        self.adapterBytes = totalBytes
        self.moduleCount = built.count
    }

    func hasModule(block: Int, target: DiTLoRATarget) -> Bool {
        projections[DiTLoRAKey(block: block, target: target)] != nil
    }

    /// Encodes the adapter delta for one projection into the caller's existing
    /// command buffer. Returns false when this adapter does not target the
    /// projection, allowing call sites to remain zero-overhead when absent.
    @discardableResult
    func encode(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        inputRows: Int,
        block: Int,
        target: DiTLoRATarget
    ) throws -> Bool {
        guard let projection = projections[DiTLoRAKey(block: block, target: target)] else {
            return false
        }
        try Self.encodeLowRankResidual(
            context: context, scratch: scratch,
            commandBuffer: commandBuffer, input: input, output: output,
            inputRows: inputRows, projection: projection)
        return true
    }

    /// Internal test seam for a precise synthetic oracle. The two GEMMs are
    /// encoded back-to-back in one command buffer; MPS `beta=1` performs the
    /// residual add directly into the base output, so there is no CPU/GPU sync
    /// and no separate add kernel.
    static func encodeLowRankResidual(
        context: MetalContext,
        scratch: BufferPool,
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        inputRows: Int,
        projection: DiTLoRAProjectionBuffers
    ) throws {
        let m = inputRows
        let k = projection.inputFeatures
        let n = projection.outputFeatures
        let r = projection.rank
        guard m > 0, k > 0, n > 0, r > 0,
              projection.effectiveScale.isFinite else {
            throw AnimapkError.validation("invalid LoRA projection shape/scale")
        }
        let scalarBytes = MemoryLayout<Float16>.stride
        let inputRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
        let outputRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: n, dataType: .float16)
        let tightInputRowBytes = k * scalarBytes
        let tightOutputRowBytes = n * scalarBytes
        guard inputRowBytes == tightInputRowBytes,
              outputRowBytes == tightOutputRowBytes else {
            throw AnimapkError.validation(
                "LoRA projection requires direct MPS rows for K=\(k) N=\(n)")
        }
        let inputBytes = try checkedProduct(m, tightInputRowBytes)
        let outputBytes = try checkedProduct(m, tightOutputRowBytes)
        guard input.length >= inputBytes, output.length >= outputBytes else {
            throw AnimapkError.validation("LoRA projection input/output buffer is too small")
        }

        let rankRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: r, dataType: .float16)
        let rankScratch = scratch.buffer(
            key: "lora.rank.\(r).rows.\(m)",
            bytes: try checkedProduct(m, rankRowBytes))

        let x = MPSMatrix(
            buffer: input,
            descriptor: MPSMatrixDescriptor(
                rows: m, columns: k, rowBytes: inputRowBytes, dataType: .float16))
        let down = MPSMatrix(
            buffer: projection.down,
            descriptor: MPSMatrixDescriptor(
                rows: r, columns: k,
                rowBytes: projection.downRowBytes, dataType: .float16))
        let low = MPSMatrix(
            buffer: rankScratch,
            descriptor: MPSMatrixDescriptor(
                rows: m, columns: r, rowBytes: rankRowBytes, dataType: .float16))
        let up = MPSMatrix(
            buffer: projection.up,
            descriptor: MPSMatrixDescriptor(
                rows: n, columns: r,
                rowBytes: projection.upRowBytes, dataType: .float16))
        let y = MPSMatrix(
            buffer: output,
            descriptor: MPSMatrixDescriptor(
                rows: m, columns: n, rowBytes: outputRowBytes, dataType: .float16))

        let downMultiply = MPSMatrixMultiplication(
            device: context.device,
            transposeLeft: false, transposeRight: true,
            resultRows: m, resultColumns: r, interiorColumns: k,
            alpha: 1, beta: 0)
        downMultiply.encode(
            commandBuffer: commandBuffer,
            leftMatrix: x, rightMatrix: down, resultMatrix: low)

        let upMultiply = MPSMatrixMultiplication(
            device: context.device,
            transposeLeft: false, transposeRight: true,
            resultRows: m, resultColumns: n, interiorColumns: r,
            alpha: Double(projection.effectiveScale), beta: 1)
        upMultiply.encode(
            commandBuffer: commandBuffer,
            leftMatrix: low, rightMatrix: up, resultMatrix: y)
    }

    private static func checkedProduct(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw AnimapkError.validation("LoRA byte size overflow")
            }
            result = next
        }
        return result
    }
}
