import Foundation

/// A prevalidated byte span relative to the start of an execution range.
struct AnimapkRelativeSpan: Equatable {
    let offset: UInt64
    let length: UInt64

    var range: Range<UInt64> { offset..<(offset + length) }
}

/// All regions of one tensor, expressed relative to its execution range.
struct AnimapkTensorSpans: Equatable {
    let tensor: AnimapkTensor
    let blob: AnimapkRelativeSpan
    let data: AnimapkRelativeSpan
    let scale: AnimapkRelativeSpan?
    let zero: AnimapkRelativeSpan?
}

/// One contiguous mmap range copied into the one-slot streaming ring.
struct AnimapkExecutionRange: Equatable {
    let logicalIndex: Int
    let fileOffset: UInt64
    let length: UInt64
    let tensors: [AnimapkTensorSpans]

    var fileRange: Range<UInt64> { fileOffset..<(fileOffset + length) }
}

/// Row regions for a quantized rank-2 tensor, relative to that tensor's blob.
struct AnimapkQuantizedRowSpans: Equatable {
    let row: Int
    let data: AnimapkRelativeSpan
    let scale: AnimapkRelativeSpan
    let zero: AnimapkRelativeSpan
}

enum AnimapkRangeBuilder {
    static func executionRanges(
        tensors: [AnimapkTensor],
        prefix: String,
        count: Int
    ) throws -> [AnimapkExecutionRange] {
        var result: [AnimapkExecutionRange] = []
        result.reserveCapacity(count)

        for logicalIndex in 0..<count {
            let exactPrefix = "\(prefix)\(logicalIndex)."
            result.append(try executionRange(
                tensors: tensors, exactPrefix: exactPrefix, logicalIndex: logicalIndex))
        }

        let physical = result.sorted { $0.fileOffset < $1.fileOffset }
        for pair in zip(physical, physical.dropFirst()) {
            guard pair.0.fileRange.upperBound <= pair.1.fileRange.lowerBound else {
                throw AnimapkError.validation(
                    "execution ranges \(pair.0.logicalIndex) and \(pair.1.logicalIndex) overlap")
            }
        }
        return result
    }

    static func executionRange(
        tensors: [AnimapkTensor], exactPrefix: String, logicalIndex: Int
    ) throws -> AnimapkExecutionRange {
        let members = tensors.filter { $0.name.hasPrefix(exactPrefix) }
        guard !members.isEmpty else {
            throw AnimapkError.validation("no tensors found for \(exactPrefix)")
        }
        let start = members.map(\.blobOffset).min()!
        var end = start
        for tensor in members {
            end = max(end, try checkedAdd(tensor.blobOffset, tensor.blobSize, label: tensor.name))
        }
        var spans: [AnimapkTensorSpans] = []
        for tensor in members.sorted(by: { $0.blobOffset < $1.blobOffset }) {
            guard tensor.blobOffset % 16_384 == 0 else {
                throw AnimapkError.validation("blob \(tensor.name) is not 16 KB aligned")
            }
            let blob = try relativeSpan(
                absoluteOffset: tensor.blobOffset, length: tensor.blobSize,
                rangeStart: start, rangeEnd: end, label: "\(tensor.name) blob")
            let dataAbsolute = try checkedAdd(
                tensor.blobOffset, tensor.dataOffset, label: "\(tensor.name) data")
            let data = try relativeSpan(
                absoluteOffset: dataAbsolute, length: tensor.dataSize,
                rangeStart: start, rangeEnd: end, label: "\(tensor.name) data")
            let scale = try optionalSpan(
                offset: tensor.scaleOffset, length: tensor.scaleSize, tensor: tensor,
                rangeStart: start, rangeEnd: end, label: "scale")
            let zero = try optionalSpan(
                offset: tensor.zeroOffset, length: tensor.zeroSize, tensor: tensor,
                rangeStart: start, rangeEnd: end, label: "zero")
            spans.append(AnimapkTensorSpans(
                tensor: tensor, blob: blob, data: data, scale: scale, zero: zero))
        }
        return AnimapkExecutionRange(
            logicalIndex: logicalIndex, fileOffset: start, length: end - start, tensors: spans)
    }

    static func tensorSpans(_ tensor: AnimapkTensor) throws -> AnimapkTensorSpans {
        let start = tensor.blobOffset
        let end = try checkedAdd(start, tensor.blobSize, label: tensor.name)
        let blob = AnimapkRelativeSpan(offset: 0, length: tensor.blobSize)
        let dataAbsolute = try checkedAdd(start, tensor.dataOffset, label: "\(tensor.name) data")
        let data = try relativeSpan(
            absoluteOffset: dataAbsolute, length: tensor.dataSize,
            rangeStart: start, rangeEnd: end, label: "\(tensor.name) data")
        let scale = try optionalSpan(
            offset: tensor.scaleOffset, length: tensor.scaleSize, tensor: tensor,
            rangeStart: start, rangeEnd: end, label: "scale")
        let zero = try optionalSpan(
            offset: tensor.zeroOffset, length: tensor.zeroSize, tensor: tensor,
            rangeStart: start, rangeEnd: end, label: "zero")
        return AnimapkTensorSpans(tensor: tensor, blob: blob, data: data, scale: scale, zero: zero)
    }

    private static func optionalSpan(
        offset: UInt64?, length: UInt64?, tensor: AnimapkTensor,
        rangeStart: UInt64, rangeEnd: UInt64, label: String
    ) throws -> AnimapkRelativeSpan? {
        guard offset != nil || length != nil else { return nil }
        guard let offset, let length else {
            throw AnimapkError.validation("\(tensor.name) has incomplete \(label) metadata")
        }
        let absolute = try checkedAdd(tensor.blobOffset, offset, label: "\(tensor.name) \(label)")
        return try relativeSpan(
            absoluteOffset: absolute, length: length,
            rangeStart: rangeStart, rangeEnd: rangeEnd, label: "\(tensor.name) \(label)")
    }

    private static func relativeSpan(
        absoluteOffset: UInt64, length: UInt64,
        rangeStart: UInt64, rangeEnd: UInt64, label: String
    ) throws -> AnimapkRelativeSpan {
        let absoluteEnd = try checkedAdd(absoluteOffset, length, label: label)
        guard absoluteOffset >= rangeStart, absoluteEnd <= rangeEnd else {
            throw AnimapkError.validation("\(label) lies outside its execution range")
        }
        return AnimapkRelativeSpan(offset: absoluteOffset - rangeStart, length: length)
    }

    static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64, label: String) throws -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw AnimapkError.validation("byte offset overflow for \(label)") }
        return sum
    }

    static func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, label: String) throws -> UInt64 {
        let (product, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw AnimapkError.validation("byte length overflow for \(label)") }
        return product
    }
}

struct DiTBlockLocator {
    static let blockCount = 28
    let blocks: [AnimapkExecutionRange]

    init(file: AnimapkFile) throws {
        blocks = try AnimapkRangeBuilder.executionRanges(
            tensors: file.tensors,
            prefix: "model.diffusion_model.blocks.",
            count: Self.blockCount
        )
    }

    func block(_ logicalIndex: Int) throws -> AnimapkExecutionRange {
        guard blocks.indices.contains(logicalIndex) else {
            throw AnimapkError.validation("DiT block index \(logicalIndex) is out of range")
        }
        return blocks[logicalIndex]
    }
}

/// Compact per-block stream ranges for the ANE-native hybrid pack.
///
/// The packer physically places the ten ANE projection blobs after the tensors
/// still consumed by Metal. This locator excludes those ANE-native blobs and
/// returns only the contiguous Metal prefix so the streaming ring never copies
/// ~63 MiB/block of projection weights that ANE reads from its prepared model
/// cache instead.
struct DiTANEHybridMetalLocator {
    static let nativeFormat = "ane_u8_per_row_fp32_v1"
    static let expectedNativePerBlock = 10
    let blocks: [AnimapkExecutionRange]

    init(file: AnimapkFile) throws {
        guard file.component == "dit", file.quantScheme == "w8-ane-hybrid-v1" else {
            throw AnimapkError.validation("ANE hybrid Metal locator requires w8-ane-hybrid-v1 DiT pack")
        }
        var ranges: [AnimapkExecutionRange] = []
        ranges.reserveCapacity(DiTBlockLocator.blockCount)
        for logicalIndex in 0..<DiTBlockLocator.blockCount {
            ranges.append(try Self.metalRange(tensors: file.tensors, logicalIndex: logicalIndex))
        }
        self.blocks = ranges
    }

    static func metalRange(
        tensors: [AnimapkTensor], logicalIndex: Int
    ) throws -> AnimapkExecutionRange {
        let prefix = "model.diffusion_model.blocks.\(logicalIndex)."
        let all = tensors.filter { $0.name.hasPrefix(prefix) }
        let native = all.filter { $0.quantizationFormat == Self.nativeFormat }
        let metal = all.filter { $0.quantizationFormat != Self.nativeFormat }
        guard all.count == 20, native.count == Self.expectedNativePerBlock, metal.count == 10 else {
            throw AnimapkError.validation(
                "ANE hybrid block \(logicalIndex) must contain 10 Metal + 10 native projection tensors")
        }
        let range = try AnimapkRangeBuilder.executionRange(
            tensors: metal, exactPrefix: prefix, logicalIndex: logicalIndex)
        let metalEnd = range.fileRange.upperBound
        guard let firstNative = native.map(\.blobOffset).min(), firstNative >= metalEnd else {
            throw AnimapkError.validation(
                "ANE hybrid block \(logicalIndex) projection blobs interleave its Metal stream range")
        }
        return range
    }

    func block(_ logicalIndex: Int) throws -> AnimapkExecutionRange {
        guard blocks.indices.contains(logicalIndex) else {
            throw AnimapkError.validation("DiT ANE hybrid block index \(logicalIndex) is out of range")
        }
        return blocks[logicalIndex]
    }
}

struct DiTFinalLayerLocator {
    let range: AnimapkExecutionRange

    init(file: AnimapkFile) throws {
        range = try AnimapkRangeBuilder.executionRange(
            tensors: file.tensors,
            exactPrefix: "model.diffusion_model.final_layer.",
            logicalIndex: -1)
        guard range.tensors.count == 3 else {
            throw AnimapkError.validation("DiT final layer must contain exactly 3 tensors")
        }
    }
}

/// Contiguous input/timestep tensors immediately following the adapter/final ranges.
struct DiTPreparationLocator {
    let range: AnimapkExecutionRange

    init(file: AnimapkFile) throws {
        let names = Set([
            "model.diffusion_model.t_embedder.1.linear_1.weight",
            "model.diffusion_model.t_embedder.1.linear_2.weight",
            "model.diffusion_model.t_embedding_norm.weight",
            "model.diffusion_model.x_embedder.proj.1.weight"
        ])
        range = try AnimapkRangeBuilder.executionRange(
            tensors: file.tensors.filter { names.contains($0.name) },
            exactPrefix: "model.diffusion_model.", logicalIndex: -2)
        guard Set(range.tensors.map(\.tensor.name)) == names else {
            throw AnimapkError.validation("DiT preparation range is incomplete")
        }
    }
}

struct QwenLayerLocator {
    static let layerCount = 28
    let embedding: AnimapkTensorSpans
    let embeddingFileOffset: UInt64
    let layers: [AnimapkExecutionRange]
    private let quantGroup: Int

    init(file: AnimapkFile) throws {
        guard let tensor = file.tensor(named: "model.embed_tokens.weight") else {
            throw AnimapkError.validation("Qwen embedding tensor is missing")
        }
        guard tensor.blobOffset % 16_384 == 0 else {
            throw AnimapkError.validation("Qwen embedding blob is not 16 KB aligned")
        }
        embedding = try AnimapkRangeBuilder.tensorSpans(tensor)
        embeddingFileOffset = tensor.blobOffset
        layers = try AnimapkRangeBuilder.executionRanges(
            tensors: file.tensors, prefix: "model.layers.", count: Self.layerCount)
        guard let group = file.quantGroup, group > 0 else {
            throw AnimapkError.validation("Qwen quantization group is missing or invalid")
        }
        quantGroup = group
    }

    func layer(_ logicalIndex: Int) throws -> AnimapkExecutionRange {
        guard layers.indices.contains(logicalIndex) else {
            throw AnimapkError.validation("Qwen layer index \(logicalIndex) is out of range")
        }
        return layers[logicalIndex]
    }

    func embeddingRow(_ row: Int) throws -> AnimapkQuantizedRowSpans {
        let tensor = embedding.tensor
        guard tensor.shape.count == 2 else {
            throw AnimapkError.validation("Qwen embedding must be rank 2")
        }
        let rows = tensor.shape[0]
        let columns = tensor.shape[1]
        guard rows > 0, row >= 0, row < rows, columns > 0 else {
            throw AnimapkError.validation("Qwen embedding row \(row) is out of range")
        }

        let dataBytesPerRow: UInt64
        switch tensor.storage {
        case .w8:
            dataBytesPerRow = UInt64(columns)
        case .w4:
            guard columns.isMultiple(of: 2) else {
                throw AnimapkError.validation("odd-width W4 embedding rows are unsupported")
            }
            dataBytesPerRow = UInt64(columns / 2)
        default:
            throw AnimapkError.validation("Qwen embedding is not quantized")
        }
        let groupsPerRow = (columns + quantGroup - 1) / quantGroup
        let parameterBytesPerRow = UInt64(groupsPerRow * MemoryLayout<UInt16>.size)
        let expectedData = try AnimapkRangeBuilder.checkedMultiply(
            UInt64(rows), dataBytesPerRow, label: "embedding data")
        let expectedParameters = try AnimapkRangeBuilder.checkedMultiply(
            UInt64(rows), parameterBytesPerRow, label: "embedding parameters")
        guard embedding.data.length == expectedData,
              embedding.scale?.length == expectedParameters,
              embedding.zero?.length == expectedParameters,
              let scale = embedding.scale, let zero = embedding.zero else {
            throw AnimapkError.validation("Qwen embedding row layout does not match metadata")
        }
        return AnimapkQuantizedRowSpans(
            row: row,
            data: AnimapkRelativeSpan(
                offset: embedding.data.offset + UInt64(row) * dataBytesPerRow,
                length: dataBytesPerRow),
            scale: AnimapkRelativeSpan(
                offset: scale.offset + UInt64(row) * parameterBytesPerRow,
                length: parameterBytesPerRow),
            zero: AnimapkRelativeSpan(
                offset: zero.offset + UInt64(row) * parameterBytesPerRow,
                length: parameterBytesPerRow)
        )
    }

}

/// Prevalidated execution ranges for the six lllite adapter blocks plus its
/// independently streamed embedding/final tensors.
struct LLMAdapterLocator {
    static let blockCount = 6
    let embedding: AnimapkTensorSpans
    let embeddingFileOffset: UInt64
    let blocks: [AnimapkExecutionRange]
    let final: AnimapkExecutionRange
    private let quantGroup: Int

    init(file: AnimapkFile) throws {
        let prefix = "model.diffusion_model.llm_adapter."
        guard let embeddingTensor = file.tensor(named: prefix + "embed.weight") else {
            throw AnimapkError.validation("adapter embedding tensor is missing")
        }
        embedding = try AnimapkRangeBuilder.tensorSpans(embeddingTensor)
        embeddingFileOffset = embeddingTensor.blobOffset
        blocks = try AnimapkRangeBuilder.executionRanges(
            tensors: file.tensors, prefix: prefix + "blocks.", count: Self.blockCount)
        let finalTensorNames = Set([
            prefix + "norm.weight", prefix + "out_proj.bias", prefix + "out_proj.weight"
        ])
        final = try AnimapkRangeBuilder.executionRange(
            tensors: file.tensors.filter { finalTensorNames.contains($0.name) },
            exactPrefix: prefix, logicalIndex: Self.blockCount)
        guard let group = file.quantGroup, group > 0 else {
            throw AnimapkError.validation("adapter quantization group is missing or invalid")
        }
        quantGroup = group
        let finalNames = Set(final.tensors.map(\.tensor.name))
        guard finalNames == finalTensorNames else {
            throw AnimapkError.validation("adapter final range contains unexpected tensors")
        }
    }

    func block(_ logicalIndex: Int) throws -> AnimapkExecutionRange {
        guard blocks.indices.contains(logicalIndex) else {
            throw AnimapkError.validation("adapter block index \(logicalIndex) is out of range")
        }
        return blocks[logicalIndex]
    }

    func embeddingRow(_ row: Int) throws -> AnimapkQuantizedRowSpans {
        let tensor = embedding.tensor
        guard tensor.shape.count == 2,
              (tensor.storage == .w4 || tensor.storage == .w8) else {
            throw AnimapkError.validation("adapter embedding must be rank-2 W4 or W8")
        }
        let rows = tensor.shape[0], columns = tensor.shape[1]
        guard rows > 0, row >= 0, row < rows, columns > 0 else {
            throw AnimapkError.validation("adapter embedding row \(row) is out of range")
        }
        let dataBytesPerRow = tensor.storage == .w4
            ? UInt64((columns + 1) / 2)
            : UInt64(columns)
        let groupsPerRow = (columns + quantGroup - 1) / quantGroup
        let parameterBytesPerRow = UInt64(groupsPerRow * MemoryLayout<UInt16>.size)
        guard embedding.data.length == UInt64(rows) * dataBytesPerRow,
              embedding.scale?.length == UInt64(rows) * parameterBytesPerRow,
              embedding.zero?.length == UInt64(rows) * parameterBytesPerRow,
              let scale = embedding.scale, let zero = embedding.zero else {
            throw AnimapkError.validation("adapter embedding row layout does not match metadata")
        }
        return AnimapkQuantizedRowSpans(
            row: row,
            data: AnimapkRelativeSpan(
                offset: embedding.data.offset + UInt64(row) * dataBytesPerRow,
                length: dataBytesPerRow),
            scale: AnimapkRelativeSpan(
                offset: scale.offset + UInt64(row) * parameterBytesPerRow,
                length: parameterBytesPerRow),
            zero: AnimapkRelativeSpan(
                offset: zero.offset + UInt64(row) * parameterBytesPerRow,
                length: parameterBytesPerRow))
    }

    /// Returns one source-FP16 embedding row. FP16 rows have no scale/zero
    /// spans, so they are exposed separately from the quantized-row API.
    func embeddingDataRow(_ row: Int) throws -> AnimapkRelativeSpan {
        let tensor = embedding.tensor
        guard tensor.shape.count == 2, tensor.storage == .fp16 else {
            throw AnimapkError.validation("adapter embedding is not FP16")
        }
        let rows = tensor.shape[0], columns = tensor.shape[1]
        guard rows > 0, row >= 0, row < rows, columns == 1_024 else {
            throw AnimapkError.validation("adapter FP16 embedding row \(row) is out of range")
        }
        let bytesPerRow = UInt64(columns * MemoryLayout<Float16>.stride)
        guard embedding.data.length == UInt64(rows) * bytesPerRow,
              (embedding.scale?.length ?? 0) == 0,
              (embedding.zero?.length ?? 0) == 0 else {
            throw AnimapkError.validation("adapter FP16 embedding row layout does not match metadata")
        }
        return AnimapkRelativeSpan(
            offset: embedding.data.offset + UInt64(row) * bytesPerRow,
            length: bytesPerRow)
    }
}
