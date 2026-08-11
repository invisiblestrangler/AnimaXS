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

private enum AnimapkRangeBuilder {
    static func executionRanges(
        tensors: [AnimapkTensor],
        prefix: String,
        count: Int
    ) throws -> [AnimapkExecutionRange] {
        var result: [AnimapkExecutionRange] = []
        result.reserveCapacity(count)

        for logicalIndex in 0..<count {
            let exactPrefix = "\(prefix)\(logicalIndex)."
            let members = tensors.filter { $0.name.hasPrefix(exactPrefix) }
            guard !members.isEmpty else {
                throw AnimapkError.validation("no tensors found for \(exactPrefix)")
            }

            let start = members.map(\.blobOffset).min()!
            var end = start
            for tensor in members {
                let blobEnd = try checkedAdd(tensor.blobOffset, tensor.blobSize, label: tensor.name)
                end = max(end, blobEnd)
            }
            let length = end - start
            var spans: [AnimapkTensorSpans] = []
            spans.reserveCapacity(members.count)
            for tensor in members.sorted(by: { $0.blobOffset < $1.blobOffset }) {
                guard tensor.blobOffset % 16_384 == 0 else {
                    throw AnimapkError.validation("blob \(tensor.name) is not 16 KB aligned")
                }
                let blob = try relativeSpan(
                    absoluteOffset: tensor.blobOffset, length: tensor.blobSize,
                    rangeStart: start, rangeEnd: end, label: "\(tensor.name) blob")
                let dataAbsolute = try checkedAdd(tensor.blobOffset, tensor.dataOffset, label: "\(tensor.name) data")
                let data = try relativeSpan(
                    absoluteOffset: dataAbsolute, length: tensor.dataSize,
                    rangeStart: start, rangeEnd: end, label: "\(tensor.name) data")
                let scale = try optionalSpan(
                    offset: tensor.scaleOffset, length: tensor.scaleSize, tensor: tensor,
                    rangeStart: start, rangeEnd: end, label: "scale")
                let zero = try optionalSpan(
                    offset: tensor.zeroOffset, length: tensor.zeroSize, tensor: tensor,
                    rangeStart: start, rangeEnd: end, label: "zero")
                spans.append(AnimapkTensorSpans(tensor: tensor, blob: blob, data: data, scale: scale, zero: zero))
            }
            result.append(AnimapkExecutionRange(
                logicalIndex: logicalIndex, fileOffset: start, length: length, tensors: spans))
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
