import XCTest
import Metal
@testable import AnimaXS

final class AttentionExecutorTests: XCTestCase {
    private func requireContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        print("ATTENTION_METAL_DEVICE=\(context.device.name)")
        return context
    }

    private func makeHalfBuffer(_ values: [Float16], context: MetalContext) -> MTLBuffer {
        let bits = values.map(\.bitPattern)
        return bits.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                      options: .storageModeShared)!
        }
    }

    private func readHalf(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Float(Float16(bitPattern: pointer[$0])) }
    }

    func testRepresentativeSelfAttentionIsQueryTiled() async throws {
        let context = try requireContext()
        let count = 1_024, dim = 64
        let query = (0..<(count * dim)).map {
            Float16(sin(Float($0 * 13 % 997) * 0.017) * 0.25)
        }
        let key = (0..<(count * dim)).map {
            Float16(cos(Float($0 * 7 % 991) * 0.019) * 0.25)
        }
        let value = (0..<(count * dim)).map {
            Float16(Float(($0 * 17 % 101) - 50) / 50)
        }
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: count * dim * 2, options: .storageModeShared))
        let executor = AttentionExecutor(context: context)
        XCTAssertEqual(try executor.maximumScoreScratchBytes(keyCount: count), 128 * count * 2)

        try await executor.execute(
            query: makeHalfBuffer(query, context: context),
            key: makeHalfBuffer(key, context: context),
            value: makeHalfBuffer(value, context: context), output: output,
            heads: 1, queryCount: count, keyCount: count, headDim: dim)

        let actual = readHalf(output, count: count * dim)
        for row in [0, 127, 128, 511, 1_023] {
            let expected = attentionRow(
                row: row, query: query, key: key, value: value,
                keyCount: count, dim: dim)
            for column in 0..<dim {
                XCTAssertEqual(actual[row * dim + column], expected[column], accuracy: 0.012,
                               "self row \(row), column \(column)")
            }
        }
        print("ATTENTION_SELF_TILED=PASS Q=1024 K=1024 tile=128")
    }

    func testCrossAttentionRetainsAll512RowsAndTailTile() async throws {
        let context = try requireContext()
        let queryCount = 129, keyCount = 512, dim = 64
        let query = [Float16](repeating: 0, count: queryCount * dim)
        let key = [Float16](repeating: 0, count: keyCount * dim)
        var value = [Float16](repeating: 0, count: keyCount * dim)
        for column in 0..<dim { value[(keyCount - 1) * dim + column] = Float16(column + 1) }
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * dim * 2, options: .storageModeShared))

        try await AttentionExecutor(context: context).execute(
            query: makeHalfBuffer(query, context: context),
            key: makeHalfBuffer(key, context: context),
            value: makeHalfBuffer(value, context: context), output: output,
            heads: 1, queryCount: queryCount, keyCount: keyCount, headDim: dim)

        let actual = readHalf(output, count: queryCount * dim)
        for row in [0, 127, 128] {
            for column in 0..<dim {
                XCTAssertEqual(actual[row * dim + column], Float(column + 1) / 512,
                               accuracy: 0.0001, "cross row \(row), column \(column)")
            }
        }
        print("ATTENTION_CROSS_512=PASS Q=129 K=512")
    }

    private func attentionRow(
        row: Int, query: [Float16], key: [Float16], value: [Float16],
        keyCount: Int, dim: Int
    ) -> [Float] {
        let scale = 1 / sqrt(Float(dim))
        var scores = [Float](repeating: 0, count: keyCount)
        for keyRow in 0..<keyCount {
            var dot: Float = 0
            for column in 0..<dim {
                dot += Float(query[row * dim + column]) * Float(key[keyRow * dim + column])
            }
            scores[keyRow] = Float(Float16(dot * scale))
        }
        let maximum = scores.max()!
        let exponentials = scores.map { exp($0 - maximum) }
        let sum = exponentials.reduce(0, +)
        let probabilities = exponentials.map { Float(Float16($0 / sum)) }
        var result = [Float](repeating: 0, count: dim)
        for keyRow in 0..<keyCount {
            for column in 0..<dim {
                result[column] += probabilities[keyRow] * Float(value[keyRow * dim + column])
            }
        }
        return result.map { Float(Float16($0)) }
    }
}
