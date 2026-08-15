import XCTest
import Metal
import MetalPerformanceShaders
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

    func testGroupedQueryAttentionUsesContiguousKVHeads() async throws {
        let context = try requireContext()
        let heads = 4, kvHeads = 2, dim = 4
        let query = [Float16](repeating: 0, count: heads * dim)
        let key = [Float16](repeating: 0, count: kvHeads * dim)
        let value = [Float16](repeating: 2, count: dim)
            + [Float16](repeating: 9, count: dim)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: heads * dim * 2, options: .storageModeShared))

        try await AttentionExecutor(context: context).execute(
            query: makeHalfBuffer(query, context: context),
            key: makeHalfBuffer(key, context: context),
            value: makeHalfBuffer(value, context: context), output: output,
            heads: heads, queryCount: 1, keyCount: 1, headDim: dim,
            keyValueHeads: kvHeads)

        let actual = readHalf(output, count: heads * dim)
        XCTAssertEqual(actual, [Float](repeating: 2, count: dim * 2)
            + [Float](repeating: 9, count: dim * 2))
        print("ATTENTION_GQA_GROUPED=PASS mapping=0,0,1,1")
    }

    func testFP32ScoresAndSoftmaxImprovesIndependentReferenceParity() async throws {
        let context = try requireContext()
        let queryCount = 19, keyCount = 37, dim = 64
        let query = (0..<(queryCount * dim)).map {
            Float16(sin(Float(($0 * 31 + 7) % 997) * 0.023) * 1.7)
        }
        let key = (0..<(keyCount * dim)).map {
            Float16(cos(Float(($0 * 17 + 11) % 991) * 0.029) * 1.5)
        }
        let value = (0..<(keyCount * dim)).map {
            Float16(sin(Float(($0 * 13 + 5) % 983) * 0.019) * 2.0)
        }
        let legacyOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * dim * 2, options: .storageModeShared))
        let fp32Out = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * dim * 2, options: .storageModeShared))

        let q = makeHalfBuffer(query, context: context)
        let k = makeHalfBuffer(key, context: context)
        let v = makeHalfBuffer(value, context: context)
        try await AttentionExecutor(context: context, numerics: .legacy).execute(
            query: q, key: k, value: v, output: legacyOut,
            heads: 1, queryCount: queryCount, keyCount: keyCount, headDim: dim)
        let fp32Executor = AttentionExecutor(context: context, numerics: .fp32ScoresAndSoftmax)
        XCTAssertEqual(try fp32Executor.maximumScoreScratchBytes(keyCount: keyCount, queryCount: queryCount),
                       queryCount * keyCount * 4,
                       "FP32 scratch should be bounded by the real query count")
        try await fp32Executor.execute(
            query: q, key: k, value: v, output: fp32Out,
            heads: 1, queryCount: queryCount, keyCount: keyCount, headDim: dim)

        let legacy = readHalf(legacyOut, count: queryCount * dim)
        let fp32 = readHalf(fp32Out, count: queryCount * dim)
        let reference = (0..<queryCount).flatMap {
            attentionRowFP32(row: $0, query: query, key: key, value: value,
                             keyCount: keyCount, dim: dim)
        }
        let legacyMetrics = metrics(legacy, reference)
        let fp32Metrics = metrics(fp32, reference)
        print("ATTN_MODE=legacy ATTN_COSINE=\(legacyMetrics.cosine) ATTN_RMSE=\(legacyMetrics.rmse) ATTN_MAX_ABS=\(legacyMetrics.maxAbs)")
        print("ATTN_MODE=fp32_scores_softmax ATTN_COSINE=\(fp32Metrics.cosine) ATTN_RMSE=\(fp32Metrics.rmse) ATTN_MAX_ABS=\(fp32Metrics.maxAbs)")
        XCTAssertLessThan(fp32Metrics.rmse, legacyMetrics.rmse)
        XCTAssertLessThanOrEqual(fp32Metrics.maxAbs, legacyMetrics.maxAbs)
    }

    // MARK: - Runtime attention tile-size experiments (§18.3)

    /// Every allowed attention tile row must reproduce the baseline (128)
    /// result within the existing tolerance, because only the query-row
    /// dispatch granularity changes — the attention math is unchanged.
    func testAllTileRowsMatchBaseline() async throws {
        let context = try requireContext()
        let queryCount = 1_024, keyCount = 1_024, dim = 64
        let query = (0..<(queryCount * dim)).map {
            Float16(sin(Float($0 * 13 % 997) * 0.017) * 0.25)
        }
        let key = (0..<(queryCount * dim)).map {
            Float16(cos(Float($0 * 7 % 991) * 0.019) * 0.25)
        }
        let value = (0..<(queryCount * dim)).map {
            Float16(Float(($0 * 17 % 101) - 50) / 50)
        }
        let q = makeHalfBuffer(query, context: context)
        let k = makeHalfBuffer(key, context: context)
        let v = makeHalfBuffer(value, context: context)

        let referenceOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * dim * 2, options: .storageModeShared))
        try await AttentionExecutor(context: context, tileRows: 128).execute(
            query: q, key: k, value: v, output: referenceOut,
            heads: 1, queryCount: queryCount, keyCount: keyCount, headDim: dim)
        let reference = readHalf(referenceOut, count: queryCount * dim)

        for tileRows in [256, 512, 1024] {
            let collector = MetricsCollector()
            let output = try XCTUnwrap(context.device.makeBuffer(
                length: queryCount * dim * 2, options: .storageModeShared))
            let executor = AttentionExecutor(context: context, tileRows: tileRows)
            executor.metrics = collector
            try await executor.execute(
                query: q, key: k, value: v, output: output,
                heads: 1, queryCount: queryCount, keyCount: keyCount, headDim: dim)
            let actual = readHalf(output, count: queryCount * dim)
            for index in actual.indices {
                XCTAssertEqual(actual[index], reference[index], accuracy: 0.012,
                               "tile \(tileRows) diverges at \(index)")
            }
            let expectedTiles = (queryCount + tileRows - 1) / tileRows
            XCTAssertEqual(collector.snapshot().attentionQueryTiles, expectedTiles,
                           "attention query-tile counter for tile \(tileRows)")
        }
    }

    /// The attention query-tile counter counts every (head, tile) execution.
    func testAttentionQueryTileCounterCountsHeadsAndTiles() async throws {
        let context = try requireContext()
        let heads = 2, queryCount = 130, keyCount = 64, dim = 8
        // Per-head layout: [heads, queryCount, dim].
        let zeros = [Float16](repeating: 0, count: heads * queryCount * dim)
        let zeroKey = [Float16](repeating: 0, count: keyCount * dim)
        let value = [Float16](repeating: 1, count: keyCount * dim)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: heads * queryCount * dim * 2, options: .storageModeShared))
        let collector = MetricsCollector()
        let executor = AttentionExecutor(context: context, tileRows: 128)
        executor.metrics = collector
        try await executor.execute(
            query: makeHalfBuffer(zeros, context: context),
            key: makeHalfBuffer(zeroKey, context: context),
            value: makeHalfBuffer(value, context: context), output: output,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: dim)
        // 2 heads × ceil(130/128) = 2 × 2 = 4 tiles.
        XCTAssertEqual(collector.snapshot().attentionQueryTiles, 4)
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

    private func attentionRowFP32(
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
            scores[keyRow] = dot * scale
        }
        let maximum = scores.max()!
        let exponentials = scores.map { exp($0 - maximum) }
        let sum = exponentials.reduce(0, +)
        var result = [Float](repeating: 0, count: dim)
        for keyRow in 0..<keyCount {
            let probability = exponentials[keyRow] / sum
            for column in 0..<dim {
                result[column] += probability * Float(value[keyRow * dim + column])
            }
        }
        return result
    }

    private func metrics(_ actual: [Float], _ expected: [Float])
        -> (cosine: Float, rmse: Float, maxAbs: Float) {
        var dot: Float = 0, actualNorm: Float = 0, expectedNorm: Float = 0
        var squaredError: Float = 0, maxAbs: Float = 0
        for (a, e) in zip(actual, expected) {
            dot += a * e
            actualNorm += a * a
            expectedNorm += e * e
            let error = abs(a - e)
            squaredError += error * error
            maxAbs = max(maxAbs, error)
        }
        return (dot / sqrt(actualNorm * expectedNorm),
                sqrt(squaredError / Float(actual.count)), maxAbs)
    }
}
