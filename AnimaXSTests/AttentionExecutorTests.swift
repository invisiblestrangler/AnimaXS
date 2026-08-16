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
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: dim,
            keyValueHeads: 1)
        // 2 heads × ceil(130/128) = 2 × 2 = 4 tiles.
        XCTAssertEqual(collector.snapshot().attentionQueryTiles, 4)
    }

    // MARK: - P4 strided token-major attention (§9)

    /// P4: strided token-major attention must reproduce the legacy head-major
    /// result. The token-major buffer is [queryCount, modelDim] with heads
    /// occupying headDim contiguous columns per row; values are DISTINGUISHABLE
    /// per token/head so a layout mistake (head mixing, wrong stride, wrong
    /// offset) shows up as a large error.
    func testStridedTokenMajorMatchesLegacyHeadMajor() async throws {
        let context = try requireContext()
        // Use a production-realistic headDim=128 so the strided per-head MPS
        // matrix offsets (head * headDim * 2 bytes) and rowBytes are 256-byte
        // aligned, exactly as in the DiT shape. A smaller headDim would produce
        // unaligned offsets that MPSMatrix rejects/corrupts.
        let heads = 2, headDim = 128, modelDim = heads * headDim
        let queryCount = 7, keyCount = 5
        func tokenMajorValues(rows: Int) -> [Float16] {
            (0..<(rows * modelDim)).map { index in
                let token = index / modelDim
                let head = (index % modelDim) / headDim
                let column = index % headDim
                return Float16(Float(token * 1000 + head * 100 + column) / 997 - 0.5)
            }
        }
        let query = tokenMajorValues(rows: queryCount)
        let key = tokenMajorValues(rows: keyCount)
        let value = tokenMajorValues(rows: keyCount)

        let qBuffer = makeHalfBuffer(query, context: context)
        let kBuffer = makeHalfBuffer(key, context: context)
        let vBuffer = makeHalfBuffer(value, context: context)

        // Reference: legacy head-major path with manually transposed buffers.
        let qHead = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * modelDim * 2, options: .storageModeShared))
        let kHead = try XCTUnwrap(context.device.makeBuffer(
            length: keyCount * modelDim * 2, options: .storageModeShared))
        let vHead = try XCTUnwrap(context.device.makeBuffer(
            length: keyCount * modelDim * 2, options: .storageModeShared))
        let referenceOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * modelDim * 2, options: .storageModeShared))
        try encodeHeadMajor(
            context: context, input: qBuffer, output: qHead,
            rows: queryCount, heads: heads, headDim: headDim)
        try encodeHeadMajor(
            context: context, input: kBuffer, output: kHead,
            rows: keyCount, heads: heads, headDim: headDim)
        try encodeHeadMajor(
            context: context, input: vBuffer, output: vHead,
            rows: keyCount, heads: heads, headDim: headDim)
        try await AttentionExecutor(context: context).execute(
            query: qHead, key: kHead, value: vHead, output: referenceOut,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim)
        // The legacy path writes HEAD-MAJOR [heads, rows, headDim]; the strided
        // path writes TOKEN-MAJOR [rows, modelDim]. Transpose the reference back
        // to token-major so the two layouts are directly comparable.
        let referenceTokenMajor = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * modelDim * 2, options: .storageModeShared))
        try encodeHeadMajor(
            context: context, input: referenceOut, output: referenceTokenMajor,
            rows: queryCount, heads: heads, headDim: headDim, direction: 0)
        let reference = readHalf(referenceTokenMajor, count: queryCount * modelDim)

        // Strided token-major path (P4): same buffers, no transposes.
        let stridedOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * modelDim * 2, options: .storageModeShared))
        try await AttentionExecutor(context: context).execute(
            query: qBuffer, key: kBuffer, value: vBuffer, output: stridedOut,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
            layout: .tokenMajor(tokenStride: modelDim))
        let strided = readHalf(stridedOut, count: queryCount * modelDim)

        for index in strided.indices {
            XCTAssertEqual(strided[index], reference[index], accuracy: 0.002,
                           "P4 strided token-major diverges at \(index)")
        }
        print("ATTENTION_STRIDED_TOKEN_MAJOR=PASS heads=\(heads) Q=\(queryCount) K=\(keyCount) modelDim=\(modelDim)")
    }

    /// P7-A: the streaming/online-softmax MPS backend must be numerically
    /// equivalent to the strided token-major MPS reference (which P4 proved
    /// equals legacy head-major) on the same token-major input. Uses an
    /// aligned headDim=128 shape so the strided MPS head offsets are valid.
    func testStreamingMPSMatchesStridedReference() async throws {
        let context = try requireContext()
        let heads = 2, headDim = 128, modelDim = heads * headDim
        let queryCount = 5, keyCount = 7
        let stride = modelDim
        var q = [Float16](repeating: 0, count: queryCount * stride)
        var k = [Float16](repeating: 0, count: keyCount * stride)
        var v = [Float16](repeating: 0, count: keyCount * stride)
        for token in 0..<queryCount {
            for col in 0..<stride {
                q[token * stride + col] = Float16(sin(Float(token * 31 + col)) * 0.3)
            }
        }
        for token in 0..<keyCount {
            for col in 0..<stride {
                k[token * stride + col] = Float16(cos(Float(token * 17 + col)) * 0.3)
                v[token * stride + col] = Float16(Float(token % 5) / 5)
            }
        }
        let qBuf = makeHalfBuffer(q, context: context)
        let kBuf = makeHalfBuffer(k, context: context)
        let vBuf = makeHalfBuffer(v, context: context)
        let refOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * stride * 2, options: .storageModeShared))
        let streamOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * stride * 2, options: .storageModeShared))
        // Strided reference (P4-proven == legacy).
        try await AttentionExecutor(context: context).execute(
            query: qBuf, key: kBuf, value: vBuf, output: refOut,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
            layout: .tokenMajor(tokenStride: stride))
        // Streaming online-softmax MPS (P7-A).
        try await AttentionExecutor(context: context, attentionBackend: .streamingMPS).execute(
            query: qBuf, key: kBuf, value: vBuf, output: streamOut,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
            layout: .tokenMajor(tokenStride: stride))
        let ref = readHalf(refOut, count: queryCount * stride)
        let stream = readHalf(streamOut, count: queryCount * stride)
        for index in ref.indices {
            XCTAssertEqual(stream[index], ref[index], accuracy: 0.01,
                           "streaming MPS diverges at \(index)")
        }
        print("ATTENTION_STREAMING_MPS=PASS heads=\(heads) Q=\(queryCount) K=\(keyCount)")
    }

    /// P7-B: the DiT-specialized pure-Metal Flash backend must be numerically
    /// equivalent to the strided reference on the DiT shape (heads=16,
    /// headDim=128, token-major 2048) with small token counts for speed.
    func testMetalFlashMatchesStridedReference() async throws {
        let context = try requireContext()
        let heads = 16, headDim = 128, modelDim = heads * headDim
        let queryCount = 4, keyCount = 8
        let stride = modelDim
        var q = [Float16](repeating: 0, count: queryCount * stride)
        var k = [Float16](repeating: 0, count: keyCount * stride)
        var v = [Float16](repeating: 0, count: keyCount * stride)
        for token in 0..<queryCount {
            for col in 0..<stride {
                q[token * stride + col] = Float16(sin(Float(token * 13 + col) * 0.02))
            }
        }
        for token in 0..<keyCount {
            for col in 0..<stride {
                k[token * stride + col] = Float16(cos(Float(token * 7 + col) * 0.02))
                v[token * stride + col] = Float16(Float(token % 7) / 7)
            }
        }
        let qBuf = makeHalfBuffer(q, context: context)
        let kBuf = makeHalfBuffer(k, context: context)
        let vBuf = makeHalfBuffer(v, context: context)
        let refOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * stride * 2, options: .storageModeShared))
        let flashOut = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * stride * 2, options: .storageModeShared))
        try await AttentionExecutor(context: context).execute(
            query: qBuf, key: kBuf, value: vBuf, output: refOut,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
            layout: .tokenMajor(tokenStride: stride))
        try await AttentionExecutor(context: context, attentionBackend: .metalFlash).execute(
            query: qBuf, key: kBuf, value: vBuf, output: flashOut,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
            layout: .tokenMajor(tokenStride: stride))
        let ref = readHalf(refOut, count: queryCount * stride)
        let flash = readHalf(flashOut, count: queryCount * stride)
        for index in ref.indices {
            XCTAssertEqual(flash[index], ref[index], accuracy: 0.01,
                           "metal Flash diverges at \(index)")
        }
        print("ATTENTION_METAL_FLASH=PASS heads=\(heads) Q=\(queryCount) K=\(keyCount)")
    }

    /// P7-B: the DiT-specialized Flash backend must reject a non-DiT shape
    /// (e.g. headDim != 128) loudly, never silently corrupt.
    func testMetalFlashRejectsNonDiTHeadDim() async throws {
        let context = try requireContext()
        let heads = 4, headDim = 32, modelDim = heads * headDim
        let queryCount = 4, keyCount = 4
        let qBuf = makeHalfBuffer([Float16](repeating: 0, count: queryCount * modelDim), context: context)
        let kBuf = makeHalfBuffer([Float16](repeating: 0, count: keyCount * modelDim), context: context)
        let vBuf = makeHalfBuffer([Float16](repeating: 0, count: keyCount * modelDim), context: context)
        let out = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * modelDim * 2, options: .storageModeShared))
        do {
            try await AttentionExecutor(context: context, attentionBackend: .metalFlash).execute(
                query: qBuf, key: kBuf, value: vBuf, output: out,
                heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
                layout: .tokenMajor(tokenStride: modelDim))
            XCTFail("metalFlash must reject a non-DiT headDim")
        } catch {
            // Expected: the backend refuses the unsupported shape.
        }
        print("ATTENTION_METAL_FLASH_REJECT_NON_DIT=PASS")
    }

    /// P4: full DiT shape descriptors — self 1024/1024 and cross 1024/512 —
    /// with tokenStride 2048. Executes real attention at full shape.
    func testStridedTokenMajorFullDiTShapes() async throws {
        let context = try requireContext()
        let heads = 16, headDim = 128, modelDim = 2048
        func tokenMajorValues(rows: Int) -> [Float16] {
            (0..<(rows * modelDim)).map { index in
                Float16(Float((index * 7 + 3) % 1009) / 2000 - 0.25)
            }
        }
        let query = tokenMajorValues(rows: 1024)
        let selfKeyValue = tokenMajorValues(rows: 1024)
        let crossKeyValue = tokenMajorValues(rows: 512)
        let qBuffer = makeHalfBuffer(query, context: context)
        let selfKV = makeHalfBuffer(selfKeyValue, context: context)
        let crossKV = makeHalfBuffer(crossKeyValue, context: context)
        let layout = AttentionInputLayout.tokenMajor(tokenStride: modelDim)

        for (name, keyCount, kv) in [("self", 1024, selfKV), ("cross", 512, crossKV)] {
            let output = try XCTUnwrap(context.device.makeBuffer(
                length: 1024 * modelDim * 2, options: .storageModeShared))
            try await AttentionExecutor(context: context).execute(
                query: qBuffer, key: kv, value: kv, output: output,
                heads: heads, queryCount: 1024, keyCount: keyCount, headDim: headDim,
                layout: layout)
            let actual = readHalf(output, count: 1024 * modelDim)
            XCTAssertTrue(actual.allSatisfy(\.isFinite), "\(name) attention produced non-finite values")
            XCTAssertGreaterThan(actual.reduce(0) { $0 + abs($1) }, 0, "\(name) attention output must not be all zero")
            print("ATTENTION_STRIDED_\(name.uppercased())=PASS Q=1024 K=\(keyCount) modelDim=2048")
        }
    }

    /// P4: no head mixing — each head reads exactly its headDim contiguous
    /// columns of every token row. Uniform K/V forces output = uniform V, so
    /// this only proves plumbing; the parity test above proves numerics.
    func testStridedTokenMajorNoHeadMixing() async throws {
        let context = try requireContext()
        let heads = 2, headDim = 4, modelDim = heads * headDim
        let rows = 3
        // Token-major values where every head of every token differs.
        var values = [Float16](repeating: 0, count: rows * modelDim)
        for token in 0..<rows {
            for head in 0..<heads {
                for column in 0..<headDim {
                    values[token * modelDim + head * headDim + column] =
                        Float16(Float(token * 10 + head) + Float(column) / 10)
                }
            }
        }
        // Q = all zeros (uniform scores), V = token-major distinct values.
        let zeros = [Float16](repeating: 0, count: rows * modelDim)
        let qBuffer = makeHalfBuffer(zeros, context: context)
        let vBuffer = makeHalfBuffer(values, context: context)
        // K all zeros: with Q=0 every score is 0, softmax uniform over key rows.
        let kBuffer = makeHalfBuffer(zeros, context: context)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: rows * modelDim * 2, options: .storageModeShared))
        try await AttentionExecutor(context: context).execute(
            query: qBuffer, key: kBuffer, value: vBuffer, output: output,
            heads: heads, queryCount: rows, keyCount: rows, headDim: headDim,
            layout: .tokenMajor(tokenStride: modelDim))
        let actual = readHalf(output, count: rows * modelDim)
        // Output = mean over key rows of V per head; head h output lives in
        // columns [h*headDim, (h+1)*headDim) of each token row.
        for token in 0..<rows {
            for head in 0..<heads {
                for column in 0..<headDim {
                    var expected: Float = 0
                    for keyRow in 0..<rows {
                        expected += Float(values[keyRow * modelDim + head * headDim + column])
                    }
                    expected /= Float(rows)
                    XCTAssertEqual(actual[token * modelDim + head * headDim + column],
                                   expected, accuracy: 0.05,
                                   "head \(head) column \(column) of token \(token) mixed")
                }
            }
        }
        print("ATTENTION_STRIDED_NO_HEAD_MIXING=PASS heads=\(heads) headDim=\(headDim)")
    }

    /// P4 metrics: the strided token-major backend must not record any
    /// transpose bytes, and must still count query tiles (2 heads × tiles).
    func testStridedTokenMajorRecordsZeroTransposeBytes() async throws {
        let context = try requireContext()
        let heads = 2, headDim = 8, modelDim = heads * headDim
        let queryCount = 130, keyCount = 64
        let zeros = [Float16](repeating: 0, count: queryCount * modelDim)
        let zeroKey = [Float16](repeating: 0, count: keyCount * modelDim)
        let value = [Float16](repeating: 1, count: keyCount * modelDim)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: queryCount * modelDim * 2, options: .storageModeShared))
        let collector = MetricsCollector()
        let executor = AttentionExecutor(context: context, tileRows: 128)
        executor.metrics = collector
        try await executor.execute(
            query: makeHalfBuffer(zeros, context: context),
            key: makeHalfBuffer(zeroKey, context: context),
            value: makeHalfBuffer(value, context: context), output: output,
            heads: heads, queryCount: queryCount, keyCount: keyCount, headDim: headDim,
            layout: .tokenMajor(tokenStride: modelDim))
        let snapshot = collector.snapshot()
        XCTAssertEqual(snapshot.transposeBytes, 0,
                       "strided token-major backend must not record transpose traffic")
        XCTAssertEqual(snapshot.attentionQueryTiles, 4,
                       "2 heads × ceil(130/128) = 4 tiles")
        print("ATTENTION_STRIDED_TRANSPOSE_BYTES=PASS transposeBytes=\(snapshot.transposeBytes)")
    }

    /// P4-F: the strided backend must FAIL loudly on unsupported numerics
    /// combinations instead of silently corrupting the layout.
    func testStridedTokenMajorRejectsUnsupportedCombinations() async throws {
        let context = try requireContext()
        let heads = 2, headDim = 4, modelDim = heads * headDim
        let buffer = try XCTUnwrap(context.device.makeBuffer(
            length: 4 * modelDim * 2, options: .storageModeShared))
        // fp32ScoresAndSoftmax is not supported on the strided path.
        do {
            try await AttentionExecutor(context: context, numerics: .fp32ScoresAndSoftmax).execute(
                query: buffer, key: buffer, value: buffer, output: buffer,
                heads: heads, queryCount: 4, keyCount: 4, headDim: headDim,
                layout: .tokenMajor(tokenStride: modelDim))
            XCTFail("fp32ScoresAndSoftmax on token-major layout must throw")
        } catch let AnimapkError.validation(message) {
            XCTAssertTrue(message.contains("token-major"), "unexpected message: \(message)")
        }
        // GQA is not expressible in token-major without a strided K/V gather.
        do {
            try await AttentionExecutor(context: context).execute(
                query: buffer, key: buffer, value: buffer, output: buffer,
                heads: heads, queryCount: 4, keyCount: 4, headDim: headDim,
                keyValueHeads: 1, layout: .tokenMajor(tokenStride: modelDim))
            XCTFail("GQA with token-major layout must throw")
        } catch let AnimapkError.validation(message) {
            XCTAssertTrue(message.contains("keyValueHeads == heads"), "unexpected message: \(message)")
        }
        print("ATTENTION_STRIDED_REJECTS_UNSUPPORTED=PASS")
    }

    /// P4-B helper: transpose a token-major [rows, heads*headDim] buffer into
    /// head-major [heads, rows, headDim] using the production kernel.
    private func encodeHeadMajor(
        context: MetalContext, input: MTLBuffer, output: MTLBuffer,
        rows: Int, heads: Int, headDim: Int, direction: UInt32 = 1
    ) throws {
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let pipeline = try context.pipeline(named: "transpose_token_head_half")
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var t = UInt32(rows), h = UInt32(heads), d = UInt32(headDim)
        var dir = direction
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&t, length: 4, index: 2)
        encoder.setBytes(&h, length: 4, index: 3)
        encoder.setBytes(&d, length: 4, index: 4)
        encoder.setBytes(&dir, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: rows * heads * headDim, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
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
