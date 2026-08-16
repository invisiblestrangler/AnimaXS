import XCTest
import Metal
@testable import AnimaXS

final class DiTBlockExecutorTests: XCTestCase {
    private func requireContext() throws -> MetalContext {
        guard let context = MetalContext() else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        return context
    }

    private func buffer<T>(_ values: [T], context: MetalContext) -> MTLBuffer {
        values.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                      options: .storageModeShared)!
        }
    }

    func testHalfBoundaryConversionsAndHeadLayoutRoundTrip() throws {
        let context = try requireContext()
        let tokens = 3, heads = 2, headDim = 4
        let values = (0..<(tokens * heads * headDim)).map { Float($0) / 7 - 1 }
        let source = buffer(values, context: context)
        let tokenHalf = try XCTUnwrap(context.device.makeBuffer(length: values.count * 2))
        let headHalf = try XCTUnwrap(context.device.makeBuffer(length: values.count * 2))
        let roundTripHalf = try XCTUnwrap(context.device.makeBuffer(length: values.count * 2))
        let output = try XCTUnwrap(context.device.makeBuffer(length: values.count * 4))
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())

        try encodeUnary(context, command, "float_to_half", source, tokenHalf, values.count)
        try encodeTranspose(context, command, tokenHalf, headHalf,
                            tokens: tokens, heads: heads, headDim: headDim, toHeadMajor: true)
        try encodeTranspose(context, command, headHalf, roundTripHalf,
                            tokens: tokens, heads: heads, headDim: headDim, toHeadMajor: false)
        try encodeUnary(context, command, "half_to_float", roundTripHalf, output, values.count)
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)

        let head = headHalf.contents().bindMemory(to: Float16.self, capacity: values.count)
        for h in 0..<heads {
            for t in 0..<tokens {
                for d in 0..<headDim {
                    let expected = Float16(values[(t * heads + h) * headDim + d])
                    XCTAssertEqual(head[(h * tokens + t) * headDim + d].bitPattern,
                                   expected.bitPattern)
                }
            }
        }
        let result = output.contents().bindMemory(to: Float.self, capacity: values.count)
        for index in values.indices {
            XCTAssertEqual(result[index], Float(Float16(values[index])), accuracy: 0)
        }
    }

    func testCrossAttentionHeadRMSNormUsesSharedWeight() throws {
        let context = try requireContext()
        let rows = 3, dim = 128
        let inputValues = (0..<(rows * dim)).map { Float16(Float(($0 * 17) % 29 - 14) / 8) }
        let weightValues = (0..<dim).map { Float16(Float(($0 % 9) + 1) / 8) }
        let input = buffer(inputValues.map(\.bitPattern), context: context)
        let weight = buffer(weightValues.map(\.bitPattern), context: context)
        let output = try XCTUnwrap(context.device.makeBuffer(length: rows * dim * 2))
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let pipeline = try context.pipeline(named: "rmsnorm_heads_half")
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var rowCount = UInt32(rows), epsilon: Float = 1e-6
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let actual = output.contents().bindMemory(to: Float16.self, capacity: rows * dim)
        for row in 0..<rows {
            var squareSum: Float = 0
            for d in 0..<dim {
                let value = Float(inputValues[row * dim + d])
                squareSum += value * value
            }
            let inverse = 1 / sqrt(squareSum / Float(dim) + epsilon)
            for d in 0..<dim {
                let expected = Float16(Float(inputValues[row * dim + d]) * inverse * Float(weightValues[d]))
                XCTAssertEqual(actual[row * dim + d].bitPattern, expected.bitPattern)
            }
        }
    }

    /// Definitive E009 gate. It is manual because neither the 1.18 GB pack nor the
    /// 347 MB diagnostic directory belongs in git. Both paths are available on the
    /// project workstation and may also be injected into a dedicated macOS runner.
    func testRealPackBlock0AgainstSameW4Oracle() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundledPack = bundledFixture(named: "anima-turbo-v1.0-xsmax-w4.animapk")
        let bundledOracle = bundledFixture(named: "block0_input_x.f32")
        guard let packs = environment["ANIMAXS_PACKS_DIR"]
                ?? bundledPack?.deletingLastPathComponent().path,
              let oracle = environment["ANIMAXS_BLOCK0_ORACLE_DIR"]
                ?? bundledOracle?.deletingLastPathComponent().path else {
            throw XCTSkip("ANIMAXS_PACKS_DIR/ANIMAXS_BLOCK0_ORACLE_DIR not set")
        }
        let context = try requireContext()
        let file = try AnimapkFile(url: URL(fileURLWithPath: packs)
            .appendingPathComponent("anima-turbo-v1.0-xsmax-w4.animapk"))
        func floats(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: URL(fileURLWithPath: oracle).appendingPathComponent(name))
            guard data.count.isMultiple(of: 4) else { throw AnimapkError.validation("invalid oracle file") }
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let inputValues = try floats("block0_input_x.f32")
        let contextHalf = try floats("block0_cross_ctx.f32").map(Float16.init)
        let expected = try floats("block0_swift_output.f32")
        let residual = buffer(inputValues, context: context)
        let emb = buffer(try floats("block0_emb.f32"), context: context)
        let adaln = buffer(try floats("block0_adaln_lora.f32"), context: context)
        let cross = buffer(contextHalf.map(\.bitPattern), context: context)
        let rope = buffer(try floats("block0_rope.f32"), context: context)
        try await DiTBlockExecutor(context: context, file: file).execute(
            blockIndex: 0, residual: residual, emb: emb, adalnLora: adaln,
            crossContext: cross, rope: rope)
        let pointer = residual.contents().bindMemory(to: Float.self, capacity: expected.count)
        var squareError = 0.0, dot = 0.0, actualNorm = 0.0, expectedNorm = 0.0
        var maxAbsolute = 0.0
        for i in expected.indices {
            let actual = Double(pointer[i]), reference = Double(expected[i])
            XCTAssertTrue(actual.isFinite)
            let error = abs(actual - reference)
            maxAbsolute = max(maxAbsolute, error)
            squareError += error * error
            dot += actual * reference
            actualNorm += actual * actual
            expectedNorm += reference * reference
        }
        let rmse = sqrt(squareError / Double(expected.count))
        let cosine = dot / sqrt(actualNorm * expectedNorm)
        print("E009_BLOCK0 maxAbs=\(maxAbsolute) rmse=\(rmse) cosine=\(cosine)")
        XCTAssertGreaterThanOrEqual(cosine, 0.999)
        XCTAssertLessThan(rmse, 0.1)
    }

    func testRealPackAllBlocksAreFinite() async throws {
        let bundledPack = bundledFixture(named: "anima-turbo-v1.0-xsmax-w4.animapk")
        let bundledSize = try bundledPack?.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard ProcessInfo.processInfo.environment["ANIMAXS_RUN_ALL_DIT_BLOCKS"] == "1"
                || bundledSize > 1_000_000_000 else {
            throw XCTSkip("ANIMAXS_RUN_ALL_DIT_BLOCKS not set")
        }
        let (packs, oracle) = try realFixtureDirectories()
        let context = try requireContext()
        let file = try AnimapkFile(url: packs.appendingPathComponent(
            "anima-turbo-v1.0-xsmax-w4.animapk"))
        let inputValues = try fixtureFloats("block0_input_x.f32", in: oracle)
        let residual = buffer(inputValues, context: context)
        let emb = buffer(try fixtureFloats("block0_emb.f32", in: oracle), context: context)
        let adaln = buffer(try fixtureFloats("block0_adaln_lora.f32", in: oracle), context: context)
        let contextHalf = try fixtureFloats("block0_cross_ctx.f32", in: oracle).map(Float16.init)
        let cross = buffer(contextHalf.map(\.bitPattern), context: context)
        let rope = buffer(try fixtureFloats("block0_rope.f32", in: oracle), context: context)
        var completed: [Int] = []
        try await DitForward(context: context, file: file).execute(
            residual: residual, emb: emb, adalnLora: adaln,
            crossContext: cross, rope: rope
        ) { logicalIndex, current in
            completed.append(logicalIndex)
            let values = current.contents().bindMemory(to: Float.self, capacity: inputValues.count)
            XCTAssertTrue((0..<inputValues.count).allSatisfy { values[$0].isFinite },
                          "non-finite residual after block \(logicalIndex)")
        }
        XCTAssertEqual(completed, Array(0..<28))
        let values = residual.contents().bindMemory(to: Float.self, capacity: inputValues.count)
        var minimum = Float.infinity, maximum = -Float.infinity
        for i in 0..<inputValues.count { minimum = min(minimum, values[i]); maximum = max(maximum, values[i]) }
        print("H006_ALL_BLOCKS=PASS min=\(minimum) max=\(maximum)")
    }

    private func realFixtureDirectories() throws -> (packs: URL, oracle: URL) {
        let environment = ProcessInfo.processInfo.environment
        let bundledPack = bundledFixture(named: "anima-turbo-v1.0-xsmax-w4.animapk")
        let bundledOracle = bundledFixture(named: "block0_input_x.f32")
        guard let packs = environment["ANIMAXS_PACKS_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledPack?.deletingLastPathComponent(),
              let oracle = environment["ANIMAXS_BLOCK0_ORACLE_DIR"]
                .map(URL.init(fileURLWithPath:)) ?? bundledOracle?.deletingLastPathComponent() else {
            throw XCTSkip("real DiT pack/oracle fixture not available")
        }
        return (packs, oracle)
    }

    // MARK: - P5 cross-attention K/V cache

    /// P5: a fresh cache starts empty (every block not ready) and per-block
    /// readiness is tracked independently.
    func testCrossKVCacheStartsEmptyAndTracksReadyPerBlock() throws {
        let context = try requireContext()
        guard let cache = CrossKVCache(device: context.device,
                                       options: .storageModeShared) else {
            throw XCTSkip("CrossKVCache could not be allocated")
        }
        for block in 0..<CrossKVCache.blockCount {
            XCTAssertFalse(cache.isReady(block), "block \(block) must start not-ready")
        }
        cache.markReady(3)
        cache.markReady(17)
        XCTAssertTrue(cache.isReady(3))
        XCTAssertTrue(cache.isReady(17))
        XCTAssertFalse(cache.isReady(0))
        XCTAssertFalse(cache.isReady(28 - 1))
    }

    /// P5: cache offsets/size are one contiguous buffer, K and V per block,
    /// sized for the DiT cross shape (512 × 2048 half = 2 MiB each).
    func testCrossKVCacheOffsetsAndSize() throws {
        let context = try requireContext()
        guard let cache = CrossKVCache(device: context.device,
                                       options: .storageModeShared) else {
            throw XCTSkip("CrossKVCache could not be allocated")
        }
        let tensorBytes = 512 * 2048 * MemoryLayout<Float16>.stride
        XCTAssertEqual(CrossKVCache.tensorBytes, tensorBytes)
        XCTAssertEqual(CrossKVCache.blockStride, tensorBytes * 2)
        XCTAssertEqual(CrossKVCache.blockCount, 28)
        // Block b's K lives at b*blockStride; V immediately after K.
        for block in 0..<CrossKVCache.blockCount {
            XCTAssertEqual(cache.kOffset(block: block), block * CrossKVCache.blockStride)
            XCTAssertEqual(cache.vOffset(block: block),
                           block * CrossKVCache.blockStride + CrossKVCache.tensorBytes)
        }
        // One contiguous buffer sized for all blocks (no 56 independent buffers).
        XCTAssertEqual(cache.buffer.length, 28 * CrossKVCache.blockStride)
    }

    /// P5: the production cache is `.storageModePrivate` (CPU never reads it).
    func testCrossKVCacheStorageModePrivateByDefault() throws {
        let context = try requireContext()
        guard let cache = CrossKVCache(device: context.device) else {
            throw XCTSkip("CrossKVCache could not be allocated")
        }
        XCTAssertEqual(cache.buffer.storageMode, .private)
    }

    /// P5: a cache write then read back reproduces the exact bytes (exact
    /// blit copy, no approximation). Uses the K region for one block.
    func testCrossKVCacheWriteReadBackIsExact() throws {
        let context = try requireContext()
        guard let cache = CrossKVCache(device: context.device,
                                       options: .storageModeShared) else {
            throw XCTSkip("CrossKVCache could not be allocated")
        }
        let count = 512 * 2048
        let src = try XCTUnwrap(context.device.makeBuffer(
            length: count * MemoryLayout<Float16>.stride, options: .storageModeShared))
        let pattern = (0..<count).map { Float16(Float($0 % 1000) / 100) }
        src.contents().bindMemory(to: Float16.self, capacity: count).update(from: pattern, count: count)
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeBlitCommandEncoder())
        encoder.copy(from: src, sourceOffset: 0, to: cache.buffer,
                     destinationOffset: cache.kOffset(block: 5), size: count * 2)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let read = cache.buffer.contents().advanced(by: cache.kOffset(block: 5))
            .bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count where i % 997 == 0 {
            XCTAssertEqual(read[i].bitPattern, pattern[i].bitPattern, "index \(i)")
        }
        cache.markReady(5)
        XCTAssertTrue(cache.isReady(5))
    }

    // MARK: - P3-A fused LayerNorm+AdaLN+to-half ABI (E-fused)

    /// Locks the fused modulation offset UNIT. The fused kernels read scale at
    /// `modulation + modulationOffset` — pointer arithmetic on a float*, i.e.
    /// float ELEMENTS, not bytes. For dim = 2048 the correct offset is 2048,
    /// and the old byte-unit bug (dim*4 = 8192) would walk 2048 floats past
    /// the end of the real 6144-float modulation buffer (3 × 2048 chunks).
    func testFusedModulationElementOffsetIsFloatElements() {
        XCTAssertEqual(DiTBlockExecutor.fusedModulationElementOffset(columns: 2048), 2048)
        // Explicit regression guard: the byte-unit value must NEVER come back.
        XCTAssertNotEqual(DiTBlockExecutor.fusedModulationElementOffset(columns: 2048), 2048 * 4)
        XCTAssertEqual(DiTBlockExecutor.fusedModulationElementOffset(columns: 1), 1)
    }

    /// Synthetic parity test for `dit_layernorm_modulate_to_half`: drives the
    /// production fused ABI (via the internal kernel seam) against an EXACT
    /// 6144-float modulation buffer (shift [0..<2048], scale [2048..<4096],
    /// gate/third chunk [4096..<6144]) — no slack, so a bogus element offset
    /// of 8192 could not hide behind an oversized allocation. The GPU output
    /// must match a CPU LayerNorm→(1+scale)→+shift→fp16 reference, proving
    /// scale was read from element 2048 (the fixed semantics).
    func testFusedLayerNormModulateToHalfParityAgainstCPU() throws {
        let context = try requireContext()
        let rows = 3
        let columns = DiTBlockExecutor.dim // 2048
        let residual = makeFusedResidual(rows: rows, columns: columns)
        let modulation = makeFusedModulation(columns: columns)
        XCTAssertEqual(modulation.count, 6144)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: rows * columns * MemoryLayout<Float16>.stride))
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())

        try DiTBlockExecutor.encodeFusedNormModulateKernel(
            context: context, command: command,
            residual: buffer(residual, context: context),
            modulation: buffer(modulation, context: context),
            output: output, rows: rows, columns: columns,
            modulationElementOffset: DiTBlockExecutor.fusedModulationElementOffset(columns: columns),
            emulatesBF16: false, monitor: nil, probe: .selfProjectionInput)
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)

        let expected = fusedReference(residual: residual, modulation: modulation,
                                      rows: rows, columns: columns)
        let actual = output.contents().bindMemory(to: Float16.self, capacity: rows * columns)
        for index in 0..<(rows * columns) {
            XCTAssertTrue(actual[index].isFinite, "non-finite output at \(index)")
            // Both sides compute in fp32 and round once to fp16; allow a
            // couple of fp16 ulps for rsqrt/summation-order differences.
            XCTAssertEqual(Float(actual[index]), expected[index], accuracy: 5e-3,
                           "index \(index)")
        }
    }

    /// Same parity check for the probe variant
    /// (`dit_layernorm_modulate_to_half_probe`), which reads scale from the
    /// SAME element offset and additionally records health stats. Clean
    /// stats (no NaN/Inf/overflow flags) confirm the probe saw the correct
    /// scale values rather than garbage from an out-of-bounds read.
    func testFusedLayerNormModulateToHalfProbeParityAgainstCPU() throws {
        let context = try requireContext()
        let rows = 2
        let columns = DiTBlockExecutor.dim
        let residual = makeFusedResidual(rows: rows, columns: columns)
        let modulation = makeFusedModulation(columns: columns)
        let monitor = try NumericalMonitor(context: context)
        let output = try XCTUnwrap(context.device.makeBuffer(
            length: rows * columns * MemoryLayout<Float16>.stride))
        let command = try XCTUnwrap(context.commandQueue.makeCommandBuffer())

        try DiTBlockExecutor.encodeFusedNormModulateKernel(
            context: context, command: command,
            residual: buffer(residual, context: context),
            modulation: buffer(modulation, context: context),
            output: output, rows: rows, columns: columns,
            modulationElementOffset: DiTBlockExecutor.fusedModulationElementOffset(columns: columns),
            emulatesBF16: false, monitor: monitor, probe: .selfProjectionInput)
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)

        let expected = fusedReference(residual: residual, modulation: modulation,
                                      rows: rows, columns: columns)
        let actual = output.contents().bindMemory(to: Float16.self, capacity: rows * columns)
        for index in 0..<(rows * columns) {
            XCTAssertTrue(actual[index].isFinite, "non-finite output at \(index)")
            XCTAssertEqual(Float(actual[index]), expected[index], accuracy: 5e-3,
                           "index \(index)")
        }
        XCTAssertEqual(monitor.warningCount(), 0)
        XCTAssertNil(monitor.earliestIssue)
    }

    /// Synthetic residual with real per-row variance (alternating sign,
    /// varying magnitude, row offset) so LayerNorm is well-defined.
    private func makeFusedResidual(rows: Int, columns: Int) -> [Float] {
        (0..<(rows * columns)).map { index in
            let i = index % columns
            let row = index / columns
            return Float(((i * 37) % 101) - 50) / 13 + Float(row)
        }
    }

    /// EXACT 3 × `columns` modulation layout: shift, then scale, then the
    /// gate/third chunk filled with arbitrary values. 6144 floats for
    /// columns = 2048 — nothing more, matching production.
    private func makeFusedModulation(columns: Int) -> [Float] {
        var values = [Float](repeating: 0, count: columns * 3)
        for i in 0..<columns {
            values[i] = Float((i % 13) - 6) / 10            // shift chunk
            values[columns + i] = Float((i % 11) - 5) / 8   // scale chunk (at element `columns`)
            values[2 * columns + i] = Float((i % 7) - 3)    // gate/third chunk (unused here)
        }
        return values
    }

    /// CPU reference for the fused kernels: per-row LayerNorm (biased
    /// variance, the same eps the shader receives), then
    /// normalized * (1 + scale) + shift with scale at element `columns`,
    /// rounded once through Float16 (matching the GPU's half() store).
    private func fusedReference(residual: [Float], modulation: [Float],
                                rows: Int, columns: Int) -> [Float] {
        let epsilon = DiTBlockExecutor.eps
        var expected = [Float](repeating: 0, count: residual.count)
        for row in 0..<rows {
            var sum: Float = 0
            for i in 0..<columns { sum += residual[row * columns + i] }
            let mean = sum / Float(columns)
            var squareSum: Float = 0
            for i in 0..<columns {
                let centered = residual[row * columns + i] - mean
                squareSum += centered * centered
            }
            let inverse = 1 / sqrt(squareSum / Float(columns) + epsilon)
            for i in 0..<columns {
                let normalized = (residual[row * columns + i] - mean) * inverse
                let scale = modulation[columns + i]
                let shift = modulation[i]
                expected[row * columns + i] = Float(Float16(normalized * (1 + scale) + shift))
            }
        }
        return expected
    }

    private func fixtureFloats(_ name: String, in directory: URL) throws -> [Float] {
        let data = try Data(contentsOf: directory.appendingPathComponent(name))
        guard data.count.isMultiple(of: 4) else { throw AnimapkError.validation("invalid oracle file") }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func bundledFixture(named name: String) -> URL? {
        guard let root = Bundle(for: Self.self).resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: nil) else { return nil }
        for case let url as URL in enumerator where url.lastPathComponent == name { return url }
        return nil
    }

    private func encodeUnary(
        _ context: MetalContext, _ command: MTLCommandBuffer, _ name: String,
        _ input: MTLBuffer, _ output: MTLBuffer, _ count: Int
    ) throws {
        let pipeline = try context.pipeline(named: name)
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var n = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(32, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeTranspose(
        _ context: MetalContext, _ command: MTLCommandBuffer,
        _ input: MTLBuffer, _ output: MTLBuffer,
        tokens: Int, heads: Int, headDim: Int, toHeadMajor: Bool
    ) throws {
        let pipeline = try context.pipeline(named: "transpose_token_head_half")
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        var t = UInt32(tokens), h = UInt32(heads), d = UInt32(headDim)
        var direction: UInt32 = toHeadMajor ? 1 : 0
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&t, length: 4, index: 2)
        encoder.setBytes(&h, length: 4, index: 3)
        encoder.setBytes(&d, length: 4, index: 4)
        encoder.setBytes(&direction, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: tokens * heads * headDim, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
