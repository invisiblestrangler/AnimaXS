import Foundation
import Metal

/// V8 device-only ANE native-projection correctness sweep.
///
/// V7 cleared one production-shaped W8 projection, the real P5 private cache,
/// repeated evaluation determinism, and foreground evaluation while a second
/// prepared model was loaded/unloaded through the shared private client. Its
/// standalone layout fixture was invalid: 37x64 produced an 8 KiB IOSurface on
/// a device whose VM page is 16 KiB, so the surface was rejected before either
/// layout kernel ran.
///
/// V8 re-tests the layout bridge with page-valid shapes, then checks every one
/// of the 280 ANE-native projection tensors across all 28 blocks. Each prepared
/// program is loaded independently; fused self-QKV is checked as Q/K/V against
/// separate CPU row references. Projection I/O is written/read directly in the
/// IOSurface plane-major layout, deliberately bypassing both layout kernels.
/// Production inference and scheduler behavior are unchanged.
enum ANERingProbe {
    private static let relativeRMSELimit: Float = 0.06
    private static let targetSampleRows = 128

    private struct Measurement {
        let block: Int
        let name: String
        let relativeRMSE: Float
        let maxAbs: Float
        let spatialBitMismatches: Int
        let finite: Bool
        let evaluationMS: Double
        let errorText: String?

        var passed: Bool {
            errorText == nil
                && finite
                && relativeRMSE < ANERingProbe.relativeRMSELimit
                && spatialBitMismatches == 0
        }

        var compact: String {
            if let errorText { return "\(name) ERROR=\(errorText)" }
            return "\(name) rel=\(five(relativeRMSE)) maxAbs=\(six(maxAbs)) "
                + "spatialBits=\(spatialBitMismatches) eval=\(two(evaluationMS))ms"
        }
    }

    private struct LayoutResult {
        let name: String
        let rawMaxAbs: Float
        let roundTripMaxAbs: Float
        let errorText: String?

        var passed: Bool {
            errorText == nil && rawMaxAbs == 0 && roundTripMaxAbs == 0
        }

        var line: String {
            if let errorText { return "FAIL layout-\(name): \(errorText)" }
            let status = passed ? "PASS" : "FAIL"
            return "\(status) layout-\(name): rawMaxAbs=\(eight(rawMaxAbs)) "
                + "roundTripMaxAbs=\(eight(roundTripMaxAbs))"
        }
    }

    /// Diagnostic-local mirror of DiTBlockExecutor's exact production bridge.
    private final class ProbeKernels {
        private let context: MetalContext

        init(context: MetalContext) { self.context = context }

        func tokenToANE(
            _ command: MTLCommandBuffer,
            source: MTLBuffer,
            destination: MTLBuffer,
            rows: Int,
            channels: Int,
            planeStrideElements: UInt
        ) throws {
            try bridge(
                command, kernel: "dit_token_to_ane_f16",
                source: source, destination: destination,
                rows: rows, channels: channels,
                planeStrideElements: planeStrideElements)
        }

        func aneToToken(
            _ command: MTLCommandBuffer,
            source: MTLBuffer,
            destination: MTLBuffer,
            rows: Int,
            channels: Int,
            planeStrideElements: UInt
        ) throws {
            try bridge(
                command, kernel: "dit_ane_to_token_f16",
                source: source, destination: destination,
                rows: rows, channels: channels,
                planeStrideElements: planeStrideElements)
        }

        private func bridge(
            _ command: MTLCommandBuffer,
            kernel: String,
            source: MTLBuffer,
            destination: MTLBuffer,
            rows: Int,
            channels: Int,
            planeStrideElements: UInt
        ) throws {
            let pipeline = try context.pipeline(named: kernel)
            guard let encoder = command.makeComputeCommandEncoder() else {
                throw AnimapkError.validation("V8 failed to create \(kernel) encoder")
            }
            var rows32 = UInt32(rows)
            var channels32 = UInt32(channels)
            var stride32 = UInt32(planeStrideElements)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(source, offset: 0, index: 0)
            encoder.setBuffer(destination, offset: 0, index: 1)
            encoder.setBytes(&rows32, length: 4, index: 2)
            encoder.setBytes(&channels32, length: 4, index: 3)
            encoder.setBytes(&stride32, length: 4, index: 4)
            let tw = max(1, min(pipeline.threadExecutionWidth, channels))
            let th = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / tw, 8))
            encoder.dispatchThreads(
                MTLSize(width: channels, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
            encoder.endEncoding()
        }
    }

    private final class ModelFactory {
        private let file: AnimapkFile

        init(file: AnimapkFile) { self.file = file }

        func projection(
            block: Int, suffix: String, label: String
        ) throws -> A12ANEProjectionModel {
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                throw AnimapkError.validation("V8 unknown projection suffix: \(suffix)")
            }
            let tensor = try ANEW8NativePack.tensor(
                file: file, block: block, suffix: suffix)
            guard let digest = tensor.blobSHA256 else {
                throw AnimapkError.validation("V8 native tensor hash missing: \(suffix)")
            }
            let key = ANEW8NativePack.projectionCacheKey(
                block: block, tag: spec.tag, hash: digest)
            return try A12ANEProjectionModel(
                preparedInputChannels: UInt(spec.columns),
                outputChannels: UInt(spec.rows),
                spatial: UInt(spec.spatial),
                label: label,
                cacheKey: key)
        }

        func qkv(block: Int) throws -> A12ANEQKVModel {
            func digest(_ suffix: String) throws -> String {
                let tensor = try ANEW8NativePack.tensor(
                    file: file, block: block, suffix: suffix)
                guard let digest = tensor.blobSHA256 else {
                    throw AnimapkError.validation("V8 QKV hash missing: \(suffix)")
                }
                return digest
            }
            let key = ANEW8NativePack.qkvCacheKey(
                block: block,
                q: try digest("self_attn.q_proj.weight"),
                k: try digest("self_attn.k_proj.weight"),
                v: try digest("self_attn.v_proj.weight"))
            return try A12ANEQKVModel(
                preparedChannels: UInt(DiTBlockExecutor.dim),
                spatial: UInt(DiTBlockExecutor.tokens),
                label: "v8_b\(block)_self_qkv",
                cacheKey: key)
        }
    }

    private static func makeInput(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { index in
            state &*= 6_364_136_223_846_793_005
            state &+= 1_442_695_040_888_963_407
            let unit = Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF)
            let value = (unit * 2 - 1) * 0.2 + Float((index % 17) - 8) * 0.001
            return Float(Float16(value))
        }
    }

    private static func zero(_ buffer: MTLBuffer) {
        buffer.contents().initializeMemory(
            as: UInt8.self, repeating: 0, count: buffer.length)
    }

    private static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var value: Float = 0
        for index in lhs.indices {
            value = max(value, abs(lhs[index] - rhs[index]))
        }
        return value
    }

    private static func relativeRMSE(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !actual.isEmpty else { return .infinity }
        var error2: Double = 0
        var reference2: Double = 0
        for index in actual.indices {
            let delta = Double(actual[index] - expected[index])
            let reference = Double(expected[index])
            error2 += delta * delta
            reference2 += reference * reference
        }
        return Float(sqrt(error2 / max(reference2, 1e-30)))
    }

    private static func sampledRows(total: Int) -> [Int] {
        guard total > 0 else { return [] }
        if total <= targetSampleRows { return Array(0..<total) }

        var rows = Set<Int>()
        let boundary = [
            0, 1, 2, 31, 63, 64, 65, 127, 128, 129,
            total / 4, total / 2 - 1, total / 2, total / 2 + 1,
            (3 * total) / 4, total - 3, total - 2, total - 1
        ]
        for row in boundary where row >= 0 && row < total { rows.insert(row) }

        for index in 0..<64 {
            rows.insert(Int((Int64(index) * Int64(total - 1)) / 63))
        }
        var state = UInt64(total) ^ 0xA12A_4E45_5638_0001
        while rows.count < targetSampleRows {
            state &*= 6_364_136_223_846_793_005
            state &+= 1_442_695_040_888_963_407
            rows.insert(Int(state % UInt64(total)))
        }
        return rows.sorted()
    }

    /// Direct plane-major write: no token->ANE layout kernel is involved.
    private static func fillSurface(
        _ surface: A12ANESurface,
        vector: [Float],
        positions: [Int]
    ) throws {
        let channels = Int(surface.channels)
        let spatial = Int(surface.spatial)
        let stride = Int(surface.planeStrideElements)
        guard vector.count == channels,
              positions.allSatisfy({ $0 >= 0 && $0 < spatial }) else {
            throw AnimapkError.validation("V8 surface fill shape/position mismatch")
        }
        zero(surface.metalBuffer)
        let capacity = surface.metalBuffer.length / MemoryLayout<UInt16>.stride
        let pointer = surface.metalBuffer.contents().bindMemory(
            to: UInt16.self, capacity: capacity)
        for channel in 0..<channels {
            let bits = Float16(vector[channel]).bitPattern
            let base = channel * stride
            for position in positions {
                pointer[base + position] = bits
            }
        }
    }

    /// Direct plane-major read: no ANE->token layout kernel is involved.
    private static func readSamples(
        _ surface: A12ANESurface,
        rows: [Int],
        positions: [Int],
        requireSpatialEquality: Bool
    ) throws -> (values: [Float], spatialBitMismatches: Int) {
        let channels = Int(surface.channels)
        let spatial = Int(surface.spatial)
        let stride = Int(surface.planeStrideElements)
        guard rows.allSatisfy({ $0 >= 0 && $0 < channels }),
              positions.allSatisfy({ $0 >= 0 && $0 < spatial }),
              let firstPosition = positions.first else {
            throw AnimapkError.validation("V8 surface sample shape/position mismatch")
        }
        let capacity = surface.metalBuffer.length / MemoryLayout<UInt16>.stride
        let pointer = surface.metalBuffer.contents().bindMemory(
            to: UInt16.self, capacity: capacity)
        var values: [Float] = []
        values.reserveCapacity(rows.count * positions.count)
        var mismatches = 0
        for position in positions {
            for row in rows {
                let bits = pointer[row * stride + position]
                values.append(Float(Float16(bitPattern: bits)))
                if requireSpatialEquality && position != firstPosition {
                    let baseline = pointer[row * stride + firstPosition]
                    if bits != baseline { mismatches += 1 }
                }
            }
        }
        return (values, mismatches)
    }

    private static func cpuProjectionRows(
        file: AnimapkFile,
        block: Int,
        suffix: String,
        input: [Float],
        rows: [Int]
    ) throws -> [Float] {
        guard let spec = ANEW8NativePack.spec(suffix: suffix),
              input.count == spec.columns else {
            throw AnimapkError.validation("V8 CPU projection shape mismatch: \(suffix)")
        }
        let tensor = try ANEW8NativePack.tensor(
            file: file, block: block, suffix: suffix)
        guard let scaleBytes = file.scaleBytes(tensor),
              let biasBytes = file.zeroBytes(tensor) else {
            throw AnimapkError.validation("V8 native row parameters missing: \(suffix)")
        }
        let quantized = file.dataBytes(tensor).bindMemory(to: UInt8.self)
        let scales = scaleBytes.bindMemory(to: Float.self)
        let biases = biasBytes.bindMemory(to: Float.self)
        guard quantized.count == spec.rows * spec.columns,
              scales.count == spec.rows,
              biases.count == spec.rows,
              rows.allSatisfy({ $0 >= 0 && $0 < spec.rows }) else {
            throw AnimapkError.validation("V8 native row payload mismatch: \(suffix)")
        }

        var result: [Float] = []
        result.reserveCapacity(rows.count)
        for row in rows {
            let base = row * spec.columns
            let scale = scales[row]
            let bias = biases[row]
            var sum: Float = 0
            for column in 0..<spec.columns {
                let weight = Float(quantized[base + column]) * scale + bias
                sum += input[column] * weight
            }
            result.append(sum)
        }
        return result
    }

    private static func compare(
        block: Int,
        name: String,
        expectedOnePosition: [Float],
        output: A12ANESurface,
        rows: [Int],
        positions: [Int],
        evaluationMS: Double
    ) -> Measurement {
        do {
            let sampled = try readSamples(
                output, rows: rows, positions: positions,
                requireSpatialEquality: true)
            var expected: [Float] = []
            expected.reserveCapacity(expectedOnePosition.count * positions.count)
            for _ in positions { expected.append(contentsOf: expectedOnePosition) }
            return Measurement(
                block: block,
                name: name,
                relativeRMSE: relativeRMSE(sampled.values, expected),
                maxAbs: maxAbsDiff(sampled.values, expected),
                spatialBitMismatches: sampled.spatialBitMismatches,
                finite: sampled.values.allSatisfy(\.isFinite)
                    && expected.allSatisfy(\.isFinite),
                evaluationMS: evaluationMS,
                errorText: nil)
        } catch {
            return failure(block: block, name: name, error: error)
        }
    }

    private static func failure(
        block: Int, name: String, error: Error
    ) -> Measurement {
        Measurement(
            block: block,
            name: name,
            relativeRMSE: .infinity,
            maxAbs: .infinity,
            spatialBitMismatches: Int.max,
            finite: false,
            evaluationMS: 0,
            errorText: error.localizedDescription)
    }

    private static func layoutCase(
        name: String,
        context: MetalContext,
        kernels: ProbeKernels,
        rows: Int,
        channels: Int
    ) -> LayoutResult {
        do {
            let surface = try A12ANESurface(
                device: context.device,
                channels: UInt(channels), spatial: UInt(rows))
            let count = rows * channels
            guard let source = context.device.makeBuffer(
                    length: count * MemoryLayout<Float16>.stride,
                    options: .storageModeShared),
                  let roundTrip = context.device.makeBuffer(
                    length: count * MemoryLayout<Float16>.stride,
                    options: .storageModeShared),
                  let encode = context.commandQueue.makeCommandBuffer() else {
                throw AnimapkError.validation("V8 layout Metal allocation failed")
            }
            let values = makeInput(
                count: count,
                seed: (UInt64(rows) << 32) ^ UInt64(channels) ^ 0x1A90_0008)
            let sourcePointer = source.contents().bindMemory(
                to: UInt16.self, capacity: count)
            for index in values.indices {
                sourcePointer[index] = Float16(values[index]).bitPattern
            }
            zero(surface.metalBuffer)
            try kernels.tokenToANE(
                encode, source: source, destination: surface.metalBuffer,
                rows: rows, channels: channels,
                planeStrideElements: surface.planeStrideElements)
            encode.commit()
            encode.waitUntilCompleted()
            if let error = encode.error { throw error }

            let channelSamples = sampledRows(total: channels)
            let positionSamples = Array(Set([
                0, 1, rows / 2, max(0, rows - 2), rows - 1
            ])).sorted()
            let raw = try readSamples(
                surface, rows: channelSamples, positions: positionSamples,
                requireSpatialEquality: false).values
            var rawExpected: [Float] = []
            rawExpected.reserveCapacity(raw.count)
            for position in positionSamples {
                for channel in channelSamples {
                    rawExpected.append(values[position * channels + channel])
                }
            }
            let rawMax = maxAbsDiff(raw, rawExpected)

            guard let decode = context.commandQueue.makeCommandBuffer() else {
                throw AnimapkError.validation("V8 layout decode allocation failed")
            }
            zero(roundTrip)
            try kernels.aneToToken(
                decode, source: surface.metalBuffer, destination: roundTrip,
                rows: rows, channels: channels,
                planeStrideElements: surface.planeStrideElements)
            decode.commit()
            decode.waitUntilCompleted()
            if let error = decode.error { throw error }
            let pointer = roundTrip.contents().bindMemory(
                to: UInt16.self, capacity: count)
            var roundTripMax: Float = 0
            for index in values.indices {
                let actual = Float(Float16(bitPattern: pointer[index]))
                roundTripMax = max(roundTripMax, abs(actual - values[index]))
            }
            return LayoutResult(
                name: name,
                rawMaxAbs: rawMax,
                roundTripMaxAbs: roundTripMax,
                errorText: nil)
        } catch {
            return LayoutResult(
                name: name,
                rawMaxAbs: .infinity,
                roundTripMaxAbs: .infinity,
                errorText: error.localizedDescription)
        }
    }

    private static func projection(
        file: AnimapkFile,
        factory: ModelFactory,
        block: Int,
        name: String,
        suffix: String,
        inputVector: [Float],
        inputSurface: A12ANESurface,
        outputSurface: A12ANESurface,
        positions: [Int]
    ) -> Measurement {
        do {
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                throw AnimapkError.validation("V8 projection spec missing: \(suffix)")
            }
            let rows = sampledRows(total: spec.rows)
            let expected = try cpuProjectionRows(
                file: file, block: block, suffix: suffix,
                input: inputVector, rows: rows)
            try fillSurface(inputSurface, vector: inputVector, positions: positions)
            zero(outputSurface.metalBuffer)
            let model = try factory.projection(
                block: block, suffix: suffix,
                label: "v8_b\(block)_\(name)")
            defer { model.invalidate() }
            var evaluationMS = 0.0
            _ = try model.evaluateInput(
                inputSurface, output: outputSurface,
                milliseconds: &evaluationMS)
            return compare(
                block: block, name: name,
                expectedOnePosition: expected,
                output: outputSurface,
                rows: rows, positions: positions,
                evaluationMS: evaluationMS)
        } catch {
            return failure(block: block, name: name, error: error)
        }
    }

    private static func qkv(
        file: AnimapkFile,
        factory: ModelFactory,
        block: Int,
        inputVector: [Float],
        surfaces: ANEW8DiTSurfaces,
        positions: [Int]
    ) -> [Measurement] {
        let suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight"
        ]
        let names = ["self_q", "self_k", "self_v"]
        let outputs = [surfaces.q, surfaces.k, surfaces.v]
        do {
            let rows = sampledRows(total: DiTBlockExecutor.dim)
            let expected = try suffixes.map { suffix in
                try cpuProjectionRows(
                    file: file, block: block, suffix: suffix,
                    input: inputVector, rows: rows)
            }
            try fillSurface(
                surfaces.tokenInput, vector: inputVector, positions: positions)
            for output in outputs { zero(output.metalBuffer) }
            let model = try factory.qkv(block: block)
            defer { model.invalidate() }
            var evaluationMS = 0.0
            _ = try model.evaluateInput(
                surfaces.tokenInput,
                qOutput: surfaces.q,
                kOutput: surfaces.k,
                vOutput: surfaces.v,
                milliseconds: &evaluationMS)
            return (0..<3).map { index in
                compare(
                    block: block,
                    name: names[index],
                    expectedOnePosition: expected[index],
                    output: outputs[index],
                    rows: rows,
                    positions: positions,
                    evaluationMS: evaluationMS)
            }
        } catch {
            return names.map { failure(block: block, name: $0, error: error) }
        }
    }

    private static func sweepBlock(
        file: AnimapkFile,
        factory: ModelFactory,
        surfaces: ANEW8DiTSurfaces,
        block: Int
    ) -> [Measurement] {
        let tokenPositions = [
            0, DiTBlockExecutor.tokens / 2, DiTBlockExecutor.tokens - 1
        ]
        let contextPositions = [
            0, DiTBlockExecutor.contextTokens / 2,
            DiTBlockExecutor.contextTokens - 1
        ]
        let tokenInput = makeInput(
            count: DiTBlockExecutor.dim,
            seed: 0x7100_0000 + UInt64(block))
        let contextInput = makeInput(
            count: DiTBlockExecutor.contextDim,
            seed: 0x7200_0000 + UInt64(block))
        let hiddenInput = makeInput(
            count: DiTBlockExecutor.hidden,
            seed: 0x7300_0000 + UInt64(block))

        var result = qkv(
            file: file, factory: factory, block: block,
            inputVector: tokenInput, surfaces: surfaces,
            positions: tokenPositions)
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "self_o", suffix: "self_attn.output_proj.weight",
            inputVector: tokenInput,
            inputSurface: surfaces.tokenInput,
            outputSurface: surfaces.tokenOutput,
            positions: tokenPositions))
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "cross_q", suffix: "cross_attn.q_proj.weight",
            inputVector: tokenInput,
            inputSurface: surfaces.tokenInput,
            outputSurface: surfaces.q,
            positions: tokenPositions))
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "cross_k", suffix: "cross_attn.k_proj.weight",
            inputVector: contextInput,
            inputSurface: surfaces.contextInput,
            outputSurface: surfaces.contextK,
            positions: contextPositions))
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "cross_v", suffix: "cross_attn.v_proj.weight",
            inputVector: contextInput,
            inputSurface: surfaces.contextInput,
            outputSurface: surfaces.contextV,
            positions: contextPositions))
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "cross_o", suffix: "cross_attn.output_proj.weight",
            inputVector: tokenInput,
            inputSurface: surfaces.tokenInput,
            outputSurface: surfaces.tokenOutput,
            positions: tokenPositions))
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "mlp1", suffix: "mlp.layer1.weight",
            inputVector: tokenInput,
            inputSurface: surfaces.tokenInput,
            outputSurface: surfaces.hidden,
            positions: tokenPositions))
        result.append(projection(
            file: file, factory: factory, block: block,
            name: "mlp2", suffix: "mlp.layer2.weight",
            inputVector: hiddenInput,
            inputSurface: surfaces.hidden,
            outputSurface: surfaces.tokenOutput,
            positions: tokenPositions))
        return result
    }

    static func run() async -> String {
        var lines = [
            "ANE native projection sweep v8",
            "Diagnostic only; production inference and scheduler are unchanged.",
            "V7 layout failure was an invalid 8 KiB fixture on a 16 KiB-page device.",
            "V8 checks page-valid layout plus every native projection tensor/program family across all 28 blocks.",
            "Projection checks bypass layout kernels via direct plane-major IOSurface I/O.",
            "sampleRows≈\(targetSampleRows) per tensor; positions=3; relRMSE limit=\(relativeRMSELimit).",
            ""
        ]

        do {
            guard A12ANEIsAvailable() else {
                lines.append("SKIP: ANE unavailable: \(A12ANERuntimeStatus())")
                return lines.joined(separator: "\n")
            }
            guard let context = MetalContext() else {
                lines.append("FAIL: Metal unavailable")
                return lines.joined(separator: "\n")
            }
            guard let ditEntry = ModelManifest.entries.first(where: { $0.component == .dit }) else {
                lines.append("FAIL: DiT manifest entry missing")
                return lines.joined(separator: "\n")
            }
            let store = try ModelStore()
            let ditURL = await store.localURL(for: ditEntry)
            guard FileManager.default.fileExists(atPath: ditURL.path) else {
                lines.append("FAIL: installed DiT pack missing")
                return lines.joined(separator: "\n")
            }
            let file = try AnimapkFile(url: ditURL)
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                lines.append("FAIL: prepared ANE cache incomplete; prepare/import the native W8 pack first")
                return lines.joined(separator: "\n")
            }

            let factory = ModelFactory(file: file)
            let kernels = ProbeKernels(context: context)
            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("")

            lines.append("LAYOUT — page-valid independent checks")
            let layouts = [
                layoutCase(
                    name: "production-token-1024x2048",
                    context: context, kernels: kernels,
                    rows: DiTBlockExecutor.tokens,
                    channels: DiTBlockExecutor.dim),
                layoutCase(
                    name: "production-context-512x1024",
                    context: context, kernels: kernels,
                    rows: DiTBlockExecutor.contextTokens,
                    channels: DiTBlockExecutor.contextDim),
                layoutCase(
                    name: "padded-37x128",
                    context: context, kernels: kernels,
                    rows: 37, channels: 128)
            ]
            lines.append(contentsOf: layouts.map(\.line))
            lines.append("")

            // Match production lifetime and eliminate IOSurface allocator churn
            // from the 28-block projection sweep.
            let surfaces = try ANEW8DiTSurfaces(device: context.device)
            lines.append("PROJECTIONS — all 28 blocks, direct plane-major I/O")
            var measurements: [Measurement] = []
            measurements.reserveCapacity(ModelConstants.ditBlocks * 10)
            for block in 0..<ModelConstants.ditBlocks {
                let current = sweepBlock(
                    file: file, factory: factory,
                    surfaces: surfaces, block: block)
                measurements.append(contentsOf: current)
                let failures = current.filter { !$0.passed }
                let valid = current.filter { $0.errorText == nil }
                if let worst = valid.max(by: { $0.relativeRMSE < $1.relativeRMSE }) {
                    let status = failures.isEmpty ? "PASS" : "FAIL"
                    lines.append(
                        "b\(twoInt(block)) \(status) failures=\(failures.count) "
                        + "worst=\(worst.name) rel=\(five(worst.relativeRMSE)) "
                        + "maxAbs=\(six(worst.maxAbs))")
                } else {
                    lines.append("b\(twoInt(block)) FAIL failures=\(failures.count) no valid measurements")
                }
                for failure in failures {
                    lines.append("  -> \(failure.compact)")
                }
            }

            lines.append("")
            lines.append("FAMILY WORST CASES")
            let families = [
                "self_q", "self_k", "self_v", "self_o",
                "cross_q", "cross_k", "cross_v", "cross_o",
                "mlp1", "mlp2"
            ]
            for family in families {
                let valid = measurements.filter {
                    $0.name == family && $0.errorText == nil
                }
                if let worst = valid.max(by: { $0.relativeRMSE < $1.relativeRMSE }) {
                    lines.append(
                        "\(family) worst=b\(twoInt(worst.block)) "
                        + "rel=\(five(worst.relativeRMSE)) "
                        + "maxAbs=\(six(worst.maxAbs)) "
                        + "spatialBits=\(worst.spatialBitMismatches)")
                } else {
                    lines.append("\(family) unavailable")
                }
            }

            let layoutFailures = layouts.filter { !$0.passed }
            let projectionFailures = measurements.filter { !$0.passed }
            let validCount = measurements.filter { $0.errorText == nil }.count
            lines.append("")
            lines.append(
                "summary: layout=\(layouts.count - layoutFailures.count)/\(layouts.count) pass "
                + "projectionTensors=\(measurements.count - projectionFailures.count)/\(measurements.count) pass "
                + "validMeasurements=\(validCount) programsAttempted=224")

            if let failure = layoutFailures.first {
                lines.append("FIRST BAD BOUNDARY: layout-\(failure.name)")
                lines.append("Interpretation: a page-valid layout check failed; inspect the two Metal bridge kernels before ANE math.")
            } else if let failure = projectionFailures.first {
                lines.append("FIRST BAD BOUNDARY: block \(failure.block) / \(failure.name)")
                lines.append("Interpretation: this prepared ANE projection disagrees with the native pack row contract or differs across identical spatial positions. Fix the projection/model ABI before attention or scheduler policy.")
            } else {
                lines.append("ALL V8 NATIVE PROJECTION BOUNDARIES PASSED")
                lines.append("Interpretation: page-valid layout, fused QKV ordering, rectangular cross K/V, both MLP shapes, and sampled rows from every native projection tensor are clean across all 28 blocks.")
                lines.append("Next target: instrument one real ANE hybrid block at self/cross/MLP residual boundaries and then inside RMSNorm/RoPE/attention/gating.")
            }
        } catch {
            lines.append("FAIL V8 setup threw: \(error.localizedDescription)")
        }
        return lines.joined(separator: "\n")
    }

    private static func two(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    private static func five(_ value: Float) -> String {
        String(format: "%.5f", value)
    }

    private static func six(_ value: Float) -> String {
        String(format: "%.6g", value)
    }

    private static func eight(_ value: Float) -> String {
        String(format: "%.8g", value)
    }

    private static func twoInt(_ value: Int) -> String {
        String(format: "%02d", value)
    }
}
