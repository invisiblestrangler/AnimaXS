import Foundation
import Dispatch
import Metal

/// V7 device-only ANE correctness microscope.
///
/// This replaces only the old timing probe. Production inference is unchanged.
/// It tests the lowest correctness boundaries that can explain both a finite
/// woven/checker output and seed-dependent late NaN/Inf:
/// 1. the exact token-major <-> Espresso channel-major Metal bridge,
/// 2. a prepared native-W8 ANE projection against a per-row CPU reference,
/// 3. the real generation-local P5 private-buffer storage path,
/// 4. repeat determinism of one loaded ANE projection,
/// 5. foreground evaluate while another prepared model load/unload churns the
///    same private `_ANEClient sharedConnection` used by production.
enum ANERingProbe {
    private struct Outcome {
        let name: String
        let passed: Bool
        let detail: String

        var line: String {
            let status = passed ? "PASS" : "FAIL"
            return "\(status) \(name): \(detail)"
        }
    }

    /// Diagnostic-local mirror of DiTBlockExecutor's production ANE layout
    /// bridge. Keeping it here avoids exposing production-private helpers just
    /// for a probe while still using the exact same kernels and dispatch shape.
    private final class ProbeKernels {
        private let context: MetalContext

        init(context: MetalContext) {
            self.context = context
        }

        func encodeTokenToANE(
            commandBuffer: MTLCommandBuffer,
            source: MTLBuffer,
            sourceOffsetBytes: Int = 0,
            destination: MTLBuffer,
            destinationOffsetBytes: Int = 0,
            tokenCount: Int,
            channels: Int,
            planeStrideElements: UInt
        ) throws {
            try encodeBridge(
                commandBuffer: commandBuffer,
                kernel: "dit_token_to_ane_f16",
                source: source,
                sourceOffsetBytes: sourceOffsetBytes,
                destination: destination,
                destinationOffsetBytes: destinationOffsetBytes,
                rows: tokenCount,
                channels: channels,
                planeStrideElements: planeStrideElements)
        }

        func encodeANEToToken(
            commandBuffer: MTLCommandBuffer,
            source: MTLBuffer,
            sourceOffsetBytes: Int = 0,
            destination: MTLBuffer,
            destinationOffsetBytes: Int = 0,
            tokenCount: Int,
            channels: Int,
            planeStrideElements: UInt
        ) throws {
            try encodeBridge(
                commandBuffer: commandBuffer,
                kernel: "dit_ane_to_token_f16",
                source: source,
                sourceOffsetBytes: sourceOffsetBytes,
                destination: destination,
                destinationOffsetBytes: destinationOffsetBytes,
                rows: tokenCount,
                channels: channels,
                planeStrideElements: planeStrideElements)
        }

        private func encodeBridge(
            commandBuffer: MTLCommandBuffer,
            kernel: String,
            source: MTLBuffer,
            sourceOffsetBytes: Int,
            destination: MTLBuffer,
            destinationOffsetBytes: Int,
            rows: Int,
            channels: Int,
            planeStrideElements: UInt
        ) throws {
            let pipeline = try context.pipeline(named: kernel)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw AnimapkError.validation("V7 failed to create \(kernel) encoder")
            }
            var rows32 = UInt32(rows)
            var channels32 = UInt32(channels)
            var stride32 = UInt32(planeStrideElements)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(source, offset: sourceOffsetBytes, index: 0)
            encoder.setBuffer(destination, offset: destinationOffsetBytes, index: 1)
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

    private final class ProjectionFactory: @unchecked Sendable {
        private let file: AnimapkFile

        init(file: AnimapkFile) {
            self.file = file
        }

        func make(block: Int, suffix: String, label: String) throws -> A12ANEProjectionModel {
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                throw AnimapkError.validation("V7 unknown projection suffix \(suffix)")
            }
            let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
            guard let digest = tensor.blobSHA256 else {
                throw AnimapkError.validation("V7 native tensor hash missing: \(suffix)")
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
    }

    private final class ChurnState: @unchecked Sendable {
        private let lock = NSLock()
        private var finished = false
        private var errorText: String?

        func finish(error: Error? = nil) {
            lock.lock()
            if let error { errorText = error.localizedDescription }
            finished = true
            lock.unlock()
        }

        var isFinished: Bool {
            lock.lock(); defer { lock.unlock() }
            return finished
        }

        var error: String? {
            lock.lock(); defer { lock.unlock() }
            return errorText
        }
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        var result: Float = 0
        for i in a.indices { result = max(result, abs(a[i] - b[i])) }
        return result
    }

    private static func relativeRMSE(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var errorSquared: Double = 0
        var referenceSquared: Double = 0
        for i in a.indices {
            let delta = Double(a[i] - b[i])
            errorSquared += delta * delta
            let reference = Double(b[i])
            referenceSquared += reference * reference
        }
        return Float(sqrt(errorSquared / max(referenceSquared, 1e-30)))
    }

    private static func finite(_ values: [Float]) -> Bool {
        values.allSatisfy(\.isFinite)
    }

    private static func makeInput(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { index in
            state &*= 6_364_136_223_846_793_005
            state &+= 1_442_695_040_888_963_407
            let unit = Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF)
            return (unit * 2 - 1) * 0.2 + Float((index % 17) - 8) * 0.001
        }
    }

    private static func f16Rounded(_ values: [Float]) -> [Float] {
        values.map { Float(Float16($0)) }
    }

    private static func readF16Buffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Float(Float16(bitPattern: pointer[$0])) }
    }

    private static func writeF16Buffer(_ values: [Float], to buffer: MTLBuffer) {
        let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: values.count)
        for index in values.indices {
            pointer[index] = Float16(values[index]).bitPattern
        }
    }

    private static func zero(_ buffer: MTLBuffer) {
        buffer.contents().initializeMemory(
            as: UInt8.self, repeating: 0, count: buffer.length)
    }

    /// CPU reference for the exact native pack contract:
    ///     weight[row,col] = U8[row,col] * scale[row] + bias[row]
    /// Inputs are already rounded to FP16 because that is what ANE receives.
    private static func cpuProjectionReference(
        file: AnimapkFile,
        block: Int,
        suffix: String,
        input: [Float],
        tokenCount: Int
    ) throws -> [Float] {
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            throw AnimapkError.validation("V7 CPU reference unknown suffix \(suffix)")
        }
        let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
        guard input.count == tokenCount * spec.columns,
              let scaleBytes = file.scaleBytes(tensor),
              let biasBytes = file.zeroBytes(tensor) else {
            throw AnimapkError.validation("V7 CPU reference payload mismatch")
        }
        let quantized = file.dataBytes(tensor).bindMemory(to: UInt8.self)
        let scales = scaleBytes.bindMemory(to: Float.self)
        let biases = biasBytes.bindMemory(to: Float.self)
        guard quantized.count == spec.rows * spec.columns,
              scales.count == spec.rows,
              biases.count == spec.rows else {
            throw AnimapkError.validation("V7 native W8 row payload sizes disagree with metadata")
        }

        var output = [Float](repeating: 0, count: tokenCount * spec.rows)
        for token in 0..<tokenCount {
            let inputBase = token * spec.columns
            let outputBase = token * spec.rows
            for row in 0..<spec.rows {
                let weightBase = row * spec.columns
                let scale = scales[row]
                let bias = biases[row]
                var sum: Float = 0
                for column in 0..<spec.columns {
                    let weight = Float(quantized[weightBase + column]) * scale + bias
                    sum += input[inputBase + column] * weight
                }
                output[outputBase + row] = sum
            }
        }
        return output
    }

    private static func runSafely(
        name: String,
        _ body: () throws -> Outcome
    ) -> Outcome {
        do { return try body() }
        catch {
            return Outcome(
                name: name, passed: false,
                detail: "threw: \(error.localizedDescription)")
        }
    }

    private static func layoutRoundTrip(
        context: MetalContext,
        kernels: ProbeKernels
    ) throws -> Outcome {
        let tokens = 37
        let channels = 64
        let elementCount = tokens * channels
        let surface = try A12ANESurface(
            device: context.device, channels: UInt(channels), spatial: UInt(tokens))
        guard let source = context.device.makeBuffer(
                length: elementCount * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let roundTrip = context.device.makeBuffer(
                length: elementCount * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            return Outcome(name: "layout-roundtrip", passed: false, detail: "Metal allocation failed")
        }
        let sourceValues = makeInput(count: elementCount, seed: 0xA11CE)
        writeF16Buffer(sourceValues, to: source)
        zero(surface.metalBuffer)
        zero(roundTrip)
        try kernels.encodeTokenToANE(
            commandBuffer: commandBuffer,
            source: source,
            destination: surface.metalBuffer,
            tokenCount: tokens,
            channels: channels,
            planeStrideElements: surface.planeStrideElements)
        try kernels.encodeANEToToken(
            commandBuffer: commandBuffer,
            source: surface.metalBuffer,
            destination: roundTrip,
            tokenCount: tokens,
            channels: channels,
            planeStrideElements: surface.planeStrideElements)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let decoded = readF16Buffer(roundTrip, count: elementCount)
        let expected = f16Rounded(sourceValues)
        let maxDiff = maxAbsDiff(decoded, expected)
        return Outcome(
            name: "layout-roundtrip",
            passed: maxDiff == 0,
            detail: String(
                format: "tokens=%d channels=%d planeStride=%llu maxAbs=%.8g",
                tokens, channels, UInt64(surface.planeStrideElements), maxDiff))
    }

    private static func projectionVsCPU(
        file: AnimapkFile,
        context: MetalContext,
        kernels: ProbeKernels,
        factory: ProjectionFactory
    ) throws -> Outcome {
        let block = 0
        let suffix = "cross_attn.q_proj.weight"
        let tokenCount = 4
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            return Outcome(name: "w8-vs-cpu", passed: false, detail: "projection spec missing")
        }
        let rawInput = makeInput(count: tokenCount * spec.columns, seed: 0xC0FFEE)
        let input = f16Rounded(rawInput)
        let cpu = try cpuProjectionReference(
            file: file, block: block, suffix: suffix,
            input: input, tokenCount: tokenCount)

        let inputSurface = try A12ANESurface(
            device: context.device,
            channels: UInt(spec.columns), spatial: UInt(spec.spatial))
        let outputSurface = try A12ANESurface(
            device: context.device,
            channels: UInt(spec.rows), spatial: UInt(spec.spatial))
        guard let tokenInput = context.device.makeBuffer(
                length: tokenCount * spec.columns * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let tokenOutput = context.device.makeBuffer(
                length: tokenCount * spec.rows * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let encode = context.commandQueue.makeCommandBuffer() else {
            return Outcome(name: "w8-vs-cpu", passed: false, detail: "Metal allocation failed")
        }
        writeF16Buffer(input, to: tokenInput)
        zero(inputSurface.metalBuffer)
        zero(outputSurface.metalBuffer)
        try kernels.encodeTokenToANE(
            commandBuffer: encode,
            source: tokenInput,
            destination: inputSurface.metalBuffer,
            tokenCount: tokenCount,
            channels: spec.columns,
            planeStrideElements: inputSurface.planeStrideElements)
        encode.commit()
        encode.waitUntilCompleted()
        if let error = encode.error { throw error }

        let model = try factory.make(block: block, suffix: suffix, label: "V7 CPU compare")
        defer { model.invalidate() }
        var evaluationMS = 0.0
        _ = try model.evaluateInput(
            inputSurface, output: outputSurface, milliseconds: &evaluationMS)

        guard let decode = context.commandQueue.makeCommandBuffer() else {
            return Outcome(name: "w8-vs-cpu", passed: false, detail: "decode command buffer failed")
        }
        zero(tokenOutput)
        try kernels.encodeANEToToken(
            commandBuffer: decode,
            source: outputSurface.metalBuffer,
            destination: tokenOutput,
            tokenCount: tokenCount,
            channels: spec.rows,
            planeStrideElements: outputSurface.planeStrideElements)
        decode.commit()
        decode.waitUntilCompleted()
        if let error = decode.error { throw error }

        let ane = readF16Buffer(tokenOutput, count: tokenCount * spec.rows)
        let rel = relativeRMSE(ane, cpu)
        let maxDiff = maxAbsDiff(ane, cpu)
        return Outcome(
            name: "w8-vs-cpu",
            // This is a gross correctness gate, not a bit-exact precision gate.
            // ANE accumulation/boundaries need not match Float32 CPU exactly.
            passed: finite(cpu) && finite(ane) && rel < 0.05,
            detail: String(
                format: "eval=%.2fms relRMSE=%.6g maxAbs=%.6g cpuFinite=%@ aneFinite=%@",
                evaluationMS, rel, maxDiff,
                String(finite(cpu)), String(finite(ane))))
    }

    private static func p5RoundTrip(context: MetalContext) throws -> Outcome {
        guard let cache = CrossKVCache(device: context.device) else {
            return Outcome(name: "p5-private-cache", passed: false, detail: "real CrossKVCache allocation failed")
        }
        let block = 13
        let byteCount = CrossKVCache.tensorBytes
        let elementCount = byteCount / MemoryLayout<Float16>.stride
        guard let source = context.device.makeBuffer(
                length: byteCount, options: .storageModeShared),
              let decoded = context.device.makeBuffer(
                length: byteCount, options: .storageModeShared),
              let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            return Outcome(name: "p5-private-cache", passed: false, detail: "Metal allocation failed")
        }
        let values = makeInput(count: elementCount, seed: 0xB117)
        writeF16Buffer(values, to: source)
        zero(decoded)
        blit.copy(
            from: source, sourceOffset: 0,
            to: cache.buffer, destinationOffset: cache.kOffset(block: block),
            size: byteCount)
        blit.copy(
            from: cache.buffer, sourceOffset: cache.kOffset(block: block),
            to: decoded, destinationOffset: 0,
            size: byteCount)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let expected = f16Rounded(values)
        let actual = readF16Buffer(decoded, count: elementCount)
        let maxDiff = maxAbsDiff(expected, actual)
        return Outcome(
            name: "p5-private-cache",
            passed: maxDiff == 0,
            detail: String(
                format: "block=%d bytes=%d maxAbs=%.8g",
                block, byteCount, maxDiff))
    }

    private static func makeStableProjectionFixture(
        file: AnimapkFile,
        context: MetalContext,
        kernels: ProbeKernels,
        factory: ProjectionFactory,
        block: Int,
        suffix: String,
        tokenCount: Int,
        seed: UInt64
    ) throws -> (
        model: A12ANEProjectionModel,
        input: A12ANESurface,
        output: A12ANESurface,
        tokenOutput: MTLBuffer,
        rows: Int
    ) {
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            throw AnimapkError.validation("V7 stable fixture projection spec missing")
        }
        let inputSurface = try A12ANESurface(
            device: context.device,
            channels: UInt(spec.columns), spatial: UInt(spec.spatial))
        let outputSurface = try A12ANESurface(
            device: context.device,
            channels: UInt(spec.rows), spatial: UInt(spec.spatial))
        guard let tokenInput = context.device.makeBuffer(
                length: tokenCount * spec.columns * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let tokenOutput = context.device.makeBuffer(
                length: tokenCount * spec.rows * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let encode = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("V7 stable fixture Metal allocation failed")
        }
        let values = makeInput(count: tokenCount * spec.columns, seed: seed)
        writeF16Buffer(values, to: tokenInput)
        zero(inputSurface.metalBuffer)
        try kernels.encodeTokenToANE(
            commandBuffer: encode,
            source: tokenInput,
            destination: inputSurface.metalBuffer,
            tokenCount: tokenCount,
            channels: spec.columns,
            planeStrideElements: inputSurface.planeStrideElements)
        encode.commit()
        encode.waitUntilCompleted()
        if let error = encode.error { throw error }
        let model = try factory.make(block: block, suffix: suffix, label: "V7 stable projection")
        return (model, inputSurface, outputSurface, tokenOutput, spec.rows)
    }

    private static func evaluateStableFixture(
        context: MetalContext,
        kernels: ProbeKernels,
        fixture: (
            model: A12ANEProjectionModel,
            input: A12ANESurface,
            output: A12ANESurface,
            tokenOutput: MTLBuffer,
            rows: Int
        ),
        tokenCount: Int
    ) throws -> [Float] {
        var evaluationMS = 0.0
        _ = try fixture.model.evaluateInput(
            fixture.input, output: fixture.output, milliseconds: &evaluationMS)
        guard let decode = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("V7 stable decode command buffer failed")
        }
        try kernels.encodeANEToToken(
            commandBuffer: decode,
            source: fixture.output.metalBuffer,
            destination: fixture.tokenOutput,
            tokenCount: tokenCount,
            channels: fixture.rows,
            planeStrideElements: fixture.output.planeStrideElements)
        decode.commit()
        decode.waitUntilCompleted()
        if let error = decode.error { throw error }
        return readF16Buffer(fixture.tokenOutput, count: tokenCount * fixture.rows)
    }

    private static func repeatedDeterminism(
        file: AnimapkFile,
        context: MetalContext,
        kernels: ProbeKernels,
        factory: ProjectionFactory
    ) throws -> Outcome {
        let tokenCount = 8
        let fixture = try makeStableProjectionFixture(
            file: file, context: context, kernels: kernels, factory: factory,
            block: 0, suffix: "cross_attn.q_proj.weight",
            tokenCount: tokenCount, seed: 0xDE71)
        defer { fixture.model.invalidate() }
        let baseline = try evaluateStableFixture(
            context: context, kernels: kernels,
            fixture: fixture, tokenCount: tokenCount)
        var worst: Float = 0
        for _ in 0..<11 {
            let current = try evaluateStableFixture(
                context: context, kernels: kernels,
                fixture: fixture, tokenCount: tokenCount)
            worst = max(worst, maxAbsDiff(baseline, current))
        }
        return Outcome(
            name: "repeat-determinism",
            passed: finite(baseline) && worst == 0,
            detail: String(format: "12 evaluations worstMaxAbs=%.8g", worst))
    }

    private static func sharedClientConcurrency(
        file: AnimapkFile,
        context: MetalContext,
        kernels: ProbeKernels,
        factory: ProjectionFactory
    ) throws -> Outcome {
        let tokenCount = 8
        let fixture = try makeStableProjectionFixture(
            file: file, context: context, kernels: kernels, factory: factory,
            block: 0, suffix: "cross_attn.q_proj.weight",
            tokenCount: tokenCount, seed: 0x51A1)
        defer { fixture.model.invalidate() }
        let baseline = try evaluateStableFixture(
            context: context, kernels: kernels,
            fixture: fixture, tokenCount: tokenCount)

        let churnQueue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-v7-client-churn",
            qos: .userInitiated)
        let churnState = ChurnState()
        let started = DispatchSemaphore(value: 0)
        churnQueue.async {
            started.signal()
            do {
                // Reuse one already-prepared projection key. Reconstructing and
                // invalidating it repeatedly exercises private loadModel/unloadModel
                // without accumulating residency or first-touching dozens of models.
                for index in 0..<24 {
                    let model = try factory.make(
                        block: 1,
                        suffix: "cross_attn.output_proj.weight",
                        label: "V7 churn \(index)")
                    model.invalidate()
                }
                churnState.finish()
            } catch {
                churnState.finish(error: error)
            }
        }
        started.wait()

        var worst: Float = 0
        var foregroundError: String?
        var evaluations = 0
        // Stay in the foreground until the background churn has actually
        // overlapped, while bounding diagnostic time even if private unload is slow.
        for index in 0..<512 {
            do {
                let current = try evaluateStableFixture(
                    context: context, kernels: kernels,
                    fixture: fixture, tokenCount: tokenCount)
                worst = max(worst, maxAbsDiff(baseline, current))
                evaluations += 1
                if !finite(current) {
                    foregroundError = "foreground iteration \(index) became non-finite"
                    break
                }
            } catch {
                foregroundError = "foreground iteration \(index): \(error.localizedDescription)"
                break
            }
            if index >= 63 && churnState.isFinished { break }
        }
        // If the foreground loop reaches its cap first, allow background cleanup
        // to finish before objects leave scope. This wait is outside the measured
        // correctness condition; the overlap already happened above.
        while !churnState.isFinished {
            Thread.sleep(forTimeInterval: 0.005)
        }
        if let backgroundError = churnState.error {
            foregroundError = "background churn: \(backgroundError)"
        }

        return Outcome(
            name: "shared-client-concurrency",
            passed: foregroundError == nil && worst == 0,
            detail: foregroundError ?? String(
                format: "%d foreground evals + 24 load/unload cycles worstMaxAbs=%.8g",
                evaluations, worst))
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE correctness microscope v7",
            "Diagnostic only; production inference is unchanged.",
            "Order: layout -> W8 CPU reference -> real P5 storage -> determinism -> shared-client concurrency.",
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
                lines.append("FAIL: prepared ANE cache incomplete; import/prepare the native W8 pack first")
                return lines.joined(separator: "\n")
            }

            let kernels = ProbeKernels(context: context)
            let factory = ProjectionFactory(file: file)
            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client selectors: \(A12ANEClientCapabilitySummary())")
            lines.append("pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme ?? "n/a")")
            lines.append("")

            let tests = [
                runSafely(name: "layout-roundtrip") {
                    try layoutRoundTrip(context: context, kernels: kernels)
                },
                runSafely(name: "w8-vs-cpu") {
                    try projectionVsCPU(
                        file: file, context: context,
                        kernels: kernels, factory: factory)
                },
                runSafely(name: "p5-private-cache") {
                    try p5RoundTrip(context: context)
                },
                runSafely(name: "repeat-determinism") {
                    try repeatedDeterminism(
                        file: file, context: context,
                        kernels: kernels, factory: factory)
                },
                runSafely(name: "shared-client-concurrency") {
                    try sharedClientConcurrency(
                        file: file, context: context,
                        kernels: kernels, factory: factory)
                }
            ]
            lines.append(contentsOf: tests.map(\.line))
            lines.append("")

            if let firstFailure = tests.first(where: { !$0.passed }) {
                lines.append("FIRST BAD BOUNDARY: \(firstFailure.name)")
                switch firstFailure.name {
                case "layout-roundtrip":
                    lines.append("Interpretation: token<->ANE layout bridge is corrupt before private ANE math.")
                case "w8-vs-cpu":
                    lines.append("Interpretation: native W8 projection semantics/layout disagree with the pack reference.")
                case "p5-private-cache":
                    lines.append("Interpretation: the real generation-local private P5 storage/blit path corrupts bytes.")
                case "repeat-determinism":
                    lines.append("Interpretation: repeated evaluation of one loaded private ANE model is not deterministic.")
                case "shared-client-concurrency":
                    lines.append("Interpretation: concurrent private load/unload and evaluate through the shared client is unsafe; serialize client operations before changing model math.")
                default:
                    break
                }
            } else {
                lines.append("ALL V7 MICRO-BOUNDARIES PASSED")
                lines.append("Next target: a complete block-level ANE-vs-dequantized-MPS intermediate comparison; corruption is above the isolated primitives.")
            }
        } catch {
            lines.append("FAIL probe setup threw: \(error.localizedDescription)")
        }
        return lines.joined(separator: "\n")
    }
}
