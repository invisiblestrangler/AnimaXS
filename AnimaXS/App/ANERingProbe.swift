import Foundation
import Dispatch
import Metal

/// V7 device-only ANE correctness microscope.
///
/// This deliberately replaces the old timing probe without touching production
/// inference. It isolates the four boundaries that can explain a finite woven
/// image or seed-dependent late NaN/Inf:
/// 1. token-major <-> Espresso channel-major layout bridge,
/// 2. prepared W8 projection vs the native pack's per-row dequant contract,
/// 3. private P5 cache blit round-trip,
/// 4. foreground evaluation while another prepared model is loaded/unloaded
///    through the same private `_ANEClient sharedConnection`.
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

    private final class ChurnWorker: @unchecked Sendable {
        let file: AnimapkFile
        init(file: AnimapkFile) { self.file = file }

        func makeProjection(block: Int, suffix: String, label: String) throws -> A12ANEProjectionModel {
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                throw AnimapkError.validation("V7 unknown projection suffix \(suffix)")
            }
            let tensor = try ANEW8NativePack.tensor(
                file: file, block: block, suffix: suffix)
            guard let digest = tensor.blobSHA256 else {
                throw AnimapkError.validation("V7 native tensor hash missing")
            }
            let key = ANEW8NativePack.projectionCacheKey(
                block: block, tag: spec.tag, hash: digest)
            return try A12ANEProjectionModel(
                preparedInputChannels: UInt(spec.columns),
                outputChannels: UInt(spec.rows),
                sequenceLength: UInt(ANEW8NativePack.expectedSequenceLength),
                cacheKey: key,
                label: label)
        }

        func makeQKV(block: Int, label: String) throws -> A12ANEQKVModel {
            let q = try ANEW8NativePack.tensor(file: file, block: block, suffix: "self.q.weight")
            let k = try ANEW8NativePack.tensor(file: file, block: block, suffix: "self.k.weight")
            let v = try ANEW8NativePack.tensor(file: file, block: block, suffix: "self.v.weight")
            guard let qHash = q.blobSHA256, let kHash = k.blobSHA256, let vHash = v.blobSHA256 else {
                throw AnimapkError.validation("V7 fused QKV native tensor hash missing")
            }
            let key = ANEW8NativePack.qkvCacheKey(block: block, qHash: qHash, kHash: kHash, vHash: vHash)
            return try A12ANEQKVModel(
                preparedInputChannels: UInt(ANEW8NativePack.expectedHiddenSize),
                outputChannels: UInt(ANEW8NativePack.expectedHiddenSize),
                sequenceLength: UInt(ANEW8NativePack.expectedSequenceLength),
                cacheKey: key,
                label: label)
        }
    }

    private static func elapsedMS(_ start: UInt64, _ end: UInt64) -> Double {
        Double(end - start) / 1_000_000.0
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        var result: Float = 0
        for i in a.indices {
            result = max(result, abs(a[i] - b[i]))
        }
        return result
    }

    private static func meanAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var sum: Double = 0
        for i in a.indices {
            sum += Double(abs(a[i] - b[i]))
        }
        return Float(sum / Double(a.count))
    }

    private static func relativeRMSE(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return .infinity }
        var err2: Double = 0
        var ref2: Double = 0
        for i in a.indices {
            let d = Double(a[i] - b[i])
            err2 += d * d
            let r = Double(b[i])
            ref2 += r * r
        }
        return Float(sqrt(err2 / max(ref2, 1e-30)))
    }

    private static func finite(_ x: [Float]) -> Bool {
        x.allSatisfy(\.isFinite)
    }

    private static func makeInput(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { i in
            state &*= 6_364_136_223_846_793_005
            state &+= 1_442_695_040_888_963_407
            let u = Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF)
            return (u * 2 - 1) * 0.2 + Float((i % 17) - 8) * 0.001
        }
    }

    private static func readF16Buffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Float(Float16(bitPattern: ptr[$0])) }
    }

    private static func writeF16Buffer(_ values: [Float], to buffer: MTLBuffer) {
        let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: values.count)
        for i in values.indices {
            ptr[i] = Float16(values[i]).bitPattern
        }
    }

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
        let rows = spec.rows
        let columns = spec.columns
        guard input.count == tokenCount * columns else {
            throw AnimapkError.validation("V7 CPU reference input count mismatch")
        }

        var output = [Float](repeating: 0, count: tokenCount * rows)
        try tensor.withANEUInt8PerRow { quantized, scales, biases in
            for token in 0..<tokenCount {
                let xBase = token * columns
                let yBase = token * rows
                for row in 0..<rows {
                    let wBase = row * columns
                    let scale = scales[row]
                    let bias = biases[row]
                    var sum: Float = 0
                    for col in 0..<columns {
                        let weight = Float(quantized[wBase + col]) * scale + bias
                        sum += input[xBase + col] * weight
                    }
                    output[yBase + row] = sum
                }
            }
        }
        return output
    }

    private static func projectionOutput(
        _ model: A12ANEProjectionModel,
        input: MTLBuffer,
        inputOffsetBytes: Int,
        output: MTLBuffer,
        outputOffsetBytes: Int,
        outputCount: Int
    ) throws -> [Float] {
        try model.evaluateInput(
            input,
            inputOffsetBytes: UInt(inputOffsetBytes),
            output: output,
            outputOffsetBytes: UInt(outputOffsetBytes))
        let offset = outputOffsetBytes / MemoryLayout<UInt16>.stride
        let ptr = output.contents().bindMemory(to: UInt16.self, capacity: offset + outputCount)
        return (0..<outputCount).map { Float(Float16(bitPattern: ptr[offset + $0])) }
    }

    private static func layoutRoundTrip(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        kernels: AnimaKernels
    ) throws -> Outcome {
        let tokens = 37
        let channels = 64
        let tokenCount = tokens * channels
        let planeStride = ANEW8DiTExecutor.alignedANEPlaneStrideElements(sequence: tokens)
        let aneCount = planeStride * channels
        let sourceValues = makeInput(count: tokenCount, seed: 0xA11CE)

        guard let source = device.makeBuffer(length: tokenCount * 2, options: .storageModeShared),
              let ane = device.makeBuffer(length: aneCount * 2, options: .storageModeShared),
              let roundTrip = device.makeBuffer(length: tokenCount * 2, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return Outcome(name: "layout-roundtrip", passed: false, detail: "Metal allocation failed")
        }
        writeF16Buffer(sourceValues, to: source)
        try kernels.encodeTokenToANE(
            commandBuffer: commandBuffer,
            source: source,
            sourceOffsetBytes: 0,
            destination: ane,
            destinationOffsetBytes: 0,
            tokenCount: tokens,
            channels: channels,
            planeStrideElements: planeStride)
        try kernels.encodeANEToToken(
            commandBuffer: commandBuffer,
            source: ane,
            sourceOffsetBytes: 0,
            destination: roundTrip,
            destinationOffsetBytes: 0,
            tokenCount: tokens,
            channels: channels,
            planeStrideElements: planeStride)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let decoded = readF16Buffer(roundTrip, count: tokenCount)
        let sourceF16 = sourceValues.map { Float(Float16($0)) }
        let maxDiff = maxAbsDiff(decoded, sourceF16)
        return Outcome(
            name: "layout-roundtrip",
            passed: maxDiff == 0,
            detail: String(format: "tokens=%d channels=%d planeStride=%d maxAbs=%.8g", tokens, channels, planeStride, maxDiff))
    }

    private static func projectionVsCPU(
        file: AnimapkFile,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        kernels: AnimaKernels
    ) throws -> Outcome {
        let block = 0
        let suffix = "cross.q.weight"
        let tokenCount = 4
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            return Outcome(name: "w8-vs-cpu", passed: false, detail: "missing projection spec")
        }
        let inputValues = makeInput(count: tokenCount * spec.columns, seed: 0xC0FFEE)
        let cpu = try cpuProjectionReference(
            file: file, block: block, suffix: suffix, input: inputValues, tokenCount: tokenCount)
        let planeStride = ANEW8DiTExecutor.alignedANEPlaneStrideElements(sequence: ANEW8NativePack.expectedSequenceLength)
        let inputANECount = planeStride * spec.columns
        let outputANECount = planeStride * spec.rows
        guard let tokenInput = device.makeBuffer(length: tokenCount * spec.columns * 2, options: .storageModeShared),
              let aneInput = device.makeBuffer(length: inputANECount * 2, options: .storageModeShared),
              let aneOutput = device.makeBuffer(length: outputANECount * 2, options: .storageModeShared),
              let tokenOutput = device.makeBuffer(length: ANEW8NativePack.expectedSequenceLength * spec.rows * 2, options: .storageModeShared),
              let encode = commandQueue.makeCommandBuffer(),
              let decode = commandQueue.makeCommandBuffer() else {
            return Outcome(name: "w8-vs-cpu", passed: false, detail: "Metal allocation failed")
        }
        writeF16Buffer(inputValues, to: tokenInput)
        try kernels.encodeTokenToANE(
            commandBuffer: encode,
            source: tokenInput,
            sourceOffsetBytes: 0,
            destination: aneInput,
            destinationOffsetBytes: 0,
            tokenCount: tokenCount,
            channels: spec.columns,
            planeStrideElements: planeStride)
        encode.commit()
        encode.waitUntilCompleted()

        let worker = ChurnWorker(file: file)
        let model = try worker.makeProjection(block: block, suffix: suffix, label: "V7 cpu compare")
        defer { model.invalidate() }
        try model.evaluateInput(
            aneInput,
            inputOffsetBytes: 0,
            output: aneOutput,
            outputOffsetBytes: 0)

        try kernels.encodeANEToToken(
            commandBuffer: decode,
            source: aneOutput,
            sourceOffsetBytes: 0,
            destination: tokenOutput,
            destinationOffsetBytes: 0,
            tokenCount: tokenCount,
            channels: spec.rows,
            planeStrideElements: planeStride)
        decode.commit()
        decode.waitUntilCompleted()
        let ane = readF16Buffer(tokenOutput, count: tokenCount * spec.rows)
        let rel = relativeRMSE(ane, cpu)
        let maxDiff = maxAbsDiff(ane, cpu)
        return Outcome(
            name: "w8-vs-cpu",
            passed: finite(ane) && rel < 0.02,
            detail: String(format: "relRMSE=%.6g maxAbs=%.6g cpuFinite=%@ aneFinite=%@", rel, maxDiff, String(finite(cpu)), String(finite(ane))))
    }

    private static func p5RoundTrip(
        device: MTLDevice,
        commandQueue: MTLCommandQueue
    ) -> Outcome {
        let count = 8192
        guard let source = device.makeBuffer(length: count * 2, options: .storageModeShared),
              let cache = device.makeBuffer(length: count * 2, options: .storageModePrivate),
              let decoded = device.makeBuffer(length: count * 2, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            return Outcome(name: "p5-private-blit", passed: false, detail: "Metal allocation failed")
        }
        let values = makeInput(count: count, seed: 0xB117)
        writeF16Buffer(values, to: source)
        blit.copy(from: source, sourceOffset: 0, to: cache, destinationOffset: 0, size: count * 2)
        blit.copy(from: cache, sourceOffset: 0, to: decoded, destinationOffset: 0, size: count * 2)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let expected = values.map { Float(Float16($0)) }
        let actual = readF16Buffer(decoded, count: count)
        let maxDiff = maxAbsDiff(expected, actual)
        return Outcome(
            name: "p5-private-blit",
            passed: maxDiff == 0,
            detail: String(format: "count=%d maxAbs=%.8g", count, maxDiff))
    }

    private static func repeatedDeterminism(
        file: AnimapkFile,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        kernels: AnimaKernels
    ) throws -> Outcome {
        let block = 0
        let suffix = "cross.q.weight"
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            return Outcome(name: "repeat-determinism", passed: false, detail: "missing spec")
        }
        let tokenCount = 8
        let planeStride = ANEW8DiTExecutor.alignedANEPlaneStrideElements(sequence: ANEW8NativePack.expectedSequenceLength)
        guard let tokenInput = device.makeBuffer(length: tokenCount * spec.columns * 2, options: .storageModeShared),
              let aneInput = device.makeBuffer(length: planeStride * spec.columns * 2, options: .storageModeShared),
              let aneOutput = device.makeBuffer(length: planeStride * spec.rows * 2, options: .storageModeShared),
              let tokenOutput = device.makeBuffer(length: ANEW8NativePack.expectedSequenceLength * spec.rows * 2, options: .storageModeShared),
              let encode = commandQueue.makeCommandBuffer() else {
            return Outcome(name: "repeat-determinism", passed: false, detail: "Metal allocation failed")
        }
        let values = makeInput(count: tokenCount * spec.columns, seed: 0xDE71)
        writeF16Buffer(values, to: tokenInput)
        try kernels.encodeTokenToANE(
            commandBuffer: encode,
            source: tokenInput,
            sourceOffsetBytes: 0,
            destination: aneInput,
            destinationOffsetBytes: 0,
            tokenCount: tokenCount,
            channels: spec.columns,
            planeStrideElements: planeStride)
        encode.commit()
        encode.waitUntilCompleted()

        let model = try ChurnWorker(file: file).makeProjection(block: block, suffix: suffix, label: "V7 deterministic")
        defer { model.invalidate() }
        var baseline: [Float]?
        var worst: Float = 0
        for _ in 0..<12 {
            try model.evaluateInput(aneInput, inputOffsetBytes: 0, output: aneOutput, outputOffsetBytes: 0)
            guard let decode = commandQueue.makeCommandBuffer() else {
                return Outcome(name: "repeat-determinism", passed: false, detail: "decode command buffer failed")
            }
            try kernels.encodeANEToToken(
                commandBuffer: decode,
                source: aneOutput,
                sourceOffsetBytes: 0,
                destination: tokenOutput,
                destinationOffsetBytes: 0,
                tokenCount: tokenCount,
                channels: spec.rows,
                planeStrideElements: planeStride)
            decode.commit()
            decode.waitUntilCompleted()
            let current = readF16Buffer(tokenOutput, count: tokenCount * spec.rows)
            if let baseline {
                worst = max(worst, maxAbsDiff(baseline, current))
            } else {
                baseline = current
            }
        }
        return Outcome(
            name: "repeat-determinism",
            passed: worst == 0,
            detail: String(format: "12 evaluations worstMaxAbs=%.8g", worst))
    }

    private static func sharedClientConcurrency(
        file: AnimapkFile,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        kernels: AnimaKernels
    ) throws -> Outcome {
        let worker = ChurnWorker(file: file)
        let stableBlock = 0
        let stableSuffix = "cross.q.weight"
        guard let spec = ANEW8NativePack.spec(suffix: stableSuffix) else {
            return Outcome(name: "shared-client-concurrency", passed: false, detail: "missing spec")
        }
        let tokenCount = 8
        let planeStride = ANEW8DiTExecutor.alignedANEPlaneStrideElements(sequence: ANEW8NativePack.expectedSequenceLength)
        guard let tokenInput = device.makeBuffer(length: tokenCount * spec.columns * 2, options: .storageModeShared),
              let aneInput = device.makeBuffer(length: planeStride * spec.columns * 2, options: .storageModeShared),
              let aneOutput = device.makeBuffer(length: planeStride * spec.rows * 2, options: .storageModeShared),
              let tokenOutput = device.makeBuffer(length: ANEW8NativePack.expectedSequenceLength * spec.rows * 2, options: .storageModeShared),
              let encode = commandQueue.makeCommandBuffer() else {
            return Outcome(name: "shared-client-concurrency", passed: false, detail: "Metal allocation failed")
        }
        let values = makeInput(count: tokenCount * spec.columns, seed: 0x51A1)
        writeF16Buffer(values, to: tokenInput)
        try kernels.encodeTokenToANE(
            commandBuffer: encode,
            source: tokenInput,
            sourceOffsetBytes: 0,
            destination: aneInput,
            destinationOffsetBytes: 0,
            tokenCount: tokenCount,
            channels: spec.columns,
            planeStrideElements: planeStride)
        encode.commit()
        encode.waitUntilCompleted()

        let stable = try worker.makeProjection(block: stableBlock, suffix: stableSuffix, label: "V7 foreground")
        defer { stable.invalidate() }

        func evaluateAndDecode() throws -> [Float] {
            try stable.evaluateInput(aneInput, inputOffsetBytes: 0, output: aneOutput, outputOffsetBytes: 0)
            guard let cb = commandQueue.makeCommandBuffer() else {
                throw AnimapkError.validation("V7 decode command buffer failed")
            }
            try kernels.encodeANEToToken(
                commandBuffer: cb,
                source: aneOutput,
                sourceOffsetBytes: 0,
                destination: tokenOutput,
                destinationOffsetBytes: 0,
                tokenCount: tokenCount,
                channels: spec.rows,
                planeStrideElements: planeStride)
            cb.commit()
            cb.waitUntilCompleted()
            return readF16Buffer(tokenOutput, count: tokenCount * spec.rows)
        }

        let baseline = try evaluateAndDecode()
        let churnQueue = DispatchQueue(label: "AnimaXS.V7.churn", qos: .userInitiated)
        let group = DispatchGroup()
        group.enter()
        var churnError: Error?
        churnQueue.async {
            defer { group.leave() }
            do {
                for i in 0..<24 {
                    let block = 10 + (i % 6)
                    let model = try worker.makeProjection(
                        block: block,
                        suffix: "cross.o.weight",
                        label: "V7 churn b\(block)")
                    model.invalidate()
                }
            } catch {
                churnError = error
            }
        }

        var worst: Float = 0
        var failedEval: String?
        for i in 0..<64 {
            do {
                let current = try evaluateAndDecode()
                worst = max(worst, maxAbsDiff(baseline, current))
                if !finite(current) {
                    failedEval = "foreground iteration \(i) non-finite"
                    break
                }
            } catch {
                failedEval = "foreground iteration \(i): \(error.localizedDescription)"
                break
            }
        }
        group.wait()
        if let churnError {
            failedEval = "background churn: \(churnError.localizedDescription)"
        }
        return Outcome(
            name: "shared-client-concurrency",
            passed: failedEval == nil && worst == 0,
            detail: failedEval ?? String(format: "64 foreground evals + 24 load/unload cycles worstMaxAbs=%.8g", worst))
    }

    @MainActor
    static func run() async -> String {
        var lines: [String] = [
            "ANE correctness microscope v7",
            "Diagnostic only; production inference is unchanged.",
            "Tests: layout -> W8 CPU reference -> P5 private blit -> determinism -> shared-client load/unload concurrency.",
            "",
        ]
        do {
            guard A12ANERuntime.runtimeStatus.available else {
                lines.append("SKIP: AppleNeuralEngine runtime unavailable")
                return lines.joined(separator: "\n")
            }
            guard let device = MTLCreateSystemDefaultDevice(),
                  let commandQueue = device.makeCommandQueue() else {
                lines.append("FAIL: Metal device/queue unavailable")
                return lines.joined(separator: "\n")
            }
            let kernels = try AnimaKernels(device: device)
            let file = try ProductionModelStore.shared.resolveDiTPackForDiagnostics()
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                lines.append("FAIL: prepared ANE cache missing; run/import native W8 pack first")
                return lines.joined(separator: "\n")
            }
            lines.append("pack=\(file.url.lastPathComponent) scheme=\(file.manifest.quantScheme ?? "unknown")")
            lines.append("runtime selectors: \(A12ANERuntime.runtimeStatus.selectorSummary)")
            lines.append("")

            let tests: [Outcome] = [
                try layoutRoundTrip(device: device, commandQueue: commandQueue, kernels: kernels),
                try projectionVsCPU(file: file, device: device, commandQueue: commandQueue, kernels: kernels),
                p5RoundTrip(device: device, commandQueue: commandQueue),
                try repeatedDeterminism(file: file, device: device, commandQueue: commandQueue, kernels: kernels),
                try sharedClientConcurrency(file: file, device: device, commandQueue: commandQueue, kernels: kernels),
            ]
            lines.append(contentsOf: tests.map(\.line))
            lines.append("")
            if let firstFailure = tests.first(where: { !$0.passed }) {
                lines.append("FIRST BAD BOUNDARY: \(firstFailure.name)")
                switch firstFailure.name {
                case "layout-roundtrip":
                    lines.append("Interpretation: token<->ANE bridge/layout is corrupt before private ANE math.")
                case "w8-vs-cpu":
                    lines.append("Interpretation: native W8 projection semantics/layout disagree with the pack reference.")
                case "p5-private-blit":
                    lines.append("Interpretation: private cache storage/blit path corrupts exact K/V reuse.")
                case "repeat-determinism":
                    lines.append("Interpretation: repeated evaluation of one loaded model is not deterministic.")
                case "shared-client-concurrency":
                    lines.append("Interpretation: overlapping private load/unload with foreground evaluate is unsafe; serialize ANE client operations before touching math/P5.")
                default:
                    break
                }
            } else {
                lines.append("ALL V7 MICRO-BOUNDARIES PASSED")
                lines.append("Next target: compare a complete production DiT block ANE vs dequantized-MPS intermediate tensors, because corruption is above the isolated primitives.")
            }
        } catch {
            lines.append("FAIL probe threw: \(error.localizedDescription)")
        }
        return lines.joined(separator: "\n")
    }
}
