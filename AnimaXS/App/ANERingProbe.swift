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
        var line: String { "\(passed ? \"PASS\" : \"FAIL\") \(name): \(detail)" }
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
                spatial: UInt(spec.spatial),
                label: label,
                cacheKey: key)
        }
    }

    private final class ErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [String] = []
        func add(_ error: Error) {
            lock.lock(); messages.append(String(describing: error)); lock.unlock()
        }
        var summary: String {
            lock.lock(); defer { lock.unlock() }
            return messages.joined(separator: " | ")
        }
        var isEmpty: Bool {
            lock.lock(); defer { lock.unlock() }
            return messages.isEmpty
        }
    }

    static func run() async -> String {
        var lines: [String] = [
            "ANE correctness microscope v7",
            "Purpose: isolate checker/woven output and late NaN/Inf without a full diffusion run.",
            "Production inference is NOT modified by this probe."
        ]
        var outcomes: [Outcome] = []

        do {
            guard let context = MetalContext() else {
                throw AnimapkError.validation("Metal unavailable")
            }
            guard let ditEntry = ModelManifest.entries.first(where: { $0.component == .dit }) else {
                throw AnimapkError.validation("DiT manifest entry missing")
            }
            let store = try ModelStore()
            let ditURL = await store.localURL(for: ditEntry)
            guard FileManager.default.fileExists(atPath: ditURL.path) else {
                throw AnimapkError.validation("installed DiT pack missing")
            }
            let file = try AnimapkFile(url: ditURL)
            try ANEW8NativePack.validate(file: file)
            guard ANEW8ModelPreparer.isPrepared(file: file) else {
                throw AnimapkError.validation("ANE prepared-model cache is incomplete")
            }
            guard A12ANEIsAvailable() else {
                throw AnimapkError.validation(
                    "A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
            }

            lines.append("runtime: \(A12ANERuntimeStatus())")
            lines.append("client: \(A12ANEClientCapabilitySummary())")
            lines.append(
                "pack: \(ditURL.lastPathComponent) scheme=\(file.quantScheme) " +
                "prepared=\(ANEW8NativePack.expectedPreparedModelCount)")
            lines.append("")

            outcomes.append(try layoutRoundTrip(context: context))
            outcomes.append(try projectionReference(context: context, file: file))
            outcomes.append(try p5PrivateBlitRoundTrip(context: context))
            outcomes.append(try repeatDeterminism(context: context, file: file))
            outcomes.append(try sharedClientConcurrency(context: context, file: file))
        } catch {
            lines.append("FATAL SETUP ERROR: \(error)")
        }

        if !outcomes.isEmpty {
            lines.append("")
            lines.append("V7 CORRECTNESS SCORECARD")
            lines.append(contentsOf: outcomes.map(\.line))
            lines.append("")
            if let firstFailure = outcomes.first(where: { !$0.passed }) {
                lines.append("FIRST FAILING BOUNDARY: \(firstFailure.name)")
            } else {
                lines.append("ALL MICRO-BOUNDARIES PASSED")
                lines.append(
                    "If the image is still woven, the next probe must compare a real " +
                    "ANE block residual against the legacy/CPU oracle; do not blame VAE processOut.")
            }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Test 1: exact production layout kernels

    private static func layoutRoundTrip(context: MetalContext) throws -> Outcome {
        let rows = 37
        let channels = 19
        let planeStrideElements = ((rows * 2 + 63) & ~63) / 2
        let count = rows * channels
        guard let token = context.device.makeBuffer(
                length: count * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let ane = context.device.makeBuffer(
                length: channels * planeStrideElements * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let roundTrip = context.device.makeBuffer(
                length: count * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("V7 layout buffers unavailable")
        }

        let input = token.contents().bindMemory(to: Float16.self, capacity: count)
        for index in 0..<count {
            let value = Float(((index * 37) % 251) - 125) / 31.0
            input[index] = Float16(value)
        }
        memset(ane.contents(), 0xA5, ane.length)
        memset(roundTrip.contents(), 0, roundTrip.length)

        try encodeLayout(
            context: context, command: command, kernel: "dit_token_to_ane_f16",
            source: token, destination: ane, rows: rows, channels: channels,
            planeStrideElements: planeStrideElements)
        try encodeLayout(
            context: context, command: command, kernel: "dit_ane_to_token_f16",
            source: ane, destination: roundTrip, rows: rows, channels: channels,
            planeStrideElements: planeStrideElements)
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }

        let output = roundTrip.contents().bindMemory(to: Float16.self, capacity: count)
        var mismatches = 0
        for index in 0..<count where output[index].bitPattern != input[index].bitPattern {
            mismatches += 1
        }
        return Outcome(
            name: "layout token<->ANE",
            passed: mismatches == 0,
            detail: "rows=\(rows) channels=\(channels) stride=\(planeStrideElements) mismatches=\(mismatches)/\(count)")
    }

    private static func encodeLayout(
        context: MetalContext,
        command: MTLCommandBuffer,
        kernel: String,
        source: MTLBuffer,
        destination: MTLBuffer,
        rows: Int,
        channels: Int,
        planeStrideElements: Int
    ) throws {
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("V7 failed to make \(kernel) encoder")
        }
        var r = UInt32(rows)
        var c = UInt32(channels)
        var stride = UInt32(planeStrideElements)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        encoder.setBytes(&r, length: 4, index: 2)
        encoder.setBytes(&c, length: 4, index: 3)
        encoder.setBytes(&stride, length: 4, index: 4)
        let total = rows * channels
        let width = min(max(1, total), pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Test 2: W8 projection math/orientation

    private static func projectionReference(
        context: MetalContext, file: AnimapkFile
    ) throws -> Outcome {
        let block = 0
        let suffix = "cross_attn.q_proj.weight"
        guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
            throw AnimapkError.validation("V7 crossQ spec missing")
        }
        let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
        let native = try ANEW8NativePack.nativeWeight(
            file: file, tensor: tensor,
            expectedRows: spec.rows, expectedColumns: spec.columns)
        guard let digest = tensor.blobSHA256 else {
            throw AnimapkError.validation("V7 crossQ digest missing")
        }
        let key = ANEW8NativePack.projectionCacheKey(
            block: block, tag: spec.tag, hash: digest)
        let model = try A12ANEProjectionModel(
            preparedInputChannels: UInt(spec.columns),
            outputChannels: UInt(spec.rows),
            spatial: UInt(spec.spatial),
            label: "v7_reference_b0_cross_q",
            cacheKey: key)
        defer { model.invalidate() }

        let input = try A12ANESurface(
            device: context.device, channels: UInt(spec.columns), spatial: UInt(spec.spatial))
        let output = try A12ANESurface(
            device: context.device, channels: UInt(spec.rows), spatial: UInt(spec.spatial))
        memset(input.metalBuffer.contents(), 0, input.byteCount)
        memset(output.metalBuffer.contents(), 0, output.byteCount)

        let activeInput = 17
        let inStride = Int(input.planeStrideElements)
        let inputHalf = input.metalBuffer.contents().bindMemory(
            to: Float16.self, capacity: input.byteCount / 2)
        for spatial in 0..<spec.spatial {
            let value = Float((spatial % 9) - 4) / 4.0
            inputHalf[activeInput * inStride + spatial] = Float16(value)
        }

        var evaluationMS = 0.0
        _ = try model.evaluateInput(input, output: output, milliseconds: &evaluationMS)

        let q = native.q.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
        let scale = native.scaleF32.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let bias = native.biasF32.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard q.count == spec.rows * spec.columns,
              scale.count == spec.rows, bias.count == spec.rows else {
            throw AnimapkError.validation("V7 native W8 payload shape mismatch")
        }

        let outStride = Int(output.planeStrideElements)
        let actual = output.metalBuffer.contents().bindMemory(
            to: Float16.self, capacity: output.byteCount / 2)
        let sampleSpatial = [0, 1, 7, 31, spec.spatial / 2, spec.spatial - 1]
        var rowSquare = 0.0
        var transposeSquare = 0.0
        var rowMax = 0.0
        var samples = 0

        for row in stride(from: 0, to: min(spec.rows, 512), by: 7) {
            let rowWeight = Float(q[row * spec.columns + activeInput]) * scale[row] + bias[row]
            let transposeWeight =
                Float(q[activeInput * spec.columns + row]) * scale[row] + bias[row]
            for spatial in sampleSpatial {
                let x = Float(inputHalf[activeInput * inStride + spatial])
                let y = Float(actual[row * outStride + spatial])
                let rowError = Double(y - x * rowWeight)
                let transposeError = Double(y - x * transposeWeight)
                rowSquare += rowError * rowError
                transposeSquare += transposeError * transposeError
                rowMax = max(rowMax, abs(rowError))
                samples += 1
            }
        }

        let rowRMSE = sqrt(rowSquare / Double(max(1, samples)))
        let transposeRMSE = sqrt(transposeSquare / Double(max(1, samples)))
        let pass = rowRMSE < 0.01 && rowRMSE <= transposeRMSE
        return Outcome(
            name: "W8 projection reference",
            passed: pass,
            detail: String(
                format: "eval=%.2fms rowMajorRMSE=%.6g rowMax=%.6g transposeRMSE=%.6g samples=%d",
                evaluationMS, rowRMSE, rowMax, transposeRMSE, samples))
    }

    // MARK: - Test 3: real private P5 buffer copy path

    private static func p5PrivateBlitRoundTrip(context: MetalContext) throws -> Outcome {
        guard let cache = CrossKVCache(device: context.device) else {
            return Outcome(
                name: "P5 private-buffer blit",
                passed: false,
                detail: "CrossKVCache allocation failed")
        }
        let bytes = CrossKVCache.tensorBytes
        guard let source = context.device.makeBuffer(
                length: bytes, options: .storageModeShared),
              let destination = context.device.makeBuffer(
                length: bytes, options: .storageModeShared),
              let command = context.commandQueue.makeCommandBuffer(),
              let blit = command.makeBlitCommandEncoder() else {
            throw AnimapkError.validation("V7 P5 blit buffers unavailable")
        }
        let src = source.contents().bindMemory(to: UInt8.self, capacity: bytes)
        for index in 0..<bytes {
            src[index] = UInt8(truncatingIfNeeded: index &* 131 &+ 17)
        }
        memset(destination.contents(), 0, bytes)
        blit.copy(
            from: source, sourceOffset: 0,
            to: cache.buffer, destinationOffset: cache.kOffset(block: 13), size: bytes)
        blit.copy(
            from: cache.buffer, sourceOffset: cache.kOffset(block: 13),
            to: destination, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }

        let dst = destination.contents().bindMemory(to: UInt8.self, capacity: bytes)
        var mismatches = 0
        for index in 0..<bytes where src[index] != dst[index] {
            mismatches += 1
            if mismatches > 1024 { break }
        }
        return Outcome(
            name: "P5 private-buffer blit",
            passed: mismatches == 0,
            detail: "bytes=\(bytes) mismatches=\(mismatches)")
    }

    // MARK: - Test 4: same model/input must be deterministic

    private static func repeatDeterminism(
        context: MetalContext, file: AnimapkFile
    ) throws -> Outcome {
        let worker = ChurnWorker(file: file)
        let model = try worker.makeProjection(
            block: 0, suffix: "cross_attn.q_proj.weight", label: "v7_repeat_cross_q")
        defer { model.invalidate() }
        let input = try A12ANESurface(
            device: context.device,
            channels: UInt(DiTBlockExecutor.dim),
            spatial: UInt(DiTBlockExecutor.tokens))
        let output = try A12ANESurface(
            device: context.device,
            channels: UInt(DiTBlockExecutor.dim),
            spatial: UInt(DiTBlockExecutor.tokens))
        fillDeterministicSurface(input)

        var ms = 0.0
        _ = try model.evaluateInput(input, output: output, milliseconds: &ms)
        let baseline = Data(bytes: output.metalBuffer.contents(), count: output.byteCount)
        var changedRuns = 0
        var firstMismatch = -1
        var totalMS = ms

        for _ in 0..<8 {
            var value = 0.0
            _ = try model.evaluateInput(input, output: output, milliseconds: &value)
            totalMS += value
            let now = Data(bytes: output.metalBuffer.contents(), count: output.byteCount)
            if now != baseline {
                changedRuns += 1
                if firstMismatch < 0 {
                    firstMismatch = firstDifferentByte(baseline, now)
                }
            }
        }
        return Outcome(
            name: "repeat ANE determinism",
            passed: changedRuns == 0,
            detail: String(
                format: "9 evals totalANE=%.2fms changedRuns=%d firstByte=%d",
                totalMS, changedRuns, firstMismatch))
    }

    // MARK: - Test 5: production's shared-client overlap assumption

    private static func sharedClientConcurrency(
        context: MetalContext, file: AnimapkFile
    ) throws -> Outcome {
        let worker = ChurnWorker(file: file)
        let foreground = try worker.makeProjection(
            block: 0, suffix: "cross_attn.q_proj.weight", label: "v7_fg_cross_q")
        defer { foreground.invalidate() }
        let input = try A12ANESurface(
            device: context.device,
            channels: UInt(DiTBlockExecutor.dim),
            spatial: UInt(DiTBlockExecutor.tokens))
        let output = try A12ANESurface(
            device: context.device,
            channels: UInt(DiTBlockExecutor.dim),
            spatial: UInt(DiTBlockExecutor.tokens))
        fillDeterministicSurface(input)

        var initialMS = 0.0
        _ = try foreground.evaluateInput(input, output: output, milliseconds: &initialMS)
        let baseline = Data(bytes: output.metalBuffer.contents(), count: output.byteCount)

        let queue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-v7-churn",
            qos: .userInitiated)
        let errors = ErrorBox()
        var changedRuns = 0
        var firstMismatch = -1
        var foregroundMS = initialMS

        for iteration in 0..<12 {
            let group = DispatchGroup()
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    let background = try worker.makeProjection(
                        block: 1 + (iteration % 4),
                        suffix: "cross_attn.q_proj.weight",
                        label: "v7_churn_\(iteration)")
                    background.invalidate()
                } catch {
                    errors.add(error)
                }
            }

            var value = 0.0
            _ = try foreground.evaluateInput(input, output: output, milliseconds: &value)
            foregroundMS += value
            group.wait()

            let now = Data(bytes: output.metalBuffer.contents(), count: output.byteCount)
            if now != baseline {
                changedRuns += 1
                if firstMismatch < 0 {
                    firstMismatch = firstDifferentByte(baseline, now)
                }
            }
        }

        let pass = changedRuns == 0 && errors.isEmpty
        let errorSuffix = errors.isEmpty ? "" : " churnErrors=\(errors.summary)"
        return Outcome(
            name: "shared-client load/unload overlap",
            passed: pass,
            detail: String(
                format: "12 overlapped evals fgANE=%.2fms changedRuns=%d firstByte=%d",
                foregroundMS, changedRuns, firstMismatch) + errorSuffix)
    }

    private static func fillDeterministicSurface(_ surface: A12ANESurface) {
        memset(surface.metalBuffer.contents(), 0, surface.byteCount)
        let stride = Int(surface.planeStrideElements)
        let ptr = surface.metalBuffer.contents().bindMemory(
            to: Float16.self, capacity: surface.byteCount / 2)
        let channels = min(Int(surface.channels), 64)
        let spatial = Int(surface.spatial)
        for channel in 0..<channels {
            for position in 0..<spatial {
                let raw = ((channel * 29 + position * 17) % 127) - 63
                ptr[channel * stride + position] = Float16(Float(raw) / 64.0)
            }
        }
    }

    private static func firstDifferentByte(_ a: Data, _ b: Data) -> Int {
        let count = min(a.count, b.count)
        return a.withUnsafeBytes { aBytes in
            b.withUnsafeBytes { bBytes in
                let ap = aBytes.bindMemory(to: UInt8.self)
                let bp = bBytes.bindMemory(to: UInt8.self)
                for index in 0..<count where ap[index] != bp[index] {
                    return index
                }
                return a.count == b.count ? -1 : count
            }
        }
    }
}
