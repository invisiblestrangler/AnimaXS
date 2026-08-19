import Foundation
import Metal

/// Single-file A12 capture used by the ANE Oracle E parity lane.
///
/// File format (`.oraclee`):
/// - bytes 0...7: ASCII `AXOECAP1`
/// - UInt32 LE version at byte 8
/// - UInt64 LE manifest offset at byte 16
/// - UInt64 LE manifest length at byte 24
/// - raw payloads from byte 64 onward
/// - JSON manifest at the recorded tail offset
///
/// Payloads include the exact device initial latent, exact device adapter cross
/// context, the exact step-0 prepared residual/embedding/AdaLN-LoRA state, and
/// a deterministic 65,536-value FP32 sample after self/cross/MLP for all 28 DiT
/// blocks at diffusion step 0. The diagnostic deliberately stops immediately
/// after block 27 MLP: no final layer, Euler update, later diffusion steps, or
/// VAE are needed to localize A12-vs-Oracle-E divergence.
final class OracleEDeviceCaptureWriter {
    static let sampleCount = 65_536
    static let residualElements = 1_024 * 2_048
    static let headerBytes = 64
    static let requiredCompletedPayloadNames: Set<String> = [
        "initial_latent",
        "cross_context",
        "prepared_residual",
        "prepared_embedding",
        "prepared_adaln_lora",
    ]

    struct PayloadRecord: Codable {
        let name: String
        let offset: UInt64
        let byteCount: Int
        let elementCount: Int
        let dtype: String
        let shape: [Int]
    }

    struct SampleStats: Codable {
        let finiteCount: Int
        let nanCount: Int
        let posInfCount: Int
        let negInfCount: Int
        let min: Float?
        let max: Float?
        let mean: Double?
        let std: Double?
        let l2: Double?
        let maxAbs: Float?
    }

    struct CheckpointRecord: Codable {
        let step: Int
        let block: Int
        let branch: String
        let offset: UInt64
        let byteCount: Int
        let sampleCount: Int
        let sampleStride: Int
        let sampleDtype: String
        let residualElements: Int
        let sampleStats: SampleStats
    }

    struct Manifest: Codable {
        let schema: Int
        let producer: String
        let status: String
        let error: String?
        let createdUTC: String
        let seed: UInt64
        let ditVariantID: String
        let linearBackend: String
        let pingPongWeightStreaming: Bool
        let numericalMonitoring: Bool
        let conditioningSource: String
        let initialLatentSource: String
        let preparedStateSource: String
        let sampleCountTarget: Int
        let sampling: String
        let expectedStep0Checkpoints: Int
        let completedStep0Checkpoints: Int
        let payloads: [PayloadRecord]
        let checkpoints: [CheckpointRecord]
    }

    private let url: URL
    private let handle: FileHandle
    private let seed: UInt64
    private let ditVariantID: String
    private let optimization: InferenceOptimizationConfig
    private let createdUTC: String
    private var payloads: [PayloadRecord] = []
    private var checkpoints: [CheckpointRecord] = []
    private var didFinish = false

    init(
        url: URL,
        seed: UInt64,
        ditVariantID: String,
        optimization: InferenceOptimizationConfig
    ) throws {
        self.url = url
        self.seed = seed
        self.ditVariantID = ditVariantID
        self.optimization = optimization
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        self.createdUTC = formatter.string(from: Date())

        let fm = FileManager.default
        try? fm.removeItem(at: url)
        guard fm.createFile(atPath: url.path, contents: nil) else {
            throw AnimapkError.validation("failed to create Oracle E device capture file")
        }
        self.handle = try FileHandle(forWritingTo: url)

        var header = Data(repeating: 0, count: Self.headerBytes)
        let magic = Array("AXOECAP1".utf8)
        header.replaceSubrange(0..<magic.count, with: magic)
        var version = UInt32(1).littleEndian
        withUnsafeBytes(of: &version) { bytes in
            header.replaceSubrange(8..<12, with: bytes)
        }
        handle.write(header)
    }

    deinit {
        if !didFinish {
            try? handle.close()
        }
    }

    func appendPayload(
        name: String,
        buffer: MTLBuffer,
        elementCount: Int,
        shape: [Int]
    ) throws {
        guard !payloads.contains(where: { $0.name == name }) else {
            throw AnimapkError.validation("duplicate Oracle E payload \(name)")
        }
        guard elementCount > 0,
              shape.allSatisfy({ $0 > 0 }),
              shape.reduce(1, *) == elementCount else {
            throw AnimapkError.validation("Oracle E payload \(name) has invalid shape")
        }
        let byteCount = elementCount * MemoryLayout<Float>.stride
        guard buffer.length >= byteCount else {
            throw AnimapkError.validation("Oracle E payload \(name) buffer is too small")
        }
        let offset = handle.offsetInFile
        handle.write(Data(bytes: buffer.contents(), count: byteCount))
        payloads.append(PayloadRecord(
            name: name,
            offset: offset,
            byteCount: byteCount,
            elementCount: elementCount,
            dtype: "float32-le",
            shape: shape))
    }

    func captureBranch(step: Int, block: Int, branch: String, residual: MTLBuffer) throws {
        guard step == 0 else { return }
        guard (0..<28).contains(block), ["self", "cross", "mlp"].contains(branch) else {
            throw AnimapkError.validation(
                "invalid Oracle E checkpoint step=\(step) block=\(block) branch=\(branch)")
        }
        let requiredBytes = Self.residualElements * MemoryLayout<Float>.stride
        guard residual.length >= requiredBytes else {
            throw AnimapkError.validation("Oracle E residual buffer is too small")
        }
        let keyAlreadyExists = checkpoints.contains { record in
            record.step == step && record.block == block && record.branch == branch
        }
        guard !keyAlreadyExists else {
            throw AnimapkError.validation(
                "duplicate Oracle E checkpoint block=\(block) branch=\(branch)")
        }

        let stride = max(1, Self.residualElements / Self.sampleCount)
        let pointer = residual.contents().bindMemory(
            to: Float.self, capacity: Self.residualElements)
        var sample = [Float]()
        sample.reserveCapacity(Self.sampleCount)

        var finiteCount = 0
        var nanCount = 0
        var posInfCount = 0
        var negInfCount = 0
        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude
        var maxAbs: Float = 0
        var sum: Double = 0
        var sumSquares: Double = 0

        var index = 0
        while index < Self.residualElements && sample.count < Self.sampleCount {
            let value = pointer[index]
            sample.append(value)
            if value.isFinite {
                finiteCount += 1
                minimum = Swift.min(minimum, value)
                maximum = Swift.max(maximum, value)
                maxAbs = Swift.max(maxAbs, abs(value))
                let d = Double(value)
                sum += d
                sumSquares += d * d
            } else if value.isNaN {
                nanCount += 1
            } else if value.sign == .plus {
                posInfCount += 1
            } else {
                negInfCount += 1
            }
            index += stride
        }

        let stats: SampleStats
        if finiteCount > 0 {
            let mean = sum / Double(finiteCount)
            let variance = Swift.max(0, sumSquares / Double(finiteCount) - mean * mean)
            stats = SampleStats(
                finiteCount: finiteCount,
                nanCount: nanCount,
                posInfCount: posInfCount,
                negInfCount: negInfCount,
                min: minimum,
                max: maximum,
                mean: mean,
                std: sqrt(variance),
                l2: sqrt(sumSquares),
                maxAbs: maxAbs)
        } else {
            stats = SampleStats(
                finiteCount: 0,
                nanCount: nanCount,
                posInfCount: posInfCount,
                negInfCount: negInfCount,
                min: nil, max: nil, mean: nil, std: nil, l2: nil, maxAbs: nil)
        }

        let offset = handle.offsetInFile
        sample.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress {
                handle.write(Data(bytes: base, count: bytes.count))
            }
        }
        checkpoints.append(CheckpointRecord(
            step: step,
            block: block,
            branch: branch,
            offset: offset,
            byteCount: sample.count * MemoryLayout<Float>.stride,
            sampleCount: sample.count,
            sampleStride: stride,
            sampleDtype: "float32-le",
            residualElements: Self.residualElements,
            sampleStats: stats))
    }

    @discardableResult
    func finish(status: String, error: String? = nil) throws -> URL {
        guard !didFinish else { return url }
        if status == "completed" {
            guard checkpoints.count == 84 else {
                throw AnimapkError.validation(
                    "Oracle E capture incomplete: \(checkpoints.count)/84 checkpoints")
            }
            let payloadNames = Set(payloads.map(\.name))
            let missing = Self.requiredCompletedPayloadNames.subtracting(payloadNames)
            guard missing.isEmpty else {
                throw AnimapkError.validation(
                    "Oracle E capture missing payloads: \(missing.sorted().joined(separator: ", "))")
            }
        }
        let manifest = Manifest(
            schema: 1,
            producer: "AnimaXS A12 Oracle E device capture",
            status: status,
            error: error,
            createdUTC: createdUTC,
            seed: seed,
            ditVariantID: ditVariantID,
            linearBackend: optimization.linearBackend.rawValue,
            pingPongWeightStreaming: optimization.pingPongWeightStreaming,
            numericalMonitoring: optimization.numericalMonitoring,
            conditioningSource: "device Qwen + device LLM adapter FP32 cross-context",
            initialLatentSource: "device SeededRNG(seed:)",
            preparedStateSource: "device DiTPreparationExecutor at diffusion step 0",
            sampleCountTarget: Self.sampleCount,
            sampling: "flatten token-major FP32 residual; stride=max(1,numel//65536); first 65536 values",
            expectedStep0Checkpoints: 84,
            completedStep0Checkpoints: checkpoints.count,
            payloads: payloads,
            checkpoints: checkpoints)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let manifestData = try encoder.encode(manifest)
        let manifestOffset = handle.offsetInFile
        handle.write(manifestData)
        let manifestLength = UInt64(manifestData.count)

        handle.seek(toFileOffset: 16)
        var offsetLE = manifestOffset.littleEndian
        withUnsafeBytes(of: &offsetLE) { handle.write(Data($0)) }
        var lengthLE = manifestLength.littleEndian
        withUnsafeBytes(of: &lengthLE) { handle.write(Data($0)) }
        try handle.synchronize()
        try handle.close()
        didFinish = true
        return url
    }
}

private enum OracleEDeviceCaptureControl: Error {
    case step0Complete
}

/// Diagnostic-only runner. It intentionally does not go through the production
/// GenerationCoordinator and therefore cannot interfere with an active user
/// generation. Callers must ensure no generation is running before invoking it.
///
/// Correctness configuration is fixed here rather than read from persisted UI
/// toggles: ANE hybrid W8, one synchronous weight slot (ping-pong OFF), legacy
/// known-good Metal boundaries, risky fused/no-copy/direct-QGEMM paths OFF.
struct OracleEDeviceCaptureRunner {
    let context: MetalContext

    func run(
        prompt: String,
        seed: UInt64,
        models: ResolvedModels,
        outputURL: URL
    ) async throws -> URL {
        var optimization = InferenceOptimizationConfig.currentBaseline
        optimization.linearBackend = .aneHybridW8
        optimization.pingPongWeightStreaming = false
        optimization.fusedNormModulation = false
        optimization.fusedMLPActivation = false
        optimization.stridedTokenMajorAttention = false
        optimization.noCopyWeightSource = false
        optimization.attentionBackend = .legacyHeadMajorMPS

        let numerics = DiTNumericsPolicy.fromVariantID(models.dit.variant.id)
        if let reason = InferenceOptimizationConfig.blockingReason(
            for: optimization, numerics: numerics, ditVariantID: models.dit.variant.id) {
            throw AnimapkError.validation(reason)
        }

        let writer = try OracleEDeviceCaptureWriter(
            url: outputURL,
            seed: seed,
            ditVariantID: models.dit.variant.id,
            optimization: optimization)

        do {
            let qwenTokenizer = try TokenizerLoader.qwen()
            let qwenIDs = qwenTokenizer.encode(text: prompt, addSpecialTokens: false)
            guard !qwenIDs.isEmpty else {
                throw GenerationError.tokenizer("Qwen tokenizer produced no tokens")
            }
            let t5Tokenizer = try TokenizerLoader.t5()
            let t5IDs = t5Tokenizer.encode(text: prompt, addSpecialTokens: false) + [1]
            let t5Weights = [Float](repeating: 1.0, count: t5IDs.count)

            guard let qwenOutput = context.device.makeBuffer(
                length: qwenIDs.count * QwenEncoderMetal.hidden * 4,
                options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate Oracle E Qwen output")
            }
            do {
                let encoder = try QwenEncoderMetal(
                    context: context, file: try AnimapkFile(url: models.textEncoder.url))
                try await encoder.execute(
                    tokenIDs: qwenIDs, output: qwenOutput, layerCompleted: nil)
                withExtendedLifetime(encoder) {}
            }

            guard let cross = context.device.makeBuffer(
                length: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden * 4,
                options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate Oracle E cross context")
            }
            do {
                let adapter = try LLMAdapterMetal(
                    context: context, file: try AnimapkFile(url: models.dit.url))
                try await adapter.execute(
                    qwenContext: qwenOutput,
                    contextTokens: qwenIDs.count,
                    t5IDs: t5IDs,
                    t5Weights: t5Weights,
                    output: cross,
                    layerCompleted: nil)
                withExtendedLifetime(adapter) {}
            }

            let latentCount = DiffusionSampler.latentElements
            guard let initialLatent = context.device.makeBuffer(
                length: latentCount * 4, options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate Oracle E initial latent")
            }
            var rng = SeededRNG(seed: seed)
            let initialPointer = initialLatent.contents().bindMemory(
                to: Float.self, capacity: latentCount)
            for index in 0..<latentCount {
                initialPointer[index] = rng.nextNormal()
            }

            try writer.appendPayload(
                name: "initial_latent",
                buffer: initialLatent,
                elementCount: latentCount,
                shape: [1, 16, 64, 64])
            try writer.appendPayload(
                name: "cross_context",
                buffer: cross,
                elementCount: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden,
                shape: [1, 512, 1_024])

            let ditFile = try AnimapkFile(url: models.dit.url)
            _ = try ANEW8ModelPreparer.ensurePrepared(file: ditFile)
            let sampler = try DiffusionSampler(
                context: context,
                file: ditFile,
                optimization: optimization,
                numerics: numerics)
            guard let unusedOutput = context.device.makeBuffer(
                length: latentCount * 4, options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate Oracle E sampler output")
            }

            do {
                try await sampler.executeDiagnostic(
                    initialLatent: initialLatent,
                    crossContext: cross,
                    outputLatent: unusedOutput,
                    startStep: 0,
                    diagnosticStepPrepared: { step, residual, embedding, adalnLora, _, _ in
                        guard step == 0 else { return }
                        try writer.appendPayload(
                            name: "prepared_residual",
                            buffer: residual,
                            elementCount: DiTPreparationExecutor.tokens * DiTPreparationExecutor.hidden,
                            shape: [DiTPreparationExecutor.tokens, DiTPreparationExecutor.hidden])
                        try writer.appendPayload(
                            name: "prepared_embedding",
                            buffer: embedding,
                            elementCount: DiTPreparationExecutor.hidden,
                            shape: [DiTPreparationExecutor.hidden])
                        try writer.appendPayload(
                            name: "prepared_adaln_lora",
                            buffer: adalnLora,
                            elementCount: DiTPreparationExecutor.adaln,
                            shape: [DiTPreparationExecutor.adaln])
                    },
                    diagnosticBranchFilter: { step, _ in step == 0 },
                    diagnosticBranchCompleted: { step, block, branch, residual in
                        try writer.captureBranch(
                            step: step, block: block, branch: branch, residual: residual)
                        if step == 0 && block == 27 && branch == "mlp" {
                            throw OracleEDeviceCaptureControl.step0Complete
                        }
                    })
                throw AnimapkError.validation(
                    "Oracle E diagnostic unexpectedly completed full diffusion")
            } catch OracleEDeviceCaptureControl.step0Complete {
                // Expected fast stop: all 84 step-0 branch checkpoints exist.
            }

            withExtendedLifetime(sampler) {}
            return try writer.finish(status: "completed")
        } catch {
            _ = try? writer.finish(status: "failed", error: error.localizedDescription)
            throw error
        }
    }
}
