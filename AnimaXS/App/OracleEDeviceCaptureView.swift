import SwiftUI
import Metal

extension View {
    /// Keeps DEBUG-only Oracle E UI out of production while giving AnimaXSApp
    /// one unconditional modifier call that parses identically in every build.
    @ViewBuilder
    func oracleEDebugOverlay() -> some View {
        #if DEBUG
        self.overlay(alignment: .topLeading) {
            OracleEDeviceCaptureLauncher()
                .padding(.top, 8)
                .padding(.leading, 12)
        }
        #else
        self
        #endif
    }
}

#if DEBUG
/// Debug-only launcher for the ANE Oracle E physical-device parity capture.
/// It is intentionally separate from production Generate/Diagnostics state so
/// the experiment can be removed cleanly after A12 localization is complete.
struct OracleEDeviceCaptureLauncher: View {
    @State private var presented = false

    var body: some View {
        Button {
            presented = true
        } label: {
            Label("Oracle E", systemImage: "waveform.path.ecg")
                .font(.caption.bold())
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Open Oracle E device capture")
        .sheet(isPresented: $presented) {
            NavigationStack {
                OracleEDeviceCaptureSheet()
            }
        }
    }
}

@MainActor
private struct OracleEDeviceCaptureSheet: View {
    /// Exact positive prompt from the pinned Oracle-V2 control configuration.
    private static let oraclePrompt = "masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, blue eyes, black witch hat, dark blue robe, holding a glowing lantern, moonlit forest, fireflies, detailed anime illustration"
    private static let oracleSeed: UInt64 = 123_456_789

    @Environment(\.dismiss) private var dismiss
    @State private var models: ResolvedModels?
    @State private var modelStatus = "Checking installed packs…"
    @State private var captureStatus = "Idle"
    @State private var captureURL: URL?
    @State private var modulationProbeURL: URL?
    @State private var isRunning = false

    var body: some View {
        Form {
            Section("Pinned control") {
                LabeledContent("Seed", value: String(Self.oracleSeed))
                Text(Self.oraclePrompt)
                    .font(.caption)
                    .textSelection(.enabled)
                Text("Runs exactly one DiT traversal (step 0) and stops after block 27 MLP. No VAE decode and no later diffusion steps.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Pack gate") {
                Text(modelStatus)
                    .font(.caption)
                if let models {
                    LabeledContent("DiT variant", value: models.dit.variant.id)
                    LabeledContent("DiT pack", value: models.dit.variant.displayFilename)
                }
                Button("Refresh local packs") {
                    Task { await refreshModels() }
                }
                .disabled(isRunning)
            }

            Section("A12 capture") {
                Button(isRunning ? "Capturing…" : "Capture Oracle E step 0") {
                    Task { await runCapture() }
                }
                .disabled(isRunning || models?.dit.variant.id != ModelManifest.ditW8ANEV1.id)

                Text(captureStatus)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)

                if let captureURL {
                    ShareLink(item: captureURL) {
                        Label("Share .oraclee capture", systemImage: "square.and.arrow.up")
                    }
                    .font(.caption)
                }

                if let modulationProbeURL {
                    ShareLink(item: modulationProbeURL) {
                        Label("Share Oracle E2 modulation probe", systemImage: "square.and.arrow.up")
                    }
                    .font(.caption)
                }
            }

            Section("Safety") {
                Text("Do not start this while a normal generation is running. The capture uses the same ANE/Metal resources and is deliberately a standalone diagnostic lane.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                Text("The outputs contain only activations, conditioning, initial noise and metadata; they never copy model weights.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Oracle E · A12")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Close") { dismiss() }
                    .disabled(isRunning)
            }
        }
        .interactiveDismissDisabled(isRunning)
        .task { await refreshModels() }
    }

    private func refreshModels() async {
        guard !isRunning else { return }
        do {
            let store = try ModelStore()
            let resolved = try await store.resolveInstalledModels()
            models = resolved
            if resolved.dit.variant.id == ModelManifest.ditW8ANEV1.id {
                modelStatus = "Ready: exact ANE-native W8 pack is installed."
            } else {
                modelStatus = "Blocked: import the w8-ane-v1 DiT pack before running Oracle E."
            }
        } catch {
            models = nil
            modelStatus = "Blocked: \(error.localizedDescription)"
        }
    }

    private func runCapture() async {
        guard !isRunning else { return }
        guard let models else {
            captureStatus = "Blocked: model set is not resolved."
            return
        }
        guard models.dit.variant.id == ModelManifest.ditW8ANEV1.id else {
            captureStatus = "Blocked: DiT variant must be w8-ane-v1."
            return
        }
        guard let context = MetalContext() else {
            captureStatus = "Blocked: Metal is unavailable."
            return
        }

        isRunning = true
        captureURL = nil
        modulationProbeURL = nil
        captureStatus = "Running pinned step-0 ANE capture…"
        defer { isRunning = false }

        do {
            let documents = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true)
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd-HHmmss"
            let stamp = formatter.string(from: Date())
            let destination = documents.appendingPathComponent(
                "AnimaXS-OracleE-A12-\(stamp).oraclee",
                isDirectory: false)
            let runner = OracleEDeviceCaptureRunner(context: context)
            let url = try await runner.run(
                prompt: Self.oraclePrompt,
                seed: Self.oracleSeed,
                models: models,
                outputURL: destination)
            captureURL = url
            captureStatus = "Full capture complete; running isolated block-0 W8 modulation probe…"

            let probeDestination = documents.appendingPathComponent(
                "AnimaXS-OracleE2-Modulation-A12-\(stamp).json",
                isDirectory: false)
            do {
                let probe = OracleE2StandaloneModulationProbe(context: context)
                let probeURL = try await probe.run(
                    models: models,
                    captureURL: url,
                    outputURL: probeDestination)
                modulationProbeURL = probeURL
                captureStatus = "PASS: 9/9 block-0 self substages + 84/84 branch checkpoints + isolated SiLU/W1/W2 modulation probe captured. Share both files for comparison."
            } catch {
                captureStatus = "PARTIAL: full .oraclee capture succeeded, but the isolated modulation probe failed: \(error.localizedDescription)"
            }
        } catch {
            captureStatus = "FAILED: \(error.localizedDescription)"
        }
    }
}

/// Oracle E2 follow-up that replays only the block-0 self modulation linears on
/// the A12 after the full capture has completed. It deliberately reads the
/// exact `prepared_embedding` and `prepared_adaln_lora` payloads back from that
/// `.oraclee` file, then runs the production `silu` and direct packed W8 matvec
/// kernels against the exact verified installed pack. This separates a raw
/// A12 W8-kernel/weight-view problem from corruption introduced by the normal
/// heterogeneous block scheduling/lifetime path.
private struct OracleE2StandaloneModulationProbe {
    private static let dim = 2_048
    private static let hiddenCount = 256
    private static let modulationCount = 6_144

    let context: MetalContext

    private struct CapturePayload: Decodable {
        let name: String
        let offset: UInt64
        let byteCount: Int
        let elementCount: Int
        let dtype: String
        let shape: [Int]
    }

    private struct CaptureManifest: Decodable {
        let payloads: [CapturePayload]
    }

    private struct ProbeDocument: Encodable {
        let schema: Int
        let producer: String
        let sourceCapture: String
        let ditVariantID: String
        let ditPackFilename: String
        let ditPackSHA256: String
        let preparedEmbedding: [Float]
        let preparedAdaLNLora: [Float]
        let productionModulation: [Float]
        let isolatedSiLUEmbedding: [Float]
        let isolatedW1Hidden: [Float]
        let isolatedW2PreAdaLN: [Float]
        let isolatedPostAdaLN: [Float]
    }

    func run(
        models: ResolvedModels,
        captureURL: URL,
        outputURL: URL
    ) async throws -> URL {
        guard models.dit.variant.id == ModelManifest.ditW8ANEV1.id else {
            throw AnimapkError.validation("Oracle E2 modulation probe requires w8-ane-v1")
        }

        let manifest = try readCaptureManifest(captureURL)
        let preparedEmbedding = try readFloatPayload(
            named: "prepared_embedding", count: Self.dim,
            manifest: manifest, captureURL: captureURL)
        let preparedAdaLN = try readFloatPayload(
            named: "prepared_adaln_lora", count: Self.modulationCount,
            manifest: manifest, captureURL: captureURL)
        let productionModulation = try readFloatPayload(
            named: "stage_b00_self_modulation", count: Self.modulationCount,
            manifest: manifest, captureURL: captureURL)

        let file = try AnimapkFile(url: models.dit.url)
        let range = try DiTANEHybridMetalLocator(file: file).block(0)
        guard range.length <= UInt64(Int.max) else {
            throw AnimapkError.validation("Oracle E2 block-0 Metal range is too large")
        }
        let streamer = try WeightStreamer(
            device: context.device,
            capacity: Int(range.length),
            slotCount: 1)
        _ = try streamer.load(range, from: file, slot: 0, mode: .copied)
        let ring = streamer.buffer(for: 0)

        let prefix = "model.diffusion_model.blocks.0."
        let w1Item = try tensor(
            prefix + "adaln_modulation_self_attn.1.weight", in: range)
        let w2Item = try tensor(
            prefix + "adaln_modulation_self_attn.2.weight", in: range)
        let w1 = try DiTQuantizedWeightFactory.makeMatrix(
            w1Item, ring: ring, rows: Self.hiddenCount, columns: Self.dim,
            label: "Oracle E2 block0 self modulation W1")
        let w2 = try DiTQuantizedWeightFactory.makeMatrix(
            w2Item, ring: ring, rows: Self.modulationCount, columns: Self.hiddenCount,
            label: "Oracle E2 block0 self modulation W2")

        let embedding = try makeBuffer(values: preparedEmbedding, label: "prepared embedding")
        let adaln = try makeBuffer(values: preparedAdaLN, label: "prepared AdaLN")
        let silu = try makeBuffer(count: Self.dim, label: "SiLU embedding")
        let hidden = try makeBuffer(count: Self.hiddenCount, label: "W1 hidden")
        let preAdd = try makeBuffer(count: Self.modulationCount, label: "W2 pre-AdaLN")
        let postAdd = try makeBuffer(count: Self.modulationCount, label: "post-AdaLN")

        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create Oracle E2 modulation command buffer")
        }
        command.label = "Oracle E2 isolated block0 W8 modulation"
        try encodeSiLU(command, input: embedding, output: silu, count: Self.dim)
        try encodeMatvec(command, input: silu, weight: w1, output: hidden)
        try encodeMatvec(command, input: hidden, weight: w2, output: preAdd)
        guard let blit = command.makeBlitCommandEncoder() else {
            throw AnimapkError.validation("failed to create Oracle E2 modulation blit encoder")
        }
        blit.copy(
            from: preAdd, sourceOffset: 0,
            to: postAdd, destinationOffset: 0,
            size: Self.modulationCount * MemoryLayout<Float>.stride)
        blit.endEncoding()
        try encodeAdd(command, destination: postAdd, source: adaln, count: Self.modulationCount)

        streamer.markInFlight(0)
        do {
            try await awaitCompletion(command)
        } catch {
            streamer.complete(0)
            throw error
        }
        streamer.complete(0)

        let document = ProbeDocument(
            schema: 1,
            producer: "AnimaXS A12 Oracle E2 isolated W8 modulation probe",
            sourceCapture: captureURL.lastPathComponent,
            ditVariantID: models.dit.variant.id,
            ditPackFilename: models.dit.variant.displayFilename,
            ditPackSHA256: models.dit.variant.sha256,
            preparedEmbedding: preparedEmbedding,
            preparedAdaLNLora: preparedAdaLN,
            productionModulation: productionModulation,
            isolatedSiLUEmbedding: copyFloats(from: silu, count: Self.dim),
            isolatedW1Hidden: copyFloats(from: hidden, count: Self.hiddenCount),
            isolatedW2PreAdaLN: copyFloats(from: preAdd, count: Self.modulationCount),
            isolatedPostAdaLN: copyFloats(from: postAdd, count: Self.modulationCount))

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(document)
        try data.write(to: outputURL, options: .atomic)
        return outputURL
    }

    private func tensor(
        _ name: String,
        in range: AnimapkExecutionRange
    ) throws -> AnimapkTensorSpans {
        guard let item = range.tensors.first(where: { $0.tensor.name == name }) else {
            throw AnimapkError.validation("Oracle E2 missing tensor \(name)")
        }
        return item
    }

    private func makeBuffer(values: [Float], label: String) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = context.device.makeBuffer(
                    bytes: base, length: bytes.count, options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate Oracle E2 \(label) buffer")
            }
            return buffer
        }
    }

    private func makeBuffer(count: Int, label: String) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw AnimapkError.validation("failed to allocate Oracle E2 \(label) buffer")
        }
        return buffer
    }

    private func encodeSiLU(
        _ command: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "silu")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Oracle E2 SiLU encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeMatvec(
        _ command: MTLCommandBuffer,
        input: MTLBuffer,
        weight: QuantizedLinearWeightBuffers,
        output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(
            named: DiTQuantizedWeightFactory.matvecKernel(for: weight.storage))
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Oracle E2 W8 matvec encoder")
        }
        var columns = UInt32(weight.columns)
        var rows = UInt32(weight.rows)
        var rowStride = UInt32(weight.packedRowStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight.packed, offset: weight.packedOffset, index: 0)
        encoder.setBuffer(weight.scale, offset: weight.scaleOffset, index: 1)
        encoder.setBuffer(weight.zero, offset: weight.zeroOffset, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&columns, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBytes(&rowStride, length: 4, index: 7)
        let threads = reductionThreads(pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: weight.rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeAdd(
        _ command: MTLCommandBuffer,
        destination: MTLBuffer,
        source: MTLBuffer,
        count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "add_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create Oracle E2 add encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destination, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func reductionThreads(_ maximum: Int) -> Int {
        var result = 1
        while result * 2 <= min(256, maximum) { result *= 2 }
        return result
    }

    private func awaitCompletion(_ command: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            command.addCompletedHandler { completed in
                if let error = completed.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            command.commit()
        }
    }

    private func copyFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readCaptureManifest(_ url: URL) throws -> CaptureManifest {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let header = handle.readData(ofLength: 32)
        guard header.count == 32,
              String(data: header.subdata(in: 0..<8), encoding: .utf8) == "AXOECAP1" else {
            throw AnimapkError.validation("Oracle E2 source capture has invalid header")
        }
        let manifestOffset = readUInt64LE(header, at: 16)
        let manifestLength = readUInt64LE(header, at: 24)
        guard manifestLength <= UInt64(Int.max) else {
            throw AnimapkError.validation("Oracle E2 source manifest is too large")
        }
        handle.seek(toFileOffset: manifestOffset)
        let manifestData = handle.readData(ofLength: Int(manifestLength))
        guard manifestData.count == Int(manifestLength) else {
            throw AnimapkError.validation("Oracle E2 source manifest is truncated")
        }
        return try JSONDecoder().decode(CaptureManifest.self, from: manifestData)
    }

    private func readFloatPayload(
        named name: String,
        count: Int,
        manifest: CaptureManifest,
        captureURL: URL
    ) throws -> [Float] {
        guard let record = manifest.payloads.first(where: { $0.name == name }) else {
            throw AnimapkError.validation("Oracle E2 source capture missing \(name)")
        }
        let expectedBytes = count * MemoryLayout<Float>.stride
        guard record.dtype == "float32-le",
              record.elementCount == count,
              record.byteCount == expectedBytes,
              record.shape.reduce(1, *) == count else {
            throw AnimapkError.validation("Oracle E2 payload \(name) has unexpected layout")
        }
        let handle = try FileHandle(forReadingFrom: captureURL)
        defer { try? handle.close() }
        handle.seek(toFileOffset: record.offset)
        let data = handle.readData(ofLength: expectedBytes)
        guard data.count == expectedBytes else {
            throw AnimapkError.validation("Oracle E2 payload \(name) is truncated")
        }
        var values = [Float](repeating: 0, count: count)
        values.withUnsafeMutableBytes { destination in
            _ = data.copyBytes(to: destination)
        }
        return values
    }

    private func readUInt64LE(_ data: Data, at offset: Int) -> UInt64 {
        var value: UInt64 = 0
        for index in 0..<8 {
            value |= UInt64(data[offset + index]) << UInt64(index * 8)
        }
        return value
    }
}
#endif
