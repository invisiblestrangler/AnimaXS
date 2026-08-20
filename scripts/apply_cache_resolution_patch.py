from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text()


def write(path, text):
    (ROOT / path).write_text(text)


def replace(path, old, new, count=1):
    text = read(path)
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{path}: expected {count} occurrences, found {actual}: {old[:120]!r}")
    write(path, text.replace(old, new))


def replace_between(path, start_marker, end_marker, replacement):
    text = read(path)
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    write(path, text[:start] + replacement + text[end:])


# ---------------------------------------------------------------------------
# Resolution model + task-local geometry. Defaults stay exactly 512x512.
# ---------------------------------------------------------------------------
path = "AnimaXS/Runtime/Generation/ModelConstants.swift"
text = read(path)
marker = "enum ModelConstants {\n"
insert = '''enum GenerationResolution: Int, Codable, CaseIterable, Identifiable, Sendable {
    case square512 = 512
    case square1024 = 1024

    var id: Int { rawValue }
    var imageSize: Int { rawValue }
    var latentSize: Int { rawValue / 8 }
    var patchGrid: Int { latentSize / ModelConstants.ditPatchSpatial }
    var ditTokens: Int { patchGrid * patchGrid }
    var latentElements: Int { ModelConstants.ditLatentChannels * latentSize * latentSize }
    var label: String { "\\(rawValue)×\\(rawValue)" }
}

/// Per-generation geometry without process-global mutable shape state. Direct
/// executor/tests that do not install a value continue to run the historical
/// 512x512 geometry.
enum GenerationGeometryRuntime {
    @TaskLocal static var current: GenerationResolution = .square512
}

'''
if text.count(marker) != 1:
    raise RuntimeError("ModelConstants marker mismatch")
write(path, text.replace(marker, insert + marker))


# ---------------------------------------------------------------------------
# Multiprocedure combined-model cache: pack-keyed process state.
# ---------------------------------------------------------------------------
path = "AnimaXS/Runtime/Generation/StageConformances.swift"
text = read(path)
start = text.index("final class ANEW8MultiProcModelCache: @unchecked Sendable {")
prefix = text[:start]
new_class = r'''final class ANEW8MultiProcModelCache: @unchecked Sendable {
    static let pinnedBlocks = 6
    static let streamingSlots = 2
    static let maxResidentBlocks = pinnedBlocks + streamingSlots

    /// The expensive synthesized 10-procedure handles are process/pack state,
    /// not generation state. A Generate owns only traversal sequencing. This
    /// mirrors the stable backend's deterministic prepared cache while also
    /// avoiding donor->combined reconstruction on the second Generate.
    private final class SharedState: @unchecked Sendable {
        let loaderQueue = DispatchQueue(
            label: "com.invisiblestrangler.AnimaXS.ane-multiproc-loader",
            qos: .userInitiated)
        var namespace: String?
        var models: [Int: ANEW8MultiProcBlockModel] = [:]
        var streamSlots: [Int?] = Array(repeating: nil, count: streamingSlots)
        var residentHighWater = 0
    }

    private static let sharedState = SharedState()

    /// Called after the actual ANE-native pack is resolved and before diffusion.
    /// Re-selecting the same namespace is a no-op. A changed pack destroys the
    /// old combined handles only when no generation is active.
    static func selectNamespace(_ namespace: String) {
        let shared = sharedState
        shared.loaderQueue.sync {
            guard shared.namespace != namespace else { return }
            for model in shared.models.values where model.isLoaded {
                _ = try? model.unload()
            }
            shared.models.removeAll()
            shared.streamSlots = Array(repeating: nil, count: streamingSlots)
            shared.residentHighWater = 0
            shared.namespace = namespace
            print("ANE multiprocedure process cache namespace selected: \(namespace.prefix(12))…")
        }
    }

    private let shared = sharedState

    // Per-generation traversal state only. The production DiT loop is serial
    // 0...27; the underlying combined handles survive this facade's deinit.
    private var expectedBlock = 0
    private var nextFuture: ANEW8MultiProcLoadFuture?

    init() {
        print("ANE multiprocedure scheduler: pinned=6 streaming=2 maxResident=8 processCache=on")
    }

    func scheduledModel(for block: Int) throws -> ANEW8MultiProcScheduledModel {
        guard shared.loaderQueue.sync(execute: { shared.namespace != nil }) else {
            throw AnimapkError.validation("ANE multiprocedure process cache was not configured for the resolved pack")
        }
        guard block == expectedBlock else {
            throw AnimapkError.validation(
                "ANE multiprocedure scheduler expected block \(expectedBlock), received \(block)")
        }

        let work: ANEW8MultiProcLoadWork
        var foregroundWait = 0.0
        if block >= Self.pinnedBlocks,
           let future = nextFuture,
           future.block == block {
            let waited = try future.wait()
            work = waited.work
            foregroundWait = waited.waitMilliseconds
            nextFuture = nil
        } else {
            if let future = nextFuture {
                _ = try? future.wait()
                nextFuture = nil
            }
            work = try shared.loaderQueue.sync {
                try Self.loadBlock(block, shared: shared)
            }
        }

        expectedBlock = block == ModelConstants.ditBlocks - 1 ? 0 : block + 1

        let next: Int?
        if block == Self.pinnedBlocks - 1 {
            next = Self.pinnedBlocks
        } else if block >= Self.pinnedBlocks && block + 1 < ModelConstants.ditBlocks {
            next = block + 1
        } else {
            next = nil
        }
        if let next { nextFuture = launchPrefetch(block: next) }

        return ANEW8MultiProcScheduledModel(
            model: work.model,
            newlyLoadedMilliseconds: work.loadMilliseconds,
            compileMilliseconds: work.compileMilliseconds,
            unloadMilliseconds: work.unloadMilliseconds,
            waitMilliseconds: foregroundWait,
            residentBlocks: work.residentBlocks)
    }

    func complete(block: Int) throws {
        guard (0..<ModelConstants.ditBlocks).contains(block) else {
            throw AnimapkError.validation("ANE multiprocedure completion block is out of range")
        }
    }

    /// Cancellation/failure unloads residency for safety, but deliberately
    /// retains the synthesized handles so the next Generate can reload them.
    func abortTraversal() {
        if let future = nextFuture {
            _ = try? future.wait()
            nextFuture = nil
        }
        shared.loaderQueue.sync {
            for model in shared.models.values where model.isLoaded {
                _ = try? model.unload()
            }
            shared.streamSlots = Array(repeating: nil, count: Self.streamingSlots)
        }
        expectedBlock = 0
    }

    private func launchPrefetch(block: Int) -> ANEW8MultiProcLoadFuture {
        let future = ANEW8MultiProcLoadFuture(block: block)
        let shared = self.shared
        shared.loaderQueue.async {
            do { future.finish(.success(try Self.loadBlock(block, shared: shared))) }
            catch { future.finish(.failure(error)) }
        }
        return future
    }

    /// Loader-queue only. Streaming slots remain bounded to two while the six
    /// pinned blocks remain resident across successful generations. The final
    /// b26/b27 slot occupants may also remain resident, so the steady process
    /// cache still obeys the proven eight-model ceiling.
    private static func loadBlock(_ block: Int, shared: SharedState) throws -> ANEW8MultiProcLoadWork {
        var unloadMS = 0.0
        if block >= pinnedBlocks {
            let slot = (block - pinnedBlocks) % streamingSlots
            if let occupant = shared.streamSlots[slot], occupant != block,
               let old = shared.models[occupant], old.isLoaded {
                unloadMS += try old.unload()
            }
            shared.streamSlots[slot] = nil
        }

        var compileMS = 0.0
        var loadMS = 0.0
        let model: ANEW8MultiProcBlockModel
        if let existing = shared.models[block] {
            model = existing
            if !existing.isLoaded { loadMS = try existing.load() }
        } else {
            model = try ANEW8MultiProcBlockModel(
                block: block,
                compileMilliseconds: &compileMS,
                loadMilliseconds: &loadMS)
            shared.models[block] = model
        }

        if block >= pinnedBlocks {
            let slot = (block - pinnedBlocks) % streamingSlots
            shared.streamSlots[slot] = block
        }

        let resident = shared.models.values.reduce(into: 0) { count, value in
            if value.isLoaded { count += 1 }
        }
        shared.residentHighWater = max(shared.residentHighWater, resident)
        guard resident <= maxResidentBlocks else {
            throw AnimapkError.validation(
                "ANE multiprocedure scheduler residency invariant violated: \(resident) > \(maxResidentBlocks)")
        }
        return ANEW8MultiProcLoadWork(
            model: model,
            loadMilliseconds: loadMS,
            compileMilliseconds: compileMS,
            unloadMilliseconds: unloadMS,
            residentBlocks: resident)
    }

    deinit {
        // Drain only this generation's outstanding admission. Shared combined
        // handles and the safe eight-model resident set intentionally survive.
        if let future = nextFuture { _ = try? future.wait() }
    }
}
'''
write(path, prefix + new_class)

# Select the correct process cache namespace before diffusion preparation.
replace(
    "AnimaXS/Runtime/Generation/GenerationEngine.swift",
    """        let file = try AnimapkFile(url: fileURL)\n        let result = try ANEW8ModelPreparer.ensurePrepared(file: file)\n""",
    """        let file = try AnimapkFile(url: fileURL)\n        if optimization.linearBackend == .aneMultiProcW8 {\n            ANEW8MultiProcModelCache.selectNamespace(try ANEW8NativePack.namespace(file: file))\n        }\n        let result = try ANEW8ModelPreparer.ensurePrepared(file: file)\n""",
)


# ---------------------------------------------------------------------------
# Keep ANE program geometry at the proven 1024-token spatial and chunk larger
# image token grids through those exact programs.
# ---------------------------------------------------------------------------
path = "AnimaXS/Runtime/ANE/ANEW8MLPBackend.swift"
text = read(path)
text = text.replace("spatial: DiTBlockExecutor.tokens", "spatial: ModelConstants.ditTokensAt512")
text = text.replace("spatial: UInt(DiTBlockExecutor.tokens),\n            label: \"dit_b\\(block)_self_qkv\"", "spatial: UInt(ModelConstants.ditTokensAt512),\n            label: \"dit_b\\(block)_self_qkv\"")
text = text.replace("spatial: UInt(DiTBlockExecutor.tokens))", "spatial: UInt(ModelConstants.ditTokensAt512))")
write(path, text)

replace(
    "AnimaXS/Runtime/Metal/DiTBlockExecutor.swift",
    "    static let tokens = 1_024\n",
    "    static var tokens: Int { GenerationGeometryRuntime.current.ditTokens }\n",
)

new_ane_method = r'''    /// Shared heterogeneous ANE/Metal block graph. At 512x512 the proven ANE
    /// programs consume all 1024 image tokens at once. At 1024x1024 the DiT has
    /// 4096 image tokens; projection GEMMs are tokenwise, so the exact same
    /// 1024-spatial ANE programs are applied in four contiguous token chunks.
    /// Self/cross attention still operates over the FULL 4096-token tensors.
    private func executeANEHybridBlock(
        blockIndex: Int,
        residual: MTLBuffer,
        emb: MTLBuffer,
        adalnLora: MTLBuffer,
        crossContext: MTLBuffer,
        rope: MTLBuffer,
        slot: Int,
        prefetchIndex: Int?,
        prefetchSlot: Int,
        diagnosticBranchCompleted: DiagnosticBranchCompleted?
    ) async throws {
        guard let aneSurfaces else {
            throw AnimapkError.validation("ANE W8 backend was selected without initialized IOSurfaces")
        }
        let aneRows = ModelConstants.ditTokensAt512
        guard Self.tokens >= aneRows, Self.tokens % aneRows == 0 else {
            throw AnimapkError.validation("ANE image token count \(Self.tokens) is not chunkable by \(aneRows)")
        }
        let chunked = Self.tokens != aneRows
        let chunkCount = Self.tokens / aneRows

        metrics?.beginBlock(blockIndex)
        let crossCacheHit = crossKVCache?.isReady(blockIndex) ?? false
        let usesMultiProc = optimization.linearBackend == .aneMultiProcW8
        var schedulerCompleted = false
        defer {
            if !schedulerCompleted {
                if usesMultiProc { aneMultiProcCache?.abortTraversal() }
                else { aneModelCache?.abortTraversal() }
            }
        }

        let oldModels: ANEW8DiTModels?
        let multiModel: ANEW8MultiProcBlockModel?
        if usesMultiProc {
            guard let aneMultiProcCache else {
                throw AnimapkError.validation("ANE multiprocedure scheduler is unavailable")
            }
            let modelResult = try aneMultiProcCache.scheduledModel(for: blockIndex)
            let reportedLoad = modelResult.newlyLoadedMilliseconds + modelResult.compileMilliseconds
            if reportedLoad > 0 { metrics?.recordANEModelLoad(seconds: reportedLoad / 1_000.0) }
            if modelResult.waitMilliseconds > 0 {
                metrics?.recordHostWait(seconds: modelResult.waitMilliseconds / 1_000.0)
            }
            oldModels = nil
            multiModel = modelResult.model
        } else {
            guard let aneModelCache else {
                throw AnimapkError.validation("original ANE hybrid scheduler is unavailable")
            }
            let modelResult = try aneModelCache.scheduledModels(for: blockIndex, kvWarm: crossCacheHit)
            if modelResult.newlyLoadedMilliseconds > 0 {
                metrics?.recordANEModelLoad(seconds: modelResult.newlyLoadedMilliseconds / 1_000.0)
            }
            if modelResult.waitMilliseconds > 0 {
                metrics?.recordHostWait(seconds: modelResult.waitMilliseconds / 1_000.0)
            }
            if !crossCacheHit && !modelResult.models.hasCrossKVModels {
                throw AnimapkError.validation("ANE scheduler supplied a six-program block on cross-K/V cache miss")
            }
            oldModels = modelResult.models
            multiModel = nil
        }

        func evaluateProjection(
            old: A12ANEProjectionModel?,
            procedure: ANEW8MultiProcProcedure,
            input: A12ANESurface,
            output: A12ANESurface,
            label: String
        ) throws {
            if let multiModel {
                try evaluateANEMultiProc(
                    multiModel, procedure: procedure, input: input, output: output,
                    label: label, blockIndex: blockIndex)
            } else if let old {
                try evaluateANEProjection(old, input: input, output: output,
                                          label: label, blockIndex: blockIndex)
            } else {
                throw AnimapkError.validation("missing ANE model for \(label) block \(blockIndex)")
            }
        }

        func uploadChunk(_ token: MTLBuffer, channels: Int,
                         surface: A12ANESurface, chunk: Int, label: String) async throws {
            guard let command = context.commandQueue.makeCommandBuffer() else {
                throw AnimapkError.validation("failed to create ANE \(label) upload command buffer")
            }
            command.label = "DiT block \(blockIndex) ANE \(label) upload c\(chunk)"
            let start = ProcessInfo.processInfo.systemUptime
            try encodeTokenToANE(
                command, tokenMajor: token, aneMajor: surface.metalBuffer,
                rows: aneRows, channels: channels,
                planeStrideElements: surface.planeStrideElements,
                tokenRowOffset: chunk * aneRows)
            try await commitStandaloneCommand(
                command, encodeSeconds: ProcessInfo.processInfo.systemUptime - start)
        }

        func downloadChunk(_ surface: A12ANESurface, token: MTLBuffer, channels: Int,
                           chunk: Int, label: String) async throws {
            guard let command = context.commandQueue.makeCommandBuffer() else {
                throw AnimapkError.validation("failed to create ANE \(label) download command buffer")
            }
            command.label = "DiT block \(blockIndex) ANE \(label) download c\(chunk)"
            let start = ProcessInfo.processInfo.systemUptime
            try encodeANEToToken(
                command, aneMajor: surface.metalBuffer, tokenMajor: token,
                rows: aneRows, channels: channels,
                planeStrideElements: surface.planeStrideElements,
                tokenRowOffset: chunk * aneRows)
            try await commitStandaloneCommand(
                command, encodeSeconds: ProcessInfo.processInfo.systemUptime - start)
        }

        func projectionChunked(
            inputToken: MTLBuffer, outputToken: MTLBuffer,
            inputChannels: Int, outputChannels: Int,
            inputSurface: A12ANESurface, outputSurface: A12ANESurface,
            old: A12ANEProjectionModel?, procedure: ANEW8MultiProcProcedure,
            label: String
        ) async throws {
            for chunk in 0..<chunkCount {
                try await uploadChunk(inputToken, channels: inputChannels,
                                      surface: inputSurface, chunk: chunk, label: label)
                try evaluateProjection(old: old, procedure: procedure,
                                       input: inputSurface, output: outputSurface, label: label)
                try await downloadChunk(outputSurface, token: outputToken,
                                        channels: outputChannels, chunk: chunk, label: label)
            }
        }

        func selfQKVChunked(
            inputToken: MTLBuffer, qToken: MTLBuffer,
            kToken: MTLBuffer, vToken: MTLBuffer
        ) async throws {
            for chunk in 0..<chunkCount {
                try await uploadChunk(inputToken, channels: Self.dim,
                                      surface: aneSurfaces.tokenInput, chunk: chunk, label: "self QKV")
                if let multiModel {
                    try evaluateANEMultiProc(multiModel, procedure: .selfQ,
                        input: aneSurfaces.tokenInput, output: aneSurfaces.q,
                        label: "self Q", blockIndex: blockIndex)
                    try evaluateANEMultiProc(multiModel, procedure: .selfK,
                        input: aneSurfaces.tokenInput, output: aneSurfaces.k,
                        label: "self K", blockIndex: blockIndex)
                    try evaluateANEMultiProc(multiModel, procedure: .selfV,
                        input: aneSurfaces.tokenInput, output: aneSurfaces.v,
                        label: "self V", blockIndex: blockIndex)
                } else if let oldModels {
                    try evaluateANEQKV(oldModels.selfQKV, input: aneSurfaces.tokenInput,
                                       q: aneSurfaces.q, k: aneSurfaces.k, v: aneSurfaces.v,
                                       blockIndex: blockIndex)
                } else {
                    throw AnimapkError.validation("missing self QKV model for block \(blockIndex)")
                }
                guard let command = context.commandQueue.makeCommandBuffer() else {
                    throw AnimapkError.validation("failed to create ANE self QKV download command buffer")
                }
                let start = ProcessInfo.processInfo.systemUptime
                try encodeANEToToken(command, aneMajor: aneSurfaces.q.metalBuffer, tokenMajor: qToken,
                                     rows: aneRows, channels: Self.dim,
                                     planeStrideElements: aneSurfaces.q.planeStrideElements,
                                     tokenRowOffset: chunk * aneRows)
                try encodeANEToToken(command, aneMajor: aneSurfaces.k.metalBuffer, tokenMajor: kToken,
                                     rows: aneRows, channels: Self.dim,
                                     planeStrideElements: aneSurfaces.k.planeStrideElements,
                                     tokenRowOffset: chunk * aneRows)
                try encodeANEToToken(command, aneMajor: aneSurfaces.v.metalBuffer, tokenMajor: vToken,
                                     rows: aneRows, channels: Self.dim,
                                     planeStrideElements: aneSurfaces.v.planeStrideElements,
                                     tokenRowOffset: chunk * aneRows)
                try await commitStandaloneCommand(
                    command, encodeSeconds: ProcessInfo.processInfo.systemUptime - start)
            }
        }

        func mlpChunked(inputToken: MTLBuffer, outputToken: MTLBuffer) async throws {
            guard aneSurfaces.hidden.planeStrideElements == UInt(aneRows) else {
                throw AnimapkError.validation("ANE hidden surface unexpectedly padded at spatial \(aneRows)")
            }
            for chunk in 0..<chunkCount {
                try await uploadChunk(inputToken, channels: Self.dim,
                                      surface: aneSurfaces.tokenInput, chunk: chunk, label: "MLP1")
                try evaluateProjection(old: oldModels?.mlpUp, procedure: .mlpUp,
                                       input: aneSurfaces.tokenInput, output: aneSurfaces.hidden,
                                       label: "MLP1")
                guard let gelu = context.commandQueue.makeCommandBuffer() else {
                    throw AnimapkError.validation("failed to create ANE chunk GELU command buffer")
                }
                let geluStart = ProcessInfo.processInfo.systemUptime
                try encodeHalfComputeBoundary(gelu, aneSurfaces.hidden.metalBuffer,
                                              count: aneRows * Self.hidden)
                try encodeMLPActivation(gelu, hiddenHalf: aneSurfaces.hidden.metalBuffer,
                                        rows: aneRows)
                try await commitStandaloneCommand(
                    gelu, encodeSeconds: ProcessInfo.processInfo.systemUptime - geluStart)
                try evaluateProjection(old: oldModels?.mlpDown, procedure: .mlpDown,
                                       input: aneSurfaces.hidden, output: aneSurfaces.tokenOutput,
                                       label: "MLP2")
                try await downloadChunk(aneSurfaces.tokenOutput, token: outputToken,
                                        channels: Self.dim, chunk: chunk, label: "MLP2")
            }
        }

        let range = try blockRange(blockIndex)
        if streamer.loadedLogicalIndexes[slot] != blockIndex {
            let copyStart = ProcessInfo.processInfo.systemUptime
            let result = try streamer.load(range, from: file, slot: slot, mode: .copied)
            guard result.mode != .noCopy else {
                throw AnimapkError.validation("ANE W8 backend must not use mmap no-copy weights")
            }
            metrics?.recordWeightCopy(bytes: Int(range.length),
                                      seconds: ProcessInfo.processInfo.systemUptime - copyStart)
        }
        let weights = try ANEBlockWeights(range: range, ring: streamer.buffer(for: slot))
        streamer.markInFlight(slot)
        var slotReleased = false
        defer { if !slotReleased { streamer.complete(slot) } }

        let siluEmb = buffer("dit.siluEmb.f32", Self.dim, Float.self)

        let s0Start = ProcessInfo.processInfo.systemUptime
        guard let s0 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid self-input command buffer")
        }
        s0.label = "DiT block \(blockIndex) ANE self input"
        try encodeUnary(s0, kernel: "silu", input: emb, output: siluEmb, count: Self.dim)
        try encodeComputeBoundary(s0, siluEmb, count: Self.dim)
        let selfInput = try encodeAttentionInput(
            s0, residual: residual, siluEmb: siluEmb, adalnLora: adalnLora,
            weights: weights, cross: false)
        if !chunked {
            try encodeTokenToANE(s0, tokenMajor: selfInput.projectionInput,
                                 aneMajor: aneSurfaces.tokenInput.metalBuffer,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        }
        let s0End = ProcessInfo.processInfo.systemUptime

        let gate0 = CommandBufferGate()
        s0.addCompletedHandler { completed in
            if let error = completed.error { gate0.resume(throwing: error) }
            else { gate0.resume() }
        }
        let wait0Start = ProcessInfo.processInfo.systemUptime
        s0.commit()
        var prefetchError: Error?
        if let prefetchIndex, prefetchSlot != slot {
            do {
                let nextRange = try blockRange(prefetchIndex)
                let copyStart = ProcessInfo.processInfo.systemUptime
                let result = try streamer.load(nextRange, from: file, slot: prefetchSlot, mode: .copied)
                guard result.mode != .noCopy else {
                    throw AnimapkError.validation("ANE hybrid prefetch unexpectedly selected mmap no-copy")
                }
                metrics?.recordWeightCopy(bytes: Int(nextRange.length),
                                          seconds: ProcessInfo.processInfo.systemUptime - copyStart)
            } catch { prefetchError = error }
        }
        try await gate0.wait()
        if let prefetchError { throw prefetchError }
        recordCompletedCommand(s0, encodeSeconds: s0End - s0Start,
                               hostWindowSeconds: ProcessInfo.processInfo.systemUptime - wait0Start)

        let qToken = buffer("dit.q.token.f16", Self.tokens * Self.dim, Float16.self)
        let kToken = buffer("dit.k.token.f16", Self.tokens * Self.dim, Float16.self)
        let vToken = buffer("dit.v.token.f16", Self.tokens * Self.dim, Float16.self)
        if chunked {
            try await selfQKVChunked(inputToken: selfInput.projectionInput,
                                     qToken: qToken, kToken: kToken, vToken: vToken)
        } else {
            if let multiModel {
                try evaluateANEMultiProc(multiModel, procedure: .selfQ,
                    input: aneSurfaces.tokenInput, output: aneSurfaces.q,
                    label: "self Q", blockIndex: blockIndex)
                try evaluateANEMultiProc(multiModel, procedure: .selfK,
                    input: aneSurfaces.tokenInput, output: aneSurfaces.k,
                    label: "self K", blockIndex: blockIndex)
                try evaluateANEMultiProc(multiModel, procedure: .selfV,
                    input: aneSurfaces.tokenInput, output: aneSurfaces.v,
                    label: "self V", blockIndex: blockIndex)
            } else if let oldModels {
                try evaluateANEQKV(oldModels.selfQKV, input: aneSurfaces.tokenInput,
                                   q: aneSurfaces.q, k: aneSurfaces.k, v: aneSurfaces.v,
                                   blockIndex: blockIndex)
            } else {
                throw AnimapkError.validation("missing self QKV model for block \(blockIndex)")
            }
        }

        let s1Start = ProcessInfo.processInfo.systemUptime
        guard let s1 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid self-attention command buffer")
        }
        s1.label = "DiT block \(blockIndex) ANE self attention"
        if !chunked {
            try encodeANEToToken(s1, aneMajor: aneSurfaces.q.metalBuffer, tokenMajor: qToken,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.q.planeStrideElements)
            try encodeANEToToken(s1, aneMajor: aneSurfaces.k.metalBuffer, tokenMajor: kToken,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.k.planeStrideElements)
            try encodeANEToToken(s1, aneMajor: aneSurfaces.v.metalBuffer, tokenMajor: vToken,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.v.planeStrideElements)
        }
        let selfAttended = try encodeANEAttentionMath(
            s1, qToken: qToken, kToken: kToken, vToken: vToken,
            cross: false, blockIndex: blockIndex, rope: rope, weights: weights, slot: slot,
            projectedKVAvailable: true)
        if !chunked {
            try encodeTokenToANE(s1, tokenMajor: selfAttended,
                                 aneMajor: aneSurfaces.tokenInput.metalBuffer,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        }
        let s1End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s1, encodeSeconds: s1End - s1Start)

        let selfBranch: MTLBuffer?
        if chunked {
            let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
            try await projectionChunked(
                inputToken: selfAttended, outputToken: branch,
                inputChannels: Self.dim, outputChannels: Self.dim,
                inputSurface: aneSurfaces.tokenInput, outputSurface: aneSurfaces.tokenOutput,
                old: oldModels?.selfO, procedure: .selfO, label: "self output")
            selfBranch = branch
        } else {
            try evaluateProjection(old: oldModels?.selfO, procedure: .selfO,
                                   input: aneSurfaces.tokenInput, output: aneSurfaces.tokenOutput,
                                   label: "self output")
            selfBranch = nil
        }

        let s2Start = ProcessInfo.processInfo.systemUptime
        guard let s2 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid cross-input command buffer")
        }
        s2.label = "DiT block \(blockIndex) ANE cross input"
        if let selfBranch {
            try encodeAttentionOutputFromToken(s2, branch: selfBranch, residual: residual,
                                               modulation: selfInput.modulation, cross: false)
        } else {
            try encodeANEAttentionOutput(s2, aneOutput: aneSurfaces.tokenOutput,
                                         residual: residual, modulation: selfInput.modulation,
                                         cross: false)
        }
        let selfSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(s2, source: residual, key: "dit.diagnostic.afterSelf.f32")
        }
        let crossInput = try encodeAttentionInput(
            s2, residual: residual, siluEmb: siluEmb, adalnLora: adalnLora,
            weights: weights, cross: true)
        if !chunked {
            try encodeTokenToANE(s2, tokenMajor: crossInput.projectionInput,
                                 aneMajor: aneSurfaces.tokenInput.metalBuffer,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        }
        if !crossCacheHit {
            try encodeTokenToANE(s2, tokenMajor: crossContext,
                                 aneMajor: aneSurfaces.contextInput.metalBuffer,
                                 rows: Self.contextTokens, channels: Self.contextDim,
                                 planeStrideElements: aneSurfaces.contextInput.planeStrideElements)
        }
        let s2End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s2, encodeSeconds: s2End - s2Start)

        let crossQToken = buffer("dit.q.token.f16", Self.tokens * Self.dim, Float16.self)
        if chunked {
            try await projectionChunked(
                inputToken: crossInput.projectionInput, outputToken: crossQToken,
                inputChannels: Self.dim, outputChannels: Self.dim,
                inputSurface: aneSurfaces.tokenInput, outputSurface: aneSurfaces.q,
                old: oldModels?.crossQ, procedure: .crossQ, label: "cross Q")
        } else {
            try evaluateProjection(old: oldModels?.crossQ, procedure: .crossQ,
                                   input: aneSurfaces.tokenInput, output: aneSurfaces.q,
                                   label: "cross Q")
        }
        if !crossCacheHit {
            try evaluateProjection(old: oldModels?.crossK, procedure: .crossK,
                                   input: aneSurfaces.contextInput, output: aneSurfaces.contextK,
                                   label: "cross K")
            try evaluateProjection(old: oldModels?.crossV, procedure: .crossV,
                                   input: aneSurfaces.contextInput, output: aneSurfaces.contextV,
                                   label: "cross V")
        }

        let s3Start = ProcessInfo.processInfo.systemUptime
        guard let s3 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid cross-attention command buffer")
        }
        s3.label = "DiT block \(blockIndex) ANE cross attention"
        let crossKCount = Self.contextTokens * Self.dim
        let crossKToken = buffer("dit.k.token.f16", crossKCount, Float16.self)
        let crossVToken = buffer("dit.v.token.f16", crossKCount, Float16.self)
        if !chunked {
            try encodeANEToToken(s3, aneMajor: aneSurfaces.q.metalBuffer, tokenMajor: crossQToken,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.q.planeStrideElements)
        }
        if !crossCacheHit {
            try encodeANEToToken(s3, aneMajor: aneSurfaces.contextK.metalBuffer, tokenMajor: crossKToken,
                                 rows: Self.contextTokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.contextK.planeStrideElements)
            try encodeANEToToken(s3, aneMajor: aneSurfaces.contextV.metalBuffer, tokenMajor: crossVToken,
                                 rows: Self.contextTokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.contextV.planeStrideElements)
        }
        let crossAttended = try encodeANEAttentionMath(
            s3, qToken: crossQToken, kToken: crossKToken, vToken: crossVToken,
            cross: true, blockIndex: blockIndex, rope: rope, weights: weights, slot: slot,
            projectedKVAvailable: !crossCacheHit)
        if !chunked {
            try encodeTokenToANE(s3, tokenMajor: crossAttended,
                                 aneMajor: aneSurfaces.tokenInput.metalBuffer,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        }
        let s3End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s3, encodeSeconds: s3End - s3Start)

        let crossBranch: MTLBuffer?
        if chunked {
            let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
            try await projectionChunked(
                inputToken: crossAttended, outputToken: branch,
                inputChannels: Self.dim, outputChannels: Self.dim,
                inputSurface: aneSurfaces.tokenInput, outputSurface: aneSurfaces.tokenOutput,
                old: oldModels?.crossO, procedure: .crossO, label: "cross output")
            crossBranch = branch
        } else {
            try evaluateProjection(old: oldModels?.crossO, procedure: .crossO,
                                   input: aneSurfaces.tokenInput, output: aneSurfaces.tokenOutput,
                                   label: "cross output")
            crossBranch = nil
        }

        let s4Start = ProcessInfo.processInfo.systemUptime
        guard let s4 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid MLP-input command buffer")
        }
        s4.label = "DiT block \(blockIndex) ANE MLP input"
        if let crossBranch {
            try encodeAttentionOutputFromToken(s4, branch: crossBranch, residual: residual,
                                               modulation: crossInput.modulation, cross: true)
        } else {
            try encodeANEAttentionOutput(s4, aneOutput: aneSurfaces.tokenOutput,
                                         residual: residual, modulation: crossInput.modulation,
                                         cross: true)
        }
        let crossSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(s4, source: residual, key: "dit.diagnostic.afterCross.f32")
        }
        let (mlpModulation, mlpInput) = try encodeMLPInput(
            s4, residual: residual, siluEmb: siluEmb,
            adalnLora: adalnLora, weights: weights)
        if !chunked {
            try encodeTokenToANE(s4, tokenMajor: mlpInput,
                                 aneMajor: aneSurfaces.tokenInput.metalBuffer,
                                 rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        }
        let s4End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s4, encodeSeconds: s4End - s4Start)

        let mlpBranch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        if chunked {
            try await mlpChunked(inputToken: mlpInput, outputToken: mlpBranch)
        } else {
            try evaluateProjection(old: oldModels?.mlpUp, procedure: .mlpUp,
                                   input: aneSurfaces.tokenInput, output: aneSurfaces.hidden,
                                   label: "MLP1")
            guard aneSurfaces.hidden.planeStrideElements == UInt(Self.tokens) else {
                throw AnimapkError.validation("ANE hidden surface unexpectedly padded at spatial \(Self.tokens)")
            }
            let s5Start = ProcessInfo.processInfo.systemUptime
            guard let s5 = context.commandQueue.makeCommandBuffer() else {
                throw AnimapkError.validation("failed to create ANE hybrid GELU command buffer")
            }
            s5.label = "DiT block \(blockIndex) ANE GELU"
            try encodeHalfComputeBoundary(s5, aneSurfaces.hidden.metalBuffer,
                                          count: Self.tokens * Self.hidden)
            try encodeMLPActivation(s5, hiddenHalf: aneSurfaces.hidden.metalBuffer)
            let s5End = ProcessInfo.processInfo.systemUptime
            try await commitStandaloneCommand(s5, encodeSeconds: s5End - s5Start)
            try evaluateProjection(old: oldModels?.mlpDown, procedure: .mlpDown,
                                   input: aneSurfaces.hidden, output: aneSurfaces.tokenOutput,
                                   label: "MLP2")
        }

        let s6Start = ProcessInfo.processInfo.systemUptime
        guard let s6 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid final command buffer")
        }
        s6.label = "DiT block \(blockIndex) ANE MLP output"
        if !chunked {
            try encodeANEToToken(s6, aneMajor: aneSurfaces.tokenOutput.metalBuffer,
                                 tokenMajor: mlpBranch, rows: Self.tokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.tokenOutput.planeStrideElements)
        }
        try encodeHalfComputeBoundary(s6, mlpBranch, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(s6, values: mlpBranch,
                                    count: Self.tokens * Self.dim, probe: .mlpBranch)
        }
        try encodeGateAdd(s6, residual: residual, branch: mlpBranch, modulation: mlpModulation,
                          count: Self.tokens * Self.dim, probe: .mlpGateAdd)
        try encodeActivationBoundary(s6, residual)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbeF32(s6, values: residual,
                                       count: Self.tokens * Self.dim, probe: .mlpResidual)
        }
        let mlpSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(s6, source: residual, key: "dit.diagnostic.afterMLP.f32")
        }
        let s6End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s6, encodeSeconds: s6End - s6Start)

        streamer.complete(slot)
        slotReleased = true
        if let diagnosticBranchCompleted {
            try diagnosticBranchCompleted("self", selfSnapshot!)
            try diagnosticBranchCompleted("cross", crossSnapshot!)
            try diagnosticBranchCompleted("mlp", mlpSnapshot!)
        }
        if usesMultiProc {
            try aneMultiProcCache?.complete(block: blockIndex)
        } else {
            try aneModelCache?.complete(
                block: blockIndex,
                crossKVReady: crossKVCache?.isReady(blockIndex) ?? false)
        }
        schedulerCompleted = true
        metrics?.endBlock()
        context.refreshDiagnostics()
        metrics?.recordMemory(allocated: context.currentAllocatedSize,
                              available: UInt64(os_proc_available_memory()))
    }

'''
replace_between(
    "AnimaXS/Runtime/Metal/DiTBlockExecutor.swift",
    "    /// Shared heterogeneous ANE/Metal block graph. The old backend supplies\n",
    "    private func evaluateANEProjection(\n",
    new_ane_method,
)

# Factor token-branch postprocessing so chunked ANE outputs can enter the exact
# same boundary/gate-add path as a single surface output.
old = r'''    private func encodeANEAttentionOutput(
        _ command: MTLCommandBuffer,
        aneOutput: A12ANESurface,
        residual: MTLBuffer,
        modulation: MTLBuffer,
        cross: Bool
    ) throws {
        let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try encodeANEToToken(
            command, aneMajor: aneOutput.metalBuffer, tokenMajor: branch,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneOutput.planeStrideElements)
        try encodeHalfComputeBoundary(command, branch, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: branch, count: Self.tokens * Self.dim,
                                    probe: cross ? .crossBranch : .selfBranch)
        }
        try encodeGateAdd(command, residual: residual, branch: branch, modulation: modulation,
                          count: Self.tokens * Self.dim,
                          probe: cross ? .crossGateAdd : .selfGateAdd)
        try encodeActivationBoundary(command, residual)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbeF32(command, values: residual, count: Self.tokens * Self.dim,
                                       probe: cross ? .crossResidual : .selfResidual)
        }
    }
'''
new = r'''    private func encodeANEAttentionOutput(
        _ command: MTLCommandBuffer,
        aneOutput: A12ANESurface,
        residual: MTLBuffer,
        modulation: MTLBuffer,
        cross: Bool
    ) throws {
        let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try encodeANEToToken(
            command, aneMajor: aneOutput.metalBuffer, tokenMajor: branch,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneOutput.planeStrideElements)
        try encodeAttentionOutputFromToken(
            command, branch: branch, residual: residual,
            modulation: modulation, cross: cross)
    }

    private func encodeAttentionOutputFromToken(
        _ command: MTLCommandBuffer,
        branch: MTLBuffer,
        residual: MTLBuffer,
        modulation: MTLBuffer,
        cross: Bool
    ) throws {
        try encodeHalfComputeBoundary(command, branch, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: branch, count: Self.tokens * Self.dim,
                                    probe: cross ? .crossBranch : .selfBranch)
        }
        try encodeGateAdd(command, residual: residual, branch: branch, modulation: modulation,
                          count: Self.tokens * Self.dim,
                          probe: cross ? .crossGateAdd : .selfGateAdd)
        try encodeActivationBoundary(command, residual)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbeF32(command, values: residual, count: Self.tokens * Self.dim,
                                       probe: cross ? .crossResidual : .selfResidual)
        }
    }
'''
replace("AnimaXS/Runtime/Metal/DiTBlockExecutor.swift", old, new)

# MLP activation can operate on one ANE 1024-token chunk or the full MPS tensor.
replace(
    "AnimaXS/Runtime/Metal/DiTBlockExecutor.swift",
    """    private func encodeMLPActivation(\n        _ command: MTLCommandBuffer, hiddenHalf: MTLBuffer\n    ) throws {\n""",
    """    private func encodeMLPActivation(\n        _ command: MTLCommandBuffer, hiddenHalf: MTLBuffer, rows: Int = Self.tokens\n    ) throws {\n""",
)
path = "AnimaXS/Runtime/Metal/DiTBlockExecutor.swift"
text = read(path)
start = text.index("    private func encodeMLPActivation(")
end = text.index("    private func encodeTokenToANE(", start)
segment = text[start:end]
segment = segment.replace("Self.tokens * Self.hidden", "rows * Self.hidden")
text = text[:start] + segment + text[end:]
write(path, text)

# Layout bridge offsets let a full 4096-token buffer feed/collect 1024-token ANE chunks.
old = r'''    private func encodeTokenToANE(
        _ command: MTLCommandBuffer,
        tokenMajor: MTLBuffer,
        aneMajor: MTLBuffer,
        rows: Int,
        channels: Int,
        planeStrideElements: UInt
    ) throws {
        try encodeANELayoutBridge(
            command, kernel: "dit_token_to_ane_f16", source: tokenMajor,
            destination: aneMajor, rows: rows, channels: channels,
            planeStrideElements: planeStrideElements)
    }

    private func encodeANEToToken(
        _ command: MTLCommandBuffer,
        aneMajor: MTLBuffer,
        tokenMajor: MTLBuffer,
        rows: Int,
        channels: Int,
        planeStrideElements: UInt
    ) throws {
        try encodeANELayoutBridge(
            command, kernel: "dit_ane_to_token_f16", source: aneMajor,
            destination: tokenMajor, rows: rows, channels: channels,
            planeStrideElements: planeStrideElements)
    }

    private func encodeANELayoutBridge(
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
            throw AnimapkError.validation("failed to create \(kernel) encoder")
        }
        var rows32 = UInt32(rows)
        var channels32 = UInt32(channels)
        var stride32 = UInt32(planeStrideElements)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
'''
new = r'''    private func encodeTokenToANE(
        _ command: MTLCommandBuffer,
        tokenMajor: MTLBuffer,
        aneMajor: MTLBuffer,
        rows: Int,
        channels: Int,
        planeStrideElements: UInt,
        tokenRowOffset: Int = 0
    ) throws {
        let tokenOffset = tokenRowOffset * channels * MemoryLayout<Float16>.stride
        try encodeANELayoutBridge(
            command, kernel: "dit_token_to_ane_f16", source: tokenMajor,
            sourceOffset: tokenOffset, destination: aneMajor, destinationOffset: 0,
            rows: rows, channels: channels, planeStrideElements: planeStrideElements)
    }

    private func encodeANEToToken(
        _ command: MTLCommandBuffer,
        aneMajor: MTLBuffer,
        tokenMajor: MTLBuffer,
        rows: Int,
        channels: Int,
        planeStrideElements: UInt,
        tokenRowOffset: Int = 0
    ) throws {
        let tokenOffset = tokenRowOffset * channels * MemoryLayout<Float16>.stride
        try encodeANELayoutBridge(
            command, kernel: "dit_ane_to_token_f16", source: aneMajor,
            sourceOffset: 0, destination: tokenMajor, destinationOffset: tokenOffset,
            rows: rows, channels: channels, planeStrideElements: planeStrideElements)
    }

    private func encodeANELayoutBridge(
        _ command: MTLCommandBuffer,
        kernel: String,
        source: MTLBuffer,
        sourceOffset: Int,
        destination: MTLBuffer,
        destinationOffset: Int,
        rows: Int,
        channels: Int,
        planeStrideElements: UInt
    ) throws {
        let tightBytes = rows * channels * MemoryLayout<Float16>.stride
        guard sourceOffset >= 0, destinationOffset >= 0,
              sourceOffset + tightBytes <= source.length,
              destinationOffset < destination.length else {
            throw AnimapkError.validation("invalid \(kernel) buffer offset/range")
        }
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(kernel) encoder")
        }
        var rows32 = UInt32(rows)
        var channels32 = UInt32(channels)
        var stride32 = UInt32(planeStrideElements)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination, offset: destinationOffset, index: 1)
'''
replace("AnimaXS/Runtime/Metal/DiTBlockExecutor.swift", old, new)


# ---------------------------------------------------------------------------
# Resolution-dependent DiT/VAE geometry.
# ---------------------------------------------------------------------------
replace(
    "AnimaXS/Runtime/Sampler/DiffusionSampler.swift",
    "    static let latentElements = 16 * 64 * 64\n",
    "    static var latentElements: Int { GenerationGeometryRuntime.current.latentElements }\n",
)
replace(
    "AnimaXS/Runtime/Sampler/DiffusionSampler.swift",
    "        let ropeTable = DitRoPE.generate()\n",
    "        let grid = GenerationGeometryRuntime.current.patchGrid\n        let ropeTable = DitRoPE.generate(H: grid, W: grid)\n",
)

path = "AnimaXS/Runtime/Metal/DiTPreparationExecutor.swift"
text = read(path)
text = text.replace("    static let latentElements = 16 * 64 * 64\n", "    static var latentElements: Int { GenerationGeometryRuntime.current.latentElements }\n")
text = text.replace("    static let tokens = 1_024\n", "    static var tokens: Int { GenerationGeometryRuntime.current.ditTokens }\n")
text = text.replace("17 * 64 * 64", "17 * GenerationGeometryRuntime.current.latentSize * GenerationGeometryRuntime.current.latentSize")
text = text.replace("var height: UInt32 = 64, width: UInt32 = 64", "var height = UInt32(GenerationGeometryRuntime.current.latentSize), width = height")
write(path, text)

path = "AnimaXS/Runtime/Metal/DiTFinalLayerExecutor.swift"
text = read(path)
text = text.replace("    private static let tokens = 1_024\n", "    private static var tokens: Int { GenerationGeometryRuntime.current.ditTokens }\n")
text = text.replace("    private static let outputElements = 16 * 64 * 64\n", "    private static var outputElements: Int { GenerationGeometryRuntime.current.latentElements }\n")
text = text.replace("var height: UInt32 = 64, width: UInt32 = 64", "var height = UInt32(GenerationGeometryRuntime.current.latentSize), width = height")
write(path, text)

replace(
    "AnimaXS/Runtime/VAE/VAEDecoder.swift",
    "    private static let latentSize = 64\n    private static let outputSize = 512\n",
    "    private static var latentSize: Int { GenerationGeometryRuntime.current.latentSize }\n    private static var outputSize: Int { GenerationGeometryRuntime.current.imageSize }\n",
)

path = "AnimaXS/Runtime/VAE/Wan21LatentFormat.swift"
text = read(path)
text = text.replace("    static let elementsPerChannel = 64 * 64\n", "    static var elementsPerChannel: Int {\n        let side = GenerationGeometryRuntime.current.latentSize\n        return side * side\n    }\n")
write(path, text)


# ---------------------------------------------------------------------------
# Thread resolution from main UI through coordinator/engine and metrics.
# ---------------------------------------------------------------------------
path = "AnimaXS/Runtime/Generation/GenerationEngine.swift"
text = read(path)
old_sig = '''    func generate(\n        prompt: String,\n        seed: UInt64,\n        models: ResolvedModels,\n        noise: MTLBuffer? = nil,\n        progress: ProgressCallback? = nil,\n        metrics metricsIn: MetricsCollector? = nil,\n        optimization: InferenceOptimizationConfig = .currentBaseline\n    ) async throws -> DecodedRGBA8 {\n'''
new_sig = '''    func generate(\n        prompt: String,\n        seed: UInt64,\n        models: ResolvedModels,\n        noise: MTLBuffer? = nil,\n        progress: ProgressCallback? = nil,\n        metrics metricsIn: MetricsCollector? = nil,\n        optimization: InferenceOptimizationConfig = .currentBaseline,\n        resolution: GenerationResolution = .square512\n    ) async throws -> DecodedRGBA8 {\n        try await GenerationGeometryRuntime.$current.withValue(resolution) {\n            try await generateConfigured(\n                prompt: prompt, seed: seed, models: models, noise: noise,\n                progress: progress, metrics: metricsIn, optimization: optimization)\n        }\n    }\n\n    private func generateConfigured(\n        prompt: String,\n        seed: UInt64,\n        models: ResolvedModels,\n        noise: MTLBuffer?,\n        progress: ProgressCallback?,\n        metrics metricsIn: MetricsCollector?,\n        optimization: InferenceOptimizationConfig\n    ) async throws -> DecodedRGBA8 {\n'''
if text.count(old_sig) != 1:
    raise RuntimeError("GenerationEngine generate signature mismatch")
text = text.replace(old_sig, new_sig)
text = text.replace("        metrics.recordOptimizationConfig(optimization)\n", "        metrics.recordOptimizationConfig(optimization)\n        metrics.recordResolution(GenerationGeometryRuntime.current)\n", 1)
write(path, text)

path = "AnimaXS/Runtime/Generation/GenerationCoordinator.swift"
text = read(path)
text = text.replace(
'''        noise: MTLBuffer? = nil,\n        optimization: InferenceOptimizationConfig = .currentBaseline\n''',
'''        noise: MTLBuffer? = nil,\n        optimization: InferenceOptimizationConfig = .currentBaseline,\n        resolution: GenerationResolution = .square512\n''', 1)
text = text.replace(
'''            prompt: prompt, seed: seed, models: models,\n            noise: noise, optimization: optimization)\n''',
'''            prompt: prompt, seed: seed, models: models,\n            noise: noise, optimization: optimization, resolution: resolution)\n''', 1)
text = text.replace(
'''        noise: MTLBuffer?,\n        optimization: InferenceOptimizationConfig\n''',
'''        noise: MTLBuffer?,\n        optimization: InferenceOptimizationConfig,\n        resolution: GenerationResolution\n''', 1)
text = text.replace(
'''                    metrics: metrics,\n                    optimization: optimization)\n''',
'''                    metrics: metrics,\n                    optimization: optimization,\n                    resolution: resolution)\n''', 1)
write(path, text)

path = "AnimaXS/Runtime/Diagnostics/GenerationMetrics.swift"
text = read(path)
text = text.replace("    var optimizationConfig: InferenceOptimizationConfig?\n", "    var optimizationConfig: InferenceOptimizationConfig?\n    var resolution: GenerationResolution = .square512\n", 1)
text = text.replace("        lines.append(\"Inference configuration\")\n", "        lines.append(\"Inference configuration\")\n        lines.append(\"Resolution: \\(resolution.label)\")\n", 1)
text = text.replace("            if config.linearBackend == .aneHybridW8 && !config.crossKVCache {\n", "            if config.linearBackend.isANEW8 && !config.crossKVCache {\n", 1)
needle = '''    func recordOptimizationConfig(_ config: InferenceOptimizationConfig) {\n        metrics.optimizationConfig = config\n        metrics.numericalMonitoringDisabled = !config.numericalMonitoring\n    }\n'''
replacement = needle + '''\n    func recordResolution(_ resolution: GenerationResolution) {\n        metrics.resolution = resolution\n    }\n'''
if text.count(needle) != 1:
    raise RuntimeError("GenerationMetrics recordOptimizationConfig mismatch")
text = text.replace(needle, replacement)
write(path, text)

path = "AnimaXS/App/ContentView.swift"
text = read(path)
text = text.replace(
'''    @AppStorage("generation.lastSeed")\n    private var seedText = "1337"\n''',
'''    @AppStorage("generation.lastSeed")\n    private var seedText = "1337"\n    @AppStorage("generation.resolution")\n    private var resolutionRaw = GenerationResolution.square512.rawValue\n''', 1)
text = text.replace("                seedSection\n                generationSection\n", "                seedSection\n                resolutionSection\n                generationSection\n", 1)
seed_marker = '''    // MARK: - Generation controls\n'''
resolution_section = '''    // MARK: - Resolution\n\n    private var selectedResolution: GenerationResolution {\n        GenerationResolution(rawValue: resolutionRaw) ?? .square512\n    }\n\n    private var resolutionSection: some View {\n        Section("Resolution") {\n            Picker("Output", selection: $resolutionRaw) {\n                ForEach(GenerationResolution.allCases) { resolution in\n                    Text(resolution.label).tag(resolution.rawValue)\n                }\n            }\n            .pickerStyle(.segmented)\n            .disabled(isGenerating)\n            if selectedResolution == .square1024 {\n                Text("Experimental: 4× DiT tokens and substantially heavier attention/VAE work. ANE projection programs are reused in four exact 1024-token chunks.")\n                    .font(.caption2)\n                    .foregroundStyle(.secondary)\n            }\n        }\n    }\n\n'''
if text.count(seed_marker) != 1:
    raise RuntimeError("ContentView generation marker mismatch")
text = text.replace(seed_marker, resolution_section + seed_marker)
text = text.replace(
'''        coordinator.generate(\n            prompt: prompt, seed: seed, models: models,\n            optimization: config)\n''',
'''        coordinator.generate(\n            prompt: prompt, seed: seed, models: models,\n            optimization: config, resolution: selectedResolution)\n''', 1)
write(path, text)


# ---------------------------------------------------------------------------
# CI-level geometry invariants. No heavy model fixtures required.
# ---------------------------------------------------------------------------
path = "AnimaXSTests/SmokeTests.swift"
text = read(path)
needle = '''    func testSigmaScheduleMatchesHandoff() {\n        let expected: [Float] = [\n            1.0, 0.9546938, 0.90035903, 0.8339981, 0.7511211,\n            0.64468634, 0.50298506, 0.30500895, 0.0\n        ]\n        XCTAssertEqual(ModelConstants.sigma8Step, expected)\n        XCTAssertEqual(ModelConstants.ditBlocks, 28)\n    }\n'''
addition = needle + '''\n    func testGenerationResolutionGeometry() {\n        XCTAssertEqual(GenerationResolution.square512.imageSize, 512)\n        XCTAssertEqual(GenerationResolution.square512.latentSize, 64)\n        XCTAssertEqual(GenerationResolution.square512.patchGrid, 32)\n        XCTAssertEqual(GenerationResolution.square512.ditTokens, 1_024)\n        XCTAssertEqual(GenerationResolution.square512.latentElements, 65_536)\n\n        XCTAssertEqual(GenerationResolution.square1024.imageSize, 1_024)\n        XCTAssertEqual(GenerationResolution.square1024.latentSize, 128)\n        XCTAssertEqual(GenerationResolution.square1024.patchGrid, 64)\n        XCTAssertEqual(GenerationResolution.square1024.ditTokens, 4_096)\n        XCTAssertEqual(GenerationResolution.square1024.latentElements, 262_144)\n    }\n\n    func testGenerationGeometryTaskLocalDefaultsTo512() {\n        XCTAssertEqual(GenerationGeometryRuntime.current, .square512)\n        XCTAssertEqual(DiffusionSampler.latentElements, 65_536)\n        XCTAssertEqual(DiTBlockExecutor.tokens, 1_024)\n    }\n'''
if text.count(needle) != 1:
    raise RuntimeError("SmokeTests sigma marker mismatch")
text = text.replace(needle, addition)
write(path, text)

print("cache + 1024 patch applied successfully")
