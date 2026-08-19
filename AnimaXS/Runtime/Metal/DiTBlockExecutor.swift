import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// P5: generation-local cache of the invariant cross-attention K/V for every
/// DiT block. Cross context is fixed for a generation, so after the first
/// executed step each block's cross K/V (post-projection, post-static-boundary,
/// post-K-RMSNorm) are reused instead of being re-projected every step. EXACT
/// reuse — no approximation; Q stays dynamic and is never cached.
///
/// One contiguous `.storageModePrivate` buffer (~112 MiB for 28 blocks). The
/// CPU never reads it. If allocation fails the cache is nil and callers fall
/// back to the legacy per-step projection path (the experiment fails
/// gracefully, never crashes). The cache belongs to ONE `DiffusionSampler` /
/// one generation; it is never persisted across prompts and never enters a
/// checkpoint.
final class CrossKVCache {
    static let tensorBytes =
        DiTBlockExecutor.contextTokens * DiTBlockExecutor.dim
        * MemoryLayout<Float16>.stride
    static let blockStride = tensorBytes * 2  // K + V per block
    static let blockCount = ModelConstants.ditBlocks

    let buffer: MTLBuffer
    private(set) var ready = [Bool](repeating: false, count: blockCount)

    /// Creates the cache. Returns nil if the device cannot allocate it.
    init?(device: MTLDevice,
          options: MTLResourceOptions = .storageModePrivate) {
        let length = CrossKVCache.blockCount * CrossKVCache.blockStride
        guard let buffer = device.makeBuffer(length: length, options: options) else {
            return nil
        }
        self.buffer = buffer
    }

    func kOffset(block: Int) -> Int { block * Self.blockStride }
    func vOffset(block: Int) -> Int { kOffset(block: block) + Self.tensorBytes }
    func isReady(_ block: Int) -> Bool { ready[block] }
    func markReady(_ block: Int) { ready[block] = true }
}

/// Exact fixed-shape MiniTrainDIT block executed with one streamed block range.
/// Legacy mode keeps residual/modulation/gates in fp32 and projection/attention
/// boundaries in fp16. `bf16Compute` emulates the reference BF16 model dtype
/// while retaining the host-visible fp32 buffers.
/// This object is intentionally non-reentrant because every activation and weight scratch
/// allocation is reused between operations and calls.
final class DiTBlockExecutor {
    typealias DiagnosticBranchCompleted = (_ branch: String, _ residual: MTLBuffer) throws -> Void
    /// Diagnostic-only block-internal tensor callback. Oracle E uses this only
    /// for block 0; normal inference passes nil and pays no snapshot/readback cost.
    typealias DiagnosticStageCompleted = (_ stage: String, _ tensor: MTLBuffer) throws -> Void
    static let tokens = 1_024
    static let contextTokens = 512
    static let dim = 2_048
    static let contextDim = 1_024
    static let heads = 16
    static let headDim = 128
    static let hidden = 8_192
    static let modulationHidden = 256
    static let modulationSize = 6_144
    static let eps: Float = 1e-6

    private let context: MetalContext
    private let file: AnimapkFile
    private let blockRanges: [AnimapkExecutionRange]
    private let streamer: WeightStreamer
    private let buffers: BufferPool
    private let linear: LinearExecutor
    private let attention: AttentionExecutor
    private let activationNumerics: ActivationNumerics
    private let monitor: NumericalMonitor?
    /// Immutable optimization snapshot for this executor's lifetime (captured
    /// at Generate). Controls the P3-A/P3-B fused paths.
    private let optimization: InferenceOptimizationConfig
    /// P5: per-generation cross-attention K/V cache (nil when the toggle is off
    /// or the buffer could not be allocated). Shared across all steps of one
    /// generation; never persisted.
    private let crossKVCache: CrossKVCache?
    /// A12/H11 ANE projection offload. Non-nil only for the explicit
    /// `.aneHybridW8` backend; the legacy path never touches private runtime
    /// symbols or allocates IOSurfaces.
    private let aneModelCache: ANEW8DiTModelCache?
    private let aneSurfaces: ANEW8DiTSurfaces?
    /// Run telemetry collector (nil in tests / diagnostic-only construction).
    /// Propagated to the child linear/attention executors so their cheap tile
    /// counters land in the SAME run metrics object (never a separate one).
    var metrics: MetricsCollector? {
        didSet {
            linear.metrics = metrics
            attention.metrics = metrics
        }
    }
    private var emulatesBF16: Bool { activationNumerics == .bf16Compute }

    /// Number of weight slots backing this block's execution (2 with ping-pong
    /// ON, 1 with it OFF). Used by `DitForward` to drive the loop shape.
    var slotCount: Int { streamer.slotCount }

    init(context: MetalContext, file: AnimapkFile,
         attentionNumerics: AttentionNumerics = .legacy,
         activationNumerics: ActivationNumerics = .legacy,
         monitor: NumericalMonitor? = nil,
         optimization: InferenceOptimizationConfig = .currentBaseline,
         crossKVCache: CrossKVCache? = nil) throws {
        let ranges: [AnimapkExecutionRange]
        if optimization.linearBackend == .aneHybridW8 {
            ranges = try DiTANEHybridMetalLocator(file: file).blocks
        } else {
            ranges = try DiTBlockLocator(file: file).blocks
        }
        guard let maximum = ranges.map(\.length).max(), maximum <= UInt64(Int.max) else {
            throw AnimapkError.validation("invalid DiT execution ranges")
        }
        self.context = context
        self.file = file
        self.blockRanges = ranges
        self.optimization = optimization
        self.crossKVCache = crossKVCache
        if optimization.linearBackend == .aneHybridW8 {
            self.aneModelCache = try ANEW8DiTModelCache(file: file)
            self.aneSurfaces = try ANEW8DiTSurfaces(device: context.device)
        } else {
            self.aneModelCache = nil
            self.aneSurfaces = nil
        }
        // Ping-pong ON keeps the existing two-slot streamer; OFF measures the
        // same generation with one slot and no look-ahead prefetch.
        let slotCount = optimization.pingPongWeightStreaming ? 2 : 1
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(maximum), slotCount: slotCount)
        self.buffers = BufferPool(device: context.device)
        self.linear = LinearExecutor(
            context: context, tileRows: optimization.linearTileRows,
            directMPSIO: optimization.directLinearMPSIO,
            linearBackend: optimization.linearBackend)
        self.activationNumerics = activationNumerics
        self.monitor = monitor
        let effectiveAttention = activationNumerics == .bf16Compute && attentionNumerics == .legacy
            ? .bf16Compute : attentionNumerics
        self.attention = AttentionExecutor(
            context: context, tileRows: optimization.attentionTileRows,
            numerics: effectiveAttention, monitor: monitor,
            attentionBackend: optimization.attentionBackend)
    }

    /// Mutates `residual` in place. All input buffers are tightly packed and use these types:
    /// residual/emb/adalnLora/rope fp32, crossContext fp16.
    ///
    /// `slot` selects the weight slot the block's weights were (or will be)
    /// loaded into. `prefetchIndex`/`prefetchSlot` request the next block's
    /// weights to be copied into the other slot AFTER this block's command
    /// buffer is committed and BEFORE it is awaited, so the CPU memcpy hides
    /// behind GPU execution (Phase 12 ping-pong). The streamer itself refuses
    /// to overwrite a slot still in flight.
    func execute(
        blockIndex: Int,
        residual: MTLBuffer,
        emb: MTLBuffer,
        adalnLora: MTLBuffer,
        crossContext: MTLBuffer,
        rope: MTLBuffer,
        slot: Int = 0,
        prefetchIndex: Int? = nil,
        prefetchSlot: Int = 0,
        diagnosticBranchCompleted: DiagnosticBranchCompleted? = nil,
        diagnosticStageCompleted: DiagnosticStageCompleted? = nil
    ) async throws {
        try validateInputs(residual: residual, emb: emb, adalnLora: adalnLora,
                           crossContext: crossContext, rope: rope)
        if optimization.linearBackend == .aneHybridW8 {
            try await executeANEHybridBlock(
                blockIndex: blockIndex, residual: residual, emb: emb,
                adalnLora: adalnLora, crossContext: crossContext, rope: rope,
                slot: slot, prefetchIndex: prefetchIndex, prefetchSlot: prefetchSlot,
                diagnosticBranchCompleted: diagnosticBranchCompleted,
                diagnosticStageCompleted: diagnosticStageCompleted)
            return
        }
        let range = try blockRange(blockIndex)
        metrics?.beginBlock(blockIndex)
        // Load the block's weights unless a prefetch (prologue or previous
        // iteration) already placed them in this slot.
        if streamer.loadedLogicalIndexes[slot] != blockIndex {
            let copyStart = ProcessInfo.processInfo.systemUptime
            let result = try streamer.load(
                range, from: file, slot: slot,
                mode: optimization.noCopyWeightSource ? .noCopy : .copied)
            if result.mode == .noCopy {
                // P6: memcpy eliminated — prove it in the metrics.
                metrics?.recordMmapNoCopyBytes(result.noCopyBytes)
            } else {
                metrics?.recordWeightCopy(
                    bytes: Int(range.length),
                    seconds: ProcessInfo.processInfo.systemUptime - copyStart)
            }
        }
        let encodeStart = ProcessInfo.processInfo.systemUptime
        let weights = try BlockWeights(range: range, ring: streamer.buffer(for: slot))
        guard let command = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create DiT block command buffer")
        }

        let siluEmb = buffer("dit.siluEmb.f32", Self.dim, Float.self)
        try encodeUnary(command, kernel: "silu", input: emb, output: siluEmb, count: Self.dim)
        try encodeComputeBoundary(command, siluEmb, count: Self.dim)
        try encodeAttentionBranch(command, residual: residual, crossContext: crossContext,
                                  rope: rope, siluEmb: siluEmb, adalnLora: adalnLora,
                                  weights: weights, cross: false, slot: slot,
                                  blockIndex: blockIndex)
        let selfSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(command, source: residual, key: "dit.diagnostic.afterSelf.f32")
        }
        try encodeAttentionBranch(command, residual: residual, crossContext: crossContext,
                                  rope: rope, siluEmb: siluEmb, adalnLora: adalnLora,
                                  weights: weights, cross: true, slot: slot,
                                  blockIndex: blockIndex)
        let crossSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(command, source: residual, key: "dit.diagnostic.afterCross.f32")
        }
        try encodeMLP(command, residual: residual, siluEmb: siluEmb,
                      adalnLora: adalnLora, weights: weights)
        let mlpSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(command, source: residual, key: "dit.diagnostic.afterMLP.f32")
        }

        let encodeEnd = ProcessInfo.processInfo.systemUptime
        streamer.markInFlight(slot)
        // Metal requires the completed handler to be registered BEFORE commit
        // (addCompletedHandler after commit is a hard assertion). Store the
        // continuation in a gate, register the handler, commit, then run the
        // next-block prefetch memcpy while the GPU executes, and finally await.
        let gate = CommandBufferGate()
        let slotStreamer = streamer
        command.addCompletedHandler { [weak slotStreamer] completed in
            slotStreamer?.complete(slot)
            if let error = completed.error { gate.resume(throwing: error) }
            else { gate.resume() }
        }
        command.commit()
        // Ping-pong prefetch: copy the next block's weights into the other
        // slot while this command buffer executes on the GPU. Safe because the
        // other slot's previous user (block i-1) was already awaited. If the
        // prefetch fails, still await the in-flight buffer before propagating.
        var prefetchError: Error?
        if let prefetchIndex {
            do {
                let nextRange = try blockRange(prefetchIndex)
                let copyStart = ProcessInfo.processInfo.systemUptime
                let result = try streamer.load(
                    nextRange, from: file, slot: prefetchSlot,
                    mode: optimization.noCopyWeightSource ? .noCopy : .copied)
                if result.mode == .noCopy {
                    // P6: prefetch memcpy eliminated — record the no-copy bytes.
                    metrics?.recordMmapNoCopyBytes(result.noCopyBytes)
                } else {
                    metrics?.recordWeightCopy(
                        bytes: Int(nextRange.length),
                        seconds: ProcessInfo.processInfo.systemUptime - copyStart)
                }
            } catch {
                prefetchError = error
            }
        }
        try await gate.wait()
        if let prefetchError { throw prefetchError }
        let done = ProcessInfo.processInfo.systemUptime
        let gpuSeconds = (command.gpuStartTime > 0 && command.gpuEndTime >= command.gpuStartTime)
            ? command.gpuEndTime - command.gpuStartTime : 0
        metrics?.recordGPUCommand(seconds: gpuSeconds)
        metrics?.recordEncode(seconds: encodeEnd - encodeStart)
        metrics?.recordHostWait(seconds: (done - encodeEnd) - gpuSeconds)
        metrics?.endBlock()
        // Per-block memory sampling (cheap: no extra GPU sync — the block's
        // command buffer has already completed).
        context.refreshDiagnostics()
        metrics?.recordMemory(
            allocated: context.currentAllocatedSize,
            available: UInt64(os_proc_available_memory()))
        if let diagnosticBranchCompleted {
            try diagnosticBranchCompleted("self", selfSnapshot!)
            try diagnosticBranchCompleted("cross", crossSnapshot!)
            try diagnosticBranchCompleted("mlp", mlpSnapshot!)
        }
    }

    /// A12/H11 hybrid block path matching the device-proven split: all large
    /// W8 projection GEMMs execute on ANE while the exact production AdaLN,
    /// learned RMSNorm, RoPE, attention, GELU, probes, gates and residual
    /// boundaries remain on Metal. IOSurface-backed MTLBuffers provide the
    /// handoff; no activation is copied through CPU memory.
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
        diagnosticBranchCompleted: DiagnosticBranchCompleted?,
        diagnosticStageCompleted: DiagnosticStageCompleted?
    ) async throws {
        guard let aneModelCache, let aneSurfaces else {
            throw AnimapkError.validation("ANE hybrid backend was selected without an initialized ANE runtime")
        }
        metrics?.beginBlock(blockIndex)

        // P5 readiness is the source of truth (not diffusion step number).
        // On a hit the scheduler loads only the six dynamic ANE programs; on a
        // miss it supplies full8 so crossK/crossV can populate the exact cache.
        let crossCacheHit = crossKVCache?.isReady(blockIndex) ?? false
        var schedulerCompleted = false
        defer {
            if !schedulerCompleted { aneModelCache.abortTraversal() }
        }
        let modelResult = try aneModelCache.scheduledModels(
            for: blockIndex, kvWarm: crossCacheHit)
        if modelResult.newlyLoadedMilliseconds > 0 {
            metrics?.recordANEModelLoad(seconds: modelResult.newlyLoadedMilliseconds / 1_000.0)
        }
        if modelResult.waitMilliseconds > 0 {
            metrics?.recordHostWait(seconds: modelResult.waitMilliseconds / 1_000.0)
        }
        let models = modelResult.models
        if !crossCacheHit && !models.hasCrossKVModels {
            throw AnimapkError.validation(
                "ANE scheduler supplied a six-program block on cross-K/V cache miss")
        }

        let range = try blockRange(blockIndex)
        if streamer.loadedLogicalIndexes[slot] != blockIndex {
            let copyStart = ProcessInfo.processInfo.systemUptime
            let result = try streamer.load(range, from: file, slot: slot, mode: .copied)
            guard result.mode != .noCopy else {
                throw AnimapkError.validation("ANE hybrid backend must not use mmap no-copy weights")
            }
            metrics?.recordWeightCopy(
                bytes: Int(range.length),
                seconds: ProcessInfo.processInfo.systemUptime - copyStart)
        }
        let weights = try ANEBlockWeights(range: range, ring: streamer.buffer(for: slot))
        streamer.markInFlight(slot)
        var slotReleased = false
        defer {
            if !slotReleased { streamer.complete(slot) }
        }

        // Shared per-block embedding activation, exactly as the legacy path.
        let siluEmb = buffer("dit.siluEmb.f32", Self.dim, Float.self)

        // ---- S0 Metal: self AdaLN/input -> ANE layout -----------------------
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
        try encodeTokenToANE(
            s0, tokenMajor: selfInput.projectionInput, aneMajor: aneSurfaces.tokenInput.metalBuffer,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        let s0End = ProcessInfo.processInfo.systemUptime

        // Preserve the original ping-pong overlap: the current slot remains
        // in-flight for the whole heterogeneous block, while the other slot can
        // receive the next block immediately after the first Metal submit.
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
                metrics?.recordWeightCopy(
                    bytes: Int(nextRange.length),
                    seconds: ProcessInfo.processInfo.systemUptime - copyStart)
            } catch {
                prefetchError = error
            }
        }
        try await gate0.wait()
        if let prefetchError { throw prefetchError }
        recordCompletedCommand(
            s0, encodeSeconds: s0End - s0Start,
            hostWindowSeconds: ProcessInfo.processInfo.systemUptime - wait0Start)
        let captureSelfStages = blockIndex == 0 && diagnosticStageCompleted != nil
        if captureSelfStages, let diagnosticStageCompleted {
            try diagnosticStageCompleted("self.modulation", selfInput.modulation)
            try diagnosticStageCompleted("self.projection_input", selfInput.projectionInput)
        }

        try evaluateANEQKV(
            models.selfQKV, input: aneSurfaces.tokenInput,
            q: aneSurfaces.q, k: aneSurfaces.k, v: aneSurfaces.v,
            blockIndex: blockIndex)

        // ---- S1 Metal: exact self Q/K transforms + attention ----------------
        let s1Start = ProcessInfo.processInfo.systemUptime
        guard let s1 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid self-attention command buffer")
        }
        s1.label = "DiT block \(blockIndex) ANE self attention"
        let qToken = buffer("dit.q.token.f16", Self.tokens * Self.dim, Float16.self)
        let kToken = buffer("dit.k.token.f16", Self.tokens * Self.dim, Float16.self)
        let vToken = buffer("dit.v.token.f16", Self.tokens * Self.dim, Float16.self)
        try encodeANEToToken(s1, aneMajor: aneSurfaces.q.metalBuffer, tokenMajor: qToken,
                             rows: Self.tokens, channels: Self.dim,
                             planeStrideElements: aneSurfaces.q.planeStrideElements)
        try encodeANEToToken(s1, aneMajor: aneSurfaces.k.metalBuffer, tokenMajor: kToken,
                             rows: Self.tokens, channels: Self.dim,
                             planeStrideElements: aneSurfaces.k.planeStrideElements)
        try encodeANEToToken(s1, aneMajor: aneSurfaces.v.metalBuffer, tokenMajor: vToken,
                             rows: Self.tokens, channels: Self.dim,
                             planeStrideElements: aneSurfaces.v.planeStrideElements)
        let rawQSnapshot = captureSelfStages
            ? try encodeSnapshotHalf(s1, source: qToken, key: "dit.diagnostic.self.qRaw.f16",
                                     elements: Self.tokens * Self.dim) : nil
        let rawKSnapshot = captureSelfStages
            ? try encodeSnapshotHalf(s1, source: kToken, key: "dit.diagnostic.self.kRaw.f16",
                                     elements: Self.tokens * Self.dim) : nil
        let rawVSnapshot = captureSelfStages
            ? try encodeSnapshotHalf(s1, source: vToken, key: "dit.diagnostic.self.vRaw.f16",
                                     elements: Self.tokens * Self.dim) : nil
        let selfAttention = try encodeANEAttentionMath(
            s1, qToken: qToken, kToken: kToken, vToken: vToken,
            cross: false, blockIndex: blockIndex, rope: rope, weights: weights, slot: slot,
            projectedKVAvailable: true, capturePostTransform: captureSelfStages)
        try encodeTokenToANE(
            s1, tokenMajor: selfAttention.attended, aneMajor: aneSurfaces.tokenInput.metalBuffer,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        let s1End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s1, encodeSeconds: s1End - s1Start)
        if captureSelfStages, let diagnosticStageCompleted {
            try diagnosticStageCompleted("self.q_raw", rawQSnapshot!)
            try diagnosticStageCompleted("self.k_raw", rawKSnapshot!)
            try diagnosticStageCompleted("self.v_raw", rawVSnapshot!)
            try diagnosticStageCompleted("self.q_post_rope", selfAttention.qPostTransform!)
            try diagnosticStageCompleted("self.k_post_rope", selfAttention.kPostTransform!)
            try diagnosticStageCompleted("self.attended", selfAttention.attended)
        }

        try evaluateANEProjection(
            models.selfO, input: aneSurfaces.tokenInput, output: aneSurfaces.tokenOutput,
            label: "self output", blockIndex: blockIndex)

        // ---- S2 Metal: self residual + cross AdaLN/input --------------------
        let s2Start = ProcessInfo.processInfo.systemUptime
        guard let s2 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid cross-input command buffer")
        }
        s2.label = "DiT block \(blockIndex) ANE cross input"
        let selfBranch = try encodeANEAttentionOutput(
            s2, aneOutput: aneSurfaces.tokenOutput, residual: residual,
            modulation: selfInput.modulation, cross: false)
        let selfSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(s2, source: residual, key: "dit.diagnostic.afterSelf.f32")
        }
        let crossInput = try encodeAttentionInput(
            s2, residual: residual, siluEmb: siluEmb, adalnLora: adalnLora,
            weights: weights, cross: true)
        try encodeTokenToANE(
            s2, tokenMajor: crossInput.projectionInput, aneMajor: aneSurfaces.tokenInput.metalBuffer,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        // `crossCacheHit` was resolved before ANE model acquisition so the
        // scheduler can omit crossK/crossV entirely on a real cache hit.
        if !crossCacheHit {
            try encodeTokenToANE(
                s2, tokenMajor: crossContext, aneMajor: aneSurfaces.contextInput.metalBuffer,
                rows: Self.contextTokens, channels: Self.contextDim,
                planeStrideElements: aneSurfaces.contextInput.planeStrideElements)
        }
        let s2End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s2, encodeSeconds: s2End - s2Start)
        if captureSelfStages, let diagnosticStageCompleted {
            try diagnosticStageCompleted("self.branch", selfBranch)
        }

        try evaluateANEProjection(
            models.crossQ, input: aneSurfaces.tokenInput, output: aneSurfaces.q,
            label: "cross Q", blockIndex: blockIndex)
        if !crossCacheHit {
            try evaluateANEProjection(
                models.crossK, input: aneSurfaces.contextInput, output: aneSurfaces.contextK,
                label: "cross K", blockIndex: blockIndex)
            try evaluateANEProjection(
                models.crossV, input: aneSurfaces.contextInput, output: aneSurfaces.contextV,
                label: "cross V", blockIndex: blockIndex)
        }

        // ---- S3 Metal: exact cross Q/K transforms + attention ---------------
        let s3Start = ProcessInfo.processInfo.systemUptime
        guard let s3 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid cross-attention command buffer")
        }
        s3.label = "DiT block \(blockIndex) ANE cross attention"
        let crossQToken = buffer("dit.q.token.f16", Self.tokens * Self.dim, Float16.self)
        let crossKCount = Self.contextTokens * Self.dim
        let crossKToken = buffer("dit.k.token.f16", crossKCount, Float16.self)
        let crossVToken = buffer("dit.v.token.f16", crossKCount, Float16.self)
        try encodeANEToToken(s3, aneMajor: aneSurfaces.q.metalBuffer, tokenMajor: crossQToken,
                             rows: Self.tokens, channels: Self.dim,
                             planeStrideElements: aneSurfaces.q.planeStrideElements)
        if !crossCacheHit {
            try encodeANEToToken(s3, aneMajor: aneSurfaces.contextK.metalBuffer, tokenMajor: crossKToken,
                                 rows: Self.contextTokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.contextK.planeStrideElements)
            try encodeANEToToken(s3, aneMajor: aneSurfaces.contextV.metalBuffer, tokenMajor: crossVToken,
                                 rows: Self.contextTokens, channels: Self.dim,
                                 planeStrideElements: aneSurfaces.contextV.planeStrideElements)
        }
        let crossAttention = try encodeANEAttentionMath(
            s3, qToken: crossQToken, kToken: crossKToken, vToken: crossVToken,
            cross: true, blockIndex: blockIndex, rope: rope, weights: weights, slot: slot,
            projectedKVAvailable: !crossCacheHit)
        try encodeTokenToANE(
            s3, tokenMajor: crossAttention.attended, aneMajor: aneSurfaces.tokenInput.metalBuffer,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        let s3End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s3, encodeSeconds: s3End - s3Start)

        try evaluateANEProjection(
            models.crossO, input: aneSurfaces.tokenInput, output: aneSurfaces.tokenOutput,
            label: "cross output", blockIndex: blockIndex)

        // ---- S4 Metal: cross residual + MLP AdaLN/input ---------------------
        let s4Start = ProcessInfo.processInfo.systemUptime
        guard let s4 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid MLP-input command buffer")
        }
        s4.label = "DiT block \(blockIndex) ANE MLP input"
        _ = try encodeANEAttentionOutput(
            s4, aneOutput: aneSurfaces.tokenOutput, residual: residual,
            modulation: crossInput.modulation, cross: true)
        let crossSnapshot = try diagnosticBranchCompleted.map { _ in
            try encodeSnapshot(s4, source: residual, key: "dit.diagnostic.afterCross.f32")
        }
        let (mlpModulation, mlpInput) = try encodeMLPInput(
            s4, residual: residual, siluEmb: siluEmb,
            adalnLora: adalnLora, weights: weights)
        try encodeTokenToANE(
            s4, tokenMajor: mlpInput, aneMajor: aneSurfaces.tokenInput.metalBuffer,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneSurfaces.tokenInput.planeStrideElements)
        let s4End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s4, encodeSeconds: s4End - s4Start)

        try evaluateANEProjection(
            models.mlpUp, input: aneSurfaces.tokenInput, output: aneSurfaces.hidden,
            label: "MLP1", blockIndex: blockIndex)

        // ---- S5 Metal: exact production GELU, directly on shared H11 buffer -
        guard aneSurfaces.hidden.planeStrideElements == UInt(Self.tokens) else {
            throw AnimapkError.validation("ANE hidden surface unexpectedly padded at spatial \(Self.tokens)")
        }
        let s5Start = ProcessInfo.processInfo.systemUptime
        guard let s5 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid GELU command buffer")
        }
        s5.label = "DiT block \(blockIndex) ANE GELU"
        try encodeHalfComputeBoundary(s5, aneSurfaces.hidden.metalBuffer, count: Self.tokens * Self.hidden)
        try encodeMLPActivation(s5, hiddenHalf: aneSurfaces.hidden.metalBuffer)
        let s5End = ProcessInfo.processInfo.systemUptime
        try await commitStandaloneCommand(s5, encodeSeconds: s5End - s5Start)

        try evaluateANEProjection(
            models.mlpDown, input: aneSurfaces.hidden, output: aneSurfaces.tokenOutput,
            label: "MLP2", blockIndex: blockIndex)

        // ---- S6 Metal: MLP boundary/gate/residual ---------------------------
        let s6Start = ProcessInfo.processInfo.systemUptime
        guard let s6 = context.commandQueue.makeCommandBuffer() else {
            throw AnimapkError.validation("failed to create ANE hybrid final command buffer")
        }
        s6.label = "DiT block \(blockIndex) ANE MLP output"
        let mlpBranch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try encodeANEToToken(
            s6, aneMajor: aneSurfaces.tokenOutput.metalBuffer, tokenMajor: mlpBranch,
            rows: Self.tokens, channels: Self.dim,
            planeStrideElements: aneSurfaces.tokenOutput.planeStrideElements)
        try encodeHalfComputeBoundary(s6, mlpBranch, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(s6, values: mlpBranch, count: Self.tokens * Self.dim, probe: .mlpBranch)
        }
        try encodeGateAdd(s6, residual: residual, branch: mlpBranch, modulation: mlpModulation,
                          count: Self.tokens * Self.dim, probe: .mlpGateAdd)
        try encodeActivationBoundary(s6, residual)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbeF32(s6, values: residual, count: Self.tokens * Self.dim, probe: .mlpResidual)
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
        // Only retire after every consumer of this block's ANE outputs has
        // completed. Pinned cache-miss blocks can now shed crossK/crossV.
        try aneModelCache.complete(
            block: blockIndex,
            crossKVReady: crossKVCache?.isReady(blockIndex) ?? false)
        schedulerCompleted = true
        metrics?.endBlock()
        context.refreshDiagnostics()
        metrics?.recordMemory(
            allocated: context.currentAllocatedSize,
            available: UInt64(os_proc_available_memory()))
    }

    private func evaluateANEProjection(
        _ model: A12ANEProjectionModel,
        input: A12ANESurface,
        output: A12ANESurface,
        label: String,
        blockIndex: Int
    ) throws {
        var milliseconds = 0.0
        do {
            _ = try model.evaluateInput(input, output: output, milliseconds: &milliseconds)
        } catch {
            throw AnimapkError.validation(
                "ANE \(label) failed for block \(blockIndex): \(error.localizedDescription)")
        }
        metrics?.recordANEEvaluation(seconds: milliseconds / 1_000.0)
    }

    private func evaluateANEQKV(
        _ model: A12ANEQKVModel,
        input: A12ANESurface,
        q: A12ANESurface,
        k: A12ANESurface,
        v: A12ANESurface,
        blockIndex: Int
    ) throws {
        var milliseconds = 0.0
        do {
            _ = try model.evaluateInput(
                input, qOutput: q, kOutput: k, vOutput: v, milliseconds: &milliseconds)
        } catch {
            throw AnimapkError.validation(
                "ANE fused self QKV failed for block \(blockIndex): \(error.localizedDescription)")
        }
        metrics?.recordANEEvaluation(seconds: milliseconds / 1_000.0)
    }

    private func encodeAttentionInput(
        _ command: MTLCommandBuffer,
        residual: MTLBuffer,
        siluEmb: MTLBuffer,
        adalnLora: MTLBuffer,
        weights: any DiTAuxWeights,
        cross: Bool
    ) throws -> (modulation: MTLBuffer, projectionInput: MTLBuffer) {
        let modulation = try encodeModulation(
            command, siluEmb: siluEmb, adalnLora: adalnLora,
            w1: cross ? weights.modCross1 : weights.modSelf1,
            w2: cross ? weights.modCross2 : weights.modSelf2)
        let projectionInput = buffer("dit.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        if optimization.fusedNormModulation {
            try encodeFusedNormModulate(
                command, residual: residual, modulation: modulation,
                output: projectionInput, rows: Self.tokens, columns: Self.dim,
                probe: cross ? .crossProjectionInput : .selfProjectionInput)
            metrics?.recordFusedTrafficSaved(
                UInt64(Self.tokens * Self.dim * MemoryLayout<Float>.stride) * 2)
        } else {
            let norm = buffer("dit.norm.f32", Self.tokens * Self.dim, Float.self)
            let modulated = buffer("dit.modulated.f32", Self.tokens * Self.dim, Float.self)
            try encodeLayerNorm(command, input: residual, output: norm, rows: Self.tokens, columns: Self.dim)
            try encodeModulate(command, normalized: norm, modulation: modulation,
                               output: modulated, count: Self.tokens * Self.dim)
            try encodeFloatToComputeHalf(
                command, input: modulated, output: projectionInput,
                count: Self.tokens * Self.dim,
                probe: cross ? .crossProjectionInput : .selfProjectionInput)
        }
        return (modulation, projectionInput)
    }

    /// Consumes raw projection outputs (or cached post-transform cross K/V) and
    /// reproduces the legacy numerical boundaries, learned RMSNorm/RoPE and
    /// attention path exactly. The returned buffer is token-major fp16 and is
    /// ready for the ANE output projection.
    private struct ANEAttentionMathResult {
        let attended: MTLBuffer
        let qPostTransform: MTLBuffer?
        let kPostTransform: MTLBuffer?
    }

    private func encodeANEAttentionMath(
        _ command: MTLCommandBuffer,
        qToken: MTLBuffer,
        kToken: MTLBuffer,
        vToken: MTLBuffer,
        cross: Bool,
        blockIndex: Int,
        rope: MTLBuffer,
        weights: any DiTAuxWeights,
        slot: Int,
        projectedKVAvailable: Bool,
        capturePostTransform: Bool = false
    ) throws -> ANEAttentionMathResult {
        let keyRows = cross ? Self.contextTokens : Self.tokens
        let kTokenCount = keyRows * Self.dim
        let cacheEnabled = cross && crossKVCache != nil
        let cacheHit = cacheEnabled && (crossKVCache?.isReady(blockIndex) ?? false)

        if cacheHit, let cache = crossKVCache {
            guard let encoder = command.makeBlitCommandEncoder() else {
                throw AnimapkError.validation("failed to create ANE cross-K/V cache blit encoder")
            }
            encoder.copy(from: cache.buffer, sourceOffset: cache.kOffset(block: blockIndex),
                         to: kToken, destinationOffset: 0,
                         size: kTokenCount * MemoryLayout<Float16>.stride)
            encoder.copy(from: cache.buffer, sourceOffset: cache.vOffset(block: blockIndex),
                         to: vToken, destinationOffset: 0,
                         size: kTokenCount * MemoryLayout<Float16>.stride)
            encoder.endEncoding()
            metrics?.recordCrossKVHit()
        } else {
            guard projectedKVAvailable else {
                throw AnimapkError.validation("ANE cross K/V projection missing on cache miss")
            }
            try encodeHalfComputeBoundary(command, kToken, count: kTokenCount)
            try encodeHalfComputeBoundary(command, vToken, count: kTokenCount)
            if NumericalMonitor.detailedProbesEnabled, let monitor {
                try monitor.encodeProbe(command, values: kToken, count: kTokenCount,
                                        probe: cross ? .crossKToken : .selfKToken)
                try monitor.encodeProbe(command, values: vToken, count: kTokenCount,
                                        probe: cross ? .crossVToken : .selfVToken)
            }
            if cross {
                try encodeRMSHeads(command, input: kToken, weightOffset: weights.crossKNorm,
                                   output: kToken, rows: Self.contextTokens * Self.heads, slot: slot)
            } else {
                try encodeRMSRoPE(command, input: kToken, weightOffset: weights.selfKNorm,
                                  rope: rope, output: kToken, slot: slot)
            }
            try encodeHalfComputeBoundary(command, kToken, count: kTokenCount)
            if cacheEnabled, let cache = crossKVCache {
                guard let encoder = command.makeBlitCommandEncoder() else {
                    throw AnimapkError.validation("failed to create ANE cross-K/V cache store encoder")
                }
                encoder.copy(from: kToken, sourceOffset: 0, to: cache.buffer,
                             destinationOffset: cache.kOffset(block: blockIndex),
                             size: kTokenCount * MemoryLayout<Float16>.stride)
                encoder.copy(from: vToken, sourceOffset: 0, to: cache.buffer,
                             destinationOffset: cache.vOffset(block: blockIndex),
                             size: kTokenCount * MemoryLayout<Float16>.stride)
                encoder.endEncoding()
                cache.markReady(blockIndex)
                metrics?.recordCrossKVMiss()
            }
        }

        try encodeHalfComputeBoundary(command, qToken, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: qToken, count: Self.tokens * Self.dim,
                                    probe: cross ? .crossQToken : .selfQToken)
        }
        if cross {
            try encodeRMSHeads(command, input: qToken, weightOffset: weights.crossQNorm,
                               output: qToken, rows: Self.tokens * Self.heads, slot: slot)
        } else {
            try encodeRMSRoPE(command, input: qToken, weightOffset: weights.selfQNorm,
                              rope: rope, output: qToken, slot: slot)
        }
        try encodeHalfComputeBoundary(command, qToken, count: Self.tokens * Self.dim)
        let qPostSnapshot = capturePostTransform
            ? try encodeSnapshotHalf(command, source: qToken, key: "dit.diagnostic.self.qPostRope.f16",
                                     elements: Self.tokens * Self.dim) : nil
        let kPostSnapshot = capturePostTransform
            ? try encodeSnapshotHalf(command, source: kToken, key: "dit.diagnostic.self.kPostRope.f16",
                                     elements: kTokenCount) : nil

        let strided = try resolvedStridedAttention()
        if strided {
            let attended = buffer("dit.attended.token.f16", Self.tokens * Self.dim, Float16.self)
            try attention.encode(
                commandBuffer: command, query: qToken, key: kToken, value: vToken,
                output: attended, heads: Self.heads, queryCount: Self.tokens,
                keyCount: keyRows, headDim: Self.headDim,
                probe: cross ? .crossScores : .selfScores,
                layout: .tokenMajor(tokenStride: Self.dim))
            try encodeHalfComputeBoundary(command, attended, count: Self.tokens * Self.dim)
            if NumericalMonitor.detailedProbesEnabled, let monitor {
                try monitor.encodeProbe(command, values: attended, count: Self.tokens * Self.dim,
                                        probe: cross ? .crossAttended : .selfAttended)
            }
            return ANEAttentionMathResult(
                attended: attended, qPostTransform: qPostSnapshot, kPostTransform: kPostSnapshot)
        }

        let qHead = buffer("dit.q.head.f16", Self.tokens * Self.dim, Float16.self)
        let kHead = buffer("dit.k.head.f16", kTokenCount, Float16.self)
        let vHead = buffer("dit.v.head.f16", kTokenCount, Float16.self)
        try encodeTranspose(command, input: qToken, output: qHead, tokens: Self.tokens, toHeadMajor: true)
        try encodeTranspose(command, input: kToken, output: kHead, tokens: keyRows, toHeadMajor: true)
        try encodeTranspose(command, input: vToken, output: vHead, tokens: keyRows, toHeadMajor: true)
        let attendedHead = buffer("dit.attended.head.f16", Self.tokens * Self.dim, Float16.self)
        try attention.encode(
            commandBuffer: command, query: qHead, key: kHead, value: vHead,
            output: attendedHead, heads: Self.heads, queryCount: Self.tokens,
            keyCount: keyRows, headDim: Self.headDim,
            probe: cross ? .crossScores : .selfScores)
        try encodeHalfComputeBoundary(command, attendedHead, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: attendedHead, count: Self.tokens * Self.dim,
                                    probe: cross ? .crossAttended : .selfAttended)
        }
        let attended = buffer("dit.attended.token.f16", Self.tokens * Self.dim, Float16.self)
        try encodeTranspose(command, input: attendedHead, output: attended,
                            tokens: Self.tokens, toHeadMajor: false)
        return ANEAttentionMathResult(
            attended: attended, qPostTransform: qPostSnapshot, kPostTransform: kPostSnapshot)
    }

    private func encodeANEAttentionOutput(
        _ command: MTLCommandBuffer,
        aneOutput: A12ANESurface,
        residual: MTLBuffer,
        modulation: MTLBuffer,
        cross: Bool
    ) throws -> MTLBuffer {
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
        return branch
    }

    private func resolvedStridedAttention() throws -> Bool {
        switch optimization.attentionBackend {
        case .legacyHeadMajorMPS:
            return false
        case .stridedTokenMajorMPS:
            return optimization.stridedTokenMajorAttention
        case .streamingMPS, .metalFlash:
            guard optimization.stridedTokenMajorAttention else {
                throw AnimapkError.validation(
                    "P7 attention backend \(optimization.attentionBackend.rawValue) requires the strided token-major toggle to be ON")
            }
            return true
        }
    }

    private func encodeMLPInput(
        _ command: MTLCommandBuffer,
        residual: MTLBuffer,
        siluEmb: MTLBuffer,
        adalnLora: MTLBuffer,
        weights: any DiTAuxWeights
    ) throws -> (modulation: MTLBuffer, projectionInput: MTLBuffer) {
        let modulation = try encodeModulation(
            command, siluEmb: siluEmb, adalnLora: adalnLora,
            w1: weights.modMLP1, w2: weights.modMLP2)
        let projectionInput = buffer("dit.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        if optimization.fusedNormModulation {
            try encodeFusedNormModulate(
                command, residual: residual, modulation: modulation,
                output: projectionInput, rows: Self.tokens, columns: Self.dim,
                probe: .mlpProjectionInput)
            metrics?.recordFusedTrafficSaved(
                UInt64(Self.tokens * Self.dim * MemoryLayout<Float>.stride) * 2)
        } else {
            let norm = buffer("dit.norm.f32", Self.tokens * Self.dim, Float.self)
            let modulated = buffer("dit.modulated.f32", Self.tokens * Self.dim, Float.self)
            try encodeLayerNorm(command, input: residual, output: norm,
                                rows: Self.tokens, columns: Self.dim)
            try encodeModulate(command, normalized: norm, modulation: modulation,
                               output: modulated, count: Self.tokens * Self.dim)
            try encodeFloatToComputeHalf(
                command, input: modulated, output: projectionInput,
                count: Self.tokens * Self.dim, probe: .mlpProjectionInput)
        }
        return (modulation, projectionInput)
    }

    /// Exact copy of the existing post-MLP1 activation behavior, factored so
    /// the ANE path cannot silently change GELU or numerical-monitor semantics.
    private func encodeMLPActivation(
        _ command: MTLCommandBuffer, hiddenHalf: MTLBuffer
    ) throws {
        if optimization.fusedMLPActivation {
            try encodeFusedGELUHalf(
                command, values: hiddenHalf,
                count: Self.tokens * Self.hidden, probe: .mlpHiddenToHalf)
            metrics?.recordFusedTrafficSaved(
                UInt64(Self.tokens * Self.hidden * MemoryLayout<Float>.stride))
        } else {
            let hiddenFloat = buffer("dit.hidden.f32", Self.tokens * Self.hidden, Float.self)
            try encodeConvert(command, kernel: "half_to_float", input: hiddenHalf,
                              output: hiddenFloat, count: Self.tokens * Self.hidden)
            metrics?.recordConversionBytes(
                UInt64(Self.tokens * Self.hidden * MemoryLayout<Float>.stride))
            try encodeUnary(command, kernel: "gelu", input: hiddenFloat,
                            output: hiddenFloat, count: Self.tokens * Self.hidden)
            try encodeComputeBoundary(command, hiddenFloat, count: Self.tokens * Self.hidden)
            if let monitor {
                try encodeProbeConvert(
                    command, input: hiddenFloat, output: hiddenHalf,
                    count: Self.tokens * Self.hidden,
                    monitor: monitor, probe: .mlpHiddenToHalf)
            } else {
                try encodeConvert(command, kernel: "float_to_half", input: hiddenFloat,
                                  output: hiddenHalf, count: Self.tokens * Self.hidden)
            }
            metrics?.recordConversionBytes(
                UInt64(Self.tokens * Self.hidden * MemoryLayout<Float16>.stride))
        }
    }

    private func encodeTokenToANE(
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
        encoder.setBytes(&rows32, length: 4, index: 2)
        encoder.setBytes(&channels32, length: 4, index: 3)
        encoder.setBytes(&stride32, length: 4, index: 4)
        let tw = max(1, min(pipeline.threadExecutionWidth, channels))
        let th = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / tw, 8))
        encoder.dispatchThreads(
            MTLSize(width: channels, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        encoder.endEncoding()
        metrics?.recordConversionBytes(
            UInt64(rows * channels * MemoryLayout<Float16>.stride))
    }

    private func commitStandaloneCommand(
        _ command: MTLCommandBuffer,
        encodeSeconds: Double
    ) async throws {
        let gate = CommandBufferGate()
        command.addCompletedHandler { completed in
            if let error = completed.error { gate.resume(throwing: error) }
            else { gate.resume() }
        }
        let waitStart = ProcessInfo.processInfo.systemUptime
        command.commit()
        try await gate.wait()
        recordCompletedCommand(
            command, encodeSeconds: encodeSeconds,
            hostWindowSeconds: ProcessInfo.processInfo.systemUptime - waitStart)
    }

    private func recordCompletedCommand(
        _ command: MTLCommandBuffer,
        encodeSeconds: Double,
        hostWindowSeconds: Double
    ) {
        let gpuSeconds = (command.gpuStartTime > 0 && command.gpuEndTime >= command.gpuStartTime)
            ? command.gpuEndTime - command.gpuStartTime : 0
        metrics?.recordGPUCommand(seconds: gpuSeconds)
        metrics?.recordEncode(seconds: encodeSeconds)
        metrics?.recordHostWait(seconds: max(0, hostWindowSeconds - gpuSeconds))
    }

    private func blockRange(_ logicalIndex: Int) throws -> AnimapkExecutionRange {
        guard blockRanges.indices.contains(logicalIndex) else {
            throw AnimapkError.validation("DiT block index \(logicalIndex) is out of range")
        }
        return blockRanges[logicalIndex]
    }

    /// Load a block's weights into a slot without executing it (ping-pong
    /// prologue). The loop's in-flight guard still applies.
    func prefetch(blockIndex: Int, slot: Int) throws {
        let range = try blockRange(blockIndex)
        let copyStart = ProcessInfo.processInfo.systemUptime
        let result = try streamer.load(
            range, from: file, slot: slot,
            mode: optimization.linearBackend == .aneHybridW8
                ? .copied
                : (optimization.noCopyWeightSource ? .noCopy : .copied))
        if result.mode == .noCopy {
            // P6: prologue memcpy eliminated — record the no-copy bytes.
            metrics?.recordMmapNoCopyBytes(result.noCopyBytes)
        } else {
            metrics?.recordWeightCopy(
                bytes: Int(range.length),
                seconds: ProcessInfo.processInfo.systemUptime - copyStart)
        }
    }

    private func encodeSnapshotHalf(
        _ command: MTLCommandBuffer, source: MTLBuffer, key: String, elements: Int
    ) throws -> MTLBuffer {
        let bytes = elements * MemoryLayout<Float16>.stride
        let snapshot = buffers.buffer(key: key, bytes: bytes)
        guard source.length >= bytes, let encoder = command.makeBlitCommandEncoder() else {
            throw AnimapkError.validation("failed to create DiT FP16 diagnostic snapshot encoder")
        }
        encoder.copy(from: source, sourceOffset: 0, to: snapshot, destinationOffset: 0, size: bytes)
        encoder.endEncoding()
        return snapshot
    }

    private func encodeSnapshot(
        _ command: MTLCommandBuffer, source: MTLBuffer, key: String
    ) throws -> MTLBuffer {
        let bytes = Self.tokens * Self.dim * MemoryLayout<Float>.stride
        let snapshot = buffers.buffer(key: key, bytes: bytes)
        guard let encoder = command.makeBlitCommandEncoder() else {
            throw AnimapkError.validation("failed to create DiT diagnostic snapshot encoder")
        }
        encoder.copy(from: source, sourceOffset: 0, to: snapshot, destinationOffset: 0, size: bytes)
        encoder.endEncoding()
        return snapshot
    }

    private func encodeAttentionBranch(
        _ command: MTLCommandBuffer, residual: MTLBuffer, crossContext: MTLBuffer,
        rope: MTLBuffer, siluEmb: MTLBuffer, adalnLora: MTLBuffer,
        weights: BlockWeights, cross: Bool, slot: Int, blockIndex: Int
    ) throws {
        let modulation = try encodeModulation(
            command, siluEmb: siluEmb, adalnLora: adalnLora,
            w1: cross ? weights.modCross1 : weights.modSelf1,
            w2: cross ? weights.modCross2 : weights.modSelf2)
        let projectionInput = buffer("dit.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        if optimization.fusedNormModulation {
            // P3-A: one fused pass (LayerNorm + AdaLN + BF16 boundary + fp16
            // conversion). No dit.norm.f32 / dit.modulated.f32 intermediates.
            try encodeFusedNormModulate(command, residual: residual, modulation: modulation,
                                        output: projectionInput, rows: Self.tokens,
                                        columns: Self.dim,
                                        probe: cross ? .crossProjectionInput : .selfProjectionInput)
            // P3: the fused pass no longer materializes the two fp32
            // intermediates (norm + modulated), so record that saved traffic.
            metrics?.recordFusedTrafficSaved(
                UInt64(Self.tokens * Self.dim * MemoryLayout<Float>.stride) * 2)
        } else {
            // Legacy three-pass path, kept exactly for A/B.
            let norm = buffer("dit.norm.f32", Self.tokens * Self.dim, Float.self)
            let modulated = buffer("dit.modulated.f32", Self.tokens * Self.dim, Float.self)
            try encodeLayerNorm(command, input: residual, output: norm, rows: Self.tokens, columns: Self.dim)
            try encodeModulate(command, normalized: norm, modulation: modulation,
                               output: modulated, count: Self.tokens * Self.dim)
            try encodeFloatToComputeHalf(command, input: modulated, output: projectionInput,
                                         count: Self.tokens * Self.dim,
                                         probe: cross ? .crossProjectionInput : .selfProjectionInput)
        }

        let qToken = buffer("dit.q.token.f16", Self.tokens * Self.dim, Float16.self)
        let kTokenCount = (cross ? Self.contextTokens : Self.tokens) * Self.dim
        let kToken = buffer("dit.k.token.f16", kTokenCount, Float16.self)
        let vToken = buffer("dit.v.token.f16", kTokenCount, Float16.self)
        // P4: the strided token-major backend feeds the token-major Q/K/V
        // buffers DIRECTLY to MPS via strided per-head views, so the four
        // head-major transpose buffers/kernels are not allocated or encoded.
        // The legacy head-major path stays byte-for-byte for A/B.
        // P7: the backend selector picks between the P4 strided MPS path,
        // the P7-A streaming online-softmax MPS path, and the P7-B pure-Metal
        // Flash path. `.legacyHeadMajorMPS` always uses the transposed
        // head-major layout below; `.stridedTokenMajorMPS` honors the P4
        // boolean so turning the toggle off still selects the legacy path.
        let strided: Bool
        switch optimization.attentionBackend {
        case .legacyHeadMajorMPS:
            strided = false
        case .stridedTokenMajorMPS:
            strided = optimization.stridedTokenMajorAttention
        case .streamingMPS, .metalFlash:
            // P7 backends REQUIRE the token-major layout; when the P4 toggle
            // is off they must not silently fall back to the transposed path
            // (that would change what the device measures).
            guard optimization.stridedTokenMajorAttention else {
                throw AnimapkError.validation(
                    "P7 attention backend \(optimization.attentionBackend.rawValue) requires the strided token-major toggle to be ON")
            }
            strided = true
        }
        let queryWeight = cross ? weights.crossQ : weights.selfQ
        let keyWeight = cross ? weights.crossK : weights.selfK
        let valueWeight = cross ? weights.crossV : weights.selfV
        try linear.encode(commandBuffer: command, input: projectionInput,
                          weight: queryWeight, output: qToken, inputRows: Self.tokens,
                          family: .attentionProjection)
        let keyInput = cross ? crossContext : projectionInput
        let keyRows = cross ? Self.contextTokens : Self.tokens
        // P5: cache the invariant cross K/V across diffusion steps. On a hit we
        // skip the cross K/V projection + static boundary + K RMSNorm and blit
        // the cached (post-transform) K/V into the scratch buffers, so the
        // downstream attention sees byte-identical inputs by construction. Q is
        // always projected fresh (it is dynamic per step).
        let cacheEnabled = cross && crossKVCache != nil
        let cacheHit = cacheEnabled && (crossKVCache?.isReady(blockIndex) ?? false)
        if cacheHit, let cache = crossKVCache {
            // Blit cached K/V into the scratch buffers (exact; the cache holds
            // the state right before attention for this block).
            if let encoder = command.makeBlitCommandEncoder() {
                encoder.copy(from: cache.buffer, sourceOffset: cache.kOffset(block: blockIndex),
                             to: kToken, destinationOffset: 0, size: kTokenCount * MemoryLayout<Float16>.stride)
                encoder.copy(from: cache.buffer, sourceOffset: cache.vOffset(block: blockIndex),
                             to: vToken, destinationOffset: 0, size: kTokenCount * MemoryLayout<Float16>.stride)
                encoder.endEncoding()
            }
            metrics?.recordCrossKVHit()
        } else {
            try linear.encode(commandBuffer: command, input: keyInput,
                              weight: keyWeight, output: kToken, inputRows: keyRows,
                              family: .attentionProjection)
            try linear.encode(commandBuffer: command, input: keyInput,
                              weight: valueWeight, output: vToken, inputRows: keyRows,
                              family: .attentionProjection)
            try encodeHalfComputeBoundary(command, kToken, count: kTokenCount)
            try encodeHalfComputeBoundary(command, vToken, count: kTokenCount)
            if NumericalMonitor.detailedProbesEnabled, let monitor {
                try monitor.encodeProbe(command, values: kToken, count: kTokenCount,
                                        probe: cross ? .crossKToken : .selfKToken)
                try monitor.encodeProbe(command, values: vToken, count: kTokenCount,
                                        probe: cross ? .crossVToken : .selfVToken)
            }
            if cross {
                try encodeRMSHeads(command, input: kToken, weightOffset: weights.crossKNorm,
                                   output: kToken, rows: Self.contextTokens * Self.heads, slot: slot)
            } else {
                try encodeRMSRoPE(command, input: kToken, weightOffset: weights.selfKNorm,
                                  rope: rope, output: kToken, slot: slot)
            }
            try encodeHalfComputeBoundary(command, kToken, count: kTokenCount)
            // First use of this block's cross K/V: store the post-transform
            // K/V into the cache for reuse by later steps.
            if cacheEnabled, let cache = crossKVCache {
                if let encoder = command.makeBlitCommandEncoder() {
                    encoder.copy(from: kToken, sourceOffset: 0, to: cache.buffer,
                                 destinationOffset: cache.kOffset(block: blockIndex),
                                 size: kTokenCount * MemoryLayout<Float16>.stride)
                    encoder.copy(from: vToken, sourceOffset: 0, to: cache.buffer,
                                 destinationOffset: cache.vOffset(block: blockIndex),
                                 size: kTokenCount * MemoryLayout<Float16>.stride)
                    encoder.endEncoding()
                }
                cache.markReady(blockIndex)
                metrics?.recordCrossKVMiss()
            }
        }
        try encodeHalfComputeBoundary(command, qToken, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: qToken, count: Self.tokens * Self.dim,
                                    probe: cross ? .crossQToken : .selfQToken)
        }
        // Q is always dynamic. Self Q uses RoPE; cross Q uses RMS norm.
        if cross {
            try encodeRMSHeads(command, input: qToken, weightOffset: weights.crossQNorm,
                               output: qToken, rows: Self.tokens * Self.heads, slot: slot)
        } else {
            try encodeRMSRoPE(command, input: qToken, weightOffset: weights.selfQNorm,
                              rope: rope, output: qToken, slot: slot)
        }
        try encodeHalfComputeBoundary(command, qToken, count: Self.tokens * Self.dim)

        let attendedToken: MTLBuffer
        if strided {
            // P4: MPS strided token-major attention — Q/K/V and the attended
            // output all stay in the token-major layout; no transposes.
            let attended = buffer("dit.attended.token.f16", Self.tokens * Self.dim, Float16.self)
            try attention.encode(
                commandBuffer: command, query: qToken, key: kToken, value: vToken,
                output: attended, heads: Self.heads, queryCount: Self.tokens,
                keyCount: keyRows, headDim: Self.headDim,
                probe: cross ? .crossScores : .selfScores,
                layout: .tokenMajor(tokenStride: Self.dim))
            // The strided backend materializes the attended token-major output
            // directly, so the boundary/probe land on it (the legacy path keeps
            // them on the head-major buffer, exactly as before, for A/B).
            try encodeHalfComputeBoundary(command, attended, count: Self.tokens * Self.dim)
            if NumericalMonitor.detailedProbesEnabled, let monitor {
                try monitor.encodeProbe(command, values: attended, count: Self.tokens * Self.dim,
                                        probe: cross ? .crossAttended : .selfAttended)
            }
            attendedToken = attended
        } else {
            // Legacy head-major path: transpose in, attend, transpose out.
            let qHead = buffer("dit.q.head.f16", Self.tokens * Self.dim, Float16.self)
            let kHead = buffer("dit.k.head.f16", kTokenCount, Float16.self)
            let vHead = buffer("dit.v.head.f16", kTokenCount, Float16.self)
            try encodeTranspose(command, input: qToken, output: qHead,
                                tokens: Self.tokens, toHeadMajor: true)
            try encodeTranspose(command, input: kToken, output: kHead,
                                tokens: keyRows, toHeadMajor: true)
            try encodeTranspose(command, input: vToken, output: vHead,
                                tokens: keyRows, toHeadMajor: true)
            let attendedHead = buffer("dit.attended.head.f16", Self.tokens * Self.dim, Float16.self)
            try attention.encode(commandBuffer: command, query: qHead, key: kHead, value: vHead,
                                 output: attendedHead, heads: Self.heads, queryCount: Self.tokens,
                                 keyCount: keyRows, headDim: Self.headDim,
                                 probe: cross ? .crossScores : .selfScores)
            try encodeHalfComputeBoundary(command, attendedHead, count: Self.tokens * Self.dim)
            if NumericalMonitor.detailedProbesEnabled, let monitor {
                try monitor.encodeProbe(command, values: attendedHead, count: Self.tokens * Self.dim,
                                        probe: cross ? .crossAttended : .selfAttended)
            }
            let attended = buffer("dit.attended.token.f16", Self.tokens * Self.dim, Float16.self)
            try encodeTranspose(command, input: attendedHead, output: attended,
                                tokens: Self.tokens, toHeadMajor: false)
            attendedToken = attended
        }
        let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try linear.encode(commandBuffer: command, input: attendedToken,
                          weight: cross ? weights.crossO : weights.selfO,
                          output: branch, inputRows: Self.tokens,
                          family: .attentionProjection)
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
            try monitor.encodeProbeF32(command, values: residual,
                                       count: Self.tokens * Self.dim,
                                       probe: cross ? .crossResidual : .selfResidual)
        }
    }

    private func encodeMLP(
        _ command: MTLCommandBuffer, residual: MTLBuffer, siluEmb: MTLBuffer,
        adalnLora: MTLBuffer, weights: BlockWeights
    ) throws {
        let modulation = try encodeModulation(command, siluEmb: siluEmb, adalnLora: adalnLora,
                                              w1: weights.modMLP1, w2: weights.modMLP2)
        let projectionInput = buffer("dit.projectionInput.f16", Self.tokens * Self.dim, Float16.self)
        if optimization.fusedNormModulation {
            // P3-A: one fused pass (LayerNorm + AdaLN + BF16 boundary + fp16
            // conversion). No dit.norm.f32 / dit.modulated.f32 intermediates.
            try encodeFusedNormModulate(command, residual: residual, modulation: modulation,
                                        output: projectionInput, rows: Self.tokens,
                                        columns: Self.dim, probe: .mlpProjectionInput)
        } else {
            // Legacy three-pass path, kept exactly for A/B.
            let norm = buffer("dit.norm.f32", Self.tokens * Self.dim, Float.self)
            let modulated = buffer("dit.modulated.f32", Self.tokens * Self.dim, Float.self)
            try encodeLayerNorm(command, input: residual, output: norm, rows: Self.tokens, columns: Self.dim)
            try encodeModulate(command, normalized: norm, modulation: modulation,
                               output: modulated, count: Self.tokens * Self.dim)
            try encodeFloatToComputeHalf(command, input: modulated, output: projectionInput,
                                         count: Self.tokens * Self.dim, probe: .mlpProjectionInput)
        }
        let hiddenHalf = buffer("dit.hidden.f16", Self.tokens * Self.hidden, Float16.self)
        try linear.encode(commandBuffer: command, input: projectionInput,
                          weight: weights.mlp1, output: hiddenHalf, inputRows: Self.tokens,
                          family: .mlpUp)
        try encodeHalfComputeBoundary(command, hiddenHalf, count: Self.tokens * Self.hidden)
        if optimization.fusedMLPActivation {
            // P3-B: in-place GELU on the fp16 hidden activations (fp32 register
            // arithmetic, optional BF16 rounding). No dit.hidden.f32 (~32 MiB)
            // intermediate and no conversion passes.
            try encodeFusedGELUHalf(command, values: hiddenHalf,
                                    count: Self.tokens * Self.hidden, probe: .mlpHiddenToHalf)
            // P3: the fused path no longer materializes the fp32 hidden GELU
            // intermediate, so record that saved traffic.
            metrics?.recordFusedTrafficSaved(
                UInt64(Self.tokens * Self.hidden * MemoryLayout<Float>.stride))
        } else {
            // Legacy path, kept exactly for A/B.
            let hiddenFloat = buffer("dit.hidden.f32", Self.tokens * Self.hidden, Float.self)
            try encodeConvert(command, kernel: "half_to_float", input: hiddenHalf,
                              output: hiddenFloat, count: Self.tokens * Self.hidden)
            // P2-C: f16→f32 conversion traffic (bytes written, counted once).
            metrics?.recordConversionBytes(UInt64(Self.tokens * Self.hidden * MemoryLayout<Float>.stride))
            try encodeUnary(command, kernel: "gelu", input: hiddenFloat,
                            output: hiddenFloat, count: Self.tokens * Self.hidden)
            try encodeComputeBoundary(command, hiddenFloat, count: Self.tokens * Self.hidden)
            if let monitor {
                try encodeProbeConvert(command, input: hiddenFloat, output: hiddenHalf,
                                       count: Self.tokens * Self.hidden,
                                       monitor: monitor, probe: .mlpHiddenToHalf)
            } else {
                try encodeConvert(command, kernel: "float_to_half", input: hiddenFloat,
                                  output: hiddenHalf, count: Self.tokens * Self.hidden)
            }
            // P2-C: f32→f16 conversion traffic (bytes written, counted once).
            metrics?.recordConversionBytes(UInt64(Self.tokens * Self.hidden * MemoryLayout<Float16>.stride))
        }
        let branch = buffer("dit.branch.f16", Self.tokens * Self.dim, Float16.self)
        try linear.encode(commandBuffer: command, input: hiddenHalf,
                          weight: weights.mlp2, output: branch, inputRows: Self.tokens,
                          family: .mlpDown)
        try encodeHalfComputeBoundary(command, branch, count: Self.tokens * Self.dim)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbe(command, values: branch, count: Self.tokens * Self.dim,
                                    probe: .mlpBranch)
        }
        try encodeGateAdd(command, residual: residual, branch: branch, modulation: modulation,
                          count: Self.tokens * Self.dim, probe: .mlpGateAdd)
        try encodeActivationBoundary(command, residual)
        if NumericalMonitor.detailedProbesEnabled, let monitor {
            try monitor.encodeProbeF32(command, values: residual,
                                       count: Self.tokens * Self.dim, probe: .mlpResidual)
        }
    }

    private func encodeActivationBoundary(
        _ command: MTLCommandBuffer, _ residual: MTLBuffer
    ) throws {
        guard activationNumerics == .bf16Boundaries || activationNumerics == .bf16Compute else { return }
        try encodeUnary(command, kernel: "round_f32_to_bf16", input: residual,
                        output: residual, count: Self.tokens * Self.dim)
    }

    private func encodeComputeBoundary(
        _ command: MTLCommandBuffer, _ value: MTLBuffer, count: Int
    ) throws {
        guard emulatesBF16 else { return }
        try encodeUnary(command, kernel: "round_f32_to_bf16", input: value,
                        output: value, count: count)
    }

    private func encodeFloatToComputeHalf(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int,
        probe: NumericalMonitor.Probe
    ) throws {
        try encodeComputeBoundary(command, input, count: count)
        if let monitor {
            try encodeProbeConvert(command, input: input, output: output, count: count,
                                   monitor: monitor, probe: probe)
        } else {
            try encodeConvert(command, kernel: "float_to_half", input: input,
                              output: output, count: count)
        }
        // P2-C: f32→f16 conversion traffic (bytes written, counted once).
        metrics?.recordConversionBytes(UInt64(count * MemoryLayout<Float16>.stride))
    }

    /// float_to_half with in-kernel numerical-health recording. The probe
    /// kernel performs the identical conversion; stats land in the monitor.
    private func encodeProbeConvert(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int,
        monitor: NumericalMonitor, probe: NumericalMonitor.Probe
    ) throws {
        let pipeline = try context.pipeline(named: "float_to_half_probe")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create float_to_half_probe encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        monitor.bindProbe(encoder, probe: probe, statsIndex: 3, slotIndex: 4)
        dispatchProbe(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    /// Dispatch shape for the probe kernels: exactly one thread per element in
    /// fixed 256-thread threadgroups (the kernels reduce via threadgroup memory).
    private func dispatchProbe(
        _ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int
    ) {
        let groups = (count + 255) / 256
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeHalfComputeBoundary(
        _ command: MTLCommandBuffer, _ value: MTLBuffer, count: Int
    ) throws {
        guard emulatesBF16 else { return }
        let pipeline = try context.pipeline(named: "round_half_to_bf16")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create DiT BF16 half boundary encoder")
        }
        var count = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(value, offset: 0, index: 0)
        encoder.setBuffer(value, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeModulation(
        _ command: MTLCommandBuffer, siluEmb: MTLBuffer, adalnLora: MTLBuffer,
        w1: QuantizedLinearWeightBuffers, w2: QuantizedLinearWeightBuffers
    ) throws -> MTLBuffer {
        let hidden = buffer("dit.modulation.hidden.f32", Self.modulationHidden, Float.self)
        let output = buffer("dit.modulation.f32", Self.modulationSize, Float.self)
        try encodeMatvec(command, input: siluEmb, weight: w1, output: hidden)
        try encodeComputeBoundary(command, hidden, count: Self.modulationHidden)
        try encodeMatvec(command, input: hidden, weight: w2, output: output)
        try encodeComputeBoundary(command, output, count: Self.modulationSize)
        try encodeBinary(command, kernel: "add_f32", destination: output,
                         source: adalnLora, count: Self.modulationSize)
        try encodeComputeBoundary(command, output, count: Self.modulationSize)
        return output
    }

    private func encodeMatvec(
        _ command: MTLCommandBuffer, input: MTLBuffer,
        weight: QuantizedLinearWeightBuffers, output: MTLBuffer
    ) throws {
        let pipeline = try context.pipeline(
            named: DiTQuantizedWeightFactory.matvecKernel(for: weight.storage))
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create modulation matvec encoder")
        }
        var columns = UInt32(weight.columns), rows = UInt32(weight.rows)
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
        encoder.dispatchThreadgroups(MTLSize(width: weight.rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeLayerNorm(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        rows: Int, columns: Int
    ) throws {
        let pipeline = try context.pipeline(named: "layernorm_f32_to_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create LayerNorm encoder")
        }
        var n = UInt32(columns), epsilon = Self.eps, rowCount = UInt32(rows)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.setBytes(&epsilon, length: 4, index: 3)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeModulate(
        _ command: MTLCommandBuffer, normalized: MTLBuffer, modulation: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: "modulate_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create AdaLN encoder")
        }
        var n = UInt32(Self.dim), elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(normalized, offset: 0, index: 0)
        encoder.setBuffer(modulation, offset: Self.dim * MemoryLayout<Float>.stride, index: 1)
        encoder.setBuffer(modulation, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        encoder.setBytes(&elementCount, length: 4, index: 5)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    /// P3-A: one fused pass replacing layernorm → modulate → boundary → toHalf.
    /// The modulation buffer uses the EXACT legacy layout (shift at element 0,
    /// scale at element `dim`). The shader ABI receives `modulationOffset` in
    /// float ELEMENTS: both kernels do `device const float *scale =
    /// modulation + modulationOffset` (pointer arithmetic on a float*), so
    /// scale starts at element `dim` (2048 for the DiT) — NOT at byte offset
    /// dim*4. The legacy non-fused path is different: `setBuffer(offset:)`
    /// takes BYTES, so its modulate-step offset stays `dim*4`. Both fused
    /// kernels (`dit_layernorm_modulate_to_half` and
    /// `dit_layernorm_modulate_to_half_probe`) share this element-unit
    /// contract, so the single host-side offset covers both. The optional BF16
    /// rounding sits between modulation and the fp16 conversion, matching the
    /// legacy `encodeFloatToComputeHalf` boundary placement. When a monitor is
    /// present the probe kernel records the same stats as the legacy
    /// `float_to_half_probe` pass, on the value fed to `half()`.

    /// Scale-chunk offset for the fused kernel, in the float-ELEMENT unit the
    /// shader ABI consumes (see the P3-A comment above). Internal (not
    /// private) so the ABI unit test and the synthetic parity test can lock
    /// the unit and catch a future byte/element regression — passing a byte
    /// offset (dim*4 = 8192) here walks 2048 floats past the end of the
    /// 6144-float modulation buffer.
    static func fusedModulationElementOffset(columns: Int) -> UInt32 {
        UInt32(columns)
    }

    private func encodeFusedNormModulate(
        _ command: MTLCommandBuffer, residual: MTLBuffer, modulation: MTLBuffer,
        output: MTLBuffer, rows: Int, columns: Int, probe: NumericalMonitor.Probe
    ) throws {
        try Self.encodeFusedNormModulateKernel(
            context: context, command: command,
            residual: residual, modulation: modulation, output: output,
            rows: rows, columns: columns,
            modulationElementOffset: Self.fusedModulationElementOffset(columns: Self.dim),
            emulatesBF16: emulatesBF16, monitor: monitor, probe: probe)
    }

    /// Single source of the fused kernel ABI (internal test seam): encodes
    /// `dit_layernorm_modulate_to_half` or its probe variant with the exact
    /// production argument layout. `modulationElementOffset` is in float
    /// ELEMENTS (see the P3-A comment above). Exposed so the synthetic parity
    /// test can drive the kernel against a tight 6144-float modulation buffer
    /// without a full pack fixture; production call sites go through
    /// `encodeFusedNormModulate`.
    static func encodeFusedNormModulateKernel(
        context: MetalContext, command: MTLCommandBuffer,
        residual: MTLBuffer, modulation: MTLBuffer, output: MTLBuffer,
        rows: Int, columns: Int, modulationElementOffset: UInt32,
        emulatesBF16: Bool, monitor: NumericalMonitor?, probe: NumericalMonitor.Probe
    ) throws {
        let kernel = monitor == nil ? "dit_layernorm_modulate_to_half" : "dit_layernorm_modulate_to_half_probe"
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create fused LayerNorm/modulate encoder")
        }
        var n = UInt32(columns), epsilon = Self.eps, rowCount = UInt32(rows)
        var modulationOffset = modulationElementOffset
        var boundary: UInt32 = emulatesBF16 ? 1 : 0
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(modulation, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&rowCount, length: 4, index: 5)
        encoder.setBytes(&modulationOffset, length: 4, index: 6)
        encoder.setBytes(&boundary, length: 4, index: 7)
        if let monitor {
            monitor.bindProbe(encoder, probe: probe, statsIndex: 8, slotIndex: 9)
        }
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// P3-B: in-place GELU on the fp16 MLP hidden activations (fp32 register
    /// arithmetic + optional BF16 rounding). With a monitor the probe kernel
    /// records the same stats as the legacy `mlpHiddenToHalf` conversion.
    private func encodeFusedGELUHalf(
        _ command: MTLCommandBuffer, values: MTLBuffer, count: Int,
        probe: NumericalMonitor.Probe
    ) throws {
        let kernel = monitor == nil ? "dit_gelu_half_inplace" : "dit_gelu_half_inplace_probe"
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create fused GELU half encoder")
        }
        var elementCount = UInt32(count)
        var boundary: UInt32 = emulatesBF16 ? 1 : 0
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBytes(&elementCount, length: 4, index: 1)
        encoder.setBytes(&boundary, length: 4, index: 2)
        if let monitor {
            monitor.bindProbe(encoder, probe: probe, statsIndex: 3, slotIndex: 4)
        }
        dispatchProbe(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeRMSRoPE(
        _ command: MTLCommandBuffer, input: MTLBuffer, weightOffset: Int,
        rope: MTLBuffer, output: MTLBuffer, slot: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rms_rope_split_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create RMS/RoPE encoder")
        }
        var tokens = UInt32(Self.tokens), heads = UInt32(Self.heads), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(streamer.buffer(for: slot), offset: weightOffset, index: 1)
        encoder.setBuffer(rope, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&tokens, length: 4, index: 4)
        encoder.setBytes(&heads, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: Self.tokens * Self.heads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRMSHeads(
        _ command: MTLCommandBuffer, input: MTLBuffer, weightOffset: Int,
        output: MTLBuffer, rows: Int, slot: Int
    ) throws {
        let pipeline = try context.pipeline(named: "rmsnorm_heads_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create head RMSNorm encoder")
        }
        var rowCount = UInt32(rows), epsilon = Self.eps
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(streamer.buffer(for: slot), offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeTranspose(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        tokens: Int, toHeadMajor: Bool
    ) throws {
        let pipeline = try context.pipeline(named: "transpose_token_head_half")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create head transpose encoder")
        }
        var tokenCount = UInt32(tokens), heads = UInt32(Self.heads)
        var headDim = UInt32(Self.headDim), direction: UInt32 = toHeadMajor ? 1 : 0
        let count = tokens * Self.dim
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&tokenCount, length: 4, index: 2)
        encoder.setBytes(&heads, length: 4, index: 3)
        encoder.setBytes(&headDim, length: 4, index: 4)
        encoder.setBytes(&direction, length: 4, index: 5)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
        // P2-C: logical transpose traffic — fp16 elements materialized,
        // counted once (bytes written). Arithmetic counter, not a GPU readback.
        metrics?.recordTransposeBytes(UInt64(count * MemoryLayout<Float16>.stride))
    }

    private func encodeGateAdd(
        _ command: MTLCommandBuffer, residual: MTLBuffer, branch: MTLBuffer,
        modulation: MTLBuffer, count: Int, probe: NumericalMonitor.Probe
    ) throws {
        if let monitor {
            let pipeline = try context.pipeline(named: "gate_add_half_f32_probe")
            guard let encoder = command.makeComputeCommandEncoder() else {
                throw AnimapkError.validation("failed to create gated residual encoder")
            }
            var n = UInt32(Self.dim), elementCount = UInt32(count)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(residual, offset: 0, index: 0)
            encoder.setBuffer(branch, offset: 0, index: 1)
            encoder.setBuffer(modulation, offset: 2 * Self.dim * MemoryLayout<Float>.stride, index: 2)
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.setBytes(&elementCount, length: 4, index: 4)
            monitor.bindProbe(encoder, probe: probe, statsIndex: 5, slotIndex: 6)
            dispatchProbe(encoder, pipeline: pipeline, count: count)
            encoder.endEncoding()
            return
        }
        let pipeline = try context.pipeline(named: "gate_add_half_f32")
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create gated residual encoder")
        }
        var n = UInt32(Self.dim), elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBuffer(modulation, offset: 2 * Self.dim * MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&elementCount, length: 4, index: 4)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeUnary(
        _ command: MTLCommandBuffer, kernel: String, input: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(kernel) encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeConvert(
        _ command: MTLCommandBuffer, kernel: String, input: MTLBuffer,
        output: MTLBuffer, count: Int
    ) throws {
        try encodeUnary(command, kernel: kernel, input: input, output: output, count: count)
    }

    private func encodeBinary(
        _ command: MTLCommandBuffer, kernel: String, destination: MTLBuffer,
        source: MTLBuffer, count: Int
    ) throws {
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create \(kernel) encoder")
        }
        var elementCount = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destination, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        dispatch1D(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
    }

    private func dispatch1D(
        _ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int
    ) {
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func reductionThreads(_ maximum: Int) -> Int {
        var result = 1
        while result * 2 <= min(256, maximum) { result *= 2 }
        return result
    }

    private func buffer<T>(_ key: String, _ count: Int, _: T.Type) -> MTLBuffer {
        buffers.buffer(key: key, bytes: count * MemoryLayout<T>.stride)
    }

    private func validateInputs(
        residual: MTLBuffer, emb: MTLBuffer, adalnLora: MTLBuffer,
        crossContext: MTLBuffer, rope: MTLBuffer
    ) throws {
        let requirements = [
            (residual, Self.tokens * Self.dim * 4, "residual"),
            (emb, Self.dim * 4, "emb"),
            (adalnLora, Self.modulationSize * 4, "adalnLora"),
            (crossContext, Self.contextTokens * Self.contextDim * 2, "crossContext"),
            (rope, Self.tokens * 64 * 4 * 4, "rope"),
        ]
        for (buffer, bytes, label) in requirements where buffer.length < bytes {
            throw AnimapkError.validation("DiT \(label) buffer is too small")
        }
    }
}

/// Await gate for a committed MTLCommandBuffer whose completion handler must be
/// registered BEFORE `commit()` (Metal asserts on late handlers). The handler
/// fires on a Metal queue thread; the awaiting task resumes via the stored
/// continuation. Safe whether the handler fires before or after `wait()`.
final class CommandBufferGate {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var outcome: Result<Void, Error>?

    func resume(throwing error: Error? = nil) {
        lock.lock()
        if let continuation {
            self.continuation = nil
            lock.unlock()
            if let error { continuation.resume(throwing: error) }
            else { continuation.resume() }
        } else {
            if let error { outcome = .failure(error) }
            else { outcome = .success(()) }
            lock.unlock()
        }
    }

    func wait() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            lock.lock()
            if let outcome {
                lock.unlock()
                continuation.resume(with: outcome)
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }
}

private protocol DiTAuxWeights {
    var modSelf1: QuantizedLinearWeightBuffers { get }
    var modSelf2: QuantizedLinearWeightBuffers { get }
    var modCross1: QuantizedLinearWeightBuffers { get }
    var modCross2: QuantizedLinearWeightBuffers { get }
    var modMLP1: QuantizedLinearWeightBuffers { get }
    var modMLP2: QuantizedLinearWeightBuffers { get }
    var selfQNorm: Int { get }
    var selfKNorm: Int { get }
    var crossQNorm: Int { get }
    var crossKNorm: Int { get }
}

/// The compact Metal-only half of one ANE-native DiT block: six modulation
/// matrices plus four learned RMSNorm vectors. The ten large projections are
/// deliberately absent from this streamed range and live in prepared ANE models.
private struct ANEBlockWeights: DiTAuxWeights {
    let modSelf1, modSelf2, modCross1, modCross2, modMLP1, modMLP2: QuantizedLinearWeightBuffers
    let selfQNorm, selfKNorm, crossQNorm, crossKNorm: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model.blocks.\(range.logicalIndex)."
        var spans: [String: AnimapkTensorSpans] = [:]
        for item in range.tensors {
            guard item.tensor.name.hasPrefix(prefix),
                  item.tensor.quantizationFormat != ANEW8NativePack.tensorFormat else {
                throw AnimapkError.validation("invalid tensor in ANE Metal-only block range")
            }
            spans[String(item.tensor.name.dropFirst(prefix.count))] = item
        }
        func matrix(_ name: String, _ rows: Int, _ columns: Int) throws -> QuantizedLinearWeightBuffers {
            guard let item = spans[name] else {
                throw AnimapkError.validation("missing ANE Metal matrix \(prefix)\(name)")
            }
            return try DiTQuantizedWeightFactory.makeMatrix(
                item, ring: ring, rows: rows, columns: columns,
                label: "DiT ANE Metal \(prefix)\(name)")
        }
        func norm(_ name: String) throws -> Int {
            guard let item = spans[name], item.tensor.shape == [128],
                  item.tensor.storage == .fp16, item.data.length == 256,
                  item.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid ANE Metal norm \(prefix)\(name)")
            }
            return Int(item.data.offset)
        }
        modSelf1 = try matrix("adaln_modulation_self_attn.1.weight", 256, 2_048)
        modSelf2 = try matrix("adaln_modulation_self_attn.2.weight", 6_144, 256)
        modCross1 = try matrix("adaln_modulation_cross_attn.1.weight", 256, 2_048)
        modCross2 = try matrix("adaln_modulation_cross_attn.2.weight", 6_144, 256)
        modMLP1 = try matrix("adaln_modulation_mlp.1.weight", 256, 2_048)
        modMLP2 = try matrix("adaln_modulation_mlp.2.weight", 6_144, 256)
        selfQNorm = try norm("self_attn.q_norm.weight")
        selfKNorm = try norm("self_attn.k_norm.weight")
        crossQNorm = try norm("cross_attn.q_norm.weight")
        crossKNorm = try norm("cross_attn.k_norm.weight")
        guard spans.count == 10 else {
            throw AnimapkError.validation("ANE Metal-only DiT block must contain exactly 10 tensors")
        }
    }
}

private struct BlockWeights: DiTAuxWeights {
    let modSelf1, modSelf2, modCross1, modCross2, modMLP1, modMLP2: QuantizedLinearWeightBuffers
    let selfQ, selfK, selfV, selfO: QuantizedLinearWeightBuffers
    let crossQ, crossK, crossV, crossO: QuantizedLinearWeightBuffers
    let mlp1, mlp2: QuantizedLinearWeightBuffers
    let selfQNorm, selfKNorm, crossQNorm, crossKNorm: Int

    init(range: AnimapkExecutionRange, ring: MTLBuffer) throws {
        let prefix = "model.diffusion_model.blocks.\(range.logicalIndex)."
        var spans: [String: AnimapkTensorSpans] = [:]
        for item in range.tensors {
            guard item.tensor.name.hasPrefix(prefix) else {
                throw AnimapkError.validation("foreign tensor in DiT block range")
            }
            spans[String(item.tensor.name.dropFirst(prefix.count))] = item
        }
        func matrix(_ name: String, _ rows: Int, _ columns: Int) throws -> QuantizedLinearWeightBuffers {
            guard let item = spans[name] else {
                throw AnimapkError.validation("missing DiT matrix \(prefix)\(name)")
            }
            return try DiTQuantizedWeightFactory.makeMatrix(
                item, ring: ring, rows: rows, columns: columns,
                label: "DiT \(prefix)\(name)")
        }
        func norm(_ name: String) throws -> Int {
            guard let item = spans[name], item.tensor.shape == [128],
                  item.tensor.storage == .fp16, item.data.length == 256,
                  item.data.offset <= UInt64(Int.max) else {
                throw AnimapkError.validation("invalid DiT norm \(prefix)\(name)")
            }
            return Int(item.data.offset)
        }
        modSelf1 = try matrix("adaln_modulation_self_attn.1.weight", 256, 2_048)
        modSelf2 = try matrix("adaln_modulation_self_attn.2.weight", 6_144, 256)
        modCross1 = try matrix("adaln_modulation_cross_attn.1.weight", 256, 2_048)
        modCross2 = try matrix("adaln_modulation_cross_attn.2.weight", 6_144, 256)
        modMLP1 = try matrix("adaln_modulation_mlp.1.weight", 256, 2_048)
        modMLP2 = try matrix("adaln_modulation_mlp.2.weight", 6_144, 256)
        selfQ = try matrix("self_attn.q_proj.weight", 2_048, 2_048)
        selfK = try matrix("self_attn.k_proj.weight", 2_048, 2_048)
        selfV = try matrix("self_attn.v_proj.weight", 2_048, 2_048)
        selfO = try matrix("self_attn.output_proj.weight", 2_048, 2_048)
        crossQ = try matrix("cross_attn.q_proj.weight", 2_048, 2_048)
        crossK = try matrix("cross_attn.k_proj.weight", 2_048, 1_024)
        crossV = try matrix("cross_attn.v_proj.weight", 2_048, 1_024)
        crossO = try matrix("cross_attn.output_proj.weight", 2_048, 2_048)
        mlp1 = try matrix("mlp.layer1.weight", 8_192, 2_048)
        mlp2 = try matrix("mlp.layer2.weight", 2_048, 8_192)
        selfQNorm = try norm("self_attn.q_norm.weight")
        selfKNorm = try norm("self_attn.k_norm.weight")
        crossQNorm = try norm("cross_attn.q_norm.weight")
        crossKNorm = try norm("cross_attn.k_norm.weight")
        guard spans.count == 20 else {
            throw AnimapkError.validation("DiT block must contain exactly 20 tensors")
        }
    }
}
