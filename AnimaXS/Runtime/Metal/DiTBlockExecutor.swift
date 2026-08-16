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
    private let locator: DiTBlockLocator
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
        let locator = try DiTBlockLocator(file: file)
        guard let maximum = locator.blocks.map(\.length).max(), maximum <= UInt64(Int.max) else {
            throw AnimapkError.validation("invalid DiT execution ranges")
        }
        self.context = context
        self.file = file
        self.locator = locator
        self.optimization = optimization
        self.crossKVCache = crossKVCache
        // Ping-pong ON keeps the existing two-slot streamer; OFF measures the
        // same generation with one slot and no look-ahead prefetch.
        let slotCount = optimization.pingPongWeightStreaming ? 2 : 1
        self.streamer = try WeightStreamer(device: context.device, capacity: Int(maximum), slotCount: slotCount)
        self.buffers = BufferPool(device: context.device)
        self.linear = LinearExecutor(
            context: context, tileRows: optimization.linearTileRows,
            directMPSIO: optimization.directLinearMPSIO)
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
        diagnosticBranchCompleted: DiagnosticBranchCompleted? = nil
    ) async throws {
        try validateInputs(residual: residual, emb: emb, adalnLora: adalnLora,
                           crossContext: crossContext, rope: rope)
        let range = try locator.block(blockIndex)
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
                let nextRange = try locator.block(prefetchIndex)
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

    /// Load a block's weights into a slot without executing it (ping-pong
    /// prologue). The loop's in-flight guard still applies.
    func prefetch(blockIndex: Int, slot: Int) throws {
        let range = try locator.block(blockIndex)
        let copyStart = ProcessInfo.processInfo.systemUptime
        let result = try streamer.load(
            range, from: file, slot: slot,
            mode: optimization.noCopyWeightSource ? .noCopy : .copied)
        if result.mode == .noCopy {
            // P6: prologue memcpy eliminated — record the no-copy bytes.
            metrics?.recordMmapNoCopyBytes(result.noCopyBytes)
        } else {
            metrics?.recordWeightCopy(
                bytes: Int(range.length),
                seconds: ProcessInfo.processInfo.systemUptime - copyStart)
        }
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
                          weight: queryWeight, output: qToken, inputRows: Self.tokens)
        let keyInput = cross ? crossContext : projectionInput
        let keyRows = cross ? Self.contextTokens : Self.tokens
        // P5: cache the invariant cross K/V across diffusion steps. On a hit we
        // skip the cross K/V projection + static boundary + K RMSNorm and blit
        // the cached (post-transform) K/V into the scratch buffers, so the
        // downstream attention sees byte-identical inputs by construction. Q is
        // always projected fresh (it is dynamic per step).
        let cacheEnabled = cross && optimization.crossKVCache
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
                              weight: keyWeight, output: kToken, inputRows: keyRows)
            try linear.encode(commandBuffer: command, input: keyInput,
                              weight: valueWeight, output: vToken, inputRows: keyRows)
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
                          output: branch, inputRows: Self.tokens)
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
                          weight: weights.mlp1, output: hiddenHalf, inputRows: Self.tokens)
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
                          weight: weights.mlp2, output: branch, inputRows: Self.tokens)
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
    /// The modulation buffer uses the EXACT legacy layout (scale at
    /// `modulationOffset` = dim*4, shift at offset 0). The optional BF16
    /// rounding sits between modulation and the fp16 conversion, matching the
    /// legacy `encodeFloatToComputeHalf` boundary placement. When a monitor is
    /// present the probe kernel records the same stats as the legacy
    /// `float_to_half_probe` pass, on the value fed to `half()`.
    private func encodeFusedNormModulate(
        _ command: MTLCommandBuffer, residual: MTLBuffer, modulation: MTLBuffer,
        output: MTLBuffer, rows: Int, columns: Int, probe: NumericalMonitor.Probe
    ) throws {
        let kernel = monitor == nil ? "dit_layernorm_modulate_to_half" : "dit_layernorm_modulate_to_half_probe"
        let pipeline = try context.pipeline(named: kernel)
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw AnimapkError.validation("failed to create fused LayerNorm/modulate encoder")
        }
        var n = UInt32(columns), epsilon = Self.eps, rowCount = UInt32(rows)
        var modulationOffset = UInt32(Self.dim * MemoryLayout<Float>.stride)
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

private struct BlockWeights {
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
