from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text()
    a = text.find(start)
    if a < 0:
        raise SystemExit(f"{path}: start marker not found: {start!r}")
    b = text.find(end, a)
    if b < 0:
        raise SystemExit(f"{path}: end marker not found: {end!r}")
    p.write_text(text[:a] + replacement + "\n\n" + text[b:])


backend = r'''enum ANEW8DiTModelProfile: String, Sendable {
    case full8
    case kvWarm6

    var programCount: Int { self == .full8 ? 8 : 6 }
    var includesCrossKV: Bool { self == .full8 }
}

/// Device-measured A12 scheduler policy. These are production constants, not
/// user-facing tuning knobs: V6 measured depth-3 prefetch, bounded retire1,
/// 4 pinned full8 blocks for the first/KV-miss traversal and 8 pinned six-
/// program blocks once exact P5 K/V is warm. The theoretical peaks stay below
/// the earlier 80-program clean envelope: 64 full8, 72 kvWarm6.
enum ANEW8DiTSchedulerPolicy {
    static let prefetchDepth = 3
    static let retireDepth = 1
    static let fullPinnedBlocks = 4
    static let warmPinnedBlocks = 8
    static let measuredSafetyCeilingPrograms = 80

    static func pinnedBlocks(for profile: ANEW8DiTModelProfile) -> Int {
        profile == .full8 ? fullPinnedBlocks : warmPinnedBlocks
    }

    static func theoreticalPeakPrograms(for profile: ANEW8DiTModelProfile) -> Int {
        (pinnedBlocks(for: profile) + prefetchDepth + retireDepth) * profile.programCount
    }
}

struct ANEW8ScheduledModels {
    let models: ANEW8DiTModels
    /// Sum of private `_ANEClient loadModel` time for models that entered the
    /// scheduler while servicing this block. Background-prefetched work is
    /// charged once, when its block first consumes it.
    let newlyLoadedMilliseconds: Double
    /// Foreground wall time spent waiting for scheduler setup / a load future.
    /// Background load/unload work that was fully hidden is intentionally 0.
    let waitMilliseconds: Double
}

/// One block's private-ANE projection set. `crossK`/`crossV` stay non-optional
/// for source compatibility with old diagnostics, but a kvWarm6 set aliases
/// them to `crossQ` and marks `hasCrossKVModels == false`; production guards
/// that such a set is only used on an exact P5 cache hit. Aliased entries are
/// never evaluated or invalidated separately.
final class ANEW8DiTModels: @unchecked Sendable {
    let selfQKV: A12ANEQKVModel
    let selfO: A12ANEProjectionModel
    let crossQ: A12ANEProjectionModel
    let crossK: A12ANEProjectionModel
    let crossV: A12ANEProjectionModel
    let crossO: A12ANEProjectionModel
    let mlpUp: A12ANEProjectionModel
    let mlpDown: A12ANEProjectionModel
    let loadMilliseconds: Double

    private let lock = NSLock()
    private var ownsCrossKV: Bool
    private var invalidated = false

    var hasCrossKVModels: Bool {
        lock.lock(); defer { lock.unlock() }
        return ownsCrossKV && !invalidated
    }

    init(
        selfQKV: A12ANEQKVModel,
        selfO: A12ANEProjectionModel,
        crossQ: A12ANEProjectionModel,
        crossK: A12ANEProjectionModel?,
        crossV: A12ANEProjectionModel?,
        crossO: A12ANEProjectionModel,
        mlpUp: A12ANEProjectionModel,
        mlpDown: A12ANEProjectionModel
    ) {
        self.selfQKV = selfQKV
        self.selfO = selfO
        self.crossQ = crossQ
        self.crossK = crossK ?? crossQ
        self.crossV = crossV ?? crossQ
        self.crossO = crossO
        self.mlpUp = mlpUp
        self.mlpDown = mlpDown
        ownsCrossKV = crossK != nil && crossV != nil
        loadMilliseconds = selfQKV.loadMilliseconds + selfO.loadMilliseconds
            + crossQ.loadMilliseconds
            + (crossK?.loadMilliseconds ?? 0)
            + (crossV?.loadMilliseconds ?? 0)
            + crossO.loadMilliseconds + mlpUp.loadMilliseconds + mlpDown.loadMilliseconds
    }

    /// After a block has populated its exact generation-local P5 K/V cache,
    /// the two invariant projection programs are dead weight. Drop only those
    /// two while retaining the six dynamic programs for the warm pinned prefix.
    func dropCrossKV() {
        lock.lock()
        guard ownsCrossKV, !invalidated else { lock.unlock(); return }
        ownsCrossKV = false
        lock.unlock()
        crossK.invalidate()
        crossV.invalidate()
    }

    func invalidateAll() {
        lock.lock()
        guard !invalidated else { lock.unlock(); return }
        invalidated = true
        let invalidateCrossKV = ownsCrossKV
        ownsCrossKV = false
        lock.unlock()

        selfQKV.invalidate()
        selfO.invalidate()
        crossQ.invalidate()
        if invalidateCrossKV {
            crossK.invalidate()
            crossV.invalidate()
        }
        crossO.invalidate()
        mlpUp.invalidate()
        mlpDown.invalidate()
    }

    deinit { invalidateAll() }
}

/// Stateless loader for already-prepared Espresso programs. Pack validation and
/// prepared-cache completeness are paid exactly once at scheduler construction;
/// every hot-path load below is only model construction + `_ANEClient loadModel`.
private final class ANEW8DiTModelLoader: @unchecked Sendable {
    private let file: AnimapkFile

    init(file: AnimapkFile) throws {
        self.file = file
        try ANEW8NativePack.validate(file: file)
        guard ANEW8ModelPreparer.isPrepared(file: file) else {
            throw AnimapkError.validation(
                "ANE-native model cache is not prepared; prepare the pack before diffusion")
        }
        guard A12ANEIsAvailable() else {
            throw AnimapkError.validation("A12 ANE runtime unavailable: \(A12ANERuntimeStatus())")
        }
    }

    func load(block: Int, profile: ANEW8DiTModelProfile) throws -> ANEW8DiTModels {
        guard (0..<ModelConstants.ditBlocks).contains(block) else {
            throw AnimapkError.validation("ANE block index out of range: \(block)")
        }
        func digest(_ suffix: String) throws -> String {
            let tensor = try ANEW8NativePack.tensor(file: file, block: block, suffix: suffix)
            guard let digest = tensor.blobSHA256 else {
                throw AnimapkError.validation("ANE-native tensor hash missing")
            }
            return digest
        }
        func projection(_ suffix: String) throws -> A12ANEProjectionModel {
            guard let spec = ANEW8NativePack.spec(suffix: suffix) else {
                throw AnimapkError.validation("unknown ANE projection suffix: \(suffix)")
            }
            let key = ANEW8NativePack.projectionCacheKey(
                block: block, tag: spec.tag, hash: try digest(suffix))
            return try A12ANEProjectionModel(
                preparedInputChannels: UInt(spec.columns),
                outputChannels: UInt(spec.rows), spatial: UInt(spec.spatial),
                label: "dit_b\(block)_\(spec.tag)", cacheKey: key)
        }

        let qHash = try digest("self_attn.q_proj.weight")
        let kHash = try digest("self_attn.k_proj.weight")
        let vHash = try digest("self_attn.v_proj.weight")
        let qkvKey = ANEW8NativePack.qkvCacheKey(block: block, q: qHash, k: kHash, v: vHash)
        let selfQKV = try A12ANEQKVModel(
            preparedChannels: UInt(DiTBlockExecutor.dim),
            spatial: UInt(DiTBlockExecutor.tokens),
            label: "dit_b\(block)_self_qkv", cacheKey: qkvKey)
        let crossQ = try projection("cross_attn.q_proj.weight")
        let crossK = profile.includesCrossKV ? try projection("cross_attn.k_proj.weight") : nil
        let crossV = profile.includesCrossKV ? try projection("cross_attn.v_proj.weight") : nil
        return ANEW8DiTModels(
            selfQKV: selfQKV,
            selfO: try projection("self_attn.output_proj.weight"),
            crossQ: crossQ,
            crossK: crossK,
            crossV: crossV,
            crossO: try projection("cross_attn.output_proj.weight"),
            mlpUp: try projection("mlp.layer1.weight"),
            mlpDown: try projection("mlp.layer2.weight"))
    }
}

private final class ANEW8LoadResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<ANEW8DiTModels, Error>?
    func set(_ result: Result<ANEW8DiTModels, Error>) {
        lock.lock(); value = result; lock.unlock()
    }
    func take() -> Result<ANEW8DiTModels, Error>? {
        lock.lock(); defer { lock.unlock() }
        let result = value
        value = nil
        return result
    }
}

private final class ANEW8LoadFuture: @unchecked Sendable {
    let block: Int
    private let semaphore = DispatchSemaphore(value: 0)
    private let box = ANEW8LoadResultBox()

    init(block: Int, profile: ANEW8DiTModelProfile,
         loader: ANEW8DiTModelLoader, queue: DispatchQueue) {
        self.block = block
        queue.async {
            do { self.box.set(.success(try loader.load(block: block, profile: profile))) }
            catch { self.box.set(.failure(error)) }
            self.semaphore.signal()
        }
    }

    func wait() throws -> ANEW8DiTModels {
        semaphore.wait()
        guard let result = box.take() else {
            throw AnimapkError.validation("ANE scheduler load future returned no result")
        }
        return try result.get()
    }
}

private final class ANEW8RetireFuture: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)

    init(models: ANEW8DiTModels, queue: DispatchQueue) {
        queue.async {
            models.invalidateAll()
            self.semaphore.signal()
        }
    }

    func wait() { semaphore.wait() }
}

/// Generation-lifetime bounded private-ANE scheduler. It recognizes each
/// ordered 0...27 DiT traversal itself, so `DitForward` does not need an ANE-
/// specific loop. First/KV-miss traversals use full8 4+3+retire1. Once the
/// generation-local P5 cache is warm, blocks use six dynamic programs with an
/// 8+3+retire1 policy. The pinned prefix survives across diffusion steps.
final class ANEW8DiTModelCache: @unchecked Sendable {
    private let loader: ANEW8DiTModelLoader
    private let loadQueue = DispatchQueue(
        label: "com.invisiblestrangler.AnimaXS.ane-production-loader", qos: .userInitiated)
    private let retireQueue = DispatchQueue(
        label: "com.invisiblestrangler.AnimaXS.ane-production-retire", qos: .userInitiated)

    private var pinned: [Int: ANEW8DiTModels] = [:]
    private var futures: [Int: ANEW8LoadFuture] = [:]
    private var activeStreamed: [Int: ANEW8DiTModels] = [:]
    private var retiring: ANEW8RetireFuture?
    private var traversalProfile: ANEW8DiTModelProfile?
    private var expectedBlock = 0
    private var pendingLoadMilliseconds = 0.0
    private var pendingForegroundWaitMilliseconds = 0.0

    /// Old diagnostic callers still use the historical unbounded API. Keep it
    /// isolated from production scheduler state so V2/V3 research code compiles.
    private var legacyModelsByBlock: [Int: ANEW8DiTModels] = [:]

    init(file: AnimapkFile) throws {
        loader = try ANEW8DiTModelLoader(file: file)
        precondition(
            ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .full8)
                <= ANEW8DiTSchedulerPolicy.measuredSafetyCeilingPrograms)
        precondition(
            ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .kvWarm6)
                <= ANEW8DiTSchedulerPolicy.measuredSafetyCeilingPrograms)
    }

    /// Production API. `kvWarm` must be the actual per-block P5 readiness, not
    /// a diffusion-step number; resumed/partial runs therefore remain correct.
    func scheduledModels(for block: Int, kvWarm: Bool) throws -> ANEW8ScheduledModels {
        let profile: ANEW8DiTModelProfile = kvWarm ? .kvWarm6 : .full8
        guard block == expectedBlock else {
            throw AnimapkError.validation(
                "ANE scheduler expected block \(expectedBlock), got \(block)")
        }
        if block == 0 {
            try beginTraversal(profile: profile)
        } else if traversalProfile != profile {
            throw AnimapkError.validation(
                "ANE scheduler profile changed inside traversal at block \(block)")
        }

        let models: ANEW8DiTModels
        var waitMilliseconds = takePendingForegroundWait()
        var loadedMilliseconds = takePendingLoadMilliseconds()
        let pinnedCount = ANEW8DiTSchedulerPolicy.pinnedBlocks(for: profile)
        if block < pinnedCount {
            guard let value = pinned[block] else {
                throw AnimapkError.validation("ANE scheduler missing pinned block \(block)")
            }
            models = value
        } else {
            guard let future = futures.removeValue(forKey: block) else {
                throw AnimapkError.validation("ANE scheduler missing future for block \(block)")
            }
            let waitStart = ProcessInfo.processInfo.systemUptime
            models = try future.wait()
            waitMilliseconds += elapsedMS(since: waitStart)
            loadedMilliseconds += models.loadMilliseconds
            activeStreamed[block] = models
        }
        expectedBlock += 1
        return ANEW8ScheduledModels(
            models: models,
            newlyLoadedMilliseconds: loadedMilliseconds,
            waitMilliseconds: waitMilliseconds)
    }

    /// Called only after all ANE/Metal work for this block has completed.
    /// Pinned cache-miss blocks immediately shed crossK/crossV once the exact
    /// generation-local P5 entry exists. Streamed blocks use bounded retire1.
    func complete(block: Int, crossKVReady: Bool) throws {
        guard let profile = traversalProfile else {
            throw AnimapkError.validation("ANE scheduler completion outside traversal")
        }
        let pinnedCount = ANEW8DiTSchedulerPolicy.pinnedBlocks(for: profile)
        if block < pinnedCount {
            if crossKVReady { pinned[block]?.dropCrossKV() }
        } else {
            guard let models = activeStreamed.removeValue(forKey: block) else {
                throw AnimapkError.validation("ANE scheduler missing active streamed block \(block)")
            }
            if let previous = retiring { previous.wait() }
            retiring = ANEW8RetireFuture(models: models, queue: retireQueue)

            let newBlock = block + ANEW8DiTSchedulerPolicy.prefetchDepth
            if newBlock < ModelConstants.ditBlocks {
                futures[newBlock] = ANEW8LoadFuture(
                    block: newBlock, profile: profile, loader: loader, queue: loadQueue)
            }
        }

        if block == ModelConstants.ditBlocks - 1 {
            if let finalRetire = retiring { finalRetire.wait() }
            retiring = nil
            guard futures.isEmpty, activeStreamed.isEmpty else {
                throw AnimapkError.validation("ANE scheduler leaked streamed state at traversal end")
            }
            traversalProfile = nil
            expectedBlock = 0
        }
    }

    /// Failure/cancellation cleanup. This is intentionally stronger than the
    /// happy-path traversal end: all pinned state is released because the
    /// generation is no longer useful and must not leave private residency.
    func abortTraversal() {
        if let retiring { retiring.wait() }
        retiring = nil
        for (_, future) in futures {
            if let models = try? future.wait() { models.invalidateAll() }
        }
        futures.removeAll()
        for (_, models) in activeStreamed { models.invalidateAll() }
        activeStreamed.removeAll()
        for (_, models) in pinned { models.invalidateAll() }
        pinned.removeAll()
        traversalProfile = nil
        expectedBlock = 0
        pendingLoadMilliseconds = 0
        pendingForegroundWaitMilliseconds = 0
    }

    /// Historical diagnostic API: unbounded, no scheduler. Production must use
    /// `scheduledModels(for:kvWarm:)` so residency stays bounded.
    func models(for block: Int) throws -> (models: ANEW8DiTModels, newlyLoadedMilliseconds: Double) {
        if let cached = legacyModelsByBlock[block] { return (cached, 0) }
        let models = try loader.load(block: block, profile: .full8)
        legacyModelsByBlock[block] = models
        return (models, models.loadMilliseconds)
    }

    deinit {
        abortTraversal()
        for (_, models) in legacyModelsByBlock { models.invalidateAll() }
        legacyModelsByBlock.removeAll()
    }

    private func beginTraversal(profile: ANEW8DiTModelProfile) throws {
        guard traversalProfile == nil, futures.isEmpty, activeStreamed.isEmpty else {
            throw AnimapkError.validation("ANE scheduler began traversal with residual streamed state")
        }
        traversalProfile = profile
        expectedBlock = 0

        let setupStart = ProcessInfo.processInfo.systemUptime
        var setupLoadMilliseconds = 0.0
        let targetPinned = ANEW8DiTSchedulerPolicy.pinnedBlocks(for: profile)

        // Exact P5 transition: preserve the six useful programs in the first
        // four pins instead of throwing them away and reloading them.
        if profile == .kvWarm6 {
            for block in 0..<min(ANEW8DiTSchedulerPolicy.fullPinnedBlocks, targetPinned) {
                pinned[block]?.dropCrossKV()
            }
        }

        // A cache-allocation failure keeps every traversal in full8. If the
        // profile is warm, expand the retained prefix from 4 -> 8 using six-
        // program loads. Any incompatible stale pin is rebuilt defensively.
        for block in 0..<targetPinned {
            if let existing = pinned[block] {
                if profile == .full8 && !existing.hasCrossKVModels {
                    existing.invalidateAll()
                    pinned.removeValue(forKey: block)
                } else {
                    continue
                }
            }
            let models = try loader.load(block: block, profile: profile)
            pinned[block] = models
            setupLoadMilliseconds += models.loadMilliseconds
        }

        // If a future policy ever reduces the pinned prefix, retire surplus
        // pins now. Current measured transition only grows 4 -> 8.
        let surplus = pinned.keys.filter { $0 >= targetPinned }
        for block in surplus {
            pinned.removeValue(forKey: block)?.invalidateAll()
        }

        if targetPinned < ModelConstants.ditBlocks {
            for block in targetPinned..<min(
                ModelConstants.ditBlocks,
                targetPinned + ANEW8DiTSchedulerPolicy.prefetchDepth) {
                futures[block] = ANEW8LoadFuture(
                    block: block, profile: profile, loader: loader, queue: loadQueue)
            }
        }
        pendingLoadMilliseconds += setupLoadMilliseconds
        pendingForegroundWaitMilliseconds += elapsedMS(since: setupStart)
    }

    private func takePendingLoadMilliseconds() -> Double {
        let value = pendingLoadMilliseconds
        pendingLoadMilliseconds = 0
        return value
    }

    private func takePendingForegroundWait() -> Double {
        let value = pendingForegroundWaitMilliseconds
        pendingForegroundWaitMilliseconds = 0
        return value
    }

    private func elapsedMS(since start: Double) -> Double {
        (ProcessInfo.processInfo.systemUptime - start) * 1_000
    }
}'''

replace_between(
    "AnimaXS/Runtime/ANE/ANEW8MLPBackend.swift",
    "final class ANEW8DiTModels {",
    "/// Reused shared IOSurfaces for one block execution.",
    backend,
)

# ---- DiTBlockExecutor production integration -------------------------------
path = "AnimaXS/Runtime/Metal/DiTBlockExecutor.swift"
replace_once(
    path,
    '''        // Espresso files were materialized before diffusion. First use here
        // only constructs/loads the already-prepared ANE programs; no weight
        // quantization, hashing or model.espresso.weights write is allowed.
        let modelResult = try aneModelCache.models(for: blockIndex)
        if modelResult.newlyLoadedMilliseconds > 0 {
            metrics?.recordANEModelLoad(seconds: modelResult.newlyLoadedMilliseconds / 1_000.0)
        }
        let models = modelResult.models
''',
    '''        // P5 readiness is the source of truth (not diffusion step number).
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
''')

replace_once(
    path,
    '''        let crossCacheHit = optimization.crossKVCache && (crossKVCache?.isReady(blockIndex) ?? false)
''',
    '''        // `crossCacheHit` was resolved before ANE model acquisition so the
        // scheduler can omit crossK/crossV entirely on a real cache hit.
''')

# Both generic and ANE attention helpers should use actual cache allocation as
# truth. For non-ANE paths DiffusionSampler only allocates on the P5 toggle, so
# their behavior remains unchanged; ANE can auto-enable P5 without mutating UI.
p = Path(path)
text = p.read_text()
old = "let cacheEnabled = cross && optimization.crossKVCache"
count = text.count(old)
if count != 2:
    raise SystemExit(f"{path}: expected two cacheEnabled sites, found {count}")
text = text.replace(old, "let cacheEnabled = cross && crossKVCache != nil")
p.write_text(text)

replace_once(
    path,
    '''        streamer.complete(slot)
        slotReleased = true
        metrics?.endBlock()
        context.refreshDiagnostics()
        metrics?.recordMemory(
            allocated: context.currentAllocatedSize,
            available: UInt64(os_proc_available_memory()))
        if let diagnosticBranchCompleted {
            try diagnosticBranchCompleted("self", selfSnapshot!)
            try diagnosticBranchCompleted("cross", crossSnapshot!)
            try diagnosticBranchCompleted("mlp", mlpSnapshot!)
        }
''',
    '''        streamer.complete(slot)
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
''')

# ---- Automatic exact P5 allocation for ANE ---------------------------------
path = "AnimaXS/Runtime/Sampler/DiffusionSampler.swift"
replace_once(
    path,
    '''        // P5: per-generation cross-attention K/V cache. Created when the toggle
        // is on; if the device cannot allocate the buffer it fails gracefully
        // to nil and the legacy per-step projection path runs (never crashes).
        let cache = optimization.crossKVCache ? CrossKVCache(device: context.device) : nil
''',
    '''        // P5: per-generation exact cross-attention K/V cache. The ANE W8
        // backend always requests it because device measurements show that the
        // post-hit six-program working set materially reduces private-runtime
        // load/residency/unload cost. Allocation failure still falls back
        // safely to bounded full8 execution for every traversal.
        let cache = Self.shouldUseCrossKVCache(optimization: optimization)
            ? CrossKVCache(device: context.device) : nil
''')
replace_once(
    path,
    '''    /// Runs the eight model evaluations and writes the final fp32 latent.
''',
    '''    /// Pure policy seam: P5 remains opt-in for Metal backends, while ANE
    /// makes exact K/V reuse part of its measured production recipe.
    static func shouldUseCrossKVCache(optimization: InferenceOptimizationConfig) -> Bool {
        optimization.crossKVCache || optimization.linearBackend == .aneHybridW8
    }

    /// Runs the eight model evaluations and writes the final fp32 latent.
''')

# ---- Telemetry must report effective automatic ANE P5 ----------------------
path = "AnimaXS/Runtime/Diagnostics/GenerationMetrics.swift"
replace_once(
    path,
    '''            lines.append("Numerical monitor: \\(config.numericalMonitoring ? \"on\" : \"off\")")
            lines.append("Mmap no-copy weight source: \\(config.noCopyWeightSource ? \"on\" : \"off\")")
''',
    '''            lines.append("Numerical monitor: \\(config.numericalMonitoring ? \"on\" : \"off\")")
            if config.linearBackend == .aneHybridW8 && !config.crossKVCache {
                lines.append("Cross-attention K/V cache: auto (ANE)")
            } else {
                lines.append("Cross-attention K/V cache: \\(config.crossKVCache ? \"on\" : \"off\")")
            }
            lines.append("Mmap no-copy weight source: \\(config.noCopyWeightSource ? \"on\" : \"off\")")
''')

# ---- Diagnostics UI: explain why the standalone P5 toggle may remain off ---
path = "AnimaXS/App/DiagnosticsView.swift"
replace_once(
    path,
    '''            Text("ANE hybrid (A12/H11): W8-only experimental device backend. Self/cross-attention projection GEMMs and MLP1/MLP2 run on ANE; AdaLN, RMSNorm, RoPE, attention, GELU and residual math stay on Metal. Use for sideload/device testing, not App Store distribution.")
''',
    '''            Text("ANE hybrid (A12/H11): W8-only experimental device backend. Self/cross-attention projection GEMMs and MLP1/MLP2 run on ANE; AdaLN, RMSNorm, RoPE, attention, GELU and residual math stay on Metal. Exact cross-attention K/V caching is automatically requested for this backend even when the standalone P5 toggle is off; if its private buffer cannot be allocated, ANE falls back to bounded full8 streaming. Use for sideload/device testing, not App Store distribution.")
''')

# ---- Pure regression tests for measured scheduler policy + automatic P5 ----
path = "AnimaXSTests/DiTBlockExecutorTests.swift"
insert = r'''
    func testANEProductionSchedulerPolicyStaysInsideMeasuredCeiling() {
        XCTAssertEqual(ANEW8DiTModelProfile.full8.programCount, 8)
        XCTAssertEqual(ANEW8DiTModelProfile.kvWarm6.programCount, 6)
        XCTAssertEqual(ANEW8DiTSchedulerPolicy.prefetchDepth, 3)
        XCTAssertEqual(ANEW8DiTSchedulerPolicy.retireDepth, 1)
        XCTAssertEqual(ANEW8DiTSchedulerPolicy.pinnedBlocks(for: .full8), 4)
        XCTAssertEqual(ANEW8DiTSchedulerPolicy.pinnedBlocks(for: .kvWarm6), 8)
        XCTAssertEqual(ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .full8), 64)
        XCTAssertEqual(ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .kvWarm6), 72)
        XCTAssertLessThanOrEqual(
            ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .full8),
            ANEW8DiTSchedulerPolicy.measuredSafetyCeilingPrograms)
        XCTAssertLessThanOrEqual(
            ANEW8DiTSchedulerPolicy.theoreticalPeakPrograms(for: .kvWarm6),
            ANEW8DiTSchedulerPolicy.measuredSafetyCeilingPrograms)
    }

    func testANECrossKVCacheIsAutomaticWithoutChangingMetalDefault() {
        var config = InferenceOptimizationConfig.currentBaseline
        XCTAssertFalse(config.crossKVCache)
        XCTAssertEqual(config.linearBackend, .dequantizedMPS)
        XCTAssertFalse(DiffusionSampler.shouldUseCrossKVCache(optimization: config))

        config.crossKVCache = true
        XCTAssertTrue(DiffusionSampler.shouldUseCrossKVCache(optimization: config))

        config.crossKVCache = false
        config.linearBackend = .aneHybridW8
        XCTAssertTrue(DiffusionSampler.shouldUseCrossKVCache(optimization: config))
    }

'''
replace_once(path, "    /// Synthetic residual with real per-row variance (alternating sign,\n", insert + "    /// Synthetic residual with real per-row variance (alternating sign,\n")

# Final structural assertions: production must no longer use the old unbounded
# retrieval site, and the six-program profile must be wired to cache readiness.
block_text = Path("AnimaXS/Runtime/Metal/DiTBlockExecutor.swift").read_text()
if "aneModelCache.models(for: blockIndex)" in block_text:
    raise SystemExit("old unbounded production ANE model-cache call remains")
if "scheduledModels(\n            for: blockIndex, kvWarm: crossCacheHit)" not in block_text:
    raise SystemExit("production scheduledModels call missing")
if block_text.count("let cacheEnabled = cross && crossKVCache != nil") != 2:
    raise SystemExit("effective P5 cache allocation is not the source of truth at both sites")

print("ANE production scheduler materialized successfully")
