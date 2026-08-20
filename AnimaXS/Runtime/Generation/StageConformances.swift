import Foundation

// MARK: - Production stage conformance (K002 seams)

// The production executors already expose the exact `execute`/`decode`
// signatures the engine protocols require; conformance is declaration-only.
extension QwenEncoderMetal: PromptEncoderStage {}
extension LLMAdapterMetal: ContextAdapterStage {}
extension DiffusionSampler: DiffusionStage {}
extension VAEDecoder: VAEDecodeStage {}

// MARK: - Experimental production ANE multiprocedure backend

/// The exact Stage2J/K single-output procedure names. The private ANE model has
/// one copy of each procedure per transformer block; self Q/K/V share the
/// lowered fused-QKV donor weights through the proven deduplicated container.
enum ANEW8MultiProcProcedure: String, CaseIterable {
    case selfQ = "procedure_self_q"
    case selfK = "procedure_self_k"
    case selfV = "procedure_self_v"
    case selfO = "procedure_self_o"
    case crossQ = "procedure_cross_q"
    case crossK = "procedure_cross_k"
    case crossV = "procedure_cross_v"
    case crossO = "procedure_cross_o"
    case mlpUp = "procedure_mlp_up"
    case mlpDown = "procedure_mlp_down"
}

/// One lightweight retained `_ANEInMemoryModel` handle for a real DiT block.
/// Creation performs the one-time Stage2J/K container build + compiled-cache
/// lookup and leaves the model loaded. Streaming retirement unloads it and
/// applies the Stage2M mode-3 trim; later use calls the proven handle reload.
final class ANEW8MultiProcBlockModel: @unchecked Sendable {
    let block: Int
    private let handle: NSMutableDictionary

    init(block: Int, compileMilliseconds: inout Double,
         loadMilliseconds: inout Double) throws {
        guard (0..<ModelConstants.ditBlocks).contains(block) else {
            throw AnimapkError.validation("ANE multiprocedure block index \(block) is out of range")
        }
        guard let created = A12ANEMultiProcCreateLoadedHandle(
            UInt(block), &compileMilliseconds, &loadMilliseconds, nil)
        else {
            throw AnimapkError.validation(
                "ANE multiprocedure model creation failed for block \(block)")
        }
        self.block = block
        self.handle = created
    }

    var isLoaded: Bool { A12ANEMultiProcHandleIsLoaded(handle) }

    @discardableResult
    func load() throws -> Double {
        if isLoaded { return 0 }
        var milliseconds = 0.0
        guard A12ANEMultiProcLoadHandle(handle, &milliseconds) else {
            throw AnimapkError.validation(
                "ANE multiprocedure reload failed for block \(block): "
                + A12ANEMultiProcHandleLastError(handle))
        }
        return milliseconds
    }

    @discardableResult
    func unload() throws -> Double {
        if !isLoaded { return 0 }
        var milliseconds = 0.0
        guard A12ANEMultiProcUnloadHandle(handle, &milliseconds) else {
            throw AnimapkError.validation(
                "ANE multiprocedure unload failed for block \(block): "
                + A12ANEMultiProcHandleLastError(handle))
        }
        return milliseconds
    }

    @discardableResult
    func evaluate(
        _ procedure: ANEW8MultiProcProcedure,
        input: A12ANESurface,
        output: A12ANESurface
    ) throws -> Double {
        var milliseconds = 0.0
        guard A12ANEMultiProcEvaluateHandle(
            handle, procedure.rawValue, input, output, &milliseconds)
        else {
            throw AnimapkError.validation(
                "ANE multiprocedure \(procedure.rawValue) failed for block \(block): "
                + A12ANEMultiProcHandleLastError(handle))
        }
        return milliseconds
    }

    deinit {
        A12ANEMultiProcDestroyHandle(handle)
    }
}

/// Result returned to `DiTBlockExecutor` for one scheduler admission. Load time
/// is private-runtime loadWithQoS time whether the work ran synchronously or on
/// the prefetch queue; wait time is only the foreground time spent waiting for
/// an already-running prefetch.
struct ANEW8MultiProcScheduledModel {
    let model: ANEW8MultiProcBlockModel
    let newlyLoadedMilliseconds: Double
    let compileMilliseconds: Double
    let unloadMilliseconds: Double
    let waitMilliseconds: Double
    let residentBlocks: Int
}

private struct ANEW8MultiProcLoadWork {
    let model: ANEW8MultiProcBlockModel
    let loadMilliseconds: Double
    let compileMilliseconds: Double
    let unloadMilliseconds: Double
    let residentBlocks: Int
}

private final class ANEW8MultiProcLoadFuture: @unchecked Sendable {
    let block: Int
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var result: Result<ANEW8MultiProcLoadWork, Error>?

    init(block: Int) { self.block = block }

    func finish(_ value: Result<ANEW8MultiProcLoadWork, Error>) {
        lock.lock()
        result = value
        lock.unlock()
        semaphore.signal()
    }

    func wait() throws -> (work: ANEW8MultiProcLoadWork, waitMilliseconds: Double) {
        let start = ProcessInfo.processInfo.systemUptime
        semaphore.wait()
        let wait = (ProcessInfo.processInfo.systemUptime - start) * 1_000
        lock.lock()
        let value = result
        lock.unlock()
        guard let value else {
            throw AnimapkError.validation("ANE multiprocedure prefetch completed without a result")
        }
        return (try value.get(), wait)
    }
}

/// Device-measured production candidate: six permanently pinned block models
/// plus two alternating streaming slots. This is deliberately NOT an LRU.
///
/// Why this exact shape:
/// - the previous physical-device scheduler bakeoff selected 6 pinned + 2
///   streaming over 0+2 and 4+2;
/// - Stage2M proved 11 simultaneously resident multiprocedure block models are
///   healthy but the 12th admission wedges the private runtime;
/// - 6+2 therefore caps residency at 8, leaving substantial safety headroom;
/// - Stage2O proved a streamed model's ~14 ms hot state does not survive the
///   21-identity reuse distance, so unloaded handles are retained for cheap
///   reconstruction but are not assumed to stay runtime-hot.
final class ANEW8MultiProcModelCache: @unchecked Sendable {
    static let pinnedBlocks = 6
    static let streamingSlots = 2
    static let maxResidentBlocks = pinnedBlocks + streamingSlots

    private let loaderQueue = DispatchQueue(
        label: "com.invisiblestrangler.AnimaXS.ane-multiproc-loader",
        qos: .userInitiated)

    // Loader-queue confined state.
    private var models: [Int: ANEW8MultiProcBlockModel] = [:]
    private var streamSlots: [Int?] = Array(repeating: nil, count: streamingSlots)
    private var residentHighWater = 0

    // Inference-thread state. The production DiT loop is serial 0...27.
    private var expectedBlock = 0
    private var nextFuture: ANEW8MultiProcLoadFuture?

    init() {
        print("ANE multiprocedure scheduler: pinned=6 streaming=2 maxResident=8")
    }

    /// Returns the block model required by the ordered production traversal.
    /// At block 5 we start loading streaming block 6 while block 5 executes;
    /// every streamed block then starts the next alternating-slot admission.
    func scheduledModel(for block: Int) throws -> ANEW8MultiProcScheduledModel {
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
                // A stale future means traversal sequencing changed. Drain it
                // before touching loader state, then fail loudly rather than
                // silently admitting an extra stream model.
                _ = try? future.wait()
                nextFuture = nil
            }
            work = try loaderQueue.sync { try loadBlock(block) }
        }

        expectedBlock = block == ModelConstants.ditBlocks - 1 ? 0 : block + 1

        // The first streaming admission can overlap the final pinned block.
        // Thereafter each current stream block overlaps admission of the next
        // model into the opposite slot. The queue is serial, so ANE load calls
        // are never issued concurrently with one another.
        let next: Int?
        if block == Self.pinnedBlocks - 1 {
            next = Self.pinnedBlocks
        } else if block >= Self.pinnedBlocks && block + 1 < ModelConstants.ditBlocks {
            next = block + 1
        } else {
            next = nil
        }
        if let next {
            nextFuture = launchPrefetch(block: next)
        }

        return ANEW8MultiProcScheduledModel(
            model: work.model,
            newlyLoadedMilliseconds: work.loadMilliseconds,
            compileMilliseconds: work.compileMilliseconds,
            unloadMilliseconds: work.unloadMilliseconds,
            waitMilliseconds: foregroundWait,
            residentBlocks: work.residentBlocks)
    }

    /// The block executor calls this only after every ANE/Metal consumer of the
    /// current block is finished. Streaming retirement is intentionally delayed
    /// until that slot is reused, so the two-slot invariant exactly matches the
    /// old measured ping-pong scheduler.
    func complete(block: Int) throws {
        guard (0..<ModelConstants.ditBlocks).contains(block) else {
            throw AnimapkError.validation("ANE multiprocedure completion block is out of range")
        }
    }

    /// Failure/cancellation recovery. Waits for any in-flight loader work, then
    /// unloads all resident handles. Lightweight handles stay cached only until
    /// this scheduler object is released by the failed generation.
    func abortTraversal() {
        if let future = nextFuture {
            _ = try? future.wait()
            nextFuture = nil
        }
        loaderQueue.sync {
            for model in models.values where model.isLoaded {
                _ = try? model.unload()
            }
            streamSlots = Array(repeating: nil, count: Self.streamingSlots)
        }
        expectedBlock = 0
    }

    private func launchPrefetch(block: Int) -> ANEW8MultiProcLoadFuture {
        let future = ANEW8MultiProcLoadFuture(block: block)
        loaderQueue.async { [weak self, weak future] in
            guard let self, let future else { return }
            do { future.finish(.success(try self.loadBlock(block))) }
            catch { future.finish(.failure(error)) }
        }
        return future
    }

    /// Loader-queue only. For a streaming block the old occupant is unloaded
    /// before the new model is admitted, making >8 resident block models
    /// structurally impossible.
    private func loadBlock(_ block: Int) throws -> ANEW8MultiProcLoadWork {
        var unloadMS = 0.0
        if block >= Self.pinnedBlocks {
            let slot = (block - Self.pinnedBlocks) % Self.streamingSlots
            if let occupant = streamSlots[slot], occupant != block,
               let old = models[occupant], old.isLoaded {
                unloadMS += try old.unload()
            }
            streamSlots[slot] = nil
        }

        var compileMS = 0.0
        var loadMS = 0.0
        let model: ANEW8MultiProcBlockModel
        if let existing = models[block] {
            model = existing
            if !existing.isLoaded { loadMS = try existing.load() }
        } else {
            model = try ANEW8MultiProcBlockModel(
                block: block,
                compileMilliseconds: &compileMS,
                loadMilliseconds: &loadMS)
            models[block] = model
        }

        if block >= Self.pinnedBlocks {
            let slot = (block - Self.pinnedBlocks) % Self.streamingSlots
            streamSlots[slot] = block
        }

        let resident = models.values.reduce(into: 0) { count, value in
            if value.isLoaded { count += 1 }
        }
        residentHighWater = max(residentHighWater, resident)
        guard resident <= Self.maxResidentBlocks else {
            throw AnimapkError.validation(
                "ANE multiprocedure scheduler residency invariant violated: \(resident) > \(Self.maxResidentBlocks)")
        }
        return ANEW8MultiProcLoadWork(
            model: model,
            loadMilliseconds: loadMS,
            compileMilliseconds: compileMS,
            unloadMilliseconds: unloadMS,
            residentBlocks: resident)
    }

    deinit {
        if let future = nextFuture { _ = try? future.wait() }
        loaderQueue.sync {
            for model in models.values where model.isLoaded { _ = try? model.unload() }
            models.removeAll()
            streamSlots = Array(repeating: nil, count: Self.streamingSlots)
        }
    }
}
