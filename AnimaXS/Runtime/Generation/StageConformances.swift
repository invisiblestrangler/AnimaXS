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
