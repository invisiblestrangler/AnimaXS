import Foundation
import Metal

/// Bounded-memory uploader for locator-derived execution ranges.
///
/// One-slot mode (default) preserves the historical H006 contract: the caller
/// must await completion of work using the current slot before loading another
/// range. Two-slot ping-pong mode (Phase 12) lets the CPU memcpy the next
/// block's weights while the GPU executes the current block: the loop loads
/// block N+1 into the other slot after committing block N and before awaiting
/// it. The streamer enforces the hard invariant itself: `load` refuses to
/// overwrite a slot that an in-flight command buffer still references
/// (`markInFlight`/`complete`), regardless of caller discipline.
final class WeightStreamer {
    /// Default slot count: 1 (preparation/final layer, and any caller that
    /// does not opt into ping-pong).
    static let defaultSlotCount = 1

    let capacity: Int
    private let slotBuffers: [MTLBuffer]
    private(set) var loadedLogicalIndexes: [Int?]
    private var inFlightSlots: Set<Int> = []
    private let stateLock = NSLock()

    /// Number of weight slots.
    var slotCount: Int { slotBuffers.count }

    init(device: MTLDevice, capacity: Int, slotCount: Int = defaultSlotCount) throws {
        guard capacity > 0 else {
            throw AnimapkError.validation("weight ring capacity must be positive")
        }
        guard slotCount > 0, slotCount <= 4 else {
            throw AnimapkError.validation("weight slot count must be in 1...4")
        }
        var buffers: [MTLBuffer] = []
        for _ in 0..<slotCount {
            guard let ring = device.makeBuffer(length: capacity, options: .storageModeShared) else {
                throw AnimapkError.validation("failed to allocate weight slot")
            }
            buffers.append(ring)
        }
        self.capacity = capacity
        self.slotBuffers = buffers
        self.loadedLogicalIndexes = [Int?](repeating: nil, count: slotCount)
    }

    /// Backward-compatible single-slot accessor (preparation/final layer).
    var ring: MTLBuffer { slotBuffers[0] }

    /// The buffer backing `slot`.
    func buffer(for slot: Int) -> MTLBuffer {
        slotBuffers[slot]
    }

    /// Load a range into `slot` (defaults to slot 0, the historical contract).
    /// Throws if the slot is currently referenced by an in-flight command
    /// buffer — a slot must never be overwritten while the GPU may read it.
    func load(_ range: AnimapkExecutionRange, from file: AnimapkFile, slot: Int = 0) throws {
        guard slotBuffers.indices.contains(slot) else {
            throw AnimapkError.validation("weight slot \(slot) out of range")
        }
        stateLock.lock()
        defer { stateLock.unlock() }
        guard !inFlightSlots.contains(slot) else {
            throw AnimapkError.validation(
                "weight slot \(slot) is still referenced by an in-flight command buffer")
        }
        guard range.length <= UInt64(capacity), range.length <= UInt64(Int.max) else {
            throw AnimapkError.validation("execution range does not fit the weight slot")
        }
        let source = try file.bytes(in: range.fileRange)
        guard let base = source.baseAddress else {
            throw AnimapkError.validation("execution range has no mapped bytes")
        }
        memcpy(slotBuffers[slot].contents(), base, source.count)
        loadedLogicalIndexes[slot] = range.logicalIndex
    }

    /// Mark a slot as referenced by a command buffer that has been committed
    /// (or is about to be). `load` into that slot then fails until `complete`.
    func markInFlight(_ slot: Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard slotBuffers.indices.contains(slot) else { return }
        inFlightSlots.insert(slot)
    }

    /// Release a slot after its command buffer completed (or errored).
    func complete(_ slot: Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        inFlightSlots.remove(slot)
    }

    /// Test-only: the currently in-flight slot set.
    var inFlightSlotSet: Set<Int> {
        stateLock.lock()
        defer { stateLock.unlock() }
        return inFlightSlots
    }
}
