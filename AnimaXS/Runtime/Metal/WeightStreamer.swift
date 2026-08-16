import Foundation
import Metal

/// P6: weight source backend selection.
///
/// - `.copied`: memcpy the execution range into the slot ring (the historical
///   production path, byte-for-byte unchanged).
/// - `.noCopy`: hand Metal an `MTLBuffer` that ALIASES the already mmap'd pack
///   region via `makeBuffer(bytesNoCopy:)`, so the GPU reads the weights
///   directly from the file mapping and the CPU memcpy is eliminated.
enum WeightStorageMode: Equatable {
    case copied
    case noCopy
}

/// Outcome of a P6 `WeightStreamer.load`: which backend served the range and
/// the `MTLBuffer` the caller must bind into the command buffer. On the
/// no-copy path `buffer` aliases the file mapping; on the copied path it is
/// the slot ring buffer (exactly as before).
struct WeightLoadResult {
    let mode: WeightStorageMode
    let buffer: MTLBuffer
    /// Weight bytes served without a CPU copy (0 on the copied path).
    let noCopyBytes: UInt64
}

/// P6-B: eligibility gate + alias construction for the mmap no-copy backend.
///
/// `makeBuffer(bytesNoCopy:)` requires the backing pointer to be page-aligned,
/// so the no-copy path is used ONLY when the range's absolute file offset is
/// 4096-byte aligned; every other range falls back to the copied path. The
/// alias never frees the pointer (the `AnimapkFile` owns the mmap and must
/// outlive the buffer) and never touches past EOF.
enum WeightNoCopyPolicy {
    /// Minimum pointer alignment Metal requires for a `bytesNoCopy` buffer.
    static let pageSize = 4_096

    /// True when the range may be served zero-copy: page-aligned absolute
    /// file offset, `Int`-representable length, and fully inside the file.
    static func isEligible(range: AnimapkExecutionRange, file: AnimapkFile) -> Bool {
        let offset = range.fileRange.lowerBound
        guard offset <= UInt64(Int.max),
              Int(offset) % pageSize == 0,
              range.length <= UInt64(Int.max) else { return false }
        return range.fileRange.upperBound <= file.header.fileSize
    }

    /// Builds the aliasing `MTLBuffer` for an eligible range, or nil when the
    /// range is ineligible or the device refuses the alias. Preferred options
    /// are shared storage + write-combined CPU cache mode (best-effort hint);
    /// if the device rejects that combination it is retried with plain shared
    /// storage. The deallocator does nothing: the `AnimapkFile` owns the mmap
    /// and must outlive the buffer.
    static func makeAlias(device: MTLDevice, range: AnimapkExecutionRange,
                          file: AnimapkFile) -> MTLBuffer? {
        guard isEligible(range: range, file: file),
              let bytes = try? file.bytes(in: range.fileRange),
              let base = bytes.baseAddress else { return nil }
        let mutable = UnsafeMutableRawPointer(mutating: base)
        let length = Int(range.length)
        let preferred: MTLResourceOptions = [.storageModeShared, .cpuCacheModeWriteCombined]
        if let buffer = device.makeBuffer(bytesNoCopy: mutable, length: length,
                                          options: preferred, deallocator: nil) {
            return buffer
        }
        return device.makeBuffer(bytesNoCopy: mutable, length: length,
                                 options: .storageModeShared, deallocator: nil)
    }
}

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
    /// Per-slot buffer backing the CURRENTLY loaded range: the slot ring on
    /// the copied path, or the mmap alias on the no-copy path. `buffer(for:)`
    /// returns this so the caller binds the right memory on both paths and on
    /// the already-loaded fast path.
    private var loadedBuffers: [MTLBuffer?]
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
        self.loadedBuffers = [MTLBuffer?](repeating: nil, count: slotCount)
        self.loadedLogicalIndexes = [Int?](repeating: nil, count: slotCount)
    }

    /// Backward-compatible single-slot accessor (preparation/final layer).
    /// Returns the slot ring buffer; P6 no-copy callers should use
    /// `buffer(for:)` instead so they bind the mmap alias when one is loaded.
    var ring: MTLBuffer { slotBuffers[0] }

    /// The buffer backing the currently loaded range in `slot` (the slot ring
    /// on the copied path, the mmap alias on the no-copy path). Falls back to
    /// the slot ring when nothing has been loaded into the slot yet.
    func buffer(for slot: Int) -> MTLBuffer {
        loadedBuffers[slot] ?? slotBuffers[slot]
    }

    /// Load a range into `slot` (defaults to slot 0, the historical contract).
    /// Throws if the slot is currently referenced by an in-flight command
    /// buffer — a slot must never be overwritten while the GPU may read it.
    ///
    /// `mode` selects the P6 backend: `.copied` (default) keeps the exact
    /// historical memcpy path; `.noCopy` aliases the mmap'd pack region when
    /// the range is page-aligned and the device accepts the alias, and falls
    /// back to `.copied` otherwise (never overmaps, never misaligns, never
    /// touches past EOF). Returns which backend served the range plus the
    /// buffer the caller must bind.
    @discardableResult
    func load(_ range: AnimapkExecutionRange, from file: AnimapkFile, slot: Int = 0,
              mode: WeightStorageMode = .copied) throws -> WeightLoadResult {
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
        // P6 no-copy fast path: alias the mmap'd pack region. `makeAlias`
        // returns nil for non-page-aligned / out-of-bounds ranges or when the
        // device refuses — in every such case fall through to the copied path.
        if mode == .noCopy,
           let alias = WeightNoCopyPolicy.makeAlias(
               device: slotBuffers[slot].device, range: range, file: file) {
            loadedBuffers[slot] = alias
            loadedLogicalIndexes[slot] = range.logicalIndex
            return WeightLoadResult(mode: .noCopy, buffer: alias,
                                    noCopyBytes: UInt64(alias.length))
        }
        let source = try file.bytes(in: range.fileRange)
        guard let base = source.baseAddress else {
            throw AnimapkError.validation("execution range has no mapped bytes")
        }
        memcpy(slotBuffers[slot].contents(), base, source.count)
        loadedBuffers[slot] = slotBuffers[slot]
        loadedLogicalIndexes[slot] = range.logicalIndex
        return WeightLoadResult(mode: .copied, buffer: slotBuffers[slot], noCopyBytes: 0)
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
