import Foundation
import Metal

/// One-slot, bounded-memory uploader for locator-derived execution ranges.
/// The caller must await completion of work using the current slot before loading
/// another range. H006 deliberately serializes blocks around this contract.
final class WeightStreamer {
    let capacity: Int
    let ring: MTLBuffer
    private(set) var loadedLogicalIndex: Int?

    init(device: MTLDevice, capacity: Int) throws {
        guard capacity > 0 else {
            throw AnimapkError.validation("weight ring capacity must be positive")
        }
        guard let ring = device.makeBuffer(length: capacity, options: .storageModeShared) else {
            throw AnimapkError.validation("failed to allocate weight ring")
        }
        self.capacity = capacity
        self.ring = ring
    }

    func load(_ range: AnimapkExecutionRange, from file: AnimapkFile) throws {
        guard range.length <= UInt64(capacity), range.length <= UInt64(Int.max) else {
            throw AnimapkError.validation("execution range does not fit the weight ring")
        }
        let source = try file.bytes(in: range.fileRange)
        guard let base = source.baseAddress else {
            throw AnimapkError.validation("execution range has no mapped bytes")
        }
        memcpy(ring.contents(), base, source.count)
        loadedLogicalIndex = range.logicalIndex
    }
}
