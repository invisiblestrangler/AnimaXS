import XCTest
@testable import AnimaXS

/// Ping-pong ON/OFF seam tests (§18.4). The loop-shape decision is pure
/// (`DitForward.loopStep`), so ON → two alternating slots with a next-block
/// prefetch and OFF → one slot with NO prefetch are proven without a real pack.
final class PingPongSeamTests: XCTestCase {

    func testPingPongOnUsesTwoAlternatingSlots() {
        let blockCount = 28
        for index in 0..<blockCount {
            let step = DitForward.loopStep(logicalIndex: index, blockCount: blockCount, pingPong: true)
            XCTAssertEqual(step.slot, index % 2, "block \(index) slot")
            if index + 1 < blockCount {
                XCTAssertEqual(step.prefetchIndex, index + 1, "block \(index) prefetch index")
                XCTAssertEqual(step.prefetchSlot, (index + 1) % 2, "block \(index) prefetch slot")
            } else {
                XCTAssertNil(step.prefetchIndex, "last block must not prefetch")
            }
        }
    }

    func testPingPongOffUsesOneSlotAndNeverPrefetches() {
        let blockCount = 28
        for index in 0..<blockCount {
            let step = DitForward.loopStep(logicalIndex: index, blockCount: blockCount, pingPong: false)
            XCTAssertEqual(step.slot, 0, "OFF mode always uses slot 0")
            XCTAssertNil(step.prefetchIndex, "OFF mode must never request a prefetch")
            XCTAssertEqual(step.prefetchSlot, 0)
        }
    }

    func testLastBlockNeverPrefetchesInEitherMode() {
        let blockCount = 28
        XCTAssertNil(DitForward.loopStep(logicalIndex: 27, blockCount: blockCount, pingPong: true).prefetchIndex)
        XCTAssertNil(DitForward.loopStep(logicalIndex: 27, blockCount: blockCount, pingPong: false).prefetchIndex)
    }

    func testSlotCountConfigDrivesStreamerSlots() throws {
        // The streamer enforces the slot count the executor config requests:
        // ping-pong ON -> 2 slots, OFF -> 1 slot.
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let on = try WeightStreamer(device: device, capacity: 64, slotCount: 2)
        let off = try WeightStreamer(device: device, capacity: 64, slotCount: 1)
        XCTAssertEqual(on.slotCount, 2)
        XCTAssertEqual(off.slotCount, 1)
    }
}
