import XCTest
@testable import AnimaXS

/// CPU reference W4/W8 decoder tests — byte-compatibility mechanics validated against
/// PHASE0_2_HANDOFF/HANDOFF.md §12–§13. Group semantics: group = K index / groupSize (64),
/// so consecutive K indices within the same group share scale/zero. The exact pack-derived
/// known vectors are tested against the real packs in `RealPackDecoderTests` (gated on
/// ANIMAXS_PACKS_DIR). These tests verify nibble ordering, group math, and equation in CI.
final class QuantDecoderTests: XCTestCase {

    private func fp16Data(_ values: [Float]) -> Data {
        var d = Data()
        for v in values {
            var h = Float16(v)
            var bits = h.bitPattern.littleEndian
            d.append(contentsOf: withUnsafeBytes(of: &bits) { [UInt8]($0) })
        }
        return d
    }

    // MARK: W4 nibble order (HANDOFF §12: byte 0x6E → low=14 even, high=6 odd)

    func testW4EvenIndexLowNibble() {
        // Single group (group=64) covers both indices 0 and 1.
        let out = QuantDecoders.dequantW4(data: Data([0x6E]), scale: fp16Data([0.0004537]), zero: fp16Data([0.0]), k: 2)
        XCTAssertEqual(out[0], 14 * 0.0004537, accuracy: 1e-4, "even K index reads low nibble")
        XCTAssertEqual(out[1], 6 * 0.0004537, accuracy: 1e-4, "odd K index reads high nibble")
    }

    func testW4EquationValueEqualsQTimesScalePlusZero() {
        // q=10 (low nibble of 0x0A), scale=0.5, zero=-0.25 → 10*0.5 + (-0.25) = 4.75
        let out = QuantDecoders.dequantW4(data: Data([0x0A]), scale: fp16Data([0.5]), zero: fp16Data([-0.25]), k: 1)
        XCTAssertEqual(out[0], 4.75, accuracy: 1e-3)
    }

    func testW4GroupIndexingAlongK() {
        // Group 64: indices 0..63 share scale[0]; index 64 uses scale[1].
        var full = Data(repeating: 0, count: 64)   // covers idx 0..63
        full[0] = 0x07
        full[32] = 0x03                            // idx 64 (byte 32, low nibble)
        let out = QuantDecoders.dequantW4(data: full, scale: fp16Data([1.0, 2.0]), zero: fp16Data([0.0, 0.0]), k: 65)
        XCTAssertEqual(out[0], 7.0, accuracy: 1e-3, "idx0 group0 q=7 scale=1")
        XCTAssertEqual(out[64], 6.0, accuracy: 1e-3, "idx64 group1 q=3 scale=2")
    }

    // MARK: W8 (uint8, group 64, value = q*scale + zero)

    func testW8Equation() {
        // Indices 0 and 1 are BOTH in group 0 (group=64), so both use scale[0], zero[0].
        let out = QuantDecoders.dequantW8(data: Data([10, 20]), scale: fp16Data([0.5, 0.25]), zero: fp16Data([-1.0, 2.0]), k: 2)
        XCTAssertEqual(out[0], 10 * 0.5 + (-1.0), accuracy: 1e-3, "idx0 q=10 scale0=0.5 zero0=-1")
        XCTAssertEqual(out[1], 20 * 0.5 + (-1.0), accuracy: 1e-3, "idx1 still group0: q=20 scale0=0.5 zero0=-1")
    }

    func testW8GroupIndexing() {
        // Group 64: idx0 uses scale[0]; idx64 uses scale[1].
        var data = Data(repeating: 0, count: 65)
        data[0] = 5   // group 0
        data[64] = 9  // group 1
        let out = QuantDecoders.dequantW8(data: data, scale: fp16Data([1.0, 10.0]), zero: fp16Data([0.0, 0.0]), k: 65)
        XCTAssertEqual(out[0], 5.0, accuracy: 1e-3, "idx0 group0 q=5 scale=1")
        XCTAssertEqual(out[64], 9 * 10.0, accuracy: 1e-3, "idx64 group1 q=9 scale=10")
    }

    // MARK: fp16 reader

    func testFP16Read() {
        let out = QuantDecoders.fp16ToFloat32(fp16Data([1.5, -2.25, 0.0, 65504.0]), count: 4)
        XCTAssertEqual(out[0], 1.5)
        XCTAssertEqual(out[1], -2.25)
        XCTAssertEqual(out[2], 0.0)
        XCTAssertEqual(out[3], 65504.0)
    }
}
