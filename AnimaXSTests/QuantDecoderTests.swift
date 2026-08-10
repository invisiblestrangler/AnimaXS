import XCTest
@testable import AnimaXS

/// CPU reference W4/W8 decoder tests — byte-compatibility mechanics validated against
/// PHASE0_2_HANDOFF/HANDOFF.md §12–§13. The exact pack-derived known vectors are tested
/// against the real packs in `RealPackDecoderTests` (gated on ANIMAXS_PACKS_DIR); these
/// tests verify the nibble ordering, group math, and equation in CI without multi-GB assets.
final class QuantDecoderTests: XCTestCase {

    private func raw(_ d: Data) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: d.withUnsafeBytes { $0.baseAddress }, count: d.count)
    }

    private func fp16Data(_ values: [Float]) -> Data {
        var d = Data()
        for v in values {
            var h = Float16(v)
            var bits = h.bitPattern.littleEndian
            d.append(contentsOf: withUnsafeBytes(of: &bits) { [UInt8]($0) })
        }
        return d
    }

    // MARK: W4 nibble order (from HANDOFF §12: byte 0x6E=110 → low=14 even, high=6 odd)

    func testW4EvenIndexLowNibble() {
        // One byte 0x6E = binary 0110 1110 → low nibble 1110=14 (even idx 0), high nibble 0110=6 (odd idx 1)
        let data = Data([0x6E])
        let scale = fp16Data([0.0004537])
        let zero = fp16Data([0.0])
        var out = [Float](repeating: 0, count: 2)
        out.withUnsafeMutableBytes { ob in
            QuantDecoders.dequantW4(data: raw(data), scale: raw(scale), zero: raw(zero), k: 2, into: ob)
        }
        XCTAssertEqual(out[0], 14 * 0.0004537, accuracy: 1e-4, "even K index reads low nibble")
        XCTAssertEqual(out[1], 6 * 0.0004537, accuracy: 1e-4, "odd K index reads high nibble")
    }

    func testW4EquationValueEqualsQTimesScalePlusZero() {
        // q=10, scale=0.5, zero=-0.25 → 10*0.5 + (-0.25) = 4.75
        // Pack q=10 into low nibble of byte 0x0A (byte value 10).
        let data = Data([0x0A])
        let scale = fp16Data([0.5])
        let zero = fp16Data([-0.25])
        var out = [Float](repeating: 0, count: 1)
        out.withUnsafeMutableBytes { ob in
            QuantDecoders.dequantW4(data: raw(data), scale: raw(scale), zero: raw(zero), k: 1, into: ob)
        }
        XCTAssertEqual(out[0], 4.75, accuracy: 1e-3)
    }

    func testW4GroupIndexingAlongK() {
        // Two groups: idx0 uses scale[0], idx64 uses scale[1].
        // q=7 low nibble at idx0 (byte 0x07); q=3 at idx64 → byte index 32, low nibble 0x03.
        var data = Data(repeating: 0, count: 64) // covers idx0..63 (byte 0..31)
        data[0] = 0x07
        var data2 = Data(repeating: 0, count: 64) // covers idx64..127
        data2[0] = 0x03
        let scale = fp16Data([1.0, 2.0])
        let zero = fp16Data([0.0, 0.0])
        var out = [Float](repeating: 0, count: 128)
        var full = Data()
        full.append(data)
        full.append(data2)
        out.withUnsafeMutableBytes { ob in
            QuantDecoders.dequantW4(data: raw(full), scale: raw(scale), zero: raw(zero), k: 128, into: ob)
        }
        XCTAssertEqual(out[0], 7.0, accuracy: 1e-3, "idx0 group0 q=7 scale=1")
        XCTAssertEqual(out[64], 6.0, accuracy: 1e-3, "idx64 group1 q=3 scale=2")
    }

    // MARK: W8 (uint8, group 64, value = q*scale + zero)

    func testW8Equation() {
        let data = Data([10, 20])
        let scale = fp16Data([0.5, 0.25])
        let zero = fp16Data([-1.0, 2.0])
        var out = [Float](repeating: 0, count: 2)
        out.withUnsafeMutableBytes { ob in
            QuantDecoders.dequantW8(data: raw(data), scale: raw(scale), zero: raw(zero), k: 2, into: ob)
        }
        XCTAssertEqual(out[0], 10 * 0.5 + (-1.0), accuracy: 1e-3)
        XCTAssertEqual(out[1], 20 * 0.25 + 2.0, accuracy: 1e-3)
    }

    func testW8GroupIndexing() {
        var data = Data(repeating: 0, count: 64)
        data[0] = 5   // group 0
        data[64] = 9  // group 1 (index 64 → scale[1])
        let scale = fp16Data([1.0, 10.0])
        let zero = fp16Data([0.0, 0.0])
        var out = [Float](repeating: 0, count: 65)
        out.withUnsafeMutableBytes { ob in
            QuantDecoders.dequantW8(data: raw(data), scale: raw(scale), zero: raw(zero), k: 65, into: ob)
        }
        XCTAssertEqual(out[0], 5.0, accuracy: 1e-3)
        XCTAssertEqual(out[64], 9 * 10.0, accuracy: 1e-3)
    }

    // MARK: fp16 reader

    func testFP16Read() {
        let data = fp16Data([1.5, -2.25, 0.0, 65504.0])
        var out = [Float](repeating: 0, count: 4)
        out.withUnsafeMutableBytes { ob in
            QuantDecoders.fp16ToFloat32(raw(data), count: 4, into: ob)
        }
        XCTAssertEqual(out[0], 1.5)
        XCTAssertEqual(out[1], -2.25)
        XCTAssertEqual(out[2], 0.0)
        XCTAssertEqual(out[3], 65504.0)
    }
}
