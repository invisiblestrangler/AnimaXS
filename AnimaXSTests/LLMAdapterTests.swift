import XCTest
@testable import AnimaXS

/// Protects LLMAdapter primitives that differ from the Qwen encoder (and are easy to get
/// wrong): INTERLEAVED RoPE (rotate_half) vs Qwen's half-split, and exact GELU.
/// Math transcribed from pinned comfy/ldm/anima/model.py (rotate_half, apply_rotary_pos_emb).
final class LLMAdapterTests: XCTestCase {

    func testInterleavedRoPEIdentityAtPositionZero() {
        // pos=0 → freqs 0 → cos=1, sin=0 → output == input.
        let x = [Float](repeating: 1.0, count: 64)
        let out = LLMAdapter.ropeInterleaved(x, positions: [0], headDim: 64)
        for i in 0..<64 {
            XCTAssertEqual(out[i], 1.0, accuracy: 1e-6)
        }
    }

    func testInterleavedRoPEDiffersFromHalfSplit() {
        // The adapter uses interleaved (rotate_half); Qwen uses half-split. They must differ.
        let x = [Float](repeating: 1.0, count: 64)
        let inter = LLMAdapter.ropeInterleaved(x, positions: [3], headDim: 64)
        let halfSplit = QwenNumerics.ropeNeoX(x, positions: [3], theta: 10000.0, headDim: 64)
        var diff: Float = 0
        for i in 0..<64 { diff = max(diff, abs(inter[i] - halfSplit[i])) }
        XCTAssertGreaterThan(diff, 0.1, "interleaved and half-split RoPE must differ for pos>0")
    }

    func testInterleavedRoPESymmetricTwoHeads() {
        // Two heads at different positions must give different rotations.
        let x = [Float](repeating: 1.0, count: 128)  // 2 heads
        let a = LLMAdapter.ropeInterleaved(x, positions: [3, 7], headDim: 64)
        let a0 = Array(a[0..<64]), a1 = Array(a[64..<128])
        let e0 = LLMAdapter.ropeInterleaved([Float](repeating: 1.0, count: 64), positions: [3], headDim: 64)
        let e1 = LLMAdapter.ropeInterleaved([Float](repeating: 1.0, count: 64), positions: [7], headDim: 64)
        for i in 0..<64 {
            XCTAssertEqual(a0[i], e0[i], accuracy: 1e-6)
            XCTAssertEqual(a1[i], e1[i], accuracy: 1e-6)
        }
        XCTAssertNotEqual(e0, e1)
    }

    func testGELUExactMatchesKnownValues() {
        // torch.nn.GELU() default (exact) reference values.
        XCTAssertEqual(LLMAdapter.gelu(0.0), 0.0, accuracy: 1e-6)
        XCTAssertEqual(LLMAdapter.gelu(1.0), 0.8413448, accuracy: 1e-5)
        XCTAssertEqual(LLMAdapter.gelu(-1.0), -0.1586552, accuracy: 1e-5)
        XCTAssertEqual(LLMAdapter.gelu(2.0), 1.9544997, accuracy: 1e-5)
        XCTAssertEqual(LLMAdapter.gelu(0.5), 0.3457312, accuracy: 1e-5)
    }
}
