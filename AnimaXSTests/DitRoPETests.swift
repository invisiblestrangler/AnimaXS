import XCTest
@testable import AnimaXS

/// CPU reference tests for the DiT 3-D RoPE embedding (H004).
///
/// The weightless rope tensor is validated structurally against the pinned-ComfyUI oracle
/// (scripts/dit_rope_oracle.py, transcribed verbatim from VideoRopePosition3DEmb
/// position_embedding.py:100-163) — cosine 1.000000 on the 512×512 grid. These pure tests
/// lock the config and the exact 2×2 rotation-block math, layout, and t/h/w freq ordering
/// that are easy to get wrong.
final class DitRoPETests: XCTestCase {

    /// Config: head_dim=128 → dim_h=42, dim_w=42, dim_t=44; 64 freqs (22 temporal + 21 h + 21 w).
    func testConfig() {
        XCTAssertEqual(DitRoPE.headDim, 128)
        XCTAssertEqual(DitRoPE.dimH, 42)
        XCTAssertEqual(DitRoPE.dimW, 42)
        XCTAssertEqual(DitRoPE.dimT, 44)
        XCTAssertEqual(DitRoPE.numFreqs, 64)
    }

    /// Exact computed thetas (position_embedding.py:127-129): h/w use NTK ratio 4.0,
    /// t uses 1.0. NOT the rounded prose values 42871.1/42871.4.
    func testThetas() {
        XCTAssertEqual(DitRoPE.hTheta, Float(10000.0 * pow(4.0, 42.0 / 40.0)), accuracy: 1e-3)
        XCTAssertEqual(DitRoPE.wTheta, DitRoPE.hTheta, accuracy: 1e-6)
        XCTAssertEqual(DitRoPE.tTheta, 10000.0, accuracy: 1e-3)
        // hTheta ≈ 42870.938 (exact), not the rounded 42871.x
        XCTAssertEqual(Double(DitRoPE.hTheta), 42870.938, accuracy: 0.01)
    }

    /// Shape/count: 512×512 → T=1,H=W=32 → 1024 tokens × 64 freqs × 4 (2×2 block).
    func testShapeAndFinite() {
        let rope = DitRoPE.generate(T: 1, H: 32, W: 32)
        XCTAssertEqual(rope.count, 1024 * 64 * 4)
        for v in rope { XCTAssertTrue(v.isFinite) }
        XCTAssertEqual(rope.min() ?? 0, -1.0, accuracy: 1e-3)   // sin/cos range
        XCTAssertEqual(rope.max() ?? 0, 1.0, accuracy: 1e-3)
    }

    /// T=1 image case: temporal freqs are all identity (seq[0]=0 → angle 0 → [1,-0,0,1]),
    /// and token0 (h=0,w=0) has identity height/width blocks too.
    func testTokenZeroIsIdentity() {
        let rope = DitRoPE.generate(T: 1, H: 32, W: 32)
        // temporal [0..22), height [22..43), width [43..64) at token0 all identity
        for f in 0..<DitRoPE.numFreqs {
            let b = block(rope, 0, f)
            XCTAssertEqual(b[0], 1.0, accuracy: 1e-6, "cos = 1")
            XCTAssertEqual(b[1], 0.0, accuracy: 1e-6, "-sin = 0")
            XCTAssertEqual(b[2], 0.0, accuracy: 1e-6, "sin = 0")
            XCTAssertEqual(b[3], 1.0, accuracy: 1e-6, "cos = 1")
        }
    }

    /// Width freq index 0 (block 43) uses range[0]=0 → theta^0=1 → angle = w (radians).
    /// block = [cos(w), -sin(w), sin(w), cos(w)].
    func testWidthFirstFreqRotation() {
        let rope = DitRoPE.generate(T: 1, H: 32, W: 32)
        // token1 = (h=0, w=1): angle = 1 rad
        assertBlock(rope, 1, 43, at: [0.5403023, -0.8414710, 0.8414710, 0.5403023], accuracy: 1e-5, msg: "w=1")
        // token17 = (h=0, w=17): angle = 17 rad
        assertBlock(rope, 17, 43, at: [-0.27516335, 0.96139747, -0.96139747, -0.27516335], accuracy: 1e-5, msg: "w=17")
        // token31 = (h=0, w=31): angle = 31 rad
        assertBlock(rope, 31, 43, at: [0.91474235, 0.40403765, -0.40403765, 0.91474235], accuracy: 1e-5, msg: "w=31")
    }

    /// Height freq index 0 (block 22) uses range[0]=0 → angle = h (radians), same as width.
    func testHeightFirstFreqRotation() {
        let rope = DitRoPE.generate(T: 1, H: 32, W: 32)
        // token1 = (h=0,w=1) → height block 22 angle = h = 0 → identity
        assertBlock(rope, 1, 22, at: [1.0, -0.0, 0.0, 1.0], accuracy: 1e-6, msg: "h=0 identity")
        // token32 = (h=1, w=0) → height block 22 angle = 1 rad
        assertBlock(rope, 32, 22, at: [0.5403023, -0.8414710, 0.8414710, 0.5403023], accuracy: 1e-5, msg: "h=1")
    }

    /// Freq ordering is t, h, w: temporal blocks [0..22), height [22..43), width [43..64).
    /// At token0 all are identity, but a non-zero token distinguishes: freq 22 is height
    /// (varies with h), freq 43 is width (varies with w).
    func testTHWOrdering() {
        let rope = DitRoPE.generate(T: 1, H: 32, W: 32)
        let token = 32 + 5   // (h=1, w=5)
        // height first freq (22): angle = h = 1
        assertBlock(rope, token, 22, at: [0.5403023, -0.8414710, 0.8414710, 0.5403023], accuracy: 1e-5, msg: "height@h=1")
        // width first freq (43): angle = w = 5
        assertBlock(rope, token, 43, at: [Float(cos(5.0)), Float(-sin(5.0)), Float(sin(5.0)), Float(cos(5.0))], accuracy: 1e-5, msg: "width@w=5")
        // temporal (0): identity even for non-zero token (T=1, seq[0]=0)
        assertBlock(rope, token, 0, at: [1.0, -0.0, 0.0, 1.0], accuracy: 1e-6, msg: "temporal identity")
    }

    /// Verify the reshaped 2×2 block layout matches [cos,-sin; sin,cos]: element order in the
    /// flat tensor is (i,j) = (0,0),(0,1),(1,0),(1,1) = cos,-sin,sin,cos.
    func testBlock2x2Layout() {
        let rope = DitRoPE.generate(T: 1, H: 32, W: 32)
        // token17 width-first-freq: angle = 17 rad
        let b = block(rope, 17, 43)
        let c = Float(cos(17.0)), s = Float(sin(17.0))
        XCTAssertEqual(b[0], c, accuracy: 1e-5)   // (0,0)=cos
        XCTAssertEqual(b[1], -s, accuracy: 1e-5)  // (0,1)=-sin
        XCTAssertEqual(b[2], s, accuracy: 1e-5)   // (1,0)=sin
        XCTAssertEqual(b[3], c, accuracy: 1e-5)   // (1,1)=cos
    }

    // MARK: helpers

    private func block(_ rope: [Float], _ tok: Int, _ f: Int) -> [Float] {
        let b = (tok * DitRoPE.numFreqs + f) * 4
        return Array(rope[b..<(b + 4)])
    }

    private func assertBlock(_ rope: [Float], _ tok: Int, _ f: Int,
                             at expected: [Float], accuracy: Float, msg: String) {
        let b = block(rope, tok, f)
        XCTAssertEqual(b.count, 4, "block count")
        for i in 0..<4 {
            XCTAssertEqual(b[i], expected[i], accuracy: accuracy, "\(msg) element \(i)")
        }
    }
}