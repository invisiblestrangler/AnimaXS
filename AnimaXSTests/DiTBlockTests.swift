import XCTest
@testable import AnimaXS

/// Pure-math CPU tests for the DiT transformer `Block` primitives (H005).
///
/// These lock the block-level operations that are easy to get wrong and are NOT covered by
/// the earlier H001–H004 tests. The full weight-dependent block is validated structurally
/// against the pinned-ComfyUI oracle (scripts/dit_block0_oracle.py). Lane B against the
/// original `block_00_out` is a separate W4-vs-source-checkpoint measurement (D035). The
/// most important test here is the SPLIT-HALF RoPE
/// pairing (D030): pair p = (p, p+half), NOT adjacent (2p, 2p+1).
final class DiTBlockTests: XCTestCase {

    // MARK: Split-half RoPE pairing (D030) — the H005-critical detail

    /// headDim=8, half=4. Input [1,2,3,4, 10,20,30,40] must pair (1,10),(2,20),(3,30),(4,40)
    /// NOT (1,2),(3,4),(10,20),(30,40). Uses 4 known 2×2 identity rotations.
    func testSplitHalfRopePairingIdentity() {
        let headDim = 8, half = 4, tokens = 1, heads = 1
        let x: [Float] = [1, 2, 3, 4, 10, 20, 30, 40]
        // identity rope: cos=1, sin=0 for all 4 freqs
        var rope = [Float](repeating: 0, count: tokens * half * 4)
        for p in 0..<half { rope[p * 4 + 0] = 1.0; rope[p * 4 + 3] = 1.0 }  // [1,-0,0,1]
        let out = DiTBlockCPU.applySplitHalfRoPE(x, rope: rope, tokens: tokens, heads: heads, headDim: headDim)
        // identity rotation → output == input
        for i in 0..<x.count { XCTAssertEqual(out[i], x[i], accuracy: 1e-6, "dim \\(i)") }
    }

    /// Non-trivial 2×2 rotations to prove the pairing, not just identity.
    /// headDim=4, half=2. Input [a,b, c,d] → pairs (a,c),(b,d).
    /// freq0 rotates (a,c) by 90° (c=0,s=1): out[0]=0*a-1*c=-c, out[2]=1*a+0*c=a
    /// freq1 rotates (b,d) by 180° (c=-1,s=0): out[1]=(-1)*b-0*d=-b, out[3]=0*b+(-1)*d=-d
    func testSplitHalfRopePairingRotation() {
        let headDim = 4, half = 2, tokens = 1, heads = 1
        let x: [Float] = [1, 2, 3, 4]   // a=1,b=2,c=3,d=4
        // rope: freq0 = [[0,-1],[1,0]] (90°), freq1 = [[-1,0],[0,-1]] (180°)
        var rope = [Float](repeating: 0, count: tokens * half * 4)
        rope[0] = 0; rope[1] = -1; rope[2] = 1; rope[3] = 0      // freq0: cos=0,sin=1
        rope[4] = -1; rope[5] = 0; rope[6] = 0; rope[7] = -1      // freq1: cos=-1,sin=0
        let out = DiTBlockCPU.applySplitHalfRoPE(x, rope: rope, tokens: tokens, heads: heads, headDim: headDim)
        // out[0] (a rotated with c by 90°): -s*b? no: c=cos=0,a=1,b=c=3 → out[0]=0*1-1*3=-3
        XCTAssertEqual(out[0], -3.0, accuracy: 1e-6, "freq0 first half")
        // out[2] = s*a + c*b = 1*1 + 0*3 = 1
        XCTAssertEqual(out[2], 1.0, accuracy: 1e-6, "freq0 second half")
        // out[1] (b with d by 180°): c=-1,a=b=2,b=d=4 → -1*2 - 0*4 = -2
        XCTAssertEqual(out[1], -2.0, accuracy: 1e-6, "freq1 first half")
        // out[3] = s*a + c*b = 0*2 + (-1)*4 = -4
        XCTAssertEqual(out[3], -4.0, accuracy: 1e-6, "freq1 second half")
    }

    /// If the pairing were adjacent (2p,2p+1), the values above would differ — this test
    /// would catch that exact bug. (Sanity: adjacent pairing would give out[0]=-2, out[1]=-3.)
    func testSplitHalfPairingIsNotAdjacent() {
        let headDim = 4, half = 2, tokens = 1, heads = 1
        let x: [Float] = [1, 2, 3, 4]
        var rope = [Float](repeating: 0, count: tokens * half * 4)
        rope[0] = 0; rope[1] = -1; rope[2] = 1; rope[3] = 0
        rope[4] = -1; rope[5] = 0; rope[6] = 0; rope[7] = -1
        let out = DiTBlockCPU.applySplitHalfRoPE(x, rope: rope, tokens: tokens, heads: heads, headDim: headDim)
        // Correct (p,p+2): out[0]=-3, out[1]=-2. Adjacent would give out[0]=-2, out[1]=-3.
        XCTAssertEqual(out[0], -3.0, accuracy: 1e-6)
        XCTAssertEqual(out[1], -2.0, accuracy: 1e-6)
    }

    // MARK: LayerNorm (mean-centering, D027)

    func testLayerNormMeanCentering() {
        let x: [Float] = [0.0, 1.0, 2.0, 3.0]   // mean 1.5, var 1.25
        let std = Float(sqrt(1.25))
        let out = DiTBlockCPU.layerNorm(x)
        let expected = (0..<4).map { (x[$0] - 1.5) / std }
        for i in 0..<4 { XCTAssertEqual(out[i], expected[i], accuracy: 1e-5, "dim \\(i)") }
    }

    func testLayerNormModulated() {
        let x: [Float] = [0.0, 1.0, 2.0, 3.0]
        let scale: [Float] = [0, 1, 0, -1]
        let shift: [Float] = [1, 2, 3, 4]
        let std = Float(sqrt(1.25))
        let out = DiTBlockCPU.layerNormModulated(x, scale: scale, shift: shift)
        for i in 0..<4 {
            let ln = (x[i] - 1.5) / std
            XCTAssertEqual(out[i], ln * (1 + scale[i]) + shift[i], accuracy: 1e-5, "dim \\(i)")
        }
    }

    // MARK: Per-head RMSNorm (shared [128] weight, no mean subtraction)

    func testRmsNormHeadsSharedWeight() {
        let headDim = 4, tokens = 2, heads = 2
        let weight: [Float] = [1, 2, 3, 4]   // shared across all heads/tokens
        // x[t,h,d] with simple values
        var x = [Float](repeating: 0, count: tokens * heads * headDim)
        for t in 0..<tokens { for h in 0..<heads { for d in 0..<headDim { x[(t * heads + h) * headDim + d] = Float((t + 1) * 10 + (h + 1) + d) } } }
        let out = DiTBlockCPU.rmsNormHeads(x, weight: weight, tokens: tokens, heads: heads, headDim: headDim)
        // independently recompute head (0,0)
        let t = 0, h = 0
        let base = (t * heads + h) * headDim
        var sum: Float = 0
        for d in 0..<headDim { let v = x[base + d]; sum += v * v }
        let inv = 1.0 / sqrt(sum / Float(headDim) + DiTBlockCPU.eps)
        for d in 0..<headDim {
            XCTAssertEqual(out[base + d], x[base + d] * inv * weight[d], accuracy: 1e-5, "head0 dim \\(d)")
        }
        // head (1,0) — same weight, different x
        let base2 = (0 * heads + 1) * headDim
        var sum2: Float = 0
        for d in 0..<headDim { let v = x[base2 + d]; sum2 += v * v }
        let inv2 = 1.0 / sqrt(sum2 / Float(headDim) + DiTBlockCPU.eps)
        for d in 0..<headDim {
            XCTAssertEqual(out[base2 + d], x[base2 + d] * inv2 * weight[d], accuracy: 1e-5, "head1 dim \\(d)")
        }
    }

    // MARK: Attention (scale + stable softmax + V accumulation)

    func testAttentionScaleAndSoftmax() {
        let headDim = 4, sk = 3
        // q: 2 tokens; k: 3 tokens; v: 3 tokens
        let q: [Float] = [1, 0, 0, 0,   0, 1, 0, 0]   // q0 aligns with k[0], q1 with k[1]
        let k: [Float] = [1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0]
        let v: [Float] = [10, 20, 30, 40,   50, 60, 70, 80,   90, 100, 110, 120]
        let out = DiTBlockCPU.attentionHead(q, k, v, headDim: headDim)
        let scale = 1.0 / sqrt(Float(headDim))
        // q0 · k = [1*scale, 0, 0]; softmax → exp(1*scale)/sum; q1 similarly with k[1]
        func softmaxWeights(_ logits: [Float]) -> [Float] {
            let m = logits.max()!
            let e = logits.map { exp($0 - m) }
            let s = e.reduce(0, +)
            return e.map { $0 / s }
        }
        let logits0 = [q[0]*k[0]+q[1]*k[1]+q[2]*k[2]+q[3]*k[3],
                       q[0]*k[4]+q[1]*k[5]+q[2]*k[6]+q[3]*k[7],
                       q[0]*k[8]+q[1]*k[9]+q[2]*k[10]+q[3]*k[11]].map { $0 * scale }
        let w0 = softmaxWeights(logits0)
        var exp0 = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim { for j in 0..<sk { exp0[d] += w0[j] * v[j * headDim + d] } }
        for d in 0..<headDim { XCTAssertEqual(out[d], exp0[d], accuracy: 1e-5, "q0 dim \\(d)") }

        let logits1 = [q[4]*k[0]+q[5]*k[1]+q[6]*k[2]+q[7]*k[3],
                       q[4]*k[4]+q[5]*k[5]+q[6]*k[6]+q[7]*k[7],
                       q[4]*k[8]+q[5]*k[9]+q[6]*k[10]+q[7]*k[11]].map { $0 * scale }
        let w1 = softmaxWeights(logits1)
        var exp1 = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim { for j in 0..<sk { exp1[d] += w1[j] * v[j * headDim + d] } }
        for d in 0..<headDim { XCTAssertEqual(out[headDim + d], exp1[d], accuracy: 1e-5, "q1 dim \\(d)") }
    }

    // MARK: Head concat order (head-major)

    func testConcatHeadsOrder() {
        let tokens = 2, heads = 3, headDim = 2
        // headsFlat[t][h][d] = t*100 + h*10 + d
        var x = [Float](repeating: 0, count: tokens * heads * headDim)
        for t in 0..<tokens { for h in 0..<heads { for d in 0..<headDim { x[(t * heads + h) * headDim + d] = Float(t * 100 + h * 10 + d) } } }
        let out = DiTBlockCPU.concatHeads(x, tokens: tokens, heads: heads, headDim: headDim)
        // token0: [0,1,10,11,20,21] (head0 dims 0..1, head1 dims 2..3, head2 dims 4..5)
        XCTAssertEqual(out[0], 0.0); XCTAssertEqual(out[1], 1.0)
        XCTAssertEqual(out[2], 10.0); XCTAssertEqual(out[3], 11.0)
        XCTAssertEqual(out[4], 20.0); XCTAssertEqual(out[5], 21.0)
        // token1 starts at 6
        XCTAssertEqual(out[6], 100.0); XCTAssertEqual(out[7], 101.0)
        XCTAssertEqual(out[8], 110.0); XCTAssertEqual(out[9], 111.0)
    }

    // MARK: GELU + gate residual

    func testExactGELU() {
        XCTAssertEqual(DiTBlockCPU.gelu(0.0), 0.0, accuracy: 1e-6)
        XCTAssertEqual(DiTBlockCPU.gelu(1.0), 0.84134474, accuracy: 1e-5)
        XCTAssertEqual(DiTBlockCPU.gelu(-1.0), -0.15865526, accuracy: 1e-5)
    }
}
