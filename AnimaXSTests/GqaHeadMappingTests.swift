import XCTest
@testable import AnimaXS

/// Protects the Qwen GQA Q→KV head grouping against regression.
///
/// Pinned ComfyUI `comfy/ops.repeat_kv_for_gqa` expands K/V heads with
/// `repeat_interleave(n_rep, dim=head_dim)`, producing contiguous grouped heads
/// `[K0,K0,K1,K1,...,K7,K7]` for n_rep=2. Therefore Q head h reads KV head
/// `h / repeatFactor` where `repeatFactor = queryHeads / kvHeads`.
///
/// The previous (WRONG) mapping was `h % kvHeads` (round-robin), which scrambled
/// which V-vector each Q head attends to and produced full-28 cosine ≈ −0.04 vs the
/// golden. See docs/QWEN_ENCODER_DEBUG.md and scripts/qwen_comfy_oracle.py.
final class GqaHeadMappingTests: XCTestCase {

    /// General GQA rule (mirrors QwenEncoderCPU.runLayer). Must stay in sync.
    static func gqaKVHead(queryHead: Int, queryHeads: Int, kvHeads: Int) -> Int {
        precondition(queryHeads % kvHeads == 0, "query heads must be divisible by kv heads")
        let repeatFactor = queryHeads / kvHeads
        return queryHead / repeatFactor
    }

    func testQwenHeadMappingTable() {
        // Qwen3-0.6B: 16 Q heads, 8 KV heads → repeat factor 2.
        let expected: [Int] = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
        for h in 0..<16 {
            let got = Self.gqaKVHead(queryHead: h, queryHeads: 16, kvHeads: 8)
            XCTAssertEqual(got, expected[h],
                           "Q head \(h) must map to KV head \(expected[h]), got \(got)")
        }
        // Explicit anchors called out in the runbook.
        let anchors: [(Int, Int)] = [(0,0),(1,0),(2,1),(3,1),(14,7),(15,7)]
        for (q, kv) in anchors {
            XCTAssertEqual(Self.gqaKVHead(queryHead: q, queryHeads: 16, kvHeads: 8), kv)
        }
        // MUST NOT use modulo.
        for h in 0..<16 {
            XCTAssertNotEqual(Self.gqaKVHead(queryHead: h, queryHeads: 16, kvHeads: 8),
                              h % 8,
                              "modulo-8 GQA grouping is wrong for Q head \(h)")
        }
    }

    func testRepeatFactorIsGeneral() {
        XCTAssertEqual(Self.gqaKVHead(queryHead: 0, queryHeads: 32, kvHeads: 8), 0)
        XCTAssertEqual(Self.gqaKVHead(queryHead: 7, queryHeads: 32, kvHeads: 8), 1)
        XCTAssertEqual(Self.gqaKVHead(queryHead: 8, queryHeads: 32, kvHeads: 8), 2)
        XCTAssertEqual(Self.gqaKVHead(queryHead: 31, queryHeads: 32, kvHeads: 8), 7)
    }

    /// Synthetic 1-token attention: with seq=1, softmax = 1.0, so the output for each
    /// Q head equals the V-vector of the KV head it maps to. Each KV head k is filled
    /// with a distinct constant, so the output directly reveals which KV head each Q
    /// head received.
    func testSyntheticAttentionRevealsKVHead() {
        let qHeads = 16, kvHeads = 8, headDim = 4
        let repeatFactor = qHeads / kvHeads
        let seq = 1

        // V: each KV head k = constant (k+1)*10 + 0.5 along every dim.
        // K and Q chosen so scores are identical across KV heads (all-ones), so only
        // the GQA head assignment matters; with a single token the softmax is 1.0.
        var v = [[Float]](repeating: [Float](repeating: 0, count: kvHeads * headDim), count: seq)
        for k in 0..<kvHeads {
            for d in 0..<headDim {
                v[0][k * headDim + d] = Float(k + 1) * 10.0 + 0.5
            }
        }
        // K: flat [seq, kvHeads*headDim], all ones → identical scores.
        var kAll = [Float](repeating: 1.0, count: seq * kvHeads * headDim)

        for qh in 0..<qHeads {
            let kvHead = Self.gqaKVHead(queryHead: qh, queryHeads: qHeads, kvHeads: kvHeads)
            let qHead = [Float](repeating: 1.0, count: headDim)
            // k/v slices for the mapped KV head across all positions.
            var kSlice = [Float](repeating: 0, count: seq * headDim)
            var vSlice = [Float](repeating: 0, count: seq * headDim)
            let kvBase = kvHead * headDim
            for s in 0..<seq {
                for d in 0..<headDim {
                    kSlice[s * headDim + d] = kAll[s * (kvHeads * headDim) + kvBase + d]
                    vSlice[s * headDim + d] = v[s][kvBase + d]
                }
            }
            let mask = QwenNumerics.causalMask(position: 0, kvLen: seq)  // seq=1 → all 0
            let out = QwenNumerics.scaledDotAttention(
                q: qHead, k: kSlice, v: vSlice,
                qLen: 1, kvLen: seq, headDim: headDim,
                scale: 1.0 / Float(headDim).squareRoot(), mask: mask)
            let expectedVal = Float(kvHead + 1) * 10.0 + 0.5
            for d in 0..<headDim {
                XCTAssertEqual(out[d], expectedVal, accuracy: 1e-4,
                               "Q head \(qh) → KV head \(kvHead), expected V const \(expectedVal)")
            }
        }
    }
}
