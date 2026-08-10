import XCTest
@testable import AnimaXS

/// CPU reference tests for the DiT timestep embedder (H002).
///
/// The weight-dependent path (Linear1→SiLU→Linear2 + RMSNorm) is validated structurally
/// against the pinned-ComfyUI oracle (scripts/dit_input_timestep_oracle.py, cosine
/// 1.000000 on the real W4 pack). These pure tests lock the sinusoidal math against
/// torch reference values (predict2.py Timesteps.forward) so CI runs with no packs.
final class TimestepEmbedderTests: XCTestCase {

    /// predict2.py:219-238 — sinusoidal embedding at sigma=1.0, verified against torch:
    ///   exponent = -log(10000)*arange(1024)/1024; emb = sigma*exp(exponent)
    ///   out = cat([cos(emb), sin(emb)], dim=-1)
    /// Reference (torch, fp32): cos first 8 = [0.54030234, 0.5478152, 0.5552175, 0.56251031,
    ///   0.56969506, 0.57677263, 0.58374441, 0.59061158]; sin first 8 = [0.84147096, 0.83659935,
    ///   0.83170521, 0.82679027, 0.8218562, 0.81690472, 0.81193745, 0.80695599].
    func testSinusoidalMatchesTorchReference() {
        let s = TimestepEmbedder.sinusoidal(sigma: 1.0)
        XCTAssertEqual(s.count, 2048)

        let cosRef: [Float] = [0.54030234, 0.5478152, 0.5552175, 0.56251031,
                               0.56969506, 0.57677263, 0.58374441, 0.59061158]
        for i in 0..<8 {
            XCTAssertEqual(s[i], cosRef[i], accuracy: 1e-6, "cos dim \(i)")
        }
        let sinRef: [Float] = [0.84147096, 0.83659935, 0.83170521, 0.82679027,
                               0.8218562, 0.81690472, 0.81193745, 0.80695599]
        for i in 0..<8 {
            XCTAssertEqual(s[1024 + i], sinRef[i], accuracy: 1e-6, "sin dim \(i)")
        }
        // angle[1] = 10000^(-1/1024) = 0.99104583
        XCTAssertEqual(Float(cos(0.99104583)), s[1], accuracy: 1e-6)
    }

    /// SiLU must match torch: silu(x) = x * sigmoid(x).
    func testSiLU() {
        XCTAssertEqual(TimestepEmbedderTests.silu(0.0), 0.0, accuracy: 1e-6)
        XCTAssertEqual(TimestepEmbedderTests.silu(1.0), 0.7310586, accuracy: 1e-6)
        XCTAssertEqual(TimestepEmbedderTests.silu(-1.0), -0.2689414, accuracy: 1e-6)
        XCTAssertEqual(TimestepEmbedderTests.silu(2.0), 1.7615942, accuracy: 1e-6)
    }

    private static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }
}
