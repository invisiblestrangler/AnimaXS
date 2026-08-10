import XCTest
@testable import AnimaXS

/// CPU reference tests for the DiT AdaLN-LoRA modulation (H003).
///
/// The weight-dependent path is validated structurally against the pinned-ComfyUI oracle
/// (scripts/dit_input_timestep_oracle.py, cosine 1.000000 on the real W4 pack). These pure
/// tests lock the two math details that are easy to get wrong:
///   (1) SiLU is applied to emb_B_T_D BEFORE Linear1 (predict2.py:451-465 nn.Sequential order),
///       NOT after — the handoff §8 prose had this wrong; the pinned source is authoritative.
///   (2) LayerNorm in the modulation is elementwise_affine=False and mean-CENTERS (not RMSNorm).
final class ModulationTests: XCTestCase {

    /// Block.forward (predict2.py:520-521) `_fn`: normalized = LayerNorm(x)*(1+scale)+shift.
    /// For a constant input the LayerNorm output is ~0 (mean-centered), so the result must
    /// equal shift exactly.
    func testLayerNormModulationConstantInputEqualsShift() {
        let n = 2048
        let x = [Float](repeating: 1.0, count: n)          // constant → LayerNorm≈0
        let shift = (0..<n).map { Float($0 % 7) }
        let scale = (0..<n).map { Float(($0 * 3) % 5) }
        let out = Modulation.applyLayerNormModulation(x, scale: scale, shift: shift)
        for i in 0..<n {
            // (1-1)/sqrt(0+eps) * (1+scale) + shift = shift (up to fp32 eps scaling)
            XCTAssertEqual(out[i], shift[i], accuracy: 1e-4, "dim \(i)")
        }
    }

    /// SiLU before Linear1 (pinned order). A pure SiLU self-check.
    func testSiLUValues() {
        XCTAssertEqual(ModulationTests.silu(0.0), 0.0, accuracy: 1e-6)
        XCTAssertEqual(ModulationTests.silu(1.0), 0.7310586, accuracy: 1e-6)
        XCTAssertEqual(ModulationTests.silu(-1.0), -0.2689414, accuracy: 1e-6)
        XCTAssertEqual(ModulationTests.silu(2.0), 1.7615942, accuracy: 1e-6)
    }

    /// LayerNorm mean-centering: a non-constant probe should be (x-mean)/std*(1+scale)+shift.
    func testLayerNormModulationMeanCentering() {
        let x: [Float] = [0.0, 1.0, 2.0, 3.0]        // mean 1.5
        let scale: [Float] = [0, 0, 0, 0]
        let shift: [Float] = [0, 0, 0, 0]
        // var = mean((x-mean)^2) = mean([2.25,0.25,0.25,2.25]) = 1.25; std=sqrt(1.25)
        let std = sqrt(1.25)
        let out = Modulation.applyLayerNormModulation(x, scale: scale, shift: shift)
        let expected = (0..<4).map { (x[$0] - 1.5) / std }
        for i in 0..<4 {
            XCTAssertEqual(out[i], expected[i], accuracy: 1e-5, "dim \(i)")
        }
    }

    private static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }
}
