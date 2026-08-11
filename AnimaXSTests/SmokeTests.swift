import XCTest
@testable import AnimaXS

final class SmokeTests: XCTestCase {
    func testPlaceholder() {
        XCTAssertEqual(2 + 2, 4)
    }

    func testSigmaScheduleMatchesHandoff() {
        let expected: [Float] = [
            1.0, 0.9546938, 0.90035903, 0.8339981, 0.7511211,
            0.64468634, 0.50298506, 0.30500895, 0.0
        ]
        XCTAssertEqual(ModelConstants.sigma8Step, expected)
        XCTAssertEqual(ModelConstants.ditBlocks, 28)
    }

    func testEulerSamplerMatchesReferenceVector() throws {
        var latent = [Float](repeating: 1, count: 4)
        for step in 0..<8 {
            latent = try EulerSampler.cpuStep(
                latent: latent, denoised: [Float](repeating: 0, count: 4),
                sigma: EulerSampler.sigmas[step], nextSigma: EulerSampler.sigmas[step + 1])
            for value in latent {
                XCTAssertEqual(value, EulerSampler.sigmas[step + 1], accuracy: 2e-7)
            }
        }
        XCTAssertThrowsError(try EulerSampler.cpuStep(
            latent: [1], denoised: [0], sigma: 0, nextSigma: 0))
    }
}
