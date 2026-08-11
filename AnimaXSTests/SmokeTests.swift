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

    func testFlowVelocityConversionFeedsEulerContract() throws {
        let latent: [Float] = [1, -2, 0.5]
        let velocity: [Float] = [0.25, 1.5, -4]
        let denoised = try DiffusionSampler.cpuDenoised(
            latent: latent, velocity: velocity, sigma: 0.5)
        XCTAssertEqual(denoised, [0.875, -2.75, 2.5])
        let next = try EulerSampler.cpuStep(
            latent: latent, denoised: denoised, sigma: 0.5, nextSigma: 0.25)
        for index in next.indices {
            XCTAssertEqual(next[index], latent[index] - 0.25 * velocity[index], accuracy: 1e-7)
        }
        XCTAssertThrowsError(try DiffusionSampler.cpuDenoised(
            latent: [1], velocity: [], sigma: 1))
    }
}
