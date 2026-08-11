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

    func testSeededRNGIsDeterministicAndStatisticallySane() throws {
        var first = SeededRNG(seed: 1_337)
        var repeatSeed = SeededRNG(seed: 1_337)
        var other = SeededRNG(seed: 1_338)
        let a = try first.normal(count: 100_000)
        XCTAssertEqual(a, try repeatSeed.normal(count: 100_000))
        XCTAssertNotEqual(Array(a.prefix(16)), try other.normal(count: 16))
        let mean = a.reduce(0.0) { $0 + Double($1) } / Double(a.count)
        let variance = a.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / Double(a.count)
        XCTAssertLessThan(abs(mean), 0.02)
        XCTAssertEqual(sqrt(variance), 1.0, accuracy: 0.02)
        XCTAssertTrue(a.allSatisfy(\.isFinite))
        XCTAssertThrowsError(try first.normal(count: -1))
    }

    func testCheckpointRoundTripIsBitExactAndAtomic() throws {
        let count = 16 * 64 * 64
        let latent = (0..<count).map { Float($0 % 257) / 19 - 3 }
        let checkpoint = try GenerationCheckpoint(
            latent: latent, step: 4, prompt: "checkpoint test", seed: 1_337,
            width: 512, height: 512,
            modelHashes: ModelHashes(dit: "dit-hash", textEncoder: "te-hash", vae: "vae-hash"))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-checkpoint-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("generation.json")
        try checkpoint.writeAtomically(to: url)
        let loaded = try GenerationCheckpoint.load(from: url)
        XCTAssertEqual(loaded, checkpoint)
        let restored = try loaded.latentValues()
        XCTAssertEqual(restored.map(\.bitPattern), latent.map(\.bitPattern))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path),
                       ["generation.json"])
    }

    func testIncrementalModelManifestSHA256() throws {
        XCTAssertEqual(ModelManifest.entries.map(\.component), [.dit, .textEncoder, .vae])
        XCTAssertEqual(Set(ModelManifest.entries.map(\.filename)).count, 3)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-sha-\(UUID().uuidString)")
        try Data("abc".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(
            try ModelManifest.sha256(of: url, chunkBytes: 1),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let synthetic = ModelManifestEntry(
            filename: url.lastPathComponent, size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            url: url, component: .dit)
        XCTAssertNoThrow(try ModelManifest.verify(url, against: synthetic))
        XCTAssertThrowsError(try ModelManifest.sha256(of: url, chunkBytes: 0))
    }
}
