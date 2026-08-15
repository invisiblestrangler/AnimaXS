import CryptoKit
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
        let expectedPrefix: [Float] = [
            0.22878295, -0.78915966, -1.7395047, 0.3901934,
            -0.89549154, -1.4645813, 1.1782961, 0.78697103,
        ]
        for (actual, expected) in zip(a.prefix(expectedPrefix.count), expectedPrefix) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
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

    /// The DiT slot accepts the W8-v2 pack as an alternate variant; its pinned
    /// size/SHA must match the HuggingFace LFS metadata exactly (verified
    /// 2026-08-15 via the tree API). This pins the constants so a
    /// digit-transcription error (like the 2_232_973_560 vs 2_232_975_360 flip
    /// that rejected every import) is caught in CI without downloading the
    /// 2.23 GB pack.
    func testDiTSlotAcceptsPinnedW8V2Variant() throws {
        let dit = try XCTUnwrap(ModelManifest.entries.first { $0.component == .dit })
        let w8 = try XCTUnwrap(dit.alternates.first)
        XCTAssertEqual(dit.alternates.count, 1, "the DiT slot accepts exactly one alternate (W8-v2)")
        XCTAssertEqual(w8.size, 2_232_975_360,
                       "must match HuggingFace LFS size for the pinned revision")
        XCTAssertEqual(
            w8.sha256,
            "8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130")
        // The primary W4 variant must remain untouched.
        XCTAssertEqual(dit.size, 1_179_435_008)
        XCTAssertEqual(
            dit.sha256,
            "ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25")
    }

    /// A file matching no accepted DiT variant must be rejected by the
    /// variant matcher (import/verify gate).
    func testDiTRejectsNonVariantFile() throws {
        let dit = try XCTUnwrap(ModelManifest.entries.first { $0.component == .dit })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-w8variant-\(UUID().uuidString)")
        // 3 bytes of "abc": matches no production variant, must be rejected.
        try Data("abc".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try ModelManifest.matchedVariant(of: url, against: dit))
    }

    /// 5.6 — `ModelManifest.sha256(of:chunkBytes:)` must hash a file correctly
    /// when a deliberately tiny chunk forces many iterations, and agree with
    /// CryptoKit's one-shot digest. Uses a deterministic few-KiB fixture, not a
    /// production pack, so CI stays fast.
    func testModelManifestSHA256MultiChunkMatchesOneShot() throws {
        let size = 64 << 10 // 64 KiB fixture
        var data = Data(capacity: size)
        for i in 0..<size {
            data.append(UInt8((i &* 13 &+ 29) & 0xFF))
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-sha-multichunk-\(UUID().uuidString)")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let expected = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        // 7-byte chunks force ~9000 stream iterations.
        XCTAssertEqual(try ModelManifest.sha256(of: url, chunkBytes: 7), expected)
        // The default 1 MiB path must agree too.
        XCTAssertEqual(try ModelManifest.sha256(of: url), expected)
    }

    func testModelStoreSyntheticDownloadVerifyAndReadyState() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnimaXS-store-\(UUID().uuidString)", isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let models = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let downloaded = source.appendingPathComponent("tiny.animapk")
        try Data("abc".utf8).write(to: downloaded)
        let entry = ModelManifestEntry(
            filename: "tiny.animapk", size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            url: URL(string: "https://example.invalid/tiny.animapk")!, component: .dit)
        let store = try ModelStore(
            directory: models,
            downloader: { _ in downloaded },
            availableCapacity: { _ in Int64.max })
        let initialState = await store.state(for: .dit)
        XCTAssertEqual(initialState, .missing)
        let installed = try await store.download(entry)
        XCTAssertEqual(try Data(contentsOf: installed), Data("abc".utf8))
        let readyState = await store.state(for: .dit)
        XCTAssertEqual(readyState, .ready(installed))
        let repeated = try await store.download(entry)
        XCTAssertEqual(repeated, installed)
        XCTAssertTrue(try installed.resourceValues(
            forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup ?? false)
    }

    func testWanVAENumerics() throws {
        var latent = [Float](repeating: 0, count: 16)
        latent[0] = 0.5
        let decoded = try VAENumerics.decodeLatent(latent)
        XCTAssertEqual(decoded[0], ModelConstants.latentsStd[0] + ModelConstants.latentsMean[0],
                       accuracy: 1e-6)
        XCTAssertEqual(decoded[1], ModelConstants.latentsMean[1], accuracy: 1e-6)

        let normalized = try VAENumerics.channelRMSNorm(
            [3, 0, 4, 0], gamma: [1, 2], channels: 2, spatial: 2)
        XCTAssertEqual(normalized[0], Float(3.0 / 5.0 * sqrt(2.0)), accuracy: 1e-6)
        XCTAssertEqual(normalized[2], Float(4.0 / 5.0 * sqrt(2.0) * 2.0), accuracy: 1e-6)
        XCTAssertEqual(normalized[1], 0, accuracy: 0)
        XCTAssertEqual(normalized[3], 0, accuracy: 0)

        XCTAssertEqual(VAENumerics.silu([0])[0], 0, accuracy: 0)
        XCTAssertEqual(
            try VAENumerics.nearestExact2x([1, 2, 3, 4], channels: 1, height: 2, width: 2),
            [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4])
    }

    func testRGBConversionClampsAndUsesCHWOrder() throws {
        let rgba = try RGBConverter.rgba8(
            [-1, 1, 0, 2, -2, 0], width: 2, height: 1)
        XCTAssertEqual(rgba, [0, 128, 0, 255, 255, 255, 128, 255])
        XCTAssertEqual(try RGBConverter.image(
            [-1, 1, 0, 2, -2, 0], width: 2, height: 1).size.width, 2)
    }
}
