import XCTest
import Metal
import Tokenizers
@testable import AnimaXS

/// L001: real end-to-end prompt → RGBA8 inference on the production path.
///
/// Chain (no mocks, no CPU substitutes):
///   canonical prompt → Qwen tokenizer → T5 tokenizer
///   → QwenEncoderMetal.execute(tokenIDs:output:)
///   → LLMAdapterMetal.execute(qwenContext:contextTokens:t5IDs:t5Weights:output:)
///   → DiffusionSampler.execute(initialLatent:crossContext:outputLatent:)
///   → VAEDecoder.decode(latent:) → DecodedRGBA8
///
/// Production topology (K002 §5.1 / guide §14): exactly THREE packs — Qwen
/// text encoder, DiT (serves BOTH the adapter and the sampler), and VAE.
/// There is NO separate adapter pack in canonical production inference.
/// (A dedicated `g003-adapter-w4.animapk` may still appear in isolated G003
/// adapter-parity tests; it is not part of full inference.)
///
/// Fixture-gated: in ordinary CI the model packs are absent, so this test
/// `XCTSkip`s (the `full-inference.yml` workflow injects the real packs via
/// ANIMAXS_*_PACK and requires a PASS). When packs are present it asserts:
///   - tokenization invariants;
///   - 8 completed Euler steps × 28 block callbacks;
///   - final latent finite + regression vs the D057 floor (cosine ≥ 0.65 vs
///     the canonical source-BF16 final latent);
///   - decoded image health (512×512, RGBA byte count, alpha 255, non-degenerate);
///   - a measured final-RGB tolerance once established (see DECISIONS).
final class FullInferenceTests: XCTestCase {
    private var context: MetalContext?

    override func setUpWithError() throws {
        context = MetalContext()
    }

    override func tearDownWithError() throws {
        context = nil
    }

    func testCanonicalProductionInference() async throws {
        guard let context else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }

        // ---- Fixture resolution (env-first, bundled fallback). Exactly three
        // production packs; the adapter uses the SAME DiT pack as the sampler.
        let qwenURL = try requiredFixture(
            envKey: "ANIMAXS_QWEN_PACK", name: "qwen3-0.6b-xsmax-w8.animapk")
        let ditURL = try requiredFixture(
            envKey: "ANIMAXS_DIFFUSION_PACK", name: "anima-turbo-v1.0-xsmax-w4.animapk")
        let vaeURL = try requiredFixture(
            envKey: "ANIMAXS_VAE_PACK", name: "qwen-image-vae-xsmax-fp16.animapk")
        let noiseURL = try requiredFixture(
            envKey: "ANIMAXS_NOISE_FILE", name: "case1_noise.f32")

        let overall = Date()
        // The canonical golden (case1_danbooru_seed1337.npz, seed 1337) was
        // generated with this EXACT prompt (see fixtures.json / extract
        // script). The production chain must regenerate the same conditioning,
        // or the final latent/RGB regression against case1_final_latent.f32 and
        // case1_decoded_rgb8.bin is apples-to-oranges and cannot pass. (A
        // shorter/different prompt produced latent cosine 0.488 / RGB 0.438 in
        // the first full run — see DECISIONS D073.)
        let prompt = "masterpiece, best quality, score_7, safe, 1girl, long brown hair, blue eyes, school uniform, cherry blossom, outdoors, looking at viewer, smile, depth of field, highres, absurdres"

        // ---- 1. Tokenization (production TokenizerLoader semantics) ----
        // Qwen: encode(prompt, no specials); T5: + [1] EOS; t5Weights all 1.0.
        let qwenTokenizer = try TokenizerLoader.qwen()
        let qwenTokenIDs = qwenTokenizer.encode(text: prompt, addSpecialTokens: false)
        XCTAssertFalse(qwenTokenIDs.isEmpty, "Qwen tokenizer produced no tokens")
        XCTAssertLessThanOrEqual(qwenTokenIDs.count, QwenEncoderMetal.maximumTokens,
                                 "Qwen token count within supported maximum")

        let t5Tokenizer = try TokenizerLoader.t5()
        let t5IDs = t5Tokenizer.encode(text: prompt, addSpecialTokens: false) + [1]
        let t5Weights = [Float](repeating: 1.0, count: t5IDs.count)
        XCTAssertEqual(t5Weights.count, t5IDs.count, "T5 weights/IDs invariant")
        XCTAssertLessThanOrEqual(t5IDs.count, LLMAdapterMetal.maximumTokens,
                                 "T5 token count within supported maximum")

        // ---- 2. Qwen text encoding ----
        let qwen = try QwenEncoderMetal(context: context, file: AnimapkFile(url: qwenURL))
        let qwenOutput = try XCTUnwrap(context.device.makeBuffer(
            length: qwenTokenIDs.count * QwenEncoderMetal.hidden * 4, options: .storageModeShared))
        let qwenStart = Date()
        try await qwen.execute(tokenIDs: qwenTokenIDs, output: qwenOutput, layerCompleted: nil)
        let qwenSeconds = Date().timeIntervalSince(qwenStart)
        XCTAssertTrue(isFinite(qwenOutput), "Qwen output must be finite")

        // ---- 3. Adapter → crossContext [512, 1024] fp32 (DiT pack) ----
        let adapter = try LLMAdapterMetal(context: context, file: AnimapkFile(url: ditURL))
        let cross = try XCTUnwrap(context.device.makeBuffer(
            length: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden * 4,
            options: .storageModeShared))
        let adapterStart = Date()
        try await adapter.execute(
            qwenContext: qwenOutput, contextTokens: qwenTokenIDs.count,
            t5IDs: t5IDs, t5Weights: t5Weights, output: cross, layerCompleted: nil)
        let adapterSeconds = Date().timeIntervalSince(adapterStart)
        XCTAssertTrue(isFinite(cross), "cross-context must be finite")

        // ---- 4. Diffusion: canonical golden noise → final latent ----
        // Count exactly 8 completed Euler steps and 8*28 block callbacks.
        let noiseValues = try floats(from: noiseURL)
        let initial = makeBuffer(noiseValues, on: context.device)
        let finalLatent = try XCTUnwrap(context.device.makeBuffer(
            length: DiffusionSampler.latentElements * 4, options: .storageModeShared))
        let sampler = try DiffusionSampler(context: context, file: AnimapkFile(url: ditURL))
        let diffusionStart = Date()
        var stepSeconds: [Double] = []
        var completedSteps = 0
        var blockCallbacks = 0
        var stepLatents: [[Float]] = []
        try await sampler.execute(
            initialLatent: initial, crossContext: cross, outputLatent: finalLatent,
            blockProgress: { _, _ in
                blockCallbacks += 1
            },
            stepCompleted: { step, _, _, _, latent in
                stepSeconds.append(Date().timeIntervalSince(diffusionStart))
                completedSteps += 1
                stepLatents.append(self.read(latent))
                print("FULL_DIFFUSION_STEP_\(step)_SECONDS=\(String(format: "%.2f", stepSeconds[step]))")
            })
        let diffusionSeconds = Date().timeIntervalSince(diffusionStart)
        XCTAssertEqual(completedSteps, 8, "exactly 8 Euler steps completed")
        XCTAssertEqual(blockCallbacks, 8 * ModelConstants.ditBlocks,
                       "exactly 8*28 block callbacks")
        XCTAssertTrue(isFinite(finalLatent), "final latent must be finite")
        for (index, latent) in stepLatents.enumerated() {
            XCTAssertTrue(latent.allSatisfy(\.isFinite),
                          "completed-step latent \(index) must be finite")
        }

        // ---- Final latent regression (D057/D059 floor: cosine ≥ 0.65 vs
        // source-BF16 canonical final latent). The canonical reference is the
        // committed case1_final_latent.f32 (== golden NPZ final_latent). ----
        let finalValues = read(finalLatent)
        if let reference = bundledFixture(named: "case1_final_latent.f32") {
            let refValues = try floats(from: reference)
            if refValues.count == finalValues.count {
                let cosine = cosineSimilarity(finalValues, refValues)
                let rmse = rmse(finalValues, refValues)
                let maxAbs = maxAbsolute(finalValues, refValues)
                print("FULL_FINAL_LATENT_COSINE=\(cosine)")
                print("FULL_FINAL_LATENT_RMSE=\(rmse)")
                print("FULL_FINAL_LATENT_MAXABS=\(maxAbs)")
                // The authoritative floor is cosine ≥ 0.65 (D057/D059). The
                // W4 recurrent path deviates from source-BF16 by design.
                XCTAssertGreaterThanOrEqual(cosine, 0.65,
                                            "final latent cosine must meet the D057 floor")
            }
        }

        // ---- 5. VAE decode → DecodedRGBA8 ----
        let vae = try VAEDecoder(context: context, file: AnimapkFile(url: vaeURL))
        let vaeStart = Date()
        let image = try await vae.decode(latent: finalLatent)
        let vaeSeconds = Date().timeIntervalSince(vaeStart)
        let totalSeconds = Date().timeIntervalSince(overall)

        // ---- Image health assertions ----
        XCTAssertEqual(image.width, 512, "decoded image width")
        XCTAssertEqual(image.height, 512, "decoded image height")
        XCTAssertEqual(image.bytes.count, 512 * 512 * 4, "RGBA byte count")
        for pixel in 0..<(512 * 512) {
            XCTAssertEqual(image.bytes[pixel * 4 + 3], 255, "alpha at pixel \(pixel)")
        }
        XCTAssertFalse(isMonochrome(image.bytes), "image must not be a single constant color")
        XCTAssertGreaterThan(dynamicRange(image.bytes), 32,
                             "image must have meaningful dynamic range")

        // ---- Final RGB regression ----
        // Canonical image-space reference: `case1_decoded_rgb8.bin` (512×512×3
        // UInt8, per-pixel RGB interleave) derived from the authoritative
        // case1_danbooru_seed1337.npz `decoded_rgb` via the EXACT production
        // display transform (AnimaKernels.metal `vae_position_to_rgba8`):
        //   byte = (uchar)(clamp((v+1)*0.5,0,1)*255 + 0.5)
        // Production RGBA8 is decoded to RGB8 (drop alpha) and BOTH sides are
        // mapped back to [-1,1] via /255*2-1, so the comparison is an
        // apples-to-apples UInt8↔UInt8 regression of the final image.
        // Metrics: FULL_RGB_COSINE / FULL_RGB_RMSE / FULL_RGB_MAE / FULL_RGB_MAXABS.
        // Regression gate: cosine ≥ 0.65. Calibrated (DECISIONS D074) from the
        // first correct-prompt full L003 run: latent cosine 0.6946 (≥ D057
        // floor 0.65) and RGB cosine 0.7035 against the canonical source-BF16
        // reference. The W4/W8-quantized production chain is expected to
        // deviate from the BF16 source (cumulative quantization), so a tight
        // 0.9 gate would falsely reject a correct pipeline; 0.65 keeps a small
        // justified margin below the measured 0.7035 while still catching any
        // gross regression (broken pipeline / wrong conditioning drops far below).
        if let rgbRefURL = bundledFixture(named: "case1_decoded_rgb8.bin") {
            let refBytes = try Data(contentsOf: rgbRefURL)
            let refRGB = refBytes.map { Float($0) / 255.0 * 2.0 - 1.0 }
            let rgbBytes = image.bytes.enumerated()
                .filter { $0.offset % 4 != 3 }
                .map { $0.element }
            if rgbBytes.count == refRGB.count && refRGB.count == 512 * 512 * 3 {
                let rgbValues = rgbBytes.map { Float($0) / 255.0 * 2.0 - 1.0 }
                let cosine = cosineSimilarity(rgbValues, refRGB)
                let rmse = rmse(rgbValues, refRGB)
                let mae = mae(rgbValues, refRGB)
                let maxAbs = maxAbsolute(rgbValues, refRGB)
                print("FULL_RGB_COSINE=\(cosine)")
                print("FULL_RGB_RMSE=\(rmse)")
                print("FULL_RGB_MAE=\(mae)")
                print("FULL_RGB_MAXABS=\(maxAbs)")
                XCTAssertGreaterThanOrEqual(cosine, 0.65,
                                            "final RGB cosine must meet the 0.65 regression floor (D074)")
            } else {
                XCTFail("RGB8 reference byte count mismatch (got \(rgbBytes.count), ref \(refRGB.count))")
            }
        }

        // ---- Observational metrics ----
        print("FULL_QWEN_SECONDS=\(String(format: "%.2f", qwenSeconds))")
        print("FULL_ADAPTER_SECONDS=\(String(format: "%.2f", adapterSeconds))")
        print("FULL_DIFFUSION_SECONDS=\(String(format: "%.2f", diffusionSeconds))")
        print("FULL_VAE_SECONDS=\(String(format: "%.2f", vaeSeconds))")
        print("FULL_TOTAL_SECONDS=\(String(format: "%.2f", totalSeconds))")
        print("FULL_INFERENCE=PASS")
    }

    // MARK: - Required-mode fixture resolution

    /// Returns the env-injected pack path, or the bundled fixture if present.
    /// When `ANIMAXS_REQUIRE_FULL_INFERENCE=1` (dedicated workflow after packs
    /// were downloaded), a missing fixture is a FAILURE, not a skip.
    private func requiredFixture(envKey: String, name: String) throws -> URL {
        if let path = env(envKey) { return URL(fileURLWithPath: path) }
        if let bundled = bundledFixture(named: name) { return bundled }
        if env("ANIMAXS_REQUIRE_FULL_INFERENCE") == "1" {
            // Hard failure: a missing required fixture in required mode must
            // not pass silently. XCTest treats a thrown non-skip error as a
            // failed test (and the workflow also greps for skip markers).
            throw NSError(
                domain: "AnimaXS.FullInference",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "required full-inference fixture \(name) is missing (env \(envKey))"])
        }
        throw XCTSkip("full inference pack \(name) not available (A005-gated fixture)")
    }

    // MARK: - Helpers

    private func env(_ key: String) -> String? {
        ProcessInfo.processInfo.environment[key]
    }

    private func bundledFixture(named name: String) -> URL? {
        guard let root = Bundle(for: Self.self).resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: nil) else { return nil }
        for case let url as URL in enumerator where url.lastPathComponent == name { return url }
        return nil
    }

    private func floats(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func makeBuffer(_ values: [Float], on device: MTLDevice) -> MTLBuffer {
        let buffer = device.makeBuffer(
            length: values.count * 4, options: .storageModeShared)!
        buffer.contents().copyMemory(from: values, byteCount: values.count * 4)
        return buffer
    }

    private func read(_ buffer: MTLBuffer) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / 4)
        return Array(UnsafeBufferPointer(start: pointer, count: buffer.length / 4))
    }

    private func isFinite(_ buffer: MTLBuffer) -> Bool {
        read(buffer).allSatisfy(\.isFinite)
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<n {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y; na += x * x; nb += y * y
        }
        return dot / (sqrt(na) * sqrt(nb))
    }

    private func rmse(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var sum = 0.0
        for i in 0..<n { let d = Double(a[i] - b[i]); sum += d * d }
        return sqrt(sum / Double(n))
    }

    private func maxAbsolute(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var m = 0.0
        for i in 0..<n { m = max(m, abs(Double(a[i] - b[i]))) }
        return m
    }

    private func mae(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var sum = 0.0
        for i in 0..<n { sum += abs(Double(a[i] - b[i])) }
        return sum / Double(n)
    }

    private func isMonochrome(_ bytes: [UInt8]) -> Bool {
        guard let first = bytes.first else { return true }
        return bytes.allSatisfy { $0 == first }
    }

    private func dynamicRange(_ bytes: [UInt8]) -> Int {
        let rgb = bytes.enumerated().filter { $0.offset % 4 != 3 }.map { Int($0.element) }
        return (rgb.max() ?? 0) - (rgb.min() ?? 0)
    }
}
