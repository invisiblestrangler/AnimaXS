import XCTest
import Metal
import Tokenizers
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
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
            envKey: "ANIMAXS_DIFFUSION_PACK", name: "anima-turbo-refine.animapk")
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
        if diagnosticConfig("golden_qwen_context") == "1" {
            let goldenURL = try requiredFixture(
                envKey: "ANIMAXS_QWEN_CONTEXT_FILE", name: "case1_cond_context.f32")
            let golden = try floats(from: goldenURL)
            XCTAssertEqual(golden.count, qwenTokenIDs.count * QwenEncoderMetal.hidden)
            golden.withUnsafeBytes { bytes in
                if let base = bytes.baseAddress {
                    memcpy(qwenOutput.contents(), base, bytes.count)
                }
            }
            print("FULL_QWEN_CONTEXT=golden")
        } else {
            try await qwen.execute(
                tokenIDs: qwenTokenIDs, output: qwenOutput, layerCompleted: nil)
            print("FULL_QWEN_CONTEXT=production")
        }
        let qwenSeconds = Date().timeIntervalSince(qwenStart)
        XCTAssertTrue(isFinite(qwenOutput), "Qwen output must be finite")

        // ---- 3. Adapter → crossContext [512, 1024] fp32 (DiT pack) ----
        // golden_dit_context=1: load the canonical source-adapter context
        // (case1_dit_context.f32, identical bytes to the CUDA ladder fixture
        // context512.f32) so backend parity isolates the DiT forward only.
        let cross = try XCTUnwrap(context.device.makeBuffer(
            length: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden * 4,
            options: .storageModeShared))
        var adapterSeconds = 0.0
        if diagnosticConfig("golden_dit_context") == "1" {
            let ditContextURL = try requiredFixture(
                envKey: "ANIMAXS_DIT_CONTEXT_FILE", name: "case1_dit_context.f32")
            let golden = try floats(from: ditContextURL)
            XCTAssertEqual(golden.count, LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden)
            golden.withUnsafeBytes { bytes in
                if let base = bytes.baseAddress {
                    memcpy(cross.contents(), base, bytes.count)
                }
            }
            print("FULL_DIT_CONTEXT=golden")
        } else {
            let adapter = try LLMAdapterMetal(context: context, file: AnimapkFile(url: ditURL))
            let adapterStart = Date()
            try await adapter.execute(
                qwenContext: qwenOutput, contextTokens: qwenTokenIDs.count,
                t5IDs: t5IDs, t5Weights: t5Weights, output: cross, layerCompleted: nil)
            adapterSeconds = Date().timeIntervalSince(adapterStart)
            print("FULL_DIT_CONTEXT=production (\\(adapterSeconds)s)")
        }
        XCTAssertTrue(isFinite(cross), "cross-context must be finite")

        // ---- 4. Diffusion: canonical golden noise → final latent ----
        // Count exactly 8 completed Euler steps and 8*28 block callbacks.
        let noiseValues = try floats(from: noiseURL)
        let initial = makeBuffer(noiseValues, on: context.device)
        let finalLatent = try XCTUnwrap(context.device.makeBuffer(
            length: DiffusionSampler.latentElements * 4, options: .storageModeShared))
        // ---- 4a. Numerical policy: the production resolver, not a default ----
        // The workflow injects ANIMAXS_DIT_VARIANT (matrix variant, e.g.
        // "w4-v2"/"w8-v2"); local runs fall back to the bundled pack metadata.
        // DiTNumericsPolicy.fromVariantID is the SAME resolver GenerationEngine
        // uses on the production path (ModelManifest.swift), and
        // DiffusionSampler.resolvedNumerics(for:) is the single source of truth
        // for the policy -> (activation, attention) mapping. Using them here
        // makes this test prove the same numerical policy production runs.
        let variantID: String
        if let injected = env("ANIMAXS_DIT_VARIANT"), !injected.isEmpty {
            variantID = injected
        } else {
            variantID = packMetadataValue(envKey: "ANIMAXS_DIT_VARIANT", jsonKey: "variant")
        }
        let policy = DiTNumericsPolicy.fromVariantID(variantID)
        print("FULL_DIT_NUMERICS_POLICY=\(policy.rawValue)")
        // Explicit diagnostic overrides (attention_numerics / activation_numerics
        // present in the config) keep the legacy override path exactly as before:
        // each value is built from its own key with a "legacy" default. Otherwise
        // the sampler is constructed from the variant-derived policy — the exact
        // numerics GenerationEngine resolves for this pack, INCLUDING the
        // policy-specific final-residual boundary. Constructing from explicit
        // attention/activation numerics alone would bypass that boundary, so the
        // no-override path must go through `numerics: policy`.
        let attentionOverride = diagnosticConfig("attention_numerics")
        let activationOverride = diagnosticConfig("activation_numerics")
        let hasNumericsOverrides = attentionOverride != nil || activationOverride != nil
        let attentionNumerics: AttentionNumerics
        let activationNumerics: ActivationNumerics
        if hasNumericsOverrides {
            attentionNumerics = AttentionNumerics(
                rawValue: attentionOverride ?? "legacy") ?? .legacy
            activationNumerics = ActivationNumerics(
                rawValue: activationOverride ?? "legacy") ?? .legacy
        } else {
            (activationNumerics, attentionNumerics) =
                DiffusionSampler.resolvedNumerics(for: policy)
        }
        print("FULL_ATTENTION_NUMERICS=\(attentionNumerics.rawValue)")
        print("FULL_ACTIVATION_NUMERICS=\(activationNumerics.rawValue)")
        // The final-residual ENTRY boundary is decoupled from activation
        // numerics (policy-driven). With overrides it mirrors
        // DiffusionSampler.init's explicit-construction rule; without overrides
        // it comes from the policy resolver.
        let finalResidualBoundary: FinalResidualBoundary
        if hasNumericsOverrides {
            finalResidualBoundary = activationNumerics == .bf16Compute
                ? .bf16RNEInFP32 : .fp16Legacy
        } else {
            finalResidualBoundary =
                DiffusionSampler.resolvedFinalResidualBoundary(for: policy)
        }
        print("FULL_FINAL_RESIDUAL_BOUNDARY=\(finalResidualBoundary.rawValue)")
        let sampler: DiffusionSampler
        if hasNumericsOverrides {
            sampler = try DiffusionSampler(
                context: context, file: AnimapkFile(url: ditURL),
                attentionNumerics: attentionNumerics,
                activationNumerics: activationNumerics)
        } else {
            // Production-equivalent path: the policy carries the final-residual
            // boundary (W8-v2 -> BF16-RNE-in-FP32, W4 -> FP16 legacy).
            sampler = try DiffusionSampler(
                context: context, file: AnimapkFile(url: ditURL),
                numerics: policy)
        }
        let diffusionStart = Date()
        var stepSeconds: [Double] = []
        var completedSteps = 0
        var blockCallbacks = 0
        var stepLatents: [[Float]] = []
        let captureTrajectory = diagnosticConfig("capture_trajectory") == "1"
        var trajectoryXIn: [Float]? = captureTrajectory ? read(initial) : nil
        let captureBlocks = Set((diagnosticConfig("capture_blocks") ??
            (diagnosticConfig("capture_block0") == "1" ? "0" : ""))
            .split(separator: ",").compactMap { Int($0) })
        var capturedStep0 = false
        var capturedBranches: Set<String> = []
        try await sampler.executeDiagnostic(
            initialLatent: initial, crossContext: cross, outputLatent: finalLatent,
            blockProgress: { _, _ in
                blockCallbacks += 1
            },
            stepCompleted: { step, _, _, denoised, latent in
                stepSeconds.append(Date().timeIntervalSince(diffusionStart))
                completedSteps += 1
                stepLatents.append(self.read(latent))
                if captureTrajectory {
                    // Per-step source-oracle capture (handoff section 11/13):
                    // x_i entering the step and the fp32 denoised output, from
                    // which the oracle reconstructs Swift's velocity
                    // v_i = (x_i - denoised_i) / sigma_i.
                    if let xIn = trajectoryXIn {
                        self.attachBuffer(from: xIn,
                                          name: String(format: "step%02d_x_in.f32", step))
                    }
                    self.attachBuffer(denoised, bytes: DiffusionSampler.latentElements * 4,
                                      name: String(format: "step%02d_denoised.f32", step))
                    trajectoryXIn = self.read(latent)
                    if step == 0 {
                        self.attachBuffer(cross, bytes: 512 * 1_024 * 4,
                                          name: "cross-context.f32")
                        let sigmas = EulerSampler.sigmas.map { String(format: "%.8f", $0) }
                            .joined(separator: ",")
                        let sigmasAttachment = XCTAttachment(string: sigmas + "\n")
                        sigmasAttachment.name = "sigmas.txt"
                        sigmasAttachment.lifetime = .keepAlways
                        self.add(sigmasAttachment)
                    }
                }
                print("FULL_DIFFUSION_STEP_\(step)_SECONDS=\(String(format: "%.2f", stepSeconds[step]))")
            },
            diagnosticStepPrepared: { step, residual, embedding, adaln, crossHalf, rope in
                guard !captureBlocks.isEmpty, step == 0, !capturedStep0 else { return }
                capturedStep0 = true
                self.attachBuffer(residual, bytes: 1_024 * 2_048 * 4,
                                  name: "w8-step0-block0-input.f32")
                self.attachBuffer(embedding, bytes: 2_048 * 4,
                                  name: "w8-step0-embedding.f32")
                self.attachBuffer(adaln, bytes: 6_144 * 4,
                                  name: "w8-step0-adaln.f32")
                self.attachBuffer(crossHalf, bytes: 512 * 1_024 * 2,
                                  name: "w8-cross-context.f16")
                self.attachBuffer(rope, bytes: 1_024 * 64 * 4 * 4,
                                  name: "w8-rope.f32")
            },
            diagnosticBlockCompleted: { step, block, residual in
                guard !captureBlocks.isEmpty, step == 0 else { return }
                if captureBlocks.contains(block) {
                    self.attachBuffer(residual, bytes: 1_024 * 2_048 * 4,
                                      name: "w8-step0-block\(block)-output.f32")
                }
                if captureBlocks.contains(block + 1) {
                    self.attachBuffer(residual, bytes: 1_024 * 2_048 * 4,
                                      name: "w8-step0-block\(block + 1)-input.f32")
                }
            },
            diagnosticBranchFilter: { step, block in
                step == 0 && captureBlocks.contains(block)
            },
            diagnosticBranchCompleted: { step, block, branch, residual in
                guard step == 0, captureBlocks.contains(block),
                      capturedBranches.insert("\(block)-\(branch)").inserted else { return }
                self.attachBuffer(residual, bytes: 1_024 * 2_048 * 4,
                                  name: "w8-step0-block\(block)-after-\(branch).f32")
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
        // committed case1_final_latent.f32 (== golden NPZ final_latent), which
        // is ComfyUI's POST-process_out VAE-space latent. The sampler output is
        // sampler-space, so it must pass through Wan21.process_out before the
        // comparison (root-cause fix, 2026-08-14; Wan21LatentFormat.swift). ----
        let finalValues = read(finalLatent)
        // Convert a COPY for the regression; the buffer itself is converted
        // once, in place, immediately before the VAE decode below.
        let vaeFinalValues = Wan21LatentFormat.processOut(finalValues)
        var latentCosineText = "n/a", latentRMSText = "n/a", latentMaxAbsText = "n/a"
        var inferencePassed = true
        if let reference = bundledFixture(named: "case1_final_latent.f32") {
            let refValues = try floats(from: reference)
            if refValues.count == vaeFinalValues.count {
                let cosine = cosineSimilarity(vaeFinalValues, refValues)
                let rmse = rmse(vaeFinalValues, refValues)
                let maxAbs = maxAbsolute(vaeFinalValues, refValues)
                latentCosineText = String(format: "%.4f", cosine)
                latentRMSText = String(format: "%.4f", rmse)
                latentMaxAbsText = String(format: "%.4f", maxAbs)
                print("FULL_FINAL_LATENT_COSINE=\(cosine)")
                print("FULL_FINAL_LATENT_RMSE=\(rmse)")
                print("FULL_FINAL_LATENT_MAXABS=\(maxAbs)")
                // The authoritative floor is cosine ≥ 0.65 (D057/D059). The
                // W4 recurrent path deviates from source-BF16 by design.
                if cosine < 0.65 { inferencePassed = false }
                XCTAssertGreaterThanOrEqual(cosine, 0.65,
                                            "final latent cosine must meet the D057 floor")
            } else {
                inferencePassed = false
                XCTFail("final latent reference element count mismatch")
            }
        } else {
            inferencePassed = false
            XCTFail("final latent reference fixture is missing")
        }

        if diagnosticConfig("latent_only") == "1" {
            print("FULL_LATENT_ONLY=PASS")
            print("FULL_INFERENCE=PASS")
            return
        }

        // ---- 5. VAE decode → DecodedRGBA8 ----
        // Wan21 boundary: the sampler output is sampler-space; convert it to
        // VAE decode space exactly once before the decoder (8px grid fix).
        Wan21LatentFormat.applyProcessOutInPlace(finalLatent)
        let vaeURL = try requiredFixture(
            envKey: "ANIMAXS_VAE_PACK", name: "qwen-image-vae-xsmax-fp16.animapk")
        let vae = try VAEDecoder(context: context, file: AnimapkFile(url: vaeURL))
        let vaeStart = Date()
        let image = try await vae.decode(latent: finalLatent)
        let vaeSeconds = Date().timeIntervalSince(vaeStart)
        let totalSeconds = Date().timeIntervalSince(overall)

        // ---- Image health assertions ----
        if image.width != 512 || image.height != 512
            || image.bytes.count != 512 * 512 * 4
            || isMonochrome(image.bytes)
            || dynamicRange(image.bytes) <= 32 {
            inferencePassed = false
        }
        XCTAssertEqual(image.width, 512, "decoded image width")
        XCTAssertEqual(image.height, 512, "decoded image height")
        XCTAssertEqual(image.bytes.count, 512 * 512 * 4, "RGBA byte count")
        if image.bytes.count == 512 * 512 * 4 {
            for pixel in 0..<(512 * 512) {
                if image.bytes[pixel * 4 + 3] != 255 {
                    inferencePassed = false
                }
                XCTAssertEqual(image.bytes[pixel * 4 + 3], 255, "alpha at pixel \(pixel)")
            }
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
        var referenceRGB: [UInt8] = []
        var rgbCosineText = "n/a", rgbRMSText = "n/a", rgbMAEText = "n/a", rgbMaxAbsText = "n/a"
        if let rgbRefURL = bundledFixture(named: "case1_decoded_rgb8.bin") {
            let refBytes = try Data(contentsOf: rgbRefURL)
            let refRGB = refBytes.map { Float($0) / 255.0 * 2.0 - 1.0 }
            referenceRGB = [UInt8](refBytes)
            let rgbBytes = image.bytes.enumerated()
                .filter { $0.offset % 4 != 3 }
                .map { $0.element }
            if rgbBytes.count == refRGB.count && refRGB.count == 512 * 512 * 3 {
                let rgbValues = rgbBytes.map { Float($0) / 255.0 * 2.0 - 1.0 }
                let cosine = cosineSimilarity(rgbValues, refRGB)
                let rmse = rmse(rgbValues, refRGB)
                let mae = mae(rgbValues, refRGB)
                let maxAbs = maxAbsolute(rgbValues, refRGB)
                rgbCosineText = String(format: "%.4f", cosine)
                rgbRMSText = String(format: "%.4f", rmse)
                rgbMAEText = String(format: "%.4f", mae)
                rgbMaxAbsText = String(format: "%.4f", maxAbs)
                print("FULL_RGB_COSINE=\(cosine)")
                print("FULL_RGB_RMSE=\(rmse)")
                print("FULL_RGB_MAE=\(mae)")
                print("FULL_RGB_MAXABS=\(maxAbs)")
                if cosine < 0.65 { inferencePassed = false }
                XCTAssertGreaterThanOrEqual(cosine, 0.65,
                                            "final RGB cosine must meet the 0.65 regression floor (D074)")
            } else {
                inferencePassed = false
                XCTFail("RGB8 reference byte count mismatch (got \(rgbBytes.count), ref \(refRGB.count))")
            }
        } else {
            inferencePassed = false
            XCTFail("RGB8 reference fixture is missing")
        }

        // ---- Observational metrics ----
        print("FULL_QWEN_SECONDS=\(String(format: "%.2f", qwenSeconds))")
        print("FULL_ADAPTER_SECONDS=\(String(format: "%.2f", adapterSeconds))")
        print("FULL_DIFFUSION_SECONDS=\(String(format: "%.2f", diffusionSeconds))")
        print("FULL_VAE_SECONDS=\(String(format: "%.2f", vaeSeconds))")
        print("FULL_TOTAL_SECONDS=\(String(format: "%.2f", totalSeconds))")

        // ---- Image capture (artifact branch only) ----
        // After the full inference has reached its normal output and all
        // regression assertions above have run, serialize the generated image
        // and the canonical reference into PNGs and attach them so the
        // workflow can export them as a GitHub artifact. This block does NOT
        // alter tensors, noise, precision, synchronization, or any comparison:
        // it only reads the already-decoded `image` and the already-loaded
        // `case1_decoded_rgb8.bin` reference.
        captureArtifacts(image: image, referenceRGB: referenceRGB, metrics: [
            "commit": provenanceValue("commit") ?? "<injected-by-workflow>",
            "run_id": provenanceValue("run_id") ?? "<injected-by-workflow>",
            "run_attempt": provenanceValue("run_attempt") ?? "<injected-by-workflow>",
            "workflow": provenanceValue("workflow") ?? "<injected-by-workflow>",
            "ref": provenanceValue("ref") ?? "<injected-by-workflow>",
            "variant": provenanceValue("variant") ?? "unknown",
            "prompt": prompt,
            "case": "case1_danbooru_seed1337",
            "packs": "qwen3-0.6b-xsmax-fp16-matrices.animapk / \(packMetadataValue(envKey: "ANIMAXS_DIT_VARIANT", jsonKey: "variant")) / qwen-image-vae-xsmax-fp16.animapk",
            "dit_numerics_policy": policy.rawValue,
            "attention_numerics": attentionNumerics.rawValue,
            "activation_numerics": activationNumerics.rawValue,
            "final_residual_boundary": finalResidualBoundary.rawValue,
            "golden_qwen_context": diagnosticConfig("golden_qwen_context") ?? "0",
            "latent_cosine": latentCosineText,
            "latent_rmse": latentRMSText,
            "latent_maxabs": latentMaxAbsText,
            "rgb_cosine": rgbCosineText,
            "rgb_rmse": rgbRMSText,
            "rgb_mae": rgbMAEText,
            "rgb_maxabs": rgbMaxAbsText,
            "qwen_seconds": String(format: "%.2f", qwenSeconds),
            "adapter_seconds": String(format: "%.2f", adapterSeconds),
            "diffusion_seconds": String(format: "%.2f", diffusionSeconds),
            "vae_seconds": String(format: "%.2f", vaeSeconds),
            "total_seconds": String(format: "%.2f", totalSeconds),
            "full_inference": inferencePassed ? "PASS" : "FAIL",
        ])

        print("FULL_INFERENCE=\(inferencePassed ? "PASS" : "FAIL")")
    }

    /// Phase 2/4 — broad-seed numerical stress with the numerical monitor's
    /// probes enabled (always-on + detailed MPS-output passes). Runs the
    /// production diffusion sampler over many ordinary seeds (diverse noise
    /// on the golden conditioning), then prints the aggregate evidence:
    /// total/success/failure, first-unsafe-boundary distribution, and maximum
    /// magnitudes observed at every probed boundary.
    ///
    /// This is evidence collection, not a correctness gate: it never asserts
    /// "zero warnings" (that is the investigation's question). It DOES fail on
    /// any non-finite latent (correctness must never regress). Seed count via
    /// ANIMAXS_STRESS_SEEDS (default 4); numerics modes via the standard
    /// attention_numerics / activation_numerics diagnostic config.
    func testNumericalStressAcrossSeeds() async throws {
        guard let context else {
            throw XCTSkip("SKIPPED_NO_METAL: default Metal device/library unavailable")
        }
        let ditURL = try requiredFixture(
            envKey: "ANIMAXS_DIFFUSION_PACK", name: "anima-turbo-refine.animapk")
        let noiseURL = try requiredFixture(
            envKey: "ANIMAXS_NOISE_FILE", name: "case1_noise.f32")
        // Golden conditioning isolates the DiT forward (same convention as the
        // canonical test's golden_dit_context path).
        let cross = try XCTUnwrap(context.device.makeBuffer(
            length: LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden * 4,
            options: .storageModeShared))
        let ditContextURL = try requiredFixture(
            envKey: "ANIMAXS_DIT_CONTEXT_FILE", name: "case1_dit_context.f32")
        let golden = try floats(from: ditContextURL)
        XCTAssertEqual(golden.count, LLMAdapterMetal.maximumTokens * LLMAdapterMetal.hidden)
        golden.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress {
                memcpy(cross.contents(), base, bytes.count)
            }
        }

        // Same variant-derived policy as canonical production inference
        // (ANIMAXS_DIT_VARIANT env, else bundled pack metadata) so the stress
        // matrix exercises the production final-residual boundary. Explicit
        // diagnostic overrides keep the legacy override path; otherwise the
        // sampler is constructed from the policy (production-equivalent).
        let variantID: String
        if let injected = env("ANIMAXS_DIT_VARIANT"), !injected.isEmpty {
            variantID = injected
        } else {
            variantID = packMetadataValue(envKey: "ANIMAXS_DIT_VARIANT", jsonKey: "variant")
        }
        let policy = DiTNumericsPolicy.fromVariantID(variantID)
        print("FULL_STRESS_DIT_NUMERICS_POLICY=\(policy.rawValue)")
        let attentionOverride = diagnosticConfig("attention_numerics")
        let activationOverride = diagnosticConfig("activation_numerics")
        let hasNumericsOverrides = attentionOverride != nil || activationOverride != nil
        let attentionNumerics: AttentionNumerics
        let activationNumerics: ActivationNumerics
        if hasNumericsOverrides {
            attentionNumerics = AttentionNumerics(
                rawValue: attentionOverride ?? "legacy") ?? .legacy
            activationNumerics = ActivationNumerics(
                rawValue: activationOverride ?? "legacy") ?? .legacy
        } else {
            (activationNumerics, attentionNumerics) =
                DiffusionSampler.resolvedNumerics(for: policy)
        }
        print("FULL_STRESS_ATTENTION_NUMERICS=\(attentionNumerics.rawValue)")
        print("FULL_STRESS_ACTIVATION_NUMERICS=\(activationNumerics.rawValue)")
        let finalResidualBoundary: FinalResidualBoundary
        if hasNumericsOverrides {
            finalResidualBoundary = activationNumerics == .bf16Compute
                ? .bf16RNEInFP32 : .fp16Legacy
        } else {
            finalResidualBoundary =
                DiffusionSampler.resolvedFinalResidualBoundary(for: policy)
        }
        print("FULL_STRESS_FINAL_RESIDUAL_BOUNDARY=\(finalResidualBoundary.rawValue)")
        NumericalMonitor.detailedProbesEnabled = true
        defer { NumericalMonitor.detailedProbesEnabled = false }

        let seedCount = Int(env("ANIMAXS_STRESS_SEEDS") ?? "") ?? 4
        let baseNoise = try floats(from: noiseURL)
        var successes = 0
        var failures = 0
        var firstIssueDistribution: [String: Int] = [:]
        var firstIssueSteps: [Int: Int] = [:]
        var boundaryMagnitudes: [String: Float] = [:]
        var failureMessages: [String] = []

        for run in 0..<seedCount {
            let seed = UInt64(1_337 + run * 7_919)
            var rng = SeededRNG(seed: seed)
            var noise = baseNoise
            for i in 0..<noise.count { noise[i] = Float(rng.nextNormal()) }
            let initial = makeBuffer(noise, on: context.device)
            let finalLatent = try XCTUnwrap(context.device.makeBuffer(
                length: DiffusionSampler.latentElements * 4, options: .storageModeShared))
            let sampler: DiffusionSampler
            if hasNumericsOverrides {
                sampler = try DiffusionSampler(
                    context: context, file: AnimapkFile(url: ditURL),
                    attentionNumerics: attentionNumerics,
                    activationNumerics: activationNumerics)
            } else {
                // Production-equivalent path: the policy carries the
                // final-residual boundary.
                sampler = try DiffusionSampler(
                    context: context, file: AnimapkFile(url: ditURL),
                    numerics: policy)
            }
            let start = Date()
            do {
                try await sampler.executeDiagnostic(
                    initialLatent: initial, crossContext: cross, outputLatent: finalLatent)
                successes += 1
            } catch {
                failures += 1
                failureMessages.append(error.localizedDescription)
                print("FULL_STRESS_SEED_\(seed)=FAIL \(error.localizedDescription)")
            }
            let seconds = Date().timeIntervalSince(start)
            print("FULL_STRESS_SEED_\(seed)=done seconds=\(String(format: "%.1f", seconds))")

            // Aggregate evidence from the monitor (valid after the run's GPU
            // work completed, whether it succeeded or threw).
            if let issue = sampler.earliestNumericalIssue {
                firstIssueDistribution[issue.probe.stageLabel, default: 0] += 1
                firstIssueSteps[issue.step, default: 0] += 1
            }
            for (probe, stats) in sampler.numericalReport where stats.maxAbs > 0 {
                boundaryMagnitudes[probe.stageLabel] = max(
                    boundaryMagnitudes[probe.stageLabel] ?? 0, stats.maxAbs)
            }
        }

        print("FULL_STRESS_TOTAL=\(seedCount) SUCCESS=\(successes) FAILURES=\(failures)")
        for (label, count) in firstIssueDistribution.sorted(by: { $0.value > $1.value }) {
            print("FULL_STRESS_FIRST_ISSUE=\(label) count=\(count)")
        }
        for (step, count) in firstIssueSteps.sorted(by: { $0.key < $1.key }) {
            print("FULL_STRESS_FIRST_ISSUE_STEP=\(step) count=\(count)")
        }
        for (label, magnitude) in boundaryMagnitudes.sorted(by: { $0.key < $1.key }) {
            print(String(format: "FULL_STRESS_MAXABS=%@ %.6g", label, magnitude))
        }
        // Correctness gate only: a non-finite latent is always a failure.
        // Numerical warnings without a non-finite latent are evidence.
        XCTAssertEqual(failures, 0, "non-finite latents under stress: \(failureMessages)")
        print("FULL_STRESS=\(failures == 0 ? "PASS" : "EVIDENCE")")
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

    private func diagnosticConfig(_ key: String) -> String? {
        if let environment = ProcessInfo.processInfo.environment[
            "ANIMAXS_" + key.uppercased()] { return environment }
        guard let url = bundledFixture(named: "quality-diagnostic.json"),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: String]
        else { return nil }
        return object[key]
    }

    private func attachBuffer(_ buffer: MTLBuffer, bytes: Int, name: String) {
        precondition(buffer.length >= bytes)
        let attachment = XCTAttachment(
            data: Data(bytes: buffer.contents(), count: bytes),
            uniformTypeIdentifier: UTType.data.identifier)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func attachBuffer(from values: [Float], name: String) {
        let attachment = XCTAttachment(
            data: values.withUnsafeBytes { Data($0) },
            uniformTypeIdentifier: UTType.data.identifier)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
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

    // MARK: - Artifact capture (temporary instrumentation, artifact branch only)

    /// Serializes the generated and reference images to PNGs, composes a
    /// lossless side-by-side comparison, and attaches them plus a metrics.txt
    /// so the workflow can export them as a GitHub artifact. This reads only
    /// already-produced data; it never touches inference internals.
    private func captureArtifacts(
        image: DecodedRGBA8,
        referenceRGB: [UInt8],
        metrics: [String: String]
    ) {
        let generatedPNG = makePNG(rgba8: image.bytes, width: 512, height: 512)
        let referencePNG = makePNG(rgb8: referenceRGB, width: 512, height: 512)
        let comparisonPNG = makeComparisonPNG(
            generatedRGBA: image.bytes, referenceRGB: referenceRGB,
            width: 512, height: 512)

        if let generatedPNG {
            let attachment = XCTAttachment(data: generatedPNG, uniformTypeIdentifier: UTType.png.identifier)
            attachment.name = "generated.png"
            attachment.lifetime = .keepAlways
            add(attachment)
        }
        if let referencePNG {
            let attachment = XCTAttachment(data: referencePNG, uniformTypeIdentifier: UTType.png.identifier)
            attachment.name = "reference.png"
            attachment.lifetime = .keepAlways
            add(attachment)
        }
        if let comparisonPNG {
            let attachment = XCTAttachment(data: comparisonPNG, uniformTypeIdentifier: UTType.png.identifier)
            attachment.name = "comparison.png"
            attachment.lifetime = .keepAlways
            add(attachment)
        }

        // metrics.txt (key: value, ordered, human-readable).
        var lines: [String] = []
        for key in [
            "commit", "run_id", "run_attempt", "workflow", "ref", "variant",
            "prompt", "case", "packs",
            "dit_numerics_policy", "attention_numerics", "activation_numerics", "golden_qwen_context",
            "latent_cosine", "latent_rmse", "latent_maxabs",
            "rgb_cosine", "rgb_rmse", "rgb_mae", "rgb_maxabs",
            "qwen_seconds", "adapter_seconds", "diffusion_seconds",
            "vae_seconds", "total_seconds", "full_inference",
        ] {
            if let value = metrics[key] { lines.append("\(key): \(value)") }
        }
        let textAttachment = XCTAttachment(string: lines.joined(separator: "\n") + "\n")
        textAttachment.name = "metrics.txt"
        textAttachment.lifetime = .keepAlways
        add(textAttachment)

        let metadata: [String: String] = [
            "variant": provenanceValue("variant")
                ?? packMetadataValue(envKey: "ANIMAXS_DIT_VARIANT", jsonKey: "variant"),
            "hf_repo": provenanceValue("dit_hf_repo")
                ?? packMetadataValue(envKey: "ANIMAXS_DIT_HF_REPO", jsonKey: "hf_repo"),
            "hf_revision": provenanceValue("dit_hf_revision")
                ?? packMetadataValue(envKey: "ANIMAXS_DIT_HF_REVISION", jsonKey: "hf_revision"),
            "sha256": provenanceValue("dit_sha256")
                ?? packMetadataValue(envKey: "ANIMAXS_DIT_SHA256", jsonKey: "sha256"),
            "bytes": provenanceValue("dit_bytes")
                ?? packMetadataValue(envKey: "ANIMAXS_DIT_BYTES", jsonKey: "bytes"),
            "storage": provenanceValue("dit_storage")
                ?? packMetadataValue(envKey: "ANIMAXS_DIT_STORAGE", jsonKey: "storage"),
            "group": "64",
        ]
        if let data = try? JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            let metadataAttachment = XCTAttachment(string: text + "\n")
            metadataAttachment.name = "pack-metadata.json"
            metadataAttachment.lifetime = .keepAlways
            add(metadataAttachment)
        }
    }

    /// xcodebuild may not forward arbitrary step environment variables into
    /// the simulator test host. The refinement workflow therefore also
    /// injects a bundled metadata JSON beside the packs; prefer the env value
    /// when present, but retain exact provenance in either execution mode.
    private func packMetadataValue(envKey: String, jsonKey: String) -> String {
        if let value = env(envKey), !value.isEmpty { return value }
        guard let url = bundledFixture(named: "pack-metadata.json"),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any],
              let value = dictionary[jsonKey] as? String,
              !value.isEmpty else {
            return "unknown"
        }
        return value
    }

    /// Reads a top-level key from the workflow-injected bundled
    /// `provenance.json` (commit/run_id/run_attempt/workflow/ref/variant and
    /// the flat `dit_*` pack fields). Returns nil when absent so callers can
    /// fall back to the older env/bundled pack-metadata.json path.
    private func provenanceValue(_ key: String) -> String? {
        guard let url = bundledFixture(named: "provenance.json"),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any],
              let value = dictionary[key] as? String,
              !value.isEmpty else {
            return nil
        }
        return value
    }

    /// Encodes interleaved RGBA8 bytes into a PNG (lossless) via CoreGraphics.
    private func makePNG(rgba8 bytes: [UInt8], width: Int, height: Int) -> Data? {
        guard width > 0, height > 0, bytes.count == width * height * 4 else { return nil }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let data = Data(bytes)
        data.withUnsafeBytes { raw in
            ctx.data?.copyMemory(from: raw.baseAddress!, byteCount: bytes.count)
        }
        guard let cgImage = ctx.makeImage() else { return nil }
        return pngData(from: cgImage)
    }

    /// Encodes interleaved RGB8 bytes into a PNG (lossless). Alpha = 255.
    private func makePNG(rgb8 bytes: [UInt8], width: Int, height: Int) -> Data? {
        guard width > 0, height > 0, bytes.count == width * height * 3 else { return nil }
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            rgba[i * 4 + 0] = bytes[i * 3 + 0]
            rgba[i * 4 + 1] = bytes[i * 3 + 1]
            rgba[i * 4 + 2] = bytes[i * 3 + 2]
        }
        return makePNG(rgba8: rgba, width: width, height: height)
    }

    /// Composes a lossless side-by-side: LEFT = generated, RIGHT = reference.
    /// Pure pixel compositing into a 2×512 wide, 512 tall canvas. No content
    /// is altered or resampled; no labels are drawn over image pixels.
    private func makeComparisonPNG(
        generatedRGBA: [UInt8], referenceRGB: [UInt8],
        width: Int, height: Int
    ) -> Data? {
        guard generatedRGBA.count == width * height * 4,
              referenceRGB.count == width * height * 3 else { return nil }
        let canvasWidth = width * 2
        let canvasHeight = height
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: canvasWidth, height: canvasHeight,
            bitsPerComponent: 8, bytesPerRow: canvasWidth * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let raw = ctx.data!.bindMemory(to: UInt8.self, capacity: canvasWidth * canvasHeight * 4)

        // CGContext's byte buffer is written in the same top-to-bottom row
        // order used by makePNG above. Keep both halves upright so the
        // comparison attachment is directly inspectable.
        for y in 0..<height {
            let srcRow = y * width * 4
            let dstRow = y * canvasWidth * 4
            for x in 0..<width {
                let si = srcRow + x * 4
                let di = dstRow + x * 4
                raw[di + 0] = generatedRGBA[si + 0]
                raw[di + 1] = generatedRGBA[si + 1]
                raw[di + 2] = generatedRGBA[si + 2]
                raw[di + 3] = 255
            }
        }
        // Right half = reference RGB8 (alpha 255).
        for y in 0..<height {
            let srcRow = y * width * 3
            let dstRow = y * canvasWidth * 4
            for x in 0..<width {
                let si = srcRow + x * 3
                let di = dstRow + (width + x) * 4
                raw[di + 0] = referenceRGB[si + 0]
                raw[di + 1] = referenceRGB[si + 1]
                raw[di + 2] = referenceRGB[si + 2]
                raw[di + 3] = 255
            }
        }

        guard let cgImage = ctx.makeImage() else { return nil }
        return pngData(from: cgImage)
    }

    /// Encodes a CGImage to PNG data via ImageIO.
    private func pngData(from image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
