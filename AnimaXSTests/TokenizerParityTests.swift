import XCTest
import Tokenizers
@testable import AnimaXS

/// Tokenizer parity tests. The reference token IDs come from the Python `transformers`
/// oracle (`scripts/gen_tokenizer_ref.py` → `Fixtures/tokenizer_reference_ids.json`) and
/// are cross-validated against the case1/2/3 goldens. The Swift tokenizers load from the
/// bundled tokenizer.json files and must produce EXACTLY the reference IDs.
final class TokenizerParityTests: XCTestCase {

    private struct Fixture: Codable {
        let prompt: String
        let qwen_ids: [Int]
        let qwen_len: Int
        let t5_ids: [Int]
        let t5_len: Int
    }

    private func loadFixture() throws -> [String: Fixture] {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "tokenizer_reference_ids", withExtension: "json"))
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: Fixture].self, from: data)
    }

    /// Qwen tokenizer produces exactly the reference IDs (no start/end token).
    func testQwenParityAllCases() throws {
        let qwen = try TokenizerLoader.qwen()
        let fixtures = try loadFixture()
        for (caseName, fx) in fixtures.sorted(by: { $0.key < $1.key }) {
            let ids = qwen.encode(text: fx.prompt, addSpecialTokens: false)
            XCTAssertEqual(ids, fx.qwen_ids, "Qwen tokenizer mismatch for \(caseName)")
            XCTAssertEqual(ids.count, fx.qwen_len, "Qwen length mismatch for \(caseName)")
        }
    }

    /// T5 tokenizer produces the reference IDs (prompt tokens + trailing </s> id 1).
    func testT5ParityAllCases() throws {
        let t5 = try TokenizerLoader.t5()
        let fixtures = try loadFixture()
        for (caseName, fx) in fixtures.sorted(by: { $0.key < $1.key }) {
            // Reference T5 = encode(prompt, no specials) + [1] (trailing </s>).
            let base = t5.encode(text: fx.prompt, addSpecialTokens: false)
            let ids = base + [1]
            XCTAssertEqual(ids, fx.t5_ids, "T5 tokenizer mismatch for \(caseName)")
            XCTAssertEqual(ids.count, fx.t5_len, "T5 length mismatch for \(caseName)")
        }
    }

    /// T5 EOS token id must be 1 (used for the trailing </s>).
    func testT5EOSToken() throws {
        let t5 = try TokenizerLoader.t5()
        // The reference appends </s> (id 1) after the prompt tokens.
        XCTAssertEqual(t5.convertTokenToId("</s>"), 1)
    }

    /// The feature-off path is intentionally the exact historical tokenizer
    /// recipe: no combined caption, no sign allocation, all adapter weights 1.
    func testEmptyNegativePromptPreservesBaselineConditioning() throws {
        let fixtures = try loadFixture()
        for (caseName, fx) in fixtures.sorted(by: { $0.key < $1.key }) {
            let compiled = try GenerationEngine.compileConditioning(
                positive: fx.prompt, negative: "")
            XCTAssertEqual(compiled.qwenIDs, fx.qwen_ids, "Qwen baseline drift for \(caseName)")
            XCTAssertEqual(compiled.t5IDs, fx.t5_ids, "T5 baseline drift for \(caseName)")
            XCTAssertEqual(compiled.t5Weights, [Float](repeating: 1, count: fx.t5_ids.count))
            XCTAssertNil(compiled.negPiPValueSigns)
        }
    }

    func testPromptWeightParserMatchesAnimaReferenceRules() {
        XCTAssertEqual(
            GenerationEngine.parsePromptWeights("(cat)"),
            [PromptWeightSegment(text: "cat", weight: 1.1)])
        XCTAssertEqual(
            GenerationEngine.parsePromptWeights("(cat:1.5)"),
            [PromptWeightSegment(text: "cat", weight: 1.5)])

        let nested = GenerationEngine.parsePromptWeights("a (b (c):2) d")
        XCTAssertEqual(nested.count, 4)
        XCTAssertEqual(nested[0], PromptWeightSegment(text: "a ", weight: 1))
        XCTAssertEqual(nested[1], PromptWeightSegment(text: "b ", weight: 2))
        XCTAssertEqual(nested[2].text, "c")
        XCTAssertEqual(nested[2].weight, 2.2, accuracy: 0.0001)
        XCTAssertEqual(nested[3], PromptWeightSegment(text: " d", weight: 1))

        XCTAssertEqual(
            GenerationEngine.parsePromptWeights(#"literal \(paren\)"#),
            [PromptWeightSegment(text: "literal (paren)", weight: 1)])
    }

    /// Qwen receives syntax-stripped text, while T5 is tokenized per weighted
    /// segment. Magnitude stays in adapter weights and a negative inline weight
    /// is represented only by the separate V sign sidecar.
    func testInlineWeightedPromptSplitsMagnitudeFromNegPiPSign() throws {
        let prompt = "portrait, (silver hair:1.5), (watermark:-0.4)"
        let compiled = try GenerationEngine.compileConditioning(
            positive: prompt, negative: "")

        let qwen = try TokenizerLoader.qwen()
        XCTAssertEqual(
            compiled.qwenIDs,
            qwen.encode(text: "portrait, silver hair, watermark", addSpecialTokens: false))

        let t5 = try TokenizerLoader.t5()
        let pieces: [(String, Float, Float)] = [
            ("portrait, ", 1, 1),
            ("silver hair", 1.5, 1),
            (", ", 1, 1),
            ("watermark", 0.4, -1),
        ]
        var expectedIDs: [Int] = []
        var expectedWeights: [Float] = []
        var expectedSigns: [Float] = []
        for (text, weight, sign) in pieces {
            let ids = t5.encode(text: text, addSpecialTokens: false)
            expectedIDs.append(contentsOf: ids)
            expectedWeights.append(contentsOf: repeatElement(weight, count: ids.count))
            expectedSigns.append(contentsOf: repeatElement(sign, count: ids.count))
        }
        expectedIDs.append(1)
        expectedWeights.append(1)
        expectedSigns.append(1)

        XCTAssertEqual(compiled.t5IDs, expectedIDs)
        XCTAssertEqual(compiled.t5Weights.count, expectedWeights.count)
        for index in expectedWeights.indices {
            XCTAssertEqual(compiled.t5Weights[index], expectedWeights[index], accuracy: 0.0001)
        }

        let signs = try XCTUnwrap(compiled.negPiPValueSigns)
        XCTAssertEqual(signs.count, LLMAdapterMetal.maximumTokens)
        XCTAssertEqual(Array(signs.prefix(expectedSigns.count)), expectedSigns)
        XCTAssertTrue(signs.dropFirst(expectedSigns.count).allSatisfy { $0 == 1 })
    }

    func testEscapedParenthesesAreLiteralAndDoNotAllocateNegPiPMask() throws {
        let compiled = try GenerationEngine.compileConditioning(
            positive: #"portrait \(smiling\)"#, negative: "")
        let clean = "portrait (smiling)"
        let qwen = try TokenizerLoader.qwen()
        let t5 = try TokenizerLoader.t5()
        XCTAssertEqual(compiled.qwenIDs, qwen.encode(text: clean, addSpecialTokens: false))
        XCTAssertEqual(compiled.t5IDs, t5.encode(text: clean, addSpecialTokens: false) + [1])
        XCTAssertEqual(compiled.t5Weights, [Float](repeating: 1, count: compiled.t5IDs.count))
        XCTAssertNil(compiled.negPiPValueSigns)
    }

    /// Two familiar UI boxes compile to ONE Qwen caption + ONE T5 target stream.
    /// Only rows sourced from the negative box receive -1 in the separate V mask;
    /// the adapter token weights stay positive so cross-K is unchanged.
    func testNegativePromptCompilesToVOnlySigns() throws {
        let positive = "1girl, silver hair"
        let negative = "watermark, blurry"
        let compiled = try GenerationEngine.compileConditioning(
            positive: positive, negative: negative)

        let qwen = try TokenizerLoader.qwen()
        XCTAssertEqual(
            compiled.qwenIDs,
            qwen.encode(text: positive + ", " + negative, addSpecialTokens: false))

        let t5 = try TokenizerLoader.t5()
        let positiveIDs = t5.encode(text: positive + ", ", addSpecialTokens: false)
        let negativeIDs = t5.encode(text: negative, addSpecialTokens: false)
        XCTAssertEqual(compiled.t5IDs, positiveIDs + negativeIDs + [1])
        XCTAssertEqual(compiled.t5Weights, [Float](repeating: 1, count: compiled.t5IDs.count))

        let signs = try XCTUnwrap(compiled.negPiPValueSigns)
        XCTAssertEqual(signs.count, LLMAdapterMetal.maximumTokens)
        XCTAssertTrue(signs[..<positiveIDs.count].allSatisfy { $0 == 1 })
        let negativeRange = positiveIDs.count..<(positiveIDs.count + negativeIDs.count)
        XCTAssertTrue(signs[negativeRange].allSatisfy { $0 == -1 })
        XCTAssertEqual(signs[positiveIDs.count + negativeIDs.count], 1, "EOS must remain positive")
        XCTAssertTrue(signs[(positiveIDs.count + negativeIDs.count + 1)...].allSatisfy { $0 == 1 })
    }

    func testNegativeBoxNestedWeightMultipliesOuterNegPiPWeight() throws {
        let positive = "portrait"
        let negative = "watermark, (bad hands:1.25)"
        let compiled = try GenerationEngine.compileConditioning(
            positive: positive, negative: negative)
        let t5 = try TokenizerLoader.t5()
        let positiveIDs = t5.encode(text: positive + ", ", addSpecialTokens: false)
        let plainNegativeIDs = t5.encode(text: "watermark, ", addSpecialTokens: false)
        let weightedNegativeIDs = t5.encode(text: "bad hands", addSpecialTokens: false)

        let plainStart = positiveIDs.count
        let weightedStart = plainStart + plainNegativeIDs.count
        let weightedEnd = weightedStart + weightedNegativeIDs.count
        XCTAssertTrue(compiled.t5Weights[plainStart..<weightedStart].allSatisfy { $0 == 1 })
        for value in compiled.t5Weights[weightedStart..<weightedEnd] {
            XCTAssertEqual(value, 1.25, accuracy: 0.0001)
        }
        let signs = try XCTUnwrap(compiled.negPiPValueSigns)
        XCTAssertTrue(signs[plainStart..<weightedEnd].allSatisfy { $0 == -1 })
        XCTAssertEqual(signs[weightedEnd], 1, "EOS must remain positive")
    }

    /// The mask is lexical task state, not process-global mutable state. A
    /// detached diagnostics-style task cannot inherit it, and the value is gone
    /// immediately after the diffusion scope exits.
    func testNegPiPContextIsTaskLocal() async throws {
        var signs = [Float](repeating: 1, count: LLMAdapterMetal.maximumTokens)
        signs[7] = -1
        let scope = try XCTUnwrap(NegPiPGenerationContext.make(signs: signs))
        XCTAssertNil(NegPiPGenerationContext.snapshot())

        try await NegPiPGenerationContext.$active.withValue(scope) {
            let snapshot = try XCTUnwrap(NegPiPGenerationContext.snapshot())
            XCTAssertEqual(snapshot.lease, scope.lease)
            XCTAssertEqual(snapshot.signs[7], -1)
            let detachedSnapshot = await Task.detached {
                NegPiPGenerationContext.snapshot()
            }.value
            XCTAssertNil(detachedSnapshot, "detached diagnostics work must not inherit generation NegPiP")
        }
        XCTAssertNil(NegPiPGenerationContext.snapshot())
    }

    func testAllPositiveNegPiPMaskIsZeroWork() throws {
        let signs = [Float](repeating: 1, count: LLMAdapterMetal.maximumTokens)
        XCTAssertNil(try NegPiPGenerationContext.make(signs: signs))
        XCTAssertNil(NegPiPGenerationContext.snapshot())
    }
}
