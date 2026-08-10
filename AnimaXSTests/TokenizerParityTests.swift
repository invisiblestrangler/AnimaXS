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
}
