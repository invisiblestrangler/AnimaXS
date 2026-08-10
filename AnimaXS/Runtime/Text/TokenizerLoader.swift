import Foundation
import Hub
import Tokenizers

/// Loads the Qwen + T5 tokenizers from the bundled `tokenizer.json` files (flat, unique
/// filenames to avoid Xcode resource-copy collisions). No HF network at runtime.
///
/// Parity (VALIDATED against goldens, see DECISIONS D014):
///   - Qwen (Qwen2Tokenizer → BPETokenizer): encode(prompt, no specials) — NO start/end token.
///     The exact Qwen2Tokenizer pre-tokenization regex is baked into qwen_tokenizer.json.
///   - T5 (T5Tokenizer → UnigramTokenizer): encode(prompt, no specials) + one trailing </s> (id 1).
enum TokenizerLoader {
    enum TokenizerError: Error, CustomStringConvertible {
        case missingResource(String)
        case parse(String)

        var description: String {
            switch self {
            case .missingResource(let n): return "tokenizer resource not found: \(n)"
            case .parse(let m): return "tokenizer parse error: \(m)"
            }
        }
    }

    /// Load the Qwen (BPETokenizer) from qwen_tokenizer.json.
    static func qwen() throws -> Tokenizer {
        try load(named: "qwen_tokenizer", tokenizerClass: "Qwen2Tokenizer")
    }

    /// Load the T5 (UnigramTokenizer) from t5_tokenizer.json.
    static func t5() throws -> Tokenizer {
        try load(named: "t5_tokenizer", tokenizerClass: "T5Tokenizer")
    }

    private static func load(named base: String, tokenizerClass: String) throws -> Tokenizer {
        guard let url = Bundle.main.url(forResource: base, withExtension: "json") else {
            throw TokenizerError.missingResource(base)
        }
        let data: Data
        do { data = try Data(contentsOf: url) } catch {
            throw TokenizerError.parse("cannot read \(url.path)")
        }
        let jsonObject: Any
        do { jsonObject = try JSONSerialization.jsonObject(with: data) } catch {
            throw TokenizerError.parse("invalid JSON in \(url.path): \(error)")
        }
        guard let nsdict = jsonObject as? NSDictionary else {
            throw TokenizerError.parse("tokenizer.json is not an object")
        }
        // Bridge NSDictionary → [NSString: Any] for Config(_ dictionary: [NSString: Any]).
        var bridged: [NSString: Any] = [:]
        nsdict.enumerateKeysAndObjects { key, value, _ in
            bridged[NSString(string: String(describing: key))] = value
        }
        let tokenizerData = Config(bridged)
        let tokenizerConfig: Config = [
            "tokenizer_class": tokenizerClass,
            "clean_up_tokenization_spaces": false,
        ]
        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}
