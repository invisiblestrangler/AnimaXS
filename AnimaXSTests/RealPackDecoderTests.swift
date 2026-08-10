import XCTest
@testable import AnimaXS

/// Real-pack verification, gated on the `ANIMAXS_PACKS_DIR` environment variable pointing
/// at a directory containing the three `.animapk` packs. CI does not set this (no 2 GB
/// download on push); the owner sets it to run the definitive parser/decode check.
/// When unset, these tests are SKIPPED (reported via XCTSkip), not silently passed.
final class RealPackDecoderTests: XCTestCase {

    private static let packEnv = ProcessInfo.processInfo.environment["ANIMAXS_PACKS_DIR"]

    private func requirePackEnv() throws {
        guard let dir = Self.packEnv else {
            throw XCTSkip("ANIMAXS_PACKS_DIR not set — real-pack verification skipped (CI never downloads packs).")
        }
        let fm = FileManager.default
        for name in ["anima-turbo-v1.0-xsmax-w4.animapk",
                     "qwen3-0.6b-xsmax-w8.animapk",
                     "qwen-image-vae-xsmax-fp16.animapk"] {
            guard fm.fileExists(atPath: dir + "/" + name) else {
                throw XCTSkip("Missing \(name) in \(dir)")
            }
        }
    }

    func testRealPackSHA256AndCRC() throws {
        try requirePackEnv()
        let dir = Self.packEnv!
        let expected = [
            "anima-turbo-v1.0-xsmax-w4.animapk": "ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25",
            "qwen3-0.6b-xsmax-w8.animapk": "ba59e4d1797de5f6512aeafcecf3f38e1f62a47313a2a400b949c9018d84ceab",
            "qwen-image-vae-xsmax-fp16.animapk": "10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447",
        ]
        for (name, sha) in expected {
            let url = URL(fileURLWithPath: dir + "/" + name)
            let data = try Data(contentsOf: url) // 2 GB max — acceptable for the real-pack gate only
            XCTAssertEqual(data.sha256Hex, sha, "SHA-256 mismatch for \(name)")
            let file = try AnimapkFile(url: url)
            XCTAssertEqual(file.crc32MismatchCount(), 0, "CRC-32 mismatch in \(name)")
        }
    }

    func testRealW4KnownVector() throws {
        try requirePackEnv()
        let url = URL(fileURLWithPath: Self.packEnv! + "/anima-turbo-v1.0-xsmax-w4.animapk")
        let file = try AnimapkFile(url: url)
        let t = try XCTUnwrap(file.tensor(named: "model.diffusion_model.blocks.0.mlp.layer1.weight"))
        let out = QuantDecoders.dequantW4(data: file.dataBytes(t).data,
                                          scale: file.scaleBytes(t)!.data,
                                          zero: file.zeroBytes(t)!.data,
                                          k: 8)
        // HANDOFF.md §12: dequantized first 8 values
        let ref: [Float] = [0.00304, -0.00059, -0.00014, 0.00032, -0.00104, -0.00150, -0.00014, -0.00195]
        for i in 0..<8 {
            XCTAssertEqual(out[i], ref[i], accuracy: 5e-5, "W4 real-pack index \(i)")
        }
    }

    func testRealW8KnownVector() throws {
        try requirePackEnv()
        let url = URL(fileURLWithPath: Self.packEnv! + "/qwen3-0.6b-xsmax-w8.animapk")
        let file = try AnimapkFile(url: url)
        let t = try XCTUnwrap(file.tensor(named: "model.embed_tokens.weight"))
        let out = QuantDecoders.dequantW8(data: file.dataBytes(t).data,
                                          scale: file.scaleBytes(t)!.data,
                                          zero: file.zeroBytes(t)!.data,
                                          k: 8)
        // HANDOFF.md §13: dequantized first 8 values
        let ref: [Float] = [-0.00342, 0.03285, -0.07002, -0.01990, -0.00540, -0.01727, -0.03046, 0.00516]
        for i in 0..<8 {
            XCTAssertEqual(out[i], ref[i], accuracy: 5e-5, "W8 real-pack index \(i)")
        }
    }
}
