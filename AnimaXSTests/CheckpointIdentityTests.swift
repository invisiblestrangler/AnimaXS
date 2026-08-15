import XCTest
@testable import AnimaXS

/// P1-B: checkpoint identity must use the ACTUAL resolved model hashes
/// (`ResolvedModels.hashes`), so a W4 checkpoint can never resume against a
/// W8 pack and vice versa. `CheckpointStore.validate` rejects a hash mismatch.
final class CheckpointIdentityTests: XCTestCase {

    /// Builds a `ResolvedModels` whose DiT variant is the given id, and whose
    /// hashes therefore reflect that variant (not the hardcoded W4 primary).
    private func resolvedModels(ditVariantID: String) -> ResolvedModels {
        let ditVariant: ModelVariantDescriptor = ditVariantID == "w8-v2"
            ? ModelManifest.ditW8V2 : ModelManifest.ditW4
        return ResolvedModels(
            textEncoderURL: URL(fileURLWithPath: "/tmp/qwen.animapk"),
            textEncoderVariant: ModelVariantDescriptor(
                id: "textEncoder", displayFilename: "qwen3-0.6b-xsmax-w8.animapk",
                size: 635_305_984,
                sha256: "ba59e4d1797de5f6512aeafcecf3f38e1f62a47313a2a400b949c9018d84ceab"),
            ditURL: URL(fileURLWithPath: "/tmp/dit.animapk"),
            ditVariant: ditVariant,
            vaeURL: URL(fileURLWithPath: "/tmp/vae.animapk"),
            vaeVariant: ModelVariantDescriptor(
                id: "vae", displayFilename: "qwen-image-vae-xsmax-fp16.animapk",
                size: 256_163_840,
                sha256: "10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447"))
    }

    private func checkpoint(modelHashes: ModelHashes) throws -> GenerationCheckpoint {
        try GenerationCheckpoint(
            latent: [Float](repeating: 0.5, count: 16 * 64 * 64),
            step: 3, prompt: "test", seed: 7,
            width: 512, height: 512, modelHashes: modelHashes)
    }

    private func assertValidate(
        _ store: CheckpointStore, checkpoint: GenerationCheckpoint,
        models: ResolvedModels, line: UInt = #line
    ) throws {
        _ = try store.validate(
            checkpoint, prompt: "test", seed: 7,
            resolution: (512, 512), modelHashes: models.hashes)
    }

    func testW4CheckpointWithW4ResolvedModelsAccepted() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory)
        let w4 = resolvedModels(ditVariantID: "w4")
        let cp = try checkpoint(modelHashes: w4.hashes)
        XCTAssertNoThrow(try assertValidate(store, checkpoint: cp, models: w4))
    }

    func testW8CheckpointWithW8ResolvedModelsAccepted() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory)
        let w8 = resolvedModels(ditVariantID: "w8-v2")
        let cp = try checkpoint(modelHashes: w8.hashes)
        XCTAssertNoThrow(try assertValidate(store, checkpoint: cp, models: w8))
    }

    func testW4CheckpointWithW8ResolvedModelsRejected() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory)
        let w4 = resolvedModels(ditVariantID: "w4")
        let w8 = resolvedModels(ditVariantID: "w8-v2")
        let cp = try checkpoint(modelHashes: w4.hashes) // W4 checkpoint
        XCTAssertThrowsError(try assertValidate(store, checkpoint: cp, models: w8)) { error in
            XCTAssertTrue(error.localizedDescription.contains("model hashes"))
        }
    }

    func testW8CheckpointWithW4ResolvedModelsRejected() throws {
        let store = try CheckpointStore(directory: FileManager.default.temporaryDirectory)
        let w4 = resolvedModels(ditVariantID: "w4")
        let w8 = resolvedModels(ditVariantID: "w8-v2")
        let cp = try checkpoint(modelHashes: w8.hashes) // W8 checkpoint
        XCTAssertThrowsError(try assertValidate(store, checkpoint: cp, models: w4)) { error in
            XCTAssertTrue(error.localizedDescription.contains("model hashes"))
        }
    }

    func testResolvedModelsHashesReflectResolvedVariantNotW4Primary() {
        // The resolved W8 hashes must be the W8 SHA, not the hardcoded W4 primary.
        let w8 = resolvedModels(ditVariantID: "w8-v2")
        XCTAssertEqual(w8.hashes.dit, ModelManifest.ditW8V2.sha256)
        let w4 = resolvedModels(ditVariantID: "w4")
        XCTAssertEqual(w4.hashes.dit, ModelManifest.ditW4.sha256)
    }
}
