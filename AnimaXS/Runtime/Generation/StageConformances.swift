import Foundation

// MARK: - Production stage conformance (K002 seams)

// The production executors already expose the exact `execute`/`decode`
// signatures the engine protocols require; conformance is declaration-only.
extension QwenEncoderMetal: PromptEncoderStage {}
extension LLMAdapterMetal: ContextAdapterStage {}
extension DiffusionSampler: DiffusionStage {}
extension VAEDecoder: VAEDecodeStage {}
