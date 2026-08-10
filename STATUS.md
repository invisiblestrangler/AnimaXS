# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — GREEN CI (parser/decoder/tokenizer verified); Qwen encoder layer-0 verified, full-28 parity in progress
- **Current task:** Debug Qwen encoder full-28 parity vs golden cond_context (see docs/QWEN_ENCODER_DEBUG.md)
- **Last green commit:** d2c9dc4 + CI run 31410534456 (all 3 jobs pass, encoder compiles)
- **Current CI run:** GREEN
- **What currently works:**
  - **CI fully green**: Xcode 26.3 ARM device build (incl. Metal shaders + swift-transformers), 21 simulator tests pass
  - Swift ANMA parser: JSON-authoritative, CRC-32, alignment, ranges (validated vs real packs, 0 mismatches)
  - CPU W4/W8 decoders — known vectors byte-exact vs HANDOFF.md
  - **Tokenizer parity CI-verified**: Qwen (exact regex) + T5 (Unigram) byte-exact on 3 canonical prompts
  - Bundled qwen_tokenizer.json + t5_tokenizer.json (flat unique names, no collisions)
  - Fixtures: case1 golden (noise/sigmas/t5ids/final_latent/cond checksums) + tokenizer ref IDs
  - MetalContext + AnimaKernels.metal compile
  - **Qwen encoder CPU reference (QwenNumerics + QwenEncoderCPU): layer-0 numerics EXACT** (cosine 1.0 vs independent Python reference); full 28-layer NOT yet matching golden (in-progress)
- **What currently fails:** Qwen encoder full-28 parity (cosine ~-0.04 vs golden; shared Swift==Python reference misunderstanding — documented). Everything else green.
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. Debug Qwen encoder parity (isolate first diverging layer; check optimized_attention/pytorch path, mask value finfo.min/4, position_ids)
  2. E002/E006 — Metal W4/W8 dequant kernels + MPS linear
  3. G001 — LLM adapter
