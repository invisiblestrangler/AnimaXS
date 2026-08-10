# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — GREEN CI (build + parser/decoder/tokenizer tests pass); next: Qwen encoder (F005)
- **Current task:** F005 Qwen text encoder (embed gather → 28 layers → cond_context)
- **Last green commit:** 402acaa (fixed T5 reference fixture) — CI fully green
- **Current CI run:** ALL GREEN (project-consistency + iphone-build + simulator-tests)
- **What currently works:**
  - **CI fully green**: Xcode 26.3 ARM device build passes (incl. Metal shaders + swift-transformers), all 21 simulator tests pass
  - Swift ANMA parser: header/JSON/table, JSON-authoritative, CRC-32, alignment, ranges (validated vs real packs, 0 mismatches)
  - CPU W4/W8 decoders (Data-based) — known vectors byte-exact vs HANDOFF.md
  - **Tokenizer parity CI-verified**: Qwen (Qwen2 BPE, exact regex) + T5 (Unigram) produce byte-exact reference IDs on all 3 canonical prompts
  - Bundled qwen_tokenizer.json + t5_tokenizer.json (flat unique names)
  - Fixtures: case1 golden (noise/sigmas/t5ids/final_latent/cond checksums) + tokenizer ref IDs
  - MetalContext + AnimaKernels.metal compile in device build
- **What currently fails:** Nothing. (RealPackDecoderTests correctly skip without ANIMAXS_PACKS_DIR.)
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. F005 — Qwen encoder (embed gather from W8 → fp32 residual → 28 streamed layers → cond_context)
  2. E002/E006 — Metal W4/W8 dequant kernels + MPS linear executor
  3. G001 — LLM adapter (T5 gather → 6 blocks → 512×1024 conditioning)
