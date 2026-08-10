# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — tokenizer parity (Qwen/T5) + parser/decoders + Metal skeleton
- **Current task:** CI-verify TokenizerParityTests (F004); then Metal dequant kernels + Qwen encoder (F005)
- **Last green commit:** 51a8877 (regenerated xcodeproj with flat tokenizer resources)
- **Current CI run:** dispatched (awaiting result on 51a8877)
- **What currently works:**
  - Repo `invisiblestrangler/AnimaXS` populated; xcodeproj committed+in sync (project-consistency green)
  - iphone-build PASSES (Xcode 26.3 ARM device build incl. Metal shaders + swift-transformers)
  - Swift ANMA parser: header/JSON/table, JSON-authoritative, CRC-32, alignment, ranges (validated vs real packs, 0 mismatches)
  - CPU W4/W8 decoders (Data-based) — known vectors byte-exact vs HANDOFF.md
  - Tokenizer parity rule D014 VALIDATED: Qwen=encode(no_specials), T5=encode+[</s>]
  - Bundled qwen_tokenizer.json (exact Qwen2 regex) + t5_tokenizer.json — byte-exact vs golden oracle
  - Fixtures: case1 golden (noise/sigmas/t5ids/final_latent/cond checksums) + tokenizer ref IDs
  - MetalContext + AnimaKernels.metal (dequant/norm/activations/euler/patchify) compile in device build
- **What currently fails:** None known. reference-tests previously failed on wrong W8 group-semantics test (fixed). simulator-tests had one 20-min hang (mitigated with timeout-minutes + SPM cache).
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. CI-green on tokenizer parity (F004)
  2. E002 — Metal dequant kernels + MPS linear (validate w4/w8 GPU decode)
  3. F005 — Qwen encoder (embed gather → 28 layers → cond_context)
