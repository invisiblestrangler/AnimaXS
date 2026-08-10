# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — Qwen encoder + LLM adapter validated; **DiT input (H001) + timestep embedder (H002) DONE and validated vs oracle**. Next: H003 AdaLN modulation.
- **Current task:** H003 AdaLN LoRA modulation (shift/scale/gate chunk ordering per branch; fp32) + CPU test.
- **Last green commit:** 870741f (xcodeproj regen for H001/H002). CI run 31428466054 — all 3 jobs GREEN (project-consistency, iphone-build, simulator-tests incl. new DiTInputTests + TimestepEmbedderTests).
- **Current CI run:** GREEN (31428466054 on 870741f).
- **What currently works:**
  - **CI fully green**: Xcode 26.3 ARM device build (incl. Metal shaders + swift-transformers), simulator tests pass
  - Swift ANMA parser (JSON-authoritative, CRC-32, alignment, ranges) — validated vs real packs
  - CPU W4/W8 decoders — known vectors byte-exact vs HANDOFF.md
  - **Tokenizer parity CI-verified**: Qwen (exact regex) + T5 (Unigram) byte-exact
  - **Qwen encoder full-28 parity RESOLVED**: Swift W8 == pinned-Comfy oracle cosine 1.000000; vs golden 0.992164 (W8 quantization)
  - **LLMAdapter (G001/G002) DONE + structurally validated**: cosine 1.000000 vs oracle
  - **DiT input (H001) DONE**: `DiTInput.swift` patchify 2×2 → 1024×68 → x_embedder → [1024,2048] fp32. Cosine 1.000000 vs oracle.
  - **Timestep (H002) DONE**: `TimestepEmbedder.swift` sigma→sinusoidal 2048 + RMSNorm/MLP (adaln 6144). Cosine 1.000000 vs oracle.
  - Oracle scripts: `scripts/qwen_comfy_oracle.py`, `anima_adapter_oracle.py`, `dit_input_timestep_oracle.py` + fixtures in `scripts/oracle_out/`
- **What currently fails:** Nothing blocking. All validated components green. A12 device-only items pending (no physical device).
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. Push H001/H002 + DiTInputTests/TimestepEmbedderTests + oracle; regenerate xcodeproj (bootstrap-project); verify CI green
  2. H003 — AdaLN LoRA modulation (shift/scale/gate chunk ordering per branch; fp32) + CPU test
  3. H004 — DiT 3-D RoPE (T/H/W 42/42/44, theta 42871.1/10000, 2×2 blocks; CPU first)
