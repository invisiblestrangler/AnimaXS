# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — Qwen encoder + LLM adapter validated; **DiT input (H001), timestep (H002), AdaLN modulation (H003), 3-D RoPE (H004) DONE and validated vs oracle**. Next: H005 DiT block 0 end-to-end.
- **Current task:** H005 DiT block 0 end-to-end vs golden block_00_out (needs H001–H004 + Metal attention/MPS primitives).
- **Last green commit:** a138854 (H004 docs prep). **H004 source is ready for CI** (DitRoPE.swift + DitRoPETests + dit_rope_oracle.py) — pending xcodeproj regen + CI run.
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
  - **AdaLN modulation (H003) DONE**: `Modulation.swift` SiLU→Linear1→Linear2→chunk shift/scale/gate (fp32) + LayerNorm apply. Cosine 1.000000 vs oracle (SiLU-before-Linear1 per pinned source, D026).
  - **DiT 3-D RoPE (H004) DONE**: `DitRoPE.swift` weightless [1024,64,2,2] (dim 42/42/44, thetas 42870.938/10000 computed exactly, freq order t,h,w, 2×2 `[cos,-sin,sin,cos]` blocks). Cosine 1.000000 vs oracle (`scripts/dit_rope_oracle.py`). CPU tests `DitRoPETests`.
  - Oracle scripts: `scripts/qwen_comfy_oracle.py`, `anima_adapter_oracle.py`, `dit_input_timestep_oracle.py`, `dit_rope_oracle.py` + fixtures in `scripts/oracle_out/`
- **What currently fails:** Nothing blocking. All validated components green. A12 device-only items pending (no physical device).
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. Push H004 (DitRoPE + DitRoPETests + dit_rope_oracle); regenerate xcodeproj (bootstrap-project); verify CI green
  2. H005 — DiT block 0 end-to-end vs golden block_00_out
  3. H006 — Full 28-block loop with WeightStreamer ring
