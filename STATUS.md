# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — Qwen encoder parity RESOLVED; LLM adapter (G001/G002) DONE and validated structurally. Next: DiT input/timestep (H001/H002).
- **Current task:** Commit LLMAdapter + regression tests + oracle; green CI; then resume RUNBOOK (H001 DiT input).
- **Last green commit:** 5d8bb86 (Qwen GQA/final-norm fix, CI run 31420780844 GREEN)
- **Current CI run:** GREEN (31420780844 on 5d8bb86). Adapter commit pending.
- **What currently works:**
  - **CI fully green**: Xcode 26.3 ARM device build (incl. Metal shaders + swift-transformers), simulator tests pass
  - Swift ANMA parser (JSON-authoritative, CRC-32, alignment, ranges) — validated vs real packs
  - CPU W4/W8 decoders — known vectors byte-exact vs HANDOFF.md
  - **Tokenizer parity CI-verified**: Qwen (exact regex) + T5 (Unigram) byte-exact
  - **Qwen encoder full-28 parity RESOLVED**: Swift W8 == pinned-Comfy oracle cosine 1.000000; vs golden 0.992164 (W8 quantization)
  - **LLMAdapter (G001/G002) DONE + structurally validated**: Swift == pinned-Comfy oracle (same W4 weights) cosine **1.000000** (weighted [1,47,1024] and padded [1,512,1024]); all finite. Source-cited details in D021.
  - Oracle scripts: `scripts/qwen_comfy_oracle.py`, `scripts/anima_adapter_oracle.py` + fixtures in `scripts/oracle_out/`
- **What currently fails:** Nothing blocking. All validated components green. A12 device-only items pending (no physical device).
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. Push LLMAdapter + LLMAdapterTests + oracle; regenerate xcodeproj (bootstrap-project); verify CI green
  2. H001 — DiT input (17-ch, patchify 2×2 → 1024 tokens ×68, input proj → 2048 fp32 residual) + H002 timestep embedder
  3. E002/E006 — Metal W4/W8 dequant kernels + MPS linear
