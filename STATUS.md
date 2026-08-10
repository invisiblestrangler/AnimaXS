# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — Qwen encoder full-28 parity **RESOLVED** (GQA grouping + final-norm fixes; structural oracle validates kernel). Next: LLM adapter (G001).
- **Current task:** Commit GQA/final-norm fix + regression test + oracle; green CI; then resume RUNBOOK (G001 LLM adapter).
- **Last green commit:** a414159 (pre-fix baseline)
- **Current CI run:** GREEN (run 31411533185 on a414159). New commit pending.
- **What currently works:**
  - **CI fully green**: Xcode 26.3 ARM device build (incl. Metal shaders + swift-transformers), 21 simulator tests pass
  - Swift ANMA parser (JSON-authoritative, CRC-32, alignment, ranges) — validated vs real packs
  - CPU W4/W8 decoders — known vectors byte-exact vs HANDOFF.md
  - **Tokenizer parity CI-verified**: Qwen (exact regex) + T5 (Unigram) byte-exact on 3 canonical prompts
  - **Qwen encoder full-28 parity RESOLVED (2026-08-10):**
    - Fixed GQA grouping: `kvHead = qHead / 2` (was `qHead % 8`) — pinned ComfyUI repeat_interleave grouping (D018)
    - Applied final RMSNorm to cond_context (golden is post-final-norm) (D019)
    - Swift W8 == pinned-ComfyUI W8 structural oracle: **cosine 1.000000, maxAbs 0.0004**
    - Swift W8 vs golden cond_context: **cosine 0.992164, rmse 0.434, maxAbs 47.5** (W8 quantization deviation, not kernel error)
    - Oracle script `scripts/qwen_comfy_oracle.py` + per-layer fixtures `scripts/oracle_out/qwen_oracle_layers.npz`
- **What currently fails:** Nothing blocking. All validated components green. A12 device-only items still pending (no physical device).
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. Push GQA/final-norm fix + GqaHeadMappingTests + oracle; regenerate xcodeproj (bootstrap-project workflow); verify CI green
  2. G001 — LLM adapter (T5 ids → W4 embedding gather → 6 blocks → out_proj → RMSNorm → T5-weight multiply → zero-pad to 512 → [1,512,1024])
  3. E002/E006 — Metal W4/W8 dequant kernels + MPS linear
