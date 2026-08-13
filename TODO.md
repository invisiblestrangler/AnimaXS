# AnimaXS Final Quality TODO

Reset on 2026-08-13 at base `45c28f4` for the final image-quality investigation.
Historical work remains in git history and `DECISIONS.md`.

Completion criterion: a canonical generated image genuinely comparable to the
reference from W4 and/or W8 using the updated iPhone XS Max pipeline. The old
0.65 regression floors and inference completion are not quality acceptance.

## Current phase

Phase 4/6 — BF16 compute-mode matrix running (true `bf16_compute` was the
missing highest-priority experiment; run #29 was a legacy control).

## Phase 1 — reset and baseline

- [x] Refresh `origin/main` and record base `45c28f4`.
- [x] Preserve the live primary checkout and create `investigate/dit-quality-runtime`.
- [x] Read historical decisions and verify latest W4/W8 metrics/provenance.
- [x] Reset `TODO.md`, `STATUS.md`, and `TEST_MATRIX.md`.
- [x] Audit attention, DiT, oracle, and full-inference implementations.
- [x] Create and launch branch-only diagnostics.

## Phase 2 — same-W8 reference

- [x] Generate exact-dequantized-W8 source evidence on Linux Actions.
- [x] Compare high-precision and Swift-like mixed precision at block 0.
- [x] Record cosine, RMSE, maxAbs, norms, and exact provenance (run
  `31678571617`). Metal-vs-exact-W8 relative L2 is `0.00019` self,
  `0.00088` cross, and `0.00023` MLP; block 0 runtime is healthy.

## Phase 3 — attention investigation

- [x] Add minimal legacy and FP32-score/softmax diagnostic modes.
- [x] Add an independent deterministic attention precision comparison.
- [x] Run focused macOS attention precision tests.
- [x] Reject FP32 attention as the primary fix — local RMSE improved ~5x,
  but final W8 latent cosine improved only `+0.00047` in run `31676322657`.
- [x] Run same-W8 block-0 parity.
- [x] Run sparse first-divergence localization across blocks 0/7/14/21/27.

## Phase 4 — localize divergence

- [x] Produce same-W8 step-0 branch-delta evidence and machine-readable CSV.
- [x] Reject selective late-branch FP16, all-block FP16, and all-DiT FP16 ceilings.
- [x] Reject BF16 residual-boundary emulation (`31690018615`).
- [x] Confirm run #29 (`31701142683`) was legacy/production, not BF16.
- [x] Fix workflow efficiency (manual-only dispatch, pack once + macOS matrix,
      pack reuse) and artifact provenance (provenance.json, real commit/run_id).
- [x] Add `scripts/measure_grid_carrier.py` and record the defective baseline.
- [ ] Run the true `bf16_compute` full-image matrix (production + golden Qwen)
      with a legacy-golden control (pack reuse from run #29).
- [ ] Apply the BF16/golden decision tree (handoff §9).

## Phase 5 — promote winners

- [x] Run eight-step W8 FP32-attention latent-only inference; insufficient.
- [x] Run full-image source-FP16 weight ceilings; visible grid persists.
- [ ] Run full canonical RGB inference only for strong latent candidates.
- [ ] Inspect images for grid/etched/checker artifacts and natural detail.
- [ ] Once shared runtime improves, compare corrected W4 and W8.
- [ ] If BF16 is insufficient: per-step trajectory exports + source scheduler
      oracle + sampler/preconditioning audit (handoff §11–§16).

## Phase 6 — production and acceptance

- [ ] Select and simplify the production implementation/model format.
- [ ] Preserve streaming, tiled attention, and bounded iPhone scratch memory.
- [ ] Record before/after latent and RGB metrics plus image artifacts.
- [ ] Pass generic iPhone build, simulator tests, and canonical inference.
- [ ] Clean diagnostics, update manifest/docs/decisions, integrate latest main.
- [ ] Open the final PR and complete the requirement-by-requirement audit.
