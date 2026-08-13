# AnimaXS Final Quality TODO

Reset on 2026-08-13 at base `45c28f4` for the final image-quality investigation.
Historical work remains in git history and `DECISIONS.md`.

Completion criterion: a canonical generated image genuinely comparable to the
reference from W4 and/or W8 using the updated iPhone XS Max pipeline. The old
0.65 regression floors and inference completion are not quality acceptance.

## Current phase

Phase 2/4 — same-W8 reference and first-divergence localization.

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
- [ ] Run first-divergence localization across later blocks/steps.

## Phase 4 — localize divergence

- [ ] Produce same-W8 step-0 block drift evidence and machine-readable CSV.
- [ ] Localize the first meaningful bad block/branch.
- [ ] If attention is insufficient, test the next repeated precision/runtime boundary.

## Phase 5 — promote winners

- [x] Run eight-step W8 FP32-attention latent-only inference; insufficient.
- [ ] Run full canonical RGB inference only for strong latent candidates.
- [ ] Inspect images for grid/etched/checker artifacts and natural detail.
- [ ] Once shared runtime improves, compare corrected W4 and W8.

## Phase 6 — production and acceptance

- [ ] Select and simplify the production implementation/model format.
- [ ] Preserve streaming, tiled attention, and bounded iPhone scratch memory.
- [ ] Record before/after latent and RGB metrics plus image artifacts.
- [ ] Pass generic iPhone build, simulator tests, and canonical inference.
- [ ] Clean diagnostics, update manifest/docs/decisions, integrate latest main.
- [ ] Open the final PR and complete the requirement-by-requirement audit.

## Operational visibility

- [ ] Send Telegram progress/final reports when `tg_send` becomes available.
  Current environment exposes neither the command nor a Telegram connector;
  persistent state and repository tracking are being updated instead.
