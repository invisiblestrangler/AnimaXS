# AnimaXS Final Quality TODO

Reset on 2026-08-13 at base `45c28f4` for the final image-quality investigation.
Historical work remains in git history and `DECISIONS.md`.

Completion criterion: a canonical generated image genuinely comparable to the
reference from W4 and/or W8 using the updated iPhone XS Max pipeline. The old
0.65 regression floors and inference completion are not quality acceptance.

## Current phase

Phase 1 — baseline evidence and runtime/oracle audit.

## Phase 1 — reset and baseline

- [x] Refresh `origin/main` and record base `45c28f4`.
- [x] Preserve the live primary checkout and create `investigate/dit-quality-runtime`.
- [x] Read historical decisions and verify latest W4/W8 metrics/provenance.
- [x] Reset `TODO.md`, `STATUS.md`, and `TEST_MATRIX.md`.
- [ ] Audit attention, DiT, oracle, and full-inference implementations.
- [ ] Create and launch branch-only diagnostics.

## Phase 2 — same-W8 reference

- [ ] Generate exact-dequantized-W8 source evidence on Linux Actions.
- [ ] Compare high-precision and Swift-like mixed precision at block 0.
- [ ] Record cosine, RMSE, maxAbs, norms, and exact provenance.

## Phase 3 — attention investigation

- [ ] Add minimal legacy and FP32-score/softmax diagnostic modes.
- [ ] Add an independent deterministic attention precision comparison.
- [ ] Run focused macOS attention and block-0 parity tests.
- [ ] Reject or promote the attention hypothesis from evidence.

## Phase 4 — localize divergence

- [ ] Produce same-W8 step-0 block drift evidence and machine-readable CSV.
- [ ] Localize the first meaningful bad block/branch.
- [ ] If attention is insufficient, test the next repeated precision/runtime boundary.

## Phase 5 — promote winners

- [ ] Run eight-step W8 latent-only inference for earned candidates.
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
