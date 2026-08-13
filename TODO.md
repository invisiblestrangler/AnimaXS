# AnimaXS Refine TODO

This checklist is the execution index for `RUNBOOK.md`. A box is checked only
when its validation evidence exists in the repository, workflow, or recorded
artifact.

## Repository and documentation

- [x] Verify current `origin/main` and create `refine/packing-v2-w4-w8`.
- [x] Archive superseded operational documents.
- [x] Install the refinement runbook.
- [ ] Refresh status, handoff, and test matrix after the result cycle.
- [x] Preserve `DECISIONS.md` append-only and record refinement decisions.

## Determinism and legacy preservation

- [ ] Reset the selected simulator before pack-free CI tests.
- [ ] Run normal CI after runtime changes and confirm D072 determinism.
- [x] Preserve the recovered v1 packer unchanged and record its SHA.

## Packer v2 and verification

- [ ] Implement dry-plan and bounded-memory/direct output writing.
- [ ] Keep ANMA v1, 16-KiB blobs, group 64, row-reset, and W4 nibble order.
- [ ] Use stored-FP16 scale/zero values when selecting Q.
- [ ] Implement deterministic W4 MSE clipping and W8 affine quantization.
- [ ] Implement precision maps, finite checks, streaming statistics, provenance,
      per-blob SHA-256, and an independent verifier.
- [ ] Pass synthetic W4 `[2,68]` and W8 `[2,65]` parser/decoder regressions.

## Hugging Face and packing workflow

- [x] Resolve and pin source revision `f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b`.
- [x] Confirm source LFS SHA `c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174`.
- [ ] Configure `HF_TOKEN` securely as a GitHub secret without logging it.
- [ ] Pack W4-v2 and W8-v2 on separate fresh Ubuntu jobs.
- [ ] Verify and publish both packs directly to dedicated HF model repos.
- [ ] Record exact HF revisions, sizes, and SHA-256 values.

## Runtime

- [ ] Add one W4/W8 DiT matrix factory and central direct-matvec dispatch.
- [ ] Generalize preparation, adapter, block, and final-layer matrices.
- [ ] Add and test the W8 FP32 Metal matvec kernel.
- [ ] Confirm legacy W4 behavior remains unchanged.

## Full inference and decision

- [ ] Add exact-revision W4-v2/W8-v2 matrix inference workflow.
- [ ] Keep Qwen, VAE, prompt, seed/noise, sigmas, and graph identical.
- [ ] Complete both canonical inferences and capture generated/reference/
      comparison PNGs plus metrics and timings.
- [ ] Visually compare legacy W4, W4-v2, and W8-v2.
- [ ] Select W4-v2, W8-v2, or one evidence-backed mixed follow-up.
- [ ] Append the final decision and refresh all handoff documents.
