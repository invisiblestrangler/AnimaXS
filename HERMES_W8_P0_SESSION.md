# Hermes W8 P0 Session Ledger

**Branch:** fix/w8-range-and-fused-adaln
**Base HEAD (origin/main):** a82abef4364dde1924cce517fac1e5a1f35de1ab
**Working HEAD:** a82abef4364dde1924cce517fac1e5a1f35de1ab (branch created)

## Task status
- Task 1 (W8 final residual boundary): `verified` (Subagent 1; Hermes inspected runtime + test + CI diffs + static §12 checks; all pass)
- Task 2 (numerical magnitude telemetry): `verified` (Subagent 2; Hermes inspected source + test diffs; all refs resolve; preserved warnings:0 + not-collected)
- Task 3 (fused AdaLN offset): `verified` (Subagent 3; Hermes inspected host+test diffs; shader math matches CPU reference; static §12 fused checks pass; fused OFF in baseline)
- Task 4 (docs/state): `verified` (Subagent 4; Hermes inspected doc diffs: D005/D006 appended, TODO P0 top section, STATUS item-2 clarification; history preserved)

## Commits
- `ea5e471` W8 final-residual boundary + magnitude telemetry (+ CI gate + runbook docs)
- `78d9888` fused AdaLN offset float-element fix + ABI/parity tests
- HEAD = `78d9888`

## CI
- CI Gate A (normal CI `ci.yml`): run `31950731666` — **PASS** (project-consistency, iphone-build, simulator-tests all success, HEAD 78d9888)
- CI Gate B (full-inference-refine.yml W4/W8): run `31951006939` — **PASS**. W4: w4Legacy, full_inference PASS, latent_cos 0.8231, coherent image. W8: w8LegacyStabilized, block legacy/legacy, full_inference PASS, latent_cos 0.9108, rgb_cos 0.8616, coherent image. W8 pack identity intact (bytes 2232975360, sha256 8b63c7fd...130). FINAL_RESIDUAL_BOUNDARY gates enforced by workflow (exit 1 on mismatch) and green.

## Deviations
- (none)

## Physical device validation
- PENDING — user must perform W8-v2 run on physical iPhone XS Max (§13).
