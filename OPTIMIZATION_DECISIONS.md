# OPTIMIZATION_DECISIONS

Semantic decisions affecting implementation. Append-only.

## D001 — Baseline branch is origin/main f89b1d8, not local fix/w8-import-refactor
Date: 2026-08-15
HEAD: f89b1d867883f04b2d9aa4b59e9322b36111c8b4
Decision: Create the optimization branch `opt/a12-sustained-io` from `origin/main` at `f89b1d8` (the planner-verified baseline, squash-merged PR #16). Do not branch from the local `fix/w8-import-refactor` branch.
Reason: The runbook P0 mandates branching from the newer `origin/main`. The local branch is a divergent pre-squash line with 4 extra commits that are already superseded by the squash-merged `f89b1d8` (which contains additional ModelStore/GenerationMetrics fixes beyond the local refactor commits). `f89b1d8` is the exact SHA the runbook designates as baseline.
Evidence: `git merge-base --is-ancestor a646a12 origin/main` → NO (divergent). `git diff --stat 6e8623d f89b1d8` shows f89b1d8 carries +25 lines in ModelStore.swift and extra DECISIONS/STATUS docs not in the local line.
Alternatives rejected: Branching from local `fix/w8-import-refactor` (stale, divergent, would inherit pre-squash artifacts and lose the f89b1d8 ModelStore fixes).
Revisit only if: origin/main is force-pushed or the baseline SHA changes.
## D002 — P4 strided token-major attention: extend AttentionExecutor, do not rewrite
Date: 2026-08-15
HEAD: b3aeb14 (P4 code committed)
Decision: Add an `AttentionInputLayout` enum (.headMajor / .tokenMajor(tokenStride:)) with the legacy `encode(...)` defaulting to head-major; add a parallel `encodeTokenMajor` path + `tokenMajorHeadMatrix` strided-view helper inside AttentionExecutor rather than touching Qwen/VAE/adapter call sites (they keep default head-major exactly). DiTBlockExecutor gates its 4 transpose kernels on a new `stridedTokenMajorAttention` config toggle (default OFF).
Reason: Runbook §9 requires (a) legacy head-major behavior identical for A/B, (b) no blind mutation of generic attention users, (c) score scratch stays tight (rows×keyCount, not token-dim stride), (d) cross-attention handled via the same row-stride helper with differing row counts (1024 vs 512). A layout parameter keeps the executor single-source without changing the legacy path.
Alternatives rejected: (1) New DiT-only attention class — duplicates softmax/QK/PV plumbing and metrics counting. (2) Rewriting the main encode loop unconditionally — would risk Qwen/VAE/adapter layouts.
Revisit only if: MPS on a physical device rejects strided descriptors (then implement auto-fallback per P4-F after device tests).
## D003 — P9 presets: keep baseline + individual A/B controls, add named combinations
Date: 2026-08-16
HEAD: d16e317 (P9 presets + tests committed; CI green on 31924467039)
Decision: Add an `InferencePreset` enum (baseline/current1024Control/fusedTraffic/stridedMPS/stridedMPSKV/noCopyCandidate/streamingMPSCandidate/metalFlashCandidate/directQGEMMCandidate/allCandidate) mapping to a concrete `InferenceOptimizationConfig`. Do NOT remove the per-toggle/per-backend controls. Do NOT change `currentBaseline` to a `recommendedA12` preset. `allCandidate` is one test configuration, not an automatic best.
Reason: Runbook §14 mandates named presets for the §17 physical-device benchmark matrix while retaining every individual A/B control. §14 forbids forcing a winner without physical A12 data, so no `recommendedA12` preset is added until the user measures on the XS Max. Presets are pure conveniences over the existing fields — snapshot-at-Generate immutability is unchanged.
Details: `InferenceOptimizationSettings.setPreset(_:)` applies the combination and persists every control plus an `activePreset` marker; invalid persisted preset raw values sanitize to `nil`; `resetToBaseline()` clears the marker. `baseline` preset == `currentBaseline` exactly.
Alternatives rejected: (1) Auto-selecting `allCandidate` as a new default — violates §14 (no device data). (2) Adding `recommendedA12` now — would claim a winner without measurements. (3) Replacing the individual controls with only presets — removes A/B isolation the runbook requires.
Revisit only if: the user supplies physical XS Max benchmark results (runbook §17) → then add `recommendedA12` with the proven winner.
