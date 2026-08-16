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

## D004 — Device stabilization (fix/device-stability-no-checkpoint): checkpoint removed, W8 locked to legacy, P8/P6 quarantined, fatal-Metal poisoning
Date: 2026-08-16
Branch: fix/device-stability-no-checkpoint (off 1756ab8, PR #17)
Decision:
1. Remove checkpoint/resume from production entirely (Checkpoint.swift/CheckpointStore.swift deleted, no Resume UI, no cold-launch resume, GenerationEngine always starts at step 0, no per-step 256 KiB fp32 latent CPU readback). Checkpointing no longer justifies its complexity/per-step overhead for the sub-100 s target.
2. W8-v2 production resolves to `w8LegacyStabilized` (legacy attention/activation numerics), NOT the incomplete FP16-backed BF16 emulation. `DiTNumericsPolicy.fromVariantID("w8-v2")` → `w8LegacyStabilized`; `w8BF16Experimental` remains only for explicit diagnostics and is not claimed range-safe internally.
3. Full-inference CI must execute the production policy: `testCanonicalProductionInference` derives numerics from `DiTNumericsPolicy.fromVariantID(ANIMAXS_DIT_VARIANT)` via `DiffusionSampler.resolvedNumerics`; `full-inference-refine.yml` asserts per-variant markers (w4-v2 → w4Legacy, w8-v2 → w8LegacyStabilized).
4. Quarantine P8 `directQuantized`/`hybrid` from production/device settings: physical A12 measured ~10× slower than `dequantizedMPS` (performance regression, not proven incorrect). Settings sanitize/reject; `directQGEMMCandidate`/`allCandidate` force `dequantizedMPS`; Diagnostics hides them with a visible note; kernel stays research code.
5. Disable P6 mmap no-copy after a physical A12 GPU page fault (`kIOGPUCommandBufferCallback` ErrorPageFault): `noCopyWeightSource` cannot be true in production/device settings; re-enable only with a future GPU-read hardware proof.
6. Fatal Metal command-buffer faults (.pageFault/.invalidResource/.internal or narrow IOGPU text fallback) poison the generation context (`metalContextPoisoned`) — failed with "Fatal GPU fault. Restart AnimaXS before generating again.", Generate blocked until restart, no context recreate/retry. Cooperative cancellation never poisons.
7. Manual Import is local-only (`.borderless` buttons; Import can never cross-trigger Download) and per-component model operations are single-flight (`activeOperations` guard). App launch never auto-downloads model packs.
8. Prompt/seed persist across relaunch via `generation.lastPrompt` / `generation.lastSeed` (`@AppStorage`), Randomize included.
Reason: The physical XS Max exposed ambiguities that simulator/CI could not catch (W8 BF16 path unproven, A12 page fault, ~10× P8 regression, checkpoint complexity for a sub-100 s target, accidental Download from Import). The optimization branch's "ready for physical A12 validation" claim was premature; these constraints define the actual device-tested baseline.
Evidence: Commit 90ca169 (checkpoint removal), 82026cc (W8 legacy), 27d3eb7 (full-inference policy), e536bd7 (P8 quarantine), b14b88b (P6 disable + poisoning), f64eb5c (Import/single-flight), a4a1c60 (prompt/seed persistence), b25b845/f647908/605300f (fresh-run ownership, preset drift/compat validator, monitor labels). CI Gate A green on run 31932848703.
Alternatives rejected: Keeping checkpointing (per-step CPU/disk work with no consumer for the target); letting production W8 auto-select BF16 emulation (its FP16 storage does not preserve BF16 exponent range — the leading W8 NaN/checkerboard hypothesis); letting P8/P6 participate in the device search (measured ~10× regression / page fault); auto-downloading missing packs at launch (network side effects).
Revisit only if: physical XS Max results (stabilization plan §17) or a new GPU-read hardware proof for P6 arrives; the W4/W8 full-inference gate (plan §15) changes the picture; a planner-approved BF16 range fix makes `w8BF16Experimental` production-safe.

## D005 — W8 final residual range is independent of block activation numerics
Date: 2026-08-16
Branch: fix/w8-range-and-fused-adaln
HEAD: 78d9888
Decision:
1. The W8 final-residual ENTRY boundary is decoupled from block attention/activation numerics via a new `FinalResidualBoundary` enum. Production W8 (`DiTNumericsPolicy.w8LegacyStabilized`) keeps `.legacy`/`.legacy` block attention/activation numerics but uses `.bf16RNEInFP32` at final-layer entry: the large FP32 residual is rounded to BF16 RNE, retained in FP32 storage, and LayerNorm runs in FP32 over those values. The W8 large residual is NEVER converted to FP16 at final-layer entry.
2. W4 keeps its `.fp16Legacy` final boundary — byte-for-byte unchanged.
3. Whole-model `w8BF16Experimental` is still NOT promoted to production.
4. No W8 repack was required or done — pack identity unchanged (bytes 2232975360, sha256 8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130).
Reason: On the physical iPhone XS Max, the previous final `float_to_half` conversion overflowed above 65,504 after block 28/28, poisoning LayerNorm with NaN and corrupting output. Block activation numerics were already range-safe as legacy; only the final-residual entry range was unsafe, so the two are now independent decisions.
Details: `FinalResidualBoundary` separates the entry-boundary choice from the block numerics policy. Production W8 = legacy blocks + `.bf16RNEInFP32` residual entry; W4 = `.fp16Legacy` (unchanged). Full `w8BF16Experimental` remains a diagnostics-only option.
Evidence: CI Gate A PASS (run 31950731666); CI Gate B PASS (run 31951006939, W8 full-inference produced a coherent image, latent_cos 0.9108). Physical iPhone XS Max W8-v2 retest remains PENDING — macOS Metal runner CI does not prove device correctness.
Alternatives rejected: Keeping the FP16 final-residual conversion (device overflow/NaN); promoting full `w8BF16Experimental` to production (unproven range safety); repacking W8 to shrink activation magnitudes (pack identity unchanged, not needed).
Revisit only if: the physical XS Max W8-v2 retest fails — then re-evaluate the final boundary and/or a planner-approved promotion of `w8BF16Experimental`.

## D006 — Fused AdaLN shader offset is measured in float elements
Date: 2026-08-16
Branch: fix/w8-range-and-fused-adaln
HEAD: 78d9888
Decision: The fused LayerNorm+AdaLN+to-half shader's modulation offset is passed in float elements, not bytes. The host now passes `fusedModulationElementOffset(columns:)` = 2048 (float elements), replacing the previous byte count (dim*4 = 8192).
Reason: The shader adds the offset to a `float*`, so the host-passed byte count was interpreted as 8192 floats, walking 2048 floats past the real 6144-float modulation buffer — the source of the A12 GPU page fault.
Details: Old host passed dim*4 = 8192 (bytes); the shader reads the value as float elements, so it addressed 8192 floats against a 6144-float modulation buffer. The host now passes 2048 float elements. `fusedNormModulation` remains OFF in the production baseline (`InferenceOptimizationConfig.currentBaseline.fusedNormModulation == false`).
Evidence: CI Gate A PASS (run 31950731666); CI Gate B PASS (run 31951006939). Physical XS Max benchmarking of the fused path is still PENDING, so the option stays non-default.
Alternatives rejected: Converting the offset inside the shader (keeps the ambiguous byte/element unit contract); leaving the offset as-is (guaranteed out-of-bounds read / page fault).
Revisit only if: physical XS Max benchmarking of the fused path (with the corrected offset) shows a regression — then revisit the offset units and buffer layout.
