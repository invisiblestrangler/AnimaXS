# AnimaXS Optimization Phase — DECISIONS.md

Branch: `opt/reliability-telemetry` (off `origin/main` @ `aa87947`)
Start: 2026-08-15

## Scope (from the optimization runbook)

1. Reliability: eliminate/harden intermittent non-finite diffusion latent failures.
2. Observability: trustworthy on-device telemetry (unplugged phone is the benchmark).
3. Performance: reduce generation time without correctness changes.
No thermal logic. No silent clamping. No global FP32. Small, reversible commits.

## Architecture facts verified against source (2026-08-15)

- **Precision layout (legacy mode):** residual/modulation/norm fp32; all MPS GEMM
  paths (Q/K/V/O, MLP, final projection) run fp16 with W4/W8 dequantized once per
  block into a reusable fp16 scratch (`LinearExecutor`); attention scores fp16
  (legacy) via MPS; softmax fp16-storage/fp32-accumulate; gate-add fp32.
- **FP32→FP16 conversion points (`float_to_half` kernel):** crossContext (step),
  projectionInput (self/cross/mlp), MLP hidden→half, final residual/normalized/
  modulated/projectionInput. MSL `half(x)` on |x| > 65504 → ±Inf; NaN propagates.
- **FP16→FP32:** `gate_add_half_f32` adds `float(branch) * gate` into fp32 residual.
  An Inf branch poisons the fp32 residual; next LayerNorm turns Inf → NaN (rsqrt(Inf)=0).
- **Weight streaming:** one slot (`WeightStreamer.ring`, capacity = max DiT block
  range ≈ 38,993,920 B). Serialization pattern per block: CPU memcpy (load) → encode →
  commit → await completion. GPU idles during every load+encode.
- **Per-block execution:** one command buffer per DiT block; per-block CPU readback
  only in diagnostic mode.
- **Failure reporting:** final CPU-side `isFinite` check after Euler (too late to
  attribute origin); error strings contain literal `\(step)` interpolation bugs
  (DiffusionSampler.swift:100/157, CheckpointStore.swift:66/79, GenerationEngine.swift:246/297).
- **Existing alternative precision paths:** `AttentionNumerics.fp32ScoresAndSoftmax`,
  `ActivationNumerics.bf16Boundaries` / `.bf16Compute` (both store in fp16 buffers —
  BF16 exponent range NOT preserved; runbook warning applies).

## Experiments / decisions (chronological)

### D200 — Phase 1: error interpolation fix
- All six literal `\(...)` sites fixed to real interpolation; numerical failure
  message now human-indexed (1-based step/block) with stage attribution hooks.
- Decision: new `NumericalFailure` formatted via a pure, unit-tested builder so
  the message shape is stable and testable.

### D201 — Phase 3: numerical-health instrumentation design
- Decision: `NumericalMonitor` owns one small GPU stats buffer (per-probe slot:
  flags uint + maxAbs float bits + firstIndex uint). Writes via atomics.
- In-kernel probes (always-on, ~free since kernels already touch every element):
  `float_to_half` (records input NaN/±Inf/|x|≥65504 pre-conversion + maxAbs),
  `gate_add_half_f32` (records non-finite branch/gate entering fp32 residual),
  `attention_softmax_rows` (records non-finite/overflowing scores input).
- Standalone probe passes (flag-gated diagnostic mode): MPS outputs (q/k/v tokens,
  attended, branch, projected) and fp32 residual after self/cross/mlp stages.
- Attribution: probe slots encode stage; sampler tags current (step, block) so the
  first-unsafe report reads "step N/8, block M/28, <stage>: <condition>".
- Rejected: copying full tensors to CPU per boundary (bandwidth waste); a global
  "GPU %" gauge (not a real API).

### D202 — Phase 7/8: telemetry design
- Decision: `GenerationMetrics` value type collected by engine + sampler + streamer:
  stage wall times, per-step/per-block wall, weight copy time+bytes, Metal encode
  time, GPU command time (command-buffer GPU timestamps), host wait time, peak Metal
  allocation, min available process memory. Summary rendered in-app (post-gen) and
  in the Diagnostics screen.
- Implemented: `MetricsCollector` (per-run, injected by coordinator → engine →
  sampler → executors). Stage timing in engine; per-block copy/encode/gpu/wait in
  `DiTBlockExecutor` (GPU time from `MTLCommandBuffer.gpuStartTime/gpuEndTime`,
  guarded when unavailable); per-block memory sampling via
  `MTLCurrentAllocatedSize` + `os_proc_available_memory`; numerical warnings from
  the monitor report. UI: "Run metrics" section in ContentView + DiagnosticsView
  with share/copy. `other` = wall − Σ(stages), so counters reconcile with wall.

### D203 — Phase 12: two-slot ping-pong streamer
- Memory headroom verified: +1 slot ≈ +39 MB (max block range 38,993,920 B) vs
  ~1.4 GB peak — headroom exists, implemented exactly two slots.
- `WeightStreamer` v2: `slotCount` slots, `load(range, from:, slot:)`, hard
  in-flight guard (`markInFlight`/`complete` — a slot referenced by a committed
  command buffer refuses overwrite). Backward-compatible `ring` = slot 0.
- `DiTBlockExecutor.execute` gains `slot` + `prefetchIndex/prefetchSlot`;
  `DitForward` runs the ping-pong loop: prologue loads block 0 → for each block
  commit → prefetch next block into the other slot (CPU memcpy overlaps GPU) →
  await. Block N's previous slot user (block N-1) is always awaited first, so
  the overlap is race-free; outputs byte-identical (same bytes at same offsets).
- Prefetch failure still awaits the in-flight buffer before propagating.
- Tests: slot independence, in-flight overwrite refusal, lifecycle tracking,
  slot-count validation.

### D204 — Phase 2/4: stress harness
- `FullInferenceTests.testNumericalStressAcrossSeeds`: production diffusion
  sampler over N ordinary seeds (diverse SeededRNG noise on golden
  conditioning), detailed probes enabled, aggregates first-unsafe-boundary
  distribution (by probe and step) + max magnitudes per boundary. Prints
  `FULL_STRESS_*` markers; gates ONLY on non-finite latents (correctness), never
  on warnings (that is the investigation's question). Seed count via
  `ANIMAXS_STRESS_SEEDS` (default 4).
- Metrics summary now includes per-probe numerical details
  ("Numerical warnings: 2 (self-attention scores: Inf detected; …)") so the
  in-app report is actionable without a cable.
- `DiffusionSampler.numericalReport` / `earliestNumericalIssue` accessors for
  the harness.

### D205 — Two-slot streamer crash found by CI + fixed
- The first full-inference run on the branch CRASHED the test host:
  `-[_MTLCommandBuffer addCompletedHandler:]:976: failed assertion 'Completed
  handler provided after commit call'` — the ping-pong restructure committed the
  block command buffer BEFORE registering its completion handler (Metal hard
  assertion). The golden-case test therefore never ran its correctness gates.
- Fix: `CommandBufferGate` — the handler is registered BEFORE `commit()`, then
  the next-block prefetch memcpy runs, then `wait()` resumes via the stored
  continuation. The gate is correct in both orderings (resume-before-wait and
  wait-before-resume, with error propagation). Regression tests cover both.
- Lesson: never move `addCompletedHandler` after `commit()` when restructuring
  Metal awaits; keep handler registration adjacent to commit.

### D206 — Terminal (8/8) checkpoint must never offer Resume; clear after completion
- **Bug (user-reported):** after a generation completes, the app could show only
  "Resume" ("Checkpoint: 8/8 steps — Resume available.") instead of Generate.
  Root cause: a checkpoint at `step == samplerSteps` (8) means diffusion FULLY
  completed — nothing left to resume — yet `canResume` accepted it
  (`step <= 8` in `.idle`/`.cancelled`). Such checkpoints were also retained +
  persisted (the step-7 callback saves step 8), so a cancel-during-VAE or a
  cold launch with a stale 8/8 file suppressed the Generate button entirely
  (no discard affordance in `.idle`).
- **Fix (4 layers):**
  1. `canResume` requires `step < samplerSteps` (1...7 only).
  2. The checkpoint callback never retains/persists a terminal checkpoint
     (`nextStep >= samplerSteps` → skip).
  3. On `.completed`, `clearCheckpoint()` drops the retained + persisted
     checkpoint (the user-facing "cache" is cleared after every generation).
  4. Cold-launch init removes a terminal checkpoint found on disk (legacy
     files) instead of retaining it.
  - Plus race hardening: a `generationEpoch` (bumped per run + per clear)
    + `isGenerating` guard invalidates any in-flight checkpoint-save task that
    could otherwise resurrect a stale checkpoint after completion or a fresh
    Generate (the LifecycleSampler harness queues all saves synchronously, so
    this is a real ordering hazard, not theoretical).
  - Defense-in-depth: `CheckpointStore.validate` rejects step 8 too; the
    ContentView caption now shows the real step count.
- Tests: completed-generation clears checkpoint + offers Generate not Resume;
  terminal checkpoint on disk cleared at cold launch; 7/8 still offers resume
  (boundary guard); validate rejects terminal step. Also fixed pre-existing
  literal `\(...)` escapes in the test files' temp-dir paths and failure
  messages (same bug class as D200).

## Open questions
- Which boundary is the actual first-unsafe site on A12? (Instrument → stress →
  attribute; do not guess.)

---

## Runtime inference-optimization experiments (D207) — decisions (append-only)

- **Performance experiments are runtime-configurable in one build.** Linear
  and attention tile rows (128/256/512/1024), direct MPS linear I/O, ping-pong
  weight streaming, numerical monitoring, and DiT pack (W4/W8) are selectable
  at runtime in Diagnostics. No environment variables or compile flags: one
  installed binary can run every experiment, so GitHub Actions is not the
  per-variation inner loop.
- **Current behavior remains the baseline.** `InferenceOptimizationConfig.currentBaseline`
  (L128, A128, direct OFF, ping-pong ON, monitor ON, W4) exactly reproduces
  current HEAD. "Reset to current baseline" restores it.
- **W8 is Diagnostics-only.** The experimental W8 v2 pack is isolated from the
  production `ModelManifest.entries` topology; it is imported/removed manually
  and selected per-run in Diagnostics. No auto-download, no HF credentials in
  the app, and no production-role collision.
- **W8 checkpointing is disabled in this batch.** The production W4 hash set
  does not describe the W8 pack, and a W8 checkpoint must not resurrect
  unrelated production Resume state. Production W4 keeps the current
  completed-8/8 cleanup behavior exactly.
- **Thermal/power values are observational telemetry and never generation
  gates.** Start/end power, battery, thermal, and Low Power Mode are recorded
  in the summary so runs under different conditions are discarded, never used
  to throttle or block generation.

---

## W8 v2 import crash fix — decisions (append-only)

- **W8 import is now a single-pass bounded-memory stream.** The reported
  real-device crash during `anima-turbo-v1.0-xsmax-w8-v2.animapk` import was
  traced to the two-pass implementation: a full SHA-256 pass over the 2.233 GB
  source followed by a second full `FileManager.copyItem` pass, with no
  per-chunk temporary-lifetime boundary. `ExperimentalDiTPackStore.importPack`
  now streams the source into staging EXACTLY ONCE (`read → hash → write` per
  chunk) at a fixed 1 MiB chunk size inside an explicit `autoreleasepool`, then
  checks the pinned byte count and SHA-256 before atomic install + receipt.
  This removes ~2.23 GB of redundant source-side I/O and bounds memory.
- **The production `ModelManifest.sha256(of:chunkBytes:)` helper is hardened**
  with an `autoreleasepool` boundary per chunk (unchanged hash behavior, chunk
  size, and semantics — no mmap, no `Data(contentsOf:)`). It is used by the
  production model integrity / Diagnostics deep-integrity paths, which can also
  hash multi-hundred-MB / multi-GB files.
- **`ExperimentalDiTPackStore` no longer has a `Verifier` injection seam** —
  the pre-copy full-SHA verifier became dead code once the stream performs
  verify-in-place. Removed cleanly; tests updated. `chunkBytes` is injectable
  (default 1 MiB) so tests force many stream iterations without a real pack.
- **Import state/UI races fixed:** the catalog publishes `.verifying` BEFORE
  awaiting the store; the W8 row hides Import while `.verifying` (a second
  multi-GB import can't be queued); and W8 Import/Remove are disabled while a
  generation is active. Security-scoped file access remains held for the entire
  stream (no earlier `stopAccessingSecurityScopedResource`, no in-memory copy).
- **Regression tests use synthetic multi-chunk fixtures** (a few KiB–1 MiB with
  4 KiB chunks), never the 2.23 GB pack: many-iteration success, SHA-mismatch
  leaves no installed/staging file, stat size gate creates no staging,
  re-import replaces, catalog `.verifying` in-flight, and a 7-byte-chunk
  `ModelManifest.sha256` multi-iteration regression. No production pack in CI.
- **Future cleanup (NOT part of this fix):** the production
  `ModelStore.importPack` manual-import path has the same broad two-pass pattern
  (`ModelManifest.verify(source)` → `installVerified(...)` → `copyItem`). It was
  left unchanged to avoid destabilizing download/repair flows; if it can be
  reused safely, the new single-pass helper should be shared there later.
- **The original device crash is not claimed to be mathematically proven as
  jetsam** until the new build is re-tested on the physical iPhone XS Max (see
  DEVICE_TESTS.md). This patch removes the concrete high-risk multi-GB import
  behavior already identified in the code.

---

## W8 v2 becomes a normal DiT pack (main page) — decisions (append-only)

- **The experimental W8 special path is removed.** `ExperimentalDiTPackStore`,
  `ExperimentalDiTPackCatalog`, `ExperimentalDiTManifest`, and the
  `DiTPackVariant` picker/per-run substitution are deleted. K's directive:
  "have it load the normal way like w4 at main page and remove it from the
  diagnostics page. The user can then import either w4 or w8-v2."
- **The `.dit` slot accepts either W4 (primary) or W8-v2 (alternate).**
  `ModelManifestEntry.allVariants` lists accepted (size, SHA-256) pairs;
  `matchedVariant`/`verify` accept the primary or any alternate. Whichever
  pack the user imports into the Models row is the DiT a generation uses.
  Importing W8 over W4 replaces it; discovery trusts the receipt for either.
- **Receipts record the ACTUAL matched variant** (size + SHA-256), so a
  relaunch trusts whichever pack (W4 or W8-v2) was installed without re-hashing.
- **The single-pass streaming import moved into the normal path.**
  `ModelStore.verifyAndStage` streams source → staging exactly once (1 MiB
  chunks, `autoreleasepool` per chunk), computes SHA-256 during the copy, and
  matches against the accepted variants. It is shared by download and manual
  import. The production `ModelStore.importPack` no longer does a separate
  full-SHA pass + `FileManager.copyItem`. This preserves the W8 crash fix on
  the normal path while removing the old two-pass behavior.
- **Generation config no longer carries a DiT variant.** Checkpointing is
  always on (the slot holds one verified pack, so no separate experimental
  state can collide). Run metrics report the actual DiT pack filename.
- **Tests:** multi-chunk streaming import, alternate-variant import/replace/
  receipt, SHA-mismatch cleanup, and the W8-v2 manifest pin live in
  `ModelStoreTests`/`SmokeTests`. No 2.23 GB pack in CI.

---

## Device stabilization (2026-08-16, branch `fix/device-stability-no-checkpoint`) — decisions (append-only)

Built on `1756ab8` (PR #17, A12 sustained-performance + reliability). These decisions
supersede the earlier experiment-era statements above wherever they conflict — in
particular "W8 checkpointing is disabled in this batch" and "Checkpointing is always
on" described the pre-stabilization branches; **checkpointing is now removed entirely**.

1. **Checkpoint/resume is removed from production** (`90ca169`). It no longer
   justifies its complexity/per-step overhead for the sub-100 s target.
   `Checkpoint.swift`/`CheckpointStore.swift` are deleted; there is no Resume UI,
   no cold-launch resume, no checkpoint identity, and no per-step latent CPU snapshot
   (`readFloats(latent...)` is gone from the production path — the 256 KiB fp32
   readback no longer runs every diffusion step). Background/memory warning simply
   cooperatively cancels.
2. **W8-v2 production temporarily uses stabilized legacy numerics** (`82026cc`).
   `DiTNumericsPolicy` is `w4Legacy` / `w8LegacyStabilized` / `w8BF16Experimental`;
   `fromVariantID("w8-v2")` → `w8LegacyStabilized` → legacy attention/activation
   numerics (the mode behind the coherent full-inference CI output). The FP16-backed
   BF16 emulation remains experimental, is not claimed range-safe internally, and is
   only reachable via explicit diagnostics.
3. **Full-inference CI derives its numerical policy from the production resolver**
   (`27d3eb7`). `testCanonicalProductionInference` builds the sampler via
   `DiTNumericsPolicy.fromVariantID(ANIMAXS_DIT_VARIANT)` →
   `DiffusionSampler.resolvedNumerics` (same as `GenerationEngine`), prints
   `FULL_DIT_NUMERICS_POLICY`, and `full-inference-refine.yml` gates each matrix
   variant on its expected marker (w4-v2 → `w4Legacy`, w8-v2 → `w8LegacyStabilized`).
4. **P8 `directQuantized`/`hybrid` is quarantined from production/device presets**
   (`e536bd7`). Physical A12 measurement showed approximately a 10× slowdown versus
   `dequantizedMPS` (performance regression, not proven incorrect). Device settings
   migrate/reject it; `directQGEMMCandidate`/`allCandidate` force `dequantizedMPS`;
   Diagnostics hides the backends/presets with a visible A12 note. The kernel remains
   research code.
5. **P6 mmap no-copy is disabled** (`b14b88b`) after a physical A12 GPU page fault
   (`kIOGPUCommandBufferCallback` ErrorPageFault). `noCopyWeightSource` cannot be true
   in production/device settings (persisted true migrates to false; presets force
   false). Not re-enabled without a future GPU-read hardware proof.
6. **Fatal Metal command-buffer faults poison the process generation context**
   (`b14b88b`). `metalContextPoisoned` is set on `.pageFault`/`.invalidResource`/
   `.internal` (or a narrow IOGPU page-fault text fallback); state fails with "Fatal
   GPU fault. Restart AnimaXS before generating again.", `isMetalAvailable` goes false,
   Generate is blocked until restart. No context recreate, no retry. Cooperative
   cancellation and ordinary failures never poison.
7. **Manual Import is local-only** (`f64eb5c`). Action buttons in model rows are
   `.borderless` so tapping Import can never cross-trigger Download; the picker target
   is cleared immediately on completion; `ModelStore` has a true per-component
   single-flight guard (`activeOperations`) for download/importPack/repair/verifyExisting
   (same-component overlap fails with "operation already active"). App launch and model
   discovery never auto-download packs.
8. **Prompt/seed persist across relaunch** (`a4a1c60`) via `generation.lastPrompt` /
   `generation.lastSeed` (`@AppStorage`), including Randomize.
9. **Fresh Generate owns the image/metrics surface** (`b25b845`); **Diagnostics preset
   marker cannot lie after manual edits, with a central compatibility validator**
   (`f647908`, blocks P6 no-copy, non-`dequantizedMPS` backends, and experimental BF16 +
   strided attention based on the RESOLVED policy); **numerical-monitor gate/add probe
   labels are disambiguated** (`605300f`).

**Physical XS Max validation after this patch is PENDING user test.** CI green on
simulator/macOS does not prove real-device success. See the stabilization plan §17
configuration and DEVICE_TESTS.md.
