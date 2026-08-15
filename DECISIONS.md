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
