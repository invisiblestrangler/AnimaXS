# AnimaXS Optimization Phase — Reliability + Telemetry + Performance (PR #11)

Date: 2026-08-15
Branch: `opt/reliability-telemetry`
PR: invisiblestrangler/AnimaXS#11
Base: `main` @ `aa87947`
HEAD: `74ac2d2` (clean, pushed)

## Summary of the phase

Three priorities from the runbook, in order: **reliability** (harden the
intermittent non-finite diffusion latent failures), **observability** (in-app
telemetry so the unplugged device is the benchmark), **performance** (two-slot
ping-pong weight streaming). No thermal logic, no silent clamping, no global
FP32. All CI green; golden-case correctness preserved.

## Repository state

- Branch: `opt/reliability-telemetry`
- HEAD: `74ac2d27e5537cdcaed6d5226794d3ab8e1fbcc0`
- Commits added: 13 (D200–D205 + regen/CI fixes)
- Working tree: clean

## Reliability (Phase 1, 2, 3, 5)

### D200 — Numerical failure reporting
- Fixed all six literal `\(...)` interpolation sites (DiffusionSampler,
  CheckpointStore, GenerationEngine) that printed a literal `step \(step)`.
- New `NumericalFailure` (Error + LocalizedError) builds a 1-based, attributed
  message: `Numerical failure at diffusion step 5/8, block 17/28, cross-attention output: Inf detected.`
- Unit-tested (NumericalFailureTests).

### D201/D204 — Numerical-health instrumentation
- `NumericalMonitor` owns a small GPU stats buffer (uint32 quad per probe slot,
  relaxed atomics). Probe kernels (always-on, ~zero cost since they already
  touch every element):
  - `float_to_half_probe` (projection inputs, MLP hidden, final-layer
    conversions, cross-context): records NaN/±Inf/FP16-overflow + maxAbs +
    firstIndex.
  - `gate_add_half_f32_probe`: records branch health + residual poisoning.
  - `attention_softmax_rows(_f32)_probe`: records score health.
  - Standalone `probe_f16_stats`/`probe_f32_stats` passes (MPS outputs +
    velocity/denoised/Euler), gated behind `NumericalMonitor.detailedProbesEnabled`.
- Attribution: per-block cheap CPU reads of the 512-byte stats buffer pin the
  FIRST unsafe (step, block, stage, condition).
- FP32 residual probes treat halfOverflow as informational (the residual is
  FP32 by design), so only true NaN/Inf/poisoning warn/attribute.
- Stress harness `FullInferenceTests.testNumericalStressAcrossSeeds` runs the
  production sampler over N ordinary seeds with detailed probes and prints the
  aggregate evidence.

### D205 — CI-found crash (reliability of the phase itself)
- First full-inference run CRASHED the test host:
  `addCompletedHandler after commit call` (Metal hard assertion) — the
  ping-pong restructure registered the handler after commit.
- Fix: `CommandBufferGate` registers the handler BEFORE commit; prefetch runs
  while GPU executes; `wait()` resumes via stored continuation (both orderings
  safe). Regression tests (CommandBufferGateTests).

## Telemetry (Phase 7, 8)

- `MetricsCollector`/`GenerationMetrics`: total wall + stage times (text
  encode, adapter, DiT, VAE, other), per-step + per-block wall, weight copy
  time+bytes, Metal encode time, measured GPU command time (MTLCommandBuffer
  GPU timestamps, guarded), host wait time (reconciles with wall), peak/current
  Metal allocation, min available process memory, thermal state, numerical
  warning count + per-probe details.
- Wired through GenerationEngine → DiffusionSampler → preparation/forward/
  final-layer/euler/DiTBlockExecutor.
- Post-run compact summary in ContentView "Run metrics" + DiagnosticsView
  "Last generation metrics", shareable.
- Overhead: negligible — stage/step/block timing is wall-clock arithmetic;
  memory sampled once per block; probes are in-kernel.

## Performance (Phase 11, 12)

- Measured the one-slot serialization: per block, the CPU memcpy (≈39 MB) ran
  while the GPU was idle (block submit → await → next copy).
- Implemented a bounded **two-slot ping-pong weight streamer** (`WeightStreamer`
  v2, exactly 2 slots in the DiT path): block N+1's weights are copied into the
  other slot while block N executes on the GPU. Hard in-flight guard prevents
  overwriting a slot a committed command buffer references. Outputs are
  byte-identical to the one-slot pattern.
- Memory headroom verified: +1 slot ≈ +39 MB vs ~1.4 GB peak.
- Correctness preserved: golden-case latent cos 0.866 / RGB cos 0.8198
  (FULL_INFERENCE=PASS), unchanged from the pre-streamer baseline.

## Evidence (CI)

### CI run (PR #11, latest green)
- project-consistency PASS
- iphone-build PASS
- simulator-tests PASS (incl. NumericalFailureTests, NumericalMonitorTests,
  GenerationMetricsTests, CommandBufferGateTests, WeightStreamerTests)

### full-inference run 31863687670 (golden case + stress)
- FULL_INFERENCE=PASS; latent cos 0.866, RGB cos 0.8198 (regression floor 0.65)
- FULL_TOTAL_SECONDS=88.53 (macOS simulator, not device)
- FULL_DIFFUSION_SECONDS=73.08, FULL_VAE_SECONDS=7.50, FULL_QWEN_SECONDS=3.62

### Numerical stress (4 seeds, golden conditioning, detailed probes)
- FULL_STRESS_TOTAL=4 SUCCESS=4 FAILURES=0
- Max magnitudes at boundaries (representative):
  - self-attention residual 281774, cross-attention residual 281748,
    MLP residual 281935 (fp32 by design)
  - cross-attention projection input 751, cross-attention scores 9.38,
    self-attention Q projection 356
  - MLP hidden conversion 238, MLP output 731
  - final-layer residual conversion 43102 (this is the FP16 boundary feeding
    final-layer LayerNorm — the largest FP32→FP16 crossing observed)
  - velocity conversion 5.74, Euler update 4.74, FLOW denoised 2.63
- No NaN/Inf was observed at any boundary across all 4 seeds; all 4 generations
  finished finite. No first-issue attribution fired (informational overflow
  only), consistent with the intermittent nature of the original failures.

## Remaining opportunities (evidence-based)

- The **final-layer residual conversion (maxAbs 43102) is the closest FP32→FP16
  crossing to the FP16 overflow threshold (65504)**. This is a candidate for the
  intermittent non-finite failures: if a particular generation's residual peaks
  cross 65504 at this boundary, `half()` yields ±Inf and the fp32 residual is
  poisoned. This boundary is a prime target for selective FP32 storage /
  splitting, pending on-device reproduction. Not changed yet — the runbook
  requires evidence, and CI/simulator did not reproduce a failure.
- **Per-block copy/GPU/wait decomposition** on the unplugged device will show
  whether the ping-pong hides the memcpy fully; if host wait still dominates,
  deeper scheduling work is justified.
- Weight-streaming copy time should now overlap GPU time; on-device
  measurement will confirm.

## What still requires the physical iPhone XS Max (unplugged)

1. Baseline runs (3–4 seeds/prompts) → "Run metrics" per run.
2. Stress: Randomize over ordinary prompts; capture any failure's attributed
   message + the metrics text.
3. Ping-pong comparison: build parent commit, repeat same seeds, compare total
   time, DiT time, weight-copy time, peak memory, and image correctness.
4. Report: share the metrics text (or screenshot) with prompt/seed + unplugged
   status. See OPTIMIZATION_DEVICE_RUNBOOK.md.

## Guardrails honored

- No thermal control logic added; thermal recorded as passive metadata only.
- No silent clamping; failures surface with correct attribution.
- No global FP32; only evidence-driven, narrow hardening is eligible.
- No custom W4 GEMM rewrite (no profiling evidence it is the bottleneck).
- No "bad seed" hunt; seeds are test diversity.
- Tethered/CI time is not presented as the device benchmark.
