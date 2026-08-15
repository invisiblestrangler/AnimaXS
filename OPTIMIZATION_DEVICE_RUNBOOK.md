# AnimaXS Optimization Phase — Device Runbook (unplugged iPhone XS Max)

Branch: `opt/reliability-telemetry` (PR #11). Build and install this branch's
build on the physical XS Max. Everything below is done **unplugged** — charging
materially heats this device and roughly doubles inference time, so tethered
wall-clock numbers are not the benchmark.

## What changed (what you are validating)

1. **Numerical failure reporting (Phase 1).** Error strings interpolate
   correctly and use 1-based human indexing. A failure now reads like:
   `Numerical failure at diffusion step 5/8, block 17/28, cross-attention output: Inf detected.`
2. **Numerical-health probes (Phase 3).** Every FP16 boundary (projection
   inputs, MLP hidden, final-layer conversions, attention scores, gated branch
   adds) records NaN/±Inf/overflow + max magnitude on the GPU, always-on with
   negligible cost. MPS-output probes exist behind the detailed flag (used by
   the CI stress test).
3. **In-app telemetry (Phase 7/8).** After every generation (success, failure,
   or cancel) a "Run metrics" section appears: total/stage times, average block
   time, measured GPU command time, weight-copy time, host wait time, peak Metal
   allocation, minimum available memory, numerical warnings with per-boundary
   details. Shareable via the share sheet. Also visible in Diagnostics → "Last
   generation metrics".
4. **Two-slot ping-pong weight streaming (Phase 12).** Block weights for block
   N+1 are memcpy'd into the second slot while block N executes on the GPU,
   hiding CPU copy behind GPU work. Outputs must be identical (validated by CI
   golden-case cosine ≥ 0.65). Peak Metal allocation should rise by ~39 MB.

## Baseline procedure (runbook Phase 10) — BEFORE optimizing further

1. Install the build. Launch, import/verify the three packs (or let them
   download) so model rows are `ready`.
2. **Unplug the phone.** Let it idle 2 minutes.
3. Use the default prompt and seed 1337. Tap Generate. Do NOT touch the phone
   until the image appears.
4. Record from the "Run metrics" section (or share the text):
   - total time; text encode; DiT; VAE; other;
   - average block wall time; measured GPU command time; weight copy time;
     host wait time;
   - peak Metal allocation; minimum available process memory;
   - numerical warnings + details.
5. Repeat with 3–4 different seeds/prompts (Randomize). Record each. These are
   the baseline runs (multiple, because the old phone varies run to run).

## Numerical stress (runbook Phase 4) — on-device

Generate repeatedly with Randomize over ordinary prompts. The app now tells you
when a boundary went unsafe:
- Successful runs: "Numerical warnings: N (…details…)" in Run metrics.
- A failure: the error message attributes step/block/stage.

The CI stress (full-inference workflow, `FULL_STRESS_*` markers) gives the
simulator distribution; the phone gives the A12 distribution. If any failures
occur, send the exact error message + the metrics text.

## Performance comparison (runbook Phase 13)

The two-slot streamer is already active in this build. To compare against the
one-slot baseline, build the previous commit (`d371f7b` or PR #11 parent) and
repeat the baseline procedure with the same seeds/prompts. Look at:
- total time and DiT time (must go DOWN or stay equal);
- weight copy time (should now overlap GPU time — watch host wait time);
- peak Metal allocation (expect ≈ +39 MB);
- image correctness (visually the same).

## What NOT to do

- Do not run while plugged in and treat the time as a benchmark.
- Do not expect a huge speedup: the copy time is only part of the block time,
  and the phone's clocks vary. Evidence decides; if total time regresses,
  we revert the streamer.
- Do not collect "cool phone" or thermal-mode data — no thermal logic exists.

## Reporting

Share the metrics text (share button) or screenshot it. Include: prompt, seed,
whether plugged/unplugged, and the metrics block. I'll fold it into the phase
report and the HF evidence repo.
