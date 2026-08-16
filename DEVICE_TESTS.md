# DEVICE_TESTS — AnimaXS A12 / physical-device record

> This file records results ONLY from a physical iPhone XS Max (A12 / Apple5) or, where explicitly noted,
> a GitHub Actions M1 simulator for CI-only characterization. Nothing here counts as A12 validation unless
> it was produced on the actual device.

## A12 DEVICE ACCEPTANCE: **PENDING** (physical-device runs 2026-08-15 observed; stabilization patch 2026-08-16 NOT yet device-tested)

A physical iPhone XS Max **has** been used in this project (first real-device run, 2026-08-15).
Only the facts below were observed on it. Everything else on this page remains pending until the
retest checklist at the bottom is executed against the stabilization build
(`fix/device-stability-no-checkpoint`, HEAD 605300f). The device-stabilization patch
(checkpoint removed, W8 legacy numerics, P8/P6 quarantined/disabled, fatal-Metal poisoning,
Import local-only, prompt/seed persistence) is CI-verified on simulator/macOS only —
**CI green does not prove real-device success**.

## Observed on the physical iPhone XS Max (2026-08-15)

| # | Observation | Status |
|---|-------------|--------|
| 1 | App builds, installs, and launches on the physical device | observed |
| 2 | All three manual `.animapk` imports initially failed with a Files permission error ("couldn't be opened because you don't have permission…") | observed |
| 3 | Explicit model Download produced a generic `AnimaXS.AnimapkError error 3` message (root cause of that specific Download failure not yet determined) | observed |
| 4 | Prompt keyboard had no obvious dismissal control | observed |
| 5 | Tapping Generate could leave the status at `Ready.` with no visible error | observed |
| 6 | Diagnostics crashed the app on the physical device (crash root cause NOT yet proven — Metal/MPS assertion, jetsam, or other) | observed (2026-08-15) |
| 7 | With all three packs installed, the app became slightly warm after launch while idle | observed (2026-08-15) |
| 8 | W8-v2 manual import (2.233 GB) terminated the app during import (2026-08-15) — addressed by the single-pass streaming import (PR #15), now in the normal path; NOT yet re-verified on device | observed (2026-08-15) |
| 9 | P6 mmap no-copy weight serving produced a real GPU page fault (`kIOGPUCommandBufferCallback` ErrorPageFault, A12) — P6 is now DISABLED in production/device settings; no re-enable without a hardware proof | observed (2026-08-16) |
| 10 | P8 direct-quantized GEMM measured approximately 10× slower than `dequantizedMPS` on A12 — P8 `directQuantized`/`hybrid` are now QUARANTINED from production/device presets | observed (2026-08-16) |

**Do not state why Diagnostics crashed until evidence proves it.** The repository-side fixes
(MPS rowBytes alignment via the MPS helper, command-buffer status inspection, crash-localizing
diagnostic marker, no auto-run, no launch-time re-hash) are implemented but the physical A12
crash status is **pending retest**.

## Hosted simulator characterization (not A12)

| Check | Result | Run |
|-------|--------|-----|
| `MTLCreateSystemDefaultDevice()` | `Apple iOS simulator GPU` | `31452206651` |
| project W4 Metal kernel execution | PASS | `31452206651` |
| 2×2 fp16 `MPSMatrixMultiplication` | PASS | `31452206651` |
| MPS precision diagnostic (corrected 64-byte-aligned rowBytes) | PASS (simulator) | stabilization CI |
| Diagnostics cheap snapshot (zero hashing) | PASS (simulator) | stabilization CI |

Use standard hosted Simulator CI for functional Metal/MPS parity. Do not use these results for A12 timing, memory, GPU-family, watchdog, page-cache, or thermal claims.

## Remains unverified (to be filled on-device)

- [ ] MPS fp16 accuracy on Apple5 (K=2048/8192)
- [ ] A12 memory / jetsam threshold
- [ ] A12 performance (per-stage timing)
- [ ] A12 command-buffer watchdog limit
- [ ] A12 sustained thermal behavior
- [ ] A12 page-cache behavior (mmap/copy MB/s)
- [ ] Diagnostics hardware tests do not crash the app on A12
- [ ] Idle thermal behavior after launch with all packs installed
- [ ] W8-v2 import completes on device (single-pass streaming import)
- [ ] P6 no-copy stays unavailable/off on device (post-page-fault)
- [ ] P8 direct/hybrid cannot be selected on device (quarantined)
- [ ] Prompt/seed persist across relaunch on device
- [ ] Fatal-GPU-fault poisoning behaves (restart required) if reproduced on device

## Physical retest checklist (stabilization build)

- [ ] Clean launch starts no downloads
- [ ] Clean launch performs no redundant full-pack SHA passes
- [ ] Idle app does not retain all pack mappings
- [ ] Idle thermal state remains reasonable after launch
- [ ] Qwen3 manual Import succeeds
- [ ] DiT manual Import succeeds
- [ ] VAE manual Import succeeds
- [ ] W8-v2 manual Import succeeds (single-pass streaming; run 1 of the device matrix is W4, run G is W8)
- [ ] Relaunch rediscovers imported packs quickly
- [ ] Prompt and seed persist across relaunch (incl. Randomize)
- [ ] Keyboard can be dismissed
- [ ] Generate either starts or displays exact blocked reason
- [ ] No Resume / Discard-checkpoint UI exists (checkpointing removed)
- [ ] P6 no-copy control is unavailable/off; P8 direct/hybrid cannot be selected
- [ ] Generation reaches Tokenizing
- [ ] Generation reaches Encoding prompt
- [ ] Generation reaches Adapter
- [ ] Generation reaches Diffusion
- [ ] Generation reaches VAE
- [ ] Generation displays image
- [ ] Diagnostics screen can be opened safely
- [ ] Diagnostics does not auto-run
- [ ] Basic diagnostics completes
- [ ] Hardware diagnostics identifies per-test progress
- [ ] Deep SHA verification is explicit only
- [ ] Exporting JSON does not rerun tests
- [ ] Explicit Download works or shows precise useful failure

## Microbenchmark template (run via Diagnostics screen on device)

| Benchmark | Config | Result | Units | Date |
|-----------|--------|--------|-------|------|
| mmap/copy | 38,993,920 B DiT block, cold | — | MB/s | — |
| mmap/copy | 38,993,920 B DiT block, warm | — | MB/s | — |
| W4 dequant | 8192×2048 | — | ms | — |
| MPS GEMM | M1024 N2048 K2048 | — | ms | — |
| MPS GEMM | M1024 N8192 K2048 | — | ms | — |
| MPS GEMM | M1024 N2048 K8192 | — | ms | — |
| MPS GEMM tiled | M=128 ×8 | — | ms | — |
| Attention tile | 16h, q128, k1024, d128 | — | ms | — |

## Memory record (os_proc_available_memory / jetsam margin)

| Point | Available | Date |
|-------|-----------|------|
| idle | — | — |
| after mapping | — | — |
| after ring alloc | — | — |
| inside DiT | — | — |
| before VAE | — | — |
| VAE peak | — | — |

## Thermal (one 8-step generation)

| Metric | Result |
|--------|--------|
| total time | — |
| thermal state during | — |
| thermal state after | — |

---

## Runtime inference-performance experiments (D207) — device runbook

The app ships ONE binary with runtime-selected experiment controls in
Diagnostics → "Inference performance experiments". Every generation records its
immutable configuration and environment in the final metrics summary, so runs
under materially different conditions are discarded rather than misinterpreted.

### Rules for valid comparisons

- Use **fresh Generate** for all benchmark timings. (Checkpoint/Resume was removed from
  production in the 2026-08-16 stabilization patch, so there is no Resume path anymore.)
- Keep the iPhone **unplugged** for authoritative performance tests.
- Same prompt, seed, image dimensions, step count, text encoder, VAE, binary.
- Check the summary's power/thermal start/end lines: discard runs where one is
  charging or thermal start state differs materially.
- The app does NOT auto-run a benchmark suite — running seven generations
  back-to-back would introduce severe thermal-order bias. Run each config
  individually from the same installed binary.

### Controls (all runtime; one build)

- Linear tile rows: 128 | 256 | 512 | 1024
- Attention tile rows: 128 | 256 | 512 | 1024
- Direct MPS linear I/O: on/off (falls back to copies on stride mismatch; the
  summary counters report direct vs copied tiles)
- Ping-pong weight streaming: on/off (existing two-slot optimization vs the
  1-slot synchronous control)
- Numerical monitor: on/off (Euler finite guard stays on either way)
- DiT pack: Production W4 | W8-v2 (W8 is a normal alternate of the `.dit` slot;
  checkpointing is REMOVED for all packs — the W8 "checkpointing disabled" caveat
  below no longer applies)
- Linear backend: dequantizedMPS (P8 `directQuantized`/hybrid are QUARANTINED and
  cannot be selected for device runs)
- Mmap no-copy: unavailable/off (P6 disabled after the A12 GPU page fault)
- Reset to current baseline

### Seven-run minimum matrix

| Run | DiT | Linear tile | Attention tile | Direct I/O | Ping-pong | Monitor |
|-----|-----|-------------|----------------|------------|-----------|---------|
| A | W4 | 128 | 128 | off | on | on |
| B | W4 | 1024 | 128 | off | on | on |
| C | W4 | 1024 | 1024 | off | on | on |
| D | W4 | best | best | on | on | on |
| E | W4 | best | best | best | on | off |
| F | W4 | best | best | best | off | best |
| G | W8 v2 | best | best | best | best | best |

Conditional: if 1024 regresses vs baseline, try Linear 512 / Attention 512 with
the already-built controls (no rebuild). If 1024 wins strongly, 256/512 need
not be exhaustively tested.

### Fields to copy from each final generation summary

```
Generation: N s          DiT: N s          Average block wall time: N s
Measured GPU command time: N s             Metal encode time: N s
Weight copy/load CPU work: N s, N MB (may overlap GPU time when ping-pong is on)
Linear GEMM tiles: N      Linear input tiles: X direct / Y copied
Linear output tiles: X direct / Y copied   Attention query tiles: N
Peak Metal allocation: N GB                Minimum available process memory: N MB
Numerical warnings: N | not collected
Inference configuration block (pack, tiles, direct I/O, ping-pong, monitor, linear backend)
Environment block (Power / Battery / Thermal / Low Power Mode start -> end)
```
