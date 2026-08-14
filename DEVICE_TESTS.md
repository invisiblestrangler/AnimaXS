# DEVICE_TESTS — AnimaXS A12 / physical-device record

> This file records results ONLY from a physical iPhone XS Max (A12 / Apple5) or, where explicitly noted,
> a GitHub Actions M1 simulator for CI-only characterization. Nothing here counts as A12 validation unless
> it was produced on the actual device.

## A12 DEVICE ACCEPTANCE: **PENDING** (physical-device run 2026-08-15 observed)

A physical iPhone XS Max **has** been used in this project (first real-device run, 2026-08-15).
Only the facts below were observed on it. Everything else on this page remains pending until the
retest checklist at the bottom is executed against the stabilization build.

## Observed on the physical iPhone XS Max (2026-08-15)

| # | Observation | Status |
|---|-------------|--------|
| 1 | App builds, installs, and launches on the physical device | observed |
| 2 | All three manual `.animapk` imports initially failed with a Files permission error ("couldn't be opened because you don't have permission…") | observed |
| 3 | Explicit model Download produced a generic `AnimaXS.AnimapkError error 3` message (root cause of that specific Download failure not yet determined) | observed |
| 4 | Prompt keyboard had no obvious dismissal control | observed |
| 5 | Tapping Generate could leave the status at `Ready.` with no visible error | observed |
| 6 | Diagnostics crashed the app on the physical device (crash root cause NOT yet proven — Metal/MPS assertion, jetsam, or other) | observed |
| 7 | With all three packs installed, the app became slightly warm after launch while idle | observed |

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

## Physical retest checklist (stabilization build)

- [ ] Clean launch starts no downloads
- [ ] Clean launch performs no redundant full-pack SHA passes
- [ ] Idle app does not retain all pack mappings
- [ ] Idle thermal state remains reasonable after launch
- [ ] Qwen3 manual Import succeeds
- [ ] DiT manual Import succeeds
- [ ] VAE manual Import succeeds
- [ ] Relaunch rediscovers imported packs quickly
- [ ] Keyboard can be dismissed
- [ ] Generate either starts or displays exact blocked reason
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
