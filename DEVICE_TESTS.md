# DEVICE_TESTS — AnimaXS A12 / physical-device record

> This file records results ONLY from a physical iPhone XS Max (A12 / Apple5) or, where explicitly noted,
> a GitHub Actions M1 simulator for CI-only characterization. Nothing here counts as A12 validation unless
> it was produced on the actual device.

## A12 DEVICE ACCEPTANCE: **PENDING**

No physical iPhone XS Max has been used in this project. Do not read any CI/M1 result as A12 proof.

## Hosted simulator characterization (not A12)

| Check | Result | Run |
|-------|--------|-----|
| `MTLCreateSystemDefaultDevice()` | `Apple iOS simulator GPU` | `31452206651` |
| project W4 Metal kernel execution | PASS | `31452206651` |
| 2×2 fp16 `MPSMatrixMultiplication` | PASS | `31452206651` |

Use standard hosted Simulator CI for functional Metal/MPS parity. Do not use these results for A12 timing, memory, GPU-family, watchdog, page-cache, or thermal claims.

## Remains unverified (to be filled on-device)
- [ ] MPS fp16 accuracy on Apple5 (K=2048/8192)
- [ ] A12 memory / jetsam threshold
- [ ] A12 performance (per-stage timing)
- [ ] A12 command-buffer watchdog limit
- [ ] A12 sustained thermal behavior
- [ ] A12 page-cache behavior (mmap/copy MB/s)

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
