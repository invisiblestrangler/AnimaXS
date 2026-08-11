# TEST_MATRIX — AnimaXS

Tracks each required test, its location, whether it genuinely ran or was skipped, and the environment.

Legend: `RAN` = actually executed · `SKIP` = skipped with explicit reason · `N/A` = not applicable
Env: `CI-sim` = GitHub Actions standard `macos-15` arm64 iOS Simulator · `CI-dev` = generic device build · `A12` = physical iPhone XS Max (PENDING)

## Pure Swift / CPU (run everywhere)
| ID | Test | Location | Status | Env |
|----|------|----------|--------|-----|
| D001 | ANMA header parse | AnimaXSTests | RAN (main CI `31436850938`) | CI-sim |
| D001 | JSON full-name/shape resolution | AnimaXSTests | RAN (main CI `31436850938`) | CI-sim |
| D001 | range/alignment validation | AnimaXSTests | RAN (main CI `31436850938`) | CI-sim |
| D002 | CRC32 per-tensor | AnimaXSTests | RAN synthetic; real-pack cases explicitly skipped in normal CI | CI-sim |
| D003 | W4 nibble order + row-aware matrix dequant reference | AnimaXSTests | RAN (`31452206651`) | CI-sim |
| D004 | W8 dequant + row-aware matrix reference | AnimaXSTests | RAN (`31452206651`) | CI-sim |
| D001 | fp16 tensor reader | AnimaXSTests | RAN (main CI `31436850938`) | CI-sim |
| D007 | block-range lookup 0…27 | AnimaXSTests | pending | CI-sim |
| D008 | Qwen embedding/layer range lookup 0…27 | AnimaXSTests | pending | CI-sim |
| I001 | sampler vector | AnimaXSTests | pending | CI-sim |
| I001 | sigma constants | AnimaXSTests | pending | CI-sim |
| I004 | checkpoint serialization | AnimaXSTests | pending | CI-sim |
| D005 | SHA manifest | AnimaXSTests | pending | CI-sim |
| F004 | tokenizers exact on canonical prompts | AnimaXSTests (TokenizerParityTests) | RAN (main CI `31436850938`) | CI-sim |
| D018 | GQA head mapping table (16/8 → h/2) | AnimaXSTests (GqaHeadMappingTests) | RAN (main CI `31436850938`) | CI-sim |
| D018 | synthetic 1-token attention reveals KV head | AnimaXSTests (GqaHeadMappingTests) | RAN (main CI `31436850938`) | CI-sim |
| D021 | interleaved RoPE (rotate_half) identity/pos | AnimaXSTests (LLMAdapterTests) | RAN (main CI `31436850938`) | CI-sim |
| D021 | exact GELU known values | AnimaXSTests (LLMAdapterTests) | RAN (main CI `31436850938`) | CI-sim |
| H003 | CPU modulation reference (LayerNorm, SiLU) | AnimaXSTests (ModulationTests) | RAN (main CI `31436850938`) | CI-sim |
| H004 | CPU RoPE reference (config, thetas, t/h/w order, 2×2 layout) | AnimaXSTests (DitRoPETests) | RAN (main CI `31436850938`) | CI-sim |
| H005 | DiT block primitives (split-half RoPE, LayerNorm/AdaLN, RMSNorm, attention, concat, GELU) | AnimaXSTests (DiTBlockTests) | RAN: 9/9 PASS (`31452206651`) | CI-sim |
| J003 | GroupNorm CPU reference | AnimaXSTests | pending | CI-sim |
| H001 | DiT input patchify ordering + token count (1024×68) + inChannels 17 | AnimaXSTests (DiTInputTests) | RAN (main CI `31436850938`) | CI-sim |
| H002 | Timestep sinusoidal vs torch reference + SiLU | AnimaXSTests (TimestepEmbedderTests) | RAN (main CI `31436850938`) | CI-sim |

## Metal (where a Metal device exists)
| ID | Test | Status | Env |
|----|------|--------|-----|
| E001 | hosted Metal device + diagnostics + project-kernel execution | PASS: `Apple iOS simulator GPU`, probes populated, run `31455204105` | CI-sim |
| E001/E006 | 2×2 fp16 `MPSMatrixMultiplication` execution | PASS, permanent test run `31452206651` | CI-sim |
| E002 | row-aware W4/W8 dequant + padded bounds | PASS exact-after-fp16 for W4 `[2,68]`, W8 `[2,65]` (`31455204105`) | CI-sim |
| E003 | fp32 norms/activations/modulation/gate/residual kernels | PASS CPU parity and padded bounds (`31478699877`) | CI-sim |
| E004 | split-half RoPE + patchify/unpatchify + Euler | PASS CPU parity, round-trip and padded bounds (`31478699877`) | CI-sim |
| E005 | direct packed W4 fp32 matvec, K=68 | PASS maxAbs `2.09e-6`, cosine `0.9999999999999986` vs fp64 (`31478699877`) | CI-sim |
| D007/D008 | zero-copy DiT/Qwen logical ranges and embedding rows | PASS synthetic (`31455204105`); local real packs DiT 28/560, Qwen 28/308, embedding 151936 rows | CI-sim/pack |
| E002 | W8 dequant GPU | pending | CI-sim/A12 |
| E003 | rmsnorm/layernorm GPU | pending | CI-sim/A12 |
| E003 | GELU/SiLU GPU | pending | CI-sim/A12 |
| E003 | gate-add GPU | pending | CI-sim/A12 |
| E006 | MPS linear GEMM | PASS W4/W8 transpose, packed stride, offsets, M=133 tail (`31482950188`) | CI-sim; A12 pending |
| E008 | attention tile | PASS self Q/K=1024 and cross K=512 including row 511 (`31482950188`) | CI-sim; A12 pending |
| E004 | euler_step_f32 GPU | pending | CI-sim/A12 |
| E007 | MPS precision K=2048/8192 | PASS; MPS cosine ≥ `0.999999975`, maxAbs ≤ `1.76e-4` (`31482950188`) | CI-sim; A12 repeat pending |
| E009 | production Metal block 0 vs H005 same-W4 oracle | pending | CI-sim/manual pack-backed/A12 |

## Model integration
| ID | Test | Status | Env |
|----|------|--------|-----|
| F005 | TE final context ≈ golden (cosine 0.992, structural 1.0 vs oracle) | RAN (harness, real pack) | local |
| F007 | streamed Metal TE ≈ F005 same-W8 oracle | pending | manual pack-backed CI/A12 |
| G001 | adapter conditioning [1,512,1024] ≈ oracle (cosine 1.000000) | RAN (harness, real pack) | local |
| G001 | adapter output finite + padded tail zero | RAN (harness, real pack) | local |
| G003 | streamed Metal adapter ≈ G001 same-W4 oracle | pending | manual pack-backed CI/A12 |
| H001 | DiT input x_embedder [1024,2048] ≈ corrected row-aware oracle (cosine 1.000000, maxAbs 3.58e-7) | RAN (harness, real pack) | local |
| H002 | timestep embedding + adaln [6144] ≈ oracle (cosine 1.000000, all finite) | RAN (harness, real pack) | local |
| H003 | AdaLN shift/scale/gate all branches ≈ oracle (cosine 1.000000, all finite) | RAN (harness, real pack) | local |
| H004 | DiT 3-D RoPE [1024,64,2,2] ≈ oracle (cosine 1.000000, maxAbs 4.8e-7) | RAN (harness, no pack needed) | local |
| H005-decoder | W4 2×68 + W8 2×65 matrix groups reset at each row | PASS locally and in XCTest (`31452206651`) | local/CI-sim |
| H005 | block 0 Swift≈NumPy W4 oracle (final golden-hook invocation) | PASS locally: cosine 1.000000000, RMSE 6.18e-6, maxAbs 1.91e-4, finite | local/CI-sim |
| H005-B | block 0 W4 vs original BF16 golden | ACCEPTED source-proven quantization baseline: Swift 0.998712139; original BF16 source 0.999992303 (D035) | local |
| H006 | 28 blocks finite | pending | CI-sim/A12 |
| I002 | 8 step latents finite/≈ step_latents | pending | CI-sim/A12 |
| J002 | VAE RGB ≈ decoded_rgb | pending | CI-sim/A12 |
| L001 | full canonical final latent | pending | CI-sim/A12 |

## Build gates
| Gate | Status | Env |
|------|--------|-----|
| Xcode 26.3 generic iOS build PASS | RAN (final snapshot `31452206651`) | CI-dev |
| deployment target 18.0 | RAN (static) | — |
| Metal shaders compile | RAN (`31452206651`) | CI-dev/CI-sim |
| simulator unit tests PASS | 61 tests, 4 expected real-pack skips, 0 failures (`31455204105`) | CI-sim |
| xcodegen 2.46.0 checksum + clean regeneration | PASS (`31452206651`) | CI-sim |
