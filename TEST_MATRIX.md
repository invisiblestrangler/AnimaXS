# TEST_MATRIX — AnimaXS

Tracks each required test, its location, whether it genuinely ran or was skipped, and the environment.

Legend: `RAN` = actually executed · `SKIP` = skipped with explicit reason · `N/A` = not applicable
Env: `CI-sim` = GitHub Actions simulator · `CI-dev` = generic device build · `A12` = physical iPhone XS Max (PENDING)

## Pure Swift / CPU (run everywhere)
| ID | Test | Location | Status | Env |
|----|------|----------|--------|-----|
| D001 | ANMA header parse | AnimaXSTests | pending | CI-sim |
| D001 | JSON full-name/shape resolution | AnimaXSTests | pending | CI-sim |
| D001 | range/alignment validation | AnimaXSTests | pending | CI-sim |
| D002 | CRC32 per-tensor | AnimaXSTests | pending | CI-sim |
| D003 | W4 nibble order + dequant reference | AnimaXSTests | pending | CI-sim |
| D004 | W8 dequant reference | AnimaXSTests | pending | CI-sim |
| D001 | fp16 tensor reader | AnimaXSTests | pending | CI-sim |
| D007 | block-range lookup 0…27 | AnimaXSTests | pending | CI-sim |
| I001 | sampler vector | AnimaXSTests | pending | CI-sim |
| I001 | sigma constants | AnimaXSTests | pending | CI-sim |
| I004 | checkpoint serialization | AnimaXSTests | pending | CI-sim |
| D005 | SHA manifest | AnimaXSTests | pending | CI-sim |
| F004 | tokenizers exact on canonical prompts | AnimaXSTests | pending | CI-sim |
| H002 | CPU timestep reference | AnimaXSTests | pending | CI-sim |
| H003 | CPU modulation reference | AnimaXSTests | pending | CI-sim |
| H004 | CPU RoPE reference | AnimaXSTests | pending | CI-sim |
| J003 | GroupNorm CPU reference | AnimaXSTests | pending | CI-sim |

## Metal (where a Metal device exists)
| ID | Test | Status | Env |
|----|------|--------|-----|
| E002 | W4 dequant GPU | pending | CI-sim/A12 |
| E002 | W8 dequant GPU | pending | CI-sim/A12 |
| E003 | rmsnorm/layernorm GPU | pending | CI-sim/A12 |
| E003 | GELU/SiLU GPU | pending | CI-sim/A12 |
| E003 | gate-add GPU | pending | CI-sim/A12 |
| E006 | MPS linear GEMM | pending | CI-sim/A12 |
| E008 | attention tile | pending | CI-sim/A12 |
| E004 | euler_step_f32 GPU | pending | CI-sim/A12 |
| E007 | MPS precision K=2048/8192 | pending | CI-sim/A12 |

## Model integration
| ID | Test | Status | Env |
|----|------|--------|-----|
| F005 | TE final context ≈ golden | pending | CI-sim/A12 |
| G001 | adapter context finite/≈ ref | pending | CI-sim/A12 |
| H005 | block 0 ≈ golden | pending | CI-sim/A12 |
| H006 | 28 blocks finite | pending | CI-sim/A12 |
| I002 | 8 step latents finite/≈ step_latents | pending | CI-sim/A12 |
| J002 | VAE RGB ≈ decoded_rgb | pending | CI-sim/A12 |
| L001 | full canonical final latent | pending | CI-sim/A12 |

## Build gates
| Gate | Status | Env |
|------|--------|-----|
| Xcode 26.3 generic iOS build PASS | pending | CI-dev |
| deployment target 18.0 | pending (static) | — |
| Metal shaders compile | pending | CI-dev |
| simulator unit tests PASS | pending | CI-sim |
| xcodegen regenerates clean (git diff empty) | pending | CI-sim |
