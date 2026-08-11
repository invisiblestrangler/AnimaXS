# Execution Runbook — AnimaXS

## Build a complete Anima-Turbo inference app for iPhone XS Max / A12 / iOS 18

You are the implementation agent for a narrowly targeted engineering project.

Your job is to build, test, document, and push a complete iOS application that runs the supplied Anima-Turbo image-generation model locally on:

```text
Device: iPhone XS Max
SoC: Apple A12 Bionic
RAM: 4 GB
OS target: iOS 18.x
Actual user's device: iOS 18.6
GPU family: Apple5
```

The finished repository must:

* clone cleanly onto a Mac;
* open directly in Xcode 26.3;
* build with Xcode 26.3;
* have minimum deployment target iOS 18.0;
* install on an iPhone XS Max running iOS 18.6 after the user selects their signing team;
* download and verify the model packs;
* tokenize prompts;
* execute the Qwen text encoder;
* execute the LLM adapter;
* execute the 28-block Anima DiT;
* execute the 8-step Turbo sampler;
* decode the final image with the VAE;
* display the generated image;
* survive normal cancellation/memory/background conditions reasonably;
* contain useful diagnostics for A12-specific issues;
* build and test automatically using GitHub Actions;
* attempt a full model inference in GitHub Actions when the runner exposes a usable Metal device;
* clearly distinguish tests that genuinely ran from tests that were skipped because an actual A12 device is required.

Do not optimize for any other phone.

Do not build MLX, Core ML, ANE, Metal 3, `MTLIOCommandQueue`, or newer-GPU fast paths.

Do not turn this into a generic inference framework.

The goal is **one model on one old phone**.

---

# 0. Source-of-truth rule

You are being supplied a handoff bundle named approximately:

```text
PHASE0_2_HANDOFF.zip
```

Treat the files inside it as the model-specific source of truth.

The zip is already extracted at `/root/anima-xsmax/PHASE0_2_HANDOFF/`.

Before implementing each major subsystem, re-read the relevant handoff files rather than relying on memory.

Most important:

```text
HANDOFF.md
HANDOFF.json
RUNTIME_CONSTANTS.json
MODEL_ARCHITECTURE.json
ANIMAPK_SPEC.md
ANIMAPK_LAYOUT.json
NUMERICS.md
GOLDENS.md
TEST_RESULTS.md
KNOWN_ISSUES.md
IMPLEMENTATION_RECOMMENDATIONS.md
FILE_INDEX.md
relevant-small-artifacts/model_info.json
relevant-small-artifacts/inspect_animapk.py
relevant-small-artifacts/animapk_reader.*
relevant-small-artifacts/RESULTS.sha256
```

Observed implementation facts in these files override assumptions in this runbook.

If a value here disagrees with the handoff artifacts, investigate the discrepancy and use the experimentally confirmed value.

Never silently guess.

For particularly subtle operations—Qwen masks, adapter attention/mask behavior, VAE upsampling, causal convolution, RoPE, modulation—also inspect the **pinned reference source identified by the handoff** before implementing it.

Resolved discrepancies are recorded in `DECISIONS.md` and override stale handoff/runbook prose. In particular: Qwen output includes the final RMSNorm (D019), AdaLN applies SiLU before its first linear (D026), DiT RoPE uses the exact computed spatial theta (D029), and quantization groups reset at every matrix row (D034).

## Current implementation checkpoint (2026-08-11)

The parser, CPU correctness oracles, tokenizer parity, streamed Metal Qwen/adapter, and complete streamed Metal DiT path through final velocity are implemented and numerically validated through G003/H007. CPU implementations remain correctness oracles, not the production inference architecture.

The production Metal DiT path is validated through H007. Do **not** replace it with `DiTBlockCPU` or retain fully dequantized weights in nested Swift arrays. Continue from the completed vertical slice:

```text
D007 block/tensor range locator with mmap-backed spans
→ E001 permanent Metal execution harness
→ E002–E008 tested Metal/MPS primitives
→ E009 one production Metal block 0 matching the H005 oracle
→ H006 streamed 28-block Metal loop
→ H007 streamed final layer + source-order velocity unpatchify
→ D008/F007 streamed Metal Qwen
→ G003 streamed Metal adapter (complete)
```

Normal GitHub Actions simulator CI has been verified to execute both a project Metal kernel and `MPSMatrixMultiplication` on the standard `macos-15` arm64 runner (final snapshot run `31452206651`). Put all pack-free functional Metal/MPS tests in the normal simulator test target. A physical XS Max remains mandatory only for A12-specific performance, memory, thermal, watchdog, and device-family acceptance.

---

# 1. Persistent context/checklist system — DO THIS FIRST

This job is too large to trust to conversational context.

At the repository root create:

```text
RUNBOOK.md
TODO.md
STATUS.md
DECISIONS.md
TEST_MATRIX.md
DEVICE_TESTS.md
```

Put this complete runbook in `RUNBOOK.md`.

## TODO.md

Make a checkbox for every meaningful task with stable IDs.

Use approximately:

```text
A001... preflight and assets
B001... repository/project bootstrap
C001... CI
D001... animapk/model store
E001... Metal/MPS primitives
F001... tokenizer/text encoder
G001... LLM adapter
H001... DiT
I001... sampler/full diffusion
J001... VAE
K001... UI/resilience
L001... full CI/inference testing
M001... documentation/release/final handoff
```

Every task must include:

```text
status
dependencies
expected output
validation criterion
```

Do not check a task merely because code was written.

Check it only when its validation criterion passes.

## STATUS.md

Keep this short and current:

```text
Current milestone:
Current task:
Last green commit:
Current CI run:
What currently works:
What currently fails:
Known device-only unknowns:
Next three tasks:
```

## DECISIONS.md

Record non-obvious choices and why.

## Context-refresh rule

At:

* start of every work session;
* after context compaction;
* after a substantial failure;
* before beginning a new subsystem;
* whenever uncertain about what has already been done;

re-read:

```text
RUNBOOK.md
STATUS.md
unchecked TODO.md items
relevant handoff section
tail of DECISIONS.md
```

Never reconstruct project state from memory.

Commit frequently.

---

# 2. Hard preflight: locate the actual model assets

The supplied handoff ZIP intentionally does NOT contain the large packs.

Look first for the paths documented by `FILE_INDEX.md`, particularly:

```text
/root/anima-xsmax/results/packs/anima-turbo-v1.0-xsmax-w4.animapk
/root/anima-xsmax/results/packs/qwen3-0.6b-xsmax-w8.animapk
/root/anima-xsmax/results/packs/qwen-image-vae-xsmax-fp16.animapk
/root/anima-xsmax/results/goldens/
```

Expected packs:

```text
anima-turbo-v1.0-xsmax-w4.animapk
size:   1,179,435,008
SHA256: ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25

qwen3-0.6b-xsmax-w8.animapk
size:   635,305,984
SHA256: ba59e4d1797de5f6512aeafcecf3f38e1f62a47313a2a400b949c9018d84ceab

qwen-image-vae-xsmax-fp16.animapk
size:   256,163,840
SHA256: 10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447
```

Verify with `sha256sum`.

Also locate at least:

```text
case1_danbooru_seed1337.npz
```

Expected:

```text
size:   118,302,516
SHA256: 44d35d4f788c0a48411b0e68db66a84a79a8dcd8ef3beb842d800ceaff81a8dc
```

If the large assets are absent, do not invent replacements.

The code/project can still be built, but:

```text
model release upload
full model CI
real inference validation
```

remain blocked until the files are supplied.

Record that explicitly.

**STATUS: VERIFIED on 2026-08-10.** All four SHA-256s match exactly. Assets at the documented paths.

---

# 3. GitHub credentials and repository

The execution environment should contain the GitHub repository location and PAT supplied for this project.

Prefer environment variables such as:

```text
GH_TOKEN
GH_REPO
```

or equivalent credentials already configured by the harness.

Never:

```text
print the PAT
commit the PAT
put it in workflow YAML
put it in a git remote URL stored permanently
write it into STATUS/TODO/logs
```

Use GitHub CLI credential handling.

**STATUS: PAT found at `/root/GITHUB_PAT_ANIMAXS` (format: `PAT: <token>` line 1, repo URL line 2). Target repo: `https://github.com/invisiblestrangler/AnimaXS.git` — verified EMPTY, PUBLIC, not archived. gh CLI installed locally (v2.4.0).**

---

# 4. Verify current external facts before coding

Do not blindly trust even this runbook.

Before project bootstrap, verify:

1. Xcode 26.3 still exists on the selected GitHub Actions runner.
2. Its path.
3. Its bundled iOS SDK.
4. current `swift-transformers` compatibility.
5. current CircleStone Anima license and every upstream license inherited by the derivative, including NVIDIA Cosmos.
6. current GitHub Release size rules.

As of this runbook, Apple says Xcode 26.3 ships iOS 26.2 SDK, supports deployment to iOS 15–26.2, and supports Swift 5 and Swift 6 language modes.

As of this runbook, the public `macos-15` ARM64 GitHub runner has:

```text
3-core M1
7 GB RAM
14 GB SSD
```

and public standard runners are free/unlimited for public repositories.

The current `macos-15` image has:

```text
/Applications/Xcode_26.3.app
```

while Xcode 16.4 is the default, so CI must select Xcode 26.3 explicitly.

If these facts have changed, adapt CI while preserving the requirement that **Xcode 26.3 remains a required build**.

**STATUS: RE-VERIFIED 2026-08-11.** Final snapshot run `31452206651` used the standard arm64 `macos-15` image, selected `/Applications/Xcode_26.3.app`, verified XcodeGen 2.46.0 by SHA-256, regenerated the project cleanly, built the generic iPhone target, and ran 56 simulator tests with 0 failures (3 expected real-pack skips). The permanent project Metal-kernel and MPS GEMM tests executed successfully on `Apple iOS simulator GPU`. Current action majors were checked before updating CI.

Authoritative links used by this audit:

- GitHub runner specifications: `https://docs.github.com/en/actions/reference/runners/github-hosted-runners`
- current runner image inventory: `https://github.com/actions/runner-images/blob/main/images/macos/macos-15-arm64-Readme.md`
- Apple Simulator Metal behavior: `https://developer.apple.com/documentation/metal/developing-metal-apps-that-run-in-simulator`
- MPS matrix-multiplication semantics: `https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication`
- GitHub Release limits: `https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases`

---

# 5. Deployment target: do this correctly

Use:

```text
IPHONEOS_DEPLOYMENT_TARGET = 18.0
```

in the **app project settings**.

Do NOT set:

```text
deployment target = 26.x
```

and do not require an iOS 18.6 build SDK.

Xcode 26.3 builds against its iOS 26.2 SDK while producing an app whose minimum OS is iOS 18.0. That is what allows installation on the user's iOS 18.6 phone.

Do not globally export `IPHONEOS_DEPLOYMENT_TARGET` in CI because that can interfere with Swift Package dependencies.

Set it on the project/targets.

---

# 6. Model-license gate before public upload

The `.animapk` files are modified/packed model weights and publishing them is model distribution.

Anima is a derivative of NVIDIA Cosmos, so the release gate covers both the current CircleStone Anima license and the current NVIDIA Cosmos/Open Model terms. Before uploading packs to the public GitHub repository's Releases:

1. Fetch and archive the CURRENT official CircleStone Anima license and NVIDIA upstream model license/terms from their authoritative model pages.
2. Read every distribution, attribution, use-restriction, and derivative-work condition.
3. Confirm this project's intended use/distribution fits **both** sets of terms.
4. Distribute the required license copies and notices, including any required NVIDIA/Cosmos attribution wording.
5. Clearly state that the `.animapk` files are modified/quantized/converted derivatives.
6. Include any required website/UI/documentation acknowledgement.
7. Do not imply CircleStone or NVIDIA endorses the project.

The current license permits model/derivative distribution only subject to its terms and non-commercial restrictions, and its distribution section requires a license copy and attribution notice; derivatives also require disclosure that the model was modified.

If the current license no longer permits the intended public distribution, **do not upload the packs**. Finish the code using local asset paths and document the licensing blocker.

Do not choose an open-source license for the app's own source code unless one has been supplied by the user. Model-license files and app-code licensing are separate issues.

Do not treat this runbook as legal advice or mark A005 complete from a summary. Preserve the exact license texts/notices used by the release and record their source URLs and retrieval date in `DECISIONS.md`.

The 2026-08-11 audit observed CircleStone Non-Commercial License v1.2 at `https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md` and the upstream NVIDIA Open Model License dated 2025-04-30 in `https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/blob/main/README.md`. The latter currently requires its license/Notice attribution and “Built on NVIDIA Cosmos” acknowledgement. Re-fetch these exact sources at A005; do not rely on this observation if either revision changes.

**STATUS: PENDING — both license chains must pass A005 before L002 uploads model packs.**

---

# 7. Technical facts that MUST drive the implementation

These are confirmed by the supplied handoff.

Re-check them in the machine-readable files before coding.

## DiT

```text
model: Anima-Turbo v1.0
type: FLOW / flow-matching
blocks: 28
hidden: 2048
heads: 16
head dim: 128
MLP: 2048 → 8192 → 2048, GELU
latent channels: 16
effective input channels after padding-mask concat: 17
patch: 2×2 spatial, T=1
512 image: latent 64×64 → 32×32 = 1024 DiT tokens
cross context: 512 × 1024
```

### CRITICAL NUMERICAL FACT

The DiT residual stream produced measured values around:

```text
261,120
```

which is far beyond fp16's finite maximum.

Therefore:

```text
DiT residual stream = Float32
```

This is non-negotiable.

Normalize/modulate from Float32 and cast only appropriate branch/computation inputs to Float16.

Attention/MLP branch output may be Float16, but gate/residual addition goes back into Float32.

## Quantization

```text
DiT: W4 group=64 + fp16 exclusions
TE:  W8 group=64 + fp16 norms/etc.
VAE: fp16
```

## `.animapk`

```text
magic: ANMA
version: 1
endianness: little
header: 256 bytes
alignment: 16,384
```

Every tensor blob was verified 16 KB aligned.

The binary tensor table truncates:

```text
name → 64 chars
shape → 4 dimensions
```

Therefore:

**Use JSON tensor metadata as authoritative.**

Match JSON and table records using `blob_offset`.

Do not port the existing C++ parser blindly because its binary-shape handling is insufficient for 5-D VAE tensors.

## Physical DiT ordering

The DiT pack is NOT physically in numerical block execution order.

It is string/alphabetical order:

```text
0, 1, 10...19, 2, 20...27, 3...9
```

Each transformer's block tensors ARE contiguous, each approximately:

```text
38,993,920 stored bytes
```

Therefore:

* execute logical blocks 0→27;
* look up each block's actual byte range;
* never calculate `blockOffset = firstBlock + index * blockSize`.

Also do not trust JSON `block_index` alone for blocks 0–5 because the original regex also matched LLM-adapter blocks.

Identify DiT blocks strictly by:

```text
model.diffusion_model.blocks.N.
```

## Largest weight scratch

Largest dequantized fp16 matrix:

```text
33,554,432 bytes
```

Use one reusable dequant scratch buffer.

## Text encoder

```text
Qwen3-0.6B
28 layers
hidden: 1024
Q heads: 16
KV heads: 8
head dim: 128
MLP intermediate: 3072
activation: gated SiLU
RMSNorm
RoPE theta: 1,000,000
```

The main Qwen output used as `cond_context` is the layer-27 hidden state **after final RMSNorm** (D019). Earlier handoff prose saying “without final norm” refers to a different intermediate capture path and is not authoritative for this golden.

Text encoder pack:

```text
635,305,984 bytes
embedding stored W8: ~165 MB
each TE layer stored: 16,777,216 bytes
all 28 TE layers contiguous
```

Never dequantize the full embedding table.

Gather only token rows.

## LLM adapter

Weights live in the DiT `.animapk`.

It takes:

```text
Qwen last hidden
+
T5 target token IDs
```

and produces the 512×1024 DiT cross-attention context.

Re-check its exact MLP activation, self-attention mask/causality, and cross-attention behavior from the pinned source before implementing. Do not rely on the handoff's `GELU?` question-mark description.

## Sampler

Canonical Turbo:

```text
steps: 8
CFG: 1
sampler: Euler
scheduler: sgm_uniform
```

Sigma sequence:

```text
1.0
0.95469
0.90036
0.834
0.75112
0.64469
0.50299
0.30501
0.0
```

CFG=1 skips the unconditional model pass.

Sampler arithmetic is Float32.

## VAE

```text
latent: [1,16,1,64,64]
output: [1,3,1,512,512]
3-D causal convolutions
Wan channel-wise RMS normalization (`F.normalize` over C, then ×sqrt(C)×gamma)
decoder channels include 384 → 192 → 96
```

Single-frame T=1 3D→2D folding is validated in J001. The actual model uses pinned
`comfy/ldm/wan/vae.py`: uncached T=1 `CausalConv3d` uses causal zero padding, so
the effective 2-D kernel is the **final temporal slice**, not a temporal sum.
The decoder's two `time_conv` tensors are skipped at T=1 because no feature cache exists.
See `docs/VAE_FOLD_REPORT.md`; do not substitute the Cosmos tokenizer's replication semantics.

Tiled VAE decode has also NOT been validated.

Do not start by implementing tiling.

---

# 8. Keep the architecture small

Use a normal SwiftUI application.

Suggested repository:

```text
AnimaXS/
├── AnimaXS.xcodeproj
├── project.yml
├── AnimaXS/
│   ├── App/
│   │   ├── AnimaXSApp.swift
│   │   ├── ContentView.swift
│   │   ├── GenerationViewModel.swift
│   │   └── DiagnosticsView.swift
│   │
│   ├── Runtime/
│   │   ├── ModelStore/
│   │   ├── Animapk/
│   │   ├── Metal/
│   │   ├── Text/
│   │   ├── DiT/
│   │   ├── Sampler/
│   │   ├── VAE/
│   │   └── Generation/
│   │
│   ├── Resources/
│   │   └── Tokenizers/
│   │
│   └── Shaders/
│       └── AnimaKernels.metal
│
├── AnimaXSTests/
│   └── Fixtures/
├── scripts/
├── docs/
├── .github/workflows/
├── README.md
├── RUNBOOK.md
├── TODO.md
├── STATUS.md
├── DECISIONS.md
├── TEST_MATRIX.md
└── DEVICE_TESTS.md
```

Do not create multiple framework targets.

Do not build a plugin system.

Do not build generic device abstraction layers.

Use Foundation + SwiftUI + Metal + MetalPerformanceShaders + MPSGraph + CryptoKit.

Use only the `Tokenizers` product from Hugging Face `swift-transformers` unless more is genuinely needed.

The current package supports iOS 16+, so an iOS 18 target is compatible.

Pin the dependency to a verified exact release/commit rather than tracking `main`.

---

# 9. Xcode project generation

The clean execution environment may not be a Mac.

Use **XcodeGen only as a project-generation tool**, not as a permanent runtime dependency.

Create a simple `project.yml`.

Required project settings:

```text
product: application
platform: iOS
deployment target: 18.0
Swift language mode: 5
CODE_SIGN_STYLE: Automatic
no committed DEVELOPMENT_TEAM
```

Swift 5 language mode is intentional:

* async/await remains available;
* Metal/MPS objects cause less strict-Sendable friction;
* this project gains little from making Swift 6 concurrency migration part of the inference problem.

Xcode 26.3 officially supports Swift 5 language mode.

Generate the `.xcodeproj` on a GitHub Actions macOS runner.

Commit both:

```text
project.yml
AnimaXS.xcodeproj
```

The MacBook user must NOT need XcodeGen just to clone and open the app.

CI should regenerate the project and fail if the committed project differs from `project.yml`.

Use XcodeGen 2.46.0 for both local and CI generation. CI downloads the official release zip and verifies its pinned SHA-256 (D041). Keep `generateEmptyDirectories: false`: git does not preserve empty directories, so including them makes a populated local checkout generate a different project from a clean CI checkout (D040).

**Implementation note: bootstrap job in `.github/workflows/bootstrap-project.yml` generates the xcodeproj on macos-15 and commits it back. `ci.yml` job 1 regenerates and `git diff --exit-code` to enforce consistency.**

---

# 10. Bootstrap GitHub Actions immediately

Do not wait until the app is complete before using CI.

First get a minimal SwiftUI app green.

Use:

```yaml
runs-on: macos-15
```

and explicitly:

```yaml
env:
  DEVELOPER_DIR: /Applications/Xcode_26.3.app/Contents/Developer
```

Before every important CI build print:

```bash
xcodebuild -version
xcrun --sdk iphoneos --show-sdk-version
xcrun --sdk iphonesimulator --show-sdk-version
df -h
```

Fail clearly if selected Xcode is not 26.3.

### Metal toolchain workaround

Xcode 26's Metal compiler can be an optional runner component and GitHub runner-image reports have documented it being absent on some jobs. If:

```bash
xcrun --find metal
```

fails, run:

```bash
xcodebuild -downloadComponent MetalToolchain
```

before building.

Do this conditionally because the component is large.

---

# 11. Required CI workflow

Create:

```text
.github/workflows/ci.yml
```

Required push/PR workflow:

### Job 1 — project consistency

```text
checkout
select Xcode 26.3
ensure Metal toolchain
install XcodeGen if needed
xcodegen generate
git diff --exit-code
```

### Job 2 — iPhone build

Run:

```bash
xcodebuild \
  -project AnimaXS.xcodeproj \
  -scheme AnimaXS \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

This proves that an ARM iPhone build compiles without needing signing.

### Job 3 — simulator unit tests

Discover an available iPhone simulator dynamically.

Do not hard-code:

```text
iPhone XS Max simulator
iOS 18.6 simulator
```

Xcode 26.3's bundled simulator is from the iOS 26.2 SDK generation. Use whatever compatible iPhone simulator its installation exposes.

Build/test the iOS 18-targeted app on that simulator.

### Pure-reference and pack-free Metal tests

Run these in the same simulator XCTest target as Job 3; do not compile the large Swift dependency graph a second time merely to create a fourth job:

```text
.animapk header parser
full-name/shape JSON resolution
16 KB offset checks
W4 test vector
W8 test vector
fp16 tensor reader
sampler vector
sigma schedule
RoPE CPU reference
RMSNorm CPU reference
AdaLN CPU reference
tokenization reference
model manifest/hash parser
checkpoint serialization
Metal kernel smoke
MPS matrix-multiplication smoke
```

Do not download 2 GB of model weights on every push.

---

# 12. GitHub Actions is NOT the A12 oracle

GitHub's ARM64 standard macOS runner currently has an M1-class VM with 7 GB RAM and 14 GB SSD.

That is useful for:

```text
building
simulator testing
Metal shader compilation and execution
MPS execution
CPU numerical tests
pack-free GPU/CPU parity tests
possibly a full functional inference after release assets and L001 exist
```

but it cannot prove:

```text
A12 speed
A12 jetsam threshold
Apple5 numerical behavior
A12 command-buffer watchdog limit
A12 sustained thermal behavior
A12 page-cache behavior
```

Do not claim that an M1 CI inference proves those things.

The standard hosted runner currently exposes an iOS Simulator Metal device, but every GPU test must still probe `MTLCreateSystemDefaultDevice()`. An unavailable device is an explicit `SKIPPED_NO_METAL`; a present device with a wrong result is a test failure. Never downgrade a real Metal/MPS failure to a skip.

---

# 13. Separate manual full-model workflow

Create:

```text
.github/workflows/full-inference.yml
```

Trigger only through:

```text
workflow_dispatch
```

This job should:

1. select Xcode 26.3;
2. ensure Metal compiler;
3. discover and boot a simulator;
4. execute the permanent pack-free Metal/MPS smoke tests;
5. if Metal is unavailable:

   * record `SKIPPED_NO_METAL`;
   * exit successfully;
6. if Metal exists:

   * require `FullInferenceTests` to exist;
   * run the app's full integration inference test;
   * fail the workflow on build, download, execution, timeout, NaN/Inf, or parity failure.

Do not silently label a skipped test PASS.
Do not use `continue-on-error` on the inference step. Write `PASS`, `FAIL`, or `SKIPPED_NO_METAL` to the job summary. Until L001 and L002 are complete, the manual workflow is infrastructure only and must report “not implemented” as a failure if invoked on a Metal-capable runner.

The manual integration test may download the three model packs from the GitHub Release because together they are only about 2.1 GB and the runner currently has 14 GB SSD. Monitor `df -h` during this job.

---

# 14. Model assets should use GitHub Releases, not Git

Do NOT commit `.animapk` files to normal Git.

GitHub normal repository objects are limited to 100 MB.

Do not use Git LFS unless forced.

Use one release such as:

```text
model-assets-v1
```

and upload:

```text
anima-turbo-v1.0-xsmax-w4.animapk
qwen3-0.6b-xsmax-w8.animapk
qwen-image-vae-xsmax-fp16.animapk
model-manifest.json
MODEL_LICENSE.md
MODEL_NOTICE.txt
```

GitHub currently allows up to 1000 assets per release, each under 2 GiB, with no total release-size or bandwidth limit. All three packs individually fit that limit.

Use the exact verified SHA-256s from section 2.

---

# 15. Extract small golden fixtures

Before implementation, use the existing canonical golden:

```text
case1_danbooru_seed1337.npz
```

Create small deterministic test fixtures.

Commit only a small, deterministic CPU fixture set:

```text
canonical prompt text
T5 token IDs
attention mask
Qwen final-context anchors or the full context if the size budget permits
init_noise_randn
sigma vector
final latent
small deterministic decoded-RGB anchors/slices
```

Extract compact deterministic slices/anchors plus shape/hash metadata for:

```text
step_latents
block_00_out
block_15_out
block_27_out
```

Store them as very simple:

```text
raw little-endian Float32
raw Int32/Int64 as appropriate
JSON metadata with shape + SHA256
```

Do not implement an NPZ reader in Swift just for tests.

Small fixture files can live in the **test target only**. Keep the committed fixture set near the A006 budget (about 3 MB); do not commit the complete 8 MB block tensors or the approximately 118 MB NPZ. Full tensors remain local or are fetched only by the manual pack-backed integration workflow.

Do not include tens of megabytes of debug goldens in the shipping app unless needed.

A minimal end-to-end device fixture should contain:

```text
canonical prompt
golden input noise
expected final latent
```

This allows inference validation without requiring exact production RNG parity.

---

# 16. Tokenizers

Two tokenizers are required:

```text
Qwen tokenizer
T5 tokenizer
```

Use the exact tokenizer assets corresponding to the pinned reference.

Bundle the tokenizer/config files as app resources.

Use `swift-transformers` `Tokenizers` from local files; do not require Hugging Face network access for tokenizer operation.

The library supports local tokenizer folders and currently contains T5/SentencePiece-related regression tests, but tokenization parity still MUST be checked against the supplied model reference rather than assumed.

Before accepting the dependency:

1. create Python reference token IDs for all three canonical prompts;
2. run the Swift implementation;
3. demand exact token IDs;
4. test both Qwen and T5.

If the latest stable package version disagrees:

```text
try a newer fixed commit
```

before writing your own tokenizer.

Do not continue to the text encoder with mismatched tokenization.

For v1, reject prompts whose Qwen OR T5 tokenized length exceeds 512 instead of silently applying unverified truncation behavior.

---

# 17. `.animapk` parser — implement in Swift

Do not add C++ bridging unless Swift proves impossible.

Implement:

```text
MappedFile.swift
AnimapkHeader.swift
AnimapkTensor.swift
AnimapkFile.swift
```

`MappedFile` should:

```text
open O_RDONLY
fstat
mmap PROT_READ | MAP_PRIVATE
own fd + mapping lifetime
munmap/close cleanly
```

Parse little-endian header manually.

Avoid unsafe unaligned integer loads; copy bytes safely.

Load architecture JSON with Foundation `JSONDecoder`.

Build:

```swift
[String: TensorMeta]
```

using the JSON full names/shapes.

Also index by `blobOffset`.

Validation:

```text
magic == ANMA
version == 1
declared file size == actual
all section ranges inside file
all blob ranges inside file
every blobOffset % 16384 == 0
```

Never use the binary table's truncated shape as the authoritative shape.

---

# 18. W4/W8 decoders: prove byte compatibility first

Before using GPU dequantization, implement tiny CPU reference decoders in the TEST TARGET.

W4 exact definition:

```text
group axis: K/input dimension
group size: 64
q: unsigned 0...15
scale: fp16
zero: fp16
even K index → low nibble
odd K index → high nibble

value = q * scale + zero
```

Required known W4 test:

```text
block 0 mlp.layer1
first packed bytes:
110, 135, 69, 55, 138, 137, 37, 98

first scale values include:
0.0004537
0.0004232
0.0004251
0.0004313
```

Verify the first reconstructed values against the handoff.

W8:

```text
unsigned uint8
group size 64
fp16 scale
fp16 zero
value = q*scale + zero
```

For a rank-2 `[N,K]` matrix, group indexing resets for each row:

```text
groupsPerRow = ceil(K / 64)
group = row * groupsPerRow + column / 64
```

Never use `flatIndex / 64` for matrices whose K is not divisible by 64. Required regression shapes are W4 `[2,68]` and W8 `[2,65]`, with row-1 values asserted. This is the H005 root-cause regression (D034).

Use the real embedding-table vector documented in `HANDOFF.md`.

Only after CPU tests are exact should Metal decoders be implemented.

---

# 19. Minimal runtime memory design

Start with this.

No three-slot ring.

No MTLHeap.

No dynamic tier system.

### Weight ring

One reusable:

```text
39 MB-ish .storageModeShared MTLBuffer
```

large enough for the largest contiguous DiT block range.

### Dequant scratch

One reusable fp16:

```text
33,554,432-byte
```

buffer for the largest matrix.

### Residual

One or two reusable Float32 residual buffers:

```text
1024 × 2048 × 4 ≈ 8.4 MB each
```

### Other activations

Reuse simple explicit buffers.

Do not allocate all possible tensors simultaneously.

Start with ring=1.

The CPU oracle may use `Data` and flat Swift arrays for safe test-scoped decoding. The production path must operate on validated mmap-backed byte spans and `MTLBuffer` offsets; it must not copy entire tensors into `Data` or construct large `[[Float]]` matrices.

Only implement ring=2 after a real XS Max measurement demonstrates that I/O overlap is worth the memory.

---

# 20. Weight streaming

Implement:

```text
WeightStreamer.swift
```

For logical DiT block N:

1. obtain its actual start/end from metadata;
2. optionally `madvise(..., WILLNEED)`;
3. memcpy its contiguous bytes into ring buffer;
4. execute tensors using offsets relative to block start;
5. reuse ring for next logical block.

`D007` must expose both block ranges and tensor-relative spans. Validate every offset/length before creating a buffer view. Copy each block into the one-slot ring once, then address packed data/scales/zeros by offsets relative to that ring; do not make an additional `Data` copy per tensor. If a direct mmap upload or bytes-no-copy path is considered later, require A12 measurements first.

Do not assume physical neighboring blocks correspond to logical neighbors.

For Qwen:

```text
embedding rows gathered directly from mmap
layer N → copy its contiguous 16.78 MB range → execute → reuse ring
```

Do not dequantize an entire 31.5 MB text layer at once unless necessary.

Still dequantize one matrix at a time into the shared scratch.

The mmap is read-only/file-backed. Let the OS manage clean file pages initially.

Do not add aggressive page touching, custom page-cache tricks or `newBufferWithBytesNoCopy` before measuring them on the actual XS Max.

---

# 21. MetalContext

Implement one:

```text
MetalContext
```

owning:

```text
MTLDevice
MTLCommandQueue
default metallib
compute pipeline cache
MPS helpers
```

At runtime record:

```text
supportsFamily(.apple5)
maxBufferLength
maxThreadgroupMemoryLength
recommendedMaxWorkingSetSize if available
os_proc_available_memory()
ProcessInfo.thermalState
```

Do not require newer GPU families.

Do not use:

```text
simdgroup_matrix
Metal 3-only APIs
MTLIOCommandQueue
bfloat16
```

---

# 22. First Metal kernels

Keep `AnimaKernels.metal` straightforward.

Implement and independently unit-test:

```text
dequant_w4_to_half
dequant_w8_to_half

rmsnorm_f32_to_half
rmsnorm_half_to_half
layernorm_f32_modulated_to_half

silu
gelu_exact

gate_add_fp32_into_float
add_half_into_float

rope_qk
qk_rmsnorm

patchify17
unpatchify16
unpatchify_velocity16

euler_step_f32

w4_matvec_f32
```

All norm statistics/reductions:

```text
Float32 accumulation
```

All softmax max/sum:

```text
Float32
```

Sampler:

```text
Float32
```

Timestep/AdaLN:

```text
Float32
```

Do not fuse kernels prematurely.

Match the validated H005 equations, not convenient approximations: adapter/DiT GELU is exact `0.5*x*(1+erf(x/sqrt(2)))`, LayerNorm mean-centers, RMSNorm does not, and DiT modulation/gates/residual additions remain Float32. The current tanh GELU and half-precision gate scaffold are not parity implementations and must be replaced before E003 is complete.

Correct separate Q/K norm + RoPE is preferable to a hard-to-debug fused kernel.

Fuse only after correctness.

---

# 23. Linear layers: one common implementation

For large matrix operations:

```text
packed weight
→ Metal dequant to reusable fp16 weight scratch
→ MPSMatrixMultiplication
```

MPS supports transposition of either input matrix and computes the standard matrix product form.

For weight logical shape:

```text
[N, K]
```

and activations:

```text
[M, K]
```

use the weight as the right matrix with right transpose:

```text
[M,K] × [N,K]^T → [M,N]
```

The dequant kernels must receive explicit row/column strides and use row-reset quantization groups. A synthetic non-aligned-K matrix test is required before any real-model linear is trusted.

Use straightforward contiguous row strides first; MPS publishes recommended matrix row strides, but they are an optimization rather than a prerequisite for the first correct implementation.

### M=1 precision-critical linears

For:

```text
timestep embedding
AdaLN/modulation
```

use the custom direct:

```text
w4_matvec_f32
```

kernel.

It should:

```text
read packed W4 directly
dequant group values on the fly
multiply by fp32 input
accumulate fp32
produce fp32 output
```

This avoids forcing the precision-critical modulation path through fp16.

---

# 24. Tile the M dimension of giant GEMMs

Do not launch the largest:

```text
M=1024, N=8192, K=2048
```

as one monolithic operation on A12 unless testing proves it safe.

Default first implementation:

```text
M tile = 128
```

For a 1024-row GEMM:

```text
8 tiles
```

Dequantize the weight once.

Execute all M tiles against it.

Use asynchronous command-buffer completion from a background task/actor.

Never block the SwiftUI main thread with:

```text
waitUntilCompleted()
```

A helper using a checked continuation around the command-buffer completion handler is enough.

Do not build a dynamic GFLOPS autotuner yet.

---

# 25. A12 MPS precision test

The handoff correctly identifies this as unresolved.

Build an isolated test covering:

```text
K=2048
K=8192
```

and representative model shapes.

Compare GPU output against a CPU Float32/Float64 reference.

Measure:

```text
max absolute error
RMSE
cosine similarity
```

Record it in diagnostics.

If MPS fp16 behavior is accurate enough:

```text
keep the simple path
```

If not:

```text
implement K-chunking + fp32 accumulation only for affected operations
```

Do not implement a complicated precision fallback preemptively.

---

# 26. Attention: use the simplest memory-safe baseline

Do NOT begin with a bespoke FlashAttention implementation.

At 512×512 the DiT has only:

```text
1024 query tokens
```

Use **query tiling**.

Recommended first approach:

```text
query tile = 128
```

For each tile:

```text
Q_tile
× K^T
→ scaled scores
→ softmax
→ × V
→ output tile
```

Implement this with reusable MPSGraph attention/matmul+softmax graphs if they work correctly on the current SDK/device.

The query dimension is independent under attention softmax, so splitting the query rows does not alter the result.

At self-attention:

```text
Q tile: 128
K length: 1024
heads: 16
head dim: 128
```

At cross-attention:

```text
Q tile: 128
K/V length: 512
```

This limits score-memory and reduces the size of individual GPU work submissions without implementing streaming-key online softmax.

### Fallback

Only if actual XS Max testing shows that MPSGraph:

```text
uses too much memory
times out
or is unacceptably slow
```

implement the custom online-softmax attention kernel described by the handoff.

Do not build both backends upfront.

---

# 27. Text encoder implementation

Implement Qwen first.

`QwenEncoderCPU` is the validated oracle. Production F007 must use the same equations through E002–E008, gather only requested embedding rows, stream one D008 layer range at a time, and reuse bounded Metal buffers. Do not retain all dequantized layer matrices in Swift arrays.

Pipeline:

```text
prompt
→ Qwen token IDs
→ gather W8 embedding rows
→ Float32 residual
→ 28 streamed Qwen layers
→ layer 27 output
→ apply final RMSNorm
```

Keeping the Qwen residual in Float32 is inexpensive because the sequence is small and reduces numerical risk.

For each Qwen layer re-check exact pinned-source ordering.

Expected broad structure:

```text
RMSNorm
self attention
residual
RMSNorm
gated SiLU MLP
residual
```

Q/K has per-head normalization.

RoPE theta:

```text
1,000,000
```

GQA:

```text
16 Q heads
8 KV heads
```

For simplicity, expand/broadcast K/V heads to 16 with a tiny Metal kernel or the equivalent MPSGraph broadcast rather than implementing a generic GQA abstraction.

Implement the exact causal mask used by the pinned Qwen reference.

Validation gate:

```text
canonical prompt token IDs exact
final Qwen context shape exact
all finite
cosine against supplied cond_context >= expected golden tolerance
```

Only then proceed to the adapter.

---

# 28. LLM adapter

After Qwen completes:

```text
unmap/release TE pack
map DiT pack
```

Generate T5 token IDs using the second tokenizer.

Gather adapter target embeddings directly from the W4 embedding tensor instead of dequantizing the entire `[32128,1024]` matrix.

Execute the six adapter blocks.

`LLMAdapter.swift` is the validated CPU oracle. Production G003 must execute these equations with the common Metal/MPS primitives and bounded buffers; it must not dequantize the full T5 embedding table or build large nested Swift matrices.

Use the exact pinned source to resolve:

```text
MLP activation
self-attention causality/mask
cross-attention mask
RoPE placement
normalization order
bias behavior
```

Do not guess those details.

Finally:

```text
out_proj
RMSNorm
multiply T5 weights
zero-pad sequence to 512
```

Retain:

```text
[1,512,1024]
```

conditioning for diffusion.

---

# 29. DiT input

For the first working generation support only:

```text
512×512
```

Do not add 640 yet.

Latent:

```text
[1,16,1,64,64]
```

Append the confirmed padding-mask channel:

```text
17 channels
```

Patchify:

```text
2×2
→ 1024 tokens
→ width 68
```

Apply input projection to 2048.

Store resulting residual as Float32.

---

# 30. Exact timestep/modulation path

Implement directly from model source/handoff.

Timestep is:

```text
sigma
```

not an arbitrary integer diffusion step.

The confirmed sinusoidal embedding uses:

```text
dimension 2048
base 10000
```

and then the model-specific RMSNorm/timestep embedding path.

Keep the entire:

```text
timestep embedding
AdaLN LoRA projection
shift/scale/gate creation
```

in Float32.

For each of the three block branches:

```text
self-attn
cross-attn
MLP
```

the modulation must follow the exact shift/scale/gate chunk ordering.

This is a major correctness risk.

Write standalone CPU unit tests for modulation equations before GPU integration.

---

# 31. RoPE

Do not implement generic LLM RoPE and assume it applies to DiT.

The DiT uses its own confirmed 3-D position encoding.

From the handoff, re-read and implement:

```text
axes: T, H, W
head dimension: 128
axis split:
  H 42
  W 42
  T 44

spatial theta = 10000 * 4^(42/40) = 42870.938501451725
temporal theta = 10000
```

and the exact 2×2 rotation block arrangement.

DiT RoPE applies to self-attention.

Cross-attention does not use that 3-D RoPE.

Write a CPU implementation first.

The CPU implementation has been validated against the pinned source (H004/D029). E004 must apply the same split-half mapping `(p, p+64)` in Metal and compare a tiny reference slice.

Wrong RoPE can produce plausible but completely incorrect images, so this is a hard validation gate.

---

# 32. One DiT block first

Do NOT write the full 28-block loop and debug the final image.

Implement a single block and validate it.

The completed CPU H005 gate uses canonical:

```text
prompt
golden input noise
the final block-hook invocation, sampler step 7, sigma 0.3050089478492737
```

Run from input through block 0.

If `block_00_out` has been extracted into the test fixtures:

compare:

```text
shape
finite values
max abs
RMSE
cosine
```

The handoff's suggested block correctness range is approximately:

```text
cosine >= 0.999
```

but use measured/reference tolerances rather than blindly forcing a number.

CPU H005 is green: Swift-W4 matches the independent NumPy-W4 oracle at cosine `1.000000000`; the remaining W4-vs-source-golden difference is source-proven quantization error (D032–D035).

Do not proceed directly from that CPU oracle to a CPU 28-block loop. First implement E009: one bounded-memory production Metal block 0 using the real range locator, ring/scratch buffers, Metal kernels, MPS linears, and tiled attention. It must reproduce the H005 W4 oracle within an experimentally recorded tolerance and keep residual/modulation/gates Float32.

---

# 33. Full DiT loop

Once the production Metal block 0 is correct:

```text
for logical block in 0..<28
```

For each:

1. locate real byte range;
2. load into ring;
3. execute exact block;
4. optionally emit diagnostics;
5. reuse buffers;
6. continue.

In debug/self-test mode compare selected:

```text
block 0
block 15
block 27
```

if fixtures exist.

In normal mode do not copy huge block outputs back to CPU and do not retain dequantized matrices between linears.

After block 27 execute final normalization/modulation/projection exactly as the reference specifies.

Unpatchify to:

```text
[1,16,1,64,64]
```

velocity/flow output.

---

# 34. Sampler

Implement the canonical Turbo path only.

Hardcode/reference the validated 9 sigma points in one constants file.

At each of the eight steps:

```text
run DiT once
CFG=1 → no unconditional pass
apply Euler FLOW update in Float32
```

Use the exact formula from supplied sampler vectors.

Unit-test it with `sampler_vectors.py` data.

Keep latent Float32 between steps.

After every step:

```text
check NaN/Inf
update progress
optionally checkpoint ~256 KB latent
```

---

# 35. Production RNG

Exact `torch.randn` parity is NOT required for the first app.

Do not spend days reverse-engineering PyTorch's normal generator.

Production requirements:

```text
same app version + same seed → deterministic same initial noise
reasonable standard-normal distribution
```

Implement a small deterministic seeded generator plus Box-Muller or equivalent.

Document that app seeds are initially **app-deterministic, not guaranteed ComfyUI-identical**.

For golden testing, bypass production RNG and load the exact supplied `init_noise_randn`.

This cleanly separates:

```text
runtime correctness
```

from:

```text
cross-framework RNG parity
```

---

# 36. VAE: validate 3D→2D fold BEFORE app dependence

Write:

```text
scripts/validate_vae_fold.py
```

using existing VAE weights and exact source padding behavior.

For every decoder/post-quant causal 3-D conv at T=1:

1. compute direct reference 3-D output;
2. construct proposed folded 2-D kernel;
3. compute 2-D output;
4. compare numerically.

Verify all temporal kernel/padding variants used by decoder.

Do not assume every layer folds identically.

J001 marked:

```text
VAE_2D_FOLD_VALIDATED=LAST_TEMPORAL_SLICE_CAUSAL_ZERO
```

after all 34 relevant rank-5 tensors passed within Float64 reduction ulps and all
32 kt=3 tensors disproved replication-sum folding.

If folding cannot exactly reproduce T=1 behavior, implement the actual T=1 3-D semantics instead.

---

# 37. VAE first implementation: full frame, NOT tiles

Do not start with 64×64 tile blending.

The decoder contains spatial attention and wide convolutional receptive fields, so
independent tiles change the math even though its normalization is channel-local.

Instead:

```text
final latent
→ Wan21 latent normalization
→ full-frame decoder
→ RGB
```

Use exact channel mean/std values from `model_info.json`/runtime constants.

For T=1 use chunk-0 values exactly as the reference.

### Convolutions

After fold validation, use a straightforward Metal Performance Shaders/MPSGraph 2-D convolution path.

Process one layer at a time.

Do not keep all VAE weights resident.

### Wan RMS normalization

Match pinned `wan/vae.py::RMS_norm`:

```text
normalize across channel dimension only at each (T,H,W)
PyTorch F.normalize epsilon behavior
multiply by sqrt(channel count)
multiply by learned gamma
fp16 output
```

Do not port the Cosmos tokenizer GroupNorm path; it is a different VAE implementation.

### Activations

Use exact source activation and exact upsampling method.

Re-read pinned VAE source; do not infer interpolation details.

---

# 38. Only implement tiled VAE if the XS Max actually needs it

The full 512 decoder has large activations but is still worth trying first.

If actual device diagnostics show unacceptable peak memory/jetsam:

THEN implement tiled decode.

At that point preserve exact convolution halos and the middle block's global spatial
attention. Channel-wise RMS normalization itself is local to each pixel and does not
require a global two-pass statistic.

The old candidate:

```text
64×64 latent tile
16 px overlap
```

was never validated.

Treat it as an experiment, not truth.

---

# 39. RGB output

Reference decoder can produce values slightly outside `[-1,1]`.

For display:

```text
rgb01 = clamp((rgb + 1) / 2, 0, 1)
```

convert to an image.

Produce `CGImage`/`UIImage`.

Do not retain the large Float32 RGB buffer after conversion if unnecessary.

---

# 40. Model download/store

The app must not bundle 2 GB of weights.

Create a small built-in manifest containing for each model:

```text
filename
size
SHA256
GitHub Release URL
component
```

ModelStore UI:

```text
Not Downloaded
Downloading X%
Verifying
Ready
Error
```

Download sequentially.

Store under:

```text
Application Support/AnimaXS/Models/
```

Use `URLSessionDownloadTask`.

Move temporary file atomically to final location.

Verify SHA-256 incrementally—never load a 1.18 GB file into `Data`.

Use CryptoKit streaming SHA256.

Mark model directory/files:

```text
excluded from backup
FileProtectionType.completeUntilFirstUserAuthentication
```

Before download, check available disk space and show the user the total download size.

No complicated background-download subsystem is required for v1.

---

# 41. Minimal SwiftUI

One screen is enough initially.

Include:

```text
model status/download controls
prompt TextEditor
seed
Randomize Seed
resolution: 512 only initially
Generate button
Cancel button
stage progress
step 1/8...8/8
block progress where useful
elapsed time
result image
Share button
Diagnostics button
```

Do not build:

```text
accounts
gallery database
cloud sync
prompt history database
social features
multi-model selector
settings hierarchy
```

After 512 end-to-end works, you may expose 640 experimentally.

---

# 42. GenerationCoordinator

Use one coordinator.

Only one generation may run at a time.

Conceptual sequence:

```text
tokenize
map TE
encode Qwen
unmap TE

map DiT
run adapter
create/init latent
8 diffusion steps × 28 blocks
unmap DiT

map VAE
decode
unmap VAE

publish image
```

Keep only the 512×1024 conditioning between TE/DiT.

Use `autoreleasepool` around large stages if helpful.

No component should remain mapped merely for convenience once its stage is complete.

---

# 43. Cancellation/background/checkpoint

Do the useful minimum.

After each sampler step save:

```text
latent Float32
step index
prompt
seed
resolution
model hashes
```

to an atomic checkpoint.

On Cancel:

```text
finish/abort at a safe command-buffer boundary
free resources
keep or delete checkpoint based on user intent
```

On app background:

```text
stop scheduling new GPU work
finish current safe work
checkpoint
release large GPU resources
```

On foreground:

```text
offer Resume
```

Do not attempt to keep a long Metal generation running in background.

Disable `UIApplication.shared.isIdleTimerDisabled` only while actively generating, and restore it afterward.

---

# 44. Memory pressure and thermal behavior

Do not implement five runtime tiers.

Observe:

```text
memory warning
available memory
thermal state
```

Simple policy:

### Memory warning

```text
checkpoint
cancel current generation gracefully
free ring/scratch/activation buffers
show recoverable message
```

### Thermal serious

```text
continue but pause briefly at safe block/step boundaries if needed
```

### Thermal critical

```text
checkpoint and pause/stop
```

Record the state in diagnostics.

---

# 45. Diagnostics screen — important because CI has no A12

Add a simple Diagnostics screen.

Display/export:

```text
OS version
device/GPU family support
maxBufferLength
maxThreadgroupMemoryLength
recommendedMaxWorkingSetSize
os_proc_available_memory
thermal state

model file sizes/hashes
model parser status
```

Buttons/tests:

```text
Run model-pack validation
Run W4 vector test
Run W8 vector test
Run MPS precision benchmark
Run 39 MB mmap/copy benchmark
Run representative MPS GEMM
Run attention tile benchmark
Run golden-noise inference self-test
Export diagnostics JSON
```

Do not deliberately try to crash the GPU to discover watchdog timeout.

Instead progressively measure representative operation durations.

---

# 46. Device microbenchmarks that matter

On the XS Max, capture:

### mmap/copy

```text
read/copy one 38,993,920-byte DiT block
cold
warm
MB/s
```

### W4 dequant

Largest:

```text
8192 × 2048
```

Measure time.

### MPS GEMMs

At least:

```text
M=1024 N=2048 K=2048
M=1024 N=8192 K=2048
M=1024 N=2048 K=8192
```

plus tiled M=128 versions.

### Accuracy

Compare K=2048/8192 against CPU Float32.

### Attention

Test:

```text
16 heads
q=128 tile
k=1024
d=128
```

### Memory

Record available memory at:

```text
idle
after mapping
after ring allocation
inside DiT
before VAE
VAE peak
```

### Thermal

Record total time and thermal state after one 8-step generation.

These results belong in `DEVICE_TESTS.md`.

---

# 47. Model release upload

Once license gate and pack verification pass:

```bash
gh release create model-assets-v1 ...
```

Upload the three `.animapk` files plus:

```text
model-manifest.json
MODEL_LICENSE.md
MODEL_NOTICE.txt
```

Verify download URLs from an unauthenticated request because the app must work for a normal public-repository user.

Download each once and re-check SHA256.

Then put those URLs/hashes in the built-in app manifest.

---

# 48. Tests required before calling the implementation complete

## Pure Swift/CPU unit tests

Must pass:

```text
header parse
JSON full-name/shape
range validation
W4 nibble order
W4 dequant reference
W8 dequant reference
fp16 tensor read
block-range lookup
logical block 0...27 order
sampler vector
sigma constants
checkpoint serialization
SHA manifest
tokenizers exact on canonical prompts
CPU RoPE reference
CPU timestep reference
CPU modulation reference
```

## Metal tests where Metal exists

```text
W4 dequant
W8 dequant
norm
RoPE
GELU/SiLU
gate-add
MPS linear
attention tile
sampler kernel
```

Compare with CPU reference. Pack-free synthetic Metal/MPS tests run on every push because hosted execution is verified. At minimum include non-aligned-K W4/W8 row reset, exact GELU, fp32 gate-add, norm semantics, split-half RoPE, transposed MPS linear, tiled attention, patchify/unpatchify, and Euler update.

## Model integration

```text
TE final context is finite/close to golden
adapter context finite
block 0 finite/close to golden where fixture exists
all 28 blocks finite
8 step latents finite
final latent finite
VAE RGB finite
```

## Full canonical inference

Using:

```text
canonical prompt
exact golden input noise
512×512
8 steps
CFG1
```

compare at least:

```text
final latent
```

against supplied reference with the experimentally reasonable tolerance.

Do NOT demand bit equality between A12 fp16/W4 runtime and the desktop bf16/fp32 golden.

---

# 49. GitHub full-inference attempt

After L001 and the licensed model release L002 exist, trigger the manual `full-inference.yml`.

The simulator test should:

1. check a Metal device exists;
2. download/verify packs using the real ModelStore;
3. inject golden reference noise;
4. run canonical inference;
5. measure stage timings;
6. assert no NaN/Inf;
7. compare reference checkpoints where valid;
8. save only small logs/results.

Do not upload the model files themselves as Actions artifacts.

Do not cache all model packs.

If Metal is unavailable on a future GitHub VM:

```text
SKIP with explicit reason
```

and leave actual inference as a device acceptance gate. If Metal is available, every other error is a failure; never use `continue-on-error` to turn it green.

---

# 50. Do not fake A12 validation

Unless you physically ran the app on an iPhone XS Max, NEVER write:

```text
"A12 inference passed"
"XS Max memory is safe"
"XS Max generation takes X minutes"
```

GitHub M1 testing does not count.

Instead say:

```text
A12 DEVICE TEST PENDING
```

The project can still be implementation-complete and CI-green.

The MacBook/iPhone owner then runs the built-in Diagnostics/Self Test.

---

# 51. Signing/local Mac behavior

Do not commit a development team.

Set automatic signing.

README instructions:

```text
1. git clone ...
2. open AnimaXS.xcodeproj in Xcode 26.3
3. select AnimaXS target
4. Signing & Capabilities → choose your Apple Development Team
5. connect/unlock iPhone XS Max
6. enable Developer Mode on phone if required
7. select the physical iPhone
8. Build & Run
9. download model packs in app
10. run Diagnostics first
11. generate canonical 512 image
```

The repository should need no manual source edits to perform these steps.

---

# 52. README must explain SDK vs deployment target

Explicitly document:

```text
Validated build IDE: Xcode 26.3
Build SDK under Xcode 26.3: iOS 26.2
Minimum deployment target: iOS 18.0
Target physical OS: iOS 18.6
```

Do not tell the user to find an "iOS 18.6 SDK" for this project.

The newer build SDK is intentionally compatible with the older deployment target.

---

# 53. First-image implementation order

Follow this order.

Do not jump around.

```text
1. repository + Xcode project + CI green
2. animapk parser
3. CPU W4/W8 tests
4. tokenizer/Qwen/adapter and DiT CPU references through block 0 (complete)
5. DiT block/tensor range locator and one-slot ring semantics
6. permanent Metal execution harness in normal CI
7. row-aware Metal W4/W8 dequant + exact elementwise/norm/RoPE kernels
8. direct W4 fp32 matvec + MPS linear + query-tiled attention
9. one production Metal DiT block 0 vs H005 oracle
10. streamed Metal Qwen + adapter vs their CPU oracles
11. all 28 streamed Metal blocks / one DiT forward
12. Euler sampler
13. 8-step final latent
14. validate VAE 3D→2D fold
15. full-frame VAE
16. first RGB image
17. model download UX
18. checkpoint/background/memory handling
19. manual CI full-inference attempt
20. documentation
21. A12 diagnostics/self-test
```

This order is deliberate.

Do not build polish around an inference core that has never produced an image.

---

# 54. What NOT to implement unless measurements force it

Do NOT preemptively add:

```text
ring=2/3
MTLHeap activation aliasing
newBufferWithBytesNoCopy
custom FlashAttention
custom fused dequant-GEMM
VAE tiling
Core ML
MLX
ANE
multiple device tiers
multiple model variants
Base/Aesthetic
W8 DiT
1024 resolution
dynamic GPU autotuner
background inference
generic tensor graph engine
```

Each is allowed later only if a measured XS Max problem justifies it.

---

# 55. Important expected failure hierarchy

When first inference differs from reference, debug in this order:

```text
1. tokenizer IDs
2. Qwen final context
3. adapter output/masks
4. timestep embedding
5. AdaLN shift/scale/gate ordering
6. DiT 3-D RoPE axis/order
7. block 0 output
8. later block overflow/precision
9. sampler
10. Wan21 latent normalization
11. VAE temporal fold
```

Do not debug by staring at the final image.

Find the first stage where numbers diverge.

---

# 56. Git discipline

Every major green gate gets a commit.

Examples:

```text
ci: xcode 26.3 project builds
runtime: parse ANMA v1 packs
runtime: validate W4/W8 decode
metal: dequant and MPS linear
text: qwen conditioning parity
text: adapter conditioning
dit: block zero parity
dit: full forward
sampler: eight-step latent
vae: validated single-frame decode
app: end-to-end generation
ci: manual full inference
docs: mac install instructions
```

Push frequently.

After every push monitor Actions.

Fix red CI before accumulating unrelated changes.

---

# 57. Completion gates

The project is complete only when:

### Repository

```text
[ ] public GitHub repo populated
[ ] no secrets committed
[ ] no model packs in Git history
[ ] xcodeproj committed
[ ] project.yml committed
[ ] README complete
```

### Xcode

```text
[ ] Xcode 26.3 generic iOS build PASS
[ ] deployment target iOS 18.0
[ ] no signing required for CI build
[ ] Metal shaders compile
[ ] pack-free Metal kernel and MPS execution tests PASS in hosted simulator CI
[ ] simulator unit tests PASS
```

### Runtime

```text
[ ] animapk parser PASS
[ ] W4 decode PASS
[ ] W8 decode PASS
[ ] tokenizer parity PASS
[ ] Qwen runs
[ ] adapter runs
[ ] DiT block 0 works
[ ] 28 blocks work
[ ] 8-step sampler works
[ ] VAE fold validated
[ ] full VAE works
[ ] final image produced on at least available supported test environment
```

### Model distribution

```text
[ ] current model license re-checked
[ ] required license/notice distributed
[ ] three packs uploaded as Release assets OR license blocker documented
[ ] release downloads work
[ ] SHA256s match
```

### GitHub testing

```text
[ ] required CI workflow green
[ ] manual full-inference workflow attempted
[ ] if Metal available and L001/L002 exist, full simulator inference attempted without masked failures
[ ] if Metal unavailable, explicit SKIP recorded
```

### Physical XS Max

If agent has no physical device:

```text
A12 DEVICE ACCEPTANCE = PENDING
```

That is acceptable.

Do not invent a pass.

The repository must contain everything the user needs to perform that final acceptance test from their MacBook.

---

# 58. Final report to the user

When finished, provide:

```text
GitHub repository URL
final commit SHA
model release URL/tag
Xcode version actually used in CI
iOS build SDK
deployment target
CI workflow URLs
required CI status
full-inference workflow status
whether Metal was available in CI
whether a complete image was generated in CI
pack SHA verification result
unit test count/pass count
simulator test result
known unresolved issues
exact steps for the user's first XS Max test
```

Also state separately:

```text
A12 DEVICE TESTED: YES/NO
```

If NO, list exactly what remains unverified:

```text
MPS fp16 accuracy on Apple5
A12 memory/jetsam
A12 performance
A12 watchdog behavior
A12 thermal behavior
```

Do not bury these behind general wording.

---

# 59. Definition of success

Success is NOT:

```text
"the app compiles"
```

Success is a repository containing a coherent, model-specific, tested implementation that gets as far toward complete inference as the available hardware genuinely allows, has a green Xcode 26.3 CI build, and is ready for the user to clone, sign, install and run on their iPhone XS Max.

Optimize for:

```text
correctness first
first image second
measured A12 optimization third
```

Do not optimize hypothetical problems before the baseline works.
