# Hermes P0 Bug-Fix Runbook — W8-v2 Numerical Range + Fused AdaLN GPU Fault

**Date:** 2026-08-16  
**Project:** AnimaXS  
**Priority:** P0 / emergency correctness  
**Target:** current repository state corresponding to the user-supplied `AnimaXS-main(1).zip`  
**Primary goal:** make W8-v2 reliable on the physical iPhone XS Max, then fix the known fused LayerNorm+AdaLN+to-half GPU-fault bug.  
**Out of scope:** new performance architecture, GPU-private scratch, multi-block GPU preparation, dequant cache, new QGEMM work, thermal policy changes, repacking weights.

---

# 0. Read this first — this is an implementation plan, NOT an investigation assignment

The root causes in this runbook have already been investigated against the supplied repository and the physical-device diagnostics.

Hermes must treat the findings below as implementation requirements.

Do **not** give an execution subagent an open-ended instruction such as:

- "investigate the W8 bug";
- "figure out why W8 is unstable";
- "find the GPU crash";
- "try different numerical modes";
- "experiment until something works";
- "see whether repacking helps";
- "look for optimizations while you are here."

Those are planner/investigator tasks and are already resolved enough for this patch.

If the live repository no longer matches an exact code anchor listed below, Hermes may inspect only enough surrounding code to map the already-specified fix to the renamed/refactored location. If the underlying architecture has materially changed, **stop and report the contradiction instead of improvising a new design**.

---

# 1. Hermes context-management rules — non-negotiable

Hermes is the coordinator/reviewer. Coding should be delegated to tightly scoped subagents.

## One subagent at a time

1. Only **ONE** subagent may exist/work at a time.
2. Each subagent begins with **ONE task only** in its context.
3. Do not give a subagent this entire project history.
4. Give it only:
   - the task title/number;
   - exact files to change;
   - the required code shape;
   - acceptance tests;
   - explicit "do not touch" constraints.
5. A subagent may inspect direct call sites required for its assigned change.
6. A subagent must not investigate unrelated issues.
7. A subagent must not start another subagent.
8. A subagent must not merge or push to `main`.
9. After the subagent returns, Hermes must inspect the actual diff itself.
10. Never accept "tests pass" or an agent summary as proof without inspecting the relevant source and CI result.

The purpose is to keep each implementation context tiny and prevent compaction/history drift.

## Re-read this file often

Save this file in the repository root as:

`HERMES_W8_P0_BUGFIX_PLAN.md`

Hermes must re-read:

- §0 and §1 before any edits;
- the relevant task section immediately before creating its subagent;
- the same task section immediately after the subagent returns;
- this plan after every context compaction/restart;
- §11 before declaring completion.

Do not rely on remembered instructions after a long coding session.

## Session ledger

Create/update:

`HERMES_W8_P0_SESSION.md`

Keep it SHORT. It should contain only:

- branch + current HEAD;
- Task 1 / 2 / 3 / 4 status: `pending`, `in progress`, `verified`, `blocked`;
- commit SHA for each verified task;
- CI run IDs + pass/fail;
- any deviation from this plan and exact reason;
- physical-device validation still owed by the user.

Do not turn the ledger into another huge report.

---

# 2. VPS / storage / infrastructure rules

The VPS is only an orchestration/editing machine.

## Do NOT do on the VPS

- no W4/W8/Qwen/VAE downloads;
- no multi-GB `.animapk` or `.bin` downloads;
- no model repacking;
- no inference;
- no large GitHub Actions artifact downloads;
- no Hugging Face repository clones containing weights;
- no Clore instance;
- no memory-heavy conversion scripts;
- no long-term storage of generated model files.

The VPS has limited RAM/disk and has already caused project interruptions before.

## Allowed on the VPS

- normal git operations;
- source edits;
- `rg`, `grep`, `git diff`, `git diff --check`;
- lightweight Swift/source inspection;
- GitHub CLI log inspection;
- workflow dispatch;
- small text artifacts.

If full inference needs packs, let the existing **GitHub Actions macOS runner download them directly**.

Do not copy the packs through the VPS.

---

# 3. Git discipline

Create a dedicated branch from the latest intended base.

Suggested branch:

```bash
git fetch origin
git switch -c fix/w8-range-and-fused-adaln origin/main
```

If a branch with that name already exists, do not overwrite it blindly.

Before editing:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/main
```

Record these in `HERMES_W8_P0_SESSION.md`.

Do not:

- work directly on `main`;
- force-push `main`;
- squash unrelated history;
- reformat unrelated files;
- "clean up" adjacent optimization code;
- change model assets.

Prefer one commit for the W8 correctness fix and one commit for the fused-kernel fault, followed by a small tests/docs commit only if necessary.

---

# 4. P0 scope lock

Until this runbook is complete:

## STOP all performance work

Do **not** implement:

- multi-block resident/preparation pipeline;
- GPU-private scratch arena;
- persistent dequant cache;
- new direct W4/W8 GEMM;
- directQuantized changes;
- hybrid backend changes;
- tile-size tuning;
- thermal behavior;
- new streaming strategy;
- mmap no-copy re-enablement;
- new fusions other than fixing the already-existing broken fused LayerNorm path;
- checkpoint/resume work;
- weight repacking.

The next optimization phase starts only after the physical W8-v2 retest is finite and completes an image.

## Do not repack W8-v2

The physical-device screenshot is using the intended W8-v2 pack:

```text
filename: anima-turbo-v1.0-xsmax-w8-v2.animapk
size:     2,232,975,360 bytes
sha256:   8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130
```

The supplied repository has the same manifest identity.

This patch is a **runtime numerical-boundary bug**, not evidence that the W8 pack itself must be regenerated.

---

# 5. Root cause A — W8-v2 production is forcing a large residual through FP16

This is the primary P0 bug.

## Device evidence

The physical iPhone reports:

```text
Numerical failure at diffusion step 1/8,
final layer (after block 28/28),
final-layer residual conversion:
value exceeded FP16 finite range
```

The subsequent warning chain is:

```text
final-layer residual conversion: value exceeded FP16 finite range
final-layer norm conversion: NaN
final-layer projection input: NaN
velocity conversion: NaN
FLOW denoised conversion: NaN
Euler update: NaN
```

This tells us the first known unsafe event is the final residual conversion.

The later NaNs are downstream consequences.

## Exact current source path

Current policy resolution:

`AnimaXS/Runtime/ModelStore/ModelManifest.swift`

```swift
enum DiTNumericsPolicy: String, Equatable {
    case w4Legacy
    case w8LegacyStabilized
    case w8BF16Experimental

    static func fromVariantID(_ id: String) -> DiTNumericsPolicy {
        id == "w8-v2" ? .w8LegacyStabilized : .w4Legacy
    }
}
```

Current mapping:

`AnimaXS/Runtime/Sampler/DiffusionSampler.swift`

```swift
case .w4Legacy:
    return (.legacy, .legacy)
case .w8LegacyStabilized:
    return (.legacy, .legacy)
case .w8BF16Experimental:
    return (.bf16Compute, .bf16Compute)
```

Current final-layer decision:

`AnimaXS/Runtime/Metal/DiTFinalLayerExecutor.swift`

```swift
private var emulatesBF16: Bool { activationNumerics == .bf16Compute }
```

and:

```swift
if emulatesBF16 {
    let boundaryFloat = buffer("final.boundary.f32", ...)
    try encodeUnary(command, "round_f32_to_bf16", residual, boundaryFloat, ...)
    try encodeLayerNorm(command, input: boundaryFloat, output: normalized)
} else {
    let residualHalf = buffer("final.residual.f16", ...)
    let boundaryFloat = buffer("final.boundary.f32", ...)
    try encodeProbeConvert(
        command, "float_to_half", residual, residualHalf,
        ..., probe: .finalResidualToHalf)
    try encodeUnary(command, "half_to_float", residualHalf, boundaryFloat, ...)
    try encodeLayerNorm(command, input: boundaryFloat, output: normalized)
}
```

Because production W8 uses `.legacy` activation numerics, it enters the second branch and executes:

```text
FP32 large residual
→ float_to_half
→ overflow above 65,504
→ Inf
→ LayerNorm NaN
```

That is exactly the device failure.

## Why the source model supports the fix

The pinned source in:

`scripts/animapk_cuda/comfy_stub/comfy/ldm/cosmos/predict2.py`

explicitly says the model's residual stream contains large values, keeps the residual stream FP32 when FP16 compute is used, and notes that clamping into FP16 produces quality degradation / artifacts.

Immediately before the final layer, the source casts the residual to the cross-attention dtype.

Therefore:

- clamping to ±65,504 is **not** an acceptable fix;
- silently allowing FP16 Inf is obviously not acceptable;
- switching the entire model back to the incomplete `w8BF16Experimental` path is **not** the required fix.

We need to separate:

```text
A. final residual range behavior
```

from:

```text
B. whole-model activation/attention BF16 emulation
```

Production W8 needs A without automatically enabling B.

---

# 6. Task 1 — Decouple the final residual boundary from activation numerics

## Delegate ONLY this task to Subagent 1

### Goal

Production W8-v2 must retain:

```text
block activation numerics: legacy
attention numerics:        legacy
```

for now, while its **final residual boundary** becomes:

```text
BF16 RNE semantics in FP32 storage
```

The W8 final residual must never be converted to FP16 merely because block activation numerics are `.legacy`.

W4 behavior must remain exactly unchanged.

---

## Required design

Add an explicit final-residual-boundary concept.

Suggested enum:

```swift
enum FinalResidualBoundary: String, Equatable {
    case fp16Legacy
    case bf16RNEInFP32
}
```

The name can differ slightly, but it must make the unit/semantic distinction obvious.

Do **not** use a vague boolean like:

```swift
isW8
```

inside the final executor.

Do **not** make `DiTFinalLayerExecutor` inspect a model filename.

Do **not** infer this from quantization storage type.

The pack-derived `DiTNumericsPolicy` remains the source of truth.

---

## Preferred resolution shape

Keep the existing attention/activation mapping but add one central resolver for the final residual boundary.

For example in `DiffusionSampler.swift`:

```swift
static func resolvedFinalResidualBoundary(
    for policy: DiTNumericsPolicy
) -> FinalResidualBoundary {
    switch policy {
    case .w4Legacy:
        return .fp16Legacy

    case .w8LegacyStabilized:
        return .bf16RNEInFP32

    case .w8BF16Experimental:
        return .bf16RNEInFP32
    }
}
```

Then in `DiffusionSampler.init`:

```swift
let resolvedFinalResidualBoundary: FinalResidualBoundary

if let numerics {
    (resolvedActivation, resolvedAttention) =
        Self.resolvedNumerics(for: numerics)

    resolvedFinalResidualBoundary =
        Self.resolvedFinalResidualBoundary(for: numerics)
} else {
    resolvedActivation = activationNumerics
    resolvedAttention = attentionNumerics

    // Preserve the previous explicit experimental construction behavior:
    // callers explicitly requesting bf16Compute still need the range-safe
    // final boundary.
    resolvedFinalResidualBoundary =
        activationNumerics == .bf16Compute
            ? .bf16RNEInFP32
            : .fp16Legacy
}
```

Pass that boundary into `DitForward`, and then into `DiTFinalLayerExecutor`.

---

## Required call-chain changes

Primary files:

- `AnimaXS/Runtime/Sampler/DiffusionSampler.swift`
- `AnimaXS/Runtime/Metal/DitForward.swift`
- `AnimaXS/Runtime/Metal/DiTFinalLayerExecutor.swift`
- `AnimaXS/Runtime/ModelStore/ModelManifest.swift` comments/docs only as needed
- `AnimaXSTests/ModelStoreTests.swift`
- `AnimaXSTests/DiTFinalLayerExecutorTests.swift`
- `AnimaXSTests/FullInferenceTests.swift`
- `.github/workflows/full-inference-refine.yml`

### `DitForward`

Add a parameter that carries the resolved final boundary:

```swift
init(
    context: MetalContext,
    file: AnimapkFile,
    attentionNumerics: AttentionNumerics = .legacy,
    activationNumerics: ActivationNumerics = .legacy,
    finalResidualBoundary: FinalResidualBoundary = .fp16Legacy,
    monitor: NumericalMonitor? = nil,
    optimization: InferenceOptimizationConfig = .currentBaseline,
    crossKVCache: CrossKVCache? = nil
) throws
```

Use it only when constructing `DiTFinalLayerExecutor`.

Do not feed this option into block execution.

### `DiTFinalLayerExecutor`

Store:

```swift
private let finalResidualBoundary: FinalResidualBoundary
```

Do **not** use `activationNumerics == .bf16Compute` to choose the large-residual storage boundary.

Replace the current final residual branch with explicit semantics:

```swift
switch finalResidualBoundary {
case .bf16RNEInFP32:
    let boundaryFloat = buffer(
        "final.boundary.f32",
        Self.tokens * Self.dim,
        Float.self)

    try encodeUnary(
        command,
        "round_f32_to_bf16",
        residual,
        boundaryFloat,
        Self.tokens * Self.dim)

    try encodeLayerNorm(
        command,
        input: boundaryFloat,
        output: normalized)

case .fp16Legacy:
    let residualHalf = buffer(
        "final.residual.f16",
        Self.tokens * Self.dim,
        Float16.self)

    let boundaryFloat = buffer(
        "final.boundary.f32",
        Self.tokens * Self.dim,
        Float.self)

    try encodeProbeConvert(
        command,
        "float_to_half",
        residual,
        residualHalf,
        Self.tokens * Self.dim,
        probe: .finalResidualToHalf)

    try encodeUnary(
        command,
        "half_to_float",
        residualHalf,
        boundaryFloat,
        Self.tokens * Self.dim)

    metrics?.recordConversionBytes(
        UInt64(Self.tokens * Self.dim * MemoryLayout<Float>.stride))

    try encodeLayerNorm(
        command,
        input: boundaryFloat,
        output: normalized)
}
```

Important:

- Keep `activationNumerics` and `emulatesBF16` for the *other* existing experimental compute-boundary behavior.
- Only the **large final residual entry boundary** is being decoupled here.
- Do not globally turn W8 block activations into `.bf16Compute`.
- Do not delete `w8BF16Experimental`.
- Do not change W4's final FP16 boundary.

---

## Critical testing hole to fix

Current `FullInferenceTests.testCanonicalProductionInference` does this:

1. resolves `DiTNumericsPolicy`;
2. converts the policy to only `(activation, attention)`;
3. constructs `DiffusionSampler` with explicit activation/attention.

That means a new policy-specific final residual boundary could accidentally be bypassed in CI.

This must be fixed.

### Required canonical full-inference behavior

If there are NO explicit diagnostic numerical overrides:

```swift
let sampler = try DiffusionSampler(
    context: context,
    file: AnimapkFile(url: ditURL),
    numerics: policy)
```

This is the production-equivalent path.

Only use explicit `attentionNumerics` / `activationNumerics` construction when the diagnostic configuration actually requests overrides.

Do not resolve the production policy into two fields and then throw away the final-boundary part.

### Stress test

`testNumericalStressAcrossSeeds` currently constructs the sampler with explicit:

```swift
attentionNumerics:
activationNumerics:
```

Change its default/no-override path to use the pack variant policy as well.

It may keep explicit override behavior when requested.

The stress test must therefore know the matrix variant (`ANIMAXS_DIT_VARIANT`) and derive:

```swift
let policy = DiTNumericsPolicy.fromVariantID(variantID)
```

just like canonical production inference.

---

# 7. Task 1 tests — must be written before considering W8 fixed

## 7.1 Policy resolution tests

Update/add tests in:

`AnimaXSTests/ModelStoreTests.swift`

Required assertions:

```swift
XCTAssertEqual(
    DiTNumericsPolicy.fromVariantID("w8-v2"),
    .w8LegacyStabilized)

let w8 = DiffusionSampler.resolvedNumerics(
    for: .w8LegacyStabilized)

XCTAssertEqual(w8.activation, .legacy)
XCTAssertEqual(w8.attention, .legacy)

XCTAssertEqual(
    DiffusionSampler.resolvedFinalResidualBoundary(
        for: .w8LegacyStabilized),
    .bf16RNEInFP32)

XCTAssertEqual(
    DiffusionSampler.resolvedFinalResidualBoundary(
        for: .w4Legacy),
    .fp16Legacy)
```

This proves W8 is **not** enabling the whole experimental BF16 path.

---

## 7.2 Synthetic >FP16-range final-layer test

The current test only proves:

```text
activationNumerics = .bf16Compute
```

keeps large residuals finite.

That is insufficient because production W8 is `.legacy/.legacy`.

Replace/extend it so it tests the actual new contract:

```text
activationNumerics:   legacy
final residual mode:  bf16RNEInFP32
residual magnitudes:  60,000 / 70,000 / 100,000 / 280,000
```

For each magnitude:

- alternate positive/negative values so LayerNorm has variance;
- run the final layer;
- assert every output value is finite.

The test name should say **production W8 final boundary**, not "experimental BF16."

For example:

```swift
func testW8ProductionFinalBoundaryKeepsLargeResidualFinite() async throws
```

Keep a separate explicit BF16 experimental test only if it adds value.

---

## 7.3 Prove W4 still uses its legacy boundary

Add/keep a small contract test that W4 resolves to `.fp16Legacy`.

Do not intentionally feed >65,504 through W4 and require finite results; that would redefine W4 behavior.

The requirement is:

```text
W4 path unchanged
```

not:

```text
make W4 behave like W8
```

---

## 7.4 CI must test the actual production policy

In `.github/workflows/full-inference-refine.yml`, keep:

```text
w4-v2 → w4Legacy
w8-v2 → w8LegacyStabilized
```

and add a second marker check for the final residual boundary.

Have the test print something stable such as:

```text
FULL_FINAL_RESIDUAL_BOUNDARY=fp16Legacy
```

for W4 and:

```text
FULL_FINAL_RESIDUAL_BOUNDARY=bf16RNEInFP32
```

for W8.

Then gate it:

```bash
case "${{ matrix.variant }}" in
  w4-v2)
    EXPECTED_DIT_NUMERICS_POLICY=w4Legacy
    EXPECTED_FINAL_BOUNDARY=fp16Legacy
    ;;
  w8-v2)
    EXPECTED_DIT_NUMERICS_POLICY=w8LegacyStabilized
    EXPECTED_FINAL_BOUNDARY=bf16RNEInFP32
    ;;
esac

grep -q \
  "FULL_DIT_NUMERICS_POLICY=$EXPECTED_DIT_NUMERICS_POLICY" \
  "$LOG"

grep -q \
  "FULL_FINAL_RESIDUAL_BOUNDARY=$EXPECTED_FINAL_BOUNDARY" \
  "$LOG"
```

This prevents a future false-green where CI says "W8 policy" but actually constructs a sampler that bypasses part of that policy.

---

# 8. Task 2 — Improve numerical evidence for the physical W8 retest

## Delegate ONLY this task to Subagent 2

This task is deliberately small.

The monitor already records `maxAbs`, but the normal metrics warning text throws the magnitude away.

Current `warningDetails()` builds a new `Stats` object containing only flags:

```swift
var stats = Stats()
stats.flags = flags
details.append("\(probe.stageLabel): \(stats.condition)")
```

So the physical report says only:

```text
value exceeded FP16 finite range
```

instead of telling us whether the input was ~66K, 100K, or 300K.

## Required change

Use the real stats for that slot, then mask its flags to the serious set:

```swift
var stats = stats(at: probe.rawValue, raw: raw)
stats.flags = flags
```

Append `maxAbs` when it is finite/nonzero.

Example formatting:

```text
final-layer residual conversion:
value exceeded FP16 finite range,
maxAbs=103424
```

Do not make this verbose for every healthy probe.

Only include the magnitude in warning detail for a probe that has a serious condition.

Add/modify `NumericalMonitorTests` and `GenerationMetricsTests` so:

- the condition remains present;
- the actual magnitude is present;
- a healthy run still reports `Numerical warnings: 0`;
- monitoring OFF still says `not collected`.

This instrumentation is part of the P0 validation because it gives the next physical failure enough evidence to continue without another blind build.

---

# 9. Root cause B — fused LayerNorm+AdaLN+to-half uses the wrong offset unit

This is the separate fatal-GPU-fault bug.

## Exact current host code

`AnimaXS/Runtime/Metal/DiTBlockExecutor.swift`

currently contains:

```swift
var modulationOffset =
    UInt32(Self.dim * MemoryLayout<Float>.stride)
```

For `Self.dim == 2048` this passes:

```text
8192
```

to the shader.

## Exact current shader interpretation

`AnimaXS/Shaders/AnimaKernels.metal`

does:

```metal
device const float *scale =
    modulation + modulationOffset;
```

That is pointer arithmetic in **Float elements**, not bytes.

So the shader interprets `8192` as:

```text
8192 floats
```

not:

```text
8192 bytes = 2048 floats
```

The AdaLN modulation buffer has only:

```text
6144 floats
```

Therefore the fused kernel can read beyond the end of the buffer.

That is a credible and direct explanation for the observed fatal A12 GPU fault.

The non-fused path does not have this bug because Swift's `setBuffer(... offset:)` API explicitly expects a byte offset.

The fused kernel's integer argument does not.

---

# 10. Task 3 — Fix and test fused LayerNorm+AdaLN+to-half

## Delegate ONLY this task to Subagent 3

### Required source fix

In:

`AnimaXS/Runtime/Metal/DiTBlockExecutor.swift`

change the fused shader's modulation offset to an **element offset**:

```swift
var modulationOffset = UInt32(Self.dim)
```

not:

```swift
UInt32(Self.dim * MemoryLayout<Float>.stride)
```

Update the nearby comment too.

Current stale comment says effectively:

```text
modulationOffset = dim*4
```

It must say that the shader ABI receives the offset in **float elements**, therefore:

```text
scale starts at element dim
```

Do not alter the shader to reinterpret the offset as bytes unless there is a compelling compile-time reason. The smallest correct fix is to pass the unit the shader already expects.

Both kernels share this contract:

```text
dit_layernorm_modulate_to_half
dit_layernorm_modulate_to_half_probe
```

so the host fix must cover both.

---

## Add a pure ABI helper to prevent regression

Do not leave this as another magic integer.

Suggested:

```swift
static func fusedModulationElementOffset(
    columns: Int
) -> UInt32 {
    UInt32(columns)
}
```

or equivalent.

Use that helper inside `encodeFusedNormModulate`.

Then add an XCTest:

```swift
XCTAssertEqual(
    DiTBlockExecutor.fusedModulationElementOffset(columns: 2048),
    2048)
```

The test exists specifically to prevent a future byte/element-unit regression.

Keep the helper internal/testable; do not expose it as app API.

---

## Add a synthetic fused-kernel parity test

Add a small Metal test in:

`AnimaXSTests/DiTBlockExecutorTests.swift`

or a tightly related test file.

The test should:

1. allocate residual data for one or a few rows with `columns = 2048`;
2. allocate exactly `6144` modulation floats;
3. place:
   - shift at `0..<2048`;
   - scale at `2048..<4096`;
   - gate/unused third chunk at `4096..<6144`;
4. execute `dit_layernorm_modulate_to_half`;
5. compare output against the CPU reference:
   ```text
   LayerNorm(residual)
   × (1 + scale)
   + shift
   → Float16
   ```
6. assert command buffer completes without error;
7. assert all output values are finite;
8. compare with a reasonable exact/near-FP16 tolerance.

The important thing is that the buffer is **exactly** the real three-chunk length. A bogus element offset of 8192 must not be able to hide behind an oversized test allocation.

If convenient, add the same test for the probe kernel, but do not create a large testing framework merely for that. Both kernels use the same offset ABI.

---

## Do not promote the fused option to a production default

The bug fix does **not** prove performance benefit.

Keep:

```text
InferenceOptimizationConfig.currentBaseline.fusedNormModulation = false
```

Do not make fused mode automatically enabled.

The physical user can benchmark it later after correctness is confirmed.

---

# 11. Task 4 — Documentation / state reconciliation

## Delegate ONLY this task to Subagent 4 after code + tests are green

The repo currently contains historical text saying production W8 "legacy stabilized" is the correct resolved behavior.

That history should not be erased, but current state must be unambiguous.

## `OPTIMIZATION_DECISIONS.md`

Append, do not rewrite history.

Suggested decisions:

### D005 — W8 final residual range is independent of block activation numerics

Record:

- physical XS Max showed `finalResidualToHalf` overflow after block 28/28;
- production W8 remains `.legacy/.legacy` for block attention/activation;
- production W8 final residual boundary is now BF16 RNE in FP32;
- W4 remains legacy FP16;
- full `w8BF16Experimental` is still not promoted;
- no W8 repack was required.

### D006 — Fused AdaLN shader offset is measured in float elements

Record:

- old host passed bytes (`2048*4=8192`);
- shader added the number to `float*`, so it read as 8192 floats;
- real modulation buffer is 6144 floats;
- host now passes 2048;
- fused option remains non-default until physical benchmarking.

## `TODO.md`

At the top, create a short current P0 section.

Do not leave stale text claiming:

```text
W8-v2 production resolves to stabilized legacy numerics
```

if that sentence could be read to mean its final residual is still forced through FP16.

Clarify:

```text
block attention/activation: legacy
final residual boundary: BF16-RNE-in-FP32
```

Mark physical XS Max W8 retest as pending.

## Historical plans/reports

Do not rewrite old reports to pretend the old decision was never made.

Historical documents may remain historical.

If a current root-level plan explicitly instructs agents to keep the unsafe W8 final FP16 boundary, mark that instruction as **superseded** rather than silently deleting all context.

---

# 12. Required validation sequence

Do not burn GitHub runner time after every tiny edit.

## After each subagent

Hermes runs:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Then inspect the exact files touched.

No full CI yet unless the task caused a compile-sensitive API change and local inspection cannot establish consistency.

---

## Static W8 checks

After Task 1:

```bash
git grep -n \
  "Production W8-v2 resolves to legacy numerics and does NOT take this path" \
  -- AnimaXS AnimaXSTests
```

Expected: no stale current-source assertion that production W8 must take the FP16 residual branch.

Check that the old W8 pack identity is unchanged:

```bash
git diff origin/main -- AnimaXS/Runtime/ModelStore/ModelManifest.swift
```

There must be no W8 size/hash modification unless it is comment-only.

Check production still resolves W8 to the non-experimental policy:

```bash
git grep -n "w8LegacyStabilized" \
  AnimaXS/Runtime/ModelStore/ModelManifest.swift \
  AnimaXS/Runtime/Sampler/DiffusionSampler.swift
```

Check the full-inference normal path actually supplies `numerics: policy`.

---

## Static fused-kernel checks

After Task 3:

```bash
git grep -n \
  'modulationOffset = UInt32(Self.dim \* MemoryLayout<Float>.stride)' \
  -- AnimaXS
```

Expected: **no result**.

Confirm the host uses:

```text
2048 element offset
```

not 8192.

Confirm neither Metal fused kernel changed its pointer interpretation unexpectedly.

---

## CI Gate A — normal CI

Run the project's normal CI after Tasks 1–3 and their targeted tests are complete.

Required:

- project consistency: PASS
- iPhone build: PASS
- simulator/tests: PASS
- no new fixture-gated test silently skipped if it is supposed to be unit/synthetic
- no Xcode project drift

Record run ID in the session ledger.

Do not proceed on red CI.

Fix only direct consequences of this runbook; do not opportunistically work on unrelated warnings.

---

## CI Gate B — W4/W8 full inference

Dispatch:

`.github/workflows/full-inference-refine.yml`

Required for W4:

```text
FULL_INFERENCE=PASS
FULL_DIT_NUMERICS_POLICY=w4Legacy
FULL_FINAL_RESIDUAL_BOUNDARY=fp16Legacy
```

Required for W8:

```text
FULL_INFERENCE=PASS
FULL_DIT_NUMERICS_POLICY=w8LegacyStabilized
FULL_FINAL_RESIDUAL_BOUNDARY=bf16RNEInFP32
```

Also require:

- no non-finite latent;
- no final-layer residual conversion overflow on the W8 production path;
- 8 completed steps;
- VAE completes;
- generated image artifact exists;
- W4 does not regress.

Do not claim physical-device correctness merely because the macOS Metal runner passes.

---

# 13. Physical iPhone XS Max acceptance — user performs this

Hermes has no real iPhone.

The final report must explicitly say that physical validation remains pending.

## First physical run — W8-v2 baseline

Use the known W8-v2 pack:

```text
anima-turbo-v1.0-xsmax-w8-v2.animapk
8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130
2,232,975,360 bytes
```

Use a conservative baseline first:

- `dequantizedMPS`;
- no mmap no-copy;
- fused LayerNorm OFF;
- numerical monitor ON;
- no directQuantized/hybrid;
- no other experimental backend.

Required:

- completes all 8 diffusion steps;
- VAE runs;
- image is produced;
- no `final-layer residual conversion` FP16 overflow;
- no NaN/Inf warning chain;
- no fatal GPU fault.

Save the metrics text.

## Second physical run — currently useful optimized-safe settings

After baseline W8 completes, retest with the already-known non-fatal performance controls the user wants to keep evaluating, for example:

- 1024 linear tiles;
- 1024 attention tiles;
- direct MPS I/O ON;
- ping-pong ON;
- cross-KV cache if desired;
- fused LayerNorm still OFF for this second W8 proof.

The point is to show the W8 final residual fix is independent of those toggles.

## Fused-kernel physical smoke

Only after W8 baseline correctness is proven:

- start a fresh app process if prior GPU fault poisoning exists;
- enable **only** `Fused LayerNorm+AdaLN+to-half` relative to a known-good configuration;
- run one generation;
- confirm no page fault / fatal Metal error;
- compare image plausibility and numerical warnings.

Do not benchmark its speed yet.

This runbook is a correctness phase.

---

# 14. Explicit things NOT to "fix"

The execution agent must not reinterpret this patch into a larger rewrite.

## Do not clamp W8 residuals

No:

```swift
min(max(value, -65504), 65504)
```

The pinned source explicitly warns that clamping produces quality degradation / artifacts.

## Do not switch all production W8 to `w8BF16Experimental`

That mode has broader FP16-backed limitations and is not the requested solution.

## Do not make W4 use the W8 boundary

W4 is the current known-good control.

## Do not repack weights

The runtime pack identity is already correct.

## Do not remove numerical monitoring

The monitor caught this bug.

## Do not revive mmap no-copy

It previously caused an A12 page fault.

## Do not revive directQuantized/hybrid

It is roughly 10× slower on the physical device and is quarantined.

## Do not solve the performance problem in this branch

The next optimization work will be based on the corrected W8 baseline.

---

# 15. Why this patch is intentionally narrow

The most recent physical W4 benchmark showed:

```text
Generation:                  238.7 s
DiT:                         215.5 s
Measured GPU command time:   211.0 s
Metal encode time:             1.9 s
Weight copy/load CPU work:    22.7 s (overlaps ping-pong GPU time)
Peak Metal allocation:         0.38 GB
```

Those numbers strongly suggest the next optimization phase should target GPU execution / memory traffic rather than another batch of tiny host-side changes.

The same benchmark recorded:

```text
dequantized weight bytes written: 25,312 MB
transpose bytes:                   6,272 MB
conversion bytes:                 15,458 MB
```

That future phase is likely to investigate:

- GPU-private scratch;
- GPU preparation/dequant look-ahead;
- bounded persistent dequantized-weight cache;
- larger resident working set without loading the full model.

**None of that belongs in this P0 branch.**

First make W8 trustworthy.

---

# 16. Suggested subagent prompts

Hermes should not paste the entire runbook into a subagent.

Use prompts shaped like these.

## Subagent 1 — W8 final residual boundary

```text
You have ONE task only.

Implement the planner-specified W8 production final-residual boundary fix.

Root cause is already established:
production W8-v2 uses legacy block activation/attention numerics, and
DiTFinalLayerExecutor currently uses activationNumerics == .bf16Compute to
decide whether the large final residual avoids FP16. Therefore production W8
falls through float_to_half and overflows above 65,504.

Required:
- add explicit FinalResidualBoundary (fp16Legacy / bf16RNEInFP32 or equivalent);
- W4 policy -> fp16Legacy;
- w8LegacyStabilized -> bf16RNEInFP32;
- w8BF16Experimental -> bf16RNEInFP32;
- thread it DiffusionSampler -> DitForward -> DiTFinalLayerExecutor;
- use this explicit boundary ONLY for the large final residual;
- keep W8 block attention/activation legacy;
- do not promote full w8BF16Experimental;
- do not touch W4 semantics;
- update canonical FullInferenceTests so the no-override path constructs the
  sampler with numerics: policy and therefore cannot bypass the new policy field;
- update the stress test similarly;
- add production-W8 >65,504 synthetic final-layer coverage;
- add W4/W8 boundary resolution tests.

Do not investigate alternative fixes.
Do not repack models.
Do not optimize unrelated code.
Do not push/merge.
Return the diff summary and exact tests you changed.
```

## Subagent 2 — numerical magnitude telemetry

```text
You have ONE task only.

Improve NumericalMonitor warning detail so a serious warning includes the
actual recorded maxAbs value.

Root cause is already known:
warningDetails() reconstructs Stats with only flags and discards the maxAbs
already stored in the monitor buffer.

Required:
- use stats(at:raw:) for the probe then mask stats.flags to serious flags;
- append maxAbs to serious warning detail when meaningful;
- update NumericalMonitorTests / GenerationMetricsTests;
- preserve "warnings: 0" and monitoring-off behavior exactly.

No unrelated diagnostics redesign.
No performance work.
Do not push/merge.
```

## Subagent 3 — fused AdaLN GPU fault

```text
You have ONE task only.

Fix the fused LayerNorm+AdaLN+to-half modulation offset ABI bug.

Root cause is already established:
Swift passes Self.dim * MemoryLayout<Float>.stride = 8192, while the Metal
kernel does `modulation + modulationOffset` on a float pointer, so it treats
8192 as float elements. The real modulation buffer is only 6144 floats.

Required:
- pass the offset in float elements: Self.dim = 2048;
- update stale comments;
- add a small internal pure helper for the element offset and unit-test it;
- add a synthetic Metal fused-kernel parity test using exactly 6144 modulation
  floats with shift[0..<2048], scale[2048..<4096], third chunk[4096..<6144];
- assert command completion, finite output, and parity with CPU math;
- leave fusedNormModulation OFF in the production baseline.

Do not redesign the shader ABI.
Do not benchmark.
Do not touch other optimization work.
Do not push/merge.
```

## Subagent 4 — docs/state

```text
You have ONE task only.

Reconcile the current docs after the W8 final-boundary and fused-offset fixes.

Append decisions; do not rewrite history.
Update TODO current state.
Clarify:
- W8 block attention/activation remains legacy;
- W8 final residual is BF16-RNE-in-FP32;
- W4 remains FP16 legacy;
- full experimental BF16 is still not production;
- fused modulation offset is in float elements;
- physical XS Max validation remains pending.

Do not change runtime code.
Do not push/merge.
```

---

# 17. Final Hermes review checklist

Before Hermes says "done", it must independently verify every item below.

## Source correctness

- [ ] W8-v2 manifest hash/size unchanged.
- [ ] W8-v2 still resolves to `.w8LegacyStabilized`.
- [ ] `.w8LegacyStabilized` still resolves block activation/attention to `.legacy/.legacy`.
- [ ] W8-v2 final residual boundary resolves to BF16-RNE-in-FP32.
- [ ] W4 final residual boundary remains legacy FP16.
- [ ] `DiTFinalLayerExecutor` no longer chooses the large-residual boundary solely from `activationNumerics`.
- [ ] Production W8 cannot execute `float_to_half(residual)` at final-layer entry.
- [ ] Full experimental BF16 path was not promoted globally.
- [ ] No W8 clamping was added.
- [ ] No model repack was added.

## Full-inference test correctness

- [ ] Canonical no-override path uses `numerics: policy`.
- [ ] Stress no-override path uses the pack-derived policy.
- [ ] W4/W8 CI logs print the explicit final-boundary marker.
- [ ] Workflow gates both the policy and boundary marker.
- [ ] Synthetic W8 residuals above 65,504 remain finite.

## Fused-kernel correctness

- [ ] Host offset is 2048 float elements, not 8192.
- [ ] Stale `dim*4` comment is removed/corrected.
- [ ] Unit test locks the offset unit.
- [ ] Synthetic kernel test uses exactly 6144 modulation floats.
- [ ] Fused mode remains default OFF.

## Repository discipline

- [ ] No unrelated optimization changes.
- [ ] No directQuantized revival.
- [ ] No mmap no-copy revival.
- [ ] No checkpoint work.
- [ ] No large VPS downloads.
- [ ] `git diff --check` clean.
- [ ] working tree understood/clean after commits.
- [ ] TODO/decisions reflect the actual code.
- [ ] session ledger has CI run IDs.

## CI

- [ ] normal CI green.
- [ ] full-inference W4 green.
- [ ] full-inference W8 green.
- [ ] W8 policy marker correct.
- [ ] W8 final-boundary marker correct.
- [ ] no required test silently skipped.

## Physical validation status

- [ ] Final report explicitly says physical XS Max retest is still required.
- [ ] Hermes does not claim "device fixed" from macOS/simulator CI alone.

---

# 18. Completion report format

When all source/CI work is finished, Hermes should return a short report in this exact spirit:

```text
P0 W8 + fused-kernel source patch complete.

Branch:
HEAD:

Commits:
- <sha> W8 final-residual boundary
- <sha> numerical magnitude telemetry
- <sha> fused AdaLN offset fix/tests
- <sha> docs/state (if separate)

Normal CI:
- run:
- project-consistency:
- iphone-build:
- simulator-tests:

Full inference:
- run:
- W4:
  policy:
  final boundary:
  result:
- W8:
  policy:
  final boundary:
  result:

Verified source invariants:
- W8 blocks remain legacy/legacy
- W8 final residual no longer enters FP16
- W4 path unchanged
- W8 pack identity unchanged
- fused modulation offset is 2048 float elements
- fused baseline remains OFF

Physical XS Max:
PENDING — user must perform the device acceptance run from §13.

No performance architecture changes were made in this branch.
```

Do not pad the report with speculative optimization discussion.

---

# 19. Definition of done

This branch is **source/CI complete** only when:

1. production W8-v2 has a range-safe final residual boundary without enabling the incomplete whole-model BF16 experimental path;
2. synthetic >65,504 final residual values remain finite for the production W8 boundary;
3. W4 final-boundary behavior is unchanged;
4. full-inference CI actually exercises the complete production W8 policy, including the new boundary;
5. normal CI is green;
6. W4/W8 full-inference matrix is green;
7. the fused LayerNorm+AdaLN offset is fixed from byte-count semantics to float-element semantics;
8. a synthetic fused-kernel test covers the real 6144-float modulation layout;
9. all unrelated optimization work stayed frozen;
10. docs/TODO/decisions accurately reflect the new state.

The project is **device validated** only after the user runs W8-v2 on the physical iPhone XS Max and confirms:

```text
8/8 diffusion steps
finite output
VAE completes
real image produced
no final-residual FP16 overflow
no NaN warning chain
no fatal GPU fault
```

Only after that should the project return to the high-impact optimization work.
