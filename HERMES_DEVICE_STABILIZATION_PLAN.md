# Hermes Execution Plan — AnimaXS Device Stabilization

**Date:** 2026-08-16  
**Target:** current `main` from the user-supplied repository archive / corresponding public repository state  
**Primary goal:** restore one trustworthy, repeatable on-device inference baseline before doing more performance experiments.

---

# 0. Read this first — execution rules are non-negotiable

This is an **implementation runbook**, not an investigation assignment. The root causes and required direction have already been investigated. Do not spend an agent session rediscovering them unless a concrete implementation detail genuinely contradicts this plan.

## Hermes is the coordinator; use focused subagents for implementation

Hermes must keep its own context small and act as the orchestrator/reviewer.

**Delegate implementation work to subagents, with these hard rules:**

1. **Only ONE subagent may exist/work at a time.** Never run two coding subagents concurrently.
2. Each subagent receives **exactly ONE task** from this runbook in its starting context.
3. Do not give a subagent the whole project history, old chats, or unrelated tasks. Give it only:
   - the single task number/title;
   - the exact required behavior from that section;
   - the relevant files listed in that section;
   - acceptance checks for that task.
4. A subagent must **not spawn another subagent**.
5. A subagent must not investigate unrelated bugs or redesign adjacent systems. It may inspect the specific call sites needed to implement its one task and fix direct compile/test consequences of that task.
6. The subagent must not push, merge, or rewrite repository history. Hermes reviews the work first.
7. After each subagent finishes, Hermes must independently inspect the diff and verify it against this runbook. **Do not trust a subagent summary as proof.**
8. Only after Hermes is satisfied should the task be committed and the next subagent started.

The purpose of this structure is deliberate: **every implementation agent begins with only one task in context, stays focused, finishes it, and exits before another starts.**

## Re-read this plan often

Save this file in the repository root as something easy to find, for example:

`HERMES_DEVICE_STABILIZATION_PLAN.md`

Hermes must re-read:

- this entire §0 before starting work;
- the relevant task section immediately before delegating it;
- that same section immediately after the subagent returns, before reviewing the diff;
- the plan again after any context compaction/restart;
- the final acceptance section before declaring completion.

Do not rely on memory after several tasks.

## Maintain a tiny session ledger

Create/update `HERMES_DEVICE_STABILIZATION_SESSION.md` with only concise state:

- branch + HEAD;
- task checklist (`pending / in progress / verified / blocked`);
- commit for each verified task;
- CI run IDs and outcomes;
- any deviation from this plan and the exact reason;
- remaining physical-device validation that only the user can do.

Keep this ledger short. It exists so Hermes can recover after context compaction without loading long old transcripts.

## VPS/resource rules

The VPS is a cheap orchestration box, **not an inference/weight-processing machine**.

### Absolutely do NOT do on the VPS

- Do not download W4, W8, Qwen, VAE, `.animapk`, `.bin`, model checkpoints, or large inference artifacts.
- Do not pack/repack weights on the VPS.
- Do not run heavy inference on the VPS.
- Do not use the VPS as long-term storage for artifacts.
- Do not download multi-GB GitHub Actions artifacts merely to inspect them.
- Do not clone huge Hugging Face model repositories.
- Do not start a Clore/GPU instance for this stabilization work.

### Allowed on the VPS

- normal source-code Git operations;
- small text/log retrieval;
- editing source/docs;
- `git diff`, `grep`, static source inspection;
- `gh run view`, GitHub API/log inspection;
- lightweight scripts that do not download model data or consume large RAM/disk.

If model packs or full inference are required, **GitHub Actions runners must download/use them directly** using the repository's existing workflows. Persistent large assets remain on Hugging Face / existing release storage, never on the VPS.

## Git discipline

Work on a dedicated branch, suggested:

`fix/device-stability-no-checkpoint`

Do not experiment directly on `main`.

Prefer one clean commit per verified task or one very small logical group when two edits are inseparable. Do not force-push `main`. Do not merge until final CI is green and the final report is written.

## Testing efficiency

GitHub macOS runners are slow. Do not launch a full workflow after every tiny edit.

Use this cadence:

1. local/static checks after every subagent (`git diff --check`, targeted grep, obvious source consistency);
2. **CI Gate A** after the checkpoint removal + core API refactor because it is structurally large;
3. implement the remaining stabilization tasks sequentially;
4. **CI Gate B** once all normal/unit-test changes are complete;
5. run the W4/W8 full-inference matrix once the production numerical policy and full-inference test have been aligned.

GitHub Actions jobs may run in parallel. The **one-subagent-at-a-time rule applies to agents, not CI jobs**.

---

# 1. What we are fixing and why

The physical XS Max currently exposes several independent problems that make optimization testing unreliable:

1. W8 production selects `w8BF16Emulated` / `bf16Compute`, while the green W8 full-inference CI path actually ran `legacy` attention + activation numerics. The real production W8 path therefore was never proven by that green gate.
2. The current BF16-emulation path still stores important internal tensors in FP16. Rounding an already-FP16 value to BF16-like mantissa precision **does not preserve BF16 exponent range**. This is consistent with the random W8 NaNs and with successful-but-corrupted checkerboard outputs.
3. The checkerboard can be a genuinely completed generation. It must not be dismissed as merely a stale-image UI bug. The UI *also* retains old images after later failed runs, which makes screenshots ambiguous; both facts are true.
4. P6 mmap no-copy produced a real A12 GPU page fault. It must remain disabled until a separate hardware proof exists.
5. After a fatal GPU fault the app can submit again and get `SubmissionsIgnored`; fatal Metal errors need to poison the generation context until app restart.
6. P8 `directQuantized` is approximately 10× slower on device than `dequantizedMPS`. This is expected from the current kernel structure: it processes only 8 activation rows per M tile, repeatedly revisits packed weights across `M=1024`, performs scalar custom FMAs, and uses many threadgroup barriers. It must not participate in the current optimization search.
7. Checkpoint/resume now adds complexity and per-step CPU/disk work for a feature we no longer want. The target is sub-100-second generation, so remove checkpointing entirely.
8. Import can accidentally trigger Download because adjacent default-style SwiftUI buttons live in the same `Form` row, and `ModelStore` lacks a true per-component operation token across `await` reentrancy.
9. Prompt and seed are `@State` and are lost on relaunch.
10. A fresh failed run can display a prior successful image because the coordinator intentionally retains it.
11. Diagnostics preset state can claim a preset is active after manual controls have changed.

The stabilization build must remove these sources of ambiguity before the user resumes device benchmarking.

---

# 2. Required final baseline

After this runbook is complete, a normal W8-v2 generation must resolve to:

```text
DiT pack:             w8-v2
Production numerics:  LEGACY (temporary stabilized W8 policy)
Linear backend:       dequantizedMPS
Mmap no-copy:         OFF / blocked
Checkpointing:        REMOVED
Numerical monitor:    ON by baseline default
Experimental toggles: baseline defaults unless explicitly selected
```

Important: **do not delete the BF16 experimental machinery simply to make the source smaller.** It can remain as research code, but normal production W8-v2 must no longer automatically select it.

Important: **do not delete the direct-quantized kernel solely because it is slow.** Quarantine it from production/device presets so it cannot contaminate the current optimization search. Keeping it as research/reference code is fine.

---

# 3. Task 1 — Remove checkpoint/resume from production completely

## Delegate only this task to Subagent 1

### Goal

There must be no production checkpoint persistence, Resume UI, cold-launch resume, checkpoint identity, or per-step latent snapshot copying.

The diffusion sampler may retain a generic `stepCompleted`/diagnostic hook for tests and trajectory capture. That hook is **not** a checkpoint feature and production must not use it to read/copy the latent each step.

### Primary files

- `AnimaXS/Runtime/Generation/Checkpoint.swift` — delete
- `AnimaXS/Runtime/Generation/CheckpointStore.swift` — delete
- `AnimaXS/Runtime/Generation/GenerationCoordinator.swift`
- `AnimaXS/Runtime/Generation/GenerationEngine.swift`
- `AnimaXS/Runtime/Generation/GenerationEligibility.swift`
- `AnimaXS/Runtime/Diagnostics/GenerationMetrics.swift`
- `AnimaXS/App/ContentView.swift`
- `AnimaXS/App/AnimaXSApp.swift`
- `AnimaXS/Runtime/ModelStore/ModelManifest.swift` if checkpoint-only `ModelHashes` helpers become dead
- `AnimaXSTests/GenerationCoordinatorTests.swift`
- `AnimaXSTests/GenerationEligibilityTests.swift`
- delete `AnimaXSTests/CheckpointIdentityTests.swift`
- delete `AnimaXSTests/ResumeEquivalenceTests.swift`
- any smoke tests that instantiate checkpoint-only types
- `AnimaXS.xcodeproj/project.pbxproj` must be regenerated after source/test deletion

### Required changes

1. Delete checkpoint model/store files and all persistence wiring.
2. Remove from `GenerationCoordinator`:
   - `checkpointStore`;
   - `latestCheckpoint`;
   - `generationEpoch` if it becomes checkpoint-only;
   - default checkpoint-store override;
   - cold-launch checkpoint load;
   - `canResume`;
   - `completedSteps`;
   - `resume(...)`;
   - `discardCheckpoint()`;
   - checkpoint save tasks/callbacks;
   - any special background behavior whose only purpose is preserving a checkpoint.
3. `GenerationCoordinator.generate(...)` is the only start path.
4. Background/inactive and memory warning should simply request cooperative cancellation. There is no resume state to preserve.
5. Remove Resume and Discard Checkpoint UI from `ContentView`.
6. Simplify `GenerationEligibility.evaluate(...)`: remove `canResume` and the `"A checkpoint is available"` condition.
7. Remove checkpoint text from metrics (`checkpointingEnabled`, `Checkpointing: on/off`).
8. In `GenerationEngine.generate(...)` remove production `startStep` and `checkpoint` parameters.
9. In the production diffusion call always start from step 0.
10. **Critical performance detail:** production must no longer execute this per step:

```swift
let values = readFloats(latent, count: DiffusionSampler.latentElements)
```

There is no checkpoint consumer anymore. Do not read 256 KiB of latent to CPU every diffusion step merely to maintain a dead callback.

11. Keep `DiffusionSampler.executeDiagnostic(... startStep:, stepCompleted:)` if useful to existing inference tests/trajectory capture. It is acceptable for diagnostic tests to use it. Production `GenerationEngine` should call the normal sampler with `startStep: 0` and no stepCompleted latent snapshot callback, or otherwise avoid all CPU latent snapshotting.
12. Remove `ResolvedModels.hashes` / `ModelHashes` / `ModelManifest.productionHashes()` if they become entirely checkpoint-only and no non-checkpoint code uses them.
13. Update `AnimaXSApp` lifecycle comments: backgrounding cancels; it does not retain a checkpoint.
14. Update tests to assert cancellation reaches a terminal cancelled state and a fresh Generate is available afterward.
15. There must be **zero** user-visible `Resume`, `Discard checkpoint`, or `Checkpoint: X/8` text after this task.

### Xcode project regeneration

Because Swift/test files are deleted, regenerate `AnimaXS.xcodeproj` from `project.yml`; do not leave stale PBX references.

If XcodeGen is unavailable on the VPS, use the existing `bootstrap-project.yml` workflow on the working branch (dispatch it with the branch ref) rather than downloading build tooling/large assets to the VPS. Inspect the bot commit before continuing.

### Acceptance checks

Hermes must verify:

```bash
git grep -n "GenerationCheckpoint\|CheckpointStore\|canResume\|Discard checkpoint\|Checkpoint:" -- AnimaXS AnimaXSTests
```

should return no production/checkpoint-system references. Generic English uses of “checkpoint” in historical docs or unrelated ML-source comments are not a failure.

Also verify:

```bash
git grep -n "readFloats(latent" -- AnimaXS/Runtime/Generation/GenerationEngine.swift
```

returns nothing.

Run `git diff --check`.

Then run **CI Gate A** because this is the largest API deletion. Do not proceed to Task 2 until normal CI is green or any failure has been fixed as a direct consequence of checkpoint removal.

---

# 4. Task 2 — Make W8 production use the known-good legacy numerical path

## Delegate only this task to Subagent 2

### Goal

W8-v2 must stop automatically selecting the incomplete FP16-backed BF16-emulation path. Production W8 should temporarily use legacy attention/activation numerics, matching the numerical mode that already produced coherent full-inference CI output.

Do this explicitly; do not hide the policy change behind a misleading enum name.

### Primary files

- `AnimaXS/Runtime/ModelStore/ModelManifest.swift`
- `AnimaXS/Runtime/Sampler/DiffusionSampler.swift`
- `AnimaXS/Runtime/Generation/GenerationEngine.swift` only if policy plumbing needs adjustment
- tests touching `DiTNumericsPolicy`

### Required policy shape

Use clear semantics such as:

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

Equivalent naming is acceptable only if it is equally explicit.

Map policies in `DiffusionSampler` as follows:

```text
w4Legacy             -> activation legacy, attention legacy
w8LegacyStabilized   -> activation legacy, attention legacy
w8BF16Experimental   -> activation bf16Compute, attention bf16Compute
```

The BF16 experimental case must **not** be selected by `fromVariantID("w8-v2")`.

Update misleading comments that currently claim production W8-v2 maps to BF16 emulation.

### Do NOT

- Do not clamp NaN/Inf.
- Do not silently catch numerical failures and continue.
- Do not change weights.
- Do not repack W8.
- Do not lower image/latent quality thresholds.
- Do not claim BF16 is solved.

### Tests

Add/adjust unit tests proving:

```text
variant w4    -> w4Legacy -> legacy/legacy
variant w8-v2 -> w8LegacyStabilized -> legacy/legacy
```

Also retain explicit test coverage that the experimental BF16 policy, if directly requested by a diagnostic test, still maps to bf16Compute/bf16Compute.

Hermes reviews the diff and verifies no production path still implicitly maps `w8-v2` to BF16 experimental numerics.

---

# 5. Task 3 — Fix full-inference CI so it executes the production numerical policy

## Delegate only this task to Subagent 3

### Goal

A green W4/W8 full-inference result must prove the same numerical policy that the app will use for that pack variant.

The current test reads `attention_numerics` / `activation_numerics`, defaults both to `legacy`, and ignores `ANIMAXS_DIT_VARIANT` for sampler policy. Fix that mismatch.

### Primary files

- `AnimaXSTests/FullInferenceTests.swift`
- `.github/workflows/full-inference-refine.yml`
- relevant small test helpers only

### Required test behavior

In `testCanonicalProductionInference`:

1. Read `ANIMAXS_DIT_VARIANT`.
2. If explicit diagnostic `attention_numerics` / `activation_numerics` overrides are provided, preserve the diagnostic-override path.
3. Otherwise construct the sampler from `DiTNumericsPolicy.fromVariantID(variantID)` — the same production resolver used by the app.
4. Print unambiguous markers, for example:

```text
FULL_DIT_NUMERICS_POLICY=w4Legacy
FULL_ATTENTION_NUMERICS=legacy
FULL_ACTIVATION_NUMERICS=legacy
```

or for W8:

```text
FULL_DIT_NUMERICS_POLICY=w8LegacyStabilized
FULL_ATTENTION_NUMERICS=legacy
FULL_ACTIVATION_NUMERICS=legacy
```

5. The artifact metadata must report the actually resolved numerical policy, not a stale hardcoded default.

### Workflow gate

Make `full-inference-refine.yml` verify the policy marker for each matrix variant so a future accidental mismatch cannot produce a false green:

```text
w4    -> expected w4Legacy
w8-v2 -> expected w8LegacyStabilized
```

Keep the W4 and W8 jobs parallel in the matrix.

### Do NOT

- Do not lower cosine/RMSE/RGB gates to make W8 pass.
- Do not switch to golden conditioning unless the existing workflow already deliberately uses it for that gate.
- Do not change prompt/seed/reference merely to obtain green.
- Do not use a special W8-only code path that production cannot execute.

Do not run the heavy full-inference workflow yet if later normal-source stabilization tasks are still pending; implementing and reviewing this task is enough. The final full-inference run happens in §15.

---

# 6. Task 4 — Quarantine `directQuantized` / hybrid from production device testing

## Delegate only this task to Subagent 4

### Goal

The current P8 direct packed QGEMM path is a known major regression on the XS Max (~10× slower than `dequantizedMPS`). It must not be selected accidentally by a preset, persisted setting, or combined candidate while we optimize toward sub-100 seconds.

### Why this is not a tile-tuning task

Do not spend time tuning P8 in this run. The current architecture processes a very small M tile (`TM=8`) against `M=1024`, repeatedly revisits the packed matrix, performs custom scalar FMA work, and inserts many barriers. The issue is structural enough that changing one tile constant is not the current priority.

### Primary files

- `AnimaXS/Runtime/Generation/InferenceOptimizationConfig.swift`
- `AnimaXS/App/InferenceOptimizationSettings.swift`
- `AnimaXS/App/DiagnosticsView.swift`
- `AnimaXSTests/InferenceOptimizationConfigTests.swift`
- `AnimaXSTests/InferenceOptimizationCoordinatorTests.swift` if relevant

Do **not** rewrite `LinearExecutor.swift` or `AnimaKernels.metal` as part of this task except for a tiny comment if needed. The research implementation may remain.

### Required behavior

1. `InferenceOptimizationConfig.currentBaseline.linearBackend` remains `.dequantizedMPS`.
2. Normal device settings must migrate persisted `.directQuantized` or `.hybrid` back to `.dequantizedMPS` on app launch.
3. Diagnostics must clearly disable/hide any production/device preset that selects P8 direct/hybrid, especially:
   - `directQGEMMCandidate`;
   - `allCandidate` if it currently contains hybrid/direct QGEMM.
4. Prefer a centralized compatibility/blocking reason rather than silently running a different backend after the user selects an incompatible preset.
5. A manually constructed diagnostic/research test may still instantiate `LinearExecutor(... .directQuantized)` so the kernel remains testable.
6. User-facing explanation should say the backend is temporarily disabled because it is a measured A12 performance regression, not because its correctness was proven false.

### Tests

Add tests proving:

- baseline always resolves to `dequantizedMPS`;
- persisted direct/hybrid selection is sanitized/migrated for normal app settings;
- a production Generate configuration containing direct/hybrid is rejected or made unavailable before inference;
- research-level directQuantized unit tests still compile/run where appropriate.

---

# 7. Task 5 — Disable P6 mmap no-copy and prevent fatal Metal fault cascades

## Delegate only this task to Subagent 5

### Goal

A12 produced a real `kIOGPUCommandBufferCallbackErrorPageFault` while mmap no-copy bytes were being served. P6 must not be runnable in this stabilization build. A fatal Metal fault must also make the current app process unable to submit another generation until restart.

### Primary files

- `AnimaXS/Runtime/Generation/InferenceOptimizationConfig.swift`
- `AnimaXS/App/InferenceOptimizationSettings.swift`
- `AnimaXS/App/DiagnosticsView.swift`
- `AnimaXS/Runtime/Metal/WeightStreamer.swift`
- `AnimaXS/Runtime/Animapk/MappedFile.swift` only if policy lives there
- `AnimaXS/Runtime/Generation/GenerationCoordinator.swift`
- `AnimaXS/Runtime/Generation/GenerationEligibility.swift`
- relevant tests

### P6 required behavior

1. Normal production/device configuration must not allow `noCopyWeightSource == true` to reach Metal.
2. Migrate persisted `true` to `false` at settings initialization.
3. Disable the P6 UI toggle/presets and explain that it is blocked after an A12 GPU page fault.
4. Keep the implementation as research code for later isolated hardware work.
5. Tighten `WeightNoCopyPolicy.isEligible` to require both page-aligned start and page-aligned length/end where applicable. This is correctness hardening only; **do not claim it proves the historical page fault root cause.**

### Fatal Metal context behavior

Add coordinator state such as:

```swift
@Published private(set) var metalContextPoisoned = false
```

`isMetalAvailable`/generation eligibility must become false after a fatal Metal command-buffer failure.

Classify at least these `MTLCommandBufferError.Code` values as fatal for this app process:

- `.pageFault`
- `.invalidResource`
- `.internal`

Use `NSError.domain == MTLCommandBufferErrorDomain` when available. Retain a narrow fallback for the native IOGPU error text if the device bridge does not expose the expected Metal domain/code.

After such an error:

```text
state -> failed("Fatal GPU fault. Restart AnimaXS before generating again.")
metalContextPoisoned -> true
Generate -> disabled until process restart
```

Do not automatically recreate the Metal context and do not retry the command buffer.

Checkpoint cleanup is not needed because Task 1 removed checkpointing.

### Tests

- synthetic pageFault NSError poisons context;
- invalidResource/internal poison context;
- ordinary cooperative cancellation does not poison context;
- subsequent Generate eligibility is blocked after poisoning;
- persisted no-copy true becomes false.

---

# 8. Task 6 — Fix Import accidentally triggering Download

## Delegate only this task to Subagent 6

### Goal

Tapping Import must never start a network download. Import is a local-file operation only.

### Primary files

- `AnimaXS/App/ContentView.swift`
- `AnimaXS/Runtime/ModelStore/ModelStore.swift`
- `AnimaXSTests/ModelStoreTests.swift`

### UI fix

Every action button sharing a model `Form` row must have an explicit non-row style, including Download / Import / Retry / Repair as applicable:

```swift
.buttonStyle(.borderless)
```

Do not leave adjacent default-style Buttons inside the same row.

Clear the importer target immediately when the picker completes or is cancelled:

```swift
let component = importComponent
importComponent = nil
```

then handle the result. Keep the security-scoped resource active for the entire async import as current code intends.

### Store reentrancy fix

Add a true per-component single-flight operation guard independent of published UI state, e.g.:

```swift
private var activeOperations: Set<ModelComponent> = []
```

Acquire before any destructive work or `await` and release with `defer` for:

- `download`
- `importPack`
- `repair`
- `verifyExisting` if it can overlap the same component

If `repair` calls an internal download primitive, do not acquire the same token twice. Split an internal `downloadAndInstallAssumingOperationHeld(...)` if necessary.

`repair` must acquire its token **before deleting an existing destination**.

Different model components may operate independently; only same-component overlap is blocked.

### Required regression tests

Use a suspended downloader continuation/actor-safe test seam:

1. Start Download for component A; while downloader is suspended, call Import A → Import fails with “operation already active”; downloader count remains one; import does not mutate destination.
2. Start Download A; call Repair A → Repair is rejected and does not delete the installed destination.
3. Start Import/verify A; explicit Download A cannot begin.
4. Operations on different components can still proceed independently.
5. Existing/local discovery remains side-effect free and never downloads.
6. A pure `importPack` test must use a downloader closure that fails the test if invoked; Import must still succeed locally.

Do not remove manual Download functionality. The requirement is **no automatic/accidental network action from Import or launch**.

---

# 9. Task 7 — Persist the latest prompt and seed

## Delegate only this task to Subagent 7

### Goal

The latest typed prompt and seed, including a randomized seed, survive app termination/relaunch.

### Primary file

- `AnimaXS/App/ContentView.swift`
- add a small persistence test only if the project has a suitable view-model/settings seam; do not build a large UI-test framework for this tiny feature

### Required change

Replace:

```swift
@State private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
@State private var seedText = "1337"
```

with stable `@AppStorage` keys, for example:

```swift
@AppStorage("generation.lastPrompt")
private var prompt = "masterpiece, best quality, score_7, safe, 1girl"

@AppStorage("generation.lastSeed")
private var seedText = "1337"
```

The value must persist **as the user edits it**. Do not wait until Generate succeeds.

Randomize already writes `seedText`; therefore it should automatically persist through the same binding.

Do not persist transient generation state in the same keys.

---

# 10. Task 8 — Make the image belong to the current fresh run

## Delegate only this task to Subagent 8

### Goal

A failed fresh run must not show an image from a previous successful run. This is separate from the checkerboard root cause: the checkerboard itself can be a genuinely completed bad generation, but later failures must not visually reuse it and confuse evidence.

### Primary files

- `AnimaXS/Runtime/Generation/GenerationCoordinator.swift`
- `AnimaXSTests/GenerationCoordinatorTests.swift`

### Required behavior

At the start of every accepted **fresh** Generate:

```swift
image = nil
lastMetricsText = nil // preferred if metrics are explicitly per-run
```

Clear them after guards succeed and before the run enters the first generation stage. Do not clear them merely because a blocked Generate was tapped.

When the run fails before VAE, no image should be displayed for that run.

Metrics/error should refer to the same run. If current UI architecture intentionally keeps metrics after a failure, make sure those metrics are the just-failed run, not a previous run.

### Test

- run 1 succeeds and sets an image;
- run 2 is accepted then fails before decode;
- image is nil during/after run 2;
- blocked/no-op Generate does not unexpectedly delete the prior result.

---

# 11. Task 9 — Fix Diagnostics preset-state drift and centralize compatibility blocking

## Delegate only this task to Subagent 9

### Goal

Diagnostics must never claim a named preset is active after an individual setting was manually changed, and known-disabled features must be rejected before Generate rather than failing deep in Metal.

### Primary files

- `AnimaXS/App/InferenceOptimizationSettings.swift`
- `AnimaXS/App/DiagnosticsView.swift`
- `AnimaXS/Runtime/Generation/InferenceOptimizationConfig.swift`
- `AnimaXS/Runtime/Generation/GenerationEligibility.swift`
- `AnimaXS/App/ContentView.swift`
- tests for settings/eligibility

### Preset marker behavior

1. `setPreset(...)` sets/persists the preset marker.
2. Any individual manual setting mutation clears the marker and removes its persisted key.
3. Relaunch preserves `Custom`/nil rather than falsely restoring an old preset marker.
4. `resetToBaseline()` yields exact baseline values and a clearly defined marker state. Prefer `nil/Custom` if the method semantically means “manual reset”; if a separate `setPreset(.baseline)` exists, that one may mark Baseline.
5. Diagnostics should display `Custom` when no named preset exactly describes the current controls.

### Central compatibility validator

Create one small validator, preferably in `InferenceOptimizationConfig.swift`, that can return a user-visible blocking reason for a resolved production configuration.

It must at least block:

- `noCopyWeightSource == true` (P6 disabled);
- `linearBackend != .dequantizedMPS` for normal device generation (P8 quarantined);
- an explicit experimental BF16 numerical policy combined with an attention backend/layout that does not support it.

Because normal W8 production now uses legacy numerics, do **not** incorrectly mark all W8 + strided attention combinations invalid merely because the old W8 policy used BF16. Compatibility should be based on the **actual resolved numerical policy**, not pack name alone.

Wire this same reason into:

- Generate eligibility;
- Diagnostics preset/toggle availability;
- visible blocked-reason text.

Never silently mutate a user-selected incompatible config at Generate time. Persisted bad settings may be migrated at app initialization, but an explicit current incompatible selection should be visibly blocked.

---

# 12. Task 10 — Numerical-monitor attribution cleanup (small, no new investigation)

## Delegate only this task to Subagent 10

### Goal

If a numerical failure still occurs, its label must identify the actual boundary instead of ambiguously calling multiple probes “self-attention output.” This is observability work only; do not attempt another BF16 redesign in this task.

### Primary files

- `AnimaXS/Runtime/Diagnostics/NumericalMonitor.swift`
- call sites that provide probe labels
- metrics/report formatting tests

### Required behavior

Rename ambiguous always-on gate/add probe labels so they cannot be confused with raw attention output, for example:

```text
selfGateAdd  -> self-attention branch/gate input
crossGateAdd -> cross-attention branch/gate input
mlpGateAdd   -> MLP branch/gate input
```

Keep raw Q/K/V/PV/projection labels distinct where detailed probes already exist.

If adding a per-run `detailedNumericalProbes` toggle is low-risk with the current config plumbing, it may be added, but **do not use a global mutable static flag for production runs**. It must be part of the immutable per-run optimization snapshot. If this would materially broaden the task, leave detailed probes as test-only and perform only the label cleanup now.

Do not clamp or sanitize non-finite tensors.

---

# 13. Task 11 — Documentation/state correction

## Delegate only this task to Subagent 11 after source/tests are stable

### Goal

Repository docs must describe what is actually proven, not repeat the earlier overclaim that all optimization work is complete.

### Update at least

- `DECISIONS.md`
- `OPTIMIZATION_DECISIONS.md`
- `OPTIMIZATION_IMPLEMENTATION_STATE.md`
- `OPTIMIZATION_EVIDENCE.md`
- `STATUS.md`
- `TODO.md`
- `DEVICE_TESTS.md` / device runbook if it describes checkpoint/Resume or old W8 BF16 default

Do not rewrite historical reports as if old experiments never happened. Preserve history, then append/currently mark the stabilization decisions.

### Required decisions to record

1. **Checkpoint/resume removed** from production because it no longer justifies its complexity/per-step overhead for the sub-100s target.
2. **W8-v2 production temporarily uses legacy numerics**. FP16-backed BF16 emulation remains experimental and is not claimed range-safe internally.
3. **P8 directQuantized/hybrid quarantined** from production/device presets because physical A12 measurement showed approximately a 10× slowdown versus `dequantizedMPS`; kernel remains research code.
4. **P6 mmap no-copy disabled** after physical A12 GPU page fault; not re-enabled without a future GPU-read hardware proof.
5. Fatal Metal command-buffer faults poison the process generation context until restart.
6. Manual Import is local-only; app launch must never auto-download model packs.
7. Prompt/seed now persist across launch.
8. Physical XS Max validation after this patch is **pending user test**. Do not claim real-device success from simulator/macOS CI.

Reset stale TODOs so completed historical work is not presented as current work and the remaining items are clear.

---

# 14. Hermes review checklist after EVERY subagent

Before starting the next subagent, Hermes must do all of the following:

1. Re-read that task section in this file.
2. Inspect the actual diff, not only the summary:

```bash
git status --short
git diff --stat
git diff --check
git diff -- <relevant files>
```

3. Confirm the subagent did not touch unrelated code.
4. Confirm there were no model/weight downloads or large VPS artifacts.
5. Run the lightweight/available task checks.
6. If the task is verified, commit it with a focused message.
7. Update `HERMES_DEVICE_STABILIZATION_SESSION.md` with one concise line.
8. Only then start the next subagent.

If the subagent wandered into adjacent optimization research, revert that unrelated portion rather than carrying it forward.

---

# 15. Final CI plan

Do not claim completion until these gates are performed.

## CI Gate B — normal project CI

Run the repository's normal CI on the stabilization branch after Tasks 1–10 are complete.

Required successful jobs include the repository's normal equivalents of:

- project consistency;
- iPhone build;
- simulator tests.

Fix failures caused by the patch. Do not suppress/delete legitimate tests merely to turn CI green.

## Full-inference stabilization gate

After normal CI is green, run `full-inference-refine.yml` using its W4/W8 matrix. Let the two matrix jobs run in parallel.

The GitHub runner — **not the VPS** — obtains the large packs/assets.

### Required W4 evidence

```text
variant: w4
policy:  w4Legacy
attention numerics: legacy
activation numerics: legacy
FULL_INFERENCE=PASS
```

### Required W8 evidence

```text
variant: w8-v2
policy:  w8LegacyStabilized
attention numerics: legacy
activation numerics: legacy
FULL_INFERENCE=PASS
```

The existing correctness/image gates must remain intact.

### If W8 legacy fails or produces the checkerboard in corrected CI

Stop and record it as new evidence. Do **not**:

- switch back to BF16 automatically;
- clamp values;
- lower the reference thresholds;
- change prompt/seed/reference;
- call the issue solved.

A failure under true production W8 legacy numerics would invalidate the leading numerical-policy explanation and requires a new planner investigation before more implementation.

### Artifact handling

Inspect logs/summary first. If image artifacts must be visually checked, use GitHub-hosted artifacts through the normal environment; do not pull multi-GB model artifacts onto the VPS. Small generated PNGs/metrics are fine if necessary, but do not use the VPS to cache model packs.

---

# 16. Final source assertions

Before handoff, Hermes should run targeted searches and manually inspect the results.

Examples:

```bash
# No production checkpoint/resume system remains
git grep -n "GenerationCheckpoint\|CheckpointStore\|canResume\|Discard checkpoint\|Checkpoint:" -- AnimaXS AnimaXSTests

# W8 resolver is explicitly stabilized legacy, not BF16 experimental
git grep -n "w8Legacy\|w8BF16\|fromVariantID" -- AnimaXS AnimaXSTests

# Production baseline remains dequantized MPS
git grep -n "linearBackend" -- AnimaXS/Runtime/Generation/InferenceOptimizationConfig.swift AnimaXS/App/InferenceOptimizationSettings.swift

# No-copy cannot survive as a production persisted-on default
git grep -n "noCopyWeightSource" -- AnimaXS AnimaXSTests

# Prompt + seed use persistent storage
git grep -n "generation.lastPrompt\|generation.lastSeed" -- AnimaXS/App/ContentView.swift

# Model launch/import code must not perform an automatic download
git grep -n "download(" -- AnimaXS/App/ContentView.swift AnimaXS/Runtime/ModelStore
```

Interpret grep output; do not blindly require zero matches where research/tests legitimately remain.

Also verify no new secrets, model binaries, `.animapk`, `.bin`, huge archives, or generated inference assets were committed.

---

# 17. Physical-device handoff configuration

Hermes cannot prove the XS Max result itself. The final handoff to the user must clearly request the first physical test with **exactly** this controlled configuration:

```text
Pack:                    W8-v2
Production numerics:     W8 legacy stabilized
Preset:                  Baseline
Linear backend:          dequantizedMPS
Linear tile rows:        128
Attention tile rows:     128
Direct MPS linear I/O:   off
Ping-pong streaming:     on
Numerical monitor:       on
Fused norm/modulation:   off
Fused MLP activation:    off
Strided attention:       off
Cross-KV cache:          off
Mmap no-copy:            unavailable/off
Attention backend:       legacy head-major MPS
Checkpointing:           absent
```

The test should be unplugged, because charging materially distorts XS Max thermals/performance.

First objective is **repeatable coherent images with no NaN/page fault/checkerboard**, not speed.

Only after the stable baseline is demonstrated on the device should optimization benchmarking restart. Begin with configurations that retain `dequantizedMPS`; do not include P6 no-copy or P8 directQuantized in that search.

---

# 18. Final report Hermes must produce

When finished, write a concise final report containing:

1. branch and final HEAD;
2. commits added, one line each;
3. exact checkpoint files/API/UI removed;
4. exact W8 production numerical policy now used;
5. confirmation `directQuantized`/hybrid cannot be selected for normal device generation;
6. confirmation P6 no-copy cannot be selected;
7. fatal-Metal poisoning behavior;
8. import-vs-download regression fix and tests;
9. prompt/seed persistence keys;
10. stale-image/current-run fix;
11. normal CI run ID + job outcomes;
12. W4/W8 full-inference run ID + policy markers + PASS/FAIL for each;
13. any remaining unresolved issue;
14. a prominent statement that **physical-device validation remains pending until the user runs the new build**.

Do not write “everything is solved” merely because CI is green.

---

# 19. Reusable one-task subagent prompt template

Hermes should use a prompt shaped like this for every delegation:

```text
You are an implementation subagent. You have exactly ONE task in this session:

[TASK NUMBER + TITLE]

Do not investigate unrelated issues. Do not redesign adjacent systems. Do not spawn another agent.
Do not download model weights, .animapk/.bin files, Hugging Face model repos, or large GitHub artifacts to this VPS.
Do not push, merge, or force-reset repository history.

Relevant files:
[ONLY FILES FOR THIS TASK]

Required behavior:
[PASTE ONLY THE REQUIRED BULLETS FROM THIS TASK]

Acceptance checks:
[PASTE ONLY THIS TASK'S CHECKS]

Implement the task completely, run only lightweight/local checks available here, and stop. Report:
- files changed;
- exact behavior implemented;
- checks run/results;
- any direct compile/test consequence that Hermes must handle.

Do not start another task.
```

Hermes then reviews the diff itself, commits only verified work, updates the tiny session ledger, re-reads the next section, and launches the next single-task subagent.

---

# 20. Definition of done

This stabilization run is complete only when all of the following are true:

- [ ] Checkpoint/resume production system is removed.
- [ ] No per-step checkpoint latent CPU copy/write remains in production.
- [ ] W8-v2 production resolves to an explicit stabilized legacy numerical policy.
- [ ] Full-inference tests derive the same production numerical policy from the pack variant.
- [ ] W4 and W8 CI policy markers are asserted.
- [ ] `dequantizedMPS` is the only normal device linear backend for this stabilization build.
- [ ] `directQuantized`/hybrid cannot contaminate device presets.
- [ ] mmap no-copy is unavailable for normal generation.
- [ ] Fatal GPU faults poison generation until app restart.
- [ ] Import cannot trigger Download and same-component model operations are single-flight.
- [ ] Launch/model discovery remains local-only and does not auto-download packs.
- [ ] Prompt persists across relaunch.
- [ ] Seed persists across relaunch, including Randomize.
- [ ] Fresh Generate clears a previous run's image/metrics ownership correctly.
- [ ] Diagnostics preset marker cannot lie after manual changes.
- [ ] Numerical error labels are no longer ambiguous at gate/add boundaries.
- [ ] Normal CI is green.
- [ ] Corrected W4 full inference is green.
- [ ] Corrected W8 **legacy-production-policy** full inference is green, or a failure is explicitly reported without manipulating gates.
- [ ] Docs/status accurately state what is proven and what is still pending.
- [ ] No weights/large model artifacts were downloaded to or retained on the VPS.
- [ ] Final report contains exact commits + CI run IDs.
- [ ] No claim of physical-device success is made before the user's new XS Max test.

