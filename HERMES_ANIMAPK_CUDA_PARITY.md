# Hermes Execution Instructions — AnimaXS Precision Ladder + Direct `.animapk` CUDA/Metal Parity Investigation

## Mission

Continue the AnimaXS root-cause investigation, but change the experiment design from a single FP16 control into a **controlled precision/runtime ladder**.

Do **not** spend most of the run on repeated 8-step/full-image tests. The current failure is already visible at step 0, so the investigation must first determine where the very first model forward diverges.

The primary control matrix is:

```text
A. official BF16 safetensors -> pinned upstream PyTorch/CUDA
B. same official model converted to FP16 -> same upstream PyTorch/CUDA
C. FP16-all .animapk -> direct .animapk CUDA runtime
D. W8 .animapk -> same direct .animapk CUDA runtime
E. W4 .animapk -> same direct .animapk CUDA runtime
```

Then run the same `.animapk` packs through Apple/Metal:

```text
C-Metal. FP16-all .animapk -> AnimaXS Metal
D-Metal. W8 .animapk -> AnimaXS Metal
E-Metal. W4 .animapk -> AnimaXS Metal
```

Interpretation:

```text
A -> B = BF16 -> FP16 model/storage precision effect
B -> C = .animapk/container/direct-runtime effect
C -> D = W8 quantization effect
C -> E = W4 quantization effect

C CUDA -> C Metal = Apple backend/runtime effect for identical FP16 pack
D CUDA -> D Metal = Apple backend/runtime effect for identical W8 pack
E CUDA -> E Metal = Apple backend/runtime effect for identical W4 pack
```

The most important new engineering task is a **direct CUDA `.animapk` runtime that consumes the real pack bytes**, rather than only dequantizing pack tensors into normal Python arrays/state dicts before inference.

> **Execution mandate:** finish every task you reasonably can in one continuous run. Do not stop after one failed hypothesis or inconclusive experiment. Fix ordinary infrastructure/test issues and continue. Stop only when the root cause is fixed and validated, or when a genuine external blocker makes further progress impossible. If blocked, leave exact evidence and next executable commands.

---

# 1. Critical Clore.ai instruction

## 1.1 The Clore instance is already launched by the user

The user will have an active Clore.ai GPU instance ready before you start.

Therefore:

```text
DO NOT launch another Clore instance.
DO NOT rent a second GPU.
DO NOT terminate the provided instance during normal experimentation.
```

Use the already-running instance as the main high-speed investigation machine.

The purpose is to avoid repeatedly:

- provisioning a GPU
- reinstalling CUDA/PyTorch
- redownloading the official multi-GB weights
- rebuilding dependencies
- losing Hugging Face/model caches
- losing generated fixtures between experiments

## 1.2 Keep the provided instance alive until the investigation is finished

Once connected:

- inspect the environment before reinstalling anything
- reuse valid model/cache files
- reuse the repo checkout where safe
- preserve generated source fixtures
- preserve the CUDA `.animapk` implementation while iterating

Do not shut the instance down after an individual test.

## 1.3 Hermes is responsible for closing Clore when done

When **all useful Clore work is complete**:

1. commit/push all source changes
2. persist important fixtures/results off the Clore VM
3. update `DECISIONS.md`, TODO, and `HERMES_SESSION.md`
4. send the final Telegram Clore/results update
5. **close/terminate the existing Clore instance**
6. state in the final report that the instance was shut down

Do not leave paid GPU capacity running after the work is complete.

Before termination, verify that no sole copy of useful code/evidence exists only on the Clore disk.

---

# 2. Compute policy

## VPS

The cheap VPS is orchestration-only.

Allowed:

- git
- editing
- logs/JSON
- workflow dispatch
- lightweight scripts
- Telegram updates

Do not run large model operations there.

## Clore NVIDIA GPU

Use for:

- actual pinned upstream Anima/Cosmos source execution
- source BF16 and FP16 controls
- direct `.animapk` CUDA execution
- W4/W8/FP16 precision ladder
- BF16->FP16 analysis
- tensor hooks and comparison
- rapid repeated step-0 testing
- fixture generation

## GitHub Linux

Use for cheap packaging/static/unit verification when useful.

## GitHub macOS

Use only when actual Apple/Xcode/Metal behavior needs proving.

Do not recompute the source oracle on macOS if Clore can produce and persist the source fixture once.

---

# 3. Branch and repo hygiene

At the beginning:

```bash
git fetch --all --prune
git status --short
git rev-parse HEAD
git rev-parse origin/main
git log -n 15 --oneline --decorate
```

Create/reset a dedicated investigation branch, for example:

```bash
git switch -C investigate/animapk-cuda-parity origin/main
```

Do not dirty `main`.

Do not assume the attached evidence commit is still current. Verify the repository first.

---

# 4. Re-read/context-compaction rules — mandatory

Save this file at repo root as:

```text
HERMES_ANIMAPK_CUDA_PARITY.md
```

Re-read the complete instruction file:

- at session start
- after context compaction/reset
- before an expensive macOS run
- after every major Clore result
- after every major CI result
- whenever the main hypothesis changes
- before final validation
- before shutting down Clore

Maintain a compact:

```text
HERMES_SESSION.md
```

with:

```markdown
# Current state

Branch:
HEAD:
origin/main:
Last known green CI:
Clore status:
Clore GPU:
Current hypothesis:
Control matrix status:
Direct animapk CUDA status:
Last experiment:
Last result:
Proven:
Rejected:
Blockers:
Next 3 actions:
Important GitHub run IDs:
Important pack SHAs:
Important source SHAs:
Important Clore paths:
```

Update it after each meaningful milestone and **before any context compaction**.

After compaction, first read:

1. `HERMES_ANIMAPK_CUDA_PARITY.md`
2. `HERMES_SESSION.md`
3. newest `DECISIONS.md` entries
4. current TODO

Then continue.

---

# 5. Reset the active TODO before new work

Read:

```text
DECISIONS.md
STATUS.md
NEXT_TASK_HANDOFF.md
TODO.md
.github/workflows/*
relevant scripts/*
Runtime/Animapk/*
Runtime/Metal/*
```

Cross-check documentation against actual code and current CI.

Remove/supersede stale active investigation tasks.

Do not delete historical decisions from `DECISIONS.md`; append new corrections/supersessions.

Use a clean active checklist in `TODO.md` or `HERMES_TODO.md`.

---

# 6. Telegram updates and images — mandatory

The user wants frequent progress visibility.

Send concise Telegram updates at least:

- after repo audit
- after connecting to the already-running Clore instance
- after BF16->FP16 storage analysis
- after actual-upstream vs custom-oracle validation
- after direct `.animapk` decoder validation
- after first FP16-all `.animapk` CUDA step-0 run
- after W8/W4 CUDA controls
- after precision-ladder plots are ready
- after first Metal comparison
- when the earliest real divergence is localized
- after a fix candidate
- after final validation
- immediately before/after shutting down Clore

Each update should contain:

```text
branch + HEAD
Clore/GitHub run or experiment
variant
pack SHA
key metric(s)
current conclusion
next action
```

Send images whenever useful.

For step-0 work, useful Telegram images include:

- per-block cosine plot
- per-block relative-L2 plot
- BF16/FP16/W8/W4 precision ladder
- CUDA vs Metal backend plot
- spatial velocity error heatmap
- per-patch-class error chart
- FFT/grid-carrier chart if relevant

For a full-image run, send:

- `generated.png`
- `reference.png`
- `comparison.png`
- carrier/FFT visualization

Never expose secrets/tokens in Telegram, CI logs, or artifacts.

---

# 7. Known evidence to preserve

Latest attached evidence was generated around:

```text
c17986b6470a0059575d63c8943ca1550a563008
```

Verify current HEAD before relying on it.

## Official source pin

Unless the repo deliberately superseded it:

```text
HF repo:
circlestone-labs/Anima

revision:
f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b

official Turbo safetensors SHA-256:
c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
```

A source hash mismatch invalidates the experiment.

## Latest FP16-all pack

```text
anima-turbo-v1.0-xsmax-fp16-all.animapk
bytes: 4,193,255,424
SHA-256: c17d62727beb114590febbe9dd019e5c9d523863d6e2b32b17c5872c6b0635ca
```

Both latest `legacy-production` and `bf16-production` evidence used this same stored FP16-all pack.

Therefore:

```text
bf16-production != native BF16 stored weights
```

It was FP16-all storage plus a different compute/activation path.

## Latest step-0 source-oracle result

Approximately:

```text
BF16-compute production step-0 cosine ~0.81683
legacy production step-0 cosine       ~0.81674
```

Those trajectories were almost identical.

Therefore broad BF16-emulated arithmetic has not explained the primary mismatch.

## Quantization evidence

Prior results indicate:

```text
W4 < W8 < FP16-all
```

and FP16-all improved latent similarity substantially versus W8, but the severe woven/grid artifact remained.

Thus:

```text
quantization is a real quality loss
but is not yet proven to be the primary grid cause
```

## 8-pixel carrier

Known bad output had roughly:

```text
generated carrier total ~0.01345
reference carrier total ~0.000055
ratio ~245x
```

Do not promote a fix that merely changes cosine slightly while leaving this pathology intact.

---

# 8. Do not blindly redo already-cleared branches

Read `DECISIONS.md` carefully.

Prior work already contains strong evidence around:

- input patchify order
- zero padding-mask channel
- x-embedder shape/order
- timestep embedding
- several same-pack block parity checks
- attention precision
- Qwen conditioning
- sampler/scheduler

A cheap output-layout sanity test remains worthwhile, but do not assume patchify is the leading root cause without new evidence.

The new question is more controlled:

> When the **exact same real `.animapk` bytes** are executed through CUDA and through Metal, where do their model forwards separate?

---

# 9. Core precision/runtime controls

Use the same canonical:

```text
prompt
conditioning
initial latent/noise
seed
sigma
step
RoPE convention
model config
hook locations
metrics
```

for every variant.

## A — BF16 source ground truth

```text
official BF16 safetensors
actual pinned upstream graph
PyTorch/CUDA
```

## B — FP16 source control

Same upstream graph and canonical inputs, with official weights deterministically converted/loaded as FP16.

This measures pure:

```text
A -> B
```

without `.animapk`, Swift, or Metal.

## C — FP16-all `.animapk` CUDA

Use the exact `.animapk` artifact itself as the sole weight source.

This measures:

```text
B -> C
```

## D — W8 `.animapk` CUDA

Same runtime, W8 pack.

Measures:

```text
C -> D
```

## E — W4 `.animapk` CUDA

Same runtime, W4 pack.

Measures:

```text
C -> E
```

Then run C/D/E on Metal.

---

# 10. New required engineering task — direct `.animapk` CUDA runtime

Do **not** merely decode every pack weight into a normal Python state dict and call that equivalent to production.

Build two explicit CUDA/Python modes so one implementation can validate the other.

## Mode 1 — `decoded_reference`

Purpose: simplest byte-faithful correctness oracle.

Flow:

```text
actual .animapk
-> mmap/read real bytes
-> parse actual tensor metadata/offsets
-> decode requested FP16/W8/W4 tensor
-> torch tensor
-> ordinary PyTorch/CUDA operation
```

This mode may materialize decoded tensors more freely.

It exists to prove pack interpretation and mathematical decode correctness.

## Mode 2 — `streaming_animapk`

Purpose: mirror the production AnimaXS weight lifecycle.

Flow:

```text
actual .animapk mmap
-> locate current execution/block range
-> copy raw range into reusable device/raw-byte ring
-> resolve tensor offsets inside ring
-> decode W4/W8 into reusable FP16 scratch
-> matmul/model operation
-> overwrite/reuse ring for next block
```

Do not conflate the two modes in metrics or reports.

Both must use the actual `.animapk` artifact as the sole C/D/E weight source.

---

# 11. Parse and validate the real `.animapk`

Create a clean implementation, e.g.:

```text
scripts/animapk_cuda/
  format.py
  reader.py
  quant.py
  runtime.py
  compare.py
```

Mirror the authoritative Swift reader/format semantics.

Validate:

- header/version
- JSON tensor metadata
- binary table if used
- tensor names
- shapes
- dtype/quantization mode
- blob offsets
- blob lengths
- file bounds
- alignment
- execution ranges
- pack provenance

Never substitute original safetensors when running C/D/E.

The pack is the source of truth for those variants.

---

# 12. Exact W4/W8 decoder parity

Before full inference, prove the direct pack decoder.

## FP16

Decode the exact half bytes.

## W8

Mirror current pack semantics exactly, including:

- unsigned byte `q`
- group size
- groups along K
- **groups reset per matrix row**
- FP16 scales
- FP16 zero terms
- shape/orientation

Use the repo's exact formula, approximately:

```text
weight = q * scale + zero
```

## W4

Mirror exact nibble order:

```text
even K -> low nibble
odd K  -> high nibble
```

and:

- unsigned q = 0..15
- groups along K
- group reset for every matrix row
- FP16 scale/zero
- padded/edge K behavior

Do not flatten grouping globally across rows.

## Decoder test set

Include representative tensors from:

- x_embedder
- attention Q/K/V/O
- MLP
- final projection
- odd/edge shapes
- padded K cases

Compare against an existing trusted Swift/CPU pack decoder/oracle.

Output:

```text
animapk_decoder_parity.json
animapk_decoder_parity.md
```

Do not draw model conclusions until decoder parity is established.

---

# 13. Lifecycle-faithful streaming mode

Once Mode 1 is correct, implement Mode 2.

The intended analogue of AnimaXS is:

```text
Linux mmap
-> current block/range in host memory
-> CUDA copy to reusable raw-byte device ring
-> tensor local offsets
-> W4/W8 GPU decode to reusable FP16 scratch
-> matmul
-> next range overwrites previous range
```

Do not overengineer GPUDirect Storage.

A normal host->device copy is sufficient for the semantic control.

Do not permanently expand all W4/W8 weights to FP16 in streaming mode.

A decoded reference mode may do that for debugging, but streaming mode should remain bounded/reusable.

Generate:

```text
animapk_execution_manifest.json
```

including for each block/range:

```text
range start
range length
tensor names
global offsets
local offsets
dtype
shape
blob bytes
```

If possible, generate the same manifest from Swift and compare automatically.

Fix manifest/offset disagreements before investigating math.

---

# 14. Validate custom source oracle against actual pinned upstream

This is still mandatory.

On Clore:

1. use exact source revision
2. exact official checkpoint
3. exact canonical conditioning
4. exact initial latent
5. sigma 1.0 for step 0
6. run one actual upstream DiT forward
7. capture output/hooks
8. compare to the current custom source oracle

Metrics:

```text
cosine
RMSE
relative L2
maxAbs
norms
finite
hash
```

If actual upstream and the custom oracle materially disagree, repair/localize the oracle **before** using it as the target for Swift.

If they agree extremely tightly, record that in `DECISIONS.md`.

---

# 15. BF16 -> FP16 storage analysis

Create/run:

```text
scripts/analyze_bf16_fp16_storage.py
```

For every official tensor compare:

```text
source BF16 -> FP32 reference
vs
BF16 -> FP32 -> FP16 -> FP32
```

Record:

- element count
- exact equality count/%
- cosine
- RMSE
- relative L2
- max abs error
- max relative error
- source zeros
- newly introduced zeros
- FP16 subnormals
- Inf
- NaN
- magnitude range

Output:

```text
bf16_fp16_storage_summary.json
bf16_fp16_storage_tensors.csv
bf16_fp16_top20.md
```

Pay special attention to:

- x_embedder
- timestep/modulation
- q/k norms
- attention projections
- MLP
- final normalization
- final projection
- small 1-D params/biases

---

# 16. Canonical step-0 hooks for all variants

Capture the same boundaries for A/B/C/D/E and Metal.

## Inputs/pre-block

```text
initial latent
patchified/pre-x-embedder
x_embedder output
timestep embedding
AdaLN/modulation vector
conditioning/context
RoPE anchors or deterministic hash
```

## All transformer blocks

Prefer residual input/output for every block in the **same single forward**:

```text
block00_in
block00_out
...
block27_in
block27_out
```

Do not create 28 separate expensive jobs.

## Final path

Capture:

```text
pre_final_norm
post_final_norm
pre_final_projection
post_final_projection_patched
post_unpatchify_velocity
```

Use graph-equivalent names if exact boundaries differ.

## Provenance

Each hook bundle must record:

```text
variant
backend/runtime
source revision
pack filename if applicable
pack SHA
official weight SHA
git commit
script SHA
torch/CUDA versions
GPU
tensor dtype/shape/hash
prompt
seed
sigma
canonical input hashes
```

---

# 17. Run the full precision ladder on the existing Clore instance

Execute, without reprovisioning/reinstalling between variants:

```text
A = upstream BF16
B = upstream FP16
C = FP16-all .animapk CUDA
D = W8 .animapk CUDA
E = W4 .animapk CUDA
```

Reuse:

- model cache
- source checkout
- Python environment
- canonical Qwen/conditioning fixture
- latent fixture
- comparison tooling

Do not decode VAE.

Do not run 8 denoising steps.

Step 0 is the main diagnostic battlefield.

Generate a stage table containing:

```text
stage
A_vs_B cosine/RMSE/relL2
B_vs_C cosine/RMSE/relL2
C_vs_D cosine/RMSE/relL2
C_vs_E cosine/RMSE/relL2
absolute norms
finite checks
```

Output:

```text
precision_ladder_stage_parity.csv
precision_ladder_stage_parity.json
precision_ladder_stage_parity.md
precision_ladder_cosine.png
precision_ladder_relative_l2.png
```

Send the plots to Telegram.

---

# 18. Interpret the ladder before touching Metal code

## If A -> B is already large

Anima Turbo is genuinely sensitive to BF16->FP16 in this graph.

Determine whether drift:

- is immediate
- grows gradually through blocks
- jumps at specific tensor classes

Then judge C against B, not only against A.

## If A -> B is small but B -> C is large

Focus on:

- `.animapk` interpretation
- orientation/layout
- pack metadata
- runtime graph
- direct pack execution semantics

## If B ~= C in decoded-reference mode but streaming C diverges

The problem is in the streaming/range/runtime implementation.

Fix the CUDA emulator before comparing it to Metal.

## If decoded-reference C ~= streaming C

The direct `.animapk` CUDA runtime becomes a strong backend control.

## C -> D and C -> E

These are clean measurements of actual W8/W4 quantization damage through the same runtime.

---

# 19. Cheap output-layout sanity gate

Retain one synthetic final projection/unpatchify test.

Construct collision-free logical patch values such as:

```text
value = 10000*channel + 100*patch_y + patch_x
```

Compare:

- pinned upstream rearrangement
- direct Python/CUDA implementation
- Swift/Metal implementation

Require exact positional agreement.

This is a cheap gate, not an assumption that layout is definitely the root cause.

---

# 20. GitHub macOS strategy

Once C/D/E CUDA are trustworthy, use GitHub macOS only for the Apple side.

Create/adapt a manual workflow such as:

```text
.github/workflows/quality-step0-backend-matrix.yml
```

No push trigger.

Preferred shape:

1. checkout once
2. select Xcode once
3. restore safe caches
4. obtain exact C/D/E packs
5. build once
6. load canonical input/conditioning fixture
7. execute one step-0 forward per pack
8. capture matching hooks
9. compare/upload evidence

If safe, run C/D/E sequentially in one diagnostic process/build.

If memory/disk prevents that, use a workflow matrix, but each variant must remain one-step-only.

Do not run VAE.

Do not run eight steps just to diagnose a first-forward mismatch.

---

# 21. CUDA vs Metal comparisons

Compare:

```text
C CUDA vs C-Metal
D CUDA vs D-Metal
E CUDA vs E-Metal
```

Generate:

```text
backend_parity_fp16.csv/json/md
backend_parity_w8.csv/json/md
backend_parity_w4.csv/json/md
backend_parity_overview.png
```

Plot block-by-block cosine and relative-L2.

Send `backend_parity_overview.png` to Telegram.

The earliest stage where CUDA and Metal materially separate becomes the target.

---

# 22. Decision tree

## C/D/E CUDA ~= C/D/E Metal

The pack/runtime arithmetic is probably not suffering a major Apple-specific semantic error.

Then focus more strongly on:

- source BF16 sensitivity
- source graph differences
- oracle/reference behavior
- cumulative numerical effects

## FP16 CUDA ~= Metal, but W8/W4 CUDA != Metal

Focus on Apple W4/W8 decode:

- nibble order
- byte interpretation
- group reset
- scale/zero
- padded K
- stride/orientation
- scratch dtype
- decode kernel

## CUDA and Metal agree through blocks but diverge at final path

Inspect only:

- final modulation
- final norm
- final projection
- dtype boundary
- projection orientation
- output rearrangement/unpatchify

## CUDA diverges from Metal at one specific block

Instrument only that block and immediate inputs.

## A/B/C/D/E all show a similar regular carrier

Then the carrier is not specifically a Metal-only phenomenon.

Measure whether the carrier first appears from:

- BF16->FP16
- pack execution
- quantization
- recurrent denoising

## CUDA is clean but Metal develops the carrier

This strongly isolates an Apple runtime/backend issue.

---

# 23. Native BF16 `.animapk` is evidence-driven, not the first task

Do not immediately redesign the pack format for BF16.

Only do so if A->B shows a large meaningful source precision loss, or another result specifically justifies native BF16 storage.

If implemented, require:

- explicit BF16 pack metadata
- deterministic BF16 bit storage
- parser/verifier support
- runtime support
- no hidden FP16 conversion
- exact bit-pattern tests

Run step 0 first.

Do not go straight to full image generation.

---

# 24. Full-image tests are promotion tests only

A candidate earns an 8-step latent run only if it:

- fixes the earliest known divergence
- materially improves step-0 parity
- or materially reduces the pathological spatial carrier

Sequence:

```text
unit/decoder test
-> step-0 parity
-> 8-step latent-only
-> VAE/full image
```

For final image collect:

```text
latent cosine
RGB cosine
RGB RMSE
8px carrier total
8px carrier ratio vs reference
```

Do not promote tiny cosine gains with unchanged grid.

---

# 25. Quantization optimization comes later

Once correctness is understood:

```text
BF16/FP16 truth
-> W8
-> mixed W8/FP16 or W8/BF16
-> W4 only after W8 is acceptable
```

Possible later techniques:

- per-row INT8
- learned rounding
- bias correction
- sensitive-layer high precision
- mixed precision by tensor class
- high-precision final layer/norm/modulation

Public Anima quantization projects may inspire optimization later, but they are not ground truth for this diagnostic phase.

---

# 26. Clore first actions

On the already-running instance record:

```bash
hostname
nvidia-smi
df -h
free -h
pwd
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name())
print('bf16 supported:', torch.cuda.is_bf16_supported())
PY
```

Save to:

```text
clore_environment.txt
```

Inspect existing environment/caches before installing/downloading.

Verify hashes before reusing existing model files.

Never print credentials/tokens.

---

# 27. Persist Clore results off-instance

Anything important must not exist solely on the Clore disk.

Persist:

- source code -> git commit/push
- canonical source fixtures -> intended artifact/HF/project storage
- CSV/JSON/plots -> GitHub/project artifact storage
- decisions -> `DECISIONS.md`
- current state -> `HERMES_SESSION.md`

Do this before termination.

---

# 28. Required artifacts

## Source/precision

```text
clore_environment.txt
provenance.json
bf16_fp16_storage_summary.json
bf16_fp16_storage_tensors.csv
bf16_fp16_top20.md
source_oracle_parity.json
source_oracle_parity.md
precision_ladder_stage_parity.csv
precision_ladder_stage_parity.json
precision_ladder_stage_parity.md
precision_ladder_cosine.png
precision_ladder_relative_l2.png
```

## `.animapk` runtime

```text
animapk_decoder_parity.json
animapk_decoder_parity.md
animapk_execution_manifest.json
animapk_runtime_provenance.json
```

## Backend parity

```text
backend_parity_fp16.csv/json/md
backend_parity_w8.csv/json/md
backend_parity_w4.csv/json/md
backend_parity_overview.png
```

## Full image only when earned

```text
generated.png
reference.png
comparison.png
grid-carrier.json
grid-carrier.png
```

Upload evidence even when assertions fail where practical.

---

# 29. Promotion discipline

Every significant experiment must conclude with one of:

```text
PROMOTE
REJECT
INCONCLUSIVE
```

Examples:

```text
PROMOTE — direct W8 animapk CUDA and Metal agree through block 27; divergence begins at final projection.
```

```text
REJECT — compute-policy change moves step-0 cosine <0.001 and leaves 8px carrier unchanged.
```

```text
INCONCLUSIVE — CUDA W4 decoder disagrees with trusted row-reset reference; fix decoder before judging inference.
```

Do not keep zombie hypotheses alive.

Append non-obvious conclusions to `DECISIONS.md` with:

- commit
- run/experiment ID
- source revision
- pack SHA
- metrics
- what the result proves
- what it does not prove

---

# 30. Final validation after a real fix

Once a real root cause is fixed:

1. decoder/unit tests
2. direct `.animapk` reference tests
3. Clore A/B/C/D/E control matrix
4. one-step CUDA/Metal parity
5. 8-step latent-only
6. full image + carrier analysis
7. normal CI
8. iPhone build
9. simulator tests
10. update docs/decisions/TODO/session
11. push branch
12. persist all Clore fixtures/results
13. send final Telegram evidence/images
14. **close the provided Clore instance**

---

# 31. Clore shutdown checklist — mandatory

Before shutdown:

- [ ] all useful code committed
- [ ] branch pushed
- [ ] source/pack hashes recorded
- [ ] CUDA `.animapk` implementation persisted
- [ ] source fixtures persisted
- [ ] JSON/CSV/plots persisted
- [ ] `DECISIONS.md` updated
- [ ] TODO/status updated
- [ ] `HERMES_SESSION.md` updated
- [ ] final Clore Telegram summary sent
- [ ] no remaining planned experiment needs the live GPU
- [ ] **terminate/close the existing Clore instance**
- [ ] record that shutdown in the final report

The user explicitly expects Hermes to close the Clore instance when done.

---

# 32. Final report format

```markdown
# AnimaXS precision/runtime root-cause investigation — final report

## Repository state
branch:
HEAD:
origin/main:
working tree:

## Clore
provided instance used: yes/no
GPU:
environment:
instance closed at end: yes/no

## Control matrix
A BF16 source:
B FP16 source:
C FP16-all animapk CUDA:
D W8 animapk CUDA:
E W4 animapk CUDA:

## BF16 -> FP16 conclusion
...

## Direct animapk runtime conclusion
...

## Quantization conclusion
W8:
W4:

## CUDA vs Metal conclusion
FP16-all:
W8:
W4:

## Earliest divergence
...

## Root cause
...

## Fix
...

## Step-0 before/after
...

## 8-step latent before/after
...

## Image/carrier before/after
...

## CI
normal CI:
iPhone build:
simulator tests:
quality workflow:

## Files changed
...

## Remaining device-only risks
...

## Remaining TODO
...
```

Include exact commit hashes, workflow run IDs, pack SHAs, source revision, and persisted artifact locations.

---

# 33. Immediate action checklist

- [ ] Save/re-read this instruction at repo root.
- [ ] Fetch repo and record current HEAD/origin/main.
- [ ] Create/reset `investigate/animapk-cuda-parity`.
- [ ] Re-read `DECISIONS.md`, `STATUS.md`, `NEXT_TASK_HANDOFF.md`, `TODO.md`.
- [ ] Hard-reset active investigation TODO to current unresolved work.
- [ ] Update `HERMES_SESSION.md`.
- [ ] Send initial Telegram status.
- [ ] Connect to the **already-running** Clore instance.
- [ ] Do **not** launch another Clore instance.
- [ ] Record Clore GPU/environment.
- [ ] Reuse valid caches/files where possible.
- [ ] Verify official source hash.
- [ ] Validate custom oracle against actual pinned upstream BF16.
- [ ] Run BF16->FP16 storage audit.
- [ ] Implement `decoded_reference` direct `.animapk` reader/runtime.
- [ ] Prove FP16/W8/W4 decoder parity.
- [ ] Implement `streaming_animapk` lifecycle-faithful CUDA mode.
- [ ] Compare decoded-reference vs streaming mode.
- [ ] Generate/compare execution manifests.
- [ ] Run A = upstream BF16.
- [ ] Run B = upstream FP16.
- [ ] Run C = FP16-all `.animapk` CUDA.
- [ ] Run D = W8 `.animapk` CUDA.
- [ ] Run E = W4 `.animapk` CUDA.
- [ ] Generate precision-ladder plots and send to Telegram.
- [ ] Determine A->B, B->C, C->D, C->E effects.
- [ ] Run cheap output-layout sanity test.
- [ ] Run C/D/E step-0 on Metal.
- [ ] Compare C/D/E CUDA vs Metal.
- [ ] Localize earliest backend-specific divergence.
- [ ] Fix only the implicated subsystem.
- [ ] Re-run one-step gates.
- [ ] Only if strongly improved, run 8-step latent-only.
- [ ] Only if earned, run full image/carrier validation.
- [ ] Run normal CI/iPhone build/simulator tests.
- [ ] Update decisions/status/TODO/session.
- [ ] Persist all useful Clore code/fixtures/evidence.
- [ ] Send final Telegram results/images.
- [ ] **Close/terminate the Clore instance.**
- [ ] Record shutdown in final report.

---

# 34. Central rule

The purpose of this phase is to construct a controlled evidence chain:

```text
official BF16 source
        ↓
official FP16 source
        ↓
real FP16 .animapk on CUDA
        ↓
real W8/W4 .animapk on CUDA
        ↓
the exact same .animapk packs on Metal
```

This should finally separate:

```text