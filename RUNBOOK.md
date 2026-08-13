# AnimaXS Refine-and-Improve Runbook

## Mission

AnimaXS has passed the difficult integration stage.

The complete:

```text
prompt
→ tokenizer
→ Qwen
→ LLM adapter
→ DiT
→ 8-step sampler
→ VAE
→ RGB/PNG
```

pipeline now executes end-to-end.

The project is **not in production-readiness mode**.

It is now officially in:

> **REFINE-AND-IMPROVE MODE**

The immediate objective is to rebuild the DiT packing system properly, generate improved **W4** and **W8** model packs from the original Anima Turbo source, make the runtime support both formats cleanly, and compare both using the exact same canonical full inference.

Do not waste this phase repeatedly re-proving components that already have strong parity evidence.

The current legacy W4 remains valuable as a historical baseline, but its visible patterned/dull output is **not an acceptable image-quality target**.

Historical measurements:

```text
Legacy W4 final latent cosine ≈ 0.6946
Legacy W4 RGB cosine          ≈ 0.7035
```

The earlier isolated eight-step W4 test independently produced approximately `0.6919` final-latent cosine, strongly confirming that the full pipeline reproduces the same accumulated W4 divergence rather than introducing a new VAE/PNG bug.

---

# 1. New infrastructure architecture

The old VPS-centric model-building workflow is retired.

Do **not** use Clore.ai in this refinement cycle.

Do **not** download the original 4+ GB Anima model onto the VPS.

Do **not** build new `.animapk` model files on the VPS.

Do **not** use GitHub Releases as the primary working storage for the new W4/W8 refinement packs.

The new architecture is:

```text
                 ORIGINAL MODEL
                       │
                       ▼
          Hugging Face: circlestone-labs/Anima
                       │
                       │ direct download
                       ▼
              GitHub Actions Linux
              ┌─────────────────┐
              │ packing job W4  │
              └────────┬────────┘
                       │
                       ├── verify
                       ├── quant report
                       └── upload
                            │
                            ▼
                  Hugging Face W4 repo


          Hugging Face: circlestone-labs/Anima
                       │
                       │ direct download
                       ▼
              GitHub Actions Linux
              ┌─────────────────┐
              │ packing job W8  │
              └────────┬────────┘
                       │
                       ├── verify
                       ├── quant report
                       └── upload
                            │
                            ▼
                  Hugging Face W8 repo


              Full-inference workflow
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
        download W4           download W8
        from HF               from HF
            │                     │
            ▼                     ▼
      macOS simulator        macOS simulator
      full inference         full inference
            │                     │
            ▼                     ▼
      PNG + metrics          PNG + metrics
```

GitHub Actions becomes **ephemeral compute**.

Hugging Face becomes **durable model storage**.

The VPS becomes only:

* source-control workspace;
* execution-agent workspace;
* GitHub workflow authoring environment;
* secret-bootstrap environment;
* documentation environment.

This completely removes the VPS model-storage problem.

---

# 2. Why NOT to use GitHub Actions artifacts for the model packs

Multiple artifacts per workflow are supported.

The current `actions/upload-artifact` implementation allows up to **500 artifacts from an individual job**. GitHub also supports downloading individual named artifacts or all artifacts from a workflow run.

However, **do not use Actions artifact storage for the large `.animapk` files**.

The reason is the account-level artifact storage quota.

GitHub currently documents included Actions artifact storage approximately as:

```text
GitHub Free       500 MB
GitHub Pro          1 GB
GitHub Team         2 GB
Enterprise Cloud   50 GB
```

The existing legacy W4 alone is:

```text
1,179,435,008 bytes
```

and the existing Qwen pack is another:

```text
635,305,984 bytes
```

Therefore the original assumption:

> “GitHub Actions artifacts can hold ~5 GB, therefore use them for W4/W8”

should **not** be part of the design.

Even if an individual upload could technically be accepted, the account-level storage quota can make this unreliable.

## Correct usage of Actions artifacts

Use GitHub artifacts only for small evidence:

```text
quant-report.json
packing-manifest.json
verification-report.json
SHA256SUMS
generated.png
reference.png
comparison.png
metrics.json
logs if useful
```

Set short retention, e.g.:

```yaml
retention-days: 7
```

The actual multi-GB model pack must be uploaded directly from the packing job to Hugging Face **before the ephemeral runner disappears**.

---

# 3. Why Hugging Face is the correct durable store

Hugging Face's CLI handles large-file uploads automatically.

Its current repository recommendations say individual files should preferably remain below about **200 GB**, with a hard maximum of 500 GB per individual file. A 1–3 GB `.animapk` is therefore completely ordinary for the platform.

Free public repositories are currently described as best-effort storage; the few GB involved in AnimaXS are far below the scale of the Hub's documented large-model repository recommendations.

Hugging Face also specifically recommends separate repositories for different model-weight variants.

Therefore create separate durable repositories for:

```text
AnimaXS-DiT-W4
AnimaXS-DiT-W8
```

under the Hugging Face account associated with the token supplied by the user.

This also avoids two parallel GitHub matrix jobs attempting concurrent commits into the same HF repository.

---

# 4. Qwen and VAE are NOT being rebuilt

Continue using the existing validated:

```text
qwen3-0.6b-xsmax-w8.animapk
qwen-image-vae-xsmax-fp16.animapk
```

The current release records:

```text
qwen3-0.6b-xsmax-w8.animapk
635,305,984 bytes
SHA256:
ba59e4d1797de5f6512aeafcecf3f38e1f62a47313a2a400b949c9018d84ceab

qwen-image-vae-xsmax-fp16.animapk
256,163,840 bytes
SHA256:
10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447
```

The VAE already achieved approximately `0.99981` same-pack Swift/Metal parity, so there is no evidence-driven reason to rebuild or modify it during this cycle.

Leave both components alone.

---

# 5. Synchronize the real repository first

Before modifying anything:

```bash
cd /root/AnimaXS

git status --short
git remote -v

git fetch origin --prune

echo "REMOTE MAIN:"
git ls-remote origin refs/heads/main

git switch main
git pull --ff-only origin main

echo "BASE HEAD:"
git rev-parse HEAD
git log --oneline --decorate -15

git status --short
```

Do not rely on any SHA quoted by an earlier agent report.

Do not assume the artifact branch is current.

Do not assume previous TODO documents describe the actual repository state.

Only after synchronizing:

```bash
git switch -c refine/packing-v2-w4-w8
```

If this branch already exists remotely, inspect it before deciding whether to continue or create another branch.

---

# 6. Install this runbook into the repository

Save this entire document as:

```text
RUNBOOK.md
```

The execution agent must re-read it:

```text
before every major phase
after context compaction
after an agent restart
before changing the packer
before changing Metal/runtime dtype handling
before starting packing CI
before interpreting W4/W8 results
```

Do not rely on conversational context alone.

The repository copy is authoritative for this refinement run.

---

# 7. Archive the old project-completion task documents

The project is no longer trying to “finish the original TODO list.”

Inspect first:

```bash
find . -maxdepth 2 -type f \
  \( -iname '*runbook*' \
     -o -iname '*todo*' \
     -o -iname '*handoff*' \
     -o -iname '*test_matrix*' \) \
  -print
```

Create:

```bash
mkdir -p docs/archive/pre-refine-2026-08-13
```

Archive superseded operational documents using `git mv`.

For example, only where appropriate:

```bash
git mv OLD_RUNBOOK.md docs/archive/pre-refine-2026-08-13/
git mv OLD_TODO.md docs/archive/pre-refine-2026-08-13/
```

If the current files are literally named:

```text
RUNBOOK.md
TODO.md
NEXT_TASK_HANDOFF.md
TEST_MATRIX.md
```

archive their old versions before replacing them.

## Keep DECISIONS.md

Do **not** archive or rewrite:

```text
DECISIONS.md
```

It remains useful.

It is explicitly append-only in the current repository.

Never rewrite old decisions merely because our interpretation has improved.

Append new decisions that supersede old operational choices.

---

# 8. Documentation reset

Create fresh:

```text
RUNBOOK.md
TODO.md
TEST_MATRIX.md
NEXT_TASK_HANDOFF.md
```

Update:

```text
STATUS.md
```

Suggested new project status:

> AnimaXS is in refine-and-improve mode. The complete 3-pack inference pipeline executes end-to-end and produces image output. Current work focuses on DiT packing/quantization quality, W4/W8 runtime support, canonical image-quality comparison, and CI determinism. Production-readiness work is intentionally deferred until a satisfactory model representation has been identified.

Append a new decision after the existing D074.

Suggested conceptual decision:

```text
D075: Refinement phase begins.

The integrated inference graph is considered connected and functional.
Legacy W4 source-vs-BF16 metrics (~0.6946 latent / ~0.7035 RGB) remain
historical regression evidence but are not image-quality acceptance targets.

New work focuses on reproducible DiT repacking, W4/W8 support and direct
full-image comparison. Qwen and VAE remain unchanged until evidence says
otherwise.
```

---

# 9. Fix D072 CI determinism

Do this early.

The current historical D072 says the normal CI test assumes no packs exist in simulator Application Support and occasionally observes otherwise.

Do **not** modify D072 itself.

Append a newer decision after fixing it.

One correction is worth recording:

GitHub's current documentation says standard hosted jobs normally receive a fresh VM.

Therefore the old explanation:

> “the shared hosted runner reused our old VM”

should remain historical context, not be treated as a guaranteed GitHub behavior.

The engineering fix is still simple:

> explicitly establish the simulator state that the test requires.

## Preferred workflow-only fix

After selecting the simulator but before running pack-free tests:

```yaml
- name: Reset selected simulator state
  shell: bash
  run: |
    set -euo pipefail

    UDID="${{ steps.sim.outputs.sim }}"

    if [ -z "$UDID" ]; then
      echo "::error::No simulator UDID selected"
      exit 1
    fi

    echo "Resetting selected simulator before pack-free tests"

    xcrun simctl shutdown "$UDID" 2>/dev/null || true
    xcrun simctl erase "$UDID"
```

Then allow `xcodebuild` to boot/use it normally.

If necessary:

```yaml
- name: Reset and boot selected simulator
  shell: bash
  run: |
    set -euo pipefail

    UDID="${{ steps.sim.outputs.sim }}"

    xcrun simctl shutdown "$UDID" 2>/dev/null || true
    xcrun simctl erase "$UDID"
    xcrun simctl boot "$UDID"
    xcrun simctl bootstatus "$UDID" -b
```

## Do not

Do not:

```text
weaken DiagnosticsTests
skip DiagnosticsTests
change expected FAIL to PASS-or-FAIL
delete arbitrary CoreSimulator filesystem paths
erase every simulator on the machine
re-run CI repeatedly hoping for luck
accept "green except D072" forever
```

The test's environment should be deterministic.

---

# 10. VPS storage policy

Since packing is moving entirely to GitHub-hosted Linux, large weight files should disappear from the VPS workflow.

Check:

```bash
df -h /
df -ih /

du -xh --max-depth=2 /root 2>/dev/null \
  | sort -h \
  | tail -80

find /root -xdev -type f -size +250M \
  -printf '%s %p\n' 2>/dev/null \
  | sort -n
```

The legacy W4 pack is already available from the existing public model release and is recorded with SHA:

```text
ba1ce615f03665812f05088f9239f0cb23591a0811067d57fa51773abf6f0d25
```

If a duplicate local copy is consuming significant VPS space:

1. verify its SHA;
2. verify the remote release still exists;
3. ensure nothing currently running needs the local file;
4. delete the local duplicate.

Do **not** retain multi-GB model files on the VPS “just in case.”

The new durable copies belong on Hugging Face.

---

# 11. Bootstrap the Hugging Face token into GitHub Secrets

The user has placed the Hugging Face API token somewhere under `/root`.

Never display it.

Never commit it.

Never print even a partial token.

Never put it directly into workflow YAML.

## Find candidate token files without printing contents

```bash
find /root -maxdepth 2 -type f \
  \( -iname '*hugging*' \
     -o -iname '*hf*token*' \
     -o -iname '*api*key*' \) \
  -ls
```

Identify the intended token file.

Disable shell tracing:

```bash
set +x
```

Use the existing authenticated GitHub CLI to create:

```text
HF_TOKEN
```

as a repository secret.

If the file contains only the raw token:

```bash
gh secret set HF_TOKEN \
  --repo invisiblestrangler/AnimaXS \
  < /path/to/hf_token_file
```

If it is an environment-style file such as:

```text
HF_TOKEN=hf_...
```

parse it without printing.

For example:

```bash
python3 - /path/to/hf_token_file <<'PY' \
  | gh secret set HF_TOKEN \
      --repo invisiblestrangler/AnimaXS

import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text().strip()

if text.startswith("HF_TOKEN="):
    text = text.split("=", 1)[1].strip()

if not text:
    raise SystemExit("empty Hugging Face token")

print(text, end="")
PY
```

This sends the token directly into `gh secret set`.

Then:

```bash
gh secret list --repo invisiblestrangler/AnimaXS
```

Confirm **only that the secret name exists**.

Never attempt to read it back.

---

# 12. The original source weights come directly from Hugging Face

The source model is currently available from:

```text
circlestone-labs/Anima
```

and the actual Turbo diffusion file is:

```text
split_files/diffusion_models/anima-turbo-v1.0.safetensors
```

Hugging Face currently reports this file as approximately **4.18 GB**.

The project previously identified the intended source file SHA-256 as:

```text
c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
```

That exact SHA is a **hard gate**.

---

# 13. Pin the Hugging Face source revision

Do not permanently build from:

```text
revision=main
```

because upstream `main` can change later.

The execution agent must:

1. query the current full Hugging Face repository revision;
2. download the exact Turbo file;
3. calculate its SHA-256;
4. confirm it equals:

```text
c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
```

5. record the full HF commit/revision in the workflow.

The final packing workflow should therefore contain something like:

```yaml
env:
  ANIMA_SOURCE_REPO: circlestone-labs/Anima
  ANIMA_SOURCE_REVISION: <FULL_PINNED_HF_REVISION>
  ANIMA_SOURCE_FILE: split_files/diffusion_models/anima-turbo-v1.0.safetensors
  ANIMA_SOURCE_SHA256: c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
```

Never silently accept a changed upstream source.

---

# 14. Preserve the recovered legacy packer

The attached recovered `pack_anima.py` is important historical evidence.

It says explicitly that it creates:

```text
anima-turbo-v1.0-xsmax-w4.animapk
qwen3-0.6b-xsmax-w8.animapk
qwen-image-vae-xsmax-fp16.animapk
```

Its quantization format is:

```text
W4: unsigned uint4 affine
W8: unsigned uint8 affine
group size: 64
scale: fp16
zero: fp16
decode: q * scale + zero
```

It correctly resets matrix groups on each output row.

Preserve it unchanged as:

```text
scripts/archive/pack_anima_v1_extracted.py
```

Before committing:

```bash
sha256sum scripts/archive/pack_anima_v1_extracted.py
```

The SHA measured during this investigation was:

```text
953e3ae1c2409584a0b0c7f849779400f891ad15e9e5c1b48c3851a2a4cdf185
```

Recalculate this locally before treating it as authoritative.

Do not modify the archived file.

Develop the replacement at:

```text
scripts/pack_anima.py
```

---

# 15. Why packing v2 is needed

The recovered packer works, but it has several refinement limitations.

## 15.1 It is not actually output-streaming

It reads tensors one at a time but appends each tensor's finished packed byte arrays into the global `tensors` list and only writes the final pack afterward.

That retains approximately the entire quantized output in memory.

v2 must write each finished tensor to the output file and release it.

---

## 15.2 W4 range selection is extremely basic

For every group of 64 weights, v1 simply computes:

```python
mn = seg.min()
mx = seg.max()
scale = (mx - mn) / 15
zero = mn
```

then rounds into 16 values.

There is no range optimization.

There is no outlier clipping.

There is no MSE search.

This is a plausible source of accumulated W4 degradation.

---

## 15.3 Integer Q is chosen against parameters the runtime does not use exactly

v1 calculates Q using Float32 `scale`/`zero`, then converts them to FP16 for storage afterward.

The runtime reconstructs using the stored FP16 values.

v2 should:

```text
calculate candidate affine range
→ convert scale/zero to fp16
→ convert them back to float32
→ choose Q using those exact stored values
```

That removes the inconsistency.

---

## 15.4 Mixed precision was not completed end-to-end

The v1 script already contains:

```text
--exclude-json
```

and can leave selected matrices FP16.

But current DiT-specific runtime paths still assume W4 in several places.

v2 should make:

```text
W4
W8
```

first-class per-matrix choices.

Do **not** add general FP16 rank-2 matrix execution yet unless later evidence requires it.

---

# 16. Packing-v2 design

Keep the ANMA container at version `1` unless a concrete incompatibility makes that impossible.

The current parser is header-driven and validates JSON/table/payload bounds and 16-KiB blob alignment rather than requiring the table immediately after JSON.

Therefore a bounded-memory writer can reserve a fixed metadata area without changing the runtime format.

Required behavior:

```text
Container               ANMA v1
Blob alignment          16 KiB
Group size              64
Rank <= 1               FP16
Rank 2                  W4 or W8
W4                      optimized affine
W8                      affine
Scale                    FP16
Zero                     FP16
Group reset              every matrix row
W4 nibble layout         even K low / odd K high
Output writer            bounded-memory
Source finite check      mandatory
Scale finite check       mandatory
Verification             mandatory
Quant report             mandatory
Source provenance        mandatory
Precision map            supported
Dry-plan                 mandatory
```

---

# 17. Do not change group size yet

Use:

```text
GROUP = 64
```

for both W4-v2 and W8-v2.

The existing Swift decoder already implements row-aware W4/W8 group-64 matrix decoding, including the historical `[2,68]` and `[2,65]` boundary cases.

Do not simultaneously change:

```text
bit depth
quantizer
group size
runtime
```

during the first comparison.

The experiment should remain understandable.

---

# 18. Packing-v2 CLI

Implement approximately:

```bash
python3 scripts/pack_anima.py \
  --component dit \
  --input source.safetensors \
  --out anima-turbo-v1.0-xsmax-w4-v2.animapk \
  --quant w4 \
  --group 64 \
  --w4-algorithm mseclip \
  --report anima-turbo-v1.0-xsmax-w4-v2.quant-report.json \
  --verify
```

and:

```bash
python3 scripts/pack_anima.py \
  --component dit \
  --input source.safetensors \
  --out anima-turbo-v1.0-xsmax-w8-v2.animapk \
  --quant w8 \
  --group 64 \
  --report anima-turbo-v1.0-xsmax-w8-v2.quant-report.json \
  --verify
```

Also support:

```text
--precision-map PATH
--dry-plan
--row-chunk N
--verify
```

---

# 19. Precision-map design

Support per-matrix storage now even though the first two packs are pure W4 and pure W8.

Suggested format:

```json
{
  "version": 1,
  "default": "w4",
  "overrides": [
    {
      "match": "model.diffusion_model.final_layer.*",
      "storage": "w8"
    }
  ]
}
```

Rules:

```text
rank <= 1:
    always FP16

rank == 2:
    resolved from default + overrides

allowed rank-2 storage:
    W4
    W8
```

Hard fail when:

```text
storage type is unknown
override pattern is malformed
an intended override matches zero tensors
multiple conflicting rules match one tensor
rank > 2 reaches matrix quantization unexpectedly
```

For first W4:

```json
{
  "version": 1,
  "default": "w4",
  "overrides": []
}
```

For first W8:

```json
{
  "version": 1,
  "default": "w8",
  "overrides": []
}
```

This means that if W8 is clean and W4 is not, generating one mixed W4/W8 follow-up later requires **no new pack format or runtime redesign**.

---

# 20. W4 range optimizer

Do not implement GPTQ, AWQ, calibration datasets or Hessian-based optimization yet.

That would overcomplicate the first refinement experiment.

Use a modest deterministic per-group MSE range search.

For each 64-element group, test:

```text
baseline min/max
symmetric 0.5% clipping
symmetric 1%
symmetric 2%
symmetric 3%
symmetric 5%

lower-only 1%
lower-only 2%
lower-only 5%

upper-only 1%
upper-only 2%
upper-only 5%
```

The exact implementation may be adjusted if a cleaner equivalent is found.

For every candidate:

1. determine candidate lower/upper range;
2. calculate scale/zero;
3. round scale/zero to FP16;
4. reconstruct their exact runtime Float32 values;
5. quantize Q using those values;
6. reconstruct;
7. calculate MSE against original FP32 group;
8. select lowest-MSE candidate.

Example core helper:

```python
import numpy as np

def evaluate_range(seg, bits, lo, hi):
    qmax = (1 << bits) - 1

    if hi - lo < 1e-8:
        scale16 = np.float16(1.0)
        zero16 = np.float16(lo)
        q = np.zeros(seg.shape, dtype=np.uint8)

        recon = (
            q.astype(np.float32) * float(scale16)
            + float(zero16)
        )

        mse = float(np.mean((recon - seg) ** 2))

        return mse, q, scale16, zero16

    scale16 = np.float16((hi - lo) / qmax)
    zero16 = np.float16(lo)

    scale = float(scale16)
    zero = float(zero16)

    if not np.isfinite(scale):
        return None

    if not np.isfinite(zero):
        return None

    if scale <= 0:
        return None

    q = np.clip(
        np.rint((seg - zero) / scale),
        0,
        qmax
    ).astype(np.uint8)

    recon = q.astype(np.float32) * scale + zero

    mse = float(
        np.mean((recon - seg) ** 2)
    )

    return mse, q, scale16, zero16
```

Do not implement the entire model as individual Python 64-element loops.

Vectorize group candidates across row chunks.

---

# 21. W8 algorithm

Keep W8 relatively simple.

With 256 quantization levels, first use ordinary per-group affine min/max, but still fix the FP16 parameter issue.

Conceptually:

```python
scale16 = np.float16((mx - mn) / 255.0)
zero16 = np.float16(mn)

scale = np.float32(scale16)
zero = np.float32(zero16)

q = np.clip(
    np.rint((seg - zero) / scale),
    0,
    255
).astype(np.uint8)
```

If W8 still produces a bad image, do not immediately optimize W8 ranges.

That result would be diagnostically interesting.

---

# 22. Vectorized row-chunk processing

The GitHub runner should not spend hours executing millions of tiny Python loops.

Use row chunks.

Conceptually:

```python
def process_matrix_chunk(chunk, bits):
    # chunk shape: [rows, K]

    rows, k = chunk.shape

    groups_per_row = (k + GROUP - 1) // GROUP
    padded_k = groups_per_row * GROUP

    # Build explicit validity mask so padded values cannot
    # influence the final partial group's statistics.

    ...
```

Important invariants:

```text
groups restart each row

K=68:
row 0 groups 0,1
row 1 groups 2,3

partial groups ignore padding during:
min
max
MSE

W4:
even K index -> low nibble
odd K index  -> high nibble
```

The historical row-boundary bug must never return.

---

# 23. Quantization report

Packing should produce useful information automatically.

Do not run a second multi-GB analysis pass afterward.

Accumulate statistics while quantizing:

```text
sum_x2
sum_y2
sum_xy
sum_squared_error
sum_absolute_error
max_absolute_error
element_count
q_zero_count
q_max_count
```

Then calculate:

```python
cosine = sum_xy / sqrt(sum_x2 * sum_y2)

rmse = sqrt(
    sum_squared_error / element_count
)

relative_l2 = sqrt(
    sum_squared_error / sum_x2
)

mae = (
    sum_absolute_error / element_count
)
```

Per tensor:

```json
{
  "name": "model.diffusion_model....weight",
  "shape": [2048, 2048],
  "storage": "w4",
  "group": 64,
  "cosine": 0.999,
  "rmse": 0.01,
  "relative_l2": 0.03,
  "mae": 0.005,
  "max_abs": 0.2,
  "q_zero_fraction": 0.02,
  "q_max_fraction": 0.01
}
```

Final report must contain sorted sections:

```text
worst_by_relative_l2
worst_by_cosine
worst_by_rmse
worst_by_max_abs
```

This becomes very useful if mixed W4/W8 is needed later.

---

# 24. Make output writing genuinely bounded-memory

Do not retain:

```text
data
scale
zero
```

for every tensor until the end.

Use two phases.

## Phase A — planning

Read only metadata/shapes.

Determine:

```text
tensor name
shape
storage type
data bytes
scale bytes
zero bytes
aligned blob bytes
execution index
block index
blob offset
```

For matrix `[rows, columns]`:

```python
groups_per_row = (
    columns + GROUP - 1
) // GROUP
```

W4:

```python
packed_row_bytes = (
    columns + 1
) // 2

data_size = (
    rows * packed_row_bytes
)

param_size = (
    rows
    * groups_per_row
    * 2
)
```

W8:

```python
data_size = (
    rows * columns
)

param_size = (
    rows
    * groups_per_row
    * 2
)
```

FP16:

```python
data_size = (
    rows
    * columns
    * 2
)

scale_size = 0
zero_size = 0
```

---

# 25. Reserve metadata space

Because the current ANMA parser gets offsets from the binary header and only requires sections to remain in bounds, a fixed metadata reserve is compatible with the existing parser design.

For example:

```python
METADATA_RESERVE = 4 * 1024 * 1024
```

Then:

```text
256-byte header

reserved JSON area

tensor table

align to 16 KiB

payload tensor 0
align

payload tensor 1
align

...
```

Calculate final offsets before loading full tensors.

If the final metadata unexpectedly exceeds the reserve:

```text
HARD FAIL
```

Do not silently corrupt layout.

---

# 26. Direct writer pattern

Conceptually:

```python
with open(output_path, "w+b") as dst:
    dst.truncate(final_file_size)

    with safe_open(
        source_path,
        framework="pt",
        device="cpu"
    ) as src_file:

        for plan in tensor_plans:
            tensor = src_file.get_tensor(
                plan.name
            )

            process_and_write_tensor(
                dst=dst,
                plan=plan,
                source=tensor
            )

            del tensor

    metadata = build_final_metadata(...)
    table = build_table(...)

    dst.seek(0)

    write_header(dst, ...)
    write_metadata(dst, metadata)
    write_table(dst, table)
```

The completed quantized matrix should be written directly to its planned file offset.

Do not create a second complete temporary `.animapk`.

---

# 27. Strengthen integrity without changing ANMA v1

The existing binary CRC covers the packed data region.

Keep that behavior for runtime compatibility.

Additionally put this into v2 JSON metadata:

```text
blob_sha256
```

where:

```text
blob_sha256 =
SHA256(
    data
    + scale
    + zero
)
```

This allows the v2 verifier to protect all quantization parameters without altering the ANMA binary record format.

The runtime does not have to validate this during normal inference yet.

The offline pack verifier should.

---

# 28. Provenance metadata

Every v2 pack must contain/report:

```json
{
  "packer": {
    "version": 2,
    "script_sha256": "...",
    "git_commit": "...",
    "python": "...",
    "numpy": "...",
    "torch": "...",
    "safetensors": "..."
  },

  "source": {
    "repo": "circlestone-labs/Anima",
    "revision": "...",
    "path": "split_files/diffusion_models/anima-turbo-v1.0.safetensors",
    "sha256": "c0b905..."
  },

  "quantization": {
    "default": "w4",
    "group": 64,
    "w4_algorithm": "mseclip-v1",
    "scale_dtype": "fp16",
    "zero_dtype": "fp16"
  },

  "precision_map_sha256": "...",

  "output": {
    "filename": "...",
    "bytes": 0,
    "sha256": "..."
  }
}
```

Do not put secrets in provenance.

---

# 29. Add `--dry-plan`

The packer must be able to predict resources without producing the pack.

Example:

```bash
python3 scripts/pack_anima.py \
  --component dit \
  --input source.safetensors \
  --out ignored.animapk \
  --quant w8 \
  --group 64 \
  --dry-plan
```

Output:

```text
SOURCE
------
filename:
bytes:
SHA256:

MODEL
-----
tensor count:
rank-2 tensors:
rank<=1 tensors:

OUTPUT
------
W4 tensors:
W8 tensors:
FP16 tensors:

estimated data bytes:
estimated scale bytes:
estimated zero bytes:
estimated padding:
estimated final file size:

largest source tensor:
estimated peak working memory:
```

GitHub CI should run `--dry-plan` before the real pack.

---

# 30. Pack verifier

Create:

```text
scripts/verify_animapk.py
```

Do not make verification part of undocumented packer internals only.

Validate:

```text
magic == ANMA
version == 1
component == DiT
alignment == 16384
record size == 128
declared file size == real file size

JSON in bounds
table in bounds
payload in bounds

tensor count correct

every blob:
    offset aligned
    range in file

every W4 matrix:
    expected packed bytes
    expected scale bytes
    expected zero bytes

every W8 matrix:
    expected packed bytes
    expected scale bytes
    expected zero bytes

every FP16 tensor:
    expected bytes
    no quant params

every scale:
    finite
    > 0

every zero:
    finite

CRC:
    correct

v2 blob SHA:
    correct

source provenance:
    present

whole output SHA:
    generated
```

---

# 31. Add tiny packer regression tests

No multi-GB source is needed for normal CI.

Create synthetic tensors:

```text
W4 matrix:  [2,68]
W8 matrix:  [2,65]
FP16 vector
```

Use deliberately different value ranges between row 0 and row 1.

Tests must catch:

```text
global group indexing
wrong nibble packing
partial-group padding contamination
wrong W8 row stride
scale/zero length errors
metadata corruption
```

The current project already has historical regression evidence specifically around `[2,68]` W4 and `[2,65]` W8.

---

# 32. GitHub-hosted packing resources

Use:

```yaml
runs-on: ubuntu-latest
```

The repository is public.

GitHub currently provides standard public `ubuntu-latest` runners with approximately:

```text
4 CPU
16 GB RAM
14 GB SSD
```

The source model is approximately:

```text
4.18 GB
```

Therefore **do not build W4 and W8 in the same job filesystem**.

Give each its own VM.

---

# 33. Packing workflow topology

Create:

```text
.github/workflows/pack-dit-v2.yml
```

Use a two-entry matrix:

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - variant: w4-v2
        quant: w4
        output: anima-turbo-v1.0-xsmax-w4-v2.animapk
        hf_repo_suffix: AnimaXS-DiT-W4

      - variant: w8-v2
        quant: w8
        output: anima-turbo-v1.0-xsmax-w8-v2.animapk
        hf_repo_suffix: AnimaXS-DiT-W8
```

This creates:

```text
JOB 1:
source 4.18 GB
+
W4 output only

JOB 2:
source 4.18 GB
+
W8 output only
```

Each gets its own fresh runner disk.

---

# 34. GitHub-hosted job timeout

GitHub currently limits a standard hosted job to **6 hours**.

Use:

```yaml
timeout-minutes: 330
```

That leaves headroom below the hard platform timeout.

The packer must print progress periodically:

```text
tensor 1 / N
tensor 25 / N
tensor 50 / N
...

bytes written
elapsed
current tensor
```

Do not allow a four-hour job to appear completely silent.

If packing approaches the timeout, first improve/vectorize the CPU quantizer.

Do **not** immediately add external paid infrastructure.

---

# 35. Source download in the packing job

Install only required Python dependencies.

Example:

```yaml
- name: Install packing dependencies
  run: |
    set -euo pipefail

    python3 -m pip install --upgrade pip

    python3 -m pip install \
      numpy \
      safetensors \
      huggingface_hub

    python3 -m pip install \
      torch \
      --index-url https://download.pytorch.org/whl/cpu
```

Then inspect disk:

```yaml
- name: Initial disk report
  run: |
    df -h
    du -sh "$RUNNER_TEMP" || true
```

Download only the required source file.

Do not clone the complete 34.9-GB Anima repository. The upstream HF repository as a whole is currently about 34.9 GB, while the required Turbo file is only about 4.18 GB.

Use:

```bash
hf download \
  "$ANIMA_SOURCE_REPO" \
  "$ANIMA_SOURCE_FILE" \
  --revision "$ANIMA_SOURCE_REVISION" \
  --local-dir "$RUNNER_TEMP/source"
```

Then find the resulting exact file path and verify:

```bash
sha256sum "$SOURCE"
```

Hard gate:

```bash
echo \
  "$ANIMA_SOURCE_SHA256  $SOURCE" \
  | sha256sum --check
```

If SHA differs:

```text
STOP
```

Do not pack it.

---

# 36. Watch Hugging Face cache/disk use

Immediately after source download:

```bash
df -h

du -sh \
  "$RUNNER_TEMP" \
  "$HOME/.cache/huggingface" \
  2>/dev/null || true
```

The runner has only 14 GB SSD.

If `hf download --local-dir` unexpectedly leaves a duplicate multi-GB cache and makes W8 unsafe:

1. determine where the actual source bytes live;
2. remove only redundant cache data;
3. retain the one verified source file;
4. rerun `sha256sum`;
5. continue.

Do not delete the source path while the packer still needs it.

If necessary, switch that workflow to a direct streaming HTTP download of the **same pinned HF revision** to avoid duplicate caching.

Do not redesign infrastructure over a cache implementation detail.

---

# 37. Create the Hugging Face repos automatically

The action already has:

```text
secrets.HF_TOKEN
```

Use the Hugging Face API to determine the token owner.

Example:

```python
import os
from huggingface_hub import HfApi

api = HfApi(
    token=os.environ["HF_TOKEN"]
)

identity = api.whoami()

owner = identity["name"]

print(owner)
```

Printing the account name is fine.

Do not print the token.

For W4:

```text
<owner>/AnimaXS-DiT-W4
```

For W8:

```text
<owner>/AnimaXS-DiT-W8
```

Create them if necessary:

```python
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True,
)
```

If the user account cannot create public repos or the token lacks write permission:

```text
HARD FAIL
```

Do not silently switch to another account.

---

# 38. Hugging Face model licensing is mandatory

The existing repository's license audit concludes that non-commercial redistribution of the quantized Anima derivative is permitted only with the required notices/license material.

Therefore every W4/W8 HF repo must contain:

```text
README.md
MODEL_LICENSE.md
MODEL_NOTICE.txt

licenses/
  CircleStone-NC-1.2.md
  NVIDIA-Open-Model-License.txt
```

Copy these from the audited GitHub project rather than rewriting the license text.

The HF model card must clearly state:

```text
AnimaXS converted/quantized derivative
non-commercial model use
source: circlestone-labs/Anima
Built on NVIDIA Cosmos
packing algorithm/version
source SHA
output SHA
intended iPhone/Metal runtime
experimental/refinement status
```

Do not present these experimental packs as official CircleStone releases.

---

# 39. Hugging Face repo structure

Because each precision gets its own model repo, keep the structure simple.

Example W4:

```text
README.md
MODEL_LICENSE.md
MODEL_NOTICE.txt

licenses/
  CircleStone-NC-1.2.md
  NVIDIA-Open-Model-License.txt

anima-turbo-v1.0-xsmax-w4-v2.animapk

quant-report.json
packing-manifest.json
verification-report.json
SHA256SUMS
```

W8 equivalent:

```text
README.md
MODEL_LICENSE.md
MODEL_NOTICE.txt

licenses/
  CircleStone-NC-1.2.md
  NVIDIA-Open-Model-License.txt

anima-turbo-v1.0-xsmax-w8-v2.animapk

quant-report.json
packing-manifest.json
verification-report.json
SHA256SUMS
```

No splitting.

No multipart reassembly.

No GitHub Release limit workaround.

---

# 40. Upload directly from Actions to Hugging Face

After packing and verification succeed:

```bash
hf upload \
  "$HF_REPO" \
  "$OUTPUT_PACK" \
  "$(basename "$OUTPUT_PACK")" \
  --repo-type model \
  --commit-message "Upload ${VARIANT} ANIMAPK"
```

The HF CLI handles large files automatically.

Then upload:

```text
quant report
packing manifest
verification report
SHA256SUMS
license files
model card
```

Do not delete the local output yet.

First verify it landed remotely.

---

# 41. Remote-after-upload verification

After upload, query Hugging Face metadata.

Verify:

```text
repository exists
expected filename exists
expected size exists
commit/revision recorded
```

Then download the remote pack's metadata/checksum if available.

For strongest verification, if disk permits:

```text
local pack SHA
==
recorded manifest SHA
==
HF stored file identity
```

If full redownload would endanger the 14-GB runner disk, do not redownload 2+ GB merely for ceremony.

The original local SHA plus successful HF large-file upload metadata is enough for the packing job.

The later macOS full-inference download provides an independent end-to-end download check.

---

# 42. GitHub artifact from packing job — SMALL FILES ONLY

After successful HF upload:

```yaml
- name: Upload packing evidence
  uses: actions/upload-artifact@v7
  with:
    name: pack-${{ matrix.variant }}-evidence
    retention-days: 7
    path: |
      out/*.quant-report.json
      out/*.packing-manifest.json
      out/*.verification-report.json
      out/SHA256SUMS
```

Do **not** include:

```text
*.animapk
*.safetensors
```

The current upload-artifact documentation supports many independently named artifacts; these evidence files are the appropriate use case.

---

# 43. Packing workflow skeleton

The final workflow should look conceptually like:

```yaml
name: Pack DiT v2

on:
  workflow_dispatch:

permissions:
  contents: read

env:
  ANIMA_SOURCE_REPO: circlestone-labs/Anima
  ANIMA_SOURCE_REVISION: <PINNED FULL REVISION>
  ANIMA_SOURCE_FILE: split_files/diffusion_models/anima-turbo-v1.0.safetensors
  ANIMA_SOURCE_SHA256: c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174

jobs:
  pack:
    runs-on: ubuntu-latest
    timeout-minutes: 330

    strategy:
      fail-fast: false
      matrix:
        include:
          - variant: w4-v2
            quant: w4
            output: anima-turbo-v1.0-xsmax-w4-v2.animapk
            hf_suffix: AnimaXS-DiT-W4

          - variant: w8-v2
            quant: w8
            output: anima-turbo-v1.0-xsmax-w8-v2.animapk
            hf_suffix: AnimaXS-DiT-W8

    steps:
      - uses: actions/checkout@v6

      - name: Disk before
        run: |
          df -h

      - name: Install dependencies
        run: |
          set -euo pipefail

          python3 -m pip install --upgrade pip

          python3 -m pip install \
            numpy \
            safetensors \
            huggingface_hub

          python3 -m pip install \
            torch \
            --index-url https://download.pytorch.org/whl/cpu

      - name: Download exact source
        run: |
          set -euo pipefail

          mkdir -p "$RUNNER_TEMP/source"

          hf download \
            "$ANIMA_SOURCE_REPO" \
            "$ANIMA_SOURCE_FILE" \
            --revision "$ANIMA_SOURCE_REVISION" \
            --local-dir "$RUNNER_TEMP/source"

          SOURCE="$RUNNER_TEMP/source/$ANIMA_SOURCE_FILE"

          echo \
            "$ANIMA_SOURCE_SHA256  $SOURCE" \
            | sha256sum --check

          echo "SOURCE=$SOURCE" >> "$GITHUB_ENV"

          df -h

      - name: Dry plan
        run: |
          python3 scripts/pack_anima.py \
            --component dit \
            --input "$SOURCE" \
            --out "$RUNNER_TEMP/unused.animapk" \
            --quant "${{ matrix.quant }}" \
            --group 64 \
            --dry-plan

      - name: Pack
        run: |
          set -euo pipefail

          mkdir -p out

          python3 scripts/pack_anima.py \
            --component dit \
            --input "$SOURCE" \
            --out "out/${{ matrix.output }}" \
            --quant "${{ matrix.quant }}" \
            --group 64 \
            --report "out/${{ matrix.variant }}.quant-report.json"

      - name: Verify pack
        run: |
          python3 scripts/verify_animapk.py \
            "out/${{ matrix.output }}" \
            --report \
            "out/${{ matrix.variant }}.verification-report.json"

      - name: Generate hashes and manifest
        run: |
          set -euo pipefail

          cd out

          sha256sum \
            "${{ matrix.output }}" \
            "${{ matrix.variant }}.quant-report.json" \
            "${{ matrix.variant }}.verification-report.json" \
            > SHA256SUMS

      - name: Prepare Hugging Face repository
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python3 scripts/prepare_hf_model_repo.py \
            --suffix "${{ matrix.hf_suffix }}" \
            --variant "${{ matrix.variant }}"

      - name: Upload durable model to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python3 scripts/publish_hf_pack.py \
            --suffix "${{ matrix.hf_suffix }}" \
            --pack "out/${{ matrix.output }}" \
            --report "out/${{ matrix.variant }}.quant-report.json" \
            --verification "out/${{ matrix.variant }}.verification-report.json" \
            --sha-file out/SHA256SUMS

      - name: Upload small CI evidence
        uses: actions/upload-artifact@v7
        with:
          name: pack-${{ matrix.variant }}-evidence
          retention-days: 7
          path: |
            out/*.json
            out/SHA256SUMS

      - name: Final disk report
        if: always()
        run: |
          df -h
```

The execution agent should adjust action major versions to the currently supported versions if necessary, but do not downgrade just because an old workflow uses an older major.

---

# 44. Runtime must support both W4 and W8

Do this before full W8 inference.

The existing project already has significant W8 infrastructure.

Current historical evidence includes W8 row-aware decoding and successful Qwen W8 execution.

The remaining work is primarily the DiT-specific validation/direct-matvec code that still assumes `.w4`.

Do not fork the entire DiT implementation into:

```text
DiTW4
DiTW8
```

There should be **one inference graph**.

Storage dtype is a property of each weight.

---

# 45. Create one common DiT quantized-weight factory

Create something such as:

```text
AnimaXS/Runtime/Metal/DiTQuantizedWeightFactory.swift
```

Conceptually:

```swift
enum DiTQuantizedWeightFactory {

    static func makeMatrix(
        _ item: AnimapkTensorSpans,
        ring: MTLBuffer,
        rows: Int,
        columns: Int,
        label: String
    ) throws -> QuantizedLinearWeightBuffers {

        let storage = item.tensor.storage

        guard storage == .w4 ||
              storage == .w8 else {

            throw AnimapkError.validation(
                "\(label) must be W4 or W8"
            )
        }

        guard item.tensor.shape == [
            rows,
            columns
        ] else {

            throw AnimapkError.validation(
                "\(label) shape mismatch"
            )
        }

        guard
            let scale = item.scale,
            let zero = item.zero
        else {

            throw AnimapkError.validation(
                "\(label) missing scale/zero"
            )
        }

        let rowBytes: Int

        switch storage {
        case .w4:
            rowBytes = (
                columns + 1
            ) / 2

        case .w8:
            rowBytes = columns

        default:
            fatalError(
                "guarded above"
            )
        }

        let groupsPerRow = (
            columns + 63
        ) / 64

        let expectedData =
            rows * rowBytes

        let expectedParams =
            rows
            * groupsPerRow
            * MemoryLayout<Float16>.stride

        guard
            item.data.length
                == UInt64(expectedData),

            scale.length
                == UInt64(expectedParams),

            zero.length
                == UInt64(expectedParams)
        else {

            throw AnimapkError.validation(
                "\(label) quantized layout mismatch"
            )
        }

        return QuantizedLinearWeightBuffers(
            storage: storage,

            packed: ring,
            packedOffset:
                Int(item.data.offset),

            scale: ring,
            scaleOffset:
                Int(scale.offset),

            zero: ring,
            zeroOffset:
                Int(zero.offset),

            rows: rows,
            columns: columns,

            packedRowStride:
                rowBytes
        )
    }
}
```

Use this in:

```text
DiTPreparationExecutor
DiTBlockExecutor
DiTFinalLayerExecutor
```

Do not retain three separate W4-only validators.

---

# 46. Add W8 FP32 direct matvec

The direct modulation/timestep/final-layer paths currently use a specialized W4 FP32 matvec.

Add:

```text
w8_matvec_f32
```

Copy the **existing W4 kernel's execution/reduction structure**.

Change only packed weight decoding.

Conceptually:

```metal
const uint q =
    uint(
        packed[
            row * rowStride
            + column
        ]
    );

const uint group =
    row * groupsPerRow
    + column / 64u;

const float scale =
    float(scales[group]);

const float zero =
    float(zeros[group]);

const float weight =
    float(q) * scale + zero;

sum += input[column] * weight;
```

Do not invent a different reduction order.

Do not change residual precision.

Do not change LayerNorm/RMSNorm semantics.

Do not change GELU.

Do not change RoPE.

Do not change sampler math.

Only add W8 weight decode.

---

# 47. Centralize matvec dispatch

Conceptually:

```swift
func quantizedMatvecKernel(
    storage: StorageDtype
) throws -> String {

    switch storage {

    case .w4:
        return "w4_matvec_f32"

    case .w8:
        return "w8_matvec_f32"

    default:
        throw AnimapkError.validation(
            "DiT matvec requires W4 or W8"
        )
    }
}
```

Preparation, block modulation and final-layer modulation should all use the same dispatch logic where practical.

---

# 48. Focused runtime tests only

Do not build another massive oracle project before full inference.

Add:

## W8 matvec parity

Use:

```text
K = 65 or 68
```

and more than one row.

Compare:

```text
Metal W8 FP32 matvec
vs
CPU dequantW8Matrix + FP32 dot
```

Require tight parity.

## Weight factory

Verify:

```text
W4 accepted
W8 accepted

wrong W4 length rejected
wrong W8 length rejected

missing scale rejected
missing zero rejected

rank-2 FP16 rejected for now
```

## Packer-parser integration

Use the synthetic:

```text
W4 [2,68]
W8 [2,65]
FP16 vector
```

pack.

Open it using the real Swift `AnimapkFile`.

No large model required.

---

# 49. Preserve existing W4 semantics

Adding W8 support must not change legacy W4 execution.

Before large inference:

```text
run normal tests
run synthetic W4 tests
run existing pack-free Metal tests
```

If an existing same-W4 numerical test changes merely because W8 support was added:

```text
STOP
```

Fix that first.

---

# 50. Create both packs in ONE workflow run

Once:

```text
packer v2
runtime W8 support
D072 cleanup
HF secret
```

are committed:

manually dispatch:

```text
Pack DiT v2
```

That single workflow should launch:

```text
pack / w4-v2
pack / w8-v2
```

in parallel.

Do not:

```text
run W4
inspect
change code
then run W8
```

The first comparison should use the same code revision.

Record:

```text
GitHub workflow run ID
commit SHA
W4 job ID
W8 job ID
HF W4 repo
HF W8 repo
HF upload revisions
pack SHA values
pack byte sizes
```

---

# 51. Expected disk behavior

Each packing job should contain only:

```text
source ~4.18 GB

ONE output pack

Python dependencies

small reports
```

rather than:

```text
source
+ W4
+ W8
+ duplicate output buffers
+ VPS copies
+ release-upload staging
```

The streaming writer is still important even though GitHub has 16 GB RAM, because it reduces both memory and unnecessary temporary copies.

---

# 52. Do not upload the source model anywhere

The packing workflow downloads the official source from:

```text
circlestone-labs/Anima
```

It does not re-upload that 4.18-GB original file to:

```text
GitHub artifacts
our HF repos
GitHub Releases
```

Only derived W4/W8 outputs are persisted.

The source remains identified by:

```text
HF repo
HF revision
file path
SHA256
```

---

# 53. Full-inference workflow now consumes Hugging Face

Update/create the refinement full-inference workflow.

Use two matrix jobs:

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - variant: w4-v2
        hf_repo: <OWNER>/AnimaXS-DiT-W4
        filename: anima-turbo-v1.0-xsmax-w4-v2.animapk

      - variant: w8-v2
        hf_repo: <OWNER>/AnimaXS-DiT-W8
        filename: anima-turbo-v1.0-xsmax-w8-v2.animapk
```

Each matrix entry should also contain the exact expected:

```text
HF revision
SHA256
byte size
```

from the accepted packing run.

Do not download “latest.”

---

# 54. Pin the packed HF revisions too

After packing completes, Hugging Face gives each upload a repository revision.

The full-inference workflow must use that immutable revision.

Conceptually:

```yaml
- variant: w4-v2
  hf_repo: user/AnimaXS-DiT-W4
  hf_revision: <EXACT COMMIT>
  filename: anima-turbo-v1.0-xsmax-w4-v2.animapk
  sha256: <EXACT SHA>

- variant: w8-v2
  hf_repo: user/AnimaXS-DiT-W8
  hf_revision: <EXACT COMMIT>
  filename: anima-turbo-v1.0-xsmax-w8-v2.animapk
  sha256: <EXACT SHA>
```

That makes the test reproducible even if the HF repositories receive improved packs later.

---

# 55. Full-inference download

On the macOS runner:

```bash
hf download \
  "$DIT_HF_REPO" \
  "$DIT_FILENAME" \
  --revision "$DIT_HF_REVISION" \
  --local-dir "$RUNNER_TEMP/dit"
```

Then:

```bash
echo \
  "$EXPECTED_DIT_SHA  $DIT_PATH" \
  | shasum -a 256 --check
```

No inference begins until the SHA matches.

Use the existing validated Qwen/VAE packs exactly as before.

Do not change their hashes.

---

# 56. Avoid hardcoding W4 inside FullInferenceTests

Use one stable test-only fixture alias.

For example:

```text
anima-turbo-refine.animapk
```

Workflow:

```bash
cp \
  "$DIT_PATH" \
  "AnimaXSTests/Fixtures/Case1Binary/anima-turbo-refine.animapk"
```

Then FullInferenceTests always opens:

```swift
requiredFixture(
    name: "anima-turbo-refine.animapk"
)
```

The selected variant metadata should be another small generated fixture:

```json
{
  "variant": "w8-v2",
  "hf_repo": "...",
  "hf_revision": "...",
  "sha256": "...",
  "storage": "w8",
  "group": 64
}
```

Do not maintain separate:

```text
FullInferenceW4Tests
FullInferenceW8Tests
```

The graph must be identical.

---

# 57. Canonical inputs must remain IDENTICAL

Both W4-v2 and W8-v2 use:

```text
same canonical prompt
same T5 tokens
same Qwen tokens
same seed
same initial noise
same 8 sigmas
same Qwen pack
same VAE pack
same app/runtime commit
same output conversion
same reference latent
same RGB reference
```

The current project previously found a wrong-prompt full-inference test and corrected it.

Do not repeat that mistake.

The **only intended difference** is the DiT weight representation.

---

# 58. Full inference output

Each variant saves:

```text
generated.png
reference.png
comparison.png
metrics.json
pack-metadata.json
```

These are small and **should** be uploaded as GitHub Actions artifacts.

Artifact names:

```text
anima-xs-refine-w4-v2-images
anima-xs-refine-w8-v2-images
```

Retention:

```text
14 days
```

or another sensible short value.

Unlike model weights, these artifacts are exactly what GitHub artifact storage is good for.

---

# 59. Metrics to record

For both variants:

```text
final latent cosine
final latent RMSE
final latent MAE
final latent maxAbs

RGB cosine
RGB RMSE
RGB MAE
RGB maxAbs

Qwen seconds
adapter seconds
diffusion seconds
VAE seconds
total seconds

pack bytes
pack SHA
```

Also:

```text
224 blocks executed
8 sampler steps executed
all states finite
```

Example:

```json
{
  "variant": "w8-v2",

  "pack": {
    "sha256": "...",
    "bytes": 0,
    "hf_repo": "...",
    "hf_revision": "..."
  },

  "latent": {
    "cosine": 0.0,
    "rmse": 0.0,
    "mae": 0.0,
    "max_abs": 0.0
  },

  "rgb": {
    "cosine": 0.0,
    "rmse": 0.0,
    "mae": 0.0,
    "max_abs": 0.0
  },

  "timings": {
    "qwen": 0.0,
    "adapter": 0.0,
    "diffusion": 0.0,
    "vae": 0.0,
    "total": 0.0
  }
}
```

---

# 60. Do not call `0.65` image quality

The historical `0.65` full-loop floor exists because accumulated source-BF16 vs W4 divergence was already known.

The final image from the legacy W4 pack demonstrates that passing that floor does not imply good visual quality.

During refinement distinguish:

## Structural success

Hard fail on:

```text
invalid pack
SHA mismatch
unsupported dtype
NaN/Inf
Metal error
wrong number of diffusion steps
missing block
wrong output dimensions
PNG failure
crash
```

## Quality result

Record:

```text
latent cosine
RGB cosine
visual result
```

Do not initially make a newly invented number the sole W4/W8 winner criterion.

Inspect the PNGs.

---

# 61. Main comparison table

After both jobs finish:

```text
                         LEGACY W4     W4-v2       W8-v2
----------------------------------------------------------------
pack bytes               1.179 GB      ?            ?
latent cosine             0.6946       ?            ?
latent RMSE               0.9735       ?            ?
RGB cosine                0.7035       ?            ?
RGB RMSE                  0.493        ?            ?
pattern lattice           obvious      ?            ?
color dullness            obvious      ?            ?
diffusion time            baseline     ?            ?
total time                baseline     ?            ?
```

Include the actual three images in the final agent report where practical.

---

# 62. Decision tree

## Case A — W4-v2 is clean

If W4-v2:

```text
substantially improves metrics
removes the obvious grid/lattice
restores useful colors
looks visually good
```

then:

```text
W4-v2 becomes leading candidate
W8-v2 remains high-precision reference
```

Do not create mixed precision merely because the feature exists.

---

## Case B — W8-v2 clean, W4-v2 bad

This strongly indicates W4 precision remains the main image-quality problem.

Then use the W4 quantization report.

Create **one** mixed pack.

Possible first promoted classes:

```text
AdaLN / modulation
timestep matrices
final layer
```

and/or whatever classes actually score worst in the quantization report.

Example precision map:

```json
{
  "version": 1,
  "default": "w4",

  "overrides": [
    {
      "match": "model.diffusion_model.blocks.*.adaln_modulation_*",
      "storage": "w8"
    },

    {
      "match": "model.diffusion_model.t_embedder.*",
      "storage": "w8"
    },

    {
      "match": "model.diffusion_model.final_layer.*",
      "storage": "w8"
    }
  ]
}
```

Then run one additional full inference.

---

## Case C — W8 better but still imperfect

If W8 clearly improves:

```text
latent
RGB
pattern
color
```

but remains visibly imperfect, quantization remains a major contributor.

Only then consider:

```text
selective FP16 matrices
stronger W4 optimizer
full W8 as quality mode
```

Do not modify VAE first.

---

## Case D — W4 and W8 both remain around ~0.7 with same artifact pattern

Stop repacking.

This materially weakens the “4-bit quantization quality” hypothesis.

Next investigate:

```text
full eight-step same-weight orchestration
sampler state transitions
recurrent DiT state
latent layout
```

At that point a full same-W8 Python/Swift loop oracle becomes worth the effort.

---

## Case E — W8 crashes or looks much worse than W4

Treat this as a new runtime W8 implementation bug.

Debug:

```text
W8 matrix span
W8 row stride
W8 group indexing
W8 direct matvec
W8 loader dtype
```

before changing quantizer math.

---

# 63. What NOT to do before the first W4/W8 result

Do not implement:

```text
group32
group16
GPTQ
AWQ
activation calibration
Hessian optimization
FP16 matrix path
another sampler
another VAE
new seed
new prompt
new sigmas
new source checkpoint
```

The first experiment is intentionally:

```text
same source
same GROUP=64
same runtime graph
better W4
vs
W8
```

---

# 64. Long-term model storage policy

Going forward:

## Hugging Face

Use for:

```text
production/refinement .animapk model packs
quant reports worth retaining
model cards
model license notices
packing manifests
```

## GitHub repository

Use for:

```text
source code
packer
runtime
tests
workflows
small fixtures
documentation
hash manifests
```

## GitHub Actions artifacts

Use for:

```text
PNG comparisons
metrics
logs
small CI reports
short-lived diagnostic data
```

## GitHub Releases

The existing `model-assets-v1` can remain as historical/legacy distribution.

Do not immediately delete it.

Do not upload every experimental pack there.

---

# 65. Qwen/VAE migration to Hugging Face is optional and deferred

Do not spend this cycle moving the existing Qwen/VAE files merely for architectural purity.

They already exist and work.

The immediate comparison can use:

```text
new DiT from HF
existing Qwen/VAE source
```

After a winning DiT variant exists, a later cleanup task can decide whether all three production packs should live on Hugging Face.

That migration is low priority.

---

# 66. ModelStore changes are deferred until a winner exists

Do not immediately change the shipping app's `ModelManifest` to point at experimental HF URLs.

The refinement CI can inject/download the packs independently.

Only when one of:

```text
W4-v2
W8-v2
mixed-v2
```

is chosen should production model metadata change.

That prevents churn.

---

# 67. Suggested commit sequence

Use reviewable commits.

For example:

```text
docs: enter AnimaXS refinement phase

ci: reset selected simulator before pack-free tests

tools: preserve recovered v1 packer

tools: add bounded-memory ANIMAPK v2 writer

tools: add W4 optimized and W8 quantization

tools: add quant reports and ANIMAPK verifier

test: add packer row-boundary regression fixtures

runtime: generalize DiT matrices to W4 and W8

metal: add W8 fp32 direct matvec

test: add W8 matvec parity

ci: add HF-backed W4/W8 packing workflow

ci: add HF-backed W4/W8 full inference matrix

docs: record W4/W8 refinement results
```

Push useful checkpoints.

Do not leave all of this in one giant final commit.

---

# 68. New TODO.md

Create:

```markdown
# AnimaXS Refine TODO

## Repository / docs

- [ ] Fetch and verify current origin/main.
- [ ] Create refinement branch.
- [ ] Archive superseded operational runbooks/TODOs.
- [ ] Install new RUNBOOK.md.
- [ ] Refresh STATUS.md.
- [ ] Refresh NEXT_TASK_HANDOFF.md.
- [ ] Refresh TEST_MATRIX.md.
- [ ] Keep DECISIONS.md append-only.
- [ ] Add refinement-phase decision.

## D072

- [ ] Add selected-simulator erase before pack-free CI tests.
- [ ] Run normal CI.
- [ ] Confirm DiagnosticsTests deterministic.
- [ ] Record D072 remediation as a new decision.

## Hugging Face infrastructure

- [ ] Locate HF token under /root without exposing it.
- [ ] Add HF_TOKEN as GitHub repository secret.
- [ ] Verify secret presence without reading value.
- [ ] Determine HF account namespace.
- [ ] Create/prepare AnimaXS-DiT-W4 repo.
- [ ] Create/prepare AnimaXS-DiT-W8 repo.
- [ ] Include required model licenses/notices.

## Source provenance

- [ ] Resolve full HF revision for circlestone-labs/Anima.
- [ ] Download only anima-turbo-v1.0.safetensors.
- [ ] Verify source SHA c0b905...2174.
- [ ] Pin full HF revision in packing workflow.

## Legacy packer

- [ ] Preserve recovered pack_anima.py unchanged.
- [ ] Verify/archive its SHA.
- [ ] Never edit the archived copy.

## Packer v2

- [ ] Implement dry-plan.
- [ ] Implement bounded-memory/direct output writer.
- [ ] Keep ANMA v1.
- [ ] Keep 16-KiB blob alignment.
- [ ] Keep group size 64.
- [ ] Preserve per-row group reset.
- [ ] Preserve W4 nibble convention.
- [ ] Use stored FP16 scale/zero when choosing Q.
- [ ] Implement W4 deterministic MSE-range search.
- [ ] Implement W8 affine quantization.
- [ ] Implement per-matrix W4/W8 precision map.
- [ ] Keep rank<=1 FP16.
- [ ] Add finite-value validation.
- [ ] Add streaming quant statistics.
- [ ] Add provenance metadata.
- [ ] Add blob SHA256 metadata.
- [ ] Add independent verify_animapk.py.
- [ ] Add [2,68] W4 regression.
- [ ] Add [2,65] W8 regression.

## DiT runtime

- [ ] Add common W4/W8 matrix factory.
- [ ] Generalize DiT preparation matrices.
- [ ] Generalize block matrices.
- [ ] Generalize final-layer matrices.
- [ ] Add w8_matvec_f32 Metal kernel.
- [ ] Centralize direct-matvec dispatch.
- [ ] Confirm W4 behavior unchanged.
- [ ] Add focused W8 matvec parity test.

## Packing workflow

- [ ] Create pack-dit-v2.yml.
- [ ] Use ubuntu-latest.
- [ ] Use separate matrix jobs for W4 and W8.
- [ ] Download source directly from pinned HF revision.
- [ ] Verify source SHA before packing.
- [ ] Run dry-plan.
- [ ] Pack W4-v2.
- [ ] Pack W8-v2.
- [ ] Verify both packs.
- [ ] Upload W4 directly to HF.
- [ ] Upload W8 directly to HF.
- [ ] Upload only small reports as GitHub artifacts.
- [ ] Record HF revisions and output SHAs.

## Full inference

- [ ] Add W4-v2/W8-v2 matrix workflow.
- [ ] Download exact SHA-pinned DiT packs from HF.
- [ ] Keep Qwen unchanged.
- [ ] Keep VAE unchanged.
- [ ] Keep canonical prompt.
- [ ] Keep canonical seed/noise.
- [ ] Keep sampler sigmas.
- [ ] Run W4-v2 full inference.
- [ ] Run W8-v2 full inference.
- [ ] Capture generated/reference/comparison PNGs.
- [ ] Record latent metrics.
- [ ] Record RGB metrics.
- [ ] Record timings.
- [ ] Compare visible pattern/color quality.

## Decision

- [ ] Compare legacy W4 vs W4-v2 vs W8-v2.
- [ ] Select W4-v2 if clean enough.
- [ ] Otherwise select W8-v2 if clean.
- [ ] Otherwise make ONE mixed W4/W8 follow-up if evidence supports it.
- [ ] If W8 remains equally bad, stop repacking and investigate full-loop orchestration.

## Finish cycle

- [ ] Append final decisions.
- [ ] Update STATUS.md.
- [ ] Update TEST_MATRIX.md.
- [ ] Update NEXT_TASK_HANDOFF.md.
- [ ] Record exact GitHub run IDs.
- [ ] Record exact HF repo revisions.
- [ ] Record exact model SHA256s.
- [ ] Ensure working tree is clean.
- [ ] Push all useful work.
```

---

# 69. TEST_MATRIX.md should include

At minimum:

```text
TEST                          W4-v2      W8-v2
------------------------------------------------
packer tiny fixture            □           □
offline pack verification      □           □
Swift parser opens pack        □           □
matrix byte layout             □           □
finite scale/zero              □           □
W4/W8 decoder test             □           □
direct matvec parity           □           □
normal CI green                □           □
source SHA verified            □           □
HF upload verified             □           □
HF download SHA verified       □           □
canonical full inference       □           □
8 finite sampler steps         □           □
generated PNG                  □           □
latent metrics recorded        □           □
RGB metrics recorded           □           □
timings recorded               □           □
visual comparison              □           □
```

---

# 70. Required execution-agent final report

Do not return merely:

> “All tasks complete.”

Return this exact level of information:

```text
REPOSITORY
----------
origin/main at start:
refinement branch:
final HEAD:
working tree:

DOCUMENTATION
-------------
old runbooks archived:
new RUNBOOK:
new TODO:
STATUS:
TEST_MATRIX:
DECISIONS entries added:

D072
----
workflow change:
normal CI run:
project-consistency:
iphone-build:
simulator tests:
DiagnosticsTests:
unexpected failures:

HF INFRASTRUCTURE
-----------------
HF namespace:
HF token secret configured:
W4 HF repo:
W8 HF repo:
licenses/notices present:

SOURCE
------
HF source repo:
HF source revision:
source path:
source bytes:
source SHA256:
expected SHA matched:

PACKER V1
---------
archived path:
SHA256:

PACKER V2
---------
commit:
group:
W4 algorithm:
W8 algorithm:
mixed precision support:
bounded-memory confirmed:
peak RAM if measured:

W4-v2
-----
GitHub packing run:
packing job:
filename:
bytes:
SHA256:
quant report:
verification:
HF repo:
HF revision:
HF upload verified:

W8-v2
-----
GitHub packing run:
packing job:
filename:
bytes:
SHA256:
quant report:
verification:
HF repo:
HF revision:
HF upload verified:

FULL INFERENCE W4-v2
--------------------
GitHub run:
DiT SHA verified:
Qwen SHA:
VAE SHA:
latent cosine:
latent RMSE:
latent MAE:
latent maxAbs:
RGB cosine:
RGB RMSE:
RGB MAE:
RGB maxAbs:
Qwen time:
adapter time:
diffusion time:
VAE time:
total time:
image artifact:
visual description:

FULL INFERENCE W8-v2
--------------------
GitHub run:
DiT SHA verified:
Qwen SHA:
VAE SHA:
latent cosine:
latent RMSE:
latent MAE:
latent maxAbs:
RGB cosine:
RGB RMSE:
RGB MAE:
RGB maxAbs:
Qwen time:
adapter time:
diffusion time:
VAE time:
total time:
image artifact:
visual description:

COMPARISON
----------
legacy W4:
W4-v2:
W8-v2:

pattern:
color:
prompt adherence:
detail:
metrics:
speed:
size:

RECOMMENDATION
--------------
winning current representation:
reason:

mixed follow-up required:
yes/no

remaining numerical blocker:
remaining visual blocker:
remaining infrastructure blocker:

NEXT TASK:
```

---

# 71. Hard stop conditions

Stop and investigate immediately if:

```text
source SHA differs from c0b905...2174

HF source revision cannot be pinned

HF token appears in logs

HF token lacks write access

required licensing files are missing

W4-v2 malformed

W8-v2 malformed

scale/zero contains NaN/Inf

v2 pack cannot be opened by real Swift parser

legacy W4 numerical behavior changes merely from runtime generalization

W8 synthetic matvec fails

HF uploaded model SHA cannot be reconciled with packing manifest

full-inference workflow downloads a different pack revision

canonical prompt/seed changes

GitHub packing job approaches six-hour timeout without progress

runner disk becomes critically low
```

Do not lower validation standards to get around these failures.

---

# 72. Explicitly forbidden during this cycle

Do not:

```text
use Clore.ai

store source weights on VPS

build multi-GB packs on VPS

upload multi-GB packs as GitHub Actions artifacts

split W8 into chunks

reassemble model chunks

repack Qwen

repack VAE

change the VAE graph

change PNG gamma/colors to hide defects

change the canonical prompt

change the canonical seed

change initial noise

change sampler sigmas

change source checkpoint

implement group32 before W4/W8 results

implement GPTQ/AWQ before W4/W8 results

implement FP16 rank-2 execution before W8 results

create separate DiT W4/W8 inference graphs

lower quality thresholds and call bad output fixed

commit .animapk into the GitHub source repository

print HF_TOKEN

commit HF_TOKEN

use unpinned "latest" HF weights for accepted inference
```

---

# 73. Definition of done for this refinement cycle

This cycle is done when:

```text
✓ current origin/main was verified before work

✓ project entered refinement mode

✓ old operational TODO/runbook material archived

✓ DECISIONS preserved append-only

✓ D072 made deterministic

✓ normal CI green

✓ recovered v1 packer preserved

✓ packer v2 committed

✓ packer v2 genuinely bounded-memory

✓ v2 uses stored FP16 scale/zero for quantization

✓ W4 range optimization implemented

✓ W8 implemented

✓ per-matrix W4/W8 architecture implemented

✓ pack verifier implemented

✓ synthetic row-boundary regressions green

✓ GitHub HF_TOKEN secret configured securely

✓ source downloaded directly by Actions from Hugging Face

✓ source revision pinned

✓ source SHA verified

✓ W4 and W8 packed on separate GitHub Linux runners

✓ no source or new packs stored on VPS

✓ W4 uploaded directly to dedicated HF repo

✓ W8 uploaded directly to dedicated HF repo

✓ both HF revisions recorded

✓ both model SHA256 values recorded

✓ DiT runtime supports W4/W8 through one graph

✓ W8 direct FP32 matvec tested

✓ existing W4 path remains unchanged

✓ canonical W4-v2 full inference completed

✓ canonical W8-v2 full inference completed

✓ both generated PNGs captured

✓ both latent metrics recorded

✓ both RGB metrics recorded

✓ both timing results recorded

✓ actual images visually compared

✓ evidence-based W4/W8/mixed recommendation made
```

---

# 74. Core engineering philosophy

The project no longer needs to prove that all the pieces can connect.

They already connect.

The refinement cycle should therefore optimize for **large, informative experiments** rather than a sequence of tiny speculative fixes.

The first experiment is deliberately simple:

```text
                ORIGINAL ANIMA TURBO
                       │
              exact same source SHA
                       │
              ┌────────┴─────────┐
              │                  │
              ▼                  ▼
         improved W4            W8
              │                  │
              └────────┬─────────┘
                       │
                  same runtime
                       │
                same prompt/noise
                       │
                 same Qwen/VAE
                       │
              canonical inference
                       │
              ┌────────┴─────────┐
              ▼                  ▼
           W4 PNG              W8 PNG
```

Then look at the actual result.

If W8 fixes the image, we know where to focus.

If improved W4 fixes it, even better.

If neither fixes it, stop spending time on quantization and move upstream to the full recurrent sampler/DiT loop.

That is the shortest path toward making AnimaXS produce genuinely good images rather than merely passing its existing tests.
