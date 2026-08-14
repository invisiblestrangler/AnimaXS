# Hermes Execution Instructions â€” AnimaXS CUDA Grid Reproduction, Root-Cause Isolation, and Fix

**Date:** 2026-08-14  
**Repository:** `invisiblestrangler/AnimaXS`  
**Primary working branch:** `investigate/animapk-cuda-parity`  
**Primary objective:** reproduce the visible woven/etched/grid defect on the already-running Clore CUDA instance, eliminate it there if it reproduces, then port/validate the fix on Metal/iOS.  
**Progress reporting:** use the normal Hermes execution conversation/transcript for concise milestone updates.

---

# 0. Execution mandate

I want you to finish every task you reasonably can in one continuous run.

Do **not** stop after one failed hypothesis, one inconclusive test, one infrastructure problem, or one partial result. Fix ordinary issues and continue.

Do not ask the user to manually perform routine technical steps that you can perform yourself.

Stop only when one of these is true:

1. the grid-pattern root cause is identified, fixed, and validated as far as the available CUDA/macOS infrastructure permits; or
2. the Clore/CUDA investigation is genuinely exhausted and the remaining work is specifically Apple/Metal-only; or
3. there is a genuine external blocker that cannot be worked around.

If blocked, leave exact evidence, exact paths/URLs/SHAs, exact commands attempted, and the next executable action.

The current run should prioritize **visible image correctness and the 8-pixel carrier**, not abstract cosine improvements that leave the grid unchanged.

---

# 1. The critical correction to the previous investigation

The recent CUDAâ†”Metal parity work is useful, but it did **not** identify the root cause of the visible grid pattern.

It established these important facts:

- real-graph CUDA final-latent cosine vs golden:
  - official BF16: ~`0.8110`
  - source FP16: ~`0.8130`
  - FP16-all `.animapk`: ~`0.8130`
  - W8 `.animapk`: ~`0.8090`
  - W4 `.animapk`: ~`0.6596`
- decoded `.animapk` execution and streaming `.animapk` execution were bit-identical on CUDA (`maxAbs 0.0`)
- pack payload interpretation was bit-faithful
- same-pack Metal:
  - FP16-all ~`0.8123`
  - W8 ~`0.8098`
  - W4 ~`0.6604`
- direct step-0 CUDAâ†”Metal block parity was approximately `0.99998â€“1.00000` cosine for every tested pack

That strongly clears a broad Metal-vs-CUDA DiT backend mismatch.

**However, the decisive missing experiment is image-space reproduction on CUDA.**

The current `scripts/animapk_cuda/ladder_real.py` work primarily established latent/model parity. It did not establish:

> â€œThe CUDA reproduction creates the same visible 8-pixel woven/grid artifact seen in the Metal full-image output.â€

Therefore do **not** write â€œgrid root cause solvedâ€ merely because CUDA and Metal latent trajectories agree.

Also treat this earlier claim as too strong for the grid question:

> â€œThe residual ~0.81 ceiling is a genuine numeric-accumulation/quantization effect.â€

That may describe latent similarity behavior, but it does not prove the origin of the visible grid.

The known defective Metal full-image carrier baseline is approximately:

```text
generated exact-8px carrier total: ~0.01345
reference exact-8px carrier total: ~0.00005484
ratio: ~245x
```

The true `bf16_compute`/golden-Qwen experiments changed ordinary RGB/latent similarity while leaving the grid carrier around `0.0132â€“0.0140`.

So:

> **A candidate that raises cosine but leaves the grid carrier essentially unchanged is rejected as the grid fix.**

---

# 2. First principle for this run

The strategy is now:

```text
1. Make the visible failure reproducible on CUDA/Clore.
2. Measure it automatically.
3. Inspect it visually.
4. Find the smallest CUDA-side change that removes it.
5. Prove the causal change.
6. Only then port the corresponding fix to Metal/iOS.
```

Clore is valuable precisely because it lets us iterate much faster than repeated GitHub macOS initialization.

If the grid exists on CUDA, keep the investigation on Clore until it is solved or until a specific Apple-only difference is proven.

Do not jump back to macOS simply because the production app is Metal.

---

# 3. Repository and branch rules

The branch has been updated recently. **Do not recreate it from `origin/main` and accidentally erase the pushed investigation work.**

At session start:

```bash
cd /path/to/AnimaXS

git fetch --all --prune
git status --short
git branch -vv
git rev-parse HEAD
git rev-parse origin/main
git rev-parse origin/investigate/animapk-cuda-parity
git log -n 20 --oneline --decorate --graph --all
```

Continue from:

```text
investigate/animapk-cuda-parity
```

Prefer:

```bash
git switch investigate/animapk-cuda-parity
git pull --ff-only origin investigate/animapk-cuda-parity
```

Only reset a checkout after proving there is no useful uncommitted work.

Do not dirty `main`.

Before coding, inspect the actual current branch rather than assuming this handoff contains the latest commit.

At minimum re-read:

```text
HERMES_ANIMAPK_CUDA_PARITY.md
DECISIONS.md
TODO.md
STATUS.md                    (if present)
HERMES_SESSION.md            (if present)
scripts/measure_grid_carrier.py
scripts/vae_decoder_oracle.py
scripts/publish_hf_pack.py
scripts/animapk_cuda/
.github/workflows/
AnimaXSTests/FullInferenceTests.swift
relevant Metal runtime/shader files
```

The pushed CUDA scripts are valuable; reuse them rather than replacing the investigation with a new unrelated harness.

---

# 4. Save and repeatedly re-read this instruction

Save this file at repo root as:

```text
HERMES_GRID_ROOT_CAUSE.md
```

Re-read the **complete** file:

- at session start
- after any context compaction/reset
- after the repo audit
- after connecting to Clore
- after the first CUDA images exist
- after every result that changes the leading hypothesis
- before starting a broad/full ComfyUI reproduction
- before an expensive macOS run
- before declaring the grid fixed
- immediately before shutting down Clore

Context compaction is not an excuse to forget the experiment design.

After any compaction, first read:

```text
1. HERMES_GRID_ROOT_CAUSE.md
2. HERMES_SESSION.md
3. latest DECISIONS.md entries
4. current TODO.md
5. latest experiment manifest/result summary
```

Then continue.

---

# 5. Maintain a compact live session state

Maintain:

```text
HERMES_SESSION.md
```

Recommended structure:

```markdown
# Current state

Branch:
HEAD:
origin/main:
origin/investigate/animapk-cuda-parity:
Working tree:
Last known green normal CI:

Clore status:
Clore host:
GPU:
Clore free disk:
Clore environment:
HF artifact repo:
HF latest uploaded experiment:
HF latest verified commit/revision:

Current hypothesis:
Grid reproduced on CUDA: yes/no/unknown
Clean CUDA control established: yes/no
Current best carrier:
Reference/control carrier:
Current best visual result:
Current root-cause localization:

Known-good controls:
Rejected hypotheses:
Open hypotheses:
Blockers:

Last experiment:
Last result:
PROMOTE / REJECT / INCONCLUSIVE:

Next 3 actions:

Important source revisions:
Important pack SHAs:
Important fixture hashes:
Important GitHub run IDs:
Important HF paths:
Important Clore paths:
```

Update this after every meaningful milestone and before any context compaction.

---

# 6. Frequent progress visibility

The user still wants frequent progress visibility.

Use the normal Hermes execution conversation/transcript for concise progress updates.

Provide a concise update after major milestones such as:

- repo audit
- Clore connection/environment audit
- HF storage audit
- current Clore files safely uploaded to HF
- first clean decoded control
- first CUDA BF16 image
- first CUDA W8 image
- first measured CUDA grid carrier
- first Hermes visual inspection
- first root-cause split
- first strong fix candidate
- final CUDA validation
- first Metal validation
- final shutdown

Useful update content:

```text
branch + HEAD
experiment id
variant
pack/source SHA
image carrier
latent/RGB metric when relevant
vision conclusion
current hypothesis
next action
```

Do not flood the user with low-level command output.

---

# 7. Compute policy

## 7.1 Cheap VPS: orchestration only

The VPS has very limited RAM/disk and must **not** become a model/artifact store.

Allowed on VPS:

- git operations
- editing source/docs
- small logs/JSON/markdown
- GitHub workflow dispatch
- Clore orchestration
- Hugging Face metadata/API calls
- small helper scripts
- status reporting

Do **not** use VPS for:

- multi-GB model downloads
- packing large weights
- CUDA inference
- VAE inference
- storing `.animapk` packs
- storing model checkpoints
- storing large fixture bundles
- downloading entire macOS artifacts merely to relay them somewhere else
- long-term result storage

If a large file must move between services, transfer it **directly**:

```text
Clore -> Hugging Face
GitHub Actions -> Hugging Face
Hugging Face -> Clore
```

Do not route it through the VPS.

Periodically check:

```bash
df -h
find /tmp /root -xdev -type f -size +100M -printf '%s %p\n' 2>/dev/null | sort -n
```

Delete stale large VPS files only after proving that their durable copy exists on Hugging Face or can be deterministically regenerated.

Never delete credentials, SSH keys, the working repository, or the only copy of useful source changes.

## 7.2 Clore NVIDIA GPU

Use the already-running Clore instance for:

- source BF16/FP16 controls
- direct `.animapk` CUDA execution
- rapid 8-step tests
- VAE decoding
- image generation
- carrier measurement
- true ComfyUI/source reproduction
- tensor capture
- image-space root-cause experiments
- fix iteration

The Clore instance is the main rapid-testing environment for this phase.

## 7.3 GitHub Linux

Use only when useful for:

- cheap packaging
- deterministic verification
- static/unit checks
- CI reproducibility

Do not use it merely to duplicate fast Clore tests.

## 7.4 GitHub macOS

Use macOS only when a candidate has earned Apple validation or when CUDA is proven clean while Metal remains bad.

Prefer manual-dispatch workflows.

Avoid repeated dependency/model reinitialization.

Reuse Hugging Face artifacts directly in the workflow.

---

# 8. Clore instance lifecycle

The user already has a Clore instance running.

Therefore:

```text
DO NOT launch another Clore instance.
DO NOT rent a second GPU.
DO NOT terminate the current instance during useful CUDA iteration.
```

Inspect and reuse the existing environment before reinstalling anything.

Record:

```bash
hostname
date -Is
nvidia-smi
df -h
free -h
pwd
python --version
```

And:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name())
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

Because previous Clore SSH sessions were flaky and detached children could die, prefer a persistent session such as `tmux` for long jobs.

Use checkpoint/resume for multi-variant work.

### When to close Clore

Close/terminate the instance when **either**:

1. the grid issue is solved and the CUDA fix/evidence is safely persisted; or
2. the remaining unresolved work has been proven Apple/Metal-specific and no further CUDA experiment is useful.

Do not keep paid GPU capacity alive just because macOS validation is still running.

Before termination:

```text
- all useful source changes committed
- branch pushed
- all unique generated files uploaded to Hugging Face
- HF uploads verified
- SHA256 manifest uploaded
- decisions/TODO/session updated and pushed
- no sole useful copy remains on Clore
- no remaining planned CUDA experiment requires the live GPU
```

Then terminate the existing instance and record that fact in the final report.

---

# 9. Hugging Face is the durable artifact store

This is mandatory.

The VPS is **not** the durable artifact store.

All durable generated investigation files must be transferred to Hugging Face, including the currently existing useful Clore outputs before substantial new work begins.

Source code belongs in Git/GitHub; large binary evidence belongs in Hugging Face.

## 9.1 Upload existing unique Clore work first

Audit the existing Clore paths from the previous handoff, especially:

```text
/workspace/out/ladder/
/workspace/out/ladder_real/
/workspace/out/fixture/
/workspace/packs/
/workspace/fixtures/
```

Also search for other unique result directories:

```bash
find /workspace -maxdepth 4 -type f \
  \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.png' \
     -o -name '*.npz' -o -name '*.f32' -o -name '*.animapk' -o -name '*.log' \) \
  -printf '%s %p\n' | sort -n
```

Do not blindly upload caches, Python environments, PyTorch wheels, git object stores, or reproducible upstream package caches.

The phrase â€œall files to Hugging Faceâ€ means **all durable/unique investigation artifacts and generated model assets**, not venv/cache garbage.

## 9.2 HF destination

First inspect repo docs/history and existing `.hf-publish.json`/manifests to discover whether an AnimaXS artifact repository is already established.

If an established artifact repo exists, reuse it.

If none exists, create one under the authenticated Hugging Face account, for example:

```text
<owner>/AnimaXS-investigation-artifacts
```

Use private visibility by default unless existing project precedent explicitly uses a public evidence repository and there is no sensitive material.

For already-established model-pack repositories, preserve their existing visibility/licensing conventions.

Record the exact HF repo id in:

```text
HERMES_SESSION.md
DECISIONS.md
each experiment manifest
```

## 9.3 Credentials

The previous environment used Hugging Face credentials such as:

```text
/root/HUGGINGFACE_TOKEN
```

and the repo helper `scripts/publish_hf_pack.py` expects:

```text
HF_TOKEN
```

Normalize securely without printing the token, for example:

```bash
export HF_TOKEN="$(tr -d '\r\n' < /root/HUGGINGFACE_TOKEN)"
```

If that exact path no longer exists, inspect the environment securely for the already-provisioned credential.

Never echo secrets.

Never commit tokens.

Never put tokens in CI artifacts or logs.

## 9.4 Recommended artifact layout

Use a stable experiment hierarchy such as:

```text
experiments/
  2026-08-14_grid-baseline/
    manifest.json
    SHA256SUMS
    environment/
      clore_environment.txt
    inputs/
      fixture_manifest.json
    latents/
      golden_final_latent.f32
      cuda_bf16_final_latent.f32
      cuda_fp16all_final_latent.f32
      cuda_w8_final_latent.f32
      metal_bad_final_latent.f32
    rgb/
      golden_via_cuda_vae.png
      cuda_bf16.png
      cuda_fp16all.png
      cuda_w8.png
      metal_latent_via_cuda_vae.png
      comparison.png
      crops.png
      fft_comparison.png
    metrics/
      grid_carrier_*.json
      latent_metrics.json
      rgb_metrics.json
    vision/
      hermes_vision_review.md
      nanogpt_vision_review.json
    logs/
      *.log
    scripts/
      exact_experiment_script_hashes.json
```

For subsequent experiments:

```text
experiments/<timestamp>_<short-purpose>/
```

Do not overwrite old evidence.

## 9.5 Verify uploads

Every upload bundle must include hashes:

```bash
find <bundle> -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
```

After upload:

- list the remote files
- verify remote metadata/size where possible
- re-download at least small manifests/metrics and compare hash
- for multi-GB packs, verify the recorded HF commit/revision and size/hash metadata
- record the HF commit/revision

Do not delete a sole local copy until verification succeeds.

---

# 10. Reset the active TODO to this grid investigation

The current `TODO.md` contains historical/final-quality items that can confuse a fresh execution agent.

Before major new work:

1. read all current TODO/status/decision files
2. preserve historical decisions in `DECISIONS.md`
3. hard-reset the **active** checklist around this new CUDA-grid mission
4. do not erase useful history from git

Recommended new active phases:

```text
Phase G0 â€” repo/Clore/HF audit
Phase G1 â€” image-space CUDA reproduction
Phase G2 â€” grid origin decision tree
Phase G3 â€” CUDA fix loop
Phase G4 â€” Metal port/validation
Phase G5 â€” production W8 acceptance
Phase G6 â€” cleanup, persistence, shutdown
```

Mark stale instructions as superseded rather than leaving two competing active plans.

---

# 11. Preserve and extend DECISIONS.md

Do not rewrite old history.

Append corrections/supersessions.

One important new decision should explicitly state:

> CUDAâ†”Metal latent/block parity does not by itself establish the root cause of the visible 8-pixel grid. The grid must be reproduced and measured in CUDA image space before concluding that the visible artifact is backend-independent.

Every significant experiment should end with:

```text
PROMOTE
REJECT
INCONCLUSIVE
```

Each decision entry should record:

```text
date
branch + commit
experiment id
backend
source revision
pack filename + SHA
fixture hashes
exact command/script
latent metrics
RGB metrics
8px carrier
vision result
what this proves
what this does NOT prove
next action
HF artifact path + revision
```

Do not keep zombie hypotheses alive after a discriminating result rejects them.

---

# 12. Immediate highest-priority experiment: make the grid visible on CUDA

Do this **before** the adapter ladder, W4 optimization, more attention tuning, or more broad Metal tests.

The first experiment should produce a small controlled image matrix using one fixed VAE decoding path.

## 12.1 Fixed canonical inputs

Use the exact canonical case already used by the project:

```text
case1 / seed 1337
canonical prompt
canonical initial noise/latent
canonical conditioning
canonical sigma schedule from the golden fixture
same model config
same output dimensions
```

Do not use â€œlatest ComfyUI defaults.â€

Do not silently change:

- scheduler
- sigma schedule
- seed
- prompt
- latent initialization
- context
- VAE
- output scaling
- model revision

Record hashes for every canonical binary input.

## 12.2 Save real final latent bytes

If `ladder_real.py` currently records only metrics/checkpoints and not the final latent bytes needed for decoding, modify it minimally.

Add a safe option such as:

```text
--save-final-latents DIR
--variants A,C,D
```

or create a focused script reusing its existing functions.

Do not rewrite the model runtime just to save tensors.

Persist each final latent as exact contiguous float32, plus shape/dtype/hash in manifest.

Initial variants:

```text
A = official BF16 source through real pinned CUDA graph
C = FP16-all .animapk through direct CUDA runtime
D = W8 .animapk through direct CUDA runtime
```

Do not run W4 initially unless it is nearly free to include. W4 is already known to be a large quality cliff and is not the immediate production target.

## 12.3 Build the five-lane image matrix

Use one identical Python/CUDA VAE path for all latents.

### Lane G0 â€” clean decoder control

```text
canonical golden final latent
-> same Python/CUDA VAE
-> PNG
```

This must be clean.

If this lane has the same severe grid, stop blaming the DiT and debug the decoder/PNG conversion first.

### Lane G1 â€” official source BF16 CUDA

```text
official BF16 source
-> real pinned CUDA DiT graph
-> exact canonical 8 steps
-> save final latent
-> same Python/CUDA VAE
-> PNG
```

This is the most important new image.

### Lane G2 â€” FP16-all `.animapk` CUDA

```text
FP16-all pack
-> direct .animapk CUDA runtime
-> exact canonical 8 steps
-> save final latent
-> same VAE
-> PNG
```

### Lane G3 â€” W8 `.animapk` CUDA

```text
W8 pack
-> direct .animapk CUDA runtime
-> exact canonical 8 steps
-> save final latent
-> same VAE
-> PNG
```

### Lane G4 â€” known bad Metal final latent, decoded on CUDA

Obtain a known bad Metal final latent from the existing macOS evidence, Hugging Face, or a narrowly targeted GitHub macOS run.

Do **not** use the VPS as a relay for a huge artifact.

Then:

```text
known bad Metal final latent
-> exact same Python/CUDA VAE
-> PNG
```

This lane is extremely valuable because it holds the decoder fixed and asks whether the bad appearance is already encoded in the Metal final latent.

If an existing bad Metal final latent is already available, use it.

If not, do not block G0â€“G3 waiting for it.

---

# 13. Use the existing VAE oracle rather than inventing a new decoder

The branch already contains:

```text
scripts/vae_decoder_oracle.py
```

It is designed to:

- read the real fp16 VAE `.animapk`
- execute the pinned T=1 Wan decoder graph
- accept a supplied final latent
- compare against canonical `decoded_rgb`
- emit same-pack decoded RGB as float32

Current argument shape includes approximately:

```text
--pack
--latent
--golden
--scale-factor
--emit-lane-a
--json
```

Use/reuse this implementation.

If it is too slow because it is NumPy/CPU-oriented, first prove the control, then adapt the already-validated decoder semantics to a CUDA implementation.

Do not change VAE semantics while comparing DiT variants.

The same decoder, scale factor, color/output mapping, and PNG conversion must be used for G0â€“G4.

---

# 14. PNG conversion must itself be controlled

Do not create a new arbitrary normalization that can manufacture or hide texture.

Find the canonical project conversion from decoded RGB tensor to output PNG and reuse it.

Record:

```text
input tensor range
clamp rule
[-1,1] / [0,1] mapping if applicable
rounding
channel order
image dimensions
PNG mode
```

Create one helper if needed:

```text
scripts/animapk_cuda/rgb_f32_to_png.py
```

and unit-test it against the known canonical reference conversion.

A grid conclusion is invalid if different lanes use different image normalization.

---

# 15. Measure every generated image with the existing 8px carrier metric

Use:

```text
scripts/measure_grid_carrier.py
```

Example shape:

```bash
python scripts/measure_grid_carrier.py \
  generated.png \
  reference.png \
  --json grid-carrier.json
```

For every lane record:

```text
horizontal
vertical
diagonal
total exact-8px carrier
ratio vs reference
```

Also record ordinary image metrics:

```text
RGB cosine
RMSE
PSNR if useful
```

But the grid carrier is the primary automated defect metric.

Do not declare success based only on cosine.

---

# 16. Generate visual diagnostics for Hermes

For each meaningful image experiment generate:

```text
reference.png
generated.png
side_by_side.png
difference.png
difference_high_gain.png
center_crop.png
texture_crop.png
fft_magnitude.png
fft_comparison.png
```

Prefer one contact sheet containing labeled panels:

```text
REFERENCE
CUDA BF16
CUDA FP16-ALL
CUDA W8
METAL LATENT -> CUDA VAE
ABS DIFF / HIGH GAIN
FFT
```

Keep the original unmodified images too.

The contact sheet is for visual reasoning, not as a substitute for metrics.

---

# 17. Hermes must use vision on the generated images

Use Hermes's native vision capability first.

Inspect:

- the clean reference
- CUDA BF16
- CUDA FP16-all
- CUDA W8
- Metal-latent-through-CUDA-VAE when available
- comparison/contact sheet
- high-gain difference
- FFT visualization

The visual review should explicitly answer:

```text
1. Is the regular woven/etched/grid artifact visibly present?
2. Is it approximately 8-pixel periodic?
3. Is it horizontal, vertical, checker-like, or cross-hatched?
4. Which lanes share it?
5. Which lane first looks clean?
6. Did a candidate remove the periodic texture while preserving natural detail?
7. Is any apparent improvement merely blur/smoothing?
8. Does the artifact line up with semantic edges, latent-cell boundaries, or the entire image uniformly?
```

Save the review to:

```text
vision/hermes_vision_review.md
```

Include image filenames and experiment id.

Vision is required for practical confirmation, but machine metrics remain required.

---

# 18. If Hermes vision fails: manually POST the image to Nano-GPT Xiaomi MiMo v2.5

If native Hermes vision is unavailable, errors, cannot load the image, or produces obviously unusable output, perform a manual HTTP POST.

Use the user-requested model:

```text
xiaomi/mimo-v2.5
```

Nano-GPT supports OpenAI-compatible image input through:

```text
POST https://nano-gpt.com/api/v1/chat/completions
```

with structured `image_url` parts, including base64 `data:image/png;base64,...`.

## 18.1 Prefer one labeled comparison sheet

To reduce ambiguity, first send a single labeled `comparison.png` containing the relevant images/crops.

If needed, send multiple image parts in one request.

## 18.2 API key handling

Use an already-provisioned secret such as:

```text
NANOGPT_API_KEY
NANO_GPT_API_KEY
```

or an existing protected key file.

Do not print the key.

Do not commit it.

If a key file contains a newline, strip it only in-memory.

## 18.3 Robust Python POST example

Create a temporary helper outside git or a safe script that contains **no secret literal**:

```python
import base64
import json
import mimetypes
import os
from pathlib import Path

import requests

image_path = Path("comparison.png")

key = (
    os.environ.get("NANOGPT_API_KEY")
    or os.environ.get("NANO_GPT_API_KEY")
)

if not key:
    raise SystemExit("Nano-GPT API key is not available")

mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

prompt = """You are reviewing an image-generation regression.
The panels are labeled. Inspect for a regular woven/etched/checker/grid artifact,
especially an approximately 8-pixel periodic carrier. Compare the reference
against each generated lane. State:
1) which lanes visibly contain the grid,
2) orientation/periodicity,
3) severity ranking,
4) whether any candidate truly removes the grid or merely blurs detail,
5) any clue about whether the artifact looks latent-cell/VAE-upsample aligned.
Be concrete and reference the panel labels."""

payload = {
    "model": "xiaomi/mimo-v2.5",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}"
                    },
                },
            ],
        }
    ],
    "stream": False,
}

r = requests.post(
    "https://nano-gpt.com/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=180,
)

r.raise_for_status()
result = r.json()
print(json.dumps(result, indent=2))
```

Save the raw response and an extracted markdown summary:

```text
vision/nanogpt_vision_review.json
vision/nanogpt_vision_review.md
```

If one image request fails, prefer base64 PNG/JPEG/WebP and verify the file is valid/non-empty.

Do not send secrets or unrelated private data to the vision model.

---

# 19. First decision tree after G0â€“G4

This is the most important branching logic in the run.

## Case 1 â€” G0 golden latent decoded by CUDA VAE has the grid

Then the DiT is not required to produce the problem.

Focus on:

```text
VAE implementation
VAE pack interpretation
latent normalization
RGB mapping
PNG conversion
reference mismatch
```

Do not continue DiT experiments until the same decoder can cleanly reproduce the canonical reference.

## Case 2 â€” G0 is clean, but G1 official BF16 CUDA has the grid

This is the most interesting likely outcome.

It means the grid can be reproduced completely off Apple hardware.

Then:

```text
DO NOT go back to Metal yet.
DO NOT optimize W4.
DO NOT spend time on adapter precision unless evidence points there.
```

Use Clore as the main root-cause lab.

The next task is to create:

```text
clean source/ComfyUI reproduction
vs
grid-producing current CUDA reproduction
```

on the same NVIDIA machine.

Then substitute one component at a time until the grid appears/disappears.

## Case 3 â€” G1 BF16 is clean, G2 FP16-all grids

Focus on:

```text
source graph vs direct pack graph
FP16 compute/cast semantics
pack-backed execution graph
runtime/layout differences
```

Do not blame W8 yet.

## Case 4 â€” G1/G2 clean, G3 W8 grids

Quantization/runtime becomes the leading cause.

Then investigate W8:

```text
which tensor/layer classes introduce the spatial pathology
mixed FP16/W8
higher precision Q/K or output/final layers
group quantization behavior
activation sensitivity
```

Use CUDA to fix W8 before touching Metal.

## Case 5 â€” G1/G2/G3 are visually clean, but G4 Metal latent -> CUDA VAE grids

The artifact is encoded in the Metal final latent.

That would mean broad cosine parity hid a spatially structured Metal trajectory error.

Focus Metal investigation on the earliest stage where the structured spatial error enters.

## Case 6 â€” G4 Metal latent -> CUDA VAE is clean, but Metal's own PNG is bad

Then revisit:

```text
Metal VAE
output normalization
RGB conversion
image write/display path
```

despite earlier broad VAE parity.

## Case 7 â€” G1/G2/G3 all show approximately the same grid

The defect is not primarily quantization.

Focus on the shared model-forward/golden-environment difference.

This is where a true ComfyUI/source reproduction on Clore becomes mandatory.

---

# 20. Establish a true clean CUDA/ComfyUI control if source CUDA grids

Do **not** install/use whatever â€œlatest ComfyUIâ€ happens to be available.

The golden environment has known version-sensitive behavior.

The repo already established that scheduler formulas changed across ComfyUI versions while the Swift sampler matches the recorded golden sigma schedule.

Therefore:

1. inspect golden fixture metadata and repo history
2. identify the exact or closest known golden-generation source revision/environment
3. use the exact canonical recorded sigma schedule from the fixture
4. use the official model revision/checkpoint already pinned by the project
5. use the exact prompt/seed/noise/conditioning
6. run the actual upstream graph, not a hand transcription
7. decode with the same controlled VAE
8. measure carrier and visually inspect

Important source evidence from the previous run includes:

```text
official Anima repo:
circlestone-labs/Anima

official source revision recorded previously:
f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b

official Turbo safetensors SHA-256 recorded previously:
c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174

pinned MiniTrainDIT / Comfy source used by the CUDA harness:
cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44
```

Verify these against the current repo before relying on them.

A source hash mismatch invalidates the experiment.

---

# 21. If true ComfyUI is clean while the current CUDA harness grids

This is the ideal debugging situation.

You now have both:

```text
CLEAN CUDA
BAD CUDA
```

on the same GPU.

Do not change many things at once.

Build a controlled substitution ladder.

High-priority differences to test include:

```text
actual Comfy operations vs local stubs
actual dtype/autocast boundaries
SDPA backend selection
BF16 vs FP16 activation/storage boundaries
Q/K normalization
RoPE implementation
attention mask semantics
attention scaling
residual/add precision
AdaLN/modulation precision
MLP precision/casts
final norm/projection
unpatchify/output rearrangement
conditioning adapter path if the clean and bad pipelines differ there
```

Do not resurrect the old manually transcribed FP32-attention oracle as ground truth. It was already shown to diverge from actual upstream SDPA behavior.

Make each substitution answer a causal question.

Prefer binary-search-like isolation over combinatorial testing.

---

# 22. The visible grid must stay in the loop during CUDA debugging

Because Clore is fast enough for repeated testing, do not return to a purely step-0/cosine investigation for hours without checking the actual defect.

Use two levels:

## Cheap gate

For every candidate:

```text
unit correctness
step-0 metrics
obvious tensor sanity
```

## Image gate

When a candidate changes a plausible shared root-cause component:

```text
8-step canonical latent
same VAE
PNG
8px carrier
Hermes vision
```

A full-image test is justified here because the visible artifact is the target and Clore is the fast loop.

Do not run huge variant matrices. Test the highest-information candidate first.

---

# 23. Add latent-space spatial diagnostics

The image artifact has a strong regular period, so ordinary global cosine can hide the relevant structure.

For the golden and bad final latents produce:

```text
per-channel error maps
mean absolute error map
signed mean error map
latent FFT magnitude
checker/parity statistics
horizontal/vertical neighbor correlation
row/column periodicity
```

Save:

```text
latent_spatial_error.png
latent_fft.png
latent_periodicity.json
```

Compare:

```text
golden vs CUDA BF16
golden vs CUDA FP16-all
golden vs CUDA W8
golden vs bad Metal latent
```

If the image-space grid is already predictable from a structured final-latent error, this can dramatically narrow the search.

Do not assume the VAE is creating the pattern merely because the visible period happens to line up with an upsampling scale; prove it.

---

# 24. Useful decoder-sensitivity experiment

If a bad latent produces the grid but the golden latent is clean through the same VAE, test whether the VAE is amplifying generic latent deviation or a specific structured deviation.

Let:

```text
delta = bad_latent - golden_latent
```

Decode controlled interpolations:

```text
golden + 0.25 * delta
golden + 0.50 * delta
golden + 0.75 * delta
golden + 1.00 * delta
```

For each:

```text
PNG
carrier
RGB metric
vision review
```

Also compare with a norm-matched random perturbation if cheap.

Interpretation:

- carrier growing specifically with the real structured delta suggests a structured latent error
- any equal-norm random perturbation producing the same grid suggests decoder sensitivity/amplification
- a sharp threshold may explain why broad latent cosine appears reasonably high while the image looks pathological

Do not spend hours here unless the initial G0â€“G4 result makes this diagnostic relevant.

---

# 25. Prior hypotheses already strongly weakened or cleared

Do not blindly redo these unless new image-space evidence points back to them:

```text
basic input patchify order
zero padding-mask channel
x-embedder shape/order
timestep embedding
gross W8 decoder error
decoded-vs-streaming lifecycle semantics
broad CUDA-vs-Metal block mismatch
Qwen conditioning as the primary grid cause
sampler/scheduler/Euler math
FP32 attention as the primary fix
broad BF16-emulated arithmetic as sufficient fix
source-FP16 weights as sufficient fix
```

Known evidence:

- FP32 attention improved local error substantially but barely moved final W8 latent quality and did not solve the grid.
- golden Qwen conditioning did not remove the grid.
- true BF16 compute changed ordinary quality but left the carrier essentially unchanged.
- source-FP16/full-DiT higher precision improved quality while the visible grid persisted.
- scheduler/sigma/Euler/initial-noise math was audited against the golden fixture.
- `.animapk` decoded and streaming execution matched on CUDA.
- CUDA and Metal matched extremely closely at tested DiT block boundaries.

These findings are valuable constraints, not proof of the final cause.

---

# 26. W8 is the primary production quantization target; park W4

Do not let W4 consume the run before the grid root cause is understood.

Current controlled evidence:

```text
FP16-all final latent cosine ~0.813
W8 final latent cosine      ~0.809
W4 final latent cosine      ~0.660
```

W8 is close to FP16 in the controlled latent ladder.

W4 is a genuine quality cliff.

Therefore:

```text
solve grid with source/FP16/W8 first
-> validate W8
-> only then revisit W4/mixed precision if useful
```

If W8 itself is proven to cause the grid in image space, then it becomes the immediate target.

Otherwise do not optimize W4 yet.

---

# 27. Do not prioritize the adapter experiment before image-space localization

An adapter precision ladder may still be useful later for the separate production-path quality gap.

But the immediate mission is the visible grid.

Do not spend Clore time first on:

```text
LLM adapter W8-vs-FP16 fidelity
0.7522-vs-0.81 bookkeeping
W4 scheme design
broad attention experiments
```

unless the G0â€“G4 image result specifically points there.

Grid localization comes first.

---

# 28. Fix-loop discipline on Clore

Once a clean-vs-bad CUDA difference is identified:

1. change one causal component
2. run cheap correctness gate
3. run canonical 8-step image
4. measure carrier
5. use Hermes vision
6. classify the result
7. commit useful code
8. upload evidence to HF
9. update `DECISIONS.md`
10. continue immediately

Use:

```text
PROMOTE â€” carrier collapses substantially toward the clean CUDA/reference control without destroying image detail.

REJECT â€” cosine changes but carrier is materially unchanged; or grid is merely hidden by blur.

INCONCLUSIVE â€” the test itself is invalid, mismatched, or lacks a clean control.
```

A â€œfixâ€ that just smooths the image is not a real fix.

---

# 29. What counts as a strong grid improvement

Do not hardcode the historical carrier ratio as a universal threshold because normalization/reference details can vary slightly.

Use the **clean control generated by the same exact measurement path** as the target.

A strong candidate should:

```text
reduce the exact-8px carrier by at least an order of magnitude from the bad baseline,
move clearly toward the clean control,
look visually free of the etched/woven pattern,
preserve natural detail,
avoid new checker/stripe frequencies,
and not cause a major semantic/image-quality regression.
```

For final acceptance, aim for the carrier to be close to the clean control rather than merely â€œbetter than 245x.â€

Always report absolute carrier values and ratios.

---

# 30. Only after a CUDA fix: port the corresponding change to Metal

Do not port speculative fixes.

Once a CUDA-side change removes the grid:

1. record exact before/after CUDA evidence
2. understand the semantic reason the fix works
3. identify the analogous Metal/Swift implementation
4. make the smallest equivalent Metal change
5. run focused macOS validation
6. only then run broader normal CI

Use the same:

```text
canonical prompt
seed
noise
sigmas
conditioning
pack SHA
VAE
image conversion
carrier metric
```

Prefer fetching needed packs/fixtures directly from Hugging Face in GitHub Actions.

Do not use the VPS as an artifact relay.

---

# 31. macOS validation sequence

When a candidate earns Apple validation:

```text
1. build/unit tests
2. focused one-step parity if relevant
3. canonical 8-step latent
4. full image
5. carrier metric
6. visual review
7. normal CI
8. generic iPhone build
9. simulator tests
```

A real iPhone is not assumed available in this phase.

Do not claim physical iPhone performance/quality validation unless an actual device run occurred.

If macOS reproduces the CUDA fix, the investigation can move toward production integration.

If CUDA is fixed but Metal remains bad, immediately compare the earliest structured difference rather than starting a new broad investigation.

---

# 32. Image review on macOS results

Use the same vision policy:

1. Hermes native vision first
2. if unavailable/failing, Nano-GPT `xiaomi/mimo-v2.5`
3. machine carrier metric always required

Store macOS comparison images and vision results in the same HF experiment hierarchy.


---

# 33. Existing scripts to preserve and reuse

The current branch already contains a significant CUDA parity toolkit.

At minimum preserve/reuse:

```text
scripts/animapk_cuda/reader.py
scripts/animapk_cuda/quant.py
scripts/animapk_cuda/runtime.py
scripts/animapk_cuda/upstream.py
scripts/animapk_cuda/ladder.py
scripts/animapk_cuda/ladder_real.py
scripts/animapk_cuda/real_step0_caps.py
scripts/animapk_cuda/backend_compare.py
scripts/animapk_cuda/compare.py
scripts/animapk_cuda/streaming_confirm.py
scripts/measure_grid_carrier.py
scripts/vae_decoder_oracle.py
scripts/publish_hf_pack.py
```

Add focused grid tooling around these rather than replacing them.

Suggested additions if helpful:

```text
scripts/animapk_cuda/grid_repro.py
scripts/animapk_cuda/grid_compare.py
scripts/animapk_cuda/latent_periodicity.py
scripts/animapk_cuda/make_grid_contact_sheet.py
scripts/animapk_cuda/rgb_f32_to_png.py
```

Keep each tool deterministic and manifest-driven.

---

# 34. Provenance requirements for every image experiment

Every image experiment must produce `manifest.json` containing at least:

```text
experiment id
timestamp UTC
git branch
git commit
dirty/clean state
script paths + SHA256
Clore hostname
GPU
torch version
CUDA version
source repo/revision
official source SHA
pack filenames
pack SHA256s
VAE pack SHA256
fixture/golden SHA256
prompt
seed
sigma list/hash
conditioning hash
initial latent/noise hash
variant/backend
dtype modes
attention backend/mode
output latent shape/dtype/hash
decoded RGB hash
PNG hash
carrier metrics path
vision review path
HF repo/path/revision
```

If two outputs are being compared, provenance must prove that intended controls were actually held constant.

---

# 35. Keep long-running work resumable

Previous Clore runs were disrupted.

For every expensive multi-variant run:

- write checkpoint after each variant
- flush metrics immediately
- write final latent before VAE decoding
- keep per-variant logs
- use `tmux`
- make reruns skip already verified outputs by hash
- never rely on one long process that loses all results at the end

If a run dies after producing a useful latent/image, upload that partial evidence before restarting if practical.

---

# 36. Do not duplicate giant data unnecessarily

While Clore remains active:

- keep the actively used source weights/packs/cache on Clore for speed
- upload their durable copy/provenance to HF
- do not repeatedly redownload the same 4â€“8 GB assets
- do not create many byte-identical pack copies
- use hardlinks/symlinks locally when safe
- record hashes instead of renaming identical data as new â€œvariantsâ€

On VPS:

- keep no multi-GB model copies at all unless a tiny temporary unavoidable transfer occurs
- immediately remove such temporary data after verified transfer

---

# 37. Hugging Face upload helper for experiment bundles

If no existing project helper handles arbitrary result directories, use `huggingface_hub` directly.

Example pattern:

```python
import os
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"].strip()
repo_id = os.environ["ANIMAXS_HF_ARTIFACT_REPO"]

api = HfApi(token=token)

api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",   # use the actual established repo type
    folder_path="/workspace/out/grid_experiment",
    path_in_repo="experiments/2026-08-14_grid_experiment",
    commit_message="Upload AnimaXS CUDA grid experiment evidence",
)
```

Do not blindly create a second repo if one already exists.

If the artifact repo is type `model`, use that exact type consistently.

Record the returned commit revision.

For giant individual packs, use direct `upload_file`/HF large-file support rather than copying them to a temporary staging directory.

---

# 38. Current unique evidence should be migrated to HF, not left only in the handoff ZIP or VM

The previous package contains valuable evidence such as:

```text
RESULTS_SUMMARY.md
cuda_vs_metal_block_quick.json
Metal step-0 logs
Metal provenance
quality diagnostics
sample step-0 captures
Clore ladder summaries
source oracle parity
precision ladder stage parity
```

Make sure the durable project evidence repository contains the authoritative versions.

Do not depend on a conversation attachment as the only archive.

---

# 39. Root-cause report must distinguish three questions

The final report must not conflate:

## A. Backend parity

```text
Does Metal execute the tested DiT math similarly to CUDA?
```

Current evidence: largely yes for the tested packs/blocks.

## B. Latent quality

```text
How close is final latent to the golden?
```

Current evidence: FP16/W8 ~0.81; W4 much worse.

## C. Visible grid pathology

```text
What produces the severe regular 8px image carrier?
```

Current status at the start of this run: **not yet root-caused**.

The run is successful only if question C is directly addressed.

---

# 40. Required initial output table

After the first image matrix, produce something like:

| Lane | DiT source/runtime | final latent cosine vs golden | RGB cosine | RGB RMSE | 8px carrier total | ratio vs reference/control | Hermes vision | Grid? |
|---|---|---:|---:|---:|---:|---:|---|---|
| G0 | golden latent -> fixed CUDA VAE | 1.0 | ... | ... | ... | ... | clean/bad | yes/no |
| G1 | official BF16 CUDA | ... | ... | ... | ... | ... | ... | yes/no |
| G2 | FP16-all animapk CUDA | ... | ... | ... | ... | ... | ... | yes/no |
| G3 | W8 animapk CUDA | ... | ... | ... | ... | ... | ... | yes/no |
| G4 | bad Metal latent -> fixed CUDA VAE | ... | ... | ... | ... | ... | ... | yes/no |

This table is the first major decision point.

Upload it to HF and commit a concise copy/summary to the branch.

---

# 41. Required experiment ordering

Unless new evidence invalidates it, use this order:

```text
1. repo audit
2. Clore environment audit
3. immediately secure existing unique Clore artifacts to HF
4. reset active TODO/session
5. G0 golden latent -> fixed CUDA VAE
6. G1 official BF16 CUDA -> image
7. G2 FP16-all CUDA -> image
8. G3 W8 CUDA -> image
9. carrier + Hermes vision
10. obtain/run G4 Metal-latent-through-same-VAE when available
11. apply decision tree
12. if CUDA grids: establish clean true-ComfyUI/source CUDA control
13. isolate the clean-vs-bad CUDA difference
14. implement/falsify fix candidates
15. validate final CUDA fix
16. upload everything to HF
17. port only proven fix to Metal
18. macOS image/carrier/vision validation
19. normal CI/build/simulator validation
20. final docs/decisions/TODO/session
21. verify HF archive
22. terminate Clore as soon as no more CUDA work is needed
```

Do not spend the early run on W4, adapter optimization, or cosmetic app work.

---

# 42. Final validation after a real grid fix

A root-cause fix should survive:

```text
A. clean canonical source/BF16 CUDA
B. FP16-all CUDA
C. W8 CUDA
D. same fixed VAE
E. carrier measurement
F. Hermes visual review
G. causal ablation: revert only the fix -> grid returns or metric materially regresses
H. reapply fix -> grid disappears again
```

That revert/reapply test is important.

It converts correlation into much stronger causal evidence.

Then validate the analogous Metal change:

```text
focused parity
canonical image
carrier
vision
normal CI
build
simulator tests
```

---

# 43. Final production choice

If the grid is solved:

- prefer W8 if it remains visually clean and close to the source/FP16 result
- do not switch to W4 merely for size unless W4 quality is separately fixed
- preserve streaming/bounded-memory behavior required for iPhone XS Max
- do not add a desktop-only workaround that cannot fit the iOS runtime
- keep the production implementation as simple as possible

If a fix only works in full BF16 source PyTorch but cannot be mapped to the iOS runtime, continue to identify which narrow operation/tensor needs higher precision.

---

# 44. Clore shutdown checklist â€” mandatory

Before shutting down:

- [ ] all useful source code committed
- [ ] `investigate/animapk-cuda-parity` pushed
- [ ] current HEAD recorded
- [ ] active TODO updated
- [ ] `DECISIONS.md` updated
- [ ] `HERMES_SESSION.md` updated
- [ ] all unique Clore result files uploaded to Hugging Face
- [ ] all important generated packs uploaded or already verified in their HF model repos
- [ ] all fixture/result hashes recorded
- [ ] HF repository/revision recorded
- [ ] HF remote files verified
- [ ] no useful file exists only on Clore
- [ ] no large result exists only on VPS
- [ ] no planned CUDA test still needs the GPU
- [ ] Clore instance terminated/closed
- [ ] final report explicitly says whether shutdown succeeded

If the grid is solved early and only macOS porting remains, perform this shutdown checklist **before** waiting on further macOS work.

If CUDA is proven clean and the remaining issue is Apple-only, also shut Clore down once all CUDA evidence is safely persisted.

---

# 45. Final report format

```markdown
# AnimaXS grid-pattern investigation â€” final report

## Repository state
branch:
HEAD:
origin/main:
origin/investigate/animapk-cuda-parity:
working tree:

## Clore
provided instance used:
GPU:
environment:
CUDA work finished:
instance closed:
shutdown confirmation:

## Hugging Face archive
artifact repo:
repo type:
revision:
main experiment paths:
all unique Clore artifacts uploaded:
verification:

## Initial CUDA image reproduction
G0 golden latent -> CUDA VAE:
G1 BF16 CUDA:
G2 FP16-all CUDA:
G3 W8 CUDA:
G4 Metal latent -> CUDA VAE:

## Grid carrier table
...

## Vision review
Hermes native vision used:
Nano-GPT fallback used:
model if fallback:
summary:

## Was the grid reproduced on CUDA?
yes/no
evidence:

## Earliest localized cause
...

## Root cause
...

## Fix
...

## Causal revert/reapply test
...

## CUDA before/after
latent:
RGB:
carrier:
visual:

## Metal before/after
latent:
RGB:
carrier:
visual:

## Production W8 result
...

## W4 status
...

## CI
focused macOS:
normal CI:
iPhone build:
simulator tests:

## Files changed
...

## Decisions added
...

## Remaining device-only risks
...

## Remaining TODO
...

## Final statement
Did we actually solve the visible grid pattern?
What exact evidence proves it?
Did we close the Clore instance?
```

---

# 46. Completion standard

Do not finish with:

> â€œCUDA and Metal match, therefore the grid is explained.â€

That is not enough.

A good completion is:

> â€œWe reproduced the same grid on CUDA, localized it to X by clean-vs-bad substitution, changed Y, the carrier dropped from A to B, Hermes/vision confirms the woven texture disappeared, revert/reapply proves causality, and the analogous Metal change reproduces the fix.â€

or, if CUDA is clean:

> â€œThe controlled CUDA images are clean while the known Metal latent/image path is not; the defect was localized to the Apple-side stage X, and Clore was shut down because further CUDA work no longer adds information.â€

The visible image pathology is the primary target.

Do not confuse a better cosine score with a solved grid.