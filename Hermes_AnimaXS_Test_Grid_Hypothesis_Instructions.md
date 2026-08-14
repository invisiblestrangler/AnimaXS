# Hermes Instructions — Close the AnimaXS 8px Grid Root Cause and Port the Fix

**Date:** 2026-08-14  
**Target repo:** `invisiblestrangler/AnimaXS`  
**Current investigation branch:** `investigate/animapk-cuda-parity`  
**Known reported HEAD at handoff:** `44d5d68` — **verify this; do not assume it is still current**  
**Primary goal:** prove or falsify the missing ComfyUI `Wan21.process_out()` boundary transform, fix the CUDA reference/harness first if confirmed, then port the exact fix to the Metal/iOS production path and validate that the visible 8px grid is gone.

---

## 0. Read this first — new highest-priority hypothesis

The investigation has now isolated the grid much further than before.

The existing evidence proves:

- the grid reproduces on CUDA with the **official BF16 source**, so it is not an Apple/Metal-only defect;
- FP16-all, W8, W4, CUDA, and Metal all reproduce the same family of grid artifacts;
- the VAE implementation itself is clean when given the known-clean golden final latent;
- the grid is already present when raw model/sampler-space outputs are decoded;
- model-block parity and Euler math are already extremely strong;
- the recorded golden callback tensor at the last Euler step is grid-carrying, while the golden `final_latent` is clean.

The previous investigation treated the last point as an inconsistency in the golden capture.

There is now a much stronger explanation:

> **ComfyUI does not return the raw sampler-space latent directly. For Anima, it applies the model latent-format output transform before the returned latent reaches the VAE. Anima is registered with `latent_formats.Wan21`, whose `process_out()` is a non-identity per-channel affine transform. The custom CUDA/Metal path appears to have omitted that boundary conversion and decoded sampler-space latents as VAE-space latents.**

For Wan21:

```python
vae_latent = sampler_latent * latents_std / scale_factor + latents_mean
```

and `scale_factor == 1.0`, so:

```python
vae_latent = sampler_latent * latents_std + latents_mean
```

ComfyUI's `Wan21.process_in()` is the inverse:

```python
sampler_latent = (vae_latent - latents_mean) * scale_factor / latents_std
```

Do **not** blindly copy constants from current ComfyUI `master`.  
Use the exact constants and semantics from the **pinned `comfy-ref` revision already used by this project/golden fixture**.

The current upstream ComfyUI implementation is useful as a conceptual cross-check only:

- `comfy/supported_models.py`: Anima uses `latent_formats.Wan21`
- `comfy/latent_formats.py`: `Wan21.process_in()` / `Wan21.process_out()`
- `comfy/model_base.py`: `process_latent_out()` delegates to the latent format
- `comfy/samplers.py`: `CFGGuider.inner_sample()` applies `self.inner_model.process_latent_out(samples.to(torch.float32))` **after** the sampler returns

This post-sampler transform explains why the final callback's `denoised` need not numerically equal the workflow's returned `final_latent`, even though the last Euler update itself ends at the final `denoised`.

---

# 1. Mission

You are not being asked to merely investigate another possibility.

You must:

1. **Prove or falsify** the missing-Wan21-transform hypothesis using already-saved tensors first.
2. If confirmed, identify every AnimaXS code path where sampler/model-space latents cross into VAE-space.
3. Fix the CUDA reference/runtime path.
4. Re-run the grid reproduction and demonstrate carrier collapse / clean images.
5. Port the same semantics to the Metal/iOS production path.
6. Validate Metal/macOS as far as GitHub Actions permits.
7. Re-run relevant FP16-all, W8, and W4 controls so weight precision is not accidentally conflated with the fix.
8. Update durable documentation and decisions.
9. Upload all durable artifacts to Hugging Face and hash-verify them.
10. Push clean commits.
11. When no useful CUDA work remains and all needed Clore artifacts are durable, **terminate the already-running Clore instance**. Do not leave it running unnecessarily.

Do not stop after only proving the hypothesis.  
Do not stop after only fixing CUDA.  
The desired end state is a production-path fix with evidence that the 8px grid is removed or, if the hypothesis is falsified, a tightly localized next cause supported by new evidence.

---

# 2. Mandatory context recovery before touching code

At the start of the run:

```bash
cd <repo>
git status
git branch --show-current
git rev-parse HEAD
git log --oneline -n 20
git remote -v
```

Verify the current branch, remote state, and working tree.

Then re-read the current investigation state. At minimum inspect:

- `GRID_FINDINGS_SUMMARY.md` if present
- `HERMES_GRID_ROOT_CAUSE.md`
- `HERMES_GRID_ROOT_CAUSE_MIMO_V2_5.md` if present
- `HERMES_SESSION.md`
- `DECISIONS.md`
- `STATUS.md` if present
- `TODO.md`
- the CUDA scripts under `scripts/animapk_cuda/`
- the exact pinned `comfy-ref` code used by the fixture
- VAE oracle/decoder documentation and code
- the production Metal sampler â†’ VAE boundary
- pack/runtime code for FP16-all, W8, and W4

Older instruction files, if present, are supporting context only:

- `Hermes_AnimaXS_Step0_Root_Cause_Instructions.md`
- `Hermes_AnimaXS_Animapk_CUDA_Parity_Instructions.md`

Do not blindly inherit old hypotheses from them. The latest measured evidence and this instruction file supersede stale investigative direction.

### TODO reset

The TODO file has accumulated stale work in earlier runs.

Do not simply append another block to an old list.

After reading the current state:

1. preserve any genuinely incomplete task that still matters;
2. archive or remove stale completed items;
3. rewrite `TODO.md` into a short ordered plan for **this** root-cause closure;
4. put the Wan21 saved-latent proof at the top.

---

# 3. Existing evidence you must preserve

The latest investigation reported:

## 3.1 CUDA reproduction matrix

Using the same CUDA VAE path:

| Lane | Source | Final latent cosine vs golden | 8px carrier | Result |
|---|---|---:|---:|---|
| G0 | golden final latent | 1.0 | ~0.0000548 | clean |
| G1 | official BF16, real CUDA graph | ~0.81103 | ~0.013529 | grid |
| G2 | FP16-all `.animapk` CUDA | ~0.812982 | ~0.013506 | grid |
| G3 | W8 `.animapk` CUDA | ~0.808978 | ~0.013407 | grid |
| G4 | Metal FP16-all final latent â†’ CUDA VAE | ~0.812312 | ~0.013470 | grid |
| G4 | Metal W8 final latent â†’ CUDA VAE | ~0.809816 | ~0.013462 | grid |
| G4 | Metal W4 final latent â†’ CUDA VAE | ~0.660366 | ~0.027051 | severe grid |

This established that the grid is not a Metal-only defect.

## 3.2 Decoder sensitivity

`golden + k*(G1-golden)` produced a monotonic grid increase up to ~246.7Ã— at `k=1`, while a norm-matched random perturbation produced only ~2.1Ã—.

That means the bad delta is structured.

## 3.3 Golden self-check

Reported:

- golden recorded last callback/`step_latents[7]` â†’ ~254.4Ã— grid;
- golden `final_latent` â†’ ~1Ã— clean;
- their cosine is only ~0.837.

The previous interpretation was that the golden trace itself was inconsistent.

**Re-evaluate this interpretation first.**

## 3.4 Model and sampler parity

Already established:

- raw sigma + 512-token context produced ~0.999064 cosine at the captured step-7 block-0 probe;
- CUDA/Metal block parity is extremely high;
- the implemented Euler formula is algebraically the same as ComfyUI Euler.

Do not repeat large portions of the old parity ladder unless a new result specifically requires it.

---

# 4. Phase A — zero-inference proof of the Wan21 boundary hypothesis

This is the single highest-priority action.

**Do not launch a fresh full DiT run before doing this.**

Use already-saved golden and reproduction tensors.

## A1. Identify the exact pinned ComfyUI semantics

From the project's exact `comfy-ref` revision, record:

- commit SHA;
- `Anima` model registration;
- its `latent_format`;
- exact `Wan21.latents_mean`;
- exact `Wan21.latents_std`;
- `scale_factor`;
- `process_in`;
- `process_out`;
- exact location where the sampler return is passed through `process_latent_out()`.

Save a small machine-readable provenance JSON.

Example:

```json
{
  "comfy_ref_sha": "...",
  "anima_latent_format": "Wan21",
  "scale_factor": 1.0,
  "source_files": {
    "supported_models": "...",
    "latent_formats": "...",
    "samplers": "...",
    "model_base": "..."
  }
}
```

Do not rely on remembered constants.

## A2. Test the golden final-step relationship directly

Load the recorded raw callback tensor that was called `golden step_latents[7]`.

Apply the **exact pinned** Wan21 `process_out()`:

```python
converted = raw_step7 * std / scale_factor + mean
```

Compare `converted` against the known-clean golden `final_latent`.

Record at minimum:

- cosine similarity;
- RMSE;
- max absolute difference;
- mean absolute difference;
- per-channel mean/std before conversion;
- per-channel mean/std after conversion;
- per-channel mean/std of golden final;
- tensor shapes and dtypes;
- SHA-256 of source tensors.

### Pass condition

A very high similarity / small error against `golden final_latent` would explain the supposed golden contradiction and strongly confirm the omitted boundary transform.

Do **not** choose an arbitrary numeric gate in advance if the fixture does not support one. Report exact measured values and compare them to the pre-transform baseline (`cos ~0.837`, `RMSE ~0.845` reported by the existing run).

If it becomes essentially exact, record that as decisive evidence.

## A3. Decode both sides

Through the already-validated CUDA VAE, decode:

1. raw golden step-7 callback latent;
2. Wan21-`process_out` converted step-7 latent;
3. golden `final_latent`.

For each:

- save RGB float output;
- save PNG;
- run exact-8px carrier metric;
- compare RGB cosine/RMSE to the clean reference.

Expected pattern if hypothesis is correct:

```text
raw callback                -> grid
raw callback + process_out  -> clean / near-reference
golden final                -> clean
```

## A4. Apply the transform to saved G1/G2/G3/G4 final latents

Without rerunning DiT:

- G1 official BF16 saved final sampler latent;
- G2 FP16-all;
- G3 W8;
- G4 Metal FP16-all;
- G4 Metal W8;
- G4 Metal W4 if saved in compatible form.

For each:

```text
saved sampler-space latent
    -> exact pinned Wan21.process_out
    -> same CUDA VAE
    -> PNG
    -> exact-8px carrier
```

Record before/after carrier ratios.

### Strong confirmation criterion

If the carrier collapses from the current ~245Ã— family toward the clean baseline and the visible woven/grid pattern disappears across BF16/FP16/W8/Metal, treat the missing boundary transform as the root cause.

W4 may still have quality degradation due to quantization; the key question is whether the **regular 8px grid carrier** collapses independently of overall image fidelity.

---

# 5. Phase B — if A confirms, fix the CUDA/reference path first

Do not immediately patch Metal before making the CUDA path clean, because CUDA is the fastest controlled environment.

## B1. Find the correct abstraction boundary

Search for every place where a final sampler/model-space latent is handed to:

- the VAE decoder;
- PNG/image conversion;
- saved as a supposedly VAE-space `final_latent`;
- comparison code that assumes golden final and raw sampler output share coordinates.

The fix should live at a clearly named boundary, not be scattered as magic constants.

Prefer a small explicit function, e.g. conceptually:

```python
def anima_sampler_to_vae_latent(x):
    # exact pinned Wan21.process_out semantics
    ...
```

or a generic latent-format helper if the repo already has the right abstraction.

Do not overengineer a general model framework solely for this bug.

## B2. Preserve coordinate-space naming

Stop calling all 16-channel tensors simply `latent`.

Use explicit names where practical:

- `sampler_latent`
- `model_latent`
- `vae_latent`

Add comments at the boundary:

```text
Anima/Wan21 sampler coordinate space != VAE decode coordinate space.
Apply process_out exactly once before VAE decode.
```

This is important because the earlier investigation was confused by comparing tensors from opposite sides of this boundary.

## B3. Prevent double application

Add a unit/regression test that proves the transform is applied:

- exactly once;
- at the correct boundary;
- not inside the VAE itself;
- not again after the VAE receives its canonical input.

The VAE oracle statement “latent is fed unchanged into the VAE” can remain true **inside the VAE**, but the runtime must convert from sampler space to VAE space **before** entering the decoder.

## B4. Audit `process_in()` too

Even if text-to-image currently starts from sampler-space Gaussian noise and does not require a VAEâ†’sampler conversion at initialization, audit all code paths for:

- img2img;
- VAE-encoded inputs;
- future latent reuse;
- any conditioning that carries VAE latents into the model.

If such a path exists, it must use the exact inverse `Wan21.process_in()` semantics.

Do not introduce `process_in()` into the pure-noise txt2img path unless the pinned ComfyUI path actually does so.

---

# 6. Phase C — rerun the CUDA grid reproduction after the code fix

Now rerun the smallest sufficient validation first.

## C1. Golden regression

- golden final latent â†’ VAE still clean;
- raw final-step callback â†’ process_out â†’ VAE matches clean behavior.

## C2. Official BF16 control

Run G1 again with the official BF16 source and real pinned graph.

Required outputs:

- sampler-space final latent;
- converted VAE-space final latent;
- PNG;
- carrier metric;
- comparison to reference;
- manifest with source/commit/model hashes.

If G1 is clean after conversion, this proves the defect was not pack quantization.

## C3. FP16-all and W8

Then rerun FP16-all and W8.

The goal is not necessarily pixel-perfect equality with BF16. The goal is:

- regular 8px grid removed;
- carrier close to clean/reference regime rather than ~245Ã—;
- ordinary expected precision differences remain.

## C4. W4

Run W4 as a control if practical.

Do not let W4-specific quantization quality distract from the root-cause closure. If the grid is removed but W4 quality is still worse, document that separately.

---

# 7. Phase D — port the confirmed fix to Metal/iOS

Only after CUDA confirmation.

## D1. Locate the production boundary

Find the exact point in the Metal/iOS pipeline where the final Euler/sample tensor becomes the input to the VAE.

Apply the same **pinned Wan21 `process_out()`** semantics there, exactly once.

For 16 channels:

```text
vae[c] = sampler[c] * std[c] / scale_factor + mean[c]
```

Use the exact pinned constants.

## D2. Implementation requirements

The implementation should:

- operate on all 16 channels;
- preserve tensor layout correctly;
- not accidentally broadcast across the wrong dimension;
- support the production dtype safely;
- avoid allocating huge unnecessary copies on iPhone;
- be deterministic;
- be covered by a small CPU/unit test where possible;
- not require repacking DiT weights merely to add this affine boundary transform.

Prefer fusing/in-place transformation if safe and if it materially reduces memory, but correctness first.

## D3. Add a known-vector parity test

Generate a small deterministic latent fixture and expected transformed result from pinned Python/ComfyUI.

Metal/Swift test should compare against it.

This catches:

- wrong channel order;
- missing mean;
- inverse transform by mistake;
- duplicated transform;
- dtype/broadcast errors.

## D4. Re-run Metal captures

At minimum test:

- FP16-all;
- W8;
- W4 if time/resource allows.

Capture both:

```text
raw sampler final
converted VAE-space final
```

Compare the converted Metal tensor against the equivalent CUDA conversion.

Do not compare a raw Metal sampler tensor directly against a post-`process_out` ComfyUI golden final tensor and call the difference model divergence.

---

# 8. Phase E — image-level validation is the primary gate

Cosine is diagnostic; visible image correctness and the exact 8px carrier are the real acceptance criteria.

For every important fixed lane produce:

- output PNG;
- exact-8px carrier;
- carrier ratio vs clean reference;
- RGB cosine/RMSE where useful;
- side-by-side contact sheet;
- high-gain difference;
- FFT comparison/crop if it helps.

### Acceptance target

The former failure was approximately:

```text
reference carrier â‰ˆ 0.0000548
bad lanes          â‰ˆ 0.0134â€“0.0135
bad/reference      â‰ˆ 245Ã—
```

The fix is successful when the **regular 8px carrier collapses toward the clean regime and visual review no longer shows the woven/checkerboard grid**.

Do not declare victory from latent cosine alone.

---

# 9. Vision review instructions

Use Hermes' native vision capability on the generated images/contact sheets.

Ask it specifically to inspect:

- regular orthogonal 8px/checkerboard structure;
- whether the grid remains in flat regions;
- whether detail is merely blurred rather than truly corrected;
- whether fixed lanes visually approach the clean reference;
- whether W4 has non-grid quantization degradation.

If Hermes native vision fails, use the existing manual image POST fallback through Nano-GPT with:

```text
xiaomi/mimo-v2.5
```

Do **not** use the old `meta/muse-spark-1.2-contributor` fallback; it was replaced.

Do not add Telegram sending logic. It is unnecessary for this Hermes run.

Machine metrics remain authoritative for the exact carrier; vision is complementary.

---

# 10. Resource rules

## 10.1 Cheap VPS

The VPS is orchestration-only.

It has very limited RAM/disk and must not be used for heavy operations.

Do not:

- run large-model inference there;
- repeatedly download multi-GB weights there;
- unpack huge archives there;
- duplicate model packs there;
- retain bulky artifacts there.

Use it for:

- git;
- light scripting;
- orchestration;
- small metadata;
- launching/monitoring jobs;
- pushing code.

## 10.2 Clore.ai

There is already an instance for the investigation.

- Use the **already-running** Clore instance.
- Do not rent another instance.
- Reuse existing CUDA environment, model caches, fixtures, and packs where verified.
- Use it for heavy PyTorch/CUDA inference, decoding, and fast iteration.
- Do not destroy useful caches during the run.
- Persist useful outputs to Hugging Face before deleting anything.

When the grid issue is solved, or when there is no remaining useful CUDA work:

1. upload all durable evidence to Hugging Face;
2. hash-verify remote files;
3. update `HERMES_SESSION.md` / `DECISIONS.md`;
4. commit and push code/docs;
5. **terminate the Clore instance**.

The older session note saying to leave Clore running is superseded by this instruction.

## 10.3 GitHub Actions

Use GitHub Actions intelligently:

- Linux runners: light pack/format/unit validation where useful;
- macOS runners: Xcode/Metal/Apple-specific tests;
- avoid repeated full E2E macOS jobs for every tiny edit;
- batch related Apple-side changes, then run the meaningful validation matrix;
- parallelize independent jobs where it saves wall-clock time.

The agent has no physical iPhone available in this run. Be explicit about the boundary between:

```text
CI/macOS validated
```

and

```text
real iPhone XS Max device validated
```

Do not claim physical-device validation from simulator/macOS CI.

---

# 11. Hugging Face is durable storage

The VPS/Clore filesystem is not the archive.

Use the existing durable investigation dataset:

```text
ScalingBiz/AnimaXS-investigation-artifacts
```

and preserve the existing structure where practical.

Existing experiment area reported:

```text
experiments/2026-08-14_grid-repro/
```

Create a clearly named follow-up directory, for example:

```text
experiments/2026-08-14_wan21-process-out-fix/
```

Store at minimum:

- provenance JSON;
- exact pinned Wan21 constants/source SHA;
- zero-inference conversion results;
- golden raw/converted/final comparison metrics;
- G1/G2/G3/G4 before/after metrics;
- fixed PNGs;
- contact sheets;
- FFT/high-gain views;
- relevant `.f32` latents;
- manifests;
- `SHA256SUMS`;
- logs that are actually useful;
- final summary.

After upload, verify hashes against the remote copy.

Do not use Hugging Face merely as a dump of redundant temporary files. Keep decisive evidence and manifests.

---

# 12. Parallelism and test ordering

Optimize for **fast falsification and fast proof**, not maximum test count.

Recommended order:

```text
A. zero-inference saved-latent process_out proof
        |
        +-- if falsified -> do not patch production; investigate exact boundary mismatch
        |
        +-- if confirmed
              |
              v
B. patch CUDA boundary
              |
              v
C. G1 BF16 image/carrier confirmation
              |
              +--> in parallel after G1:
              |      G2 FP16-all
              |      G3 W8
              |      W4 control
              |
              v
D. patch Metal boundary + unit fixture
              |
              v
E. macOS/Metal CI
              |
              v
F. final image/carrier evidence + docs + HF + shutdown
```

Do not burn time rerunning already-settled block-parity tests unless the new boundary result unexpectedly fails.

---

# 13. If the Wan21 hypothesis is falsified

Do not force the conclusion.

If:

```text
process_out(golden_step7)
```

does **not** closely explain `golden_final_latent`, immediately determine why.

Audit, in this order:

1. Was the saved `step_latents[7]` truly the final callback `denoised` from the same run?
2. Was the tensor reshaped/transposed/folded before storage?
3. Is the pinned Anima latent format actually Wan21 at that exact revision?
4. Is there another wrapper after the sampler in `csample.sample`?
5. Is `final_latent` saved before/after a node-level transform not yet accounted for?
6. Are callback and returned tensors from separate invocations due to capture-script behavior?
7. Only then return to context/conditioning assembly.

If the hypothesis fails, capture the **first exact operation where the two paths diverge**.

Do not regress to a broad suspect list.

---

# 14. Why the previous investigation missed this — and how not to repeat it

Document this explicitly in `DECISIONS.md` once experimentally verified.

The miss came from a conceptual boundary error:

```text
Sampler / model latent space
        !=
VAE decode latent space
```

The investigation correctly reproduced:

- model blocks;
- timestep convention;
- context shape;
- Euler update;
- VAE decoder math.

But it treated those as if they formed one uninterrupted tensor coordinate system.

The phrase:

```text
"latent fed unchanged into the VAE — no mean/std transform"
```

was true **inside the VAE decoder**, but was incorrectly generalized to mean:

```text
"the sampler output can be fed unchanged into the VAE"
```

Those are not the same claim.

ComfyUI hides the model-specific latent-format conversion in its framework boundary:

```text
sampler returns raw samples
    -> model.process_latent_out(...)
    -> latent_format.process_out(...)
    -> workflow receives VAE-space latent
```

That boundary is easy to miss when copying the visible Euler/model/VAE equations separately.

The callback semantics made this harder:

- callback sees internal sampler tensors;
- workflow receives the postprocessed sampler result;
- both were casually called "latent";
- the existing capture script also misnamed callback arguments, further obscuring which coordinate space was being recorded.

The decisive warning sign was:

```text
last callback denoised -> grid
workflow final_latent  -> clean
```

Instead of assuming one of those captures was wrong, the investigation should have traced **every operation between callback and returned sample**.

New rule for this project:

> Whenever two supposedly identical tensors disagree, trace the framework call graph from producer to consumer and explicitly record every model-specific pre/post transform before debugging numerical kernels.

Also add coordinate-space labels to future fixtures/manifests.

---

# 15. Durable state / context-compaction protocol

This run may be long. Do not rely on conversational memory.

Maintain these files continuously:

## `TODO.md`

Short, current, ordered checklist.  
Update after every meaningful phase.

## `DECISIONS.md`

For every promoted/rejected hypothesis record:

- decision ID;
- date;
- branch/commit;
- exact source revision;
- pack/model hashes;
- experiment;
- key metrics;
- what it proves;
- what it does **not** prove;
- artifact path.

Do not overwrite old decisions; append superseding decisions.

## `HERMES_SESSION.md`

Keep a compact live recovery state:

- current branch/HEAD;
- Clore state;
- most recent successful command/test;
- current hypothesis status;
- artifact locations;
- next 3 actions;
- blockers;
- uncommitted changes.

### Before any context compaction

Update:

1. `TODO.md`
2. `HERMES_SESSION.md`
3. `DECISIONS.md` if a conclusion changed
4. commit useful code/docs if coherent

### Immediately after context compaction/restart

Re-read:

1. this instruction file;
2. `HERMES_SESSION.md`;
3. `TODO.md`;
4. latest `DECISIONS.md` entries;
5. `GRID_FINDINGS_SUMMARY.md`;
6. relevant source files before editing.

Do not reconstruct state from memory.

---

# 16. Update cadence

Give frequent concise progress updates through the normal Hermes interaction so bad decisions can be caught early.

Useful update points:

- after state/context audit;
- after exact pinned Wan21 semantics are confirmed;
- immediately after the zero-inference golden transform test;
- after G1 image/carrier result;
- before changing the Metal production path;
- after macOS CI;
- before Clore shutdown.

An update should state:

```text
What I tested
What I measured
What that proves / rejects
What I am doing next
```

Do not spam low-level command-by-command logs.

No Telegram sending is required.

---

# 17. Git / branch hygiene

Stay on the investigation branch until the root cause is proven and the fix is validated.

Before edits:

```bash
git status --short
git rev-parse HEAD
git fetch --all --prune
```

Do not rewrite unrelated history.

Commit in coherent units, for example:

```text
test: prove Anima Wan21 sampler-to-VAE latent transform
fix: apply Wan21 process_out before CUDA VAE decode
fix: apply Anima latent postprocess before Metal VAE decode
test: add sampler-to-VAE latent parity fixture
docs: close 8px grid root cause investigation
```

Push frequently enough that work is durable.

Do not commit:

- secrets;
- HF tokens;
- Clore credentials;
- giant generated model files;
- temporary CUDA caches.

---

# 18. Required regression tests

At minimum add/retain tests for:

## 18.1 Mathematical transform

For known 16-channel values:

```text
process_out(x)
```

matches pinned ComfyUI.

## 18.2 Inverse

Where appropriate:

```text
process_in(process_out(x)) ~= x
```

within dtype tolerance.

## 18.3 Channel mapping

Use deliberately distinct channel values so a transpose/order mistake cannot accidentally pass.

## 18.4 Golden relationship

If confirmed:

```text
process_out(golden final callback sampler-space tensor)
    ~= golden final VAE-space tensor
```

preserve this as a regression fixture if storage size allows, otherwise use a representative crop/subtensor with provenance.

## 18.5 Image-level grid gate

Keep an automated exact-8px carrier check so a future regression cannot reintroduce the artifact while numerical parity tests still pass.

---

# 19. Final acceptance checklist

Do not call the run complete until all applicable boxes are satisfied.

- [ ] repo/branch/HEAD verified
- [ ] stale TODO reset into this run's plan
- [ ] exact pinned ComfyUI Anima latent format recorded
- [ ] exact pinned Wan21 constants recorded
- [ ] callback â†’ `process_out` â†’ golden final relationship tested
- [ ] zero-inference saved G1/G2/G3/G4 conversion test completed where tensors exist
- [ ] root cause either confirmed or falsified with hard measurements
- [ ] CUDA code fixed if confirmed
- [ ] official BF16 G1 rerun clean of regular 8px carrier
- [ ] FP16-all rerun
- [ ] W8 rerun
- [ ] W4 control run if practical
- [ ] Metal/iOS boundary fixed
- [ ] known-vector Metal/Swift transform test added
- [ ] macOS/Metal CI green for relevant jobs
- [ ] visible image review completed
- [ ] exact-8px carrier evidence recorded
- [ ] `DECISIONS.md` updated
- [ ] `HERMES_SESSION.md` finalized
- [ ] final investigation summary written
- [ ] artifacts uploaded to Hugging Face
- [ ] Hugging Face hashes verified
- [ ] code/docs committed and pushed
- [ ] no useful CUDA work remains
- [ ] Clore instance terminated

---

# 20. Final report format

Finish with a compact evidence-backed report:

```markdown
# AnimaXS 8px Grid — Final Root Cause Report

## Root cause
<one precise sentence>

## Proof
- golden callback raw -> ...
- after pinned Wan21 process_out -> ...
- G1 before/after carrier -> ...
- G2 before/after -> ...
- G3 before/after -> ...
- Metal before/after -> ...

## Code fix
- CUDA:
- Metal/iOS:
- tests:

## Remaining quality differences
- FP16:
- W8:
- W4:

## CI
- run IDs:
- jobs:
- result:

## Durable artifacts
- HF repo/path:
- manifest:
- SHA256SUMS:

## Repo
- branch:
- final HEAD:
- working tree:

## Clore
- terminated: yes/no
```

---

# 21. Primary principle

The project has already spent substantial effort proving that the kernels, quantized weights, Metal backend, CUDA backend, VAE implementation, and Euler trajectory are individually close.

Do not keep optimizing those components while a model-specific **coordinate-space boundary** remains untested.

The next decisive experiment is:

```text
raw saved sampler latent
        +
exact pinned Wan21.process_out
        â†“
same validated VAE
        â†“
measure exact 8px carrier
```

Run that first.

If it collapses the grid, implement the same boundary semantics everywhere and finish the app-side fix.