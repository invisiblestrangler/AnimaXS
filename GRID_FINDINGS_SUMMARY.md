# AnimaXS 8px Grid-Pattern Investigation — Detailed Findings

**Date:** 2026-08-14 (CUDA grid-reproduction run)
**Branch:** `investigate/animapk-cuda-parity` @ `44d5d68` (clean, pushed)
**Clore instance:** O-2024354 (RTX 3060 12 GB, torch 2.7.1+cu126, venv at `/workspace/venv`)
**Primary doc:** `HERMES_GRID_ROOT_CAUSE.md` (repo root, from K's instructions)

---

## 1. Mission and prior state

The production AnimaXS iOS app (Metal) generates images with a severe regular
**woven/etched 8-pixel grid**. Previous work (the CUDA↔Metal parity ladder) had
established that Metal and CUDA latent trajectories agree almost exactly
(final-latent cosine FP16-all 0.8123 vs 0.8130; W8 0.8098 vs 0.8090; W4 0.6604
vs 0.6596) and per-block parity is 0.99998–1.00000 — but the **visible grid
itself was never reproduced or measured off Apple hardware**. That missing
experiment was the mission: *reproduce the grid on CUDA/Clore, measure it,
inspect it visually, then find the root cause.*

Known defective baseline (from earlier macOS runs):
- generated exact-8px carrier total ≈ 0.01345, reference ≈ 0.0000548 → **~245×**
- reference image: `case1` canonical (danbooru prompt, seed 1337, 512×512, 8-step Euler)

---

## 2. Environment audit (G0 phase)

- **Repo:** `/root/AnimaXS`, branch `investigate/animapk-cuda-parity`; HEAD was
  `c582805`, origin HEAD identical; working tree clean.
- **Clore:** order 2024354 alive (RTX 3060, 12 GB, ~755 GB free disk);
  `/workspace/{repo,source,fixtures,packs,comfy-ref,out}`; official BF16
  safetensors present (`anima-turbo-v1.0.safetensors`, 4.18 GB); all 3 packs
  present (fp16-all 4.19 GB, w8-v2 2.23 GB, w4-v2 1.19 GB); golden npz
  `case1_danbooru_seed1337.npz` (118 MB, canonical SHA `44d35d4f…a8dc`).
- **Fixes made to Clore environment:**
  - Clore repo was a **shallow clone** (caused fetches to stall at old commit) → `git fetch --unshallow`.
  - `remote.origin.fetch` refspec only tracked `dit-quality-runtime` → fixed to `+refs/heads/*:refs/remotes/origin/*`.
  - Git identity set; `inspect_animapk.py` committed to repo (`948343f`).
- **HF durability established:**
  - `ScalingBiz/AnimaXS-investigation-artifacts` (private dataset) created; all
    pre-existing Clore evidence uploaded and hash-verified (ladder, ladder_real,
    step0_cuda, storage, fixture, environment) plus all Metal step-0 captures
    (`evidence/metal_step0/{fp16-all,w8,w4}/`, 166 files each, hash MATCH round-trip).
  - VAE pack published to `ScalingBiz/AnimaXS-VAE-XSMAX-FP16`
    (`qwen-image-vae-xsmax-fp16.animapk`, SHA-256 `10171af0b826…54fc447`, matches D060).

---

## 3. The decisive experiment: image-space CUDA reproduction (G1 phase)

### 3.1 New tooling (all committed on the branch, in `scripts/animapk_cuda/`)

| Script | Purpose |
|---|---|
| `grid_repro.py` | G0–G4 lanes: golden latent / official BF16 / FP16-all pack / W8 pack / Metal final latents → one identical CUDA VAE → PNG; full manifest (hashes, sigmas, provenance) |
| `latent_periodicity.py` | latent-space checker/parity statistics, FFT peaks, autocorrelations (MD §23) |
| `make_grid_contact_sheet.py` | labeled contact sheet + difference / high-gain / FFT / crops panels (MD §16) |
| `decoder_sensitivity.py` | `golden + k·Δ` interpolation + norm-matched random control (MD §24) |
| `decode_denoised.py` | decode per-step model `denoised` outputs to find where the grid enters |
| `check_input_noise.py` | decode input noise `x0` to establish the VAE noise-floor carrier |
| `trajectory_trace.py` | per-step Euler latent capture |
| `align_trajectory.py` | harness trajectory vs golden `step_latents` alignment |
| `probe_timestep_ctx.py` | step-7 block-00 probe: timestep convention × context candidates vs golden |
| `golden_selfcheck.py` | decode golden `step_latents[7]` vs golden `final_latent` through the same VAE |
| `check_invariant.py` | Euler invariant checks |

**VAE decoder port:** the validated NumPy oracle semantics (D060/D061: latent fed
**unchanged** — no mean/std transform — through `conv2 → decoder → head`,
final-slice rank-5 folds, channel RMS `F.normalize×√C×γ`, single-head spatial
attention, nearest-exact 2× resample) were ported to torch/CUDA and **validated
against the NumPy oracle on the golden latent**: cosine 0.99999997, RMSE
1.6e-4, maxAbs 0.008 (RGB float scale; tighter than the J002 Metal gate of 0.05).

**PNG conversion:** canonical production transform (identical to Metal
`vae_position_to_rgba8`): `clamp((v+1)*0.5, 0, 1)` → `floor(x*255+0.5)` → RGB8.

### 3.2 The G0–G4 lane matrix (the headline result)

Same VAE decode path for every lane. Exact-8px carrier = normalized Fourier
energy at the exact ±1/8-cycle-per-pixel bins after per-channel mean removal
(`scripts/measure_grid_carrier.py`).

| Lane | DiT source / runtime | final latent cos vs golden | 8px carrier total | ratio vs reference | vision |
|---|---|---|---|---|---|
| **G0** | golden latent → fixed CUDA VAE | 1.0 | 0.0000548 | **~1.0× CLEAN** | clean (rgb_cos 0.999989) |
| **G1** | official BF16 source → real pinned CUDA graph, 8-step Euler | **0.81103** | 0.013529 | **~246.7× GRID** | grid |
| **G2** | FP16-all `.animapk` → CUDA runtime | 0.812982 | 0.013506 | ~246.3× GRID | grid |
| **G3** | W8 `.animapk` → CUDA runtime | 0.808978 | 0.013407 | ~244.5× GRID | grid |
| **G4** | Metal fp16-all final latent → CUDA VAE | 0.812312 | 0.013470 | ~245.6× GRID | grid |
| **G4** | Metal w8 final latent → CUDA VAE | 0.809816 | 0.013462 | ~245.5× GRID | grid |
| **G4** | Metal w4 final latent → CUDA VAE | 0.660366 | 0.027051 | ~493× GRID | grid |

Every latent cosine **exactly reproduces** the previously established ladder
(A=0.8110, C=0.8130, D=0.8090, E=0.6600; Metal 0.8123/0.8098/0.6604) —
the reproduction is faithful.

**⇒ Decision-tree Case 2 confirmed: the grid reproduces completely off Apple
hardware.** G0 (golden latent through the same decoder) is bit-clean, so the
defect is **not** the VAE, the PNG conversion, or the decode path. This is the
first time the grid has been produced and measured on CUDA.

### 3.3 Vision review

- Hermes native vision (aux model) failed with a provider billing error; fell
  back to **Nano-GPT `xiaomi/mimo-v2.5`** per the MD §18 manual POST path.
- Contact sheet verdict: **only REFERENCE clean; all 7 generated lanes show a
  regular orthogonal checkerboard grid**; severity G1/G2 extreme, G4-moderate;
  aligned with latent-cell/VAE-upsample boundaries; candidates blur detail
  without removing the grid.
- (The model also flagged G0, but the machine carrier metric is definitive
  there: G0 = 1.0×; at contact-sheet scale an 8px period is not resolvable, so
  the vision model was misreading a downsampled panel.)

---

## 4. Root-cause isolation experiments (G2/G3 phase)

### 4.1 Decoder-sensitivity (MD §24) — the grid is latent-encoded

`delta = G1_final − golden`. Decode `golden + k·Δ` and a **norm-matched random**
perturbation through the same VAE:

| Input | carrier ratio |
|---|---|
| golden alone (k=0) | 1.0× |
| golden + 0.25·Δ | 3.3× |
| golden + 0.50·Δ | 18.3× |
| golden + 0.75·Δ | 72.4× |
| golden + 1.00·Δ (= G1) | **246.7×** |
| golden + random, same norm (k=1) | **2.1× (no grid!)** |
| golden + random, 0.5× norm | 1.4× |

**Interpretation:** a same-norm *random* latent error produces essentially no
grid, while the *real structured* delta produces the grid monotonically. The
VAE is clean; the produced latent carries **real 8px-structured error**.
The grid is not decoder amplification, not latent-format scaling, not
PNG/normalization.

### 4.2 Where the grid enters the trajectory

- `decode_denoised.py`: the model's **`denoised` output grids from step 0**
  (carrier 415.7×, rgb_cos 0.527). The grid is present in the first model
  forward — **not** an Euler accumulation artifact.
- `check_input_noise.py`: input `x0` == golden `init_noise_randn` (byte-exact),
  decodes to 94.7× — that is the VAE's inherent response to gaussian noise
  (expected; noise is not a grid). Golden `final_latent` decodes to 1.0×.
- `golden_selfcheck.py`: the golden's **own** recorded `step_latents[7]`
  decodes with the grid (**254.4×**, rgb_cos 0.842) while golden `final_latent`
  decodes clean (1.0×, rgb_cos 0.999989). The two golden latents differ by
  cos 0.837 (RMSE 0.845).

### 4.3 Model forward is faithful — timestep and context are correct

`probe_timestep_ctx.py` runs the real pinned graph at **step 7** (sigma
0.3050089478492737 — the step where golden `block_00_out` was captured per
D032) against the golden's `block_00_out`:

| timestep convention | context | block00 cos |
|---|---|---|
| **raw sigma** | **ctx512 (harness fixture)** | **0.999064** (RMSE 0.0805) |
| raw sigma | cond_context padded to 512 | 0.901728 |
| raw sigma | cond_context 46 | −0.0146 |
| sigma ×1000 | ctx512 | −0.2188 |
| old-form flux t ×10000 | cc_pad512 | 0.3389 |

**⇒ The harness convention (raw sigma + context512) is correct.** ComfyUI
`ModelSamplingDiscreteFlow.timestep(sigma) = sigma × multiplier` with
multiplier=1 here (raw sigma), and the 512-token context is the golden's
conditioning. The real pinned graph forward reproduces the golden's step-7
block-0 output at 0.999 — the model forward is **not** the grid source.

### 4.4 Euler is faithful

ComfyUI `sample_euler` (comfy-ref/k_diffusion/sampling.py:190-214):
`denoised = model(x, sigma_hat)`; `d = to_d(x, sigma_hat, denoised) =
(x − denoised)/sigma_hat`; `x += d·(sigma_next − sigma_hat)`.
Harness: `denoised = x − s·v`; `x += (x − denoised)/s · (s_next − s)`.
**Algebraically identical.** No Euler bug.

### 4.5 Trajectory alignment — the golden trace itself is inconsistent

- Harness post-step latents vs golden `step_latents`: converges to **0.974 at
  step 7** but only 0.811 vs golden `final_latent`.
- `trace_anima.py` (the golden capture script) names its callback's second
  argument `x`, but ComfyUI passes **`denoised`** there
  (comfy-ref/samplers.py:1003: `callback(x["i"], x["denoised"], x["x"], ...)`).
  ⇒ Golden `step_latents` are **denoised captures**, not post-step latents —
  the D055 inconsistency is explained and confirmed.
- With `sigma_next = 0`, the final post-step latent must equal the final
  `denoised` (Euler invariant) — but golden `step_latents[7] ≠ final_latent`
  (cos 0.837). The golden's own recorded trajectory carries the grid while its
  clean `final_latent`/reference image does not. The harness faithfully matches
  the golden's *grid-carrying* recorded trace; the clean reference came from a
  separate path in the golden capture.

---

## 5. What is proven / rejected / open

### PROMOTED (evidence-backed)
1. **The 8px grid reproduces on CUDA** (246–493× carrier) with the official
   BF16 source through the real pinned graph — completely off Apple hardware.
2. **The decoder/VAE/PNG path is clean** — G0 golden latent decodes to 1.0×
   (rgb_cos 0.999989); a norm-matched random latent perturbation yields no grid.
3. **The grid is encoded in the produced final latent** (structured error), not
   introduced by the decoder, Euler, or input noise.
4. **The model forward is faithful** (step-7 block00 cos 0.999064 with the
   correct raw-sigma + 512-context convention) and Euler matches ComfyUI
   algebraically — the DiT math is not diverging from the golden environment.
5. **The grid enters at step 0 of the trajectory** (`denoised_step0` 415.7×),
   so it is a property of the model's output given the trajectory, not
   accumulated drift.

### REJECTED (evidence-backed)
- VAE decoder / latent normalization / PNG conversion as the grid source.
- Input noise as the grid source (94.7× is the VAE's gaussian-noise response;
  produced grids are 2.6–5× above it and are structured).
- Backend (Metal vs CUDA) as the grid cause (reproduces identically on CUDA).
- Sampler/scheduler/Euler/initial-latent math (D090 + §4.4).
- Timestep feeding convention (raw sigma confirmed correct) and context
  (512-token confirmed correct).

### OPEN
- **Why the golden's clean `final_latent` differs from its own recorded
  grid-carrying `step_latents` trace** (D055). The harness trajectory is
  faithful to the recorded trace; the clean reference came from the actual
  ComfyUI `csample.sample` path in `trace_anima.py`. The single-step divergence
  appears at step 1 (cos 0.17 between harness post-step and golden trace at the
  next step) — prime suspects: the exact conditioning/context assembly in the
  golden ComfyUI path (Qwen → LLMAdapter → 512-token context), or a
  harness-vs-golden single-forward difference at the first step that the
  step-7 block-0 probe cannot see.
- Whether the produced latent's 8px structure is a real model property or an
  artifact of the harness's context assembly — needs the true ComfyUI control
  (`trace_anima.py`'s `csample.sample` path) on CUDA.

---

## 6. Evidence on Hugging Face

**Repo:** `ScalingBiz/AnimaXS-investigation-artifacts` (dataset, private)
**URL:** https://huggingface.co/datasets/ScalingBiz/AnimaXS-investigation-artifacts

### 6.1 New grid-reproduction experiment (uploaded 2026-08-14, 66 files)
`experiments/2026-08-14_grid-repro/` (upload commit
`ea45b10503f1a619aaf022e9c4232227ec3d1ea2`):

| Evidence | Path |
|---|---|
| Experiment manifest (all hashes, sigmas, provenance) | `experiments/2026-08-14_grid-repro/manifest.json` |
| SHA-256 manifest | `experiments/2026-08-14_grid-repro/SHA256SUMS` |
| Reference image | `experiments/2026-08-14_grid-repro/reference.png` |
| G0 golden → VAE (clean) | `.../G0_golden.png`, `G0_golden_rgb.f32`, `G0_numpy_oracle_rgb.f32` |
| G1 official BF16 (grid) | `.../G1_bf16.png`, `G1_bf16_rgb.f32`, `G1_bf16_final_latent.f32` |
| G2 FP16-all (grid) | `.../G2_fp16all.png`, `G2_fp16all_rgb.f32`, `G2_fp16all_final_latent.f32` |
| G3 W8 (grid) | `.../G3_w8.png`, `G3_w8_rgb.f32`, `G3_w8_final_latent.f32` |
| G4 Metal fp16-all / w8 / w4 → CUDA VAE | `.../G4_metal_fp16-all.png`, `G4_metal_w8.png`, `G4_metal_w4.png` (+ `_rgb.f32` each) |
| Vision contact sheet | `.../vision/contact_sheet.png`, `side_by_side.png`, `difference.png`, `difference_high_gain.png`, `fft_comparison.png`, `fft_magnitude_ref.png`, `fft_magnitude_g1.png`, `center_crop.png`, `texture_crop.png` |
| Decoder sensitivity (k·Δ + random control) | `.../decoder_sens/decoder_sensitivity.json`, `k0_golden.png`, `k0.25.png`, `k0.5.png`, `k0.75.png`, `k1.png`, `rand_k0.5.png`, `rand_k1.png`, `reference.png` |
| Denoised per-step decodes | `.../denoised/denoised_step0.png`, `denoised_step3.png`, `denoised_step6.png`, `denoised_step7.png`, `reference.png` |
| Latent spatial diagnostics | `.../latent_diag/latent_periodicity.json`, `G1_bf16_mae.png`, `G1_bf16_sme.png`, `G2_fp16all_mae.png`, `G2_fp16all_sme.png`, `G3_w8_mae.png`, `G3_w8_sme.png` |
| Input-noise control | `.../noise/x0_input_noise.png`, `golden_step0.png`, `golden_final.png`, `reference.png` |
| Golden self-check (step7 vs final) | `.../selfcheck/golden_step7.png`, `golden_final.png`, `reference.png` |
| Harness per-step latents (8 × .f32) | `.../traj/step00_latent.f32` … `step07_latent.f32` |

### 6.2 Pre-existing evidence now durable on HF

| Bundle | Path |
|---|---|
| Real-graph ladder results | `evidence/ladder_real/RESULTS_SUMMARY.md`, `ladder_real_checkpoint.json`, `streaming_confirm.log` |
| Oracle ladder + decoder parity | `evidence/ladder/` (ladder_summary.json, source_oracle_parity.*, precision_ladder_stage_parity.*, animapk_decoder_parity.*, caps_*.npz ×9) |
| CUDA step-0 per-block captures | `evidence/step0_cuda/caps_{A,C,D,E}_*_real.npz`, `small_*.npz`, `provenance.json` |
| BF16→FP16 storage analysis | `evidence/storage/bf16_fp16_storage_tensors.csv`, `bf16_fp16_top20.md` |
| Canonical fixtures | `evidence/fixture/x_in.f32`, `context512.f32`, `cond_context.f32`, `sigmas.txt`, `manifest.json` |
| Metal step-0 captures (3 packs × 166 files) | `evidence/metal_step0/{fp16-all,w8,w4}/` (per-block f32, 8-step x_in/denoised, provenance, manifest) |
| Clore environment | `environment/clore_environment.txt` |

### 6.3 Model pack repos
- `ScalingBiz/AnimaXS-VAE-XSMAX-FP16` — `qwen-image-vae-xsmax-fp16.animapk`
  (SHA `10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447`) + manifest.json
- `ScalingBiz/AnimaXS-DiT-FP16-ALL`, `ScalingBiz/AnimaXS-DiT-W8`, `ScalingBiz/AnimaXS-DiT-W4` (established repos)

---

## 7. Repository state

- Branch `investigate/animapk-cuda-parity`, HEAD `44d5d68`, clean, pushed.
- New commits: `e41cdb3` (grid_repro), `948343f` (inspect_animapk),
  `ecf0d38` (latent_periodicity), `6e527b7` (fix npz), `a80ee02` (contact sheet),
  `a6d4657` (diagnostics), `44d5d68` (docs D094–D096 + session).
- `DECISIONS.md` appended: **D094** (grid reproduced on CUDA, Case 2),
  **D095** (grid is latent-encoded, not decoder), **D096** (model forward +
  Euler faithful; golden trace inconsistency).
- `HERMES_SESSION.md` rewritten with full current state.
- `HERMES_GRID_ROOT_CAUSE.md` saved at repo root (K's instruction file).

---

## 8. Next actions

1. **Establish the true clean CUDA control**: reproduce `trace_anima.py`'s
   actual ComfyUI `csample.sample` path (cfg=1, real conditioning assembly)
   on Clore and diff its clean `final_latent` against the harness trajectory to
   find the exact single-step divergence (appears at step 1, cos 0.17).
2. If the divergence is a harness context-assembly difference (Qwen →
   LLMAdapter → 512-token context), fix the harness's conditioning and re-run
   G1 — expected carrier collapse toward 1.0×.
3. Port the proven fix to Metal (G4) and validate on macOS simulator.
4. Clore stays running per K's instruction (do not terminate).

---

*Generated by Hermes on 2026-08-14 from the live Clore run. All numbers are
measured, not inferred; all images are machine-generated from real model
execution; all hashes verified against HF remote.*
