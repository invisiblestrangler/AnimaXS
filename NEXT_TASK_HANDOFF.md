# AnimaXS — Next-Task Handoff (fresh agent starting point)

You are taking over the AnimaXS iOS inference project. **Do not** start a new implementation
and do **not** rewrite the runbook. A large amount of verified work already exists. Your job:

1. reload the durable context (below);
2. implement the **next unchecked TODO tasks (H001 and H002 — DiT input + timestep embedder)**;
3. validate numerically against the pinned source the same way the encoder/adapter were validated;
4. update durable state, commit, get green CI.

---

## 0. NON-NEGOTIABLE CONTEXT RELOAD (do this FIRST, in order)

```bash
cd /root/AnimaXS
```

Read these durable files **in order**:

```
/root/AnimaXS/RUNBOOK.md      # the full execution runbook (source of truth for order)
/root/AnimaXS/STATUS.md       # current milestone + next three tasks
/root/AnimaXS/TODO.md         # the checklist (H001/H002 are the next unchecked items)
/root/AnimaXS/DECISIONS.md    # every non-obvious choice + source-cited facts (D001–D022)
/root/AnimaXS/TEST_MATRIX.md  # test tracking
/root/AnimaXS/docs/QWEN_ENCODER_DEBUG.md  # encoder debugging + resolution (pattern to follow)
/root/AnimaXS/docs/PROGRESS_REPORT_FOR_ADVANCED_AI.md  # the prior full progress report
```

Then the **handoff bundle** (authoritative model facts):

```
/root/anima-xsmax/PHASE0_2_HANDOFF/HANDOFF.md
/root/anima-xsmax/PHASE0_2_HANDOFF/MODEL_ARCHITECTURE.json
/root/anima-xsmax/PHASE0_2_HANDOFF/RUNTIME_CONSTANTS.json
/root/anima-xsmax/PHASE0_2_HANDOFF/NUMERICS.md
/root/anima-xsmax/PHASE0_2_HANDOFF/GOLDENS.md
/root/anima-xsmax/PHASE0_2_HANDOFF/KNOWN_ISSUES.md
/root/anima-xsmax/PHASE0_2_HANDOFF/IMPLEMENTATION_RECOMMENDATIONS.md
```

Then the **pinned reference source** (commit `cbbc9da`, at `/root/comfy-ref`):
```
/root/comfy-ref/comfy/ldm/cosmos/predict2.py   # MiniTrainDIT (the DiT) — READ FOR H001–H007
/root/comfy-ref/comfy/ldm/anima/model.py        # Anima wrapper + LLMAdapter (already done)
/root/comfy-ref/comfy/ldm/modules/attention.py  # attention (for H005)
/root/comfy-ref/comfy/ops.py                     # Linear/RMSNorm/embedding ops + repeat_kv_for_gqa
```

Then the **actual current code** (to mirror its structure):
```
/root/AnimaXS/AnimaXS/Runtime/Text/QwenEncoderCPU.swift
/root/AnimaXS/AnimaXS/Runtime/Text/LLMAdapter.swift
/root/AnimaXS/AnimaXS/Runtime/Text/QwenNumerics.swift
/root/AnimaXS/AnimaXS/Runtime/Animapk/         # parser + QuantDecoders
```

**Do not rely on this handoff alone.** The durable records + pinned source are authoritative.

---

## 1. VERIFY REPO STATE

```bash
cd /root/AnimaXS
git status --short && git branch --show-current && git log --oneline -8 && git rev-parse HEAD
```

Current `main` should be at `a02582d` (last green: CI run 31423768681, all 3 jobs GREEN).
Worktree must be clean. Do NOT reset to an older commit.

If you see a CI failing `project-consistency` after adding files: the committed
`AnimaXS.xcodeproj` must be regenerated on a macOS runner. Pattern used successfully:
1. `git push` your source changes
2. `gh workflow run bootstrap-project.yml --ref main` (regenerates + commits the xcodeproj)
3. `gh workflow run ci.yml --ref main` (bootstrap's GITHUB_TOKEN push does NOT auto-trigger CI)
Always confirm `gh auth status` works before relying on this.

---

## 2. PROJECT STATE — WHAT IS DONE AND VALIDATED

Everything below is **verified**, do not redo it unless a regression test fails:

- **CI fully green**: Xcode 26.3 ARM device build (Metal shaders + swift-transformers), simulator tests pass.
- **ANMA v1 parser**: JSON-authoritative, CRC-32, alignment, ranges — validated vs all 3 real packs.
- **CPU W4/W8 decoders**: known vectors byte-exact vs HANDOFF.md. (`QuantDecoders.dequantW4/dequantW8/fp16ToFloat32`)
- **Tokenizer parity CI-verified**: Qwen (exact regex) + T5 (Unigram) byte-exact on 3 canonical prompts.
- **Qwen encoder full-28 parity RESOLVED**: Swift W8 == pinned-Comfy oracle cosine **1.000000**; vs golden `cond_context` cosine **0.992164** (pure W8 quantization). Two bugs fixed: GQA `kvHead=qHead/2` (not `%8`), and final RMSNorm applied to cond_context.
- **LLMAdapter (G001/G002) DONE + validated**: Swift == pinned-Comfy oracle cosine **1.000000** → produces `[1,512,1024]` conditioning.

### The two oracle scripts (REUSE THIS PATTERN — it is how correctness is proven)
```
/root/AnimaXS/scripts/qwen_comfy_oracle.py    # Qwen encoder oracle (W8 TE pack)
/root/AnimaXS/scripts/anima_adapter_oracle.py # LLMAdapter oracle (W4 DiT pack, llm_adapter.* tensors)
```
Both dequantize the **same weights the Swift code consumes** from the real `.animapk` packs
(via `inspect_animapk.Animapk`), transcribe the **pinned ComfyUI equations verbatim with line
citations**, and compare Swift vs oracle (structural parity) separately from vs golden
(quantization). Fixtures saved to `scripts/oracle_out/`.

**For every new DiT subsystem, build the analogous oracle.** Do NOT hand-write equations from
memory. Transcribe from `/root/comfy-ref` and cite lines.

---

## 3. THE NEXT TASKS: H001 and H002 (implement these)

From `TODO.md`:

- [ ] **H001** — DiT input: 17-ch (16 latent + padding mask), patchify 2×2 → 1024 tokens ×68, input proj → 2048 fp32 residual.
  - deps: E004, E006 · output: `DiTInput.swift` · validation: shape/token count exact
- [ ] **H002** — Timestep: sigma-based sinusoidal embed dim 2048 base 10000 + model RMSNorm/MLP path; full fp32.
  - deps: E005 · output: `TimestepEmbedder.swift` + CPU reference test · validation: CPU unit test matches

Read **RUNBOOK.md** sections for DiT input and timestep (search `# 29. DiT input` and the
timestep section), plus `MODEL_ARCHITECTURE.json` "dit" block and `comfy/ldm/cosmos/predict2.py`.

### Key DiT facts already in the handoff (re-verify in the pinned source before coding)
```
model: Anima-Turbo v1.0, type FLOW
blocks: 28, hidden 2048, heads 16, head_dim 128, MLP 2048→8192→2048 GELU
latent channels: 16, +padding-mask concat = 17 effective input channels
patch: 2×2 spatial, T=1  →  512 image → latent 64×64 → 32×32 = 1024 DiT tokens × 68 input dim
cross context: 512 × 1024  (produced by LLMAdapter)
residual stream = Float32 (activations reach ~261,000 → fp16 overflow; MUST be fp32)
DiT pack: W4 group=64 + fp16 exclusions; physical order is string-sorted, use block ranges via
   prefix `model.diffusion_model.blocks.N.` (D012); each block ≈ 38,993,920 stored bytes
```

**CRITICAL pitfalls from prior work (avoid repeating):**
- Never guess model math — transcribe from `/root/comfy-ref` and cite lines.
- Two implementations agreeing doesn't prove correctness if both share a wrong formula.
  Build the pinned-source oracle as the independent A/B/C control.
- `QuantDecoders` are Data-based; scope `withUnsafeBytes` internally (D017). Don't pass
  `baseAddress` across closures.
- DiT blocks located by metadata/prefix, never `blockOffset = first + index*size` (D012).
- TE/DiT tensors are string-sorted physically; use the JSON metadata full names (binary table
  truncates names to 64 chars and shapes to 4 dims).
- Norm vectors are stored **fp16** in the packs; projection/embedding matrices are quantized
  (Qwen W8, DiT W4). Dispatch on `t.storage == .fp16`.

---

## 4. THE PROVEN WORKFLOW (use it for H001/H002 and everything after)

At every subsystem transition:

```
1. Read STATUS.md.
2. Read next unchecked TODO items.
3. Read relevant RUNBOOK section.
4. Read relevant handoff section.
5. Read actual current source implementation (QwenEncoderCPU.swift/LLMAdapter.swift patterns).
6. Read pinned Comfy/reference code for the subsystem (predict2.py for DiT).
7. State the exact validation gate before coding.
8. Implement the smallest correct unit.
9. Validate numerically (build a pinned-source oracle; compare Swift vs oracle cosine ≥ ~1.0,
   and vs golden where available).
10. Update durable files (TODO/STATUS/DECISIONS/TEST_MATRIX).
11. Commit and push.
12. Verify CI (regenerate xcodeproj via bootstrap-project if files were added).
```

If context is compressed or you are unsure of a model detail: **STOP CODING AND RE-READ the
durable files.** Never fill a forgotten model detail from intuition.

---

## 5. ENVIRONMENT / ASSETS (all verified)

| Item | Value |
|---|---|
| Target | iPhone XS Max, A12, 4 GB, iOS 18.6, Metal Apple5 |
| Packs (SHA-256 verified) | DiT W4 `anima-turbo-v1.0-xsmax-w4.animapk` (`ba1ce6…0d25`, 1,179,435,008 B); TE W8 `qwen3-0.6b-xsmax-w8.animapk` (`ba59e4…ceab`); VAE fp16 `qwen-image-vae-xsmax-fp16.animapk` (`10171a…c447`) |
| Golden | `case1_danbooru_seed1337.npz` at `/root/anima-xsmax/results/goldens/` (`cond_context`, `block_00_out..27`, `step_latents`, `final_latent`, `decoded_rgb`, `init_noise_randn`, `sigmas_comfy`, T5 ids/weights) |
| Handoff | `/root/anima-xsmax/PHASE0_2_HANDOFF/` |
| Pinned ref | `/root/comfy-ref/` — commit `cbbc9da` (do NOT pull/update) |
| GitHub | `invisiblestrangler/AnimaXS`; PAT at `/root/GITHUB_PAT_ANIMAXS`; `gh` authenticated |
| Swift (local Linux) | `/opt/swift/usr/bin/swift` — used to build the Linux harness (`/root/anima-harness`) against the real packs |
| Python | `/root/anima-xsmax/.venv/bin/python` (torch 2.13.0+cpu, numpy) |

### Harness pattern
The Linux Swift harness at `/root/anima-harness/` (Package.swift → `Sources/harness/`) runs the
real packs (parser, decoders, Qwen, LLMAdapter). Add new DiT modules there to validate against
the oracle, then port to the app's `AnimaXS/Runtime/...`. `main.swift` is the Qwen driver
(restore it before finishing; back it up when you swap it). Note: the harness `AnimapkFile` has
the same `tensor(named:)`/`dataBytes().data` API as the app.

### Python oracle pattern
`inspect_animapk.py` (in `/root/anima-xsmax/scripts/`) dequantizes any `.animapk` tensor to
float32 (`Animapk(path).read_table()` → match by `blob_offset` to JSON `tensor_meta` for full
names → `decode(rec)`). Reuse `AdapterWeights`/`PackWeights` loader style from the two oracle
scripts.

---

## 6. REPORTING BACK

When H001/H002 are done, report:

```
Repo HEAD:
CI run + status (all 3 jobs):
H001 — DiT input: shape (17-ch, 1024 tokens, 68) exact? input_proj → 2048 residual cosine vs oracle?
H002 — timestep: sigma→sinusoidal embed dim 2048, RMSNorm/MLP path; CPU test matches oracle?
Pinned ComfyUI commit:
DiT pack SHA256:
Swift vs oracle (structural) cosine/rmse/maxAbs:
Next unchecked RUNBOOK task:
```

---

## 7. HARD DON'TS

Do not:
- rewrite the project or the runbook;
- update the pinned ComfyUI reference (commit cbbc9da is frozen);
- make multiple speculative math changes in one patch;
- skip full-tensor comparisons (shape/allFinite/maxAbs/RMSE/cosine);
- trust the old hand-written Python layer-0 reference (obsolete, moved to
  `/root/anima-harness/obsolete_shared_reference/`);
- fill forgotten model details from intuition;
- use fp16 for the DiT residual stream (must be Float32);
- move to H003+ before H001/H002 gates pass;
- optimize performance during correctness work.

Correctness and a reproducible oracle are the only priorities until the DiT is proven.
