# AnimaXS — Phase 3 Progress & Debug Report for Continuation AI

**Author:** current implementation agent
**Date:** 2026-08-10
**Purpose:** Hand off precise state to a more advanced AI that will guide the next steps. All facts are from the live repo, the real model packs, CI logs, and measured runs — **nothing fabricated**.

---

## 1. TL;DR

- **A green, CI-verified foundation exists**: repo `invisiblestrangler/AnimaXS` (public), Xcode 26.3 ARM device build passes, 21 simulator unit tests pass, parser + W4/W8 decoders + **Qwen/T5 tokenizer parity** all validated byte-exact against the real model packs and golden references.
- **The current blocker** is the **Qwen3-0.6B text encoder full-28-layer parity**: layer-0 numerics are exact (cosine 1.0 vs an independent Python reference), but the full forward does **not** match the golden `cond_context` (cosine ≈ −0.04). Both Swift and a hand-written Python reference produce the **identical** wrong output — so this is a **shared misunderstanding of one reference detail**, not a Swift typo.
- Everything downstream (LLM adapter, DiT, sampler, VAE, model download UX, release) is **not yet implemented** and is blocked behind encoder parity.
- **A12 DEVICE TESTED: NO.** Nothing on CI counts as A12 validation.

---

## 2. Environment & Assets (all verified)

| Item | Value |
|---|---|
| Device target | iPhone XS Max, A12, 4 GB RAM, iOS 18.6, Metal family Apple5 (runbook §0) |
| Model packs (SHA-256 verified exact) | DiT W4 `anima-turbo-v1.0-xsmax-w4.animapk` (1,179,435,008 B, `ba1ce6…0d25`); TE W8 `qwen3-0.6b-xsmax-w8.animapk` (635,305,984 B, `ba59e4…ceab`); VAE fp16 `qwen-image-vae-xsmax-fp16.animapk` (256,163,840 B, `10171a…c447`) |
| Golden | `case1_danbooru_seed1337.npz` (118,302,516 B, `44d35d…a8dc`) |
| Handoff | `/root/anima-xsmax/PHASE0_2_HANDOFF/` (HANDOFF.md, MODEL_ARCHITECTURE.json, RUNTIME_CONSTANTS.json, NUMERICS.md, GOLDENS.md, IMPLEMENTATION_RECOMMENDATIONS.md, etc.) |
| Reference source (pinned) | `/root/comfy-ref/comfy/` — ComfyUI commit `cbbc9da…`; text encoder at `comfy/text_encoders/llama.py`, `anima.py` |
| GitHub | `invisiblestrangler/AnimaXS` (PAT at `/root/GITHUB_PAT_ANIMAXS`, repo URL on line 2) |
| Local Swift toolchain | `/opt/swift` (Swift 6.1.3, Ubuntu 22.04) — used to build a Linux harness that runs the **real packs** |
| Local Python | `/root/anima-xsmax/.venv/bin/python` (torch 2.13.0+cpu, transformers, tokenizers, numpy) |

---

## 3. What has been ACCOMPLISHED (exact)

### 3.1 CI / build system (runbook §3, §4, §5, §9, §10, §11)
- **All 3 CI jobs GREEN** on `main` (latest `a414159`, run 31411533185):
  - `project-consistency` — regenerates `AnimaXS.xcodeproj` from `project.yml` via XcodeGen; `git diff --exit-code`.
  - `iphone-build` — `xcodebuild … -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build` with Xcode 26.3. **Passes** (includes Metal shader compilation + swift-transformers/Tokenizers).
  - `simulator-tests` — discovers an iPhone simulator dynamically; runs 21 tests, **3 skipped** (RealPackDecoderTests gate on `ANIMAXS_PACKS_DIR`), **0 failures**.
- Verified external facts (§4): Xcode 26.3 (17C529) at `/Applications/Xcode_26.3.app` on `macos-15`; default is 16.4 (16F6) so CI sets `DEVELOPER_DIR` explicitly; XcodeGen NOT preinstalled → installed via Homebrew; swift-transformers latest = **1.3.3** (2026-05-16).
- Deployment target **iOS 18.0** set on targets (not global export), builds against iOS 26.2 SDK (runbook §5, §52). Swift 5 language mode (D002).
- `bootstrap-project.yml` regenerates+commits the xcodeproj; CI's `project-consistency` enforces sync.

### 3.2 ANMA v1 parser (runbook §17, §2)
Swift `MappedFile` (mmap), `AnimapkHeader`, `AnimapkTensor`, `AnimapkFile`, `Crc32`. **Validated against the real packs** via the Linux harness: all 1,189 tensors across 3 packs CRC-32 **0 mismatches**, every `blob_offset % 16384 == 0`, declared file_size == actual, section/blob ranges in-bounds. JSON `tensor_meta` is authoritative for names/shapes (binary table truncates to 64-char names / 4-dim shapes).

### 3.3 W4/W8 CPU decoders (runbook §18)
Data-based `QuantDecoders` (scopes `withUnsafeBytes` internally — see D017 pitfall). W4 nibble order (even K→low nibble, odd→high), group 64, fp16 scale/zero. Known vectors from HANDOFF.md §12–§13 reproduced byte-exact:
- W4 block0 `mlp.layer1.weight` first-8 dequantized = `[0.00304, −0.00059, −0.00014, 0.00032, −0.00104, −0.00150, −0.00014, −0.00195]`
- W8 `embed_tokens.weight` first-8 = `[−0.00342, 0.03285, −0.07002, −0.01990, −0.00540, −0.01727, −0.03046, 0.00516]`

### 3.4 Tokenizer parity (runbook §16) — CI-VERIFIED
- **D014 (measured, goldens case1/2/3 seed1337):** Qwen = `encode(prompt, add_special_tokens=False)`, **no** start/end/trailing token → sequence lengths 46/72/122 (matches `cond_context` shapes exactly). **T5 = `encode(prompt, no_specials) + [1]`** (trailing `</s>`, id 1) → 47/85/134 (matches golden `cond_meta_t5xxl_ids`).
- Bundled `qwen_tokenizer.json` (11.4 MB) generated from reference `vocab.json`+`merges.txt` with the **exact Qwen2Tokenizer pre-tokenization regex**:
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  (Split isolated + ByteLevel(use_regex=False)). This regex is the **only** thing that reproduces the reference hyphen handling (`fur-trimmed` stays one pre-token). Verified byte-exact on all 3 prompts **plus** the tricky hyphen/URL case3.
- `t5_tokenizer.json` (2.4 MB) is the reference Unigram tokenizer; loaded via swift-transformers `T5Tokenizer`→`UnigramTokenizer`.
- `TokenizerLoader` reads flat unique-named files via `AutoTokenizer.from(tokenizerConfig:tokenizerData:)` (requires `import Hub` for `Config`, and `Config(String-var)` explicit wrapping — string *literals* auto-convert but **string variables do not**; `Config(variable)` needed).

### 3.5 Qwen encoder (runbook §27) — PARTIALLY, layer-0 exact
`QwenNumerics.swift` + `QwenEncoderCPU.swift` (pure Foundation, compiles in CI device build). **Layer-0 numerics verified EXACT** (cosine 1.0, maxAbs 4.7e-6) against an independent Python reference built from the same dequantized W8 weights (files: `/root/anima-harness/layer0_input_emb.npy`, `layer0_h1.npy`, `layer0_ref_out.npy`).

### 3.6 Metal skeleton (runbook §21, §22)
`MetalContext.swift` (device/queue/library/pipeline cache + Apple5 probe) + `AnimaKernels.metal` (dequant_w4/w8_to_half, rmsnorm, gelu, silu, gate_add, add_half_into_float, euler_step_f32, patchify17) — **compile in the device build**. Not yet runtime-tested (no Metal on this box).

---

## 4. What DIFFERED from the runbook (so far)

1. **Runner has no reference-tests job anymore** (runbook §11 job 4). I folded pure-reference tests into `simulator-tests` because compiling the huge swift-transformers SPM graph (swift-crypto/nio/atomics, ~15–25 min cold on 3-core M1) per-job caused CI hangs. Documented in `ci.yml`.
2. **Sections are NOT all 16 KB aligned** (runbook §17/ANIMAPK_SPEC §3 claim). Real packs: `json_offset=256`, `table_offset=327936` — only `payload_offset` and every `blob_offset` are 16 KB-aligned. Parser enforces **blob** alignment only (D008).
3. **JSON `component` is a string** (`"dit"`/`"te"`/`"vae"`), not the integer 1/2/3 (that's only in the binary header) (D009).
4. **TE layers are string-sorted physically** (`0,1,10..19,2,20..27,3..9`) — HANDOFF.md's "layers 0–27 in order" is **misleading**; each layer is a contiguous 16,777,216 B range but located by metadata, never arithmetic (D010).
5. **Tokenizer asset layout**: used flat unique filenames (`qwen_tokenizer.json`, `t5_tokenizer.json`, `t5_tokenizer_config.json`, `t5_special_tokens_map.json`) instead of `qwen/`+`t5/` subfolders. Xcode's resource build phase flattens by filename → two `tokenizer_config.json` collided ("Multiple commands produce"). XcodeGen `type: folder` did NOT produce a folder reference as expected; flat unique names are the robust fix.
6. **Qwen q_proj is [2048, 1024]** (16 heads × 128 = 2048 out), **k/v_proj [1024, 1024]** (8 KV × 128). RUNTIME_CONSTANTS says "head_dim 128" which is consistent once you account for GQA output dims — but a naive "q_proj [hidden,hidden]" assumption is wrong.
7. **Norm vectors are stored fp16 in the pack** (not W8), while projection/embedding matrices are W8. Encoder must dispatch on `t.storage == .fp16` (D011-related). NUMERICS.md §4 lists 353 fp16 rank≤1 tensors.
8. **RoPE is half-split rotate**, not GPT-NeoX interleaved pairs (see §5.1 below — this was corrected and layer-0 became exact).
9. **`bootstrap-project.yml`'s GITHUB_TOKEN push does not re-trigger CI** (GitHub's no-recursion rule). Added `workflow_dispatch` to `ci.yml` so CI can be run on demand after a bootstrap.
10. **XcodeGen `info:` generation** left the repo without a committed `Info.plist` (only the runner had it), breaking clean clones → switched to a committed `AnimaXS/Info.plist` + `INFOPLIST_FILE`.

---

## 5. CURRENT ISSUE — Qwen encoder full-28 parity (the blocker)

### 5.1 What is verified correct (layer 0, Swift == Python, both vs the same weights)
- Embedding row-gather from W8 (row stride 1024 data bytes, **32-byte** scale/zero stride per row = 16 fp16 groups).
- RMSNorm (fp32 accumulation, eps 1e-6).
- GQA: q_proj [2048,1024] → 16 heads × 128; k/v_proj [1024,1024] → 8 KV × 128; o_proj [1024,2048]. Q head `h` reads KV head `h % 8`.
- gemma3 per-head Q/K RMSNorm over each 128-dim head, `rms_norm_add=False` → **no** `weight+1`.
- **Half-split RoPE** (corrected from GPT-NeoX): `inv_freq[j] = 1/theta^((2j)/128)`, `freqs = pos*inv_freq`, `cos/sin` over both halves, `out[:64]=a·cos−b·sin`, `out[64:]=b·cos+a·sin`. theta=1e6. Applied per head-chunk; positions repeated per head (seq×16 for Q, seq×8 for K).
- Causal mask `triu(1)`.
- Gated SiLU MLP (gate/up [3072,1024], down [1024,3072]).
- Residual order: `x = x + attn(x)`; `x = x + mlp(x)`.
- fp16-vs-W8 storage dispatch.

Layer-0 output cosine 1.0 (maxAbs 4.7e-6) vs my Python reference. **But both are wrong vs the golden** — the reference (real ComfyUI run) must differ in one detail I haven't identified.

### 5.2 Symptoms
- Full 28-layer forward: `out first6 = [104.6, −80.1, 57.6, −108.2, 37.9, 198.2]`, cosine vs golden `cond_context` first6 `[2.01, 25.06, 0.03, −17.74, 1.36, −7.19]` ≈ **−0.04**, maxAbs ~2951. Both Swift and Python give **identical** wrong output (so it's a shared formula issue, not code).
- Magnitude grows to ~100–200 while golden stays ~2–25 → the residual/norm chain likely has a structural detail wrong that only shows after several layers (a per-layer small difference compounding).

### 5.3 Reference code examined (comfy/text_encoders/llama.py + comfy/ldm/modules/attention.py)
Key facts I confirmed from source:
- `Qwen3_06BConfig`: hidden 1024, heads 16, kv_heads 8, head_dim 128, intermediate 3072, rms_norm_eps 1e-6, rope_theta 1e6, `rms_norm_add = False`, `mlp_activation = "silu"`, `qkv_bias = False`, `final_norm: True` (config), but handoff says `layer_norm_hidden_state=False` → final norm NOT applied to `cond_context`.
- `BaseLlama.forward`: `position_ids = arange(0, seq_len)` (no shift); **causal mask** `= fill(finfo.min/4).triu_(1)` when `seq_len > 1`; `optimized_attention_for_device(x.device, mask=..., small_input=True)` → for small input returns `attention_pytorch` (if pytorch attention enabled) or `attention_basic`.
- `Attention.forward`: q_proj/k_proj/v_proj then reshape to `[batch, seq, heads, head_dim]`, `xq=self.q_norm(xq)`, `xk=self.k_norm(xk)`, `apply_rope(xq, xk, freqs_cis)`, then `optimized_attention(..., skip_reshape=True, enable_gqa=True)`.
- `attention_pytorch` uses `comfy.ops.scaled_dot_product_attention(..., scale=dim_head**-0.5, enable_gqa=True)`. Scale = `1/sqrt(128)` — matches mine.
- `attention_basic` (line ~167): `enable_gqa → repeat_kv_for_gqa(k, v, heads)`; scale `dim_head ** -0.5`.

### 5.4 Prime suspects for the shared bug (ranked)
1. **The attention mask value / semantics.** Reference uses `mask = 1.0 − attention_mask` reshaped to `[b,1,seq_len,kv]`, then `masked_fill(mask.to(bool), finfo.min/4)`, plus causal `triu_(1)` of `finfo.min/4`. My implementation uses plain `−inf` masking in softmax. With fp16/bf16, `finfo.min/4` is a **finite large-negative** number, and softmax over it still → ~0. This *shouldn't* matter in fp32, but if the reference computed in bf16, extreme values could differ. **However** — the bigger question is whether `attention_mask` from the clip path is all-ones (so `1−1=0`, mask=0) and the causal part is the only real mask. My causal mask is `j <= position` attend. Confirm the mask is applied as an **additive** bias (not a boolean select), and that masked positions use `-inf` in fp32 which my softmax handles.
2. **Whether the golden `cond_context` is the true raw last-layer hidden, or whether there's a post-processing step** (e.g., the `sd1_clip` path or a projection) I'm missing. The handoff explicitly says it's the last-layer hidden without final norm, but I should double-check against how ComfyUI captures `cond_context` for this model.
3. **RoPE frequency exponent sign / arrangement.** I verified `inv_freq = 1/(theta^((2j)/128))` and half-split. But re-verify against `precompute_freqs_cis` exactly: `theta_numerator = arange(0, head_dim, 2)`, `inv_freq = 1/(theta^(theta_numerator/head_dim))`, `emb = cat((freqs, freqs))`, and `apply_rope` uses `cos`, `sin[:64]`, `−sin[64:]`. My formula: `out[:64]=x[:64]·cos − x[64:]·sin`, `out[64:]=x[64:]·cos + x[:64]·sin`, where `sin = sin(freqs)`. Double-check the sign of the second term against `nsin = −sin[64:]`.
4. **GQA head mapping.** `enable_gqa=True` in `scaled_dot_product_attention` — torch's GQA repeats KV heads. I assume Q head `h` reads KV head `h % 8`. Confirm the reference maps `h → h // (16/8) = h // 2` vs `h % 8` — **this is a likely bug**: some GQA implementations map `q_head h → kv_head floor(h * kv_heads / heads) = h // 2`, NOT `h % 8`. If the reference uses `h // 2` and I use `h % 8`, the head pairing is wrong and would produce exactly this kind of scrambled-but-structured output. **This is the #1 suspect.**
5. **`attention_basic`/`attention_pytorch` reshape**: `_reshape_qkv_to_heads` with `skip_reshape=True` means Q/K/V are already `[batch, heads, seq, head_dim]`. My implementation iterates q_len and head correctly, but the **KV length** in my per-token loop is `seq` (full), and for `attention_basic` the softmax is over the full KV with the causal mask — equivalent to my causal mask. Confirm no KV caching/shift.

### 5.5 Decisive next debugging step (recommended)
1. **Test GQA mapping variants first**: try `kvHead = h // 2` instead of `h % 8` (and `h / (heads//kv_heads)`). This is the single most likely fix given "structured but scrambled" output. Re-run the layer-0-against-Python comparison and then the full-28 vs golden.
2. If that doesn't fix it, **compare per-layer hidden states against a true torch reference**: load the real Qwen3-0.6B weights (or dequantized-from-pack) into PyTorch, run the actual `Qwen3_06B` from `comfy/text_encoders/llama.py` on the case1 prompt in **fp32**, and capture layer-0 and final hidden. Compare to my Python reference layer-by-layer to find the FIRST divergence. This isolates the exact wrong operation.
3. Check `optimized_attention`'s actual scale and the `repeat_kv_for_gqa` implementation in `comfy/ops`.

---

## 6. Downstream work NOT yet started (blocked behind encoder parity)

Per runbook §53 order: LLM adapter (§28), DiT timestep/RoPE/modulation (§30–§31), DiT block 0 (§32), 28-block loop (§33), Euler sampler (§34), VAE 3D→2D fold validation (§36), full-frame VAE (§37), RGB (§39), model download UX (§40), checkpoint/background/memory (§43–§44), full-inference CI attempt (§49), docs/release (§47, §51, §52).

---

## 7. Repo / CI / test state (exact)

- **Repo:** `https://github.com/invisiblestrangler/AnimaXS` — 32 commits on `main`, latest `a414159`.
- **CI:** run 31411533185 on `a414159` = **GREEN** (all 3 jobs). Prior run 31410534456 on `d2c9dc4` also green. Run 31409987044 failed only because new encoder files weren't yet in the regenerated xcodeproj (fixed by bootstrap `d2c9dc4`).
- **Tests:** 21 executed, 0 failures, 3 skipped (RealPackDecoderTests need `ANIMAXS_PACKS_DIR`). Suites: AnimapkParsingTests, QuantDecoderTests, TokenizerParityTests, SmokeTests, ReferenceTestsPlaceholder.
- **To run the real-pack tests in CI** (definitive parser/decoder check on-device): set `ANIMAXS_PACKS_DIR` env and ensure the 3 packs are present. Not done on push (2 GB download, runbook §11/§49).
- **Metal in CI:** `MTLCreateSystemDefaultDevice()` not yet probed in a test; Metal shaders compile but no runtime Metal test exists yet.

---

## 8. Files that matter (paths)

| Path | Purpose |
|---|---|
| `/root/AnimaXS/RUNBOOK.md`, `TODO.md`, `STATUS.md`, `DECISIONS.md`, `TEST_MATRIX.md`, `DEVICE_TESTS.md` | persistent context (re-read before resuming) |
| `/root/AnimaXS/docs/QWEN_ENCODER_DEBUG.md` | encoder debugging notes |
| `/root/AnimaXS/AnimaXS/Runtime/Text/QwenNumerics.swift` | RMSNorm, gemma3 head norm, half-split RoPE, gated SiLU, attention |
| `/root/AnimaXS/AnimaXS/Runtime/Text/QwenEncoderCPU.swift` | 28-layer CPU reference encoder |
| `/root/AnimaXS/AnimaXS/Runtime/Text/TokenizerLoader.swift` | loads bundled tokenizers via swift-transformers |
| `/root/AnimaXS/AnimaXS/Runtime/Animapk/*` | parser + QuantDecoders |
| `/root/AnimaXS/AnimaXS/Runtime/Metal/*`, `Shaders/AnimaKernels.metal` | Metal skeleton |
| `/root/anima-harness/` | Linux Swift harness that runs the **real packs** (parser/decoder/encoder validation); source in `Sources/harness/`, refs `cond_context_case1.f32`, `qwen_ids_case1.json`, `layer0_*.npy/.f32` |
| `/root/anima-xsmax/scripts/gen_tokenizer_ref.py` | Python tokenizer oracle |
| `/root/anima-xsmax/results/goldens/*.npz` | goldens (cond_context, block_00..27_out, step_latents, final_latent, decoded_rgb, init_noise_randn) |

---

## 9. Key handoff references for the next AI

- **Qwen encoder structure:** HANDOFF.md §4 (lines 95–134) and §4.2 (adapter, lines 135–166); MODEL_ARCHITECTURE.json "text_encoder"; RUNTIME_CONSTANTS.json "text_encoder".
- **Attention/RoPE/gemma3 exact source:** `/root/comfy-ref/comfy/text_encoders/llama.py` (lines 415–580: RMSNorm add, precompute_freqs_cis, apply_rope, Attention.forward; lines 591–636: TransformerBlock; lines 728–800: BaseLlama.forward incl. causal mask & optimized_attention_for_device).
- **optimized_attention / GQA / scale:** `/root/comfy-ref/comfy/ldm/modules/attention.py` (attention_pytorch ~line 203, attention_basic ~167, `repeat_kv_for_gqa` in `comfy/ops`).
- **Tokenizer parity rule:** DECISIONS D014.
- **Pack format reality:** DECISIONS D008–D013; ANIMAPK_SPEC.md + real pack offsets.

---

## 10. Suggested immediate next actions for the continuation AI

1. **Fix GQA head mapping** — try `kvHead = h // 2` (floor `h * kv_heads / heads`) in `QwenEncoderCPU.runLayer` (and the Python reference), re-run layer-0 vs Python, then full-28 vs golden `cond_context`. Highest-probability fix.
2. If GQA isn't it, build a **true torch reference** for layer 0 and layer 27 from the dequantized weights (fp32), compare per-layer to both my Swift and Python, and find the first divergent layer.
3. Re-verify the **final norm / post-processing** of `cond_context` against how ComfyUI captures it for this model.
4. Once encoder parity passes (cosine ≥ 0.999), proceed to §28 LLM adapter, then §30–§33 DiT, §34 sampler, §36–§39 VAE, in the runbook order. Only then the UI (K001) and model download UX (D006/K001).

---

## 11. Is the buggy code in the repo? — YES

Both `AnimaXS/Runtime/Text/QwenEncoderCPU.swift` and `AnimaXS/Runtime/Text/QwenNumerics.swift` are **committed** (commits `afe795a`, xcodeproj regenerated in `d2c9dc4`, CI green on `a414159`). The advanced AI can read the exact buggy code directly from the repo. The **Python reference** (which produced the identical wrong output) is NOT in the repo — it was ad-hoc in the harness — but it shares the same misunderstanding, so fixing the Swift fixes both. The Linux harness source is under `/root/anima-harness/Sources/harness/` (not in the git repo; it's scratch).

## 12. Golden control values (text) — the ONLY Qwen-encoder control

The goldens' `cond_context` is the sole reference for the Qwen encoder. The 118 MB `.npz` is impractical to share; here are the **small, high-signal control values** (extracted to `docs/cond_context_case1_anchor.txt`). Note: **`block_00_out..block_27_out` in the goldens are DiT block outputs, NOT Qwen layers — they do NOT help isolate the Qwen encoder bug.** Only `cond_context` does.

`cond_context` = case1_danbooru seed1337, Qwen last-layer hidden (no final norm), fp32, shape `[1, 46, 1024]`:

| Token | dims 0–5 |
|---|---|
| 0 | `2.012837, 25.056881, 0.033985, -17.743534, 1.363858, -7.191565` |
| 23 (mid) | `2.814430, 2.906637, -1.134546, 4.335360, 1.339249, -21.847313` |
| 45 (last) | `-1.316896, 17.327833, -1.357964, -3.790185, 2.751323, 0.100495` |

Other anchors: token0 dims 128–129 = `1.783196, -1.033121`; token0 dim 1023 = `-0.811501`; **global max|abs| = 57.638** (min −57.638, max 40.843); per-token L2 norm (tokens 0–5) = `59.70, 85.11, 107.39, 131.04, 119.92, 121.78`; total elems = 47,104.

> **Correction to my earlier report:** I wrote "golden magnitude ~2–25" from the first-6; the true per-token L2 norms are ~60–131, so the golden values are large too. My wrong output first6 `[104.6, −80.1, …]` is *not* obviously wrong by magnitude alone — the error is **structural/orthogonal (cosine −0.04)**, reinforcing the GQA-head-pairing or formula-detail hypothesis over a simple scaling issue.

Also available (already in fixtures/verified): Qwen token IDs 46 = `[13629, 22362, 11, 1850, 4271, …]`; T5 IDs 47 (listed in §12 above as `[20975, 6, 200, 463, … 60, 7, 1]`, all weights 1.0); sigmas = `[1.0, 0.95469, 0.90036, 0.834, 0.75112, 0.64469, 0.50299, 0.30501, 0.0]`; init_noise first16 = `[0.180772, -0.069988, -0.359623, -0.915204, 0.625765, 0.02551, 0.954514, 0.064349, 0.361151, 1.167878, -1.349893, -0.510177, 0.235958, -0.239778, -0.921115, 1.543297]`; final_latent first8 = `[0.702544, 1.049438, 0.884274, 0.795114, 0.754374, 0.726681, 0.787972, 0.548165]`.

For a full exact match check, the advanced AI can either (a) read the npz from `/root/anima-xsmax/results/goldens/` if running on this box, or (b) use the already-committed fixture `AnimaXSTests/Fixtures/case1_danbooru_seed1337_fixture.json` (contains full `final_latent`, `init_noise_first16`, `sigmas`, `t5_ids`, `cond_context_first6`, and `cond_context_sha256` for integrity).
