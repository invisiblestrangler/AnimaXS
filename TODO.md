# TODO — AnimaXS Implementation Checklist

Check a task ONLY when its validation criterion passes (not when code is merely written).
Stable IDs. Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` blocked/cancelled.

**Current execution priority:** I001/I002 sampler integration, in parallel with J001 VAE fold validation, then J002–J004 and full-pipeline integration. Do not simply choose the first unchecked item in file order. A005 gates only model-pack release; A006/D005/D006 and UI/release work can proceed when their dependency path becomes relevant.

---

## A — Preflight and assets

- [x] **A001** — Locate model packs + golden npz at documented paths.
  - deps: none · output: paths confirmed · validation: files exist
- [x] **A002** — Verify SHA-256 of all 3 packs + case1_danbooru_seed1337.npz.
  - deps: A001 · output: hashes match runbook · validation: all 4 SHA-256 exact
- [x] **A003** — Verify GitHub credentials + target public repository.
  - deps: none · output: populated public repo `invisiblestrangler/AnimaXS`; `gh` authenticated · validation: `gh repo view` + `gh auth status`
- [x] **A004** — Verify current hosted build facts: standard arm64 `macos-15`, Xcode 26.3 path, iOS 26.2 SDK, runner resources, current CI action majors.
  - deps: none · output: RUNBOOK §4 + STATUS entries · validation: official docs/API and final live snapshot `31452206651`
- [ ] **A005** — Archive and evaluate the current CircleStone Anima license **and inherited NVIDIA Cosmos/Open Model terms** before distributing packs.
  - deps: none · output: exact license copies + required notices in `docs/`; source URLs/retrieval date/verdict in DECISIONS · validation: both distribution/attribution/use-restriction chains read and recorded (runbook §6)
- [ ] **A006** — Extract the small committed fixture set from `case1_danbooru_seed1337.npz`: prompt, token IDs/masks, context anchors/full context if budget permits, noise, sigmas, final latent, step/block/RGB slices, and shape/hash metadata.
  - deps: A002 · output: test-only fixtures + `fixtures.json` · validation: SHA-256 and shapes recorded, deterministic anchors checked, committed total about 3 MB; full 8 MB block arrays/118 MB NPZ stay local or manual-CI-only

## B — Repository / project bootstrap

- [x] **B001** — Persistent context files (RUNBOOK/TODO/STATUS/DECISIONS/TEST_MATRIX/DEVICE_TESTS).
  - deps: none · output: 6 files at repo root · validation: exist + current
- [x] **B002** — project.yml (XcodeGen) + app skeleton + test target; deployment target 18.0, Swift 5 mode.
  - deps: B001 · output: project.yml, minimal SwiftUI app, unit-test target · validation: xcodegen generates project without warnings
- [x] **B003** — Bootstrap job generated and committed AnimaXS.xcodeproj.
  - deps: B002 · output: committed AnimaXS.xcodeproj · validation: repo contains xcodeproj, CI generated it
- [x] **B004** — ci.yml: job1 project consistency (xcodegen generate; git diff --exit-code).
  - deps: B003 · output: green consistency job · validation: CI pass
- [x] **B005** — ci.yml: job2 generic iPhone build (Xcode 26.3, CODE_SIGNING_ALLOWED=NO).
  - deps: B003 · output: green device build · validation: CI pass
- [x] **B006** — ci.yml: job3 simulator unit tests (dynamic simulator discovery).
  - deps: B003 · output: green test job · validation: CI pass, tests ran (not skipped)

## C — CI

- [x] **C001** — Metal toolchain conditional install (`xcrun --find metal` fail → `xcodebuild -downloadComponent MetalToolchain`).
  - deps: B004 · output: shared CI step · validation: works on macos-15
- [x] **C002** — Run all pure-reference and pack-free Metal/MPS tests in the single simulator XCTest job (no duplicate dependency build).
  - deps: B006 · output: green simulator test job · validation: latest main CI executed all discovered tests; future tests are added to the same target
- [~] **C003** — `full-inference.yml` manual workflow: boot simulator, execute permanent Metal/MPS smoke, explicit `SKIPPED_NO_METAL`, then fail-fast full inference.
  - deps: B005, L001, L002 · output: workflow file + attempted run · validation: after L001/L002, run records PASS or SKIPPED_NO_METAL; no `continue-on-error`
- [x] **C004** — Push/PR CI never downloads packs; only manual full-inference may fetch release assets.
  - deps: C002 · output: enforced in YAML · validation: normal CI green without pack download; real-pack tests explicitly skip

## D — Animapk / model store

- [x] **D001** — Swift ANMA v1 parser: MappedFile (mmap), AnimapkHeader, AnimapkTensor, AnimapkFile; JSON-authoritative names/shapes; blobOffset index; full validation (magic/version/size/alignment/ranges).
  - deps: B002 · output: 4 Swift files · validation: parser unit tests pass (synthetic + real pack where available)
- [x] **D002** — CRC32 (zlib) per-tensor verification in parser.
  - deps: D001 · output: crc check API · validation: synthetic fixture crc passes; real packs 0 mismatches
- [x] **D003** — CPU reference W4 decoder (nibble order, group=64, fp16 scale/zero) + known vector test (block0 mlp.layer1 first bytes/scales).
  - deps: D001 · output: W4Decoder.swift + tests · validation: known first-values match handoff
- [x] **D004** — CPU reference W8 decoder + real embedding-row vector test.
  - deps: D001 · output: W8Decoder.swift + tests · validation: matches HANDOFF.md documented vector
- [ ] **D005** — ModelManifest: built-in manifest struct (filename/size/sha256/URL/component) + incremental CryptoKit SHA256.
  - deps: B002 · output: ModelManifest.swift · validation: unit tests on synthetic file + known hash
- [ ] **D006** — ModelStore download/verify/ready states; Application Support storage; backup exclusion + file protection; disk-space check.
  - deps: D005 · output: `ModelStore.swift` with testable state machine · validation: synthetic download/verify/storage tests plus real-pack download on manual CI/device; K001 binds these states to UI
- [x] **D007** — Zero-copy DiT locator: logical block 0…27 → validated physical range and tensor-relative data/scale/zero spans via `model.diffusion_model.blocks.N.` prefix (never `block_index`).
  - deps: D001 · output: `DiTBlockLocator` span API suitable for mmap→one-slot ring copies · validation: exactly 28 disjoint logical ranges; every tensor span is in-range/aligned; synthetic order test + real-pack audit pass; no full-tensor `Data` copies
- [x] **D008** — Zero-copy Qwen locator: embedding tensor rows plus logical layer 0…27 contiguous ranges and tensor-relative spans.
  - deps: D001, D007 shared span primitives · output: `QwenLayerLocator`/shared locator API · validation: exactly 28 logical layers in numerical order, embedding row spans checked, real-pack ranges match metadata, no full embedding/table copy

## E — Metal / MPS primitives

- [x] **E001** — Finish `MetalContext`: device/queue/library/pipeline cache plus apple5 support, buffer/threadgroup/working-set/memory/thermal diagnostics; keep permanent pack-free execution tests in normal CI.
  - deps: B002 · output: completed `MetalContext.swift` + `MetalExecutionTests` · validation: hosted kernel+MPS execution passes (permanent tests proven by `31452206651`); all probe values populate where API-supported; no newer-GPU API requirement
- [x] **E002** — Row-aware `dequant_w4_to_half` and `dequant_w8_to_half` kernels.
  - deps: E001 · output: kernels + synthetic GPU-vs-CPU tests · validation: exact after fp16 conversion for vectors and W4 `[2,68]` / W8 `[2,65]`; row-1 group reset and bounds asserted on hosted CI
- [x] **E003** — Exact arithmetic kernels: fp32-stat RMSNorm/LayerNorm, SiLU, **exact erf GELU**, fp32 modulation/gate residual add, and half→float residual add.
  - deps: E001 · output: kernels + CPU parity tests · validation: H005 equations match on hosted Metal; replace the scaffold tanh GELU and half gate before completion
- [x] **E004** — Q/K RMSNorm + split-half `(p,p+64)` DiT RoPE, patchify17/unpatchify16, and fp32 Euler update.
  - deps: E003 · output: kernels + pack-free tests · validation: small slices match H001/H004/I001 CPU references on hosted Metal
- [x] **E005** — Direct packed `w4_matvec_f32` for M=1 precision-critical timestep/AdaLN linears.
  - deps: E002 · output: kernel + non-aligned-K test · validation: row-aware dequant, fp32 accumulation, error/cosine recorded against fp64 reference
- [x] **E006** — Common MPS linear: row-aware dequant→one reusable fp16 scratch→`MPSMatrixMultiplication` `[M,K]×[N,K]ᵀ`; M-tiling (start at 128); async completion.
  - deps: E001, E002 · output: `LinearExecutor.swift` · validation: transpose/stride tests vs CPU; representative tiled shape on hosted CI; giant real shape only in pack-backed/manual test if CI time/memory permits
- [x] **E007** — Precision characterization for K=2048/8192 against CPU fp32/fp64: maxAbs, RMSE, cosine; decide simple vs chunked accumulation.
  - deps: E005, E006 · output: test + recorded diagnostics/decision · validation: synthetic representative tests on hosted CI and real-shape result where resources permit; repeat on A12 before device acceptance
- [x] **E008** — Query-tiled (start at 128) MPS/MPSGraph attention: scaled scores, fp32 softmax, ×V; self K=1024 and cross K=512.
  - deps: E006 · output: `AttentionExecutor.swift` + bounded-memory tests · validation: self/cross small and representative shapes match H005 CPU attention on hosted CI; all 512 padded cross rows retained
- [x] **E009** — Production Metal DiT block 0 vertical slice using D007 ranges, one-slot ring, reusable dequant/activation buffers, E002–E008, and no large Swift matrices.
  - deps: D007, E002–E008, H005 · output: `DiTBlockMetal`/executor + pack-backed parity test · validation: bounded memory, finite, and matches the H005 same-W4 oracle within recorded tolerance; hosted manual CI when packs/fixture are available

## F — Tokenizer / text encoder

- [x] **F001** — Pin swift-transformers 1.3.3 (verify iOS 16+ compat, local-file tokenizer operation).
  - deps: B002 · output: SPM dependency pinned · validation: builds; Tokenizers product links
- [x] **F002** — Bundle Qwen + T5 tokenizer assets (from pinned reference) as app resources.
  - deps: F001 · output: Resources/Tokenizers/* · validation: files present, no HF network needed
- [x] **F003** — Python reference token IDs for 3 canonical prompts (Qwen + T5); fixtures.
  - deps: F002 · output: ref token JSONs · validation: generated with pinned tokenizer files
- [x] **F004** — Swift tokenizer parity tests: exact token IDs on canonical prompts; reject >512 (Qwen or T5).
  - deps: F002, F003 · output: tokenizer tests · validation: exact integer match; long-prompt rejection path
- [x] **F005** — Qwen CPU reference pipeline: gather W8 embedding rows → fp32 residual → 28 layers → final RMSNorm.
  - deps: F004, D004 · output: `QwenEncoderCPU.swift` correctness oracle · validation: shape (1,seq,1024), finite, same-W8 structural parity and measured golden tolerance
  - **DONE 2026-08-10.** QwenEncoderCPU.encode() verified: shape (1,46,1024), allFinite, cosine vs golden cond_context = **0.992164** (rmse 0.434, maxAbs 47.5). The 0.999 target assumes fp16-vs-bf16 same-weights; here the comparison is W8-dequant-vs-original-bf16. Structural proof Swift W8 == pinned-Comfy W8 (scripts/qwen_comfy_oracle.py) cosine **1.000000** → kernel correct; residual 0.008 is pure W8 quantization (GOLDENS.md §5 ≥0.99 boundary). NOTE: `encode()` now applies the FINAL RMSNorm (D019) — golden cond_context is post-final-norm, contrary to earlier handoff prose.
- [x] **F006** — Qwen layer internals: GQA 16/8 head broadcast, per-head Q/K RMSNorm (gemma3), causal mask exact, RoPE theta 1e6.
  - deps: F005 · output: verified layer math · validation: per-layer cosine ≥ 0.999 vs reference where available
  - **DONE 2026-08-10.** GQA 16/8 grouped broadcast (`kvHead = qHead / 2`, D018), gemma3 per-head Q/K RMSNorm, causal mask, half-split RoPE theta 1e6 all validated against pinned-ComfyUI oracle; per-layer oracle outputs saved (scripts/oracle_out/qwen_oracle_layers.npz). Regression test GqaHeadMappingTests.
- [x] **F007** — Production streamed Metal Qwen encoder: gather only requested W8 embedding rows, one Qwen layer range at a time, reusable buffers, final RMSNorm.
  - deps: D008, E002–E008, F005–F006 · output: `QwenEncoderMetal.swift`/executor · validation: bounded memory, no large Swift arrays/Data copies, finite, and same-W8 parity against F005 at layer checkpoints/final context; pack-backed hosted CI plus A12 acceptance
  - **DONE 2026-08-11.** Hosted real-W8 run `31491046871`: layer 0/15/27 cosine `0.9999995091`/`0.9999999958`/`0.9999992347`; final cosine **0.9999992405**, RMSE `0.00430097`, maxAbs `0.172913`, 5.19 s. It gathers selected rows, streams one ~16 MB layer, uses grouped 16Q/8KV attention, and keeps fp32 residuals. A12 acceptance remains pending.

## G — LLM adapter

- [x] **G001** — LLM adapter CPU oracle: T5 ids → W4 embedding gather → 6 blocks → out_proj → RMSNorm → T5-weight multiply → zero-pad to `[1,512,1024]`.
  - deps: F005, D003 · output: `LLMAdapter.swift` correctness oracle · validation: same-W4 structural parity and finite/padding checks
  - **DONE 2026-08-10.** `AnimaXS/Runtime/Text/LLMAdapter.swift` implemented (reads W4 DiT pack `model.diffusion_model.llm_adapter.*`). Validated against pinned-ComfyUI oracle (`scripts/anima_adapter_oracle.py`, commit cbbc9da, same W4 weights): weighted [1,47,1024] cosine **1.000000** (rmse 5.2e-7, maxAbs 3.3e-6); padded [1,512,1024] cosine 1.000000, tail all-zero, all finite. Structural parity proven.
- [x] **G002** — Resolve from pinned source (not handoff guess): adapter MLP activation, self-attn causality, cross-attn mask, RoPE placement, norm order, bias behavior.
  - deps: G001 · output: DECISIONS entry + code comments · validation: source-cited, matches reference numerics
  - **DONE 2026-08-10.** All resolved from `comfy/ldm/anima/model.py` (see D021): MLP = exact GELU (nn.GELU, not SiLU); self-attn = MHA 16/16 bidirectional (no causal mask, target_attention_mask=None); cross-attn = MHA over Qwen context (source_attention_mask=None); RoPE = INTERLEAVED (rotate_half, HF-style), theta 10000, applied to Q&K of both attns (DIFFERS from Qwen's half-split); norm = RMSNorm(eps 1e-6) before each attn + before MLP + final after out_proj; biases: out_proj + MLP have bias, attn projections do NOT.
- [x] **G003** — Production Metal LLM adapter: row-gather W4 T5 embeddings, six bounded-memory blocks, out projection/RMSNorm/T5 weighting, zero-pad to `[1,512,1024]`.
  - deps: D007 span primitives, E002–E008, F007, G001–G002 · output: `LLMAdapterMetal.swift`/executor · validation: no full embedding or nested Swift matrices, same-W4 parity against G001, all finite, padded tail exactly zero
  - **DONE 2026-08-11.** Hosted real-W4 subset run `31492451065`: layer 0/5 cosine `0.9999997571`/`0.9999991391`; final cosine **0.9999984505**, RMSE `9.579e-5`, maxAbs `0.00190943`, 1.46 s; all 465 padded rows exactly zero. One ~9.6 MB block ring, selected embedding rows, exact GELU/bias, and fp32 residuals are retained. A12 acceptance remains pending.

## H — DiT

- [x] **H001** — DiT input CPU oracle: 17-ch (16 latent + padding mask), patchify 2×2 → 1024 tokens ×68, input projection → 2048 fp32 residual.
  - deps: D003 · output: `DiTInput.swift` reference · validation: shape/token count/order and same-W4 x-embed parity exact
  - **DONE 2026-08-10; decoder audit corrected 2026-08-11.** `AnimaXS/Runtime/Text/DiTInput.swift` (patchify 2×2, r=1 → (c·4+m·2+n) feature order, x_embedder Linear(68→2048) W4, fp32 residual). The original same-dump oracle proved forward-math parity but shared a bad flat W4 dump. D034 fixes matrix groups to reset per row; the refreshed independent matmul still matches Swift at cosine **1.000000000** (maxAbs `3.58e-7`). CPU tests cover patchify and row-reset decoding.
- [x] **H002** — Timestep CPU oracle: sigma-based sinusoidal embed dim 2048 base 10000 + model RMSNorm/MLP path; full fp32.
  - deps: D003 · output: `TimestepEmbedder.swift` + CPU reference test · validation: same-W4 oracle matches
  - **DONE 2026-08-10.** `AnimaXS/Runtime/Text/TimestepEmbedder.swift` (sigma→sinusoidal dim 2048 base 10000, cos[0:1024]/sin[1024:2048]; Linear1(2048→2048 no bias)→SiLU→Linear2(2048→6144 no bias) = adaln_lora_B_T_3D; RMSNorm on raw sinusoidal = t_embedding_B_T_D; all fp32). Transcribed from predict2.py Timesteps (219-238), TimestepEmbedding (241-269), t_embedding_norm (737, 881-882). Validated vs oracle (same W4 pack): raw sinusoid cosine **1.00000000**, embedding (RMSNorm'd) cosine **1.00000000**, adaln_lora [6144] cosine **1.00000000** (maxAbs 9.9e-5), all finite. CPU tests `TimestepEmbedderTests` (sinusoidal vs torch ref, SiLU).
- [x] **H003** — AdaLN LoRA modulation: shift/scale/gate chunk ordering exact per block branch (self/cross/MLP); fp32.
  - deps: H002 · output: Modulation.swift + CPU test · validation: CPU unit test equations exact
  - **DONE 2026-08-10.** `AnimaXS/Runtime/Text/Modulation.swift` (per-branch: SiLU→Linear1(2048→256)→Linear2(256→6144) → +adaln_lora_B_T_3D → chunk(3) into shift/scale/gate, each [2048], fp32; + applyLayerNormModulation = LayerNorm(elementwise_affine=False, mean-center)*(1+scale)+shift). Transcribed from `comfy/ldm/cosmos/predict2.py` Block.forward (486-516, 520-521, 542/575/590). Validated vs oracle (`scripts/dit_input_timestep_oracle.py`, commit cbbc9da, same W4 pack): block-0 shift/scale/gate all 3 branches cosine **1.00000000** (maxAbs ~3e-5), all finite. NOTE: handoff §8 said SiLU(Linear1(emb)) but pinned source (predict2.py:451-465) puts **SiLU BEFORE Linear1** — resolved in D026. CPU tests `ModulationTests` (LayerNorm, SiLU).
- [x] **H004** — DiT 3-D RoPE CPU oracle: T/H/W axes, 42/42/44 split, exact computed theta, 2×2 blocks; self-attn only.
  - deps: none (weightless CPU reference) · output: `DitRoPE.swift` + pinned-source parity tests · validation: CPU vs independent oracle cosine ≥ 0.999; Metal application parity belongs to E004
  - **DONE 2026-08-10.** `AnimaXS/Runtime/Text/DitRoPE.swift` (weightless; `VideoRopePosition3DEmb.generate_embeddings`, position_embedding.py:100-163). Config head_dim=128 → dim_h=42/dim_w=42/dim_t=44, numFreqs=64 (22 temporal + 21 h + 21 w); thetas computed exactly `10000·4.0^(42/40)≈42870.938` (h/w) and `10000` (t) — NOT the rounded prose 42871.x; freq order **t,h,w** (blocks [0..22)=temporal,[22..43)=height,[43..64)=width); each freq = 2×2 block `[cos,-sin,sin,cos]`. Validated vs pinned-ComfyUI oracle (`scripts/dit_rope_oracle.py`, commit cbbc9da): shape [1024,64,2,2] exact, cosine **1.000000000** (maxAbs 4.8e-7, all finite), anchors match (token17/f43 = [-0.27516335,0.96139747,-0.96139747,-0.27516335]). CPU tests `DitRoPETests` (config, thetas, shape/finite, t/h/w ordering, 2×2 layout, width/height rotations vs cos/sin references). Metal slice compare = SKIPPED (weightless pure-math rope; CPU-vs-oracle cosine is the authoritative H004 gate per NEXT_TASK_HANDOFF §4).
- [x] **H005** — CPU DiT block 0 end-to-end oracle vs same-W4 NumPy and source-BF16 golden.
  - deps: H001–H004, D003 · output: validated CPU oracle/reference block · validation: same-W4 structural parity plus source-proven golden tolerance; production Metal parity is E009
  - **DONE 2026-08-11.** The `0.7345` failure was a real row-boundary decoder bug: the packer resets group-64 quantization on every matrix row, while Swift flattened rows. Only DiT `x_embedder` has a non-group-aligned K (`68`), so earlier tests missed it. Row-aware W4/W8 matrix decoders plus regressions are now in place (D034).
  - **Final validation:** canonical last hook invocation = sampler step 7, sigma `0.3050089478492737`. Swift-W4 vs independent NumPy-W4: cosine `1.000000000`, RMSE `6.18e-6`, maxAbs `1.91e-4`, finite. NumPy/Swift W4 vs original BF16 golden: cosine `0.998712106`/`0.998712139`, RMSE `9.43e-2`, maxAbs `1.79`, relative L2 `5.074e-2`. Independent original-BF16 source weights vs golden reach cosine `0.999992303`; corrected W4 `x_embedder` output vs source is `0.998998606`. This is the source-proven W4 tolerance exception permitted by the H005 hard gate (D035), not an unresolved graph error.
- [x] **H006** — Full 28-block **Metal** loop with one-slot WeightStreamer ring (~39 MB), logical-order execution, real per-block ranges, and reusable buffers.
  - deps: D007, E009 · output: `WeightStreamer.swift` + `DitForward.swift` · validation: no CPU/nested-array production fallback; all blocks finite; block 0 remains at E009 parity and block 15/27 match compact/full fixtures where available
- [x] **H007** — Post-loop final norm/modulation/projection + unpatchify → [1,16,1,64,64] velocity.
  - deps: H006 · output: final projection · validation: finite, shape exact
  - **DONE 2026-08-11.** `DiTFinalLayerExecutor` streams the exact three-tensor final range, applies SiLU→W4 2048→256→4096 modulation plus `adaln_lora[0..<4096]`, preserves the source's residual-fp16 and LayerNorm-output-fp16 boundaries with fp32 statistics/AdaLN, projects 1024×2048→64 through `LinearExecutor`, and uses source-order `(p1,p2,t,C)` unpatchify. Hosted real-W4 run `31488934459`: cosine **0.9999999646**, RMSE `0.0003068520`, maxAbs `0.001953125`, finite, exact output shape. Normal CI `31488187793` passed the synthetic full-shape orchestration and nonzero 64-stride/order kernel tests plus generic iOS build.

## I — Sampler / full diffusion

- [ ] **I001** — Sigma constants (9 points) + Euler FLOW update fp32 (x_next = x + (x−denoised)/σ·dσ).
  - deps: none (CPU) · output: Sampler.swift constants + unit test · validation: matches sampler_vectors.py data
- [ ] **I002** — 8-step loop: single model pass per step (CFG=1), latent fp32, NaN/Inf check per step, progress + checkpoint after each step.
  - deps: I001, H007 · output: full diffusion run (conditioning may be injected for isolated DiT test) · validation: 8 step latents finite; matches step_latents within tolerance
- [ ] **I003** — Production RNG: deterministic seeded generator + Box-Muller (app-deterministic, documented as not ComfyUI-identical); golden path loads init_noise_randn instead.
  - deps: none · output: SeededRNG.swift + tests · validation: same seed → same noise; std-normal stats sane
- [ ] **I004** — Checkpoint serialization (latent fp32, step, prompt, seed, resolution, model hashes) + atomic write + resume.
  - deps: I002, D005 · output: Checkpoint.swift + tests · validation: round-trip exact; atomic rename

## J — VAE

- [ ] **J001** — scripts/validate_vae_fold.py: numerical 3D→2D fold validation for every decoder causal conv variant (T=1).
  - deps: A002 · output: script + report · validation: fold exact (or decision to implement true T=1 3-D)
- [ ] **J002** — Wan21 latent norm (chunk-0 mean/std exact) + full-frame decoder, 2-D conv path (fold validated) or true T=1 3-D.
  - deps: J001, E006 · output: VAEDecoder.swift · validation: decoded_rgb vs golden: max abs ≤ 0.05, PSNR ≥ 30 dB
- [ ] **J003** — GroupNorm 32 groups fp32 stats + exact eps; exact activations/upsampling from pinned source.
  - deps: J002 · output: GroupNorm.swift + tests · validation: CPU reference match
- [ ] **J004** — RGB: (rgb+1)/2 clamp → CGImage/UIImage; release fp32 buffer.
  - deps: J002 · output: RGBConverter.swift · validation: image displayed, memory released
- [ ] **J005** — Tiled VAE ONLY if device diagnostics demand it (global GroupNorm stats, 2-pass). NOT in first implementation.
  - deps: J002 + device evidence · output: decision in DECISIONS.md · validation: decision recorded, not speculative code

## K — UI / resilience

- [ ] **K001** — Minimal SwiftUI: model status/download, prompt, seed, Randomize, Generate, Cancel, progress (stage/step/block), elapsed, image, Share, Diagnostics.
  - deps: D006 · output: ContentView/GenerationViewModel · validation: builds; all controls wired
- [ ] **K002** — GenerationCoordinator: one generation at a time; map/unmap per stage; keep only 512×1024 cond between TE/DiT.
  - deps: D006, F007, G003, I002, J004 · output: `GenerationCoordinator.swift` · validation: full pipeline runs; memory maps/releases occur at stage boundaries
- [ ] **K003** — Cancellation at safe boundaries + background: stop scheduling, finish safe work, checkpoint, release; foreground Resume.
  - deps: I004, K002 · output: lifecycle handling · validation: cancel/background/resume behave without crash
- [ ] **K004** — Memory warning: checkpoint + graceful cancel + free buffers + recoverable message. Thermal: pause/stop policy.
  - deps: K002 · output: policy handlers · validation: simulated warning path works
- [ ] **K005** — Diagnostics screen + export JSON + self-test buttons (pack validation, W4/W8 vector, MPS precision, mmap benchmark, GEMM, attention tile, golden-noise self-test).
  - deps: E001–E008, D001–D005, I003 · output: DiagnosticsView + DiagnosticsEngine · validation: runs on device/simulator, exports JSON

## L — Full CI / inference testing

- [ ] **L001** — Full canonical inference integration test: prompt→production Qwen/adapter→golden noise→8-step DiT→VAE at 512/CFG1; compare checkpoints and assert finite.
  - deps: F007, G003, I002, J004, A006 · output: `FullInferenceTests.swift` · validation: final latent/RGB meet recorded source-vs-quantized tolerances; runs where Metal exists
- [ ] **L002** — Model release `model-assets-v1` (3 packs + manifest + LICENSE + NOTICE) AFTER license gate A005 passes; unauthenticated URL verification + re-hash.
  - deps: A005, D005 · output: GitHub Release · validation: unauthenticated download matches SHA-256
- [ ] **L003** — Manual full-inference run: permanent Metal/MPS smoke → verified pack download → golden noise → canonical inference → timings/finite/parity asserts → small logs only.
  - deps: L001, L002, C003 · output: workflow run record + summary · validation: PASS or explicit SKIPPED_NO_METAL; any failure with Metal present remains red

## M — Documentation / release / final handoff

- [ ] **M001** — README: clone/open/sign/build/install steps, SDK-vs-deployment-target explanation, license notices.
  - deps: B003 · output: README.md · validation: user can follow steps without source edits
- [ ] **M002** — TEST_MATRIX.md + DEVICE_TESTS.md maintained; DEVICE_TESTS captures A12 microbenchmarks when run.
  - deps: ongoing · output: docs · validation: accurate, current
- [ ] **M003** — Final report: repo URL, commit SHA, release URL, Xcode/SDK/deployment facts, CI statuses, Metal availability, image generated?, pack hashes, test counts, unresolved issues, first XS Max steps; explicit A12 DEVICE TESTED: YES/NO.
  - deps: all · output: report to user · validation: every field answered honestly
