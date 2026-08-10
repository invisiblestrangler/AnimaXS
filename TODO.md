# TODO — AnimaXS Implementation Checklist

Check a task ONLY when its validation criterion passes (not when code is merely written).
Stable IDs. Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` blocked/cancelled.

---

## A — Preflight and assets

- [x] **A001** — Locate model packs + golden npz at documented paths.
  - deps: none · output: paths confirmed · validation: files exist
- [x] **A002** — Verify SHA-256 of all 3 packs + case1_danbooru_seed1337.npz.
  - deps: A001 · output: hashes match runbook · validation: all 4 SHA-256 exact
- [x] **A003** — Verify GitHub credentials + target repo exists/empty/public.
  - deps: none · output: repo `invisiblestrangler/AnimaXS` verified empty+public; gh authenticated · validation: `gh repo view` + `gh auth status`
- [x] **A004** — Verify external facts: Xcode 26.3 on macos-15 runner, iOS 26.2 SDK, XcodeGen not preinstalled, swift-transformers latest (1.3.3).
  - deps: none · output: RUNBOOK §4 STATUS entries · validation: runner-images README + GitHub API
- [ ] **A005** — Fetch current CircleStone Anima license; evaluate distribution terms.
  - deps: none · output: license copy in docs/, verdict in DECISIONS.md · validation: distribution section read, gate recorded (runbook §6)
- [ ] **A006** — Extract small golden fixtures from case1_danbooru_seed1337.npz into AnimaXSTests/Fixtures/ (prompt, T5 ids, attn mask, qwen context, noise, sigmas, final latent, decoded-rgb slice, block_00/15/27 slices if storage allows).
  - deps: A002 · output: fixtures + fixtures.json manifest · validation: sha256 recorded, shapes correct, < ~3 MB total

## B — Repository / project bootstrap

- [x] **B001** — Persistent context files (RUNBOOK/TODO/STATUS/DECISIONS/TEST_MATRIX/DEVICE_TESTS).
  - deps: none · output: 6 files at repo root · validation: exist + current
- [ ] **B002** — project.yml (XcodeGen) + app skeleton + test target; deployment target 18.0, Swift 5 mode.
  - deps: B001 · output: project.yml, minimal SwiftUI app, unit-test target · validation: xcodegen generates project without warnings
- [ ] **B003** — Bootstrap job generates AnimaXS.xcodeproj on macOS runner and commits it back.
  - deps: B002 · output: committed AnimaXS.xcodeproj · validation: repo contains xcodeproj, CI generated it
- [ ] **B004** — ci.yml: job1 project consistency (xcodegen generate; git diff --exit-code).
  - deps: B003 · output: green consistency job · validation: CI pass
- [ ] **B005** — ci.yml: job2 generic iPhone build (Xcode 26.3, CODE_SIGNING_ALLOWED=NO).
  - deps: B003 · output: green device build · validation: CI pass
- [ ] **B006** — ci.yml: job3 simulator unit tests (dynamic simulator discovery).
  - deps: B003 · output: green test job · validation: CI pass, tests ran (not skipped)

## C — CI

- [ ] **C001** — Metal toolchain conditional install (`xcrun --find metal` fail → `xcodebuild -downloadComponent MetalToolchain`).
  - deps: B004 · output: shared CI step · validation: works on macos-15
- [ ] **C002** — ci.yml: job4 pure-reference tests job (parser, W4/W8 CPU, sampler vector, sigma, RoPE/RMSNorm/AdaLN CPU refs, manifest, checkpoint).
  - deps: D001, E001(CPU parts) · output: green reference-test job · validation: all listed tests pass on simulator
- [ ] **C003** — full-inference.yml manual workflow (workflow_dispatch; Metal probe; SKIPPED_NO_METAL path).
  - deps: B005, L001 · output: workflow file + attempted run · validation: run records PASS or SKIPPED_NO_METAL explicitly
- [ ] **C004** — CI never downloads packs on push; only full-inference job may fetch release assets.
  - deps: C002, C003 · output: enforced in YAML · validation: push CI completes < 30 min without pack download

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
  - deps: D005, K001 · output: ModelStore.swift · validation: UI states work; download+verify of real pack on device
- [ ] **D007** — Block-range lookup: logical block 0…27 → physical byte range via `model.diffusion_model.blocks.N.` prefix filter (never block_index).
  - deps: D001 · output: DiTBlockLocator · validation: 28 blocks, contiguous ranges, order test passes

## E — Metal / MPS primitives

- [ ] **E001** — MetalContext: device/queue/library/pipeline cache; probe apple5 family, maxBufferLength, maxThreadgroupMemoryLength, recommendedMaxWorkingSetSize, os_proc_available_memory, thermalState.
  - deps: B002 · output: MetalContext.swift · validation: probe values populated on device/simulator; no newer-GPU API usage
- [ ] **E002** — Metal kernels compile: dequant_w4_to_half, dequant_w8_to_half (in AnimaKernels.metal).
  - deps: E001 · output: kernels + tests vs CPU reference · validation: max abs err ~0 vs CPU decoder (fp16 compare)
- [ ] **E003** — Kernels: rmsnorm_f32_to_half, rmsnorm_half_to_half, layernorm_f32_modulated_to_half, silu, gelu, gate_add_half_into_float, add_half_into_float.
  - deps: E002 · output: kernels + unit tests · validation: fp32-stat correctness vs CPU reference
- [ ] **E004** — Kernels: rope_qk, qk_rmsnorm, patchify17, unpatchify16, euler_step_f32.
  - deps: E003 · output: kernels + unit tests · validation: CPU reference match (small slices)
- [ ] **E005** — w4_matvec_f32 direct kernel (M=1 precision-critical path: timestep embedding, AdaLN modulation).
  - deps: E004 · output: kernel + test · validation: fp32 accumulate, error vs fp64 reference < 1e-4
- [ ] **E006** — MPS linear: dequant→fp16 scratch→MPSMatrixMultiplication ([M,K]×[N,K]ᵀ); M-tiling (128) for giant GEMMs; async completion via continuation.
  - deps: E002, E001 · output: LinearExecutor.swift · validation: GEMM tests vs CPU fp32 reference; M=1024×8192×2048 tiled path works
- [ ] **E007** — MPS precision test K=2048/8192 vs CPU fp32/fp64: max abs err, RMSE, cosine → recorded in diagnostics.
  - deps: E006 · output: test + diagnostics record · validation: numbers recorded; decide simple-vs-chunked path (DECISIONS)
- [ ] **E008** — Attention: query-tiled (128) MPSGraph attention (scaled scores → softmax fp32 → ×V); self-attn K=1024, cross-attn K=512.
  - deps: E006 · output: AttentionExecutor.swift + tests · validation: matches CPU reference; memory bounded

## F — Tokenizer / text encoder

- [x] **F001** — Pin swift-transformers 1.3.3 (verify iOS 16+ compat, local-file tokenizer operation).
  - deps: B002 · output: SPM dependency pinned · validation: builds; Tokenizers product links
- [x] **F002** — Bundle Qwen + T5 tokenizer assets (from pinned reference) as app resources.
  - deps: F001 · output: Resources/Tokenizers/* · validation: files present, no HF network needed
- [x] **F003** — Python reference token IDs for 3 canonical prompts (Qwen + T5); fixtures.
  - deps: A006 · output: ref token JSONs · validation: generated with pinned tokenizer files
- [x] **F004** — Swift tokenizer parity tests: exact token IDs on canonical prompts; reject >512 (Qwen or T5).
  - deps: F002, F003 · output: tokenizer tests · validation: exact integer match; long-prompt rejection path
- [ ] **F005** — Qwen encoder pipeline: gather W8 embedding rows (no full-table dequant) → fp32 residual → 28 streamed layers → layer-27 hidden (NO final norm).
  - deps: F004, D004, D007-style TE locator, E001 · output: QwenEncoder.swift · validation: shape (1,seq,1024), finite, cosine vs cond_context ≥ 0.999
- [ ] **F006** — Qwen layer internals: GQA 16/8 head broadcast, per-head Q/K RMSNorm (gemma3), causal mask exact, RoPE theta 1e6.
  - deps: F005 · output: verified layer math · validation: per-layer cosine ≥ 0.999 vs reference where available

## G — LLM adapter

- [ ] **G001** — Adapter: T5 ids → W4 embedding gather → 6 blocks → out_proj → RMSNorm → T5-weight multiply → zero-pad to 512 → [1,512,1024] conditioning.
  - deps: F005, D003, E001 · output: LLMAdapter.swift · validation: cosine ≥ 0.999 vs reference cond; finite
- [ ] **G002** — Resolve from pinned source (not handoff guess): adapter MLP activation, self-attn causality, cross-attn mask, RoPE placement, norm order, bias behavior.
  - deps: G001 · output: DECISIONS entry + code comments · validation: source-cited, matches reference numerics

## H — DiT

- [ ] **H001** — DiT input: 17-ch (16 latent + padding mask), patchify 2×2 → 1024 tokens ×68, input proj → 2048 fp32 residual.
  - deps: E004, E006 · output: DiTInput.swift · validation: shape/token count exact
- [ ] **H002** — Timestep: sigma-based sinusoidal embed dim 2048 base 10000 + model RMSNorm/MLP path; full fp32.
  - deps: E005 · output: TimestepEmbedder.swift + CPU reference test · validation: CPU unit test matches
- [ ] **H003** — AdaLN LoRA modulation: shift/scale/gate chunk ordering exact per block branch (self/cross/MLP); fp32.
  - deps: H002 · output: Modulation.swift + CPU test · validation: CPU unit test equations exact
- [ ] **H004** — DiT 3-D RoPE: T/H/W axes, 42/42/44 split, theta 42871.1/10000, 2×2 rotation blocks; CPU impl first; self-attn only.
  - deps: E004 · output: DitRoPE.swift + CPU test + Metal slice compare · validation: hard gate — CPU slice == Metal slice, cosine ≥ 0.999
- [ ] **H005** — DiT block 0 end-to-end vs golden block_00_out (shape/finite/maxabs/RMSE/cosine).
  - deps: H001–H004, D007, E006, E008 · output: one working block · validation: cosine ≥ 0.999 (measured tolerance), DO NOT proceed if wrong
- [ ] **H006** — Full 28-block loop with WeightStreamer ring (39 MB), logical-order execution, per-block real ranges.
  - deps: H005, D007, E006 · output: DitForward.swift · validation: all blocks finite; block 15/27 match fixtures where present
- [ ] **H007** — Post-loop final norm/modulation/projection + unpatchify16 → [1,16,1,64,64] velocity.
  - deps: H006 · output: final projection · validation: finite, shape exact

## I — Sampler / full diffusion

- [ ] **I001** — Sigma constants (9 points) + Euler FLOW update fp32 (x_next = x + (x−denoised)/σ·dσ).
  - deps: none (CPU) · output: Sampler.swift constants + unit test · validation: matches sampler_vectors.py data
- [ ] **I002** — 8-step loop: single model pass per step (CFG=1), latent fp32, NaN/Inf check per step, progress + checkpoint after each step.
  - deps: I001, H007 · output: full diffusion run · validation: 8 step latents finite; matches step_latents within tolerance
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
  - deps: I002, J004 · output: GenerationCoordinator.swift · validation: full pipeline runs; memory maps released between stages
- [ ] **K003** — Cancellation at safe boundaries + background: stop scheduling, finish safe work, checkpoint, release; foreground Resume.
  - deps: I004, K002 · output: lifecycle handling · validation: cancel/background/resume behave without crash
- [ ] **K004** — Memory warning: checkpoint + graceful cancel + free buffers + recoverable message. Thermal: pause/stop policy.
  - deps: K002 · output: policy handlers · validation: simulated warning path works
- [ ] **K005** — Diagnostics screen + export JSON + self-test buttons (pack validation, W4/W8 vector, MPS precision, mmap benchmark, GEMM, attention tile, golden-noise self-test).
  - deps: E001–E008, D001–D005, I003 · output: DiagnosticsView + DiagnosticsEngine · validation: runs on device/simulator, exports JSON

## L — Full CI / inference testing

- [ ] **L001** — Full canonical inference integration test: golden noise, 512, 8 steps, CFG1; compare final latent (tolerance), assert finite.
  - deps: I002, J004, A006 · output: IntegrationTests · validation: final latent cosine ≥ 0.999 (or recorded tolerance); runs where Metal exists
- [ ] **L002** — Model release `model-assets-v1` (3 packs + manifest + LICENSE + NOTICE) AFTER license gate A005 passes; unauthenticated URL verification + re-hash.
  - deps: A005, D005 · output: GitHub Release · validation: unauthenticated download matches SHA-256
- [ ] **L003** — Manual full-inference workflow run: Metal probe → download packs via ModelStore → inject golden noise → canonical inference → stage timings → NaN/Inf asserts → small logs only.
  - deps: L001, L002, C003 · output: workflow run record · validation: PASS or SKIPPED_NO_METAL with reason

## M — Documentation / release / final handoff

- [ ] **M001** — README: clone/open/sign/build/install steps, SDK-vs-deployment-target explanation, license notices.
  - deps: B003 · output: README.md · validation: user can follow steps without source edits
- [ ] **M002** — TEST_MATRIX.md + DEVICE_TESTS.md maintained; DEVICE_TESTS captures A12 microbenchmarks when run.
  - deps: ongoing · output: docs · validation: accurate, current
- [ ] **M003** — Final report: repo URL, commit SHA, release URL, Xcode/SDK/deployment facts, CI statuses, Metal availability, image generated?, pack hashes, test counts, unresolved issues, first XS Max steps; explicit A12 DEVICE TESTED: YES/NO.
  - deps: all · output: report to user · validation: every field answered honestly
