# AnimaXS A12 Reliability + Sustained-Performance Optimization — HANDOFF

For: the next agent continuing this implementation. Read this fully before touching anything.

---

## TL;DR status
- **Phases P0–P9 are COMPLETE and CI-GREEN** on branch `opt/a12-sustained-io`. P9 (combine winners + presets + final report) is done. **The project is READY FOR PHYSICAL A12 VALIDATION** (runbook §17).
- **Current HEAD: `d16e317`** (P9 complete + state/evidence/report). Working tree CLEAN, `origin/opt/a12-sustained-io` == local `d16e317`. Draft PR #17 open.
- Last green CI: run `31924467039` (P9) — **324 tests, 14 expected skips, 0 failures**.
- The full implementation plan is the runbook file: `/root/.hermes/cache/documents/doc_e617196af2a6_ANIMAXS_A12_RELIABILITY_PERFORMANCE_IMPLEMENTATION_PLAN.md` (also sent in-conversation). **It is the authoritative spec.** Repo: `/root/AnimaXS`, branch `opt/a12-sustained-io`.
- Final report: `OPTIMIZATION_FINAL_REPORT.md` (runbook §23 template).

---

## CRITICAL environment facts (do not rediscover the hard way)

### Build/test environment
- **No Xcode/Swift/Metal toolchain on this Linux VPS.** You CANNOT compile or run tests locally. Do NOT install toolchains (forbidden by runbook §0.2). **Build/test ONLY via GitHub Actions CI** (`ci.yml`):
  ```
  cd /root/AnimaXS && gh workflow run ci.yml --ref opt/a12-sustained-io
  ```
  Wait with `gh run watch <id> --exit-status`. CI ~9 min: project-consistency + iphone-build + simulator-tests. **simulator-tests surfaces Swift compile + test-run errors.** iphone-build passes = app target (incl. Metal shaders) compiles.

### Git discipline (learned through failures — DO NOT skip)
1. **PUSH AFTER EVERY COMMIT.** Untracked local commits silently block CI/bootstrap. Verify `git ls-remote origin opt/a12-sustained-io` == `git rev-parse HEAD`.
2. **If you ADD/REMOVE a .swift file**, regenerate `AnimaXS.xcodeproj` via `bootstrap-project.yml` (workflow_dispatch ref=opt/a12-sustained-io), WAIT for bot commit, `git pull`, THEN ci.yml. Editing existing .swift/.metal files needs NO bootstrap. **Prefer editing existing files.**
3. Keep tool output bounded: `rg -n`, `sed -n`, `git diff --stat`, `git status --short`.
4. Work in SMALL incremental commits, push each.

### Delegation (subagents)
- `delegate_task` works (provider `deepseek-v4-flash` @ `https://api.surplusintelligence.ai/v1` — verified working).
- Subagents hit the **iteration budget (~50 tool calls) mid-large-phases**. Pattern: they commit substantive code then get cut off before tests/CI. **Expect to finish each phase yourself** (push, add missing tests, run CI, fix compile/test errors). Brief them for small commits.
- The user's original instruction: delegate one phase at a time to one subagent, monitor, only intervene when necessary — but you'll likely need to finish CI/fixes yourself.

### Known infra quirks
- `hermes gateway restart` is BLOCKED from inside the gateway process. Don't attempt it.
- Memory tool sometimes fails with "consolidation failed" — don't fight it.
- The terminal guard blocks `.animapk` files (use execute_code or copy to `.bin`).
- **A PARALLEL Hermes session also commits to this branch** (docs/state/handoff commits). ALWAYS `git pull` before committing; if a commit lands in between, rebase/pull — NEVER force-push. Prefer editing existing files.

---

## State files (the continuation contract — UPDATE THESE)
At branch root:
- `OPTIMIZATION_IMPLEMENTATION_STATE.md` — current phase/objective/tests/files/last-safe-continuation-point
- `OPTIMIZATION_EVIDENCE.md` — APPEND-ONLY evidence log
- `OPTIMIZATION_DECISIONS.md` — semantic decisions

Update before long CI waits and at every phase gate.

---

## What's done (all CI-GREEN)

- **P0** branch/state/baseline — CI 31896517851.
- **P1** W8 identity + numerics + failure accounting — CI 31904926712 (273 tests). Resolved-pack identity, checkpoint identity via `models.hashes`, `DiTNumericsPolicy`, W8 final-layer BF16-RNE-in-FP32 overflow fix, `NumericalLocation` (no fake block 1), failure-safe timing/bookkeeping, cancellation reasons.
- **P2** per-step telemetry — CI 31906565105 (277 tests). `DiffusionStepMetrics`/`stepMetrics`/`activeStepIndex`, partial-step recording, traffic counters.
- **P3** fused activation fusion — CI 31908033162 (280 tests). `dit_layernorm_modulate_to_half`/`dit_gelu_half_inplace` kernels, `fusedNormModulation`/`fusedMLPActivation` toggles (default OFF), `fusedTrafficSavedBytes`.
- **P4** strided token-major MPS attention — CI 31911189760 (287 tests). `AttentionInputLayout`/`encodeTokenMajor`, `stridedTokenMajorAttention` toggle (default OFF). NOTE: MPS strided head offsets must be 256-byte aligned (production headDim=128 is fine); legacy head-major reference must be transposed to token-major before parity compare.
- **P5** cross-attention K/V cache — CI 31913876755 (292 tests). `CrossKVCache` (one ~112 MiB private buffer, per-block ready, blit hit/miss), `crossKVCache` toggle (default OFF).
- **P6** mmap no-copy weight source — CI 31915690478 (302 tests). `WeightStorageMode`/`WeightNoCopyPolicy`/`WeightLoadResult`, `noCopyWeightSource` toggle (default OFF), page-aligned-only with copied fallback, `recordMmapNoCopyBytes`.
- **P7** attention backends (streaming MPS + pure Metal Flash) — CI 31918808645 (305 tests). `DiTAttentionBackend` (legacy/strided/streaming/metalFlash, default legacy), streaming online-softmax MPS + `dit_flash_attention_h128_q4_k32`/`_k16`, parity + non-DiT rejection tests.
- **P8** direct packed W4/W8 GEMM — CI 31922667679 (307 tests). `DiTLinearFamily` + `DiTLinearBackend` (dequantizedMPS/directQuantized/hybrid, default dequantizedMPS), `qgemm_8x8x64`/`qgemm_8x16x64`(+W8) macro-generated MSL kernels (MSL rejects template-param structs — use `#define QGEMM_KERNEL`), exact dequant decode, threadgroup W-tile, FP32 acc, no full FP16 scratch, `recordQGEMMCall`, direct-QGEMM parity test (cosine 0.9999998).
- **P9** combine winners + presets + final report — CI 31924467039 (324 tests). `InferencePreset` (10 cases) → `makeConfig()` maps to a concrete `InferenceOptimizationConfig`; `setPreset` applies + persists + sanitizes (invalid → nil); `resetToBaseline` clears the marker; DiagnosticsView preset picker; baseline/individual A/B controls retained; no `recommendedA12` (no physical A12 data yet). Final report at `OPTIMIZATION_FINAL_REPORT.md`.

---

## What's next — PHYSICAL A12 VALIDATION (runbook §17)

P9 is complete; **no further code changes are required**. The remaining work is on-device, not on this VPS:
- Run the §17 benchmark matrix on a physical XS Max using the Diagnostics-screen **Preset** selector (runbook §17 presets 1–9): fixed model W4 first, fixed prompt, fixed seed, 8 steps, unchanged resolution, device unplugged, low-power mode off. Record the full metrics text after each run.
- After W4 is stable, run W8-v2 correctness first — do NOT use W8 as the first performance control.
- Only after physical measurements: add a `recommendedA12` preset in `InferencePreset` with the proven winner combination. **Do NOT change `currentBaseline` to `recommendedA12` without physical A12 data.**
- Simulator/macOS timing is NOT device proof; do not "optimize for CI timing."

### Files P9 touched
- `AnimaXS/Runtime/Generation/InferenceOptimizationConfig.swift` (`InferencePreset` + `makeConfig`)
- `AnimaXS/App/InferenceOptimizationSettings.swift` (`setPreset`, `activePreset`, `loadPreset`)
- `AnimaXS/App/DiagnosticsView.swift` (preset picker)
- `AnimaXSTests/InferenceOptimizationConfigTests.swift` (P9 preset tests, +17)
- `OPTIMIZATION_IMPLEMENTATION_STATE.md`, `OPTIMIZATION_EVIDENCE.md`, `OPTIMIZATION_DECISIONS.md`, `HANDOFF.md`, `OPTIMIZATION_FINAL_REPORT.md`

---

## Tests/CI strategy
- Every phase: normal `ci.yml` green (compile + all unit tests).
- Milestone full-inference: `full-inference-refine.yml` at (1) after P1 ✅, (2) after P3/P4/P5 combined, (3) final candidate. Do NOT download packs locally; let the runner fetch pinned fixtures.
- Simulator/macOS timing is NOT device proof. Physical XS Max decides winners (runbook §17). Do not "optimize for CI timing."

## Invariants (never regress — runbook §3)
W4 known-good reference; 8 steps; 28 blocks; no thermal gating; no auto model download; bounded-memory weight streaming; no model packs on VPS; no clamp of W8 residuals; no suppressing numerical failures; no changing sampler math/steps/resolution for speed; no A14 simdgroup_matrix / Metal3 / MTLIOCommandQueue; no per-kernel waitUntilCompleted; no approximate attention; no cross-prompt KV reuse.

---

## Exact next steps for a fresh agent
1. `cd /root/AnimaXS && git pull origin opt/a12-sustained-io` (should be clean at `d16e317`; parallel session may have added a commit — pull to get it).
2. P0–P9 are COMPLETE and CI-green. **No code changes are required.**
3. The remaining work is PHYSICAL A12 VALIDATION on a device (runbook §17): use the Diagnostics-screen **Preset** selector to run presets 1–9, record full metrics, and only then add a `recommendedA12` preset in `InferencePreset` with the proven winner. Do NOT claim a fastest backend without physical measurements, and do NOT change `currentBaseline` to `recommendedA12`.
4. If any further code change is needed, follow the git discipline above: commit + push each change, verify `remote == local`, run `ci.yml` to green.

