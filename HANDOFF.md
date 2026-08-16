# AnimaXS A12 Reliability + Sustained-Performance Optimization — HANDOFF

For: the next agent continuing this implementation. Read this fully before touching anything.

---

## TL;DR status
- **Phases P0, P1, P2, P3, P4, P5, P6, P7, P8 are COMPLETE and CI-GREEN** on branch `opt/a12-sustained-io`. **Only P9 (combine winners + presets + final report) remains.**
- **Current HEAD: `bffc03a`** (P8 complete + state/evidence). Working tree CLEAN, `origin/opt/a12-sustained-io` == local `bffc03a`. Draft PR #17 open.
- Last green CI: run `31922667679` (P8) — **307 tests, 14 expected skips, 0 failures**.
- The full implementation plan is the runbook file: `/root/.hermes/cache/documents/doc_e617196af2a6_ANIMAXS_A12_RELIABILITY_PERFORMANCE_IMPLEMENTATION_PLAN.md` (also sent in-conversation). **It is the authoritative spec.** Repo: `/root/AnimaXS`, branch `opt/a12-sustained-io`.

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

---

## What's left — P9 (runbook §14, the LAST phase)

### P9 — combine winners + presets (runbook §14) + final report (§23)
- Add `enum InferencePreset: String, CaseIterable`: `baseline`, `current1024Control`, `fusedTraffic`, `stridedMPS`, `stridedMPSKV`, `noCopyCandidate`, `streamingMPSCandidate`, `metalFlashCandidate`, `directQGEMMCandidate`, `allCandidate`. Each preset maps to a specific combination of the existing config toggles/backends (linear 1024 + attention 1024 + directMPSIO + ping-pong for control; add fused; add strided; add KV; add no-copy; etc.).
- Keep ALL individual backends + A/B controls (don't remove the per-toggle/backend controls).
- Add a diagnostics preset selector + reset-to-baseline. **Do NOT change `currentBaseline` to a `recommendedA12` preset** — no physical A12 data exists yet. Add `recommendedA12` only AFTER the user tests on the device.
- Preset config persistence: `InferenceOptimizationSettings` needs a `setPreset(_:)` that applies the mapping + sanitizes invalid persisted enums; reset restores baseline. Immutable snapshot-at-Generate behavior must remain.
- Tests: preset selection maps to the right combination; reset-to-baseline restores baseline; invalid persisted enum sanitizes; snapshot unchanged mid-run.
- Then produce the **final report (runbook §23)**: repo/branch/final HEAD/correctness fixes/backends/tests/traffic changes/known limitations/physical XS Max benchmark matrix (runbook §17 presets 1-9 with fixed model/prompt/seed/steps/resolution, unplugged, low-power off)/state files/final recommendation. **Do NOT claim any backend is fastest without physical A12 measurements.**
- Ensure `OPTIMIZATION_IMPLEMENTATION_STATE.md` ends in a "ready for physical A12 validation" state; tree clean; branch pushed; `OPTIMIZATION_EVIDENCE.md` has all run IDs.

### Files P9 likely touches
- `AnimaXS/Runtime/Generation/InferenceOptimizationConfig.swift` (+ preset mapping)
- `AnimaXS/App/InferenceOptimizationSettings.swift`, `AnimaXS/App/DiagnosticsView.swift` (preset selector + apply)
- `AnimaXSTests/InferenceOptimizationConfigTests.swift` / `GenerationMetricsTests.swift` (preset tests)
- `OPTIMIZATION_IMPLEMENTATION_STATE.md`, `OPTIMIZATION_EVIDENCE.md`, `OPTIMIZATION_DECISIONS.md`, `HANDOFF.md`, and the final report file

---

## Tests/CI strategy
- Every phase: normal `ci.yml` green (compile + all unit tests).
- Milestone full-inference: `full-inference-refine.yml` at (1) after P1 ✅, (2) after P3/P4/P5 combined, (3) final candidate. Do NOT download packs locally; let the runner fetch pinned fixtures.
- Simulator/macOS timing is NOT device proof. Physical XS Max decides winners (runbook §17). Do not "optimize for CI timing."

## Invariants (never regress — runbook §3)
W4 known-good reference; 8 steps; 28 blocks; no thermal gating; no auto model download; bounded-memory weight streaming; no model packs on VPS; no clamp of W8 residuals; no suppressing numerical failures; no changing sampler math/steps/resolution for speed; no A14 simdgroup_matrix / Metal3 / MTLIOCommandQueue; no per-kernel waitUntilCompleted; no approximate attention; no cross-prompt KV reuse.

---

## Exact next steps for a fresh agent
1. `cd /root/AnimaXS && git pull origin opt/a12-sustained-io` (should be clean at `bffc03a`; parallel session may have added a commit — pull to get it).
2. Read runbook §14 (P9) + §17 (device benchmark matrix) + §23 (final report).
3. Implement P9 presets + tests, get `ci.yml` green. Commit + push, verify remote==local.
4. Write the final §23 report + update state files to "ready for physical A12 validation" + update this HANDOFF.md. Tree clean, branch pushed.
