# AnimaXS — Next Execution Agent Handoff

Updated 2026-08-12 (final completion stage: L002 verified, L003 PASS, docs reconciled).

## Repository state

```
branch: main
HEAD: 494ffb1 (docs: record L001 gate commit hash in TODO)
origin/main: 494ffb1
working-tree: clean except untracked scripts/oracle_out/block0/ (gitignored oracle artifacts — do not commit)
```

## Final CI evidence

```
Normal CI (on HEAD 494ffb1): run 31639624730 (push) — verify green (should mirror
31634877232: project-consistency ✓, iphone-build ✓, simulator-tests ✓ 151 tests /
13 expected fixture-gated skips / 0 failures).
L002: model-assets-v1 — all 3 packs unauthenticated-downloaded and SHA-256-verified
(D070); manifest/LICENSE/NOTICE byte-identical to committed copies.
L003: run 31639624310 (commit 494ffb1) SUCCESS — Metal smoke PASS, 3 packs verified,
FULL_INFERENCE=PASS: final latent cosine 0.6946 (≥ 0.65 floor, D057), final RGB
cosine 0.7035 (≥ 0.65 gate, D074). Deterministic (identical to calibration
31638695924). Full pipeline on simulator: Qwen 6.62s, adapter 2.19s, 8-step
diffusion 138.38s, VAE 9.71s, total 161.28s.
```

## What was fixed this stage

1. **full-inference.yml Bash 3.2 bug** (commit `3dae37d`): `declare -A` is unsupported
   on macOS Bash 3.2 — replaced with a `printf | while read name size sha` loop;
   hardened curl (`--fail --location --retry --retry-all-errors`), exact size+SHA checks.
2. **L003 env-injection failure** (commit `adbad52`): xcodebuild does not reliably
   forward step env vars to the simulator's test process; adopted the proven
   parity-workflow pattern — copy verified packs into `AnimaXSTests/Fixtures/Case1Binary/`
   and regenerate the project (pinned XcodeGen) so `FullInferenceTests` finds them as
   BUNDLED test resources.
3. **L001 wrong-prompt bug** (commit `f72a962`, D073): test used a short prompt while
   the canonical golden used the long prompt — first full run was apples-to-oranges
   (latent 0.488 / RGB 0.438). Now uses the exact canonical prompt.
4. **RGB gate calibration** (commit `4f84b7a`, D074): measured RGB cosine 0.7035 with
   correct conditioning; gate set to ≥ 0.65 (small margin below measured; a broken
   pipeline drops to ~0.44).
5. **RGB8 canonical fixture** (commit `3dae37d`, D071): `case1_decoded_rgb8.bin`
   (SHA `a396c4ae…7019`) derived from the SHA-verified NPZ via the exact production
   display transform; `FullInferenceTests` now reports FULL_RGB_COSINE/RMSE/MAE/MAXABS.

## Key decisions (DECISIONS.md)

- D070: L002 unauthenticated release verification evidence.
- D071: RGB8 regression fixture + initial gate.
- D072: DiagnosticsTests environmental flake (leftover packs on runner VM) — not a regression.
- D073: wrong-prompt root cause of the first L003 low metrics.
- D074: RGB gate calibrated to 0.65 from measured 0.7035.

## What cannot be proven in this phase

```
physical iPhone XS Max launch
real A12 GPU behavior
actual A12 Metal throughput
actual device peak RSS
iOS jetsam behavior
thermal behavior
real-device GPU memory reclamation timing
real-device second-generation memory stability
```

Progress files say: **CI-validated; physical A12 acceptance pending.**

## Remaining tasks (ordered by dependency)

1. **Physical A12 acceptance** — Build in Xcode 26.3, install on iPhone XS Max, record
   stage timings, peak memory, second-generation stability, thermal behavior. Tune VAE
   tile size (J005) only if device measurements justify it. Nothing else CI-verifiable
   remains open.

## First command for next agent

```bash
cd /root/AnimaXS
git pull --rebase origin main
gh run list --limit 5
# Confirm the push CI on the final HEAD is green, then proceed to physical-device
# acceptance per DEVICE_TESTS.md (no further CI work is pending).
```
