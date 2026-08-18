# AnimaXS GitHub CI Failure Debug Protocol

This is a hard project rule for ChatGPT/planner work on AnimaXS.

## Hard constraint

**Do not attempt to retrieve full GitHub Actions logs through the ChatGPT GitHub connector, `curl`, Actions log-download endpoints, or repeated API workarounds.** Those routes have repeatedly proven unreliable in this environment and have wasted substantial debugging time.

A failed job status by itself is **not** enough evidence to diagnose or change code.

## Reliable failure channel

The primary and authoritative CI-debug channel is a **GitHub bot comment on the pull request**.

For `xcodebuild`/test failures, CI must capture command output locally inside the runner, extract the useful compiler/test diagnostics, and post that excerpt to the PR. The extraction path is intentionally independent of Actions full-log retrieval.

## Required procedure

1. Identify the failed PR/check/run and the exact failing job.
2. Read the PR bot comment first.
3. If the comment contains the concrete compiler/test error, inspect the referenced source at that exact PR revision and diagnose from that evidence.
4. If the comment is missing or insufficient, **do not fetch the full Actions log externally**. Improve/trigger the CI diagnostic action so it reruns the relevant command, captures its output locally, and posts the extracted diagnostics to the PR.
5. Record the exact error text, file/line when available, failing command, PR head SHA, and run ID before modifying code.
6. Implement the smallest fix supported by that evidence.
7. Rerun CI and independently verify the replacement run. If it fails again, repeat through the bot-comment path.

## Never do this

- Guess the root cause from `generic iPhone build: failed` or another status-only signal.
- Spend tool calls trying different Actions full-log download APIs after the bot comment is absent.
- Use `curl` to obtain Actions logs.
- Claim the compiler/test failure is understood before the exact diagnostics have been surfaced.

## CI implementation

`.github/workflows/ci.yml` contains diagnostic jobs for the main CI gates. When a PR job fails, a diagnostic job reruns the relevant command on the same PR checkout, captures its output with `tee`, uses `.github/scripts/extract_ci_failure.py` to produce a compact diagnostic excerpt, and posts/updates a bot comment on the PR.

If that mechanism itself fails, fixing the bot-comment diagnostic path takes priority over guessing at the underlying source failure.
