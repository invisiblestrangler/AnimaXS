# AnimaXS GitHub CI Failure Debug Protocol

This is a hard project rule for ChatGPT/planner work on AnimaXS.

## Hard constraint

**Do not attempt to retrieve full GitHub Actions logs through the ChatGPT GitHub connector, `curl`, Actions log-download endpoints, or repeated API workarounds.** Those routes have repeatedly proven unreliable in this environment and have wasted substantial debugging time.

A failed job status by itself is **not** enough evidence to diagnose or change code.

## Reliable failure channel

The primary and authoritative CI-debug channel is a **GitHub bot comment on the pull request**.

For `xcodebuild`/test failures, the original CI job must capture command output locally on the same runner that executed the failing command. Simulator runs must also preserve the original `.xcresult` bundle. The job then extracts compact compiler/test/crash diagnostics and posts that evidence to the PR **before returning the original non-zero result**.

The diagnostic path must not rerun the failed build or simulator suite merely to obtain logs. The original runner already owns the authoritative log and result bundle.

## Required procedure

1. Identify the failed PR/check/run and exact failing gate from status metadata only.
2. Read the PR bot comment first.
3. If the comment contains the concrete compiler/test/crash error, inspect the referenced source at that exact PR revision and diagnose from that evidence.
4. If the comment is missing or insufficient, **do not fetch the full Actions log externally and do not blindly rerun the expensive command for diagnostics**. Improve the same-run capture/extractor/comment path so the original job posts enough local evidence.
5. Record the exact error text, file/line when available, failing command, PR head SHA, and run ID before modifying source code.
6. Implement the smallest fix supported by that evidence.
7. Rerun CI and independently verify the replacement run. If it fails again, repeat through the bot-comment path.

## Never do this

- Guess the root cause from `generic iPhone build: failed`, `simulator tests: failed`, or another status-only signal.
- Spend tool calls trying Actions full-log download APIs after the bot comment is absent.
- Use `curl` to obtain Actions logs.
- Use the GitHub connector to retrieve Actions job logs for CI diagnosis.
- Launch a second full simulator/build run solely to reproduce a failure that the original runner could have captured.
- Claim a compiler/test failure is understood before exact diagnostics have been surfaced.

## CI implementation

`.github/workflows/ci.yml` keeps diagnostics inside each primary CI gate:

- the command is piped through `tee` on the original runner;
- its real exit status is preserved;
- simulator failures retain an explicit `.xcresult` path;
- `.github/scripts/extract_ci_failure.py` generates a compact diagnostic excerpt from those local files;
- the same job posts/updates a bot comment on the PR;
- a final step returns the preserved failure status so the gate remains red.

There are **no separate diagnostic rerun jobs**.

Simulator testing also has a conservative fail-fast lane selected by `.github/scripts/select_impacted_tests.py`. Relevant test classes run first from the already-built test bundle. If they fail, CI stops early and comments immediately. If they pass, the authoritative **full simulator suite still runs on every green head**. Unknown/shared low-level changes bypass the shortcut and go directly to the full suite.

If the same-run comment mechanism itself fails, fixing that diagnostic path takes priority over guessing at the underlying source failure.
