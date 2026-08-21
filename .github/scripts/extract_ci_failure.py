#!/usr/bin/env python3
"""Extract compact, actionable diagnostics from a locally captured CI command log.

This intentionally operates on a log produced inside the current GitHub Actions
runner. It does not download Actions logs through the API.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
XCRESULT_RE = re.compile(r"(/.+?\.xcresult)(?:\s|$)")
PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\berror:",
        r"\bfatal error:",
        r"::error::",
        r"undefined symbols for architecture",
        r"clang: error:",
        r"\bld: ",
        r"command .* failed",
        r"failed with exit code",
        r"\*\* build failed \*\*",
        r"\*\* test failed \*\*",
        r"test case .* failed",
        r"xctassert",
        r"no such module",
        r"cannot find .* in scope",
        r"ambiguous use of",
        r"failed to emit",
        r"the following build commands failed",
        r"testing failed",
        r"xcresult issue",
        r"xcresult action status",
    )
]


def usage() -> None:
    raise SystemExit(
        "usage: extract_ci_failure.py LOG LABEL OUTPUT [EXIT_CODE] [COMMAND]"
    )


def _xc_value(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("_value")
        if isinstance(raw, (str, int, float, bool)):
            return str(raw)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def _xc_values(value: Any) -> list[Any]:
    if not isinstance(value, dict):
        return []
    values = value.get("_values")
    return values if isinstance(values, list) else []


def _append_xcresult_diagnostics(lines: list[str], label: str, exit_code: str) -> None:
    """Append compact result-bundle diagnostics captured locally on the runner.

    XCTest can report every assertion as passed while the overall test session
    still fails (for example, a runner/session teardown failure). xcodebuild's
    text tail does not always surface that reason, but the .xcresult bundle can.
    This function reads that local bundle with xcresulttool and appends only a
    compact summary/issues section for the PR diagnostic comment.
    """

    if exit_code in ("0", "unknown") or "simulator" not in label.lower():
        return

    candidates: list[Path] = []
    for line in lines:
        for match in XCRESULT_RE.finditer(line):
            candidates.append(Path(match.group(1).strip()))

    if not candidates:
        lines.extend(
            [
                "",
                "XCRESULT ISSUE: xcodebuild failed but no .xcresult path was found in captured output.",
            ]
        )
        return

    result_path = candidates[-1]
    lines.extend(["", f"XCRESULT BUNDLE: {result_path}"])
    if not result_path.exists():
        lines.append(
            "XCRESULT ISSUE: result bundle path was reported by xcodebuild but is not present on the diagnostic runner."
        )
        return

    # Modern xcresulttool summary. Keep the raw JSON compact and bounded so it
    # remains useful even if the schema evolves between Xcode releases.
    try:
        summary = subprocess.run(
            [
                "xcrun",
                "xcresulttool",
                "get",
                "test-results",
                "summary",
                "--path",
                str(result_path),
                "--compact",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if summary.returncode == 0:
            text = summary.stdout.strip()
            if text:
                if len(text) > 12_000:
                    text = text[:12_000] + "… [xcresult summary truncated]"
                lines.append(f"XCRESULT TEST SUMMARY: {text}")
        else:
            detail = (summary.stderr or summary.stdout).strip().replace("\n", " ")
            lines.append(
                f"XCRESULT ISSUE: modern test-results summary command exited {summary.returncode}: {detail[:2000]}"
            )
    except Exception as exc:  # Diagnostic code must never hide the original failure.
        lines.append(f"XCRESULT ISSUE: could not read modern test summary: {exc}")

    # The legacy root object still exposes ActionResult.issue summaries and is
    # useful for session-level failures that are absent from the normal XCTest
    # assertion summary. Xcode 16+ keeps this interface behind --legacy.
    try:
        root_result = subprocess.run(
            [
                "xcrun",
                "xcresulttool",
                "get",
                "--legacy",
                "--path",
                str(result_path),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if root_result.returncode != 0:
            detail = (root_result.stderr or root_result.stdout).strip().replace("\n", " ")
            lines.append(
                f"XCRESULT ISSUE: legacy issue-summary command exited {root_result.returncode}: {detail[:2000]}"
            )
            return

        root = json.loads(root_result.stdout)
        actions = _xc_values(root.get("actions")) if isinstance(root, dict) else []
        issue_count = 0
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            action_result = action.get("actionResult")
            if not isinstance(action_result, dict):
                continue

            status = _xc_value(action_result.get("status"))
            if status:
                lines.append(f"XCRESULT ACTION STATUS [{action_index}]: {status}")

            issues = action_result.get("issues")
            if not isinstance(issues, dict):
                continue

            for collection_name, collection_label in (
                ("errorSummaries", "error"),
                ("testFailureSummaries", "test-failure"),
            ):
                for issue in _xc_values(issues.get(collection_name)):
                    if not isinstance(issue, dict):
                        continue
                    message = _xc_value(issue.get("message")) or "(no message)"
                    issue_type = _xc_value(issue.get("issueType"))
                    location = None
                    document_location = issue.get("documentLocationInCreatingWorkspace")
                    if isinstance(document_location, dict):
                        location = _xc_value(document_location.get("url"))

                    extras = []
                    if issue_type:
                        extras.append(issue_type)
                    if location:
                        extras.append(location)
                    suffix = f" ({'; '.join(extras)})" if extras else ""
                    lines.append(
                        f"XCRESULT ISSUE [{collection_label}]: {message}{suffix}"
                    )
                    issue_count += 1

        if issue_count == 0:
            lines.append(
                "XCRESULT ISSUE: no ActionResult errorSummaries or testFailureSummaries were present in the result bundle."
            )
    except Exception as exc:  # Keep bot output useful even if xcresult schema changes.
        lines.append(f"XCRESULT ISSUE: could not extract ActionResult issues: {exc}")


def main() -> None:
    if len(sys.argv) < 4:
        usage()

    log_path = Path(sys.argv[1])
    label = sys.argv[2]
    output_path = Path(sys.argv[3])
    exit_code = sys.argv[4] if len(sys.argv) >= 5 else "unknown"
    command = sys.argv[5] if len(sys.argv) >= 6 else "(see workflow)"

    if log_path.exists():
        raw = log_path.read_text(errors="replace")
        lines = [ANSI_RE.sub("", line.rstrip()) for line in raw.splitlines()]
    else:
        lines = [f"Diagnostic capture file was not created: {log_path}"]

    _append_xcresult_diagnostics(lines, label, exit_code)

    hit_indices: list[int] = []
    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in PATTERNS):
            hit_indices.append(index)

    selected: set[int] = set()
    if hit_indices:
        for index in hit_indices:
            start = max(0, index - 4)
            end = min(len(lines), index + 6)
            selected.update(range(start, end))
        # The final tail often contains xcodebuild's command-failure summary or
        # the locally extracted xcresult session diagnostics.
        selected.update(range(max(0, len(lines) - 30), len(lines)))
    else:
        # If no known signature matched, preserve a useful tail rather than
        # pretending we found a cause.
        selected.update(range(max(0, len(lines) - 140), len(lines)))

    ordered = sorted(selected)
    excerpt_lines: list[str] = []
    previous = -2
    for index in ordered:
        if index > previous + 1 and excerpt_lines:
            excerpt_lines.append("…")
        excerpt_lines.append(f"{index + 1:>6}: {lines[index]}")
        previous = index

    # GitHub issue comments have a hard size limit. Keep plenty of room for
    # metadata added by the posting step.
    excerpt_lines = excerpt_lines[:360]
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > 45_000:
        excerpt = excerpt[:45_000] + "\n… [diagnostic excerpt truncated]"

    reproduced = exit_code not in ("0", "unknown")
    title_suffix = "" if reproduced else " — diagnostic rerun did not reproduce"
    body = f"""### CI FAILURE — {label}{title_suffix}

**Diagnostic rerun exit:** `{exit_code}`  
**Command:** `{command}`

This excerpt was captured **inside a GitHub Actions diagnostic rerun on the same PR checkout**. It does not use the Actions full-log download API, `curl`, or an external log fetch.

```text
{excerpt}
```
"""
    output_path.write_text(body)


if __name__ == "__main__":
    main()
