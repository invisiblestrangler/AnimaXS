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
import tempfile
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
        r"xcresult test detail",
        r"xcresult activity",
        r"xcresult console",
        r"xcresult crash",
    )
]

CRASH_TERMS = (
    "negpip",
    "sigabrt",
    "abort",
    "crash",
    "fatal",
    "assert",
    "exception",
    "terminating",
    "termination reason",
    "mps",
    "metal validation",
    "validatefunctionarguments",
)


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


def _run_xcresulttool(arguments: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["xcrun", "xcresulttool", *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _bounded(text: str, limit: int = 12_000) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "… [truncated]"


def _collect_interesting_strings(
    value: Any,
    path: str = "$",
    hits: list[str] | None = None,
    limit: int = 80,
) -> list[str]:
    """Collect crash-relevant string values from arbitrary xcresult JSON."""

    if hits is None:
        hits = []
    if len(hits) >= limit:
        return hits

    if isinstance(value, dict):
        for key, child in value.items():
            _collect_interesting_strings(child, f"{path}.{key}", hits, limit)
            if len(hits) >= limit:
                break
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_interesting_strings(child, f"{path}[{index}]", hits, limit)
            if len(hits) >= limit:
                break
    elif isinstance(value, str):
        lower = value.lower()
        if any(term in lower for term in CRASH_TERMS):
            compact = " ".join(value.split())
            hits.append(f"{path}: {_bounded(compact, 1200)}")

    return hits


def _append_test_failure_details(
    lines: list[str], result_path: Path, summary_payload: Any
) -> None:
    if not isinstance(summary_payload, dict):
        return
    failures = summary_payload.get("testFailures")
    if not isinstance(failures, list):
        return

    for failure in failures[:3]:
        if not isinstance(failure, dict):
            continue
        test_id = failure.get("testIdentifierURL") or failure.get("testIdentifierString")
        if not isinstance(test_id, str) or not test_id:
            continue

        details = _run_xcresulttool(
            [
                "get",
                "test-results",
                "test-details",
                "--path",
                str(result_path),
                "--test-id",
                test_id,
                "--compact",
            ]
        )
        if details.returncode == 0 and details.stdout.strip():
            lines.append(
                f"XCRESULT TEST DETAIL [{test_id}]: {_bounded(details.stdout, 14_000)}"
            )
        else:
            detail = (details.stderr or details.stdout).strip().replace("\n", " ")
            lines.append(
                f"XCRESULT ISSUE: test-details for {test_id} exited {details.returncode}: {_bounded(detail, 2000)}"
            )

        activities = _run_xcresulttool(
            [
                "get",
                "test-results",
                "activities",
                "--path",
                str(result_path),
                "--test-id",
                test_id,
                "--compact",
            ]
        )
        if activities.returncode == 0 and activities.stdout.strip():
            try:
                payload = json.loads(activities.stdout)
                hits = _collect_interesting_strings(payload)
            except Exception:
                hits = []
            if hits:
                for hit in hits:
                    lines.append(f"XCRESULT ACTIVITY [{test_id}]: {hit}")
            else:
                lines.append(
                    f"XCRESULT ACTIVITY [{test_id}]: {_bounded(activities.stdout, 6000)}"
                )
        elif activities.returncode != 0:
            detail = (activities.stderr or activities.stdout).strip().replace("\n", " ")
            lines.append(
                f"XCRESULT ISSUE: activities for {test_id} exited {activities.returncode}: {_bounded(detail, 2000)}"
            )


def _append_console_crash_messages(lines: list[str], result_path: Path) -> None:
    console = _run_xcresulttool(
        ["get", "log", "--path", str(result_path), "--type", "console", "--compact"],
        timeout=45,
    )
    if console.returncode != 0:
        detail = (console.stderr or console.stdout).strip().replace("\n", " ")
        lines.append(
            f"XCRESULT ISSUE: console-log extraction exited {console.returncode}: {_bounded(detail, 2000)}"
        )
        return
    if not console.stdout.strip():
        return

    try:
        payload = json.loads(console.stdout)
        hits = _collect_interesting_strings(payload, limit=120)
    except Exception:
        hits = []

    if hits:
        for hit in hits:
            lines.append(f"XCRESULT CONSOLE: {hit}")
        return

    # Keep a bounded fallback if the output schema is not JSON or no known
    # signature matched; this is still local runner data posted by the bot.
    raw = console.stdout
    lowered = raw.lower()
    indexes = [lowered.find(term) for term in CRASH_TERMS]
    indexes = [index for index in indexes if index >= 0]
    if indexes:
        start = max(0, min(indexes) - 2000)
        lines.append(f"XCRESULT CONSOLE: {_bounded(raw[start:start + 12_000], 12_000)}")


def _json_documents(text: str) -> list[Any]:
    try:
        return [json.loads(text)]
    except Exception:
        pass

    documents: list[Any] = []
    for raw_line in text.splitlines():
        candidate = raw_line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            documents.append(json.loads(candidate))
        except Exception:
            continue
    return documents


def _append_structured_crash_document(lines: list[str], path: Path, payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    proc_name = payload.get("procName") or payload.get("process")
    exception = payload.get("exception")
    termination = payload.get("termination")
    asi = payload.get("asi") or payload.get("applicationSpecificInformation")
    faulting = payload.get("faultingThread")

    if not any(value is not None for value in (proc_name, exception, termination, asi, faulting)):
        return False

    if proc_name is not None:
        lines.append(f"XCRESULT CRASH [{path.name}] process: {_bounded(str(proc_name), 1000)}")
    if exception is not None:
        lines.append(
            f"XCRESULT CRASH [{path.name}] exception: {_bounded(json.dumps(exception, ensure_ascii=False), 4000)}"
        )
    if termination is not None:
        lines.append(
            f"XCRESULT CRASH [{path.name}] termination: {_bounded(json.dumps(termination, ensure_ascii=False), 4000)}"
        )
    if asi is not None:
        lines.append(
            f"XCRESULT CRASH [{path.name}] application-info: {_bounded(json.dumps(asi, ensure_ascii=False), 5000)}"
        )
    if faulting is not None:
        lines.append(f"XCRESULT CRASH [{path.name}] faulting-thread: {faulting}")

    threads = payload.get("threads")
    if isinstance(faulting, int) and isinstance(threads, list) and 0 <= faulting < len(threads):
        thread = threads[faulting]
        if isinstance(thread, dict):
            frames = thread.get("frames")
            if isinstance(frames, list):
                for index, frame in enumerate(frames[:28]):
                    if not isinstance(frame, dict):
                        continue
                    symbol = frame.get("symbol") or frame.get("sourceFile") or "(unsymbolicated)"
                    image = frame.get("imageIndex")
                    offset = frame.get("imageOffset")
                    lines.append(
                        f"XCRESULT CRASH FRAME {index}: {symbol} image={image} offset={offset}"
                    )
    return True


def _append_exported_crash_diagnostics(lines: list[str], result_path: Path) -> None:
    """Export xcresult diagnostics locally and summarize crash reports only."""

    try:
        with tempfile.TemporaryDirectory(prefix="animaxs-xcresult-") as temporary:
            output_path = Path(temporary) / "diagnostics"
            exported = _run_xcresulttool(
                [
                    "export",
                    "diagnostics",
                    "--path",
                    str(result_path),
                    "--output-path",
                    str(output_path),
                ],
                timeout=45,
            )
            if exported.returncode != 0:
                detail = (exported.stderr or exported.stdout).strip().replace("\n", " ")
                lines.append(
                    f"XCRESULT ISSUE: diagnostics export exited {exported.returncode}: {_bounded(detail, 2000)}"
                )
                return

            files = [path for path in output_path.rglob("*") if path.is_file()]
            files.sort(
                key=lambda path: (
                    0 if path.suffix.lower() in {".ips", ".crash"} else 1,
                    path.name,
                )
            )

            emitted = 0
            for path in files[:80]:
                if emitted >= 80:
                    break
                try:
                    if path.stat().st_size > 8_000_000:
                        continue
                    text = path.read_text(errors="replace")
                except Exception:
                    continue

                structured = False
                for document in _json_documents(text):
                    if _append_structured_crash_document(lines, path, document):
                        structured = True
                        emitted += 1
                        if emitted >= 80:
                            break
                if structured or emitted >= 80:
                    continue

                lower = text.lower()
                if not any(term in lower for term in CRASH_TERMS):
                    continue

                source_lines = text.splitlines()
                matched = [
                    index
                    for index, source_line in enumerate(source_lines)
                    if any(term in source_line.lower() for term in CRASH_TERMS)
                ]
                if not matched and len(text) <= 12_000:
                    lines.append(f"XCRESULT CRASH [{path.name}]: {_bounded(text, 12_000)}")
                    emitted += 1
                    continue

                chosen: set[int] = set()
                for index in matched[:12]:
                    chosen.update(range(max(0, index - 3), min(len(source_lines), index + 5)))
                if chosen:
                    lines.append(f"XCRESULT CRASH FILE: {path.name}")
                    for index in sorted(chosen)[:80]:
                        lines.append(
                            f"XCRESULT CRASH [{path.name}:{index + 1}]: {_bounded(source_lines[index], 1200)}"
                        )
                    emitted += 1
    except Exception as exc:
        lines.append(f"XCRESULT ISSUE: could not export crash diagnostics: {exc}")


def _append_xcresult_diagnostics(lines: list[str], label: str, exit_code: str) -> None:
    """Append compact result-bundle diagnostics captured locally on the runner.

    XCTest can report every assertion as passed while the overall test session
    still fails (for example, a test-process crash followed by a retry). The
    .xcresult bundle is authoritative for that hidden session state.
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

    summary_payload: Any = None
    try:
        summary = _run_xcresulttool(
            [
                "get",
                "test-results",
                "summary",
                "--path",
                str(result_path),
                "--compact",
            ]
        )
        if summary.returncode == 0:
            text = summary.stdout.strip()
            if text:
                lines.append(f"XCRESULT TEST SUMMARY: {_bounded(text, 12_000)}")
                try:
                    summary_payload = json.loads(text)
                except Exception:
                    summary_payload = None
        else:
            detail = (summary.stderr or summary.stdout).strip().replace("\n", " ")
            lines.append(
                f"XCRESULT ISSUE: modern test-results summary command exited {summary.returncode}: {_bounded(detail, 2000)}"
            )
    except Exception as exc:
        lines.append(f"XCRESULT ISSUE: could not read modern test summary: {exc}")

    try:
        _append_test_failure_details(lines, result_path, summary_payload)
    except Exception as exc:
        lines.append(f"XCRESULT ISSUE: could not read failing-test details: {exc}")

    try:
        _append_console_crash_messages(lines, result_path)
    except Exception as exc:
        lines.append(f"XCRESULT ISSUE: could not read console crash messages: {exc}")

    try:
        _append_exported_crash_diagnostics(lines, result_path)
    except Exception as exc:
        lines.append(f"XCRESULT ISSUE: could not summarize exported diagnostics: {exc}")

    # The legacy root object exposes ActionResult issue summaries and remains a
    # compact cross-check for session failures. Xcode 16+ keeps it behind
    # --legacy while the newer test-results commands provide richer details.
    try:
        root_result = _run_xcresulttool(
            ["get", "--legacy", "--path", str(result_path), "--format", "json"]
        )
        if root_result.returncode != 0:
            detail = (root_result.stderr or root_result.stdout).strip().replace("\n", " ")
            lines.append(
                f"XCRESULT ISSUE: legacy issue-summary command exited {root_result.returncode}: {_bounded(detail, 2000)}"
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
    except Exception as exc:
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
        # the locally extracted xcresult session/crash diagnostics.
        selected.update(range(max(0, len(lines) - 60), len(lines)))
    else:
        selected.update(range(max(0, len(lines) - 140), len(lines)))

    ordered = sorted(selected)
    if len(ordered) > 360:
        ordered = ordered[:180] + ordered[-180:]

    excerpt_lines: list[str] = []
    previous = -2
    for index in ordered:
        if index > previous + 1 and excerpt_lines:
            excerpt_lines.append("…")
        excerpt_lines.append(f"{index + 1:>6}: {lines[index]}")
        previous = index

    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > 45_000:
        excerpt = (
            excerpt[:22_000]
            + "\n… [diagnostic excerpt middle truncated] …\n"
            + excerpt[-22_000:]
        )

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
