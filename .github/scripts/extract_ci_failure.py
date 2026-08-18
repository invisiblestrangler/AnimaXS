#!/usr/bin/env python3
"""Extract compact, actionable diagnostics from a locally captured CI command log.

This intentionally operates on a log produced inside the current GitHub Actions
runner. It does not download Actions logs through the API.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
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
    )
]


def usage() -> None:
    raise SystemExit(
        "usage: extract_ci_failure.py LOG LABEL OUTPUT [EXIT_CODE] [COMMAND]"
    )


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
        # The final tail often contains xcodebuild's command-failure summary.
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
