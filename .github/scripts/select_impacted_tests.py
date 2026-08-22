#!/usr/bin/env python3
"""Choose a conservative fail-fast XCTest subset from changed repository paths.

This selector never changes the definition of CI green: the authoritative full
simulator suite still runs after the subset succeeds. Its only purpose is to
surface obvious subsystem regressions earlier on a failing commit.

Unknown app/runtime changes deliberately fall back to the full suite. CI/docs
changes alone use a tiny smoke subset because the generic device build and the
full simulator pass still follow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TEST_TARGET = "AnimaXSTests"
MAX_TARGETED_CLASSES = 14

EXACT_SOURCE_MAP: dict[str, set[str]] = {
    "AnimaXS/Runtime/Metal/AttentionExecutor.swift": {
        "AttentionExecutorTests",
        "DiTBlockExecutorTests",
        "MetalExecutionTests",
    },
    "AnimaXS/Runtime/Metal/DiTBlockExecutor.swift": {
        "AttentionExecutorTests",
        "DiTBlockExecutorTests",
        "DiTBlockTests",
        "MetalExecutionTests",
    },
    "AnimaXS/Runtime/Generation/GenerationEngine.swift": {
        "GenerationCoordinatorTests",
        "GenerationEligibilityTests",
        "LLMAdapterTests",
        "SmokeTests",
        "TokenizerParityTests",
    },
    "AnimaXS/ContentView.swift": {
        "GenerationCoordinatorTests",
        "SmokeTests",
    },
    "AnimaXS/AnimaXSApp.swift": {"SmokeTests"},
}

PREFIX_SOURCE_MAP: tuple[tuple[str, set[str]], ...] = (
    (
        "AnimaXS/Runtime/Generation/",
        {
            "GenerationCoordinatorTests",
            "GenerationEligibilityTests",
            "GenerationMetricsTests",
            "SmokeTests",
        },
    ),
    (
        "AnimaXS/Runtime/Text/",
        {"LLMAdapterTests", "QwenEncoderMetalTests", "SmokeTests", "TokenizerParityTests"},
    ),
    (
        "AnimaXS/Runtime/Metal/",
        {
            "AttentionExecutorTests",
            "DiTBlockExecutorTests",
            "LinearExecutorTests",
            "MetalExecutionTests",
            "SmokeTests",
        },
    ),
    (
        "AnimaXS/Runtime/ModelStore/",
        {"GenerationEligibilityTests", "ModelStoreTests", "SmokeTests"},
    ),
    (
        "AnimaXS/Runtime/Sampler/",
        {"NumericalFailureTests", "NumericalMonitorTests", "SmokeTests"},
    ),
    (
        "AnimaXS/Runtime/VAE/",
        {"SmokeTests", "VAEDecoderLocatorTests", "VAEPrimitiveTests", "Wan21LatentFormatTests"},
    ),
    (
        "AnimaXS/Runtime/Animapk/",
        {
            "AnimapkParsingTests",
            "AnimapkRangeLocatorTests",
            "QuantDecoderTests",
            "SmokeTests",
            "WeightStreamerTests",
        },
    ),
    (
        "AnimaXS/Runtime/Diagnostics/",
        {"DiagnosticsRunMarkerTests", "DiagnosticsTests", "SmokeTests"},
    ),
)

FULL_SUITE_PREFIXES = (
    "AnimaXS/Runtime/ANE/",
    "AnimaXSTests/Support/",
    "AnimaXSTests/Fixtures/",
    "AnimaXS.xcodeproj/",
)
FULL_SUITE_EXACT = {
    "project.yml",
    "Package.resolved",
}


def changed_paths(base: str | None, head: str) -> list[str]:
    base = (base or "").strip()
    if not base:
        base = f"{head}^"
    command = ["git", "diff", "--name-only", "--diff-filter=ACMR", base, head, "--"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # A missing/invalid base must never make the selector less safe.
        print(
            f"warning: could not diff {base}..{head}; falling back to full suite: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return ["__FULL_SUITE_FALLBACK__"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def selection_for(paths: list[str]) -> tuple[str, list[str], list[str]]:
    tests: set[str] = {"SmokeTests"}
    reasons: list[str] = []

    if not paths:
        return "full", [], ["no changed paths resolved"]

    for path in paths:
        if path == "__FULL_SUITE_FALLBACK__":
            return "full", [], ["git diff unavailable"]
        if path in FULL_SUITE_EXACT or path.endswith(".metal"):
            return "full", [], [f"shared build/Metal surface changed: {path}"]
        if any(path.startswith(prefix) for prefix in FULL_SUITE_PREFIXES):
            return "full", [], [f"shared/ANE/test-support surface changed: {path}"]

        if path.startswith("AnimaXSTests/") and path.endswith("Tests.swift"):
            tests.add(Path(path).stem)
            reasons.append(f"changed test: {path}")
            continue

        mapped = EXACT_SOURCE_MAP.get(path)
        if mapped is not None:
            tests.update(mapped)
            reasons.append(f"exact source map: {path}")
            continue

        prefix_match = False
        for prefix, mapped_tests in PREFIX_SOURCE_MAP:
            if path.startswith(prefix):
                tests.update(mapped_tests)
                reasons.append(f"subsystem map: {path}")
                prefix_match = True
                break
        if prefix_match:
            continue

        # Workflow/docs/scripts do not broaden the fail-fast subset. They are
        # still validated by the authoritative full suite on every green head.
        if path.startswith(".github/") or path.endswith(".md"):
            reasons.append(f"CI/docs-only path: {path}")
            continue

        # UI resources can be covered by compile + smoke in the fast pass.
        if path.startswith("AnimaXS/") and not path.endswith(".swift"):
            reasons.append(f"app resource: {path}")
            continue

        # Any unclassified Swift/runtime source is too risky for prediction.
        if path.startswith("AnimaXS/"):
            return "full", [], [f"unmapped app/runtime source changed: {path}"]

        # Unknown repository surfaces are conservative too.
        return "full", [], [f"unmapped repository path changed: {path}"]

    selected = sorted(tests)
    if len(selected) > MAX_TARGETED_CLASSES:
        return "full", [], [
            f"targeted set expanded to {len(selected)} classes; full suite is more efficient"
        ]
    return "targeted", selected, reasons


def write_outputs(path: Path, mode: str, tests: list[str], reasons: list[str]) -> None:
    flags = " ".join(f"-only-testing:{TEST_TARGET}/{name}" for name in tests)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"mode={mode}\n")
        handle.write(f"flags={flags}\n")
        handle.write(f"classes={','.join(tests)}\n")
        handle.write(f"reason={' | '.join(reasons[:8])}\n")


def self_test() -> None:
    cases = [
        (["AnimaXS/Runtime/Metal/AttentionExecutor.swift"], "targeted", "AttentionExecutorTests"),
        (["AnimaXS/Runtime/ANE/ANEW8.swift"], "full", None),
        (["AnimaXS/Shaders/Kernels.metal"], "full", None),
        (["AnimaXSTests/GenerationCoordinatorTests.swift"], "targeted", "GenerationCoordinatorTests"),
        (["AnimaXS/ContentView.swift", ".github/workflows/ci.yml"], "targeted", "SmokeTests"),
        (["AnimaXS/UnknownRuntimeThing.swift"], "full", None),
    ]
    for paths, expected_mode, expected_test in cases:
        mode, tests, _ = selection_for(paths)
        assert mode == expected_mode, (paths, mode, tests)
        if expected_test is not None:
            assert expected_test in tests, (paths, tests)
    print("select_impacted_tests self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error("--output is required unless --self-test is used")

    paths = changed_paths(args.base, args.head)
    mode, tests, reasons = selection_for(paths)
    print(f"Changed paths ({len(paths)}):")
    for path in paths:
        print(f"  {path}")
    print(f"Fail-fast mode: {mode}")
    if tests:
        print("Fail-fast classes: " + ", ".join(tests))
    if reasons:
        print("Selection reason: " + " | ".join(reasons[:8]))
    write_outputs(Path(args.output), mode, tests, reasons)


if __name__ == "__main__":
    main()
