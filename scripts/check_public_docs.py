"""Guard public docs and package code against private operations leakage."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "mkdocs.yml",
    *sorted((ROOT / "docs").glob("**/*.md")),
]
PACKAGE_PATHS = sorted((ROOT / "src" / "xcalib").glob("**/*.py"))

DOC_DISALLOWED = {
    r"\bpush-hub\b": "write-side model Hub command",
    r"\bpush-dataset\b": "write-side dataset Hub command",
    r"\bpush_to_hub\b": "write-side Hub API",
    r"\bupload(?:ing|s|ed)?\b": "write-side artifact operation",
    r"\bHF_TOKEN\b": "token environment variable",
    r"\bXCALIB_ENABLE_HUB_WRITE\b": "write-side Hub command switch",
    r"\bxcalib-utc\b": "private UTC weights repo name",
    r"\bXCALIB_HF_PRIVATE_REPO\b": "private Hub routing env var",
    r"\bprivate mirror\b": "private mirror workflow detail",
    r"\bLab side\b": "internal maintainer wording",
    r"\bmaintainer\b": "internal maintainer wording",
    r"\bwrite token\b": "private token permission detail",
    r"\bread token\b": "private token permission detail",
    r"\bpartner caches\b": "private partner delivery detail",
}

PACKAGE_DISALLOWED = {
    r"\bpush_weights_to_hub\b": "write-side model Hub API",
    r"\bpush_dataset\b": "write-side dataset Hub API",
    r"\bpush_to_hub\b": "write-side Matcher Hub API",
    r"\bcmd_push_hub\b": "write-side CLI command",
    r"\bcmd_push_dataset\b": "write-side CLI command",
    r"\bXCALIB_ENABLE_HUB_WRITE\b": "write-side CLI switch",
    r"\bCommitOperationAdd\b": "Hub write-side commit operation",
    r"\bHfApi\b": "Hub write-side API client",
    r"\bcreate_repo\b": "Hub write-side repo creation",
}


def scan_paths(
    paths: list[Path],
    patterns: dict[str, str],
    failures: list[str],
) -> None:
    compiled = [
        (re.compile(pattern, re.IGNORECASE), reason)
        for pattern, reason in patterns.items()
    ]

    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for regex, reason in compiled:
                if regex.search(line):
                    rel = path.relative_to(ROOT)
                    failures.append(f"{rel}:{line_no}: {reason}: {line.strip()}")


def main() -> int:
    failures: list[str] = []
    scan_paths(DOC_PATHS, DOC_DISALLOWED, failures)
    scan_paths(PACKAGE_PATHS, PACKAGE_DISALLOWED, failures)

    if failures:
        print("Private operations terms found in public docs/package:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        "Public boundary check passed "
        f"({len(DOC_PATHS)} doc files, {len(PACKAGE_PATHS)} package files scanned)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

