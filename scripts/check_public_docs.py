"""Guard public docs and package code against private operations leakage.

The built-in checks name generic Hub write-side API terms, and match personal
machine path shapes and credential shapes; none names a particular person,
machine or private workflow. Terms that belong to a private workflow are not
listed here: pass ``--extra-patterns FILE`` to add them for a run.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
DOC_SUFFIXES = {".md", ".ipynb"}
WALK_SKIP_DIRS = {
    ".git",
    ".pixi",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "site",
    ".pytest_cache",
    ".ruff_cache",
}

# Applied to documentation. Only generic Hub terms are listed; terms from a private
# workflow come from --extra-patterns.
DOC_DISALLOWED = {
    r"\bpush_to_hub\b": "write-side Hub API",
    r"\bupload(?:ing|s|ed)?\b": "write-side artifact operation",
    r"\bHF_TOKEN\b": "token environment variable",
}

# Applied to package code.
PACKAGE_DISALLOWED = {
    r"\bpush_to_hub\b": "write-side Matcher Hub API",
    r"\bCommitOperationAdd\b": "Hub write-side commit operation",
    r"\bHfApi\b": "Hub write-side API client",
    r"\bcreate_repo\b": "Hub write-side repo creation",
}

# Applied to every tracked text file. Placeholder account names are allowed.
GENERIC_DISALLOWED = {
    # Drive-letter, JSON-escaped, WSL and MSYS forms of a user or workspace folder.
    r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]+|(?<![\w.-])/(?:mnt/)?[A-Za-z]/)"
    r"(?:Users|workspaces?)[\\/]": "personal Windows machine path",
    r"(?<![\w.-])/(?-i:home)/"
    r"(?!(?:user|username|name|me|runner|ubuntu|nvidia|jovyan|example)(?![\w.-]))"
    r"[\w.-]+": "personal Linux home path",
    r"(?<![\w.:-])/(?-i:Users)/"
    r"(?!(?:Shared|Public|user|username|name|me|example)(?![\w.-]))"
    r"[\w.-]+": "personal macOS home path",
}

# Applied to every tracked text file; reports hide anything these match.
SECRET_DISALLOWED = {
    r"\bhf_[A-Za-z0-9]{30,}\b": "Hugging Face access token shape",
    r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{20,})\b": "GitHub access token shape",
    r"\bAKIA[0-9A-Z]{16}\b": "AWS access key id shape",
    r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----": "private key block",
}
SECRET_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in SECRET_DISALLOWED]

EXTRA_PATTERNS_HELP = """\
Extra pattern file (UTF-8, one entry per line):

    REGEX => reason shown in the report

Blank lines and lines starting with # are ignored, the reason is optional, and
every REGEX is a case-insensitive Python regular expression applied to each
tracked text file. A usage error exits with status 2.
"""


def load_extra_patterns(path: Path) -> dict[str, str]:
    patterns: dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig")  # a BOM would otherwise end up in the first pattern
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, separator, reason = line.rpartition(" => ")
        if not separator:
            pattern, reason = line, "extra pattern"
        pattern = pattern.strip()
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"{path}:{line_no}: invalid regular expression: {exc}") from exc
        if compiled.search(""):
            raise ValueError(f"{path}:{line_no}: regular expression matches empty text")
        patterns[pattern] = reason.strip() or "extra pattern"
    return patterns


def list_files(root: Path) -> list[Path]:
    """Return the tracked files of a git checkout, else every file under root."""
    if (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
            return [root / name for name in names if name]
    found: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in WALK_SKIP_DIRS)
        found.extend(Path(current) / name for name in sorted(names))
    return found


def is_excluded(path: Path, excludes: list[str], skip: set[Path]) -> bool:
    if path.resolve() in skip:
        return True
    rel = path.relative_to(ROOT).as_posix()
    for pattern in excludes:
        glob = pattern.replace("\\", "/")
        base = glob.rstrip("/")
        if fnmatch.fnmatchcase(rel, glob) or rel == base or rel.startswith(base + "/"):
            return True
    return False


def excerpt(line: str, regex: re.Pattern[str]) -> str:
    """Return the offending line, trimmed around the match; credential shapes are hidden."""
    text = line.strip()
    for secret in SECRET_REGEXES:
        text = secret.sub("<redacted>", text)
    if len(text) <= 200:
        return text
    match = regex.search(text) or re.search("<redacted>", text)
    start = max(0, (match.start() if match else 0) - 80)
    return f"...{text[start:start + 200]}..."


def scan_paths(
    paths: list[Path],
    patterns: dict[str, str],
    failures: list[str],
) -> tuple[int, int]:
    """Scan text files for patterns; return (text files scanned, binary files skipped)."""
    compiled = [
        (re.compile(pattern, re.IGNORECASE | re.MULTILINE), reason)
        for pattern, reason in patterns.items()
    ]
    scanned = binary = 0

    for path in paths:
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            head = handle.read(8192)
            if b"\0" in head:
                binary += 1
                continue
            data = head + handle.read()
        scanned += 1
        text = data.decode("utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        hits: list[tuple[int, int, str]] = []
        for index, (regex, reason) in enumerate(compiled):
            if not regex.search(text):
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(
                        (line_no, index, f"{rel}:{line_no}: {reason}: {excerpt(line, regex)}")
                    )
        failures.extend(message for _, _, message in sorted(hits))
    return scanned, binary


def check_agent_contract(failures: list[str]) -> None:
    agents = ROOT / "AGENTS.md"
    if not agents.is_file():
        failures.append("AGENTS.md: missing public agent entry")

    shared = ROOT / "docs" / "agents"
    required = ("workflow.md", "code-index.md", "conventions.md")
    for name in required:
        path = shared / name
        if not path.is_file():
            failures.append(f"docs/agents/{name}: missing public contributor guide")
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(r"(?im)^Open this", text):
            failures.append(f"docs/agents/{name}: missing opening usage cue")
        if not re.search(r"(?m)^- \[[^]]+\]\(#[^)]+\)$", text):
            failures.append(f"docs/agents/{name}: missing jump list")

    mkdocs_path = ROOT / "mkdocs.yml"
    if mkdocs_path.is_file():
        mkdocs = mkdocs_path.read_text(encoding="utf-8")
        for name in required:
            if f"agents/{name}" not in mkdocs:
                failures.append(f"mkdocs.yml: agents/{name} missing from public navigation")
    else:
        failures.append("mkdocs.yml: missing, so the public navigation cannot be checked")

    paths = [agents, *sorted(shared.glob("*.md"))]
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in paths:
        if not path.exists():
            continue
        for target in link_pattern.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                failures.append(f"{path.relative_to(ROOT)}: broken link {target}")

    for relative in (
        "src/xcalib/__init__.py",
        "src/xcalib/engine/matcher.py",
        "src/xcalib/engine/exporter.py",
        "src/xcalib/engine/trainer.py",
        "src/xcalib/engine/trt.py",
        "src/xcalib/models/registry.py",
        "src/xcalib/cfg",
        "src/xcalib/protocol.py",
        "src/xcalib/calibration_refine.py",
        "src/xcalib/oneshot",
        "src/xcalib/continual",
        "src/xcalib/data",
        "src/xcalib/hub",
        "src/xcalib/gui",
        "tests/smoke",
        "tests/unit",
        "tests/integration",
    ):
        if not (ROOT / relative).exists():
            failures.append(f"docs/agents/code-index.md: missing named path {relative}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when tracked files expose write-side Hub operations, "
            "personal machine paths or credentials."
        ),
        epilog=EXTRA_PATTERNS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extra-patterns",
        metavar="FILE",
        help="add the regular expressions listed in FILE to the generic checks",
    )
    parser.add_argument(
        "--exclude",
        metavar="GLOB",
        action="append",
        default=[],
        help="skip tracked files whose repository-relative path matches GLOB (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra: dict[str, str] = {}
    skip = {SELF}
    if args.extra_patterns:
        extra_file = Path(args.extra_patterns)
        try:
            extra = load_extra_patterns(extra_file)
        except (OSError, ValueError) as exc:
            print(f"Cannot read extra patterns: {exc}", file=sys.stderr)
            return 2
        skip.add(extra_file.resolve())

    def keep(path: Path) -> bool:
        return not is_excluded(path, args.exclude, skip)

    files = [p for p in list_files(ROOT) if p.is_file() and not p.is_symlink() and keep(p)]
    doc_paths = sorted(
        {
            ROOT / "AGENTS.md",
            ROOT / "README.md",
            ROOT / "mkdocs.yml",
            *(ROOT / "docs").glob("**/*.md"),
            *(p for p in files if p.suffix.lower() in DOC_SUFFIXES),
        }
    )
    doc_paths = [p for p in doc_paths if keep(p)]
    package_paths = [p for p in sorted((ROOT / "src" / "xcalib").glob("**/*.py")) if keep(p)]

    failures: list[str] = []
    scan_paths(doc_paths, DOC_DISALLOWED, failures)
    scan_paths(package_paths, PACKAGE_DISALLOWED, failures)
    scanned, binary = scan_paths(
        files,
        {**GENERIC_DISALLOWED, **SECRET_DISALLOWED, **extra},
        failures,
    )
    check_agent_contract(failures)

    if failures:
        print("Private operations terms found in public docs/package:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    extra_note = f", {len(extra)} extra patterns" if extra else ""
    print(
        "Public boundary check passed "
        f"({len(doc_paths)} doc files, {len(package_paths)} package files, "
        f"{scanned} text files scanned, {binary} binary files skipped{extra_note})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
