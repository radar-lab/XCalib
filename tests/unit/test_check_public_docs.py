"""Unit tests for scripts/check_public_docs.py (the public-boundary guard).

The tests import the script by path and run it against small temporary trees.
Strings that match the generic path and credential patterns are assembled at run
time, so this file never contains one. (It names the generic Hub write-side API
terms the guard lists, which only apply to docs and package code.)

Set XCALIB_EXTRA_PATTERNS_FILE to an extra-patterns file to also prove that
the script and this test carry none of its patterns.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_public_docs.py"
EXTRA_ENV = "XCALIB_EXTRA_PATTERNS_FILE"
MARKER = "zz-fixture-marker"


def _cat(*parts: str) -> str:
    return "".join(parts)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_public_docs_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_skeleton(module, root: Path) -> None:
    """Create the smallest tree that satisfies check_agent_contract."""
    (root / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    guide ="Open this first.\n\n- [Jump](#jump)\n"
    (root / "docs" / "agents").mkdir(parents=True)
    for name in ("workflow.md", "code-index.md", "conventions.md"):
        (root / "docs" / "agents" / name).write_text(guide, encoding="utf-8")
    (root / "mkdocs.yml").write_text(
        "nav:\n  - agents/workflow.md\n  - agents/code-index.md\n  - agents/conventions.md\n",
        encoding="utf-8",
    )
    named = re.findall(r'"((?:src|tests)/[^"]+)"', inspect.getsource(module.check_agent_contract))
    assert len(named) >= 10
    for relative in named:
        target = root / relative
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        else:
            target.mkdir(parents=True, exist_ok=True)


class Repo:
    def __init__(self, module, root: Path, capsys, extra_file: Path):
        self.module = module
        self.root = root
        self.capsys = capsys
        self.extra_file = extra_file

    def write(self, relative: str, content: str | bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def write_extra(self, content: str) -> Path:
        self.extra_file.write_text(content, encoding="utf-8")
        return self.extra_file

    def run(self, *args: str) -> tuple[int, str, str]:
        code = self.module.main(list(args))
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err


@pytest.fixture
def repo(mod, tmp_path, monkeypatch, capsys):
    """A bare tree. The agent-contract check is switched off here: it needs a ~30-file
    skeleton, which makes every test slow on file systems with per-file overhead, and it
    has its own tests through ``contract_repo``."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "check_agent_contract", lambda failures: None)
    return Repo(mod, root, capsys, tmp_path / "extra.txt")


@pytest.fixture
def contract_repo(mod, tmp_path, monkeypatch, capsys):
    """A tree with the smallest skeleton that satisfies the real agent-contract check."""
    root = tmp_path / "repo"
    root.mkdir()
    build_skeleton(mod, root)
    monkeypatch.setattr(mod, "ROOT", root)
    return Repo(mod, root, capsys, tmp_path / "extra.txt")


# ---------------------------------------------------------------- baseline


def test_clean_tree_passes(contract_repo):
    repo = contract_repo
    repo.write("README.md", "Plain text.\n")
    repo.write("src/xcalib/core.py", "VALUE = 1\n")
    code, out, err = repo.run()
    assert code == 0
    assert out.startswith("Public boundary check passed (")
    assert "text files scanned" in out
    assert err == ""


def test_failure_message_and_exit_code(repo):
    repo.write("docs/page.md", "Run push_to_hub now.\n")
    code, out, err = repo.run()
    assert code == 1
    assert out == ""
    assert err.startswith("Private operations terms found in public docs/package:")
    assert "docs/page.md:1: write-side Hub API: Run push_to_hub now." in err


def test_agent_contract_still_runs(contract_repo):
    repo = contract_repo
    (repo.root / "docs" / "agents" / "workflow.md").unlink()
    code, _, err = repo.run()
    assert code == 1
    assert "docs/agents/workflow.md: missing public contributor guide" in err


# ------------------------------------------------- Hub write-side terms

DOC_TERMS = [
    ("Call push_to_hub here.", "write-side Hub API"),
    ("Files are uploaded.", "write-side artifact operation"),
    ("Set HF_TOKEN first.", "token environment variable"),
]


@pytest.mark.parametrize("text, reason", DOC_TERMS)
def test_write_side_doc_terms_still_fire(repo, text, reason):
    repo.write("docs/page.md", text + "\n")
    code, _, err = repo.run()
    assert code == 1
    assert f"docs/page.md:1: {reason}: {text}" in err


@pytest.mark.parametrize(
    "relative",
    ["README.md", "demo/README.md", "demo/tour.ipynb", "docs/sub/deep.md"],
)
def test_write_side_doc_terms_cover_every_doc_location(repo, relative):
    repo.write(relative, "Run push_to_hub now.\n")
    code, _, err = repo.run()
    assert code == 1
    assert f"{relative}:1: write-side Hub API" in err


PACKAGE_TERMS = [
    "push_to_hub",
    "CommitOperationAdd",
    "HfApi",
    "create_repo",
]


@pytest.mark.parametrize("term", PACKAGE_TERMS)
def test_write_side_package_terms_still_fire(repo, term):
    repo.write("src/xcalib/hub_extra.py", f"x = {term}\n")
    code, _, err = repo.run()
    assert code == 1
    assert "src/xcalib/hub_extra.py:1:" in err


def test_doc_terms_do_not_apply_outside_docs_and_package(repo):
    repo.write(".github/workflows/ci.yml", "uses: actions/upload-artifact@v4\n")
    repo.write("tests/unit/test_x.py", "# set HF_TOKEN first\n")
    repo.write("scripts/run.py", "print('uploading')\n")
    code, _, _ = repo.run()
    assert code == 0


# ----------------------------------------------------------- generic paths

PATH_CASES = [
    (_cat("C", ":\\", "Users", "\\alice\\data"), "personal Windows machine path"),
    (_cat("D", ":/", "workspace", "/proj"), "personal Windows machine path"),
    (_cat("D", ":\\\\", "workspace", "\\\\proj"), "personal Windows machine path"),
    (_cat("/mnt/", "c", "/Users", "/alice/x"), "personal Windows machine path"),
    (_cat("/", "d", "/workspace", "/x"), "personal Windows machine path"),
    (_cat("/ho", "me/alice/data"), "personal Linux home path"),
    (_cat("cd /ho", "me/alice"), "personal Linux home path"),
    (_cat("/Us", "ers/alice/Documents"), "personal macOS home path"),
]


@pytest.mark.parametrize("text, reason", PATH_CASES)
def test_generic_path_patterns_fire(repo, text, reason):
    repo.write("scripts/run.py", f"path = '{text}'\n")
    code, _, err = repo.run()
    assert code == 1
    assert f"scripts/run.py:1: {reason}:" in err


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "demo/README.md",
        "demo/tour.ipynb",
        "scripts/run.py",
        "tests/unit/test_x.py",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
        "src/xcalib/core.py",
    ],
)
def test_generic_patterns_cover_every_text_file(repo, relative):
    repo.write(relative, _cat("see C", ":\\", "Users", "\\alice\\notes\n"))
    code, _, err = repo.run()
    assert code == 1
    assert f"{relative}:1: personal Windows machine path" in err


PATH_NEGATIVES = [
    _cat("/ho", "me/runner/work/xcalib"),
    _cat("/ho", "me/user/data"),
    _cat("/ho", "me/<user>/data"),
    _cat("/ho", "me/nvidia/xcalib"),
    "$HOME/data",
    "https://example.com/home/page",
    _cat("/Us", "ers/Shared/x"),
    _cat("/api/us", "ers/list"),
    _cat("C", ":\\Program Files\\app"),
    "/data/capture/site1",
    "/opt/nvidia/tensorrt/bin/trtexec",
]


@pytest.mark.parametrize("text", PATH_NEGATIVES)
def test_generic_path_patterns_allow_placeholders(repo, text):
    repo.write("docs/page.md", f"Example: {text}\n")
    code, _, err = repo.run()
    assert code == 0, err


# ---------------------------------------------------------- credential shapes

SECRETS = [
    (_cat("hf", "_", "aB1x" * 10), "Hugging Face access token shape"),
    (_cat("gh", "p_", "aB1x" * 10), "GitHub access token shape"),
    (_cat("AK", "IA", "ABCDEFGHIJKLMNOP"), "AWS access key id shape"),
    (_cat("-----BEGIN ", "RSA PRIVATE KEY", "-----"), "private key block"),
]


@pytest.mark.parametrize("secret, reason", SECRETS)
def test_secret_shapes_fire_and_are_redacted(repo, secret, reason):
    repo.write("scripts/run.py", f"TOKEN = '{secret}'\n")
    code, _, err = repo.run()
    assert code == 1
    assert f"scripts/run.py:1: {reason}:" in err
    assert secret not in err
    assert "<redacted>" in err


def test_secret_is_hidden_on_lines_flagged_for_other_reasons(repo):
    token = _cat("hf", "_", "aB1x" * 10)
    path = _cat("C", ":\\", "Users", "\\alice")
    repo.write("scripts/run.py", f"p = '{path}'; t = '{token}'\n")
    repo.write("docs/page.md", f"Run push_to_hub with {token}\n")
    code, _, err = repo.run()
    assert code == 1
    assert "personal Windows machine path" in err
    assert "write-side Hub API" in err
    assert token not in err


def test_long_lines_are_trimmed_around_the_match(repo):
    path = _cat("C", ":\\", "Users", "\\alice")
    repo.write("demo/tour.ipynb", "x" * 5000 + " " + path + " " + "y" * 5000 + "\n")
    code, _, err = repo.run()
    assert code == 1
    report = [line for line in err.splitlines() if "demo/tour.ipynb:1:" in line]
    assert len(report) == 1
    assert len(report[0]) < 400
    assert path in report[0]


# ----------------------------------------------------------- extra patterns


def test_extra_patterns_fire_and_default_run_passes(repo):
    repo.write("demo/notes.txt", f"see {MARKER} here\n")
    extra = repo.write_extra(f"{MARKER} => synthetic marker\n")
    assert repo.run()[0] == 0
    code, out, err = repo.run("--extra-patterns", str(extra))
    assert code == 1
    assert out == ""
    assert "demo/notes.txt:1: synthetic marker: see zz-fixture-marker here" in err


def test_extra_patterns_apply_to_every_kind_of_file(repo):
    for relative in (
        "docs/page.md",
        "src/xcalib/core.py",
        "demo/tour.ipynb",
        "scripts/run.py",
        "tests/unit/test_x.py",
        ".github/workflows/ci.yml",
    ):
        repo.write(relative, f"{MARKER}\n")
    extra = repo.write_extra(f"{MARKER}\n")
    code, _, err = repo.run("--extra-patterns", str(extra))
    assert code == 1
    assert err.count("extra pattern:") == 6


def test_extra_patterns_file_format(repo):
    repo.write("a.txt", "Alpha-One and beta.two\n")
    extra = repo.write_extra(
        "# comment line\n"
        "\n"
        "   \n"
        "alpha-one => first reason\n"
        r"beta\.two" "\n"
    )
    code, _, err = repo.run("--extra-patterns", str(extra))
    assert code == 1
    assert "a.txt:1: first reason:" in err
    assert "a.txt:1: extra pattern:" in err
    assert err.count("a.txt:1:") == 2


def test_extra_patterns_file_is_not_scanned_when_inside_the_tree(repo):
    extra = repo.root / "private" / "extra.txt"
    extra.parent.mkdir()
    extra.write_text(f"{MARKER} => synthetic marker\n", encoding="utf-8")
    code, _, _ = repo.run("--extra-patterns", str(extra))
    assert code == 0


def test_invalid_extra_pattern_is_a_usage_error(repo):
    extra = repo.write_extra("fine\n(unclosed => broken\n")
    code, out, err = repo.run("--extra-patterns", str(extra))
    assert code == 2
    assert out == ""
    assert "extra.txt:2: invalid regular expression" in err


def test_extra_pattern_that_matches_empty_text_is_rejected(repo):
    extra = repo.write_extra("fine\nx* => would flag every line\n")
    code, _, err = repo.run("--extra-patterns", str(extra))
    assert code == 2
    assert "extra.txt:2: regular expression matches empty text" in err


def test_missing_extra_patterns_file_is_a_usage_error(repo):
    code, _, err = repo.run("--extra-patterns", str(repo.root / "absent.txt"))
    assert code == 2
    assert "Cannot read extra patterns" in err


# ----------------------------------------------------------------- exclusion


def test_exclude_glob_and_directory_prefix(repo):
    repo.write("overlay/notes.md", f"{MARKER}\n")
    extra = repo.write_extra(f"{MARKER} => synthetic marker\n")
    assert repo.run("--extra-patterns", str(extra))[0] == 1
    assert repo.run("--extra-patterns", str(extra), "--exclude", "overlay/*")[0] == 0
    assert repo.run("--extra-patterns", str(extra), "--exclude", "overlay")[0] == 0
    assert repo.run("--extra-patterns", str(extra), "--exclude", "overlay\\*")[0] == 0
    assert repo.run("--extra-patterns", str(extra), "--exclude", "other/*")[0] == 1


# --------------------------------------------------------------- binary data


def test_binary_files_are_skipped_and_counted(repo):
    repo.write("docs/assets/figure.png", b"\x89PNG\r\n\x1a\n\x00\x00" + MARKER.encode() + b"\x00")
    extra = repo.write_extra(f"{MARKER}\n")
    code, out, _ = repo.run("--extra-patterns", str(extra))
    assert code == 0
    assert "1 binary files skipped" in out


# ------------------------------------------------------------ tracked files


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_only_tracked_files_are_scanned_in_a_git_checkout(repo):
    extra = repo.write_extra(f"{MARKER}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo.root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo.root, check=True)
    repo.write("notes/untracked.txt", f"{MARKER}\n")
    assert repo.run("--extra-patterns", str(extra))[0] == 0
    subprocess.run(["git", "add", "notes/untracked.txt"], cwd=repo.root, check=True)
    code, _, err = repo.run("--extra-patterns", str(extra))
    assert code == 1
    assert "notes/untracked.txt:1:" in err


# ----------------------------------------------------------- the script itself


def test_script_and_tests_are_clean_under_generic_patterns(mod, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", SCRIPT.parents[1])
    failures: list[str] = []
    patterns = {**mod.GENERIC_DISALLOWED, **mod.SECRET_DISALLOWED}
    mod.scan_paths([SCRIPT, Path(__file__).resolve()], patterns, failures)
    assert failures == []


def test_script_and_tests_carry_no_extra_pattern(mod, monkeypatch):
    path = os.environ.get(EXTRA_ENV)
    if not path:
        pytest.skip(f"{EXTRA_ENV} is not set")
    extra = mod.load_extra_patterns(Path(path))
    assert extra
    monkeypatch.setattr(mod, "ROOT", SCRIPT.parents[1])
    failures: list[str] = []
    mod.scan_paths([SCRIPT, Path(__file__).resolve()], extra, failures)
    assert failures == []


def test_script_is_standard_library_only():
    tree = SCRIPT.read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:from|import) ([A-Za-z_][\w]*)", tree, flags=re.MULTILINE))
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_script_runs_standalone_from_a_copy(mod, tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    build_skeleton(mod, root)
    (root / "scripts").mkdir()
    shutil.copy(SCRIPT, root / "scripts" / "check_public_docs.py")
    command = [sys.executable, "-I", str(root / "scripts" / "check_public_docs.py")]
    clean = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout.startswith("Public boundary check passed")
    (root / "docs" / "page.md").write_text("Run push_to_hub now.\n", encoding="utf-8")
    dirty = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert dirty.returncode == 1
    assert "docs/page.md:1:" in dirty.stderr


# ------------------------------------------------------- robustness found in review


def test_a_byte_order_mark_does_not_disarm_the_first_extra_pattern(repo):
    extra = repo.extra_file
    extra.write_bytes(("﻿" + MARKER + " => synthetic marker\n").encode("utf-8"))
    repo.write("docs/notes.txt", f"see {MARKER} here\n")
    code, _, err = repo.run("--extra-patterns", str(extra))
    assert code == 1
    assert "docs/notes.txt:1: synthetic marker" in err


def test_a_tree_without_mkdocs_yml_reports_it_instead_of_crashing(contract_repo):
    (contract_repo.root / "mkdocs.yml").unlink()
    code, _, err = contract_repo.run()
    assert code == 1
    assert "mkdocs.yml: missing" in err
