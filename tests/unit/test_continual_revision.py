"""Resolve / pointer-validation tests for the revision-resolution helpers.

`xcalib.continual.refine_model` consumes the package Matcher via lazy imports.
These cover `resolve_xcalib_revision` -- which a long-running caller may invoke
once per request -- and the strict `current.json` validation, including the
(path, mtime, size) SHA memoisation. Tests that reload a model (verify / match /
train) need real checkpoints and are not covered here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from xcalib.continual.refine_model import (
    XCalibRevisionError,
    resolve_xcalib_revision,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer_layout(model_dir: Path) -> dict[str, object]:
    revision_dir = model_dir / "revisions" / "rev-000002"
    revision_dir.mkdir(parents=True)
    weights = revision_dir / "crlite.pth"
    config = revision_dir / "crlite.yaml"
    weights.write_bytes(b"checkpoint-v2")
    config.write_text("model: crlite\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "revision": "rev-000002",
        "parent_revision": "rev-000001",
        "model_name": "crlite",
        "weights": "revisions/rev-000002/crlite.pth",
        "config": "revisions/rev-000002/crlite.yaml",
        "sha256": _sha(weights),
        "published_at": "2026-07-12T00:00:00Z",
    }


def _write_pointer(model_dir: Path, payload: object) -> None:
    (model_dir / "current.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_pointer_and_legacy_layout(tmp_path: Path) -> None:
    model_dir = tmp_path / "cam_sync_model"
    revision_dir = model_dir / "revisions" / "rev-000002"
    revision_dir.mkdir(parents=True)
    weights = revision_dir / "crlite.pth"
    config = revision_dir / "crlite.yaml"
    weights.write_bytes(b"checkpoint-v2")
    config.write_text("model: crlite\n", encoding="utf-8")
    pointer = {
        "schema_version": 1,
        "revision": "rev-000002",
        "parent_revision": "rev-000001",
        "model_name": "crlite",
        "weights": "revisions/rev-000002/crlite.pth",
        "config": "revisions/rev-000002/crlite.yaml",
        "sha256": _sha(weights),
        "published_at": "2026-07-12T00:00:00Z",
    }
    (model_dir / "current.json").write_text(json.dumps(pointer), encoding="utf-8")

    ref = resolve_xcalib_revision(model_dir)

    assert ref is not None
    assert ref.revision == "rev-000002"
    assert ref.parent_revision == "rev-000001"
    assert ref.source == "pointer"
    assert ref.weights_path == weights.resolve()
    assert ref.config_path == config.resolve()
    assert ref.sha256 == _sha(weights)

    (model_dir / "current.json").unlink()
    legacy = model_dir / "crlite.pth"
    legacy.write_bytes(b"legacy")
    legacy_ref = resolve_xcalib_revision(model_dir)
    assert legacy_ref is not None
    assert legacy_ref.source == "legacy"
    assert legacy_ref.revision == f"legacy-{_sha(legacy)[:12]}"


def test_resolve_memoises_weights_sha_until_the_file_changes(tmp_path, monkeypatch):
    # resolve_xcalib_revision may be called repeatedly for one checkpoint; the
    # full-file SHA is memoised by (path, mtime, size) so it is not recomputed each
    # call.
    import xcalib.continual.refine_model as mlx

    model_dir = tmp_path / "cam_sync_model"
    model_dir.mkdir(parents=True)
    (model_dir / "crlite.pth").write_bytes(b"checkpoint")
    (model_dir / "crlite.yaml").write_text("model: crlite\n", encoding="utf-8")

    calls = {"n": 0}
    real = mlx._sha256

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(mlx, "_sha256", counting)
    mlx._SHA_CACHE.clear()

    first = resolve_xcalib_revision(model_dir)
    second = resolve_xcalib_revision(model_dir)
    assert first.sha256 == second.sha256
    assert calls["n"] == 1  # the second resolve hit the SHA cache

    (model_dir / "crlite.pth").write_bytes(b"checkpoint-with-different-size")
    third = resolve_xcalib_revision(model_dir)
    assert calls["n"] == 2 and third.sha256 != first.sha256


@pytest.mark.parametrize(
    "payload",
    [None, [], 7, {"schema_version": 1}],
    ids=["null", "list", "scalar", "wrong-object-shape"],
)
def test_pointer_rejects_non_object_or_wrong_shape(tmp_path: Path, payload: object) -> None:
    model_dir = tmp_path / "cam_sync_model"
    model_dir.mkdir()
    _write_pointer(model_dir, payload)

    with pytest.raises(XCalibRevisionError, match="current.json"):
        resolve_xcalib_revision(model_dir)


def test_pointer_rejects_boolean_schema_version(tmp_path: Path) -> None:
    model_dir = tmp_path / "cam_sync_model"
    pointer = _pointer_layout(model_dir)
    pointer["schema_version"] = True
    _write_pointer(model_dir, pointer)

    with pytest.raises(XCalibRevisionError, match="schema_version"):
        resolve_xcalib_revision(model_dir)


def test_pointer_rejects_invalid_parent_revision_type(tmp_path: Path) -> None:
    model_dir = tmp_path / "cam_sync_model"
    pointer = _pointer_layout(model_dir)
    pointer["parent_revision"] = 42
    _write_pointer(model_dir, pointer)

    with pytest.raises(XCalibRevisionError, match="parent_revision"):
        resolve_xcalib_revision(model_dir)


@pytest.mark.parametrize(
    "field",
    ["revision", "model_name", "weights", "config", "sha256", "published_at"],
)
def test_pointer_requires_nonempty_string_fields(tmp_path: Path, field: str) -> None:
    model_dir = tmp_path / "cam_sync_model"
    pointer = _pointer_layout(model_dir)
    pointer[field] = ""
    _write_pointer(model_dir, pointer)

    with pytest.raises(XCalibRevisionError, match=field):
        resolve_xcalib_revision(model_dir)


@pytest.mark.parametrize("field", ["revision", "model_name", "sha256", "published_at"])
def test_pointer_rejects_non_string_metadata(tmp_path: Path, field: str) -> None:
    model_dir = tmp_path / "cam_sync_model"
    pointer = _pointer_layout(model_dir)
    pointer[field] = 123
    _write_pointer(model_dir, pointer)

    with pytest.raises(XCalibRevisionError, match=field):
        resolve_xcalib_revision(model_dir)


@pytest.mark.parametrize("field", ["weights", "config"])
def test_pointer_rejects_non_string_paths_without_coercion(tmp_path: Path, field: str) -> None:
    model_dir = tmp_path / "cam_sync_model"
    pointer = _pointer_layout(model_dir)
    coerced_artifact = model_dir / "123"
    if field == "weights":
        coerced_artifact.write_bytes(b"coerced-checkpoint")
        pointer["sha256"] = _sha(coerced_artifact)
    else:
        coerced_artifact.write_text("model: coerced\n", encoding="utf-8")
    pointer[field] = 123
    _write_pointer(model_dir, pointer)

    with pytest.raises(XCalibRevisionError, match=field):
        resolve_xcalib_revision(model_dir)


def test_pointer_normalizes_hash_io_failure(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "cam_sync_model"
    pointer = _pointer_layout(model_dir)
    _write_pointer(model_dir, pointer)

    def fail_hash(_path):
        raise OSError("cannot read checkpoint")

    monkeypatch.setattr("xcalib.continual.refine_model._sha256", fail_hash)
    with pytest.raises(XCalibRevisionError, match="cannot resolve XCalib revision"):
        resolve_xcalib_revision(model_dir)


@pytest.mark.parametrize("bad_weights", ["../outside.pth", "revisions/rev-000002/missing.pth"])
def test_pointer_rejects_escape_or_missing_artifact(tmp_path: Path, bad_weights: str) -> None:
    model_dir = tmp_path / "cam_sync_model"
    model_dir.mkdir()
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")
    pointer = {
        "schema_version": 1,
        "revision": "rev-000002",
        "parent_revision": "rev-000001",
        "model_name": "crlite",
        "weights": bad_weights,
        "config": "revisions/rev-000002/crlite.yaml",
        "sha256": _sha(outside),
        "published_at": "2026-07-12T00:00:00Z",
    }
    (model_dir / "current.json").write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(XCalibRevisionError):
        resolve_xcalib_revision(model_dir)


def test_pointer_rejects_digest_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "cam_sync_model"
    revision_dir = model_dir / "revisions" / "rev-000001"
    revision_dir.mkdir(parents=True)
    (revision_dir / "crlite.pth").write_bytes(b"changed")
    (revision_dir / "crlite.yaml").write_text("model: crlite\n", encoding="utf-8")
    pointer = {
        "schema_version": 1,
        "revision": "rev-000001",
        "parent_revision": None,
        "model_name": "crlite",
        "weights": "revisions/rev-000001/crlite.pth",
        "config": "revisions/rev-000001/crlite.yaml",
        "sha256": "0" * 64,
        "published_at": "2026-07-12T00:00:00Z",
    }
    (model_dir / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(XCalibRevisionError, match="sha256"):
        resolve_xcalib_revision(model_dir)
