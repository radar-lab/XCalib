from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil
import sqlite3

import pytest

from xcalib.continual.refine_model import (
    XCalibArtifact,
    XCalibRevisionError,
    XCalibRevisionRef,
    resolve_xcalib_revision,
)
from xcalib.continual import publication as xcalib_publication
from xcalib.continual.publication import (
    PublicationConflict,
    PublicationDivergence,
    commit_publication,
    list_divergent_operations,
    migrate_publications,
    prepare_publication,
    prune_revision_artifacts,
    publish_revision,
    read_operation,
    recover_publications,
    remediate_divergence,
    rollback_revision,
    switch_publication_pointer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def make_refine_db(path: Path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate_publications(conn)
    return conn


def _legacy_parent(model_dir: Path) -> XCalibRevisionRef:
    model_dir.mkdir(parents=True, exist_ok=True)
    legacy_weights = model_dir / "crlite.pth"
    if not legacy_weights.exists():
        legacy_weights.write_bytes(b"legacy-parent")
    parent_digest = _sha256(legacy_weights)
    return XCalibRevisionRef(
        model_dir=model_dir,
        model_name="crlite",
        revision=f"legacy-{parent_digest[:12]}",
        parent_revision=None,
        weights_path=legacy_weights,
        config_path=None,
        sha256=parent_digest,
        source="legacy",
    )


def _temporary_artifact(
    model_dir: Path,
    parent: XCalibRevisionRef,
    revision: str = "rev-000003",
) -> XCalibArtifact:
    temporary = model_dir / f".{revision}.tmp"
    temporary.mkdir()
    weights = temporary / "crlite.pth"
    config = temporary / "crlite.yaml"
    weights.write_bytes(f"verified-candidate:{revision}".encode())
    config.write_text("model: crlite\n", encoding="utf-8")
    return XCalibArtifact(
        revision=revision,
        parent_revision=parent.revision,
        model_name="crlite",
        directory=temporary,
        weights_path=weights,
        config_path=config,
        sha256=_sha256(weights),
        base_fingerprint="tiny-base-v1",
    )


def prepared_artifact(root: Path):
    model_dir = root / "cam_sync_model"
    parent = _legacy_parent(model_dir)
    artifact = _temporary_artifact(model_dir, parent)
    return model_dir, parent, artifact


def _trust_artifacts(monkeypatch):
    monkeypatch.setattr(
        "xcalib.continual.publication.verify_xcalib_revision",
        lambda value, device="cpu": value,
    )


def test_commit_publication_skips_finalize_if_committed_during_verify(
    tmp_path, monkeypatch
):
    # Race: two committers (or a resumed recovery worker) both read `pointer_switched`
    # at the top of commit_publication; one finalizes while the other is still
    # load-verifying the destination. finalize() is NOT idempotent (it bumps
    # per-sample streak counters), so the loser must not run it a second time.
    conn = make_refine_db(tmp_path / "refine.sqlite")
    _model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "race-run", parent, artifact, {"sample": "s1"})
    operation = switch_publication_pointer(conn, operation)
    assert operation.state == "pointer_switched"

    finalized: list[str] = []
    real_verify = xcalib_publication._verify_destination

    def racing_verify(op):
        # a concurrent committer wins while we are load-verifying
        conn.execute(
            "UPDATE publication_operations SET state='committed', last_error=NULL "
            "WHERE run_id=?",
            (op.run_id,),
        )
        conn.commit()
        return real_verify(op)

    monkeypatch.setattr(xcalib_publication, "_verify_destination", racing_verify)
    commit_publication(
        conn, operation, finalize=lambda db, op: finalized.append(op.run_id)
    )
    assert finalized == []  # loser did not re-run the non-idempotent finalize
    assert read_operation(conn, "race-run").state == "committed"


def test_prepare_is_verified_idempotent_and_conflicts_on_payload_reuse(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    verified: list[str] = []
    monkeypatch.setattr(
        "xcalib.continual.publication.verify_xcalib_revision",
        lambda value, device="cpu": verified.append(value.revision) or value,
    )

    operation = prepare_publication(
        conn,
        "run-1",
        parent,
        artifact,
        commit_payload={"state_updates": [{"sample_key": "s1"}]},
    )
    repeated = prepare_publication(
        conn,
        "run-1",
        parent,
        artifact,
        commit_payload={"state_updates": [{"sample_key": "s1"}]},
    )

    assert operation == repeated
    assert operation.state == "prepared"
    assert operation.old_pointer is None
    assert operation.new_pointer["schema_version"] == 1
    assert operation.new_pointer["revision"] == "rev-000003"
    assert operation.new_pointer["weights"] == "revisions/rev-000003/crlite.pth"
    assert operation.commit_payload == {
        "state_updates": [{"sample_key": "s1"}]
    }
    assert verified == ["rev-000003", "rev-000003"]
    with pytest.raises(PublicationConflict):
        prepare_publication(
            conn,
            "run-1",
            parent,
            artifact,
            commit_payload={"state_updates": []},
        )
    assert read_operation(conn, "run-1").state == "prepared"
    assert model_dir.joinpath("current.json").exists() is False


def test_prepare_retry_after_switch_and_commit_verifies_journalled_destination(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    verified_directories: list[Path] = []

    def verify(value, device="cpu"):
        verified_directories.append(value.directory.resolve())
        return value

    monkeypatch.setattr(
        "xcalib.continual.publication.verify_xcalib_revision", verify
    )
    operation = prepare_publication(
        conn, "retry-run", parent, artifact, {"sample": "s1"}
    )
    operation = switch_publication_pointer(conn, operation)
    destination = model_dir / "revisions" / artifact.revision
    assert not artifact.directory.exists()

    verified_directories.clear()
    pointer_retry = prepare_publication(
        conn, "retry-run", parent, artifact, {"sample": "s1"}
    )
    assert pointer_retry.state == "pointer_switched"
    assert verified_directories == [destination.resolve()]

    commit_publication(conn, pointer_retry, finalize=lambda db, value: None)
    verified_directories.clear()
    committed_retry = prepare_publication(
        conn, "retry-run", parent, artifact, {"sample": "s1"}
    )
    assert committed_retry.state == "committed"
    assert verified_directories == [destination.resolve()]

    with pytest.raises(PublicationConflict, match="different publication payload"):
        prepare_publication(
            conn,
            "retry-run",
            parent,
            artifact,
            {"sample": "different"},
        )


def test_publish_retry_after_lost_response_is_idempotent(tmp_path, monkeypatch):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    _model_dir, parent, artifact = prepared_artifact(tmp_path)
    def require_existing_artifact(value, device="cpu"):
        if not value.directory.is_dir():
            raise XCalibRevisionError("artifact directory is missing")
        return value

    monkeypatch.setattr(
        "xcalib.continual.publication.verify_xcalib_revision",
        require_existing_artifact,
    )
    finalized: list[str] = []

    first = publish_revision(
        conn,
        "lost-response",
        parent,
        artifact,
        {"sample": "s1"},
        finalize=lambda db, operation: finalized.append(operation.run_id),
    )
    retried = publish_revision(
        conn,
        "lost-response",
        parent,
        artifact,
        {"sample": "s1"},
        finalize=lambda db, operation: finalized.append(operation.run_id),
    )

    assert retried == first
    assert finalized == ["lost-response"]


@pytest.mark.parametrize("mutation", ["modified", "deleted"])
def test_committed_retry_uses_frozen_parent_config_snapshot(
    tmp_path, monkeypatch, mutation
):
    import xcalib.continual.publication as implementation

    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir = tmp_path / "cam_sync_model"
    legacy = _legacy_parent(model_dir)
    _trust_artifacts(monkeypatch)
    first_artifact = _temporary_artifact(model_dir, legacy, "rev-000001")
    versioned_parent = publish_revision(
        conn,
        "publish-parent",
        legacy,
        first_artifact,
        {},
        finalize=lambda db, operation: None,
    )
    child_artifact = _temporary_artifact(
        model_dir, versioned_parent, "rev-000002"
    )
    finalized: list[str] = []
    first = publish_revision(
        conn,
        "retry-child",
        versioned_parent,
        child_artifact,
        {"sample": "s1"},
        finalize=lambda db, operation: finalized.append(operation.run_id),
    )
    parent_config = versioned_parent.config_path.resolve()
    if mutation == "modified":
        parent_config.write_text("model: modified-after-commit\n", encoding="utf-8")
    else:
        parent_config.unlink()
    real_sha256 = implementation._sha256

    def reject_parent_rehash(path):
        if Path(path).resolve() == parent_config:
            raise AssertionError("committed retry re-read mutable parent config")
        return real_sha256(path)

    monkeypatch.setattr(implementation, "_sha256", reject_parent_rehash)
    retried = publish_revision(
        conn,
        "retry-child",
        versioned_parent,
        child_artifact,
        {"sample": "s1"},
        finalize=lambda db, operation: finalized.append(operation.run_id),
    )

    assert retried == first
    assert finalized == ["retry-child"]


def test_committed_retry_rejects_forged_parent_snapshot_without_rehashing(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir = tmp_path / "cam_sync_model"
    legacy = _legacy_parent(model_dir)
    _trust_artifacts(monkeypatch)
    first_artifact = _temporary_artifact(model_dir, legacy, "rev-000001")
    versioned_parent = publish_revision(
        conn,
        "publish-parent",
        legacy,
        first_artifact,
        {},
        finalize=lambda db, operation: None,
    )
    child_artifact = _temporary_artifact(
        model_dir, versioned_parent, "rev-000002"
    )
    publish_revision(
        conn,
        "retry-child",
        versioned_parent,
        child_artifact,
        {"sample": "s1"},
        finalize=lambda db, operation: None,
    )
    forged_parent = replace(
        versioned_parent, config_path=model_dir / "missing-forged-parent.yaml"
    )

    with pytest.raises(PublicationConflict, match="different publication payload"):
        prepare_publication(
            conn,
            "retry-child",
            forged_parent,
            child_artifact,
            {"sample": "s1"},
        )


def test_prepare_rejects_dirty_caller_transaction_without_committing_it(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    _model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    conn.execute("CREATE TABLE caller_work (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO caller_work(value) VALUES ('must-rollback')")
    assert conn.in_transaction is True

    with pytest.raises(PublicationConflict, match="clean transaction"):
        prepare_publication(conn, "dirty-prepare", parent, artifact, {})

    assert conn.in_transaction is True
    with pytest.raises(KeyError):
        read_operation(conn, "dirty-prepare")
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM caller_work").fetchone()[0] == 0
    assert artifact.directory.is_dir()


@pytest.mark.parametrize("phase", ["switch", "recover"])
def test_mutating_publication_phases_reject_dirty_caller_transaction(
    tmp_path, monkeypatch, phase
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "dirty-phase", parent, artifact, {})
    conn.execute("CREATE TABLE caller_phase_work (value TEXT NOT NULL)")
    conn.commit()
    conn.execute(
        "INSERT INTO caller_phase_work(value) VALUES ('must-rollback')"
    )
    assert conn.in_transaction is True

    with pytest.raises(PublicationConflict, match="clean transaction"):
        if phase == "switch":
            switch_publication_pointer(conn, operation)
        else:
            recover_publications(conn, finalize=lambda db, value: None)

    assert conn.in_transaction is True
    assert read_operation(conn, "dirty-phase").state == "prepared"
    assert artifact.directory.is_dir()
    assert not (model_dir / "current.json").exists()
    assert not (model_dir / "revisions" / artifact.revision).exists()
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM caller_phase_work").fetchone()[0] == 0


def test_prepare_requires_candidate_parent_model_and_exact_wheel_filenames(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    with pytest.raises(PublicationConflict, match="parent"):
        prepare_publication(
            conn,
            "wrong-parent",
            parent,
            replace(artifact, parent_revision="rev-not-the-parent"),
            {},
        )

    rogue_config = model_dir / "rogue-parent.yaml"
    rogue_config.write_text("model: crlite\n")
    with pytest.raises(PublicationConflict, match="parent"):
        prepare_publication(
            conn,
            "wrong-parent-config",
            replace(parent, config_path=rogue_config),
            artifact,
            {},
        )
    with pytest.raises((ValueError, PublicationConflict), match="model"):
        prepare_publication(
            conn,
            "unsafe-model",
            parent,
            replace(artifact, model_name="../escape"),
            {},
        )

    wrong_weights = artifact.directory / "renamed.pth"
    wrong_weights.write_bytes(artifact.weights_path.read_bytes())
    with pytest.raises(PublicationConflict, match="filename|weights"):
        prepare_publication(
            conn,
            "wrong-filename",
            parent,
            replace(artifact, weights_path=wrong_weights),
            {},
        )


def test_switch_moves_verified_directory_and_atomically_writes_canonical_pointer(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "run-1", parent, artifact, {})

    switched = switch_publication_pointer(conn, operation)
    repeated = switch_publication_pointer(conn, switched)

    assert switched.state == repeated.state == "pointer_switched"
    assert not artifact.directory.exists()
    destination = model_dir / "revisions" / "rev-000003"
    assert destination.is_dir()
    assert (model_dir / "current.json").read_text(encoding="utf-8") == _canonical(
        switched.new_pointer
    )
    assert list(model_dir.glob("current.json.tmp-*")) == []
    resolved = resolve_xcalib_revision(model_dir)
    assert resolved is not None
    assert resolved.revision == "rev-000003"
    assert resolved.sha256 == artifact.sha256


def test_recovery_finishes_pointer_switched_operation_once(tmp_path, monkeypatch):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    calls: list[str] = []
    operation = prepare_publication(
        conn,
        "run-1",
        parent,
        artifact,
        commit_payload={"state_updates": [{"sample_key": "s1"}]},
    )
    switch_publication_pointer(conn, operation)
    assert read_operation(conn, "run-1").state == "pointer_switched"

    results = recover_publications(
        conn,
        finalize=lambda db, op: calls.append(op.run_id),
    )
    repeated = recover_publications(
        conn,
        finalize=lambda db, op: calls.append(op.run_id),
    )

    assert results == ["run-1:committed"]
    assert repeated == []
    assert calls == ["run-1"]
    assert read_operation(conn, "run-1").state == "committed"
    assert json.loads((model_dir / "current.json").read_text())["revision"] == (
        "rev-000003"
    )


def test_recovery_completes_prepared_from_old_pointer(tmp_path, monkeypatch):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    prepare_publication(conn, "run-1", parent, artifact, {"value": 3})
    finalized: list[dict] = []

    results = recover_publications(
        conn,
        finalize=lambda db, operation: finalized.append(operation.commit_payload),
    )

    assert results == ["run-1:committed"]
    assert finalized == [{"value": 3}]
    assert read_operation(conn, "run-1").state == "committed"
    assert json.loads((model_dir / "current.json").read_text())["revision"] == (
        "rev-000003"
    )


def test_recovery_marks_failed_and_removes_only_unreferenced_temporary_artifact(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    verify_calls = 0

    def verify(value, device="cpu"):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls > 1:
            raise XCalibRevisionError("candidate no longer verifies")
        return value

    monkeypatch.setattr(
        "xcalib.continual.publication.verify_xcalib_revision", verify
    )
    prepare_publication(conn, "run-1", parent, artifact, {})
    unrelated = model_dir / ".unrelated.tmp"
    unrelated.mkdir()

    recover_publications(conn, finalize=lambda db, operation: None)

    failed = read_operation(conn, "run-1")
    assert failed.state == "failed"
    assert "candidate no longer verifies" in (failed.last_error or "")
    assert not artifact.directory.exists()
    assert unrelated.is_dir()
    assert not (model_dir / "current.json").exists()


def test_recovery_accepts_a_new_pointer_written_before_the_journal_state_update(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "run-1", parent, artifact, {})
    destination = model_dir / "revisions" / artifact.revision
    destination.parent.mkdir()
    artifact.directory.replace(destination)
    (model_dir / "current.json").write_text(
        _canonical(operation.new_pointer), encoding="utf-8"
    )

    results = recover_publications(conn, finalize=lambda db, operation: None)

    assert results == ["run-1:committed"]
    assert read_operation(conn, "run-1").state == "committed"


def test_divergent_pointer_blocks_recovery_and_later_writes_without_overwrite(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    prepare_publication(conn, "run-1", parent, artifact, commit_payload={})
    divergent_bytes = _canonical({"revision": "rev-999999"}).encode()
    (model_dir / "current.json").write_bytes(divergent_bytes)

    with pytest.raises(PublicationDivergence):
        recover_publications(conn, finalize=lambda db, operation: None)

    assert (model_dir / "current.json").read_bytes() == divergent_bytes
    assert read_operation(conn, "run-1").state == "diverged"
    later = _temporary_artifact(model_dir, parent, revision="rev-000004")
    with pytest.raises(PublicationDivergence):
        prepare_publication(conn, "run-2", parent, later, {})
    assert (model_dir / "current.json").read_bytes() == divergent_bytes


def test_destination_collision_with_wrong_digest_diverges_closed(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    prepare_publication(conn, "run-1", parent, artifact, {})
    destination = model_dir / "revisions" / artifact.revision
    destination.mkdir(parents=True)
    (destination / "crlite.pth").write_bytes(b"wrong-candidate")
    (destination / "crlite.yaml").write_text("model: crlite\n")
    shutil.rmtree(artifact.directory)

    with pytest.raises(PublicationDivergence):
        recover_publications(conn, finalize=lambda db, operation: None)

    assert read_operation(conn, "run-1").state == "diverged"
    assert (destination / "crlite.pth").read_bytes() == b"wrong-candidate"
    assert not (model_dir / "current.json").exists()


def test_historical_revision_directory_cannot_be_renamed_as_a_new_candidate(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, legacy, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    current = publish_revision(
        conn,
        "publish-old",
        legacy,
        artifact,
        {},
        finalize=lambda db, operation: None,
    )
    historical = model_dir / "revisions" / current.revision
    disguised = XCalibArtifact(
        revision="rev-000004",
        parent_revision=current.revision,
        model_name=current.model_name,
        directory=historical,
        weights_path=historical / "crlite.pth",
        config_path=historical / "crlite.yaml",
        sha256=_sha256(historical / "crlite.pth"),
        base_fingerprint="tiny-base-v1",
    )

    with pytest.raises(PublicationConflict, match="temporary|historical|source"):
        prepare_publication(conn, "rename-history", current, disguised, {})
    assert historical.is_dir()
    assert not (model_dir / "revisions" / "rev-000004").exists()


def test_switch_revalidates_parent_weights_and_config_before_moving_candidate(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "parent-mutated", parent, artifact, {})
    parent.weights_path.write_bytes(b"mutated-parent")
    with pytest.raises(PublicationDivergence, match="parent"):
        switch_publication_pointer(conn, operation)
    assert artifact.directory.is_dir()
    assert not (model_dir / "current.json").exists()

    # Repeat with a versioned parent so an in-place config edit is detected too.
    fresh_root = tmp_path / "config-case"
    fresh_root.mkdir()
    config_conn = make_refine_db(fresh_root / "refine.sqlite")
    config_model = fresh_root / "cam_sync_model"
    legacy = _legacy_parent(config_model)
    first_artifact = _temporary_artifact(config_model, legacy, "rev-000001")
    first = publish_revision(
        config_conn,
        "publish-first",
        legacy,
        first_artifact,
        {},
        finalize=lambda db, operation: None,
    )
    second_artifact = _temporary_artifact(config_model, first, "rev-000002")
    second_operation = prepare_publication(
        config_conn, "config-mutated", first, second_artifact, {}
    )
    first.config_path.write_text("model: changed-in-place\n", encoding="utf-8")
    with pytest.raises(PublicationDivergence, match="parent|config"):
        switch_publication_pointer(config_conn, second_operation)
    assert second_artifact.directory.is_dir()


def test_publication_validates_revision_components_and_hashes_run_id_for_temp_path(
    tmp_path, monkeypatch
):
    import xcalib.continual.publication as implementation

    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    with pytest.raises(ValueError, match="revision"):
        prepare_publication(
            conn, "bad-revision", parent, replace(artifact, revision="../escape"), {}
        )
    with pytest.raises(ValueError, match="target_revision|revision"):
        rollback_revision(
            conn,
            "bad-rollback",
            model_dir,
            target_revision="..\\escape",
            finalize=lambda db, operation: None,
        )

    raw_run_id = "../unsafe\\run"
    operation = prepare_publication(conn, raw_run_id, parent, artifact, {})
    replaced_paths: list[Path] = []
    real_replace = implementation.os.replace

    def capture_replace(source, destination):
        replaced_paths.append(Path(source))
        return real_replace(source, destination)

    monkeypatch.setattr(implementation.os, "replace", capture_replace)
    switch_publication_pointer(conn, operation)
    pointer_temp = replaced_paths[-1]
    assert pointer_temp.parent == model_dir
    assert pointer_temp.name.startswith("current.json.tmp-")
    assert raw_run_id not in pointer_temp.name


def test_finalize_failure_leaves_pointer_switched_for_idempotent_retry(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)

    def fail_finalize(db, operation):
        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        publish_revision(conn, "run-1", parent, artifact, {}, fail_finalize)
    assert read_operation(conn, "run-1").state == "pointer_switched"
    assert json.loads((model_dir / "current.json").read_text())["revision"] == (
        artifact.revision
    )

    calls: list[str] = []
    results = recover_publications(
        conn, finalize=lambda db, operation: calls.append(operation.run_id)
    )
    assert results == ["run-1:committed"]
    assert calls == ["run-1"]


def test_direct_commit_revalidates_pointer_and_does_not_finalize_divergence(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "run-1", parent, artifact, {})
    operation = switch_publication_pointer(conn, operation)
    divergent_bytes = _canonical({"revision": "rev-diverged"}).encode()
    (model_dir / "current.json").write_bytes(divergent_bytes)
    finalized: list[str] = []

    with pytest.raises(PublicationDivergence):
        commit_publication(
            conn,
            operation,
            finalize=lambda db, value: finalized.append(value.run_id),
        )

    assert finalized == []
    assert read_operation(conn, "run-1").state == "diverged"
    assert (model_dir / "current.json").read_bytes() == divergent_bytes


def test_commit_rejects_dirty_caller_transaction_before_divergence_updates(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    operation = prepare_publication(conn, "run-1", parent, artifact, {})
    operation = switch_publication_pointer(conn, operation)
    conn.execute("CREATE TABLE unrelated_work (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO unrelated_work(value) VALUES ('must-rollback')")
    assert conn.in_transaction is True
    (model_dir / "current.json").write_text(
        _canonical({"revision": "rev-diverged"}), encoding="utf-8"
    )

    with pytest.raises(PublicationConflict, match="clean transaction"):
        commit_publication(conn, operation, finalize=lambda db, value: None)

    assert conn.in_transaction is True
    assert read_operation(conn, "run-1").state == "pointer_switched"
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM unrelated_work").fetchone()[0] == 0


def test_rollback_uses_the_same_journal_and_keeps_target_artifact_in_place(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir = tmp_path / "cam_sync_model"
    parent = _legacy_parent(model_dir)
    _trust_artifacts(monkeypatch)
    first_artifact = _temporary_artifact(model_dir, parent, "rev-000001")
    first = publish_revision(
        conn,
        "publish-old",
        parent,
        first_artifact,
        {},
        finalize=lambda db, operation: None,
    )
    latest_artifact = _temporary_artifact(model_dir, first, "rev-000003")
    publish_revision(
        conn,
        "publish-latest",
        first,
        latest_artifact,
        {},
        finalize=lambda db, operation: None,
    )
    target = model_dir / "revisions" / first.revision
    target_weights = target / "crlite.pth"
    target_bytes = target_weights.read_bytes()

    rolled_back = rollback_revision(
        conn,
        "rollback-1",
        model_dir,
        target_revision="rev-000001",
        finalize=lambda db, operation: None,
    )

    assert read_operation(conn, "rollback-1").state == "committed"
    assert json.loads((model_dir / "current.json").read_text())["revision"] == (
        rolled_back.revision
    )
    assert rolled_back.sha256 == _sha256(target_weights)
    assert target_weights.read_bytes() == target_bytes


def test_rollback_ignores_failed_or_uncommitted_artifact_metadata(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, latest_artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    current = publish_revision(
        conn,
        "publish-latest",
        parent,
        latest_artifact,
        {},
        finalize=lambda db, operation: None,
    )
    failed_artifact = _temporary_artifact(model_dir, current, "rev-000001")
    failed = prepare_publication(
        conn, "failed-old", current, failed_artifact, {}
    )
    conn.execute(
        "UPDATE publication_operations SET state = 'failed' WHERE run_id = ?",
        (failed.run_id,),
    )
    conn.commit()
    target = model_dir / "revisions" / failed_artifact.revision
    failed_artifact.directory.replace(target)

    with pytest.raises(PublicationConflict, match="committed|journal"):
        rollback_revision(
            conn,
            "rollback-failed",
            model_dir,
            target_revision=failed_artifact.revision,
            finalize=lambda db, operation: None,
        )


def test_pruning_keeps_current_plus_five_and_protects_nonterminal_references(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir = tmp_path / "cam_sync_model"
    parent = _legacy_parent(model_dir)
    first_parent_revision = parent.revision
    _trust_artifacts(monkeypatch)

    for index in range(1, 9):
        revision = f"rev-{index:06d}"
        artifact = _temporary_artifact(model_dir, parent, revision)
        parent = publish_revision(
            conn,
            f"publish-{index}",
            parent,
            artifact,
            {},
            finalize=lambda db, operation: None,
        )

    protected_dir = model_dir / "revisions" / "rev-000001"
    protected_artifact = XCalibArtifact(
        revision="rev-000001",
        parent_revision=first_parent_revision,
        model_name="crlite",
        directory=protected_dir,
        weights_path=protected_dir / "crlite.pth",
        config_path=protected_dir / "crlite.yaml",
        sha256=_sha256(protected_dir / "crlite.pth"),
        base_fingerprint="tiny-base-v1",
    )
    prepare_publication(
        conn,
        "protect-old-revision",
        parent,
        protected_artifact,
        {},
    )

    removed = prune_revision_artifacts(conn, model_dir, keep_previous=5)

    assert removed == ["rev-000002"]
    assert protected_dir.is_dir()
    assert (model_dir / "revisions" / "rev-000008").is_dir()
    assert all(
        (model_dir / "revisions" / f"rev-{index:06d}").is_dir()
        for index in range(3, 9)
    )
    assert (model_dir / "crlite.pth").read_bytes() == b"legacy-parent"


def test_pruning_rejects_invalid_digest_and_uncommitted_current_revision(
    tmp_path, monkeypatch
):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir = tmp_path / "cam_sync_model"
    parent = _legacy_parent(model_dir)
    _trust_artifacts(monkeypatch)
    for index in range(1, 4):
        artifact = _temporary_artifact(model_dir, parent, f"rev-{index:06d}")
        parent = publish_revision(
            conn,
            f"publish-{index}",
            parent,
            artifact,
            {},
            finalize=lambda db, operation: None,
        )
    pointer_path = model_dir / "current.json"
    valid_pointer = json.loads(pointer_path.read_text())
    invalid_digest = dict(valid_pointer, sha256="0" * 64)
    pointer_path.write_text(_canonical(invalid_digest), encoding="utf-8")
    with pytest.raises(PublicationDivergence, match="sha256|digest|resolve"):
        prune_revision_artifacts(conn, model_dir, keep_previous=1)

    uncommitted_dir = model_dir / "revisions" / "rev-999999"
    uncommitted_dir.mkdir()
    uncommitted_weights = uncommitted_dir / "crlite.pth"
    uncommitted_weights.write_bytes(b"load-valid-under-stub")
    (uncommitted_dir / "crlite.yaml").write_text("model: crlite\n")
    uncommitted_pointer = dict(
        valid_pointer,
        revision="rev-999999",
        parent_revision=valid_pointer["revision"],
        weights="revisions/rev-999999/crlite.pth",
        config="revisions/rev-999999/crlite.yaml",
        sha256=_sha256(uncommitted_weights),
    )
    pointer_path.write_text(_canonical(uncommitted_pointer), encoding="utf-8")
    with pytest.raises(PublicationConflict, match="committed"):
        prune_revision_artifacts(conn, model_dir, keep_previous=1)
    assert (model_dir / "revisions" / "rev-000001").is_dir()


def _publish(conn, run_id, parent, artifact):
    return publish_revision(conn, run_id, parent, artifact, {}, lambda db, op: None)


def _wedge(conn, model_dir, parent, revision="rev-000009", reason="recovery mismatch"):
    # Realistic wedge: only IN-FLIGHT ops diverge (a committed op is guarded from leaving
    # 'committed'). Prepare a fresh op -- which does NOT move current.json -- then mark it
    # diverged, exactly as recovery does on a pointer mismatch. This leaves the live
    # pointer and every committed revision untouched.
    artifact = _temporary_artifact(model_dir, parent, revision=revision)
    prepare_publication(conn, f"wedge-{revision}", parent, artifact, {})
    conn.execute(
        "UPDATE publication_operations SET state='diverged', last_error=? WHERE run_id=?",
        (reason, f"wedge-{revision}"),
    )
    conn.commit()
    return f"wedge-{revision}"


def test_remediate_divergence_repoints_to_target_and_unwedges(tmp_path, monkeypatch):
    # A diverged op fail-closes the whole model dir. Operator remediation re-points
    # current.json to a known-good committed revision, load-verifies it, then clears the
    # marker so refine can publish again.
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact_a = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    ref_a = _publish(conn, "run-a", parent, artifact_a)
    artifact_b = _temporary_artifact(model_dir, ref_a, revision="rev-000004")
    ref_b = _publish(conn, "run-b", ref_a, artifact_b)
    assert resolve_xcalib_revision(model_dir).revision == "rev-000004"

    wedge = _wedge(conn, model_dir, ref_b)
    # The dir is fail-closed: even a rollback (which checks divergence first) is blocked.
    with pytest.raises(PublicationDivergence):
        rollback_revision(conn, "blocked", model_dir, "rev-000003", lambda db, op: None)

    ref = remediate_divergence(conn, "fix-1", model_dir, target_revision="rev-000003")
    assert ref.revision == "rev-000003"
    assert resolve_xcalib_revision(model_dir).revision == "rev-000003"
    assert list_divergent_operations(conn, model_dir) == []
    assert read_operation(conn, wedge).state == "failed"
    assert "fix-1" in read_operation(conn, wedge).last_error
    # Unwedged, and rev-000004's committed provenance survived: rollback works again.
    rolled = rollback_revision(conn, "roll-1", model_dir, "rev-000004", lambda db, op: None)
    assert rolled.revision == "rev-000004"


def test_remediate_divergence_without_target_acknowledges_resolvable_pointer(
    tmp_path, monkeypatch
):
    # A spurious/transient marker over a still-resolvable live pointer clears with no
    # re-point: the live current.json must merely strict-resolve (schema + sha vs disk).
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    ref_a = _publish(conn, "run-a", parent, artifact)
    _wedge(conn, model_dir, ref_a, reason="transient parse blip")

    ref = remediate_divergence(conn, "fix-2", model_dir)
    assert ref.revision == "rev-000003"
    assert list_divergent_operations(conn, model_dir) == []


def test_remediate_divergence_requires_a_marker(tmp_path, monkeypatch):
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    _publish(conn, "run-a", parent, artifact)
    with pytest.raises(PublicationConflict, match="no divergent operation"):
        remediate_divergence(conn, "fix-3", model_dir)


def test_remediate_divergence_rejects_unknown_target_and_keeps_marker(
    tmp_path, monkeypatch
):
    # Safety invariant: a bad target must abort BEFORE the marker is cleared, so a failed
    # remediation never un-wedges a model that still cannot serve.
    conn = make_refine_db(tmp_path / "refine.sqlite")
    model_dir, parent, artifact = prepared_artifact(tmp_path)
    _trust_artifacts(monkeypatch)
    ref_a = _publish(conn, "run-a", parent, artifact)
    _wedge(conn, model_dir, ref_a)
    with pytest.raises(PublicationConflict, match="no committed journal entry"):
        remediate_divergence(conn, "fix-4", model_dir, target_revision="rev-999999")
    assert len(list_divergent_operations(conn, model_dir)) == 1


def test_relocating_the_model_dir_re_anchors_the_journal(tmp_path, monkeypatch):
    # Move the whole data tree (as open_store lays it out: refine.sqlite INSIDE the model
    # dir, alongside revisions/ and current.json). The stored absolute model_dir goes
    # stale, but re-opening at the new location must re-anchor every row to where the DB
    # now lives, so the journal keeps resolving/committing.
    from xcalib.continual.publication import _committed_operation_for_revision

    _trust_artifacts(monkeypatch)
    src = tmp_path / "site_v1" / "cam_model"
    src.mkdir(parents=True)
    conn = make_refine_db(src / "refine.sqlite")  # DB lives inside the model dir
    parent = _legacy_parent(src)
    ref = _publish(conn, "run-a", parent, _temporary_artifact(src, parent))
    assert ref.revision == "rev-000003"
    assert read_operation(conn, "run-a").model_dir == src.resolve()
    conn.close()

    dst = tmp_path / "site_v2" / "cam_model"
    dst.parent.mkdir(parents=True)
    shutil.move(str(src), str(dst))  # DB + revisions/ + current.json travel together

    moved = make_refine_db(dst / "refine.sqlite")  # migrate re-anchors on open
    op = read_operation(moved, "run-a")
    assert op.model_dir == dst.resolve()  # re-anchored to the NEW location, not the old
    assert resolve_xcalib_revision(dst).revision == "rev-000003"  # relative pointer resolves
    # A query keyed on the new dir finds the re-anchored committed op...
    assert _committed_operation_for_revision(moved, dst.resolve(), "rev-000003").run_id == "run-a"
    # ...and its destination artifact points at files under the new location.
    dest = xcalib_publication._destination_artifact(op)
    assert dest.weights_path == (dst / "revisions" / "rev-000003" / "crlite.pth").resolve()
