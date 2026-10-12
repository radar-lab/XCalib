from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Callable, Literal

from .refine_model import (
    XCalibArtifact,
    XCalibRevisionError,
    XCalibRevisionRef,
    resolve_xcalib_revision,
    verify_xcalib_revision,
)


PublicationState = Literal[
    "prepared", "pointer_switched", "committed", "failed", "diverged"
]


class PublicationError(RuntimeError):
    pass


class PublicationConflict(PublicationError):
    pass


class PublicationDivergence(PublicationError):
    pass


@dataclass(frozen=True)
class PublicationOperation:
    run_id: str
    model_dir: Path
    state: PublicationState
    old_pointer: dict | None
    new_pointer: dict
    artifact: XCalibArtifact
    parent: XCalibRevisionRef
    parent_config_sha256: str | None
    commit_payload: dict
    last_error: str | None
    created_at: str
    updated_at: str


_CREATE_PUBLICATIONS = """
CREATE TABLE IF NOT EXISTS publication_operations (
  run_id TEXT PRIMARY KEY,
  model_dir TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('prepared','pointer_switched','committed','failed','diverged')),
  old_pointer_json TEXT,
  new_pointer_json TEXT NOT NULL,
  artifact_json TEXT NOT NULL,
  commit_payload_json TEXT NOT NULL,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""
_POINTER_FIELDS = {
    "schema_version",
    "revision",
    "parent_revision",
    "model_name",
    "weights",
    "config",
    "sha256",
    "published_at",
}
_LIVE_CONFIG_SHA256 = object()


def _connection_model_dir(conn: sqlite3.Connection) -> Path | None:
    """The directory holding this connection's refine.sqlite -- which is, by construction
    (open_store writes ``<model_dir>/refine.sqlite``), the model dir every row belongs to.
    Deriving it from the live DB file makes the journal relocatable: the stored absolute
    ``model_dir`` becomes stale the moment the data tree is moved, but the DB physically
    travels inside its model dir, so its own location is always the truth."""
    for entry in conn.execute("PRAGMA database_list").fetchall():
        name = entry["name"] if isinstance(entry, sqlite3.Row) else entry[1]
        file = entry["file"] if isinstance(entry, sqlite3.Row) else entry[2]
        if name == "main" and file:
            return Path(file).resolve().parent
    return None


def migrate_publications(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_PUBLICATIONS)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS publication_operations_model_state "
        "ON publication_operations(model_dir, state, created_at)"
    )
    # Self-heal the anchor: re-point every row's model_dir at wherever this DB now lives.
    # File access re-derives from model_dir + the pointer's RELATIVE weights/config, so
    # re-anchoring here is all a relocated data tree needs to keep resolving/committing.
    # Idempotent (touches only rows that disagree); a no-op for the common in-place case.
    model_dir = _connection_model_dir(conn)
    if model_dir is not None:
        conn.execute(
            "UPDATE publication_operations SET model_dir = ? WHERE model_dir <> ?",
            (str(model_dir), str(model_dir)),
        )
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_pointer(value: dict) -> str:
    return _canonical_json(value) + "\n"


def _safe_revision(value: str, *, field: str = "revision") -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise ValueError(f"{field} must be one safe path component")
    if value in {".", ".."}:
        raise ValueError(f"{field} must be one safe path component")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise PublicationConflict(
            f"publication path escapes model directory: {value}"
        ) from exc
    return candidate


def _artifact_payload(artifact: XCalibArtifact) -> dict:
    return {
        "revision": artifact.revision,
        "parent_revision": artifact.parent_revision,
        "model_name": artifact.model_name,
        "directory": str(artifact.directory.resolve()),
        "weights_path": str(artifact.weights_path.resolve()),
        "config_path": str(artifact.config_path.resolve()),
        "sha256": artifact.sha256,
        "base_fingerprint": artifact.base_fingerprint,
    }


def _artifact_from_payload(payload: dict) -> XCalibArtifact:
    required = {
        "revision",
        "parent_revision",
        "model_name",
        "directory",
        "weights_path",
        "config_path",
        "sha256",
        "base_fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise PublicationConflict("journal artifact payload is incompatible")
    return XCalibArtifact(
        revision=payload["revision"],
        parent_revision=payload["parent_revision"],
        model_name=payload["model_name"],
        directory=Path(payload["directory"]),
        weights_path=Path(payload["weights_path"]),
        config_path=Path(payload["config_path"]),
        sha256=payload["sha256"],
        base_fingerprint=payload["base_fingerprint"],
    )


def _ref_payload(
    ref: XCalibRevisionRef,
    *,
    config_sha256: str | None | object = _LIVE_CONFIG_SHA256,
) -> dict:
    config_path = ref.config_path.resolve() if ref.config_path is not None else None
    if config_sha256 is _LIVE_CONFIG_SHA256:
        config_sha256 = (
            _sha256(config_path) if config_path is not None else None
        )
    return {
        "model_dir": str(ref.model_dir.resolve()),
        "model_name": ref.model_name,
        "revision": ref.revision,
        "parent_revision": ref.parent_revision,
        "weights_path": str(ref.weights_path.resolve()),
        "config_path": str(config_path) if config_path is not None else None,
        "sha256": ref.sha256,
        "source": ref.source,
        "config_sha256": config_sha256,
    }


def _ref_from_payload(payload: dict) -> tuple[XCalibRevisionRef, str | None]:
    required = {
        "model_dir",
        "model_name",
        "revision",
        "parent_revision",
        "weights_path",
        "config_path",
        "sha256",
        "source",
        "config_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise PublicationConflict("journal parent payload is incompatible")
    config_path = (
        Path(payload["config_path"])
        if payload["config_path"] is not None
        else None
    )
    return (
        XCalibRevisionRef(
            model_dir=Path(payload["model_dir"]),
            model_name=payload["model_name"],
            revision=payload["revision"],
            parent_revision=payload["parent_revision"],
            weights_path=Path(payload["weights_path"]),
            config_path=config_path,
            sha256=payload["sha256"],
            source=payload["source"],
        ),
        payload["config_sha256"],
    )


def _journal_artifact_payload(
    artifact: XCalibArtifact,
    parent: XCalibRevisionRef,
    *,
    parent_config_sha256: str | None | object = _LIVE_CONFIG_SHA256,
) -> dict:
    return {
        "artifact": _artifact_payload(artifact),
        "publication_parent": _ref_payload(
            parent, config_sha256=parent_config_sha256
        ),
    }


def _row_dict(cursor: sqlite3.Cursor, row) -> dict:
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return {
        description[0]: value
        for description, value in zip(cursor.description or (), row)
    }


def _operation_from_row(cursor: sqlite3.Cursor, row) -> PublicationOperation:
    values = _row_dict(cursor, row)
    journal_payload = json.loads(values["artifact_json"])
    if not isinstance(journal_payload, dict) or set(journal_payload) != {
        "artifact",
        "publication_parent",
    }:
        raise PublicationConflict("journal artifact wrapper is incompatible")
    parent, parent_config_sha256 = _ref_from_payload(
        journal_payload["publication_parent"]
    )
    return PublicationOperation(
        run_id=values["run_id"],
        model_dir=Path(values["model_dir"]),
        state=values["state"],
        old_pointer=(
            json.loads(values["old_pointer_json"])
            if values["old_pointer_json"] is not None
            else None
        ),
        new_pointer=json.loads(values["new_pointer_json"]),
        artifact=_artifact_from_payload(journal_payload["artifact"]),
        parent=parent,
        parent_config_sha256=parent_config_sha256,
        commit_payload=json.loads(values["commit_payload_json"]),
        last_error=values["last_error"],
        created_at=values["created_at"],
        updated_at=values["updated_at"],
    )


def read_operation(conn: sqlite3.Connection, run_id: str) -> PublicationOperation:
    cursor = conn.execute(
        "SELECT * FROM publication_operations WHERE run_id = ?", (run_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise KeyError(f"unknown publication operation: {run_id}")
    return _operation_from_row(cursor, row)


def _validate_pointer(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) != _POINTER_FIELDS:
        raise PublicationDivergence("current.json has an incompatible schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise PublicationDivergence("current.json schema_version is not integer 1")
    parent = payload["parent_revision"]
    if parent is not None and not isinstance(parent, str):
        raise PublicationDivergence("current.json parent_revision is invalid")
    for field in (
        "revision",
        "model_name",
        "weights",
        "config",
        "sha256",
        "published_at",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise PublicationDivergence(f"current.json {field} is invalid")
    _safe_revision(payload["revision"])
    return payload


def _read_pointer(model_dir: Path) -> dict | None:
    path = model_dir / "current.json"
    if not path.exists():
        return None
    text = None
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except PermissionError as exc:
            # Transient on Windows: an AV scan or a concurrent atomic replace of the
            # hot pointer briefly locks it. This is NOT divergence -- retry, and if
            # it never clears, raise a RETRYABLE conflict rather than permanently
            # bricking the model dir with a divergence marker.
            last_error = exc
            time.sleep(0.05)
    if text is None:
        raise PublicationConflict(f"current.json temporarily unreadable: {last_error}")
    try:
        return _validate_pointer(json.loads(text))
    except PublicationDivergence:
        raise
    except Exception as exc:
        raise PublicationDivergence(f"cannot parse current.json: {exc}") from exc


def _same_pointer(left: dict | None, right: dict | None) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _pointer_for_artifact(
    artifact: XCalibArtifact, *, published_at: str
) -> dict:
    revision = _safe_revision(artifact.revision)
    model_name = _safe_revision(artifact.model_name, field="model_name")
    relative_dir = Path("revisions") / revision
    return {
        "schema_version": 1,
        "revision": revision,
        "parent_revision": artifact.parent_revision,
        "model_name": model_name,
        "weights": (relative_dir / f"{model_name}.pth").as_posix(),
        "config": (relative_dir / f"{model_name}.yaml").as_posix(),
        "sha256": artifact.sha256,
        "published_at": published_at,
    }


def _refs_match(left: XCalibRevisionRef, right: XCalibRevisionRef) -> bool:
    left_config = (
        left.config_path.resolve() if left.config_path is not None else None
    )
    right_config = (
        right.config_path.resolve() if right.config_path is not None else None
    )
    return (
        left.model_dir.resolve() == right.model_dir.resolve()
        and left.model_name == right.model_name
        and left.revision == right.revision
        and left.parent_revision == right.parent_revision
        and left.sha256 == right.sha256
        and left.weights_path.resolve() == right.weights_path.resolve()
        and left_config == right_config
        and left.source == right.source
    )


def _check_model_not_diverged(conn: sqlite3.Connection, model_dir: Path) -> None:
    row = conn.execute(
        "SELECT run_id FROM publication_operations "
        "WHERE model_dir = ? AND state = 'diverged' LIMIT 1",
        (str(model_dir),),
    ).fetchone()
    if row is not None:
        run_id = row[0]
        raise PublicationDivergence(
            f"model publication is blocked by divergent operation {run_id}"
        )


def _validate_artifact_layout(
    model_dir: Path, artifact: XCalibArtifact
) -> tuple[Path, bool]:
    revision = _safe_revision(artifact.revision)
    model_name = _safe_revision(artifact.model_name, field="model_name")
    directory = _inside(model_dir, artifact.directory)
    weights = _inside(model_dir, artifact.weights_path)
    config = _inside(model_dir, artifact.config_path)
    if weights != directory / f"{model_name}.pth":
        raise PublicationConflict(
            "artifact weights filename must be exactly <model_name>.pth"
        )
    if config != directory / f"{model_name}.yaml":
        raise PublicationConflict(
            "artifact config filename must be exactly <model_name>.yaml"
        )
    final_directory = (model_dir / "revisions" / revision).resolve()
    already_final = directory == final_directory
    if not already_final:
        expected_temporary = (model_dir / f".{revision}.tmp").resolve()
        if directory != expected_temporary:
            raise PublicationConflict(
                "candidate source must be the exact hidden temporary revision directory"
            )
    return directory, already_final


def _committed_target_operation(
    conn: sqlite3.Connection,
    model_dir: Path,
    artifact: XCalibArtifact,
) -> PublicationOperation | None:
    cursor = conn.execute(
        "SELECT * FROM publication_operations "
        "WHERE model_dir = ? AND state = 'committed' "
        "ORDER BY updated_at DESC, created_at DESC, run_id DESC",
        (str(model_dir),),
    )
    for row in cursor.fetchall():
        operation = _operation_from_row(cursor, row)
        pointer = operation.new_pointer
        if (
            pointer.get("revision") == artifact.revision
            and pointer.get("model_name") == artifact.model_name
            and pointer.get("sha256") == artifact.sha256
            and operation.artifact.parent_revision == artifact.parent_revision
            and operation.artifact.base_fingerprint == artifact.base_fingerprint
        ):
            return operation
    return None


def prepare_publication(
    conn: sqlite3.Connection,
    run_id: str,
    parent: XCalibRevisionRef,
    artifact: XCalibArtifact,
    commit_payload: dict,
) -> PublicationOperation:
    if conn.in_transaction:
        raise PublicationConflict(
            "publication preparation requires a clean transaction"
        )
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(commit_payload, dict):
        raise ValueError("commit_payload must be a JSON object")
    _safe_revision(artifact.revision)
    model_dir = parent.model_dir.resolve()
    if artifact.model_name != parent.model_name:
        raise PublicationConflict(
            "candidate model_name must match the official parent model"
        )
    _safe_revision(artifact.model_name, field="model_name")
    _directory, already_final = _validate_artifact_layout(model_dir, artifact)
    commit_json = _canonical_json(commit_payload)

    try:
        existing = read_operation(conn, run_id)
    except KeyError:
        existing = None
    if existing is not None:
        expected_new = _pointer_for_artifact(
            artifact, published_at=existing.new_pointer["published_at"]
        )
        requested_artifact_json = _canonical_json(
            _journal_artifact_payload(
                artifact,
                parent,
                parent_config_sha256=existing.parent_config_sha256,
            )
        )
        persisted_artifact_json = _canonical_json(
            _journal_artifact_payload(
                existing.artifact,
                existing.parent,
                parent_config_sha256=existing.parent_config_sha256,
            )
        )
        if (
            existing.model_dir.resolve() != model_dir
            or _canonical_json(existing.new_pointer) != _canonical_json(expected_new)
            or persisted_artifact_json != requested_artifact_json
            or _canonical_json(existing.commit_payload) != commit_json
        ):
            raise PublicationConflict(
                f"run_id {run_id!r} already has a different publication payload"
            )
        if existing.state == "diverged":
            raise PublicationDivergence(
                f"publication operation {run_id!r} has diverged"
            )
        if existing.state in {"pointer_switched", "committed"}:
            _verify_destination(existing)
        elif existing.state == "prepared":
            verify_xcalib_revision(existing.artifact, device="cpu")
        return existing

    _check_model_not_diverged(conn, model_dir)
    committed_target = _committed_target_operation(
        conn, model_dir, artifact
    )
    if already_final:
        if committed_target is None:
            raise PublicationConflict(
                "an already-final artifact must have matching committed journal metadata"
            )
    elif artifact.parent_revision != parent.revision:
        raise PublicationConflict(
            "candidate parent_revision must match the official parent revision"
        )
    verified = verify_xcalib_revision(artifact, device="cpu")
    artifact_json = _canonical_json(
        _journal_artifact_payload(verified, parent)
    )

    try:
        resolved_parent = resolve_xcalib_revision(model_dir)
    except XCalibRevisionError as exc:
        raise PublicationDivergence(
            f"cannot verify current parent revision: {exc}"
        ) from exc
    if resolved_parent is None or not _refs_match(parent, resolved_parent):
        raise PublicationConflict(
            "published parent no longer matches the official model pointer"
        )
    old_pointer = _read_pointer(model_dir)
    now = _now()
    new_pointer = _pointer_for_artifact(verified, published_at=now)
    conn.execute(
        """
        INSERT INTO publication_operations (
          run_id, model_dir, state, old_pointer_json, new_pointer_json,
          artifact_json, commit_payload_json, last_error, created_at, updated_at
        ) VALUES (?, ?, 'prepared', ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            run_id,
            str(model_dir),
            _canonical_json(old_pointer) if old_pointer is not None else None,
            _canonical_json(new_pointer),
            artifact_json,
            commit_json,
            now,
            now,
        ),
    )
    conn.commit()
    return read_operation(conn, run_id)


def _set_state(
    conn: sqlite3.Connection,
    operation: PublicationOperation,
    state: PublicationState,
    *,
    error: str | None = None,
) -> PublicationOperation:
    # Never move out of a terminal fail-closed state: a suspended/resumed worker
    # that passed its checks before another connection marked the op `diverged`
    # (or `committed`) must not erase that marker by writing an in-flight state.
    conn.execute(
        "UPDATE publication_operations SET state = ?, last_error = ?, "
        "updated_at = ? WHERE run_id = ? AND state NOT IN ('diverged', 'committed')",
        (state, error, _now(), operation.run_id),
    )
    conn.commit()
    return read_operation(conn, operation.run_id)


def _diverge(
    conn: sqlite3.Connection, operation: PublicationOperation, message: str
):
    _set_state(conn, operation, "diverged", error=message)
    raise PublicationDivergence(message)


def _destination_artifact(operation: PublicationOperation) -> XCalibArtifact:
    model_dir = operation.model_dir.resolve()
    weights = _inside(model_dir, operation.new_pointer["weights"])
    config = _inside(model_dir, operation.new_pointer["config"])
    directory = weights.parent
    if config.parent != directory:
        raise PublicationDivergence("journalled weights/config directories disagree")
    return replace(
        operation.artifact,
        directory=directory,
        weights_path=weights,
        config_path=config,
    )


def _verify_destination(operation: PublicationOperation) -> XCalibArtifact:
    artifact = _destination_artifact(operation)
    if not artifact.weights_path.is_file() or not artifact.config_path.is_file():
        raise PublicationDivergence("journalled destination artifact is incomplete")
    actual = _sha256(artifact.weights_path)
    if actual != artifact.sha256:
        raise PublicationDivergence(
            f"journalled destination sha256 mismatch: expected {artifact.sha256}, got {actual}"
        )
    try:
        return verify_xcalib_revision(artifact, device="cpu")
    except XCalibRevisionError as exc:
        raise PublicationDivergence(
            f"journalled destination does not load-verify: {exc}"
        ) from exc


def _atomic_replace(src: Path, dst: Path, *, attempts: int = 20, delay: float = 0.05) -> None:
    """os.replace with a bounded retry on Windows sharing violations.

    Replacing a file a reader momentarily holds open raises PermissionError on
    Windows (ERROR_SHARING_VIOLATION; CPython opens without FILE_SHARE_DELETE).
    `current.json` is read on every match request, so a publish that lands while
    the UI polls would otherwise fail the run and leave the journal mid-flight.
    Retry briefly before giving up."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _write_pointer(model_dir: Path, run_id: str, payload: dict) -> None:
    token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
    temporary = model_dir / f"current.json.tmp-{token}"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_pointer(payload))
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, model_dir / "current.json")
    finally:
        temporary.unlink(missing_ok=True)


def _verify_prepared_parent(operation: PublicationOperation) -> None:
    try:
        current_parent = resolve_xcalib_revision(operation.model_dir)
    except XCalibRevisionError as exc:
        raise PublicationDivergence(
            f"prepared parent no longer strict-resolves: {exc}"
        ) from exc
    if current_parent is None or not _refs_match(
        current_parent, operation.parent
    ):
        raise PublicationDivergence(
            "prepared parent identity changed before pointer switch"
        )
    config_digest = (
        _sha256(current_parent.config_path)
        if current_parent.config_path is not None
        else None
    )
    if config_digest != operation.parent_config_sha256:
        raise PublicationDivergence(
            "prepared parent config changed before pointer switch"
        )


def switch_publication_pointer(
    conn: sqlite3.Connection, operation: PublicationOperation
) -> PublicationOperation:
    if conn.in_transaction:
        raise PublicationConflict(
            "publication pointer switch requires a clean transaction"
        )
    operation = read_operation(conn, operation.run_id)
    if operation.state == "committed":
        return operation
    if operation.state == "diverged":
        raise PublicationDivergence(
            operation.last_error or f"publication {operation.run_id} diverged"
        )
    if operation.state == "failed":
        raise PublicationConflict(f"publication {operation.run_id} has failed")
    _check_model_not_diverged(conn, operation.model_dir)
    try:
        current = _read_pointer(operation.model_dir)
    except PublicationDivergence as exc:
        _diverge(conn, operation, str(exc))
    if operation.state == "pointer_switched":
        if not _same_pointer(current, operation.new_pointer):
            _diverge(
                conn,
                operation,
                "official pointer changed after the publication switch",
            )
        try:
            _verify_destination(operation)
        except PublicationDivergence as exc:
            _diverge(conn, operation, str(exc))
        return operation
    if not _same_pointer(current, operation.old_pointer) and not _same_pointer(
        current, operation.new_pointer
    ):
        _diverge(
            conn,
            operation,
            "official pointer matches neither journalled old nor new payload",
        )

    destination = _destination_artifact(operation).directory
    source = operation.artifact.directory.resolve()
    if _same_pointer(current, operation.new_pointer):
        try:
            _verify_destination(operation)
        except PublicationDivergence as exc:
            _diverge(conn, operation, str(exc))
        return _set_state(conn, operation, "pointer_switched")

    try:
        _verify_prepared_parent(operation)
    except PublicationDivergence as exc:
        _diverge(conn, operation, str(exc))

    if destination.exists():
        if source != destination and source.exists():
            _diverge(
                conn,
                operation,
                "both temporary and destination revision directories exist",
            )
        try:
            _verify_destination(operation)
        except PublicationDivergence as exc:
            _diverge(conn, operation, str(exc))
    else:
        if source == destination or not source.is_dir():
            _diverge(
                conn,
                operation,
                "temporary revision disappeared before the pointer switch",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        try:
            _verify_destination(operation)
        except PublicationDivergence as exc:
            _diverge(conn, operation, str(exc))

    _write_pointer(operation.model_dir, operation.run_id, operation.new_pointer)
    return _set_state(conn, operation, "pointer_switched")


def commit_publication(
    conn: sqlite3.Connection,
    operation: PublicationOperation,
    finalize: Callable[[sqlite3.Connection, PublicationOperation], None],
) -> None:
    operation = read_operation(conn, operation.run_id)
    if operation.state == "committed":
        return
    if operation.state != "pointer_switched":
        raise PublicationConflict(
            f"cannot commit publication in state {operation.state!r}"
        )
    if conn.in_transaction:
        raise PublicationConflict("publication finalization requires a clean transaction")
    try:
        current = _read_pointer(operation.model_dir)
        if not _same_pointer(current, operation.new_pointer):
            _diverge(
                conn,
                operation,
                "official pointer diverged before publication finalization",
            )
        _verify_destination(operation)
    except PublicationDivergence as exc:
        refreshed = read_operation(conn, operation.run_id)
        if refreshed.state == "diverged":
            raise
        _diverge(conn, operation, str(exc))
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-read state inside the exclusive transaction. A concurrent committer or
        # a resumed recovery worker may have finalized this run while we were
        # load-verifying the destination (seconds-wide window). finalize() is NOT
        # idempotent -- it increments per-sample correct_streak counters -- so it
        # must run at most once. If someone already committed, roll back and return.
        row = conn.execute(
            "SELECT state FROM publication_operations WHERE run_id = ?",
            (operation.run_id,),
        ).fetchone()
        if row is None or row[0] != "pointer_switched":
            conn.rollback()
            return
        finalize(conn, operation)
        conn.execute(
            "UPDATE publication_operations SET state = 'committed', "
            "last_error = NULL, updated_at = ? WHERE run_id = ? "
            "AND state = 'pointer_switched'",
            (_now(), operation.run_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ref_from_operation(operation: PublicationOperation) -> XCalibRevisionRef:
    artifact = _destination_artifact(operation)
    if not artifact.weights_path.is_file() or _sha256(artifact.weights_path) != artifact.sha256:
        raise PublicationDivergence("published revision no longer matches its digest")
    return XCalibRevisionRef(
        model_dir=operation.model_dir.resolve(),
        model_name=artifact.model_name,
        revision=artifact.revision,
        parent_revision=artifact.parent_revision,
        weights_path=artifact.weights_path.resolve(),
        config_path=artifact.config_path.resolve(),
        sha256=artifact.sha256,
        source="pointer",
    )


def publish_revision(
    conn: sqlite3.Connection,
    run_id: str,
    parent: XCalibRevisionRef,
    artifact: XCalibArtifact,
    commit_payload: dict,
    finalize: Callable[[sqlite3.Connection, PublicationOperation], None],
) -> XCalibRevisionRef:
    operation = prepare_publication(
        conn, run_id, parent, artifact, commit_payload
    )
    operation = switch_publication_pointer(conn, operation)
    commit_publication(conn, operation, finalize)
    return _ref_from_operation(read_operation(conn, run_id))


def _remove_unreferenced_temporary(operation: PublicationOperation) -> None:
    candidate = operation.artifact.directory.resolve()
    model_dir = operation.model_dir.resolve()
    if candidate.parent != model_dir or not candidate.name.startswith("."):
        return
    try:
        current = _read_pointer(model_dir)
    except PublicationDivergence:
        return
    if current is not None:
        current_weights = _inside(model_dir, current["weights"])
        try:
            current_weights.relative_to(candidate)
            return
        except ValueError:
            pass
    if candidate.is_dir():
        shutil.rmtree(candidate)


def recover_publications(
    conn: sqlite3.Connection,
    finalize: Callable[[sqlite3.Connection, PublicationOperation], None],
) -> list[str]:
    if conn.in_transaction:
        raise PublicationConflict(
            "publication recovery requires a clean transaction"
        )
    cursor = conn.execute(
        "SELECT * FROM publication_operations "
        "WHERE state IN ('prepared', 'pointer_switched') "
        "ORDER BY created_at, run_id"
    )
    operations = [_operation_from_row(cursor, row) for row in cursor.fetchall()]
    results: list[str] = []
    for operation in operations:
        operation = read_operation(conn, operation.run_id)
        if operation.state == "prepared":
            try:
                current = _read_pointer(operation.model_dir)
            except PublicationDivergence as exc:
                _diverge(conn, operation, str(exc))
            if not _same_pointer(current, operation.old_pointer) and not _same_pointer(
                current, operation.new_pointer
            ):
                _diverge(
                    conn,
                    operation,
                    "recovery found a pointer different from both journalled values",
                )
            if _same_pointer(current, operation.old_pointer):
                try:
                    if operation.artifact.directory.is_dir():
                        verify_xcalib_revision(operation.artifact, device="cpu")
                    operation = switch_publication_pointer(conn, operation)
                except PublicationDivergence:
                    raise
                except Exception as exc:
                    operation = _set_state(
                        conn, operation, "failed", error=str(exc)
                    )
                    _remove_unreferenced_temporary(operation)
                    results.append(f"{operation.run_id}:failed")
                    continue
            else:
                operation = switch_publication_pointer(conn, operation)
        if operation.state == "pointer_switched":
            try:
                current = _read_pointer(operation.model_dir)
                if not _same_pointer(current, operation.new_pointer):
                    _diverge(
                        conn,
                        operation,
                        "pointer-switched recovery found a different official pointer",
                    )
                _verify_destination(operation)
            except PublicationDivergence as exc:
                _diverge(conn, operation, str(exc))
            commit_publication(conn, operation, finalize)
            results.append(f"{operation.run_id}:committed")
    return results


def _find_artifact_metadata(
    conn: sqlite3.Connection,
    model_dir: Path,
    revision: str,
    *,
    model_name: str,
    sha256: str,
) -> tuple[str | None, str]:
    cursor = conn.execute(
        "SELECT artifact_json, new_pointer_json FROM publication_operations "
        "WHERE model_dir = ? AND state = 'committed' "
        "ORDER BY updated_at DESC, created_at DESC",
        (str(model_dir),),
    )
    for row in cursor.fetchall():
        artifact_payload = json.loads(row[0])["artifact"]
        pointer_payload = json.loads(row[1])
        if (
            pointer_payload.get("revision") == revision
            and pointer_payload.get("model_name") == model_name
            and pointer_payload.get("sha256") == sha256
            and artifact_payload.get("sha256") == sha256
        ):
            return (
                artifact_payload.get("parent_revision"),
                artifact_payload["base_fingerprint"],
            )
    raise PublicationConflict(
        "rollback target has no matching committed journal metadata"
    )


def rollback_revision(
    conn: sqlite3.Connection,
    run_id: str,
    model_dir: str | Path,
    target_revision: str,
    finalize: Callable[[sqlite3.Connection, PublicationOperation], None],
) -> XCalibRevisionRef:
    revision = _safe_revision(target_revision, field="target_revision")
    root = Path(model_dir).resolve()
    _check_model_not_diverged(conn, root)
    parent = resolve_xcalib_revision(root)
    if parent is None:
        raise PublicationConflict("cannot rollback a model with no published parent")
    target = root / "revisions" / revision
    if target.parent.resolve() != (root / "revisions").resolve() or not target.is_dir():
        raise PublicationConflict("rollback target revision does not exist")
    weights = target / f"{parent.model_name}.pth"
    config = target / f"{parent.model_name}.yaml"
    if not weights.is_file() or not config.is_file():
        raise PublicationConflict("rollback target revision is incomplete")
    digest = _sha256(weights)
    original_parent, base_fingerprint = _find_artifact_metadata(
        conn,
        root,
        revision,
        model_name=parent.model_name,
        sha256=digest,
    )
    artifact = XCalibArtifact(
        revision=revision,
        parent_revision=original_parent,
        model_name=parent.model_name,
        directory=target,
        weights_path=weights,
        config_path=config,
        sha256=digest,
        base_fingerprint=base_fingerprint,
    )
    verify_xcalib_revision(artifact, device="cpu")
    return publish_revision(
        conn,
        run_id,
        parent,
        artifact,
        {},
        finalize,
    )


def _committed_operation_for_revision(
    conn: sqlite3.Connection, model_dir: Path, revision: str
) -> PublicationOperation | None:
    cursor = conn.execute(
        "SELECT * FROM publication_operations "
        "WHERE model_dir = ? AND state = 'committed' "
        "ORDER BY updated_at DESC, created_at DESC, run_id DESC",
        (str(model_dir),),
    )
    for row in cursor.fetchall():
        operation = _operation_from_row(cursor, row)
        if operation.new_pointer.get("revision") == revision:
            return operation
    return None


def list_divergent_operations(
    conn: sqlite3.Connection, model_dir: str | Path
) -> list[PublicationOperation]:
    """Every diverged op wedging a model dir, so an operator API can surface what needs
    acknowledging (each op's last_error explains why recovery fail-closed)."""
    root = Path(model_dir).resolve()
    cursor = conn.execute(
        "SELECT * FROM publication_operations "
        "WHERE model_dir = ? AND state = 'diverged' ORDER BY created_at, run_id",
        (str(root),),
    )
    return [_operation_from_row(cursor, row) for row in cursor.fetchall()]


def remediate_divergence(
    conn: sqlite3.Connection,
    run_id: str,
    model_dir: str | Path,
    *,
    target_revision: str | None = None,
) -> XCalibRevisionRef:
    """Operator escape hatch for a model whose publication journal is wedged 'diverged'.

    A single diverged op fail-closes the entire model dir: prepare/switch/commit/rollback
    all raise PublicationDivergence, so continual refine can never publish again. Divergence
    means the on-disk current.json stopped agreeing with the journal, so only a human who
    has inspected the dir may clear it.

    Two modes:
      * ``target_revision`` given -- re-point current.json at that previously-committed
        revision. Its stored pointer is replayed and the artifact is load-verified from
        disk (torch load + sha) BEFORE anything is written; a bad/incomplete target aborts
        with the marker still in place.
      * ``target_revision`` omitted -- acknowledge a spurious marker, trusting the live
        pointer. It must still strict-resolve (schema + sha256 vs disk) or the call aborts.

    Either way the marker is cleared (to 'failed', acknowledged) only after the live
    pointer resolves, so a failed remediation never un-wedges a model that cannot serve.
    Returns the live revision the model resolves to afterward.
    """
    if conn.in_transaction:
        raise PublicationConflict("divergence remediation requires a clean transaction")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")
    root = Path(model_dir).resolve()
    if not list_divergent_operations(conn, root):
        raise PublicationConflict("no divergent operation to remediate")

    if target_revision is not None:
        revision = _safe_revision(target_revision, field="target_revision")
        target = _committed_operation_for_revision(conn, root, revision)
        if target is None:
            raise PublicationConflict(
                f"remediation target {revision!r} has no committed journal entry"
            )
        artifact = _destination_artifact(target)
        if not artifact.weights_path.is_file() or not artifact.config_path.is_file():
            raise PublicationConflict(
                f"remediation target {revision!r} is incomplete on disk"
            )
        verify_xcalib_revision(artifact, device="cpu")
        _write_pointer(root, run_id, target.new_pointer)

    try:
        ref = resolve_xcalib_revision(root)
    except XCalibRevisionError as exc:
        raise PublicationConflict(
            f"cannot remediate: live pointer does not resolve ({exc}); "
            "supply target_revision to re-point"
        ) from exc
    if ref is None:
        raise PublicationConflict(
            "cannot remediate: model has no live pointer; supply target_revision"
        )

    # Direct UPDATE -- this IS the human override of the fail-closed guard, so _set_state
    # (which refuses to leave 'diverged') is bypassed on purpose.
    conn.execute(
        "UPDATE publication_operations SET state = 'failed', last_error = ?, "
        "updated_at = ? WHERE model_dir = ? AND state = 'diverged'",
        (f"remediated by {run_id} -> {ref.revision}", _now(), str(root)),
    )
    conn.commit()
    return ref


def prune_revision_artifacts(
    conn: sqlite3.Connection,
    model_dir: str | Path,
    keep_previous: int = 5,
) -> list[str]:
    if type(keep_previous) is not int or keep_previous < 0:
        raise ValueError("keep_previous must be a nonnegative integer")
    root = Path(model_dir).resolve()
    _check_model_not_diverged(conn, root)
    current_payload = _read_pointer(root)
    if current_payload is None:
        return []
    try:
        current_ref = resolve_xcalib_revision(root)
    except XCalibRevisionError as exc:
        raise PublicationDivergence(
            f"cannot strict-resolve current revision before pruning: {exc}"
        ) from exc
    if current_ref is None or current_ref.source != "pointer":
        raise PublicationConflict(
            "pruning requires a versioned current revision"
        )
    current_revision = _safe_revision(current_ref.revision)
    current_operation: PublicationOperation | None = None
    current_cursor = conn.execute(
        "SELECT * FROM publication_operations "
        "WHERE model_dir = ? AND state = 'committed' "
        "ORDER BY updated_at DESC, created_at DESC, run_id DESC",
        (str(root),),
    )
    for row in current_cursor.fetchall():
        operation = _operation_from_row(current_cursor, row)
        if _same_pointer(operation.new_pointer, current_payload):
            current_operation = operation
            break
    if current_operation is None:
        raise PublicationConflict(
            "current revision has no matching committed publication journal entry"
        )
    try:
        verified_current = _verify_destination(current_operation)
    except PublicationDivergence:
        raise
    if (
        verified_current.revision != current_ref.revision
        or verified_current.sha256 != current_ref.sha256
        or verified_current.weights_path.resolve() != current_ref.weights_path.resolve()
    ):
        raise PublicationDivergence(
            "current pointer and committed artifact disagree before pruning"
        )
    cursor = conn.execute(
        "SELECT new_pointer_json FROM publication_operations "
        "WHERE model_dir = ? AND state = 'committed' "
        "ORDER BY updated_at DESC, created_at DESC, run_id DESC",
        (str(root),),
    )
    committed_order: list[str] = []
    for row in cursor.fetchall():
        revision = json.loads(row[0]).get("revision")
        if (
            isinstance(revision, str)
            and revision not in committed_order
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", revision)
        ):
            committed_order.append(revision)
    if not committed_order:
        raise PublicationConflict("pruning requires a committed publication")
    keep = {current_revision}
    keep.update(
        revision
        for revision in committed_order
        if revision != current_revision
    )
    ordered_keep = [current_revision] + [
        revision
        for revision in committed_order
        if revision != current_revision
    ][:keep_previous]
    keep = set(ordered_keep)

    protected: set[str] = set()
    cursor = conn.execute(
        "SELECT new_pointer_json, artifact_json FROM publication_operations "
        "WHERE model_dir = ? AND state IN ('prepared', 'pointer_switched')",
        (str(root),),
    )
    for new_pointer_json, artifact_json in cursor.fetchall():
        pointer_revision = json.loads(new_pointer_json).get("revision")
        if isinstance(pointer_revision, str):
            protected.add(pointer_revision)
        artifact_directory = Path(
            json.loads(artifact_json)["artifact"]["directory"]
        ).resolve()
        revisions_root = (root / "revisions").resolve()
        if artifact_directory.parent == revisions_root:
            protected.add(artifact_directory.name)

    committed = set(committed_order)
    removed: list[str] = []
    revisions_dir = root / "revisions"
    if not revisions_dir.is_dir():
        return removed
    for directory in sorted(revisions_dir.iterdir(), key=lambda path: path.name):
        revision = directory.name
        if (
            directory.is_dir()
            and revision in committed
            and revision not in keep
            and revision not in protected
        ):
            shutil.rmtree(directory)
            removed.append(revision)
    return removed
