"""Durable review ledger and recoverable Apply-operation journal.

The corrections corpus + apply journal that back XCalib's continual-refine loop.
Pure-stdlib (sqlite3), no torch/engine dependency -- the foundation the rest of
``xcalib.continual`` layers on."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator, Literal, Sequence


class ReviewNotFound(LookupError):
    """An Apply request refers to a review that was never issued."""


class ReviewConflict(RuntimeError):
    """A durable review or operation identity was reused inconsistently."""


class ApplyBlocked(RuntimeError):
    """An unresolved operation prevents another writer from proceeding."""


@dataclass(frozen=True)
class IssuedReviewRecord:
    review_id: str
    scene: str
    camera: str
    frame: str
    model_revision: str
    checkpoint_sha256: str
    input_fingerprint: str
    payload_hash: str
    payload_json: str
    created_at: str


@dataclass(frozen=True)
class PendingFileRecord:
    relative_path: str
    before_sha256: str
    after_sha256: str
    after_json: str


@dataclass(frozen=True)
class ApplyOperationRecord:
    operation_id: str
    scene: str
    camera: str
    payload_hash: str
    request_json: str
    state: str
    result_json: str | None
    files: Sequence[PendingFileRecord]
    last_recovery_error: str | None


@dataclass(frozen=True)
class ReviewFrameRecord:
    review_id: str
    operation_id: str
    scene: str
    camera: str
    frame: str
    model_revision: str
    input_fingerprint: str
    verification_action: str
    applied_at: str


@dataclass(frozen=True)
class CorrectionRecord:
    sample_key: str
    observation_key: str
    scene: str
    camera: str
    frame: str
    review_id: str | None  # the issued review, or None for a direct corpus write
    input_fingerprint: str
    model_revision: str
    camera_track: str | None
    rect_json: str
    final_decision: str
    final_lidar_key: str | None
    original_proposal_json: str | None
    candidates_json: str
    provenance: str
    label_revision: int
    created_at: str
    updated_at: str
    superseded_by: str | None = None


@dataclass(frozen=True)
class CorpusCounts:
    frames: int
    corrections: int
    positives: int
    no_matches: int

    @property
    def total(self) -> int:
        return self.corrections


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', '1');
CREATE TABLE IF NOT EXISTS issued_reviews(
  review_id TEXT PRIMARY KEY, scene TEXT NOT NULL, camera TEXT NOT NULL,
  frame TEXT NOT NULL, model_revision TEXT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL, input_fingerprint TEXT NOT NULL,
  payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS apply_operations(
  operation_id TEXT PRIMARY KEY, scene TEXT NOT NULL, camera TEXT NOT NULL,
  payload_hash TEXT NOT NULL, request_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending','committed','needs_resolution')),
  result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  last_recovery_error TEXT
);
CREATE TABLE IF NOT EXISTS apply_operation_files(
  operation_id TEXT NOT NULL REFERENCES apply_operations(operation_id) ON DELETE CASCADE,
  relative_path TEXT NOT NULL, before_sha256 TEXT NOT NULL,
  after_sha256 TEXT NOT NULL, after_json TEXT NOT NULL,
  PRIMARY KEY(operation_id, relative_path)
);
CREATE TABLE IF NOT EXISTS review_frames(
  review_id TEXT PRIMARY KEY REFERENCES issued_reviews(review_id),
  operation_id TEXT NOT NULL REFERENCES apply_operations(operation_id),
  scene TEXT NOT NULL, camera TEXT NOT NULL, frame TEXT NOT NULL,
  model_revision TEXT NOT NULL, input_fingerprint TEXT NOT NULL,
  verification_action TEXT NOT NULL, applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections(
  sample_key TEXT PRIMARY KEY, observation_key TEXT NOT NULL,
  scene TEXT NOT NULL, camera TEXT NOT NULL,
  frame TEXT NOT NULL, review_id TEXT REFERENCES issued_reviews(review_id),
  input_fingerprint TEXT NOT NULL, model_revision TEXT NOT NULL,
  camera_track TEXT, rect_json TEXT NOT NULL,
  final_decision TEXT NOT NULL CHECK(final_decision IN ('positive','no_match')),
  final_lidar_key TEXT, original_proposal_json TEXT,
  candidates_json TEXT NOT NULL, provenance TEXT NOT NULL,
  label_revision INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  superseded_by TEXT REFERENCES corrections(sample_key),
  CHECK((final_decision='positive' AND final_lidar_key IS NOT NULL) OR
        (final_decision='no_match' AND final_lidar_key IS NULL))
);
CREATE INDEX IF NOT EXISTS corrections_frame_idx ON corrections(scene,camera,frame);
CREATE INDEX IF NOT EXISTS corrections_observation_idx ON corrections(observation_key,superseded_by);
CREATE INDEX IF NOT EXISTS review_frames_operation_idx ON review_frames(operation_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(row: sqlite3.Row, cls):
    names = {item.name for item in fields(cls)}
    return cls(**{name: row[name] for name in names})


def _check_one_sample_per_observation(
    corrections: "Sequence[CorrectionRecord]",
) -> None:
    """A single write may not map one observation to two sample keys."""
    observations: dict[str, str] = {}
    for row in corrections:
        prior = observations.setdefault(row.observation_key, row.sample_key)
        if prior != row.sample_key:
            raise ReviewConflict(
                "multiple sample keys describe one observation in one write"
            )


def _upsert_corrections(
    conn: sqlite3.Connection, corrections: "Sequence[CorrectionRecord]"
) -> None:
    """Insert/replace each correction, bump label_revision only when the decision
    changed, and supersede any older active correction for the same observation.

    The single source of truth for corpus writes: both the journalled apply
    (:meth:`XCalibStore.commit_apply`) and the lighter
    :meth:`XCalibStore.upsert_corrections` go through here so the two paths cannot
    drift. Caller holds an IMMEDIATE transaction.
    """
    for row in corrections:
        existing = conn.execute(
            "SELECT * FROM corrections WHERE sample_key=?", (row.sample_key,)
        ).fetchone()
        revision = row.label_revision
        created_at = row.created_at
        if existing is not None:
            unchanged = (
                existing["final_decision"] == row.final_decision
                and existing["final_lidar_key"] == row.final_lidar_key
            )
            revision = int(existing["label_revision"]) + (0 if unchanged else 1)
            created_at = existing["created_at"]

        values = {
            item.name: getattr(row, item.name) for item in fields(CorrectionRecord)
        }
        values["superseded_by"] = None
        values["label_revision"] = revision
        values["created_at"] = created_at
        conn.execute(
            """INSERT INTO corrections(
                   sample_key,observation_key,scene,camera,frame,review_id,
                   input_fingerprint,model_revision,camera_track,rect_json,
                   final_decision,final_lidar_key,original_proposal_json,
                   candidates_json,provenance,label_revision,created_at,
                   updated_at,superseded_by
               ) VALUES (
                   :sample_key,:observation_key,:scene,:camera,:frame,:review_id,
                   :input_fingerprint,:model_revision,:camera_track,:rect_json,
                   :final_decision,:final_lidar_key,:original_proposal_json,
                   :candidates_json,:provenance,:label_revision,:created_at,
                   :updated_at,:superseded_by
               ) ON CONFLICT(sample_key) DO UPDATE SET
                   observation_key=excluded.observation_key,
                   scene=excluded.scene,camera=excluded.camera,frame=excluded.frame,
                   review_id=excluded.review_id,
                   input_fingerprint=excluded.input_fingerprint,
                   model_revision=excluded.model_revision,
                   camera_track=excluded.camera_track,rect_json=excluded.rect_json,
                   final_decision=excluded.final_decision,
                   final_lidar_key=excluded.final_lidar_key,
                   original_proposal_json=excluded.original_proposal_json,
                   candidates_json=excluded.candidates_json,
                   provenance=excluded.provenance,
                   label_revision=excluded.label_revision,
                   created_at=excluded.created_at,updated_at=excluded.updated_at,
                   superseded_by=excluded.superseded_by""",
            values,
        )
        conn.execute(
            """UPDATE corrections SET superseded_by=?
               WHERE observation_key=? AND sample_key<>? AND superseded_by IS NULL""",
            (row.sample_key, row.observation_key, row.sample_key),
        )


class XCalibStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path, timeout=5.0, isolation_level=None
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        has_metadata = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if has_metadata is not None:
            existing = self.connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing is not None and existing["value"] not in {"1", "2", "3"}:
                self.connection.close()
                raise RuntimeError(
                    f"unsupported XCalib refine schema version: {existing['value']}"
                )
        try:
            self.connection.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA + "\nCOMMIT;")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.rollback()
            self.connection.close()
            raise
        version = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if version is None or version["value"] not in {"1", "2", "3"}:
            self.connection.close()
            raise RuntimeError(
                f"unsupported XCalib refine schema version: "
                f"{None if version is None else version['value']}"
            )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self, immediate: bool = False):
        if self.connection.in_transaction:
            raise RuntimeError("nested XCalibStore transaction is not allowed")
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def issue_review(self, record: IssuedReviewRecord) -> None:
        with self.transaction(immediate=True) as conn:
            existing = conn.execute(
                "SELECT * FROM issued_reviews WHERE review_id=?",
                (record.review_id,),
            ).fetchone()
            if existing is not None:
                current = _record(existing, IssuedReviewRecord)
                stable_fields = (
                    "scene",
                    "camera",
                    "frame",
                    "model_revision",
                    "checkpoint_sha256",
                    "input_fingerprint",
                    "payload_hash",
                    "payload_json",
                )
                if any(
                    getattr(current, name) != getattr(record, name)
                    for name in stable_fields
                ):
                    raise ReviewConflict(
                        f"immutable issued review {record.review_id} changed"
                    )
                return
            conn.execute(
                """INSERT INTO issued_reviews(
                       review_id,scene,camera,frame,model_revision,checkpoint_sha256,
                       input_fingerprint,payload_hash,payload_json,created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                tuple(getattr(record, item.name) for item in fields(IssuedReviewRecord)),
            )

    def get_issued_review(self, review_id: str) -> IssuedReviewRecord | None:
        row = self.connection.execute(
            "SELECT * FROM issued_reviews WHERE review_id=?", (review_id,)
        ).fetchone()
        return None if row is None else _record(row, IssuedReviewRecord)

    def begin_apply(
        self, operation: ApplyOperationRecord
    ) -> Literal["created", "pending", "committed"]:
        now = _now()
        with self.transaction(immediate=True) as conn:
            existing = conn.execute(
                "SELECT payload_hash,state FROM apply_operations WHERE operation_id=?",
                (operation.operation_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != operation.payload_hash:
                    raise ReviewConflict(
                        f"operation_id payload mismatch: {operation.operation_id}"
                    )
                if existing["state"] == "needs_resolution":
                    raise ApplyBlocked(
                        f"operation {operation.operation_id} needs_resolution"
                    )
                return existing["state"]

            blocked = conn.execute(
                """SELECT operation_id,state FROM apply_operations
                   WHERE scene=? AND camera=? AND state IN ('pending','needs_resolution')
                   ORDER BY created_at LIMIT 1""",
                (operation.scene, operation.camera),
            ).fetchone()
            if blocked is not None:
                raise ApplyBlocked(
                    f"operation {blocked['operation_id']} is {blocked['state']}"
                )

            try:
                review_ids = [
                    str(frame["review_id"])
                    for frame in json.loads(operation.request_json).get("frames", [])
                ]
            except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ReviewConflict("Apply operation request_json is invalid") from exc
            for review_id in review_ids:
                applied = conn.execute(
                    "SELECT operation_id FROM review_frames WHERE review_id=?",
                    (review_id,),
                ).fetchone()
                if applied is not None:
                    raise ReviewConflict(
                        f"issued review {review_id} was already applied by "
                        f"operation {applied['operation_id']}"
                    )

            conn.execute(
                """INSERT INTO apply_operations(
                       operation_id,scene,camera,payload_hash,request_json,state,
                       result_json,created_at,updated_at,last_recovery_error
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    operation.operation_id,
                    operation.scene,
                    operation.camera,
                    operation.payload_hash,
                    operation.request_json,
                    "pending",
                    None,
                    now,
                    now,
                    None,
                ),
            )
            conn.executemany(
                """INSERT INTO apply_operation_files(
                       operation_id,relative_path,before_sha256,after_sha256,after_json
                   ) VALUES (?,?,?,?,?)""",
                [
                    (
                        operation.operation_id,
                        item.relative_path,
                        item.before_sha256,
                        item.after_sha256,
                        item.after_json,
                    )
                    for item in operation.files
                ],
            )
        return "created"

    def get_apply_operation(self, operation_id: str) -> ApplyOperationRecord | None:
        row = self.connection.execute(
            "SELECT * FROM apply_operations WHERE operation_id=?", (operation_id,)
        ).fetchone()
        if row is None:
            return None
        file_rows = self.connection.execute(
            """SELECT relative_path,before_sha256,after_sha256,after_json
               FROM apply_operation_files WHERE operation_id=? ORDER BY relative_path""",
            (operation_id,),
        ).fetchall()
        return ApplyOperationRecord(
            operation_id=row["operation_id"],
            scene=row["scene"],
            camera=row["camera"],
            payload_hash=row["payload_hash"],
            request_json=row["request_json"],
            state=row["state"],
            result_json=row["result_json"],
            files=tuple(_record(item, PendingFileRecord) for item in file_rows),
            last_recovery_error=row["last_recovery_error"],
        )

    def commit_apply(
        self,
        operation_id: str,
        frames: Sequence[ReviewFrameRecord],
        corrections: Sequence[CorrectionRecord],
        result_json: str,
    ) -> None:
        frame_keys = [(row.scene, row.camera, row.frame) for row in frames]
        if len(frame_keys) != len(set(frame_keys)):
            raise ReviewConflict("duplicate scene/camera/frame in one Apply operation")
        _check_one_sample_per_observation(corrections)
        with self.transaction(immediate=True) as conn:
            operation = conn.execute(
                "SELECT state FROM apply_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise ReviewNotFound(f"unknown operation {operation_id}")
            if operation["state"] == "committed":
                return
            if operation["state"] != "pending":
                raise ApplyBlocked(
                    f"operation {operation_id} is {operation['state']}"
                )

            conn.executemany(
                """INSERT INTO review_frames(
                       review_id,operation_id,scene,camera,frame,model_revision,
                       input_fingerprint,verification_action,applied_at
                   ) VALUES (?,?,?,?,?,?,?,?,?)""",
                [tuple(getattr(row, item.name) for item in fields(ReviewFrameRecord))
                 for row in frames],
            )

            _upsert_corrections(conn, corrections)

            conn.execute(
                """UPDATE apply_operations
                   SET state='committed',result_json=?,updated_at=?,last_recovery_error=NULL
                   WHERE operation_id=?""",
                (result_json, _now(), operation_id),
            )

    def upsert_corrections(self, corrections: Sequence[CorrectionRecord]) -> int:
        """Write verified-pair corrections into the corpus WITHOUT the apply journal.

        The lighter path a front end without the issued-review workflow (e.g. the
        Qt review) uses to accumulate into the durable corpus: same upsert +
        observation-supersede semantics as :meth:`commit_apply`, minus the
        issued-review provenance and the apply-operation state machine. Returns the
        number of corrections written.
        """
        _check_one_sample_per_observation(corrections)
        with self.transaction(immediate=True) as conn:
            _upsert_corrections(conn, corrections)
        return len(corrections)

    def mark_apply_needs_resolution(self, operation_id: str, error: str) -> None:
        with self.transaction(immediate=True) as conn:
            conn.execute(
                """UPDATE apply_operations SET state='needs_resolution',
                   last_recovery_error=?,updated_at=? WHERE operation_id=?""",
                (error, _now(), operation_id),
            )

    def pending_apply_operations(self) -> list[ApplyOperationRecord]:
        rows = self.connection.execute(
            "SELECT operation_id FROM apply_operations WHERE state='pending' ORDER BY created_at"
        ).fetchall()
        return [self.get_apply_operation(row["operation_id"]) for row in rows]

    def get_correction(self, sample_key: str) -> CorrectionRecord | None:
        row = self.connection.execute(
            "SELECT * FROM corrections WHERE sample_key=?", (sample_key,)
        ).fetchone()
        return None if row is None else _record(row, CorrectionRecord)

    def iter_corrections(
        self,
        *,
        after_sample_key: str | None = None,
        limit: int = 1000,
        include_superseded: bool = False,
    ) -> Iterator[CorrectionRecord]:
        if limit <= 0:
            return iter(())
        clauses = []
        params: list[object] = []
        if after_sample_key is not None:
            clauses.append("sample_key > ?")
            params.append(after_sample_key)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        rows = self.connection.execute(
            f"SELECT * FROM corrections{where} ORDER BY sample_key LIMIT ?", params
        ).fetchall()
        return iter([_record(row, CorrectionRecord) for row in rows])

    def corpus_counts(self) -> CorpusCounts:
        frames = self.connection.execute(
            "SELECT COUNT(*) FROM review_frames"
        ).fetchone()[0]
        row = self.connection.execute(
            """SELECT COUNT(*) AS corrections,
                      SUM(CASE WHEN final_decision='positive' THEN 1 ELSE 0 END) positives,
                      SUM(CASE WHEN final_decision='no_match' THEN 1 ELSE 0 END) no_matches
               FROM corrections WHERE superseded_by IS NULL"""
        ).fetchone()
        return CorpusCounts(
            frames=int(frames),
            corrections=int(row["corrections"] or 0),
            positives=int(row["positives"] or 0),
            no_matches=int(row["no_matches"] or 0),
        )


def open_store(model_root: str | Path) -> XCalibStore:
    return XCalibStore(Path(model_root) / "refine.sqlite")


__all__ = [
    "ApplyBlocked",
    "ApplyOperationRecord",
    "CorpusCounts",
    "CorrectionRecord",
    "IssuedReviewRecord",
    "PendingFileRecord",
    "ReviewConflict",
    "ReviewFrameRecord",
    "ReviewNotFound",
    "XCalibStore",
    "open_store",
]
