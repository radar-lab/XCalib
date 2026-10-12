"""SQLite-backed finite XCalib refine jobs and cross-process writer leases."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Literal, Mapping
import uuid

from .store import open_store

RunStatus = Literal["queued", "running", "accepted", "rejected", "work_remaining",
                    "up_to_date", "failed", "interrupted"]
RunPhase = Literal["indexing", "caching", "classifying", "training", "validating",
                   "auditing", "publishing"]
_ACTIVE = ("queued", "running")
_TERMINAL = ("accepted", "rejected", "work_remaining", "up_to_date", "failed", "interrupted")
_LEASE_SECONDS = 30.0
# A user-cancelled run lands in 'interrupted' — already an allowed status — and is told
# apart by this exact `failure_detail`. A first-class 'cancelled' status would mean
# rebuilding the CHECK constraint on live corpora (the schema is created with
# CREATE TABLE IF NOT EXISTS, so existing DBs keep the old constraint); that rebuild is
# not worth it for a label. `DurableJobState.failure_detail` carries this to the UI.
CANCELLED_DETAIL = "cancelled by user"


class LeaseLost(RuntimeError):
    pass


class CancelUnavailable(RuntimeError):
    """The run cannot be cancelled at this instant (see `.phase`)."""

    def __init__(self, message: str, *, phase: str | None = None):
        super().__init__(message)
        self.phase = phase


class WriterLeaseConflict(RuntimeError):
    def __init__(self, owner: str):
        super().__init__(f"XCalib writer is locked by {owner}")
        self.owner = owner


@dataclass(frozen=True)
class StartResult:
    run_id: str | None
    disposition: Literal["started", "existing", "locked"]
    lock_owner: str | None = None


@dataclass(frozen=True)
class DurableJobState:
    id: str
    title: str
    scene: str
    camera: str
    status: RunStatus
    phase: RunPhase | None
    current: int
    total: int
    percent: int
    message: str
    parent_revision: str | None
    new_revision: str | None
    best_epoch: int | None
    counts: dict[str, int]
    validation: dict[str, Any] | None
    audit: dict[str, Any] | None
    gate_reasons: tuple[str, ...]
    failure_detail: str | None
    started_at: str
    finished_at: str | None


@dataclass(frozen=True)
class RefineWorkload:
    scene: str
    camera: str
    counts: dict[str, int]
    active: DurableJobState | None
    latest: DurableJobState | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS continual_runs(
 run_id TEXT PRIMARY KEY, scene TEXT NOT NULL, camera TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('queued','running','accepted','rejected',
 'work_remaining','up_to_date','failed','interrupted')),
 parent_revision TEXT, corpus_watermark TEXT NOT NULL DEFAULT '{}',
 train_config_json TEXT NOT NULL DEFAULT '{}', work_counts_json TEXT NOT NULL DEFAULT '{}',
 phase TEXT, current INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
 message TEXT NOT NULL DEFAULT '', best_epoch INTEGER, validation_json TEXT,
 audit_json TEXT, gate_reasons_json TEXT NOT NULL DEFAULT '[]', new_revision TEXT,
 failure_detail TEXT, result_json TEXT, owner_pid INTEGER, lease_token TEXT,
 heartbeat_at REAL, started_at TEXT NOT NULL, finished_at TEXT, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS continual_runs_camera_idx
 ON continual_runs(scene,camera,started_at DESC);
CREATE TABLE IF NOT EXISTS writer_leases(
 scene TEXT NOT NULL, camera TEXT NOT NULL, owner_kind TEXT NOT NULL,
 owner_id TEXT NOT NULL, token TEXT NOT NULL, owner_pid INTEGER,
 heartbeat_at REAL NOT NULL, expires_at REAL NOT NULL, PRIMARY KEY(scene,camera)
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _epoch() -> float:
    return time.time()


def _json(value: Any) -> str:
    return json.dumps(asdict(value) if is_dataclass(value) else value,
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected dataclass or mapping, got {type(value).__name__}")


def _infer_scope(root: Path) -> tuple[str, str]:
    camera = root.name[:-6] if root.name.endswith("_model") else root.name
    scene = root.parents[2].name if len(root.parents) > 2 else "default"
    return scene, camera


def _pid_live(pid: int | None) -> bool:
    """Whether `pid` is a live process.

    CRITICAL: never use `os.kill(pid, 0)` on Windows — CPython maps a signal of 0
    to `TerminateProcess(handle, 0)`, so it KILLS the probed process (and returns
    as if it were alive). Prefer psutil when it is installed (it is optional); on
    win32 fall back to OpenProcess+WaitForSingleObject; only POSIX uses
    os.kill(pid, 0).
    """
    if pid is None or pid <= 0:
        return False
    try:
        import psutil  # optional; any failure falls back to the probes below
        return psutil.pid_exists(int(pid))
    except Exception:
        pass
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ContinualJobs:
    def __init__(self, model_root: str | Path, scene: str | None = None,
                 camera: str | None = None):
        self.model_root = Path(model_root).resolve()
        inferred_scene, inferred_camera = _infer_scope(self.model_root)
        self.scene, self.camera = scene or inferred_scene, camera or inferred_camera
        self.store = open_store(self.model_root)
        self._migrate()

    @classmethod
    def for_scene(cls, data_root: str | Path, scene: str, camera: str) -> "ContinualJobs":
        from .locations import model_dir
        return cls(model_dir(data_root, scene, camera), scene, camera)

    def close(self) -> None:
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _migrate(self) -> None:
        from .publication import migrate_publications
        migrate_publications(self.store.connection)
        with self.store.transaction(immediate=True) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            version = None if row is None else str(row["value"])
            if version not in {"1", "2", "3"}:
                raise RuntimeError(f"unsupported XCalib refine schema version: {version}")
            db.executescript(_SCHEMA)
            if version != "3":
                db.execute("UPDATE metadata SET value='3' WHERE key='schema_version'")

    def _counts(self) -> dict[str, int]:
        db = self.store.connection
        corpus = db.execute(
            """SELECT COUNT(DISTINCT frame) frames,COUNT(*) corrections,
               SUM(final_decision='positive') positives,SUM(final_decision='no_match') no_matches
               FROM corrections WHERE scene=? AND camera=? AND superseded_by IS NULL""",
            (self.scene, self.camera)).fetchone()
        counts = {"frames": int(corpus["frames"] or 0),
                  "corrections": int(corpus["corrections"] or 0),
                  "positives": int(corpus["positives"] or 0),
                  "no_matches": int(corpus["no_matches"] or 0),
                  "new": 0, "hard": 0, "mastered": 0,
                  "missing_features": 0, "validation_features": 0,
                  "remaining_new": 0, "remaining_hard": 0}
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_state'").fetchone()
        if exists:
            for row in db.execute(
                "SELECT difficulty_state,COUNT(*) n FROM sample_state WHERE scene=? AND camera=? GROUP BY difficulty_state",
                (self.scene, self.camera)).fetchall():
                counts[str(row["difficulty_state"])] = int(row["n"])
            counts["missing_features"] = int(db.execute(
                """SELECT COUNT(DISTINCT frame||char(0)||feature_input_fingerprint)
                   FROM sample_state WHERE scene=? AND camera=?
                   AND feature_state IN ('missing','stale','error')""",
                (self.scene, self.camera)).fetchone()[0] or 0)
            counts["validation_features"] = int(db.execute(
                """SELECT COUNT(DISTINCT frame||char(0)||feature_input_fingerprint)
                   FROM sample_state WHERE scene=? AND camera=? AND dataset_split='validation'
                   AND feature_state IN ('missing','stale','error')""",
                (self.scene, self.camera)).fetchone()[0] or 0)
        # hard/mastered come from sample_state, which is only built while a refine
        # RUN indexes + evaluates the corpus. Until then it's empty and the footer
        # read 0/0/0 even with a full corpus. Derive "new" from the live corrections
        # instead, so the readout reflects the corpus immediately: every correction
        # a run hasn't yet classified as hard/mastered is new (unevaluated).
        counts["new"] = max(
            0, counts["corrections"] - counts["hard"] - counts["mastered"]
        )
        return counts

    def _state(self, row) -> DurableJobState:
        counts = self._counts()
        counts.update({str(k): int(v) for k, v in json.loads(row["work_counts_json"] or "{}").items()})
        current, total = int(row["current"]), int(row["total"])
        return DurableJobState(
            id=str(row["run_id"]), title=f"GT refine — {row['scene']}/{row['camera']}",
            scene=str(row["scene"]), camera=str(row["camera"]), status=row["status"],
            phase=row["phase"], current=current, total=total,
            percent=0 if total <= 0 else max(0, min(100, round(100 * current / total))),
            message=str(row["message"]), parent_revision=row["parent_revision"],
            new_revision=row["new_revision"], best_epoch=row["best_epoch"], counts=counts,
            validation=json.loads(row["validation_json"]) if row["validation_json"] else None,
            audit=json.loads(row["audit_json"]) if row["audit_json"] else None,
            gate_reasons=tuple(json.loads(row["gate_reasons_json"] or "[]")),
            failure_detail=row["failure_detail"], started_at=str(row["started_at"]),
            finished_at=row["finished_at"])

    def get(self, run_id: str) -> DurableJobState | None:
        row = self.store.connection.execute("SELECT * FROM continual_runs WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else self._state(row)

    def workload(self) -> RefineWorkload:
        rows = self.store.connection.execute(
            "SELECT * FROM continual_runs WHERE scene=? AND camera=? ORDER BY started_at DESC,run_id DESC",
            (self.scene, self.camera)).fetchall()
        states = [self._state(row) for row in rows]
        return RefineWorkload(self.scene, self.camera, self._counts(),
                              next((s for s in states if s.status in _ACTIVE), None),
                              states[0] if states else None)

    def _run(self, db, run_id: str, token: str | None):
        row = db.execute("SELECT * FROM continual_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise LeaseLost(f"unknown continual refine run {run_id}")
        if row["lease_token"] is not None and row["lease_token"] != token:
            raise LeaseLost(f"lease token mismatch for continual refine run {run_id}")
        if row["status"] not in _ACTIVE:
            raise LeaseLost(f"continual refine run {run_id} is {row['status']}")
        return row

    def assert_lease(self, run_id: str, token: str) -> None:
        now = _epoch()
        with self.store.transaction(immediate=True) as db:
            row = self._run(db, run_id, token)
            lease = db.execute("SELECT * FROM writer_leases WHERE scene=? AND camera=?",
                               (row["scene"], row["camera"])).fetchone()
            if (lease is None or lease["owner_kind"] != "continual"
                    or lease["owner_id"] != run_id or lease["token"] != token
                    or float(lease["expires_at"]) <= now):
                raise LeaseLost(f"writer lease lost for continual refine run {run_id}")

    def heartbeat(self, run_id: str, token: str, ttl_seconds: float = _LEASE_SECONDS) -> None:
        now = _epoch()
        with self.store.transaction(immediate=True) as db:
            row = self._run(db, run_id, token)
            changed = db.execute(
                """UPDATE writer_leases SET heartbeat_at=?,expires_at=?
                   WHERE scene=? AND camera=? AND owner_kind='continual'
                   AND owner_id=? AND token=?""",
                (now, now + ttl_seconds, row["scene"], row["camera"], run_id, token)).rowcount
            if changed != 1:
                raise LeaseLost(f"writer lease lost for continual refine run {run_id}")
            db.execute("""UPDATE continual_runs SET status='running',heartbeat_at=?,updated_at=?
                          WHERE run_id=? AND status IN ('queued','running')""",
                       (now, _utc_now(), run_id))

    @contextmanager
    def lease_heartbeat(self, run_id: str, token: str,
                        interval_seconds: float = 10.0) -> Iterator[None]:
        self.heartbeat(run_id, token)
        stop, lost = threading.Event(), []

        def pulse():
            beats = None
            missed = 0
            try:
                beats = ContinualJobs(self.model_root, self.scene, self.camera)
                while not stop.wait(interval_seconds):
                    try:
                        beats.heartbeat(run_id, token)
                        missed = 0
                    except sqlite3.OperationalError:
                        # Transient DB contention: the main thread holds a long
                        # IMMEDIATE transaction past busy_timeout. A missed beat is
                        # fine while the lease TTL holds; only give up after several
                        # in a row. A genuine LeaseLost still propagates.
                        missed += 1
                        if missed >= 5:
                            raise
            except BaseException as exc:
                lost.append(exc)
                stop.set()
            finally:
                if beats is not None:
                    beats.close()

        thread = threading.Thread(target=pulse, daemon=True,
                                  name=f"xcalib-refine-heartbeat-{run_id}")
        thread.start()
        try:
            yield
            if lost:
                raise lost[0]
            self.assert_lease(run_id, token)
        finally:
            stop.set()
            thread.join(timeout=max(1.0, min(interval_seconds, 10.0)))

    def record_progress(self, run_id: str, *args) -> None:
        if len(args) == 1:
            token, progress = None, args[0]
        elif len(args) == 2:
            token, progress = args
        else:
            raise TypeError("record_progress expects run_id, [token,] progress")
        now, stamp = _epoch(), _utc_now()
        with self.store.transaction(immediate=True) as db:
            row = db.execute("SELECT * FROM continual_runs WHERE run_id=?", (run_id,)).fetchone()
            # Token-less insertion is useful for service reconstruction tests only;
            # a real started row always owns a non-null token.
            if row is None and token is None:
                db.execute("""INSERT INTO continual_runs(
                    run_id,scene,camera,status,phase,current,total,message,best_epoch,
                    heartbeat_at,started_at,updated_at)
                    VALUES (?,?,?,'running',?,?,?,?,?,?,?,?)""",
                    (run_id, self.scene, self.camera, progress.phase, int(progress.current),
                     int(progress.total), str(progress.message), getattr(progress, "epoch", None),
                     now, stamp, stamp))
                return
            row = self._run(db, run_id, token)
            db.execute("""UPDATE continual_runs SET status='running',phase=?,current=?,total=?,
                          message=?,best_epoch=COALESCE(?,best_epoch),heartbeat_at=?,updated_at=?
                          WHERE run_id=?""",
                       (progress.phase, int(progress.current), int(progress.total),
                        str(progress.message), getattr(progress, "epoch", None), now, stamp, run_id))
            if token is not None:
                changed = db.execute("""UPDATE writer_leases SET heartbeat_at=?,expires_at=?
                    WHERE scene=? AND camera=? AND owner_kind='continual' AND owner_id=? AND token=?""",
                    (now, now + _LEASE_SECONDS, row["scene"], row["camera"], run_id, token)).rowcount
                if changed != 1:
                    raise LeaseLost(f"writer lease lost for continual refine run {run_id}")

    def finish(self, run_id: str, token: str, result: Any) -> None:
        outcome = str(result.outcome)
        if outcome not in {"accepted", "rejected", "work_remaining", "up_to_date"}:
            raise ValueError(f"invalid continual refine outcome: {outcome}")
        remaining = _mapping(getattr(result, "remaining", None)) or {}
        counts = {"remaining_new": int(remaining.get("new_samples", 0)),
                  "remaining_hard": int(remaining.get("hard_samples", 0)),
                  "missing_features": int(remaining.get("missing_features", 0)),
                  "validation_features": int(remaining.get("validation_features", 0))}
        parent = _mapping(getattr(result, "validation_parent", None))
        candidate = _mapping(getattr(result, "validation_candidate", None))
        validation = None if parent is None and candidate is None else {"parent": parent, "candidate": candidate}
        audit = _mapping(getattr(result, "audit", None))
        reasons = tuple(str(x) for x in getattr(result, "gate_reasons", ()))
        stamp = _utc_now()
        with self.store.transaction(immediate=True) as db:
            row = self._run(db, run_id, token)
            db.execute("""UPDATE continual_runs SET status=?,parent_revision=?,new_revision=?,best_epoch=?,
                       work_counts_json=?,validation_json=?,audit_json=?,gate_reasons_json=?,result_json=?,
                       message=?,finished_at=?,updated_at=? WHERE run_id=?""",
                (outcome, getattr(result, "parent_revision", None), getattr(result, "new_revision", None),
                 getattr(result, "best_epoch", None), _json(counts),
                 None if validation is None else _json(validation), None if audit is None else _json(audit),
                 _json(reasons), _json(result), outcome.replace("_", " "), stamp, stamp, run_id))
            db.execute("""DELETE FROM writer_leases WHERE scene=? AND camera=?
                          AND owner_kind='continual' AND owner_id=? AND token=?""",
                       (row["scene"], row["camera"], run_id, token))

    def fail(self, run_id: str, token: str, detail: str) -> None:
        stamp = _utc_now()
        with self.store.transaction(immediate=True) as db:
            row = db.execute("SELECT * FROM continual_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row["lease_token"] != token:
                raise LeaseLost(f"lease token mismatch for continual refine run {run_id}")
            if row["status"] in _TERMINAL:
                return
            db.execute("""UPDATE continual_runs SET status='failed',failure_detail=?,message=?,
                          finished_at=?,updated_at=? WHERE run_id=?""",
                       (detail, detail, stamp, stamp, run_id))
            db.execute("DELETE FROM writer_leases WHERE scene=? AND camera=? AND owner_id=? AND token=?",
                       (row["scene"], row["camera"], run_id, token))

    def cancel(self, run_id: str) -> bool:
        """Cooperatively stop an active run. Returns True if this call stopped it.

        Takes no lease token: the caller is the user/API, not the worker. It flips the
        run to a terminal status and drops its lease. The worker discovers that at its
        next `record_progress()`/`heartbeat()` — both go through `_run()`, which raises
        `LeaseLost` once the status leaves `_ACTIVE` — so it unwinds at a progress
        boundary of its own choosing instead of being killed mid-write. Its subsequent
        `fail()` then no-ops on the terminal row, so the cancel cannot be overwritten.

        We do NOT terminate the process. The refine writes a publication journal and
        switches `current.json`; killing it is precisely how you manufacture the
        divergence `remediate_divergence` exists to repair. Every phase emits progress
        frequently, so cooperative cancellation lands in well under a second.

        Refuses during `publishing` — that phase IS the pointer switch. It is short;
        retry a moment later.
        """
        stamp = _utc_now()
        with self.store.transaction(immediate=True) as db:
            row = db.execute("SELECT * FROM continual_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown continual refine run {run_id}")
            if row["status"] in _TERMINAL:
                return False  # idempotent: already stopped, keep the real outcome
            if row["phase"] == "publishing":
                raise CancelUnavailable(
                    "the run is publishing a revision; it cannot be interrupted",
                    phase="publishing",
                )
            db.execute(
                """UPDATE continual_runs SET status='interrupted',failure_detail=?,message=?,
                   finished_at=?,updated_at=? WHERE run_id=? AND status IN ('queued','running')""",
                (CANCELLED_DETAIL, CANCELLED_DETAIL, stamp, stamp, run_id))
            db.execute(
                """DELETE FROM writer_leases WHERE scene=? AND camera=?
                   AND owner_kind='continual' AND owner_id=?""",
                (row["scene"], row["camera"], run_id))
        return True

    def _acquire(self, kind: str, owner_id: str, ttl: float, *, token: str | None = None,
                 pid: int | None = None) -> str:
        token, now = token or secrets.token_urlsafe(24), _epoch()
        with self.store.transaction(immediate=True) as db:
            old = db.execute("SELECT * FROM writer_leases WHERE scene=? AND camera=?",
                             (self.scene, self.camera)).fetchone()
            if old is not None and float(old["expires_at"]) > now:
                if old["owner_kind"] == kind and old["owner_id"] == owner_id and old["token"] == token:
                    return token
                raise WriterLeaseConflict(f"{old['owner_kind']}:{old['owner_id']}")
            db.execute("""INSERT INTO writer_leases(scene,camera,owner_kind,owner_id,token,owner_pid,
                       heartbeat_at,expires_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(scene,camera)
                       DO UPDATE SET owner_kind=excluded.owner_kind,owner_id=excluded.owner_id,
                       token=excluded.token,owner_pid=excluded.owner_pid,
                       heartbeat_at=excluded.heartbeat_at,expires_at=excluded.expires_at""",
                       (self.scene, self.camera, kind, owner_id, token, pid, now, now + ttl))
        return token

    @contextmanager
    def writer_lease(self, kind: str, owner_id: str, ttl_seconds: float = 60.0) -> Iterator[str]:
        token = self._acquire(kind, owner_id, ttl_seconds, pid=os.getpid())
        stop, lost = threading.Event(), []
        interval = max(1.0, min(10.0, ttl_seconds / 3))

        def pulse():
            beats = None
            missed = 0
            try:
                beats = ContinualJobs(self.model_root, self.scene, self.camera)
                while not stop.wait(interval):
                    now = _epoch()
                    try:
                        with beats.store.transaction(immediate=True) as db:
                            changed = db.execute("""UPDATE writer_leases SET heartbeat_at=?,expires_at=?
                                WHERE scene=? AND camera=? AND owner_kind=? AND owner_id=? AND token=?""",
                                (now, now + ttl_seconds, self.scene, self.camera, kind, owner_id, token)).rowcount
                            if changed != 1:
                                raise LeaseLost(f"writer lease lost for {kind}:{owner_id}")
                        missed = 0
                    except sqlite3.OperationalError:
                        # Transient DB contention only; a genuine LeaseLost (rowcount
                        # != 1) is a different exception and still propagates. Tolerate
                        # a few missed renewals while the lease TTL holds.
                        missed += 1
                        if missed >= 5:
                            raise
            except BaseException as exc:
                lost.append(exc)
                stop.set()
            finally:
                if beats is not None:
                    beats.close()

        thread = threading.Thread(target=pulse, daemon=True, name=f"xcalib-writer-{kind}-{owner_id}")
        thread.start()
        try:
            yield token
            if lost:
                raise lost[0]
        finally:
            stop.set()
            thread.join(timeout=max(1.0, min(interval, 10.0)))
            with self.store.transaction(immediate=True) as db:
                db.execute("""DELETE FROM writer_leases WHERE scene=? AND camera=?
                              AND owner_kind=? AND owner_id=? AND token=?""",
                           (self.scene, self.camera, kind, owner_id, token))


def _data_root(root: Path, scene: str) -> Path:
    return root.parents[3] if len(root.parents) > 3 and root.parents[2].name == scene else root.parent


def start_refine_job(model_root: str | Path, scene: str, camera: str, python: str,
                     repo_root: str | Path, *,
                     worker: str | Path | None = None) -> StartResult:
    """Start one refine run in a separate process, tracked as a durable job.

    The process is ``python <worker> --data-root ... --scene ... --camera ... --run-id ...
    --lease-token ...`` run in ``repo_root``. XCalib ships no worker: yours opens
    :meth:`ContinualJobs.for_scene`, calls ``run_continual_refine`` under the lease it is
    handed (heartbeat, progress, ``finish`` or ``fail``), and exits. A relative ``worker`` is
    resolved against ``repo_root``; ``None`` means ``tools/xcalib/refine_continual.py`` under
    ``repo_root``. Raises ``FileNotFoundError``, before anything is queued, when the script is
    not a file. A run that is already active is returned instead of starting another."""
    repo = Path(repo_root).resolve()
    script = Path(worker) if worker is not None else repo / "tools" / "xcalib" / "refine_continual.py"
    if not (script if script.is_absolute() else repo / script).is_file():
        raise FileNotFoundError(f"refine worker script not found: {script}")
    root = Path(model_root).resolve()
    jobs = ContinualJobs(root, scene, camera)
    try:
        now = _epoch()
        row = jobs.store.connection.execute(
            """SELECT * FROM continual_runs WHERE scene=? AND camera=?
               AND status IN ('queued','running') ORDER BY started_at DESC LIMIT 1""",
            (scene, camera)).fetchone()
        if row is not None and row["heartbeat_at"] is not None \
                and now - float(row["heartbeat_at"]) <= _LEASE_SECONDS:
            return StartResult(str(row["run_id"]), "existing")
        recover_interrupted_runs(root, scene=scene, camera=camera)
        row = jobs.store.connection.execute(
            """SELECT run_id FROM continual_runs WHERE scene=? AND camera=?
               AND status IN ('queued','running') ORDER BY started_at DESC LIMIT 1""",
            (scene, camera)).fetchone()
        if row is not None:
            return StartResult(str(row["run_id"]), "existing")

        run_id, token = f"run-{uuid.uuid4().hex}", secrets.token_urlsafe(24)
        try:
            jobs._acquire("continual", run_id, _LEASE_SECONDS, token=token)
        except WriterLeaseConflict as exc:
            return StartResult(None, "locked", exc.owner)
        stamp = _utc_now()
        watermark = jobs.store.connection.execute(
            """SELECT MAX(label_revision) revision,MAX(updated_at) applied_at FROM corrections
               WHERE scene=? AND camera=? AND superseded_by IS NULL""",
            (scene, camera)).fetchone()
        with jobs.store.transaction(immediate=True) as db:
            db.execute("""INSERT INTO continual_runs(
                run_id,scene,camera,status,corpus_watermark,work_counts_json,phase,current,total,
                message,lease_token,heartbeat_at,started_at,updated_at)
                VALUES (?,?,?,'queued',?,?,'indexing',0,0,'queued',?,?,?,?)""",
                (run_id, scene, camera,
                 _json({"label_revision": int(watermark["revision"] or 0),
                        "applied_at": watermark["applied_at"]}),
                 _json(jobs._counts()), token, now, stamp, stamp))
        cmd = [python, str(script),
               "--data-root", str(_data_root(root, scene)), "--scene", scene,
               "--camera", camera, "--run-id", run_id, "--lease-token", token]
        env = os.environ.copy()
        env.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"})
        try:
            process = subprocess.Popen(cmd, cwd=str(repo), env=env)
        except BaseException as exc:
            jobs.fail(run_id, token, f"{type(exc).__name__}: {exc}")
            raise
        pid = process.pid if isinstance(process.pid, int) else None
        with jobs.store.transaction(immediate=True) as db:
            db.execute("UPDATE continual_runs SET owner_pid=?,updated_at=? WHERE run_id=?",
                       (pid, _utc_now(), run_id))
            db.execute("""UPDATE writer_leases SET owner_pid=? WHERE scene=? AND camera=?
                          AND owner_kind='continual' AND owner_id=? AND token=?""",
                       (pid, scene, camera, run_id, token))
        return StartResult(run_id, "started")
    finally:
        jobs.close()


def _publication_finalizer(db, operation) -> None:
    payload = operation.commit_payload if isinstance(operation.commit_payload, dict) else {}
    stamp = _utc_now()
    # A committed publication means the run's revision is live, so the run succeeded
    # regardless of a transient 'failed'/'interrupted' status a dying heartbeat or a
    # crash-during-recovery may have stamped. Reconcile any non-accepted run.
    db.execute("""UPDATE continual_runs SET status='accepted',new_revision=?,
                  validation_json=COALESCE(?,validation_json),audit_json=COALESCE(?,audit_json),
                  message='accepted',finished_at=COALESCE(finished_at,?),updated_at=?
                  WHERE run_id=? AND status != 'accepted'""",
               (operation.new_pointer.get("revision"),
                None if payload.get("validation") is None else _json(payload["validation"]),
                None if payload.get("audit") is None else _json(payload["audit"]),
                stamp, stamp, operation.run_id))


def _reconcile_committed_publications(store) -> list[str]:
    """A publication can commit normally and then a crash (or a dying heartbeat) can
    leave the run row 'failed'/'interrupted' before jobs.finish -- recover_publications
    won't touch an already-committed op, so its run keeps the wrong status while the
    model is live. Reconcile every committed publication's run to 'accepted'."""
    reconciled: list[str] = []
    committed = store.connection.execute(
        """SELECT p.run_id, p.new_pointer_json, p.commit_payload_json
           FROM publication_operations p JOIN continual_runs r ON r.run_id = p.run_id
           WHERE p.state='committed' AND r.status != 'accepted'"""
    ).fetchall()
    for op in committed:
        revision = json.loads(op["new_pointer_json"]).get("revision")
        payload = json.loads(op["commit_payload_json"] or "{}")
        stamp = _utc_now()
        with store.transaction(immediate=True) as db:
            changed = db.execute(
                """UPDATE continual_runs SET status='accepted',new_revision=?,
                   validation_json=COALESCE(?,validation_json),audit_json=COALESCE(?,audit_json),
                   message='accepted',finished_at=COALESCE(finished_at,?),updated_at=?
                   WHERE run_id=? AND status != 'accepted'""",
                (revision,
                 None if payload.get("validation") is None else _json(payload["validation"]),
                 None if payload.get("audit") is None else _json(payload["audit"]),
                 stamp, stamp, op["run_id"])).rowcount
        if changed:
            reconciled.append(f"{op['run_id']}:reconciled-accepted")
    return reconciled


def recover_interrupted_runs(model_root: str | Path, live_pids: set[int] | None = None,
                             *, scene: str | None = None,
                             camera: str | None = None) -> list[str]:
    from .train import finalize_continual_publication
    from .publication import recover_publications
    jobs = ContinualJobs(model_root, scene, camera)
    recovered: list[str] = []
    try:
        def finalize(db, operation):
            finalize_continual_publication(db, operation)
            _publication_finalizer(db, operation)

        recovered.extend(recover_publications(jobs.store.connection, finalize=finalize))
        recovered.extend(_reconcile_committed_publications(jobs.store))
        rows = jobs.store.connection.execute(
            """SELECT * FROM continual_runs WHERE status IN ('queued','running')
               AND (heartbeat_at IS NULL OR heartbeat_at < ?)""",
            (_epoch() - _LEASE_SECONDS,)).fetchall()
        for row in rows:
            pid = row["owner_pid"]
            live = int(pid) in live_pids if live_pids is not None and pid is not None else _pid_live(pid)
            if live:
                continue
            stamp = _utc_now()
            with jobs.store.transaction(immediate=True) as db:
                changed = db.execute("""UPDATE continual_runs SET status='interrupted',
                    message='worker heartbeat expired',failure_detail='worker heartbeat expired',
                    finished_at=?,updated_at=? WHERE run_id=? AND status IN ('queued','running')""",
                    (stamp, stamp, row["run_id"])).rowcount
                if changed:
                    db.execute("""DELETE FROM writer_leases WHERE scene=? AND camera=?
                                  AND owner_kind='continual' AND owner_id=?""",
                               (row["scene"], row["camera"], row["run_id"]))
                    recovered.append(f"{row['run_id']}:interrupted")
        return recovered
    finally:
        jobs.close()


def get_refine_job(model_root: str | Path, run_id: str) -> DurableJobState | None:
    with ContinualJobs(model_root) as jobs:
        return jobs.get(run_id)


def get_refine_workload(model_root: str | Path) -> RefineWorkload:
    with ContinualJobs(model_root) as jobs:
        return jobs.workload()


def cancel_refine_job(model_root: str | Path, run_id: str) -> bool:
    """Cooperatively stop `run_id`. True if this call stopped it, False if it had
    already finished. Raises CancelUnavailable while the run is publishing."""
    with ContinualJobs(model_root) as jobs:
        return jobs.cancel(run_id)


__all__ = ["ContinualJobs", "DurableJobState", "LeaseLost", "RefineWorkload",
           "RunPhase", "RunStatus", "StartResult", "WriterLeaseConflict",
           "get_refine_job", "get_refine_workload", "recover_interrupted_runs",
           "start_refine_job"]
