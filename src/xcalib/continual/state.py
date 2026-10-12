"""Durable, revision-relative state for bounded continual XCalib refine runs.

This module deliberately stores no human labels.  Those remain authoritative in
``corrections``; this is only the cache, split and curriculum bookkeeping that
lets subsequent finite runs reuse frozen-base work safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
from typing import Literal, Mapping, Sequence

import numpy as np

from .store import XCalibStore


DatasetSplit = Literal["train", "validation", "audit"]
DifficultyState = Literal["new", "hard", "mastered"]
FeatureState = Literal["missing", "ready", "stale", "error"]


@dataclass(frozen=True)
class ScoreDecision:
    mastered: bool
    positive_margin: float
    max_negative_logit: float


@dataclass(frozen=True)
class CachedFrame:
    frame: str
    input_fingerprint: str
    base_fingerprint: str
    rect_keys: tuple[str, ...]
    lidar_keys: tuple[str, ...]
    image_embeddings: np.ndarray
    lidar_embeddings: np.ndarray


@dataclass(frozen=True)
class FeatureWork:
    frame: str
    input_fingerprint: str
    dataset_split: DatasetSplit


@dataclass(frozen=True)
class SupervisionSample:
    sample_key: str
    scene: str
    camera: str
    frame: str
    frame_ordinal: int
    camera_track: str | None
    group_key: str
    dataset_split: DatasetSplit
    difficulty_state: DifficultyState
    feature_state: FeatureState
    feature_input_fingerprint: str
    label_revision: int
    label_input_fingerprint: str
    official_revision: str | None
    correct_streak: int
    positive_margin: float | None
    max_negative_logit: float | None
    served_count: int
    last_served_run: str | None
    last_served_at: str | None
    last_corrected_at: str


_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS frame_features(
  frame TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL,
  base_fingerprint TEXT NOT NULL,
  rect_keys_json TEXT NOT NULL,
  lidar_keys_json TEXT NOT NULL,
  image_dtype TEXT NOT NULL,
  image_shape_json TEXT NOT NULL,
  image_blob BLOB NOT NULL,
  lidar_dtype TEXT NOT NULL,
  lidar_shape_json TEXT NOT NULL,
  lidar_blob BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(frame,input_fingerprint,base_fingerprint)
);
CREATE INDEX IF NOT EXISTS frame_features_lookup_idx
  ON frame_features(frame,input_fingerprint,base_fingerprint);

CREATE TABLE IF NOT EXISTS sample_state(
  sample_key TEXT PRIMARY KEY,
  scene TEXT NOT NULL,
  camera TEXT NOT NULL,
  frame TEXT NOT NULL,
  frame_ordinal INTEGER NOT NULL,
  camera_track TEXT,
  group_key TEXT NOT NULL,
  dataset_split TEXT NOT NULL CHECK(dataset_split IN ('train','validation','audit')),
  difficulty_state TEXT NOT NULL CHECK(difficulty_state IN ('new','hard','mastered')),
  feature_state TEXT NOT NULL CHECK(feature_state IN ('missing','ready','stale','error')),
  feature_input_fingerprint TEXT NOT NULL DEFAULT '',
  label_revision INTEGER NOT NULL DEFAULT 0,
  label_input_fingerprint TEXT NOT NULL DEFAULT '',
  official_revision TEXT,
  correct_streak INTEGER NOT NULL DEFAULT 0,
  positive_margin REAL,
  max_negative_logit REAL,
  served_count INTEGER NOT NULL DEFAULT 0,
  last_served_run TEXT,
  last_served_at TEXT,
  last_corrected_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sample_state_feature_idx
  ON sample_state(feature_state,dataset_split,frame,feature_input_fingerprint);
CREATE INDEX IF NOT EXISTS sample_state_hard_idx
  ON sample_state(dataset_split,difficulty_state,feature_state,sample_key);

CREATE TABLE IF NOT EXISTS continual_cursors(
  cursor_kind TEXT PRIMARY KEY,
  cursor_sample_key TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS serving_events(
  run_id TEXT NOT NULL,
  sample_key TEXT NOT NULL REFERENCES sample_state(sample_key) ON DELETE CASCADE,
  page_kind TEXT NOT NULL,
  served_at TEXT NOT NULL,
  PRIMARY KEY(run_id,sample_key,page_kind)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_split(scene: str, camera: str, group_key: str) -> DatasetSplit:
    bucket = int.from_bytes(
        hashlib.sha256(f"{scene}/{camera}/{group_key}".encode()).digest()[:8],
        "big",
    ) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "audit"


def _sample(row: sqlite3.Row) -> SupervisionSample:
    return SupervisionSample(
        sample_key=str(row["sample_key"]),
        scene=str(row["scene"]),
        camera=str(row["camera"]),
        frame=str(row["frame"]),
        frame_ordinal=int(row["frame_ordinal"]),
        camera_track=row["camera_track"],
        group_key=str(row["group_key"]),
        dataset_split=row["dataset_split"],
        difficulty_state=row["difficulty_state"],
        feature_state=row["feature_state"],
        feature_input_fingerprint=str(row["feature_input_fingerprint"]),
        label_revision=int(row["label_revision"]),
        label_input_fingerprint=str(row["label_input_fingerprint"]),
        official_revision=row["official_revision"],
        correct_streak=int(row["correct_streak"]),
        positive_margin=row["positive_margin"],
        max_negative_logit=row["max_negative_logit"],
        served_count=int(row["served_count"]),
        last_served_run=row["last_served_run"],
        last_served_at=row["last_served_at"],
        last_corrected_at=str(row["last_corrected_at"]),
    )


def _encode_array(array: np.ndarray, keys: Sequence[str], label: str) -> tuple[str, str, bytes]:
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float32):
        raise ValueError(f"{label} embeddings must be float32")
    if array.ndim != 2:
        raise ValueError(f"{label} embeddings must be a rank-2 array")
    if array.shape[0] != len(keys):
        raise ValueError(f"{label} key count does not match embedding rows")
    contiguous = np.ascontiguousarray(array)
    return str(contiguous.dtype), json.dumps(list(contiguous.shape)), contiguous.tobytes()


def _decode_array(dtype_value: str, shape_value: str, blob: bytes, label: str) -> np.ndarray:
    try:
        dtype = np.dtype(dtype_value)
        shape = json.loads(shape_value)
        if dtype != np.dtype(np.float32):
            raise ValueError("dtype")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("shape")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValueError("shape")
        size = math.prod(shape)
        if len(blob) != size * dtype.itemsize:
            raise ValueError("length")
        return np.frombuffer(blob, dtype=dtype, count=size).copy().reshape(tuple(shape))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cached {label} embeddings") from exc


class ContinualState:
    """Schema-v2 state operations backed by the ``refine.sqlite`` store."""

    def __init__(self, store: XCalibStore):
        self.store = store
        self._migrate()

    def _migrate(self) -> None:
        with self.store.transaction(immediate=True) as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            version = None if row is None else row["value"]
            if version not in {"1", "2", "3"}:
                raise RuntimeError(f"unsupported XCalib refine schema version: {version}")
            db.executescript(_STATE_SCHEMA)
            if version == "1":
                db.execute(
                    "UPDATE metadata SET value='2' WHERE key='schema_version'"
                )

    def get_sample(self, sample_key: str) -> SupervisionSample | None:
        row = self.store.connection.execute(
            "SELECT * FROM sample_state WHERE sample_key=?", (sample_key,)
        ).fetchone()
        return None if row is None else _sample(row)

    def ensure_sample(
        self,
        sample_key: str,
        scene: str,
        camera: str,
        frame: str,
        frame_ordinal: int,
        camera_track: str | None,
        *,
        label_revision: int | None = None,
        label_input_fingerprint: str | None = None,
        feature_input_fingerprint: str | None = None,
    ) -> SupervisionSample:
        group_key = (
            f"track:{camera_track}" if camera_track else f"time:{frame_ordinal // 100}"
        )
        dataset_split = stable_split(scene, camera, group_key)
        now = _now()
        with self.store.transaction(immediate=True) as db:
            existing = db.execute(
                "SELECT * FROM sample_state WHERE sample_key=?", (sample_key,)
            ).fetchone()
            if existing is None:
                db.execute(
                    """INSERT INTO sample_state(
                           sample_key,scene,camera,frame,frame_ordinal,camera_track,
                           group_key,dataset_split,difficulty_state,feature_state,
                           feature_input_fingerprint,label_revision,label_input_fingerprint,
                           last_corrected_at,created_at,updated_at
                       ) VALUES (?,?,?,?,?,?,?,?,'new','missing',?,?,?,?,?,?)""",
                    (
                        sample_key, scene, camera, frame, frame_ordinal, camera_track,
                        group_key, dataset_split, feature_input_fingerprint or "",
                        0 if label_revision is None else label_revision,
                        label_input_fingerprint or "", now, now, now,
                    ),
                )
            else:
                updates: list[str] = ["updated_at=?"]
                params: list[object] = [now]
                label_changed = (
                    (label_revision is not None and label_revision != existing["label_revision"])
                    or (
                        label_input_fingerprint is not None
                        and label_input_fingerprint != existing["label_input_fingerprint"]
                    )
                )
                if label_revision is not None:
                    updates.append("label_revision=?")
                    params.append(label_revision)
                if label_input_fingerprint is not None:
                    updates.append("label_input_fingerprint=?")
                    params.append(label_input_fingerprint)
                if label_changed:
                    updates.extend((
                        "difficulty_state='new'", "official_revision=NULL",
                        "correct_streak=0", "positive_margin=NULL",
                        "max_negative_logit=NULL", "last_corrected_at=?",
                    ))
                    params.append(now)
                if (
                    feature_input_fingerprint is not None
                    and feature_input_fingerprint != existing["feature_input_fingerprint"]
                ):
                    updates.extend(("feature_input_fingerprint=?", "feature_state='stale'"))
                    params.append(feature_input_fingerprint)
                params.append(sample_key)
                db.execute(
                    f"UPDATE sample_state SET {','.join(updates)} WHERE sample_key=?", params
                )
        sample = self.get_sample(sample_key)
        assert sample is not None
        return sample

    def store_feature(self, cached: CachedFrame) -> None:
        image_dtype, image_shape, image_blob = _encode_array(
            cached.image_embeddings, cached.rect_keys, "image"
        )
        lidar_dtype, lidar_shape, lidar_blob = _encode_array(
            cached.lidar_embeddings, cached.lidar_keys, "lidar"
        )
        try:
            rect_keys_json = json.dumps(list(cached.rect_keys))
            lidar_keys_json = json.dumps(list(cached.lidar_keys))
        except (TypeError, ValueError) as exc:
            raise ValueError("cached feature keys must be JSON serializable") from exc
        now = _now()
        with self.store.transaction(immediate=True) as db:
            db.execute(
                """INSERT INTO frame_features(
                       frame,input_fingerprint,base_fingerprint,rect_keys_json,lidar_keys_json,
                       image_dtype,image_shape_json,image_blob,lidar_dtype,lidar_shape_json,
                       lidar_blob,created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(frame,input_fingerprint,base_fingerprint) DO UPDATE SET
                     rect_keys_json=excluded.rect_keys_json,lidar_keys_json=excluded.lidar_keys_json,
                     image_dtype=excluded.image_dtype,image_shape_json=excluded.image_shape_json,
                     image_blob=excluded.image_blob,lidar_dtype=excluded.lidar_dtype,
                     lidar_shape_json=excluded.lidar_shape_json,lidar_blob=excluded.lidar_blob,
                     created_at=excluded.created_at""",
                (
                    cached.frame, cached.input_fingerprint, cached.base_fingerprint,
                    rect_keys_json, lidar_keys_json, image_dtype, image_shape,
                    sqlite3.Binary(image_blob), lidar_dtype, lidar_shape,
                    sqlite3.Binary(lidar_blob), now,
                ),
            )
            db.execute(
                """UPDATE sample_state SET feature_state='ready',updated_at=?
                   WHERE frame=? AND feature_input_fingerprint=?""",
                (now, cached.frame, cached.input_fingerprint),
            )

    def get_feature(
        self, frame: str, input_fingerprint: str, base_fingerprint: str
    ) -> CachedFrame | None:
        row = self.store.connection.execute(
            """SELECT * FROM frame_features
               WHERE frame=? AND input_fingerprint=? AND base_fingerprint=?""",
            (frame, input_fingerprint, base_fingerprint),
        ).fetchone()
        if row is None:
            return None
        try:
            rect_keys = tuple(json.loads(row["rect_keys_json"]))
            lidar_keys = tuple(json.loads(row["lidar_keys_json"]))
            if not all(isinstance(key, str) for key in rect_keys + lidar_keys):
                raise ValueError("keys")
            image = _decode_array(
                row["image_dtype"], row["image_shape_json"], row["image_blob"], "image"
            )
            lidar = _decode_array(
                row["lidar_dtype"], row["lidar_shape_json"], row["lidar_blob"], "lidar"
            )
            if image.shape[0] != len(rect_keys) or lidar.shape[0] != len(lidar_keys):
                raise ValueError("key count")
        except (TypeError, ValueError, json.JSONDecodeError):
            with self.store.transaction(immediate=True) as db:
                db.execute(
                    """DELETE FROM frame_features
                       WHERE frame=? AND input_fingerprint=? AND base_fingerprint=?""",
                    (frame, input_fingerprint, base_fingerprint),
                )
                db.execute(
                    """UPDATE sample_state SET feature_state='stale',updated_at=?
                       WHERE frame=? AND feature_input_fingerprint=?""",
                    (_now(), frame, input_fingerprint),
                )
            return None
        return CachedFrame(
            frame=frame,
            input_fingerprint=input_fingerprint,
            base_fingerprint=base_fingerprint,
            rect_keys=rect_keys,
            lidar_keys=lidar_keys,
            image_embeddings=image,
            lidar_embeddings=lidar,
        )

    def feature_work_page(self, base_fingerprint: str, limit: int) -> list[FeatureWork]:
        if limit <= 0:
            return []
        with self.store.transaction(immediate=True) as db:
            rows = db.execute(
                """SELECT frame,feature_input_fingerprint,
                          MIN(CASE dataset_split WHEN 'validation' THEN 0
                              WHEN 'train' THEN 1 ELSE 2 END) AS split_priority
                   FROM sample_state
                   GROUP BY frame,feature_input_fingerprint
                   ORDER BY split_priority,frame,feature_input_fingerprint"""
            ).fetchall()
            work: list[FeatureWork] = []
            now = _now()
            for row in rows:
                frame = str(row["frame"])
                fingerprint = str(row["feature_input_fingerprint"])
                cache_row = db.execute(
                    """SELECT * FROM frame_features
                       WHERE frame=? AND input_fingerprint=? AND base_fingerprint=?""",
                    (frame, fingerprint, base_fingerprint),
                ).fetchone()
                cached = False
                if cache_row is not None:
                    try:
                        rect_keys = tuple(json.loads(cache_row["rect_keys_json"]))
                        lidar_keys = tuple(json.loads(cache_row["lidar_keys_json"]))
                        if not all(isinstance(key, str) for key in rect_keys + lidar_keys):
                            raise ValueError("keys")
                        image = _decode_array(
                            cache_row["image_dtype"], cache_row["image_shape_json"],
                            cache_row["image_blob"], "image",
                        )
                        lidar = _decode_array(
                            cache_row["lidar_dtype"], cache_row["lidar_shape_json"],
                            cache_row["lidar_blob"], "lidar",
                        )
                        cached = image.shape[0] == len(rect_keys) and lidar.shape[0] == len(lidar_keys)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        db.execute(
                            """DELETE FROM frame_features
                               WHERE frame=? AND input_fingerprint=? AND base_fingerprint=?""",
                            (frame, fingerprint, base_fingerprint),
                        )
                if cached:
                    db.execute(
                        """UPDATE sample_state SET feature_state='ready',updated_at=?
                           WHERE frame=? AND feature_input_fingerprint=?
                             AND feature_state != 'ready'""",
                        (now, frame, fingerprint),
                    )
                    continue
                db.execute(
                    """UPDATE sample_state SET feature_state='stale',updated_at=?
                       WHERE frame=? AND feature_input_fingerprint=?
                         AND feature_state='ready'""",
                    (now, frame, fingerprint),
                )
                split: DatasetSplit = (
                    "validation" if row["split_priority"] == 0
                    else "train" if row["split_priority"] == 1 else "audit"
                )
                work.append(FeatureWork(frame, fingerprint, split))
                if len(work) >= limit:
                    break
            return work

    def classification_page(self, limit: int = 512) -> list[SupervisionSample]:
        if limit <= 0:
            return []
        rows = self.store.connection.execute(
            """SELECT * FROM sample_state
               WHERE difficulty_state='new' AND feature_state='ready'
               ORDER BY CASE dataset_split WHEN 'validation' THEN 0
                            WHEN 'train' THEN 1 ELSE 2 END,sample_key
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_sample(row) for row in rows]

    def apply_parent_scores(
        self,
        official_revision: str,
        scores: Mapping[str, ScoreDecision],
        *,
        expected_labels: Mapping[str, tuple[int, str]] | None = None,
    ) -> None:
        now = _now()
        with self.store.transaction(immediate=True) as db:
            for sample_key, decision in scores.items():
                expected = None if expected_labels is None else expected_labels.get(sample_key)
                if expected is None:
                    previous = db.execute(
                        "SELECT correct_streak FROM sample_state WHERE sample_key=?",
                        (sample_key,),
                    ).fetchone()
                else:
                    previous = db.execute(
                        """SELECT state.correct_streak FROM sample_state AS state
                           JOIN corrections AS correction
                             ON correction.sample_key=state.sample_key
                           WHERE state.sample_key=?
                             AND state.label_revision=?
                             AND state.label_input_fingerprint=?
                             AND correction.label_revision=?
                             AND correction.input_fingerprint=?
                             AND correction.superseded_by IS NULL""",
                        (
                            sample_key,
                            expected[0],
                            expected[1],
                            expected[0],
                            expected[1],
                        ),
                    ).fetchone()
                if previous is None:
                    continue
                mastered = bool(decision.mastered)
                db.execute(
                    """UPDATE sample_state SET difficulty_state=?,official_revision=?,
                       correct_streak=?,positive_margin=?,max_negative_logit=?,updated_at=?
                       WHERE sample_key=?""",
                    (
                        "mastered" if mastered else "hard", official_revision,
                        int(previous["correct_streak"]) + 1 if mastered else 0,
                        decision.positive_margin, decision.max_negative_logit, now, sample_key,
                    ),
                )

    def hard_page(self, limit: int) -> list[SupervisionSample]:
        if limit <= 0:
            return []
        rows = self.store.connection.execute(
            """SELECT * FROM sample_state
               WHERE dataset_split='train' AND difficulty_state='hard' AND feature_state='ready'
               ORDER BY CASE WHEN last_served_run IS NULL THEN 0 ELSE 1 END,
                        last_corrected_at DESC, served_count ASC, last_served_at ASC,
                        sample_key ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_sample(row) for row in rows]

    def _mastered_page(self, cursor_kind: str, limit: int) -> list[SupervisionSample]:
        if limit <= 0:
            return []
        cursor = self.store.connection.execute(
            "SELECT cursor_sample_key FROM continual_cursors WHERE cursor_kind=?", (cursor_kind,)
        ).fetchone()
        after = None if cursor is None else cursor["cursor_sample_key"]

        def read(comparator: str, value: str | None, remaining: int):
            if value is None:
                return self.store.connection.execute(
                    """SELECT * FROM sample_state
                       WHERE dataset_split='train' AND difficulty_state='mastered'
                         AND feature_state='ready'
                       ORDER BY sample_key LIMIT ?""",
                    (remaining,),
                ).fetchall()
            return self.store.connection.execute(
                f"""SELECT * FROM sample_state
                    WHERE dataset_split='train' AND difficulty_state='mastered'
                      AND feature_state='ready' AND sample_key {comparator} ?
                    ORDER BY sample_key LIMIT ?""",
                (value, remaining),
            ).fetchall()

        rows = read(">", after, limit)
        if after is not None and len(rows) < limit:
            rows += read("<=", after, limit - len(rows))
        return [_sample(row) for row in rows]

    def mastered_probe_page(self, limit: int) -> list[SupervisionSample]:
        return self._mastered_page("mastered_probe", limit)

    def mastered_anchor_page(self, limit: int) -> list[SupervisionSample]:
        return self._mastered_page("mastered_anchor", limit)

    def mark_served(
        self, run_id: str, sample_keys: Sequence[str], *, page_kind: str = "hard"
    ) -> None:
        if not sample_keys:
            return
        now = _now()
        fresh: list[str] = []
        with self.store.transaction(immediate=True) as db:
            for sample_key in sample_keys:
                inserted = db.execute(
                    """INSERT OR IGNORE INTO serving_events(run_id,sample_key,page_kind,served_at)
                       VALUES (?,?,?,?)""",
                    (run_id, sample_key, page_kind, now),
                ).rowcount
                if inserted:
                    fresh.append(sample_key)
                    # Rotating mastered probes and replay anchors own only a
                    # cursor.  Updating their shared hard-page fairness fields
                    # would make one page's acknowledgement alter the other's
                    # snapshot before that other page is acknowledged.
                    if page_kind not in {"mastered_probe", "mastered_anchor"}:
                        db.execute(
                            """UPDATE sample_state SET served_count=served_count+1,
                               last_served_run=?,last_served_at=?,updated_at=? WHERE sample_key=?""",
                            (run_id, now, now, sample_key),
                        )
            if fresh and page_kind in {"mastered_probe", "mastered_anchor"}:
                db.execute(
                    """INSERT INTO continual_cursors(cursor_kind,cursor_sample_key,updated_at)
                       VALUES (?,?,?)
                       ON CONFLICT(cursor_kind) DO UPDATE SET
                         cursor_sample_key=excluded.cursor_sample_key,updated_at=excluded.updated_at""",
                    (page_kind, fresh[-1], now),
                )

    def _seed(self, sample_keys: Sequence[str], difficulty: DifficultyState) -> None:
        now = _now()
        with self.store.transaction(immediate=True) as db:
            for ordinal, sample_key in enumerate(sample_keys):
                db.execute(
                    """INSERT INTO sample_state(
                           sample_key,scene,camera,frame,frame_ordinal,camera_track,
                           group_key,dataset_split,difficulty_state,feature_state,
                           feature_input_fingerprint,label_revision,label_input_fingerprint,
                           last_corrected_at,created_at,updated_at
                       ) VALUES (?, 'seed', 'seed', ?, ?, NULL, ?, 'train', ?, 'ready',
                                 'seed', 0, '', ?, ?, ?)
                       ON CONFLICT(sample_key) DO UPDATE SET
                         dataset_split='train',difficulty_state=excluded.difficulty_state,
                         feature_state='ready',updated_at=excluded.updated_at""",
                    (sample_key, sample_key, ordinal, f"seed:{sample_key}", difficulty, now, now, now),
                )

    def seed_hard(self, sample_keys: Sequence[str]) -> None:
        self._seed(sample_keys, "hard")

    def seed_mastered(self, sample_keys: Sequence[str]) -> None:
        self._seed(sample_keys, "mastered")


__all__ = [
    "CachedFrame", "ContinualState", "DatasetSplit", "DifficultyState",
    "FeatureState", "FeatureWork", "ScoreDecision", "SupervisionSample",
    "stable_split",
]
