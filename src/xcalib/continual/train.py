"""Bounded, warm-started continual refinement for the XCalib adapter.

The service owns corpus scheduling and publication, while every tensor/model
operation goes through the ``ml_api`` object (by default
:mod:`xcalib.continual.refine_model`).  One invocation is deliberately
finite: callers that want to consume more corpus work start another durable
run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from . import refine_model as _default_ml
from .refine_model import (
    XCalibBaseEmbeddings,
    XCalibFrameInput,
    XCalibMetrics,
    XCalibRevisionRef,
    XCalibTrainConfig,
    XCalibTrainingBatch,
)
from . import locations
from .state import (
    CachedFrame,
    ContinualState,
    FeatureWork,
    ScoreDecision,
    SupervisionSample,
)
from .publication import (
    PublicationOperation,
    migrate_publications,
    publish_revision,
)
from .store import (
    CorrectionRecord,
    XCalibStore,
    open_store,
)


RefinePhase = Literal[
    "indexing",
    "caching",
    "classifying",
    "training",
    "validating",
    "auditing",
    "publishing",
]
RefineOutcome = Literal["accepted", "rejected", "work_remaining", "up_to_date"]


class ContinualRefineUnavailable(RuntimeError):
    """The requested model/corpus cannot run continual refinement."""


class _FrameInputFingerprintMismatch(ContinualRefineUnavailable):
    """The current frame content no longer matches its reviewed snapshot."""


@dataclass(frozen=True)
class RefineConfig:
    embed_limit: int = 500
    hard_limit: int = 4096
    mastered_probe_limit: int = 512
    mastered_anchor_limit: int = 256
    min_validation_samples: int = 20
    min_validation_groups: int = 2
    minimum_primary_delta: float = 0.005
    max_top1_drop: float = 0.01
    max_hard_violation_increase: float = 0.01
    max_no_match_fpr_increase: float = 0.01


@dataclass(frozen=True)
class RefineProgress:
    phase: RefinePhase
    current: int
    total: int
    message: str
    epoch: int | None = None
    max_epochs: int | None = None


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RemainingWork:
    missing_features: int = 0
    validation_features: int = 0
    new_samples: int = 0
    hard_samples: int = 0
    # Train-split-only counts: only these are trainable. Validation/audit-split
    # 'new'/'hard' samples never enter the train hard page, so they must NOT keep
    # the run reporting work_remaining (that livelocks any "run until up_to_date"
    # driver when the parent is imperfect on a held-out sample -- the steady state).
    train_new_samples: int = 0
    train_hard_samples: int = 0


@dataclass(frozen=True)
class RefineResult:
    outcome: RefineOutcome
    parent_revision: str
    new_revision: str | None = None
    best_epoch: int | None = None
    remaining: RemainingWork = RemainingWork()
    validation_parent: XCalibMetrics | None = None
    validation_candidate: XCalibMetrics | None = None
    audit: XCalibMetrics | None = None
    gate_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameLocalTrainingBatch(XCalibTrainingBatch):
    """ML batch plus read-only identities useful for contract tests/debugging."""

    lidar_keys: np.ndarray
    frame_keys: tuple[str, ...]


ProgressCallback = Callable[[RefineProgress], None]
FrameLoader = Callable[
    [str | Path, str, str, FeatureWork, Sequence[CorrectionRecord]],
    XCalibFrameInput,
]


def publication_gate(
    parent: XCalibMetrics,
    candidate: XCalibMetrics,
    config: RefineConfig = RefineConfig(),
) -> GateDecision:
    reasons: list[str] = []
    # NaN/Inf must fail closed: every check below is a "reject-if" comparison, and
    # `NaN < threshold` / `NaN > threshold` is False, so a candidate with non-finite
    # metrics would slip through as accepted and publish a broken revision.
    nonfinite = [
        name
        for name in (
            "primary_score",
            "positive_top1",
            "hard_negative_violation_rate",
            "no_match_false_positive_rate",
        )
        if not math.isfinite(getattr(candidate, name))
    ]
    if nonfinite:
        return GateDecision(False, tuple(f"candidate {name} is not finite" for name in nonfinite))
    if candidate.primary_score - parent.primary_score < config.minimum_primary_delta:
        reasons.append("primary delta below 0.005")
    if parent.positive_top1 - candidate.positive_top1 > config.max_top1_drop:
        reasons.append("positive top-1 regressed by more than 0.01")
    if (
        candidate.hard_negative_violation_rate
        - parent.hard_negative_violation_rate
        > config.max_hard_violation_increase
    ):
        reasons.append("hard-negative violations increased by more than 0.01")
    if (
        candidate.no_match_false_positive_rate
        - parent.no_match_false_positive_rate
        > config.max_no_match_fpr_increase
    ):
        reasons.append("no-match false positives increased by more than 0.01")
    return GateDecision(not reasons, tuple(reasons))


def _train_config() -> XCalibTrainConfig:
    return XCalibTrainConfig(
        max_epochs=20,
        max_steps=2000,
        wall_time_seconds=1800,
        patience=3,
        learning_rate=1e-4,
        positive_ce_weight=1.0,
        positive_acceptance_weight=0.25,
        hard_margin_weight=1.0,
        no_match_weight=1.0,
        parent_anchor_weight=1e-4,
        hard_margin=0.20,
        acceptance_threshold=0.50,
    )


def _emit(progress: ProgressCallback | None, value: RefineProgress) -> None:
    if progress is not None:
        progress(value)


def _all_corrections(store: XCalibStore) -> list[CorrectionRecord]:
    rows: list[CorrectionRecord] = []
    after: str | None = None
    while True:
        page = list(store.iter_corrections(after_sample_key=after, limit=1000))
        if not page:
            return rows
        rows.extend(page)
        after = page[-1].sample_key


def _frame_ordinals(
    data_root: str | Path, scene: str, records: Sequence[CorrectionRecord]
) -> dict[str, int]:
    lidar_dir = Path(data_root).resolve() / scene / "lidar"
    frames = {
        item.stem
        for item in lidar_dir.iterdir()
        if item.is_file() and item.suffix.lower() in {".pcd", ".bin", ".npy"}
    } if lidar_dir.is_dir() else set()
    frames.update(row.frame for row in records)
    return {frame: ordinal for ordinal, frame in enumerate(sorted(frames))}


def _rect_geometry(item: Mapping[str, object]) -> tuple[float, float, float, float]:
    rect = item.get("rect")
    if not isinstance(rect, Mapping):
        raise ContinualRefineUnavailable("frame contains an invalid 2D rect")
    try:
        values = tuple(float(rect[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualRefineUnavailable("frame contains an invalid 2D rect") from exc
    return values  # type: ignore[return-value]


def _default_frame_loader(
    data_root: str | Path,
    scene: str,
    camera: str,
    work: FeatureWork,
    records: Sequence[CorrectionRecord],
) -> XCalibFrameInput:
    # Package-native default: read the scene (SUSTechPOINTS-style layout) via
    # ``xcalib.gui.scene``.
    # Imported lazily because ``scene_loader`` imports helpers from this module
    # (``_rect_geometry`` / ``_candidate_keys`` / the mismatch error), so a top-level
    # import would be circular. Callers may still inject their own ``frame_loader``
    # (tests inject a fake).
    from .scene_loader import scene_frame_loader

    return scene_frame_loader(str(data_root), scene, camera, work, records)


def _active_samples(
    state: ContinualState, records: Sequence[CorrectionRecord]
) -> list[SupervisionSample]:
    result: list[SupervisionSample] = []
    for record in records:
        sample = state.get_sample(record.sample_key)
        if sample is not None:
            result.append(sample)
    return result


def _remaining(
    state: ContinualState, records: Sequence[CorrectionRecord]
) -> RemainingWork:
    samples = _active_samples(state, records)
    missing = {
        (sample.frame, sample.feature_input_fingerprint)
        for sample in samples
        if sample.feature_state != "ready"
    }
    validation = {
        (sample.frame, sample.feature_input_fingerprint)
        for sample in samples
        if sample.dataset_split == "validation" and sample.feature_state != "ready"
    }
    return RemainingWork(
        missing_features=len(missing),
        validation_features=len(validation),
        new_samples=sum(sample.difficulty_state == "new" for sample in samples),
        hard_samples=sum(sample.difficulty_state == "hard" for sample in samples),
        train_new_samples=sum(
            sample.difficulty_state == "new" and sample.dataset_split == "train"
            for sample in samples
        ),
        train_hard_samples=sum(
            sample.difficulty_state == "hard" and sample.dataset_split == "train"
            for sample in samples
        ),
    )


def _active_feature_work_page(
    state: ContinualState,
    base_fingerprint: str,
    active_samples: Sequence[SupervisionSample],
    limit: int,
) -> list[FeatureWork]:
    active_groups = {
        (sample.frame, sample.feature_input_fingerprint) for sample in active_samples
    }
    total_groups = int(
        state.store.connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT 1 FROM sample_state
                   GROUP BY frame,feature_input_fingerprint
               )"""
        ).fetchone()[0]
    )
    return [
        item for item in state.feature_work_page(base_fingerprint, total_groups)
        if (item.frame, item.input_fingerprint) in active_groups
    ][:limit]


def _active_rotating_page(
    state: ContinualState,
    page: Callable[[int], list[SupervisionSample]],
    active_keys: set[str],
    limit: int,
) -> list[SupervisionSample]:
    total = int(
        state.store.connection.execute(
            """SELECT COUNT(*) FROM sample_state
               WHERE dataset_split='train' AND difficulty_state='mastered'
                 AND feature_state='ready'"""
        ).fetchone()[0]
    )
    return [sample for sample in page(total) if sample.sample_key in active_keys][:limit]


def _active_hard_page(
    state: ContinualState, active_keys: set[str], limit: int
) -> list[SupervisionSample]:
    total = int(
        state.store.connection.execute(
            """SELECT COUNT(*) FROM sample_state
               WHERE dataset_split='train' AND difficulty_state='hard'
                 AND feature_state='ready'"""
        ).fetchone()[0]
    )
    return [
        sample for sample in state.hard_page(total)
        if sample.sample_key in active_keys
    ][:limit]


def _cached_for_sample(
    state: ContinualState,
    sample: SupervisionSample,
    base_fingerprint: str | None,
) -> CachedFrame | None:
    fingerprint = base_fingerprint
    if fingerprint is None:
        row = state.store.connection.execute(
            """SELECT base_fingerprint FROM frame_features
               WHERE frame=? AND input_fingerprint=?
               ORDER BY created_at DESC,base_fingerprint LIMIT 1""",
            (sample.frame, sample.feature_input_fingerprint),
        ).fetchone()
        fingerprint = None if row is None else str(row[0])
    if fingerprint is None:
        return None
    return state.get_feature(
        sample.frame, sample.feature_input_fingerprint, fingerprint
    )


def _candidate_keys(record: CorrectionRecord) -> tuple[str, ...]:
    try:
        candidates = json.loads(record.candidates_json)
    except json.JSONDecodeError as exc:
        raise ContinualRefineUnavailable(
            f"invalid candidates_json for sample {record.sample_key}"
        ) from exc
    if not isinstance(candidates, list):
        raise ContinualRefineUnavailable(
            f"invalid candidates_json for sample {record.sample_key}"
        )
    keys: list[str] = []
    for candidate in candidates:
        key = candidate.get("lidar_key") if isinstance(candidate, dict) else None
        if isinstance(key, str) and key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _replaced_key(record: CorrectionRecord) -> str | None:
    if record.original_proposal_json is None:
        return None
    try:
        proposal = json.loads(record.original_proposal_json)
    except json.JSONDecodeError as exc:
        raise ContinualRefineUnavailable(
            f"invalid original proposal for sample {record.sample_key}"
        ) from exc
    if not isinstance(proposal, dict) or proposal.get("decision") != "positive":
        return None
    key = proposal.get("lidar_key")
    if not isinstance(key, str) or key == record.final_lidar_key:
        return None
    return key


def _finite_score(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


def _sample_decision(
    sample: SupervisionSample,
    record: CorrectionRecord,
    cached: CachedFrame,
    scores,
    config: XCalibTrainConfig,
) -> ScoreDecision:
    try:
        row = cached.rect_keys.index(sample.sample_key)
    except ValueError:
        return ScoreDecision(False, -1_000_000.0, 1_000_000.0)
    lidar_index = {key: index for index, key in enumerate(cached.lidar_keys)}
    visible = [lidar_index[key] for key in _candidate_keys(record) if key in lidar_index]
    retrieval = np.asarray(scores.retrieval_scores, dtype=float)
    probabilities = np.asarray(scores.acceptance_probabilities, dtype=float)
    if retrieval.shape != probabilities.shape or retrieval.shape != (
        len(cached.rect_keys), len(cached.lidar_keys)
    ):
        raise ContinualRefineUnavailable("ML scores do not match cached feature keys")
    if scores.acceptance_logits is None:
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        acceptance = np.log(clipped / (1.0 - clipped))
    else:
        acceptance = np.asarray(scores.acceptance_logits, dtype=float)
        if acceptance.shape != retrieval.shape:
            raise ContinualRefineUnavailable("ML acceptance logits have an invalid shape")

    max_acceptance = (
        max(float(acceptance[row, index]) for index in visible)
        if visible else -1_000_000.0
    )
    max_acceptance = _finite_score(max_acceptance, 1_000_000.0)
    if record.final_decision == "no_match":
        threshold_logit = math.log(
            config.acceptance_threshold / (1.0 - config.acceptance_threshold)
        )
        return ScoreDecision(
            max_acceptance < threshold_logit,
            0.0,
            max_acceptance,
        )

    target = lidar_index.get(record.final_lidar_key or "")
    if target is None or target not in visible:
        return ScoreDecision(False, -1_000_000.0, max_acceptance)
    target_score = float(retrieval[row, target])
    negatives = [index for index in visible if index != target]
    margin = (
        target_score - max(float(retrieval[row, index]) for index in negatives)
        if negatives else 1_000_000.0
    )
    margin = _finite_score(margin, -1_000_000.0)
    top = max(visible, key=lambda index: float(retrieval[row, index]))
    return ScoreDecision(top == target and margin >= config.hard_margin, margin, max_acceptance)


def _score_samples(
    samples: Sequence[SupervisionSample],
    state: ContinualState,
    handle,
    ml_api,
    base_fingerprint: str,
    train_config: XCalibTrainConfig,
    records_by_sample: Mapping[str, CorrectionRecord],
) -> tuple[dict[str, ScoreDecision], tuple[str, ...]]:
    grouped: dict[tuple[str, str], list[SupervisionSample]] = {}
    for sample in samples:
        grouped.setdefault(
            (sample.frame, sample.feature_input_fingerprint), []
        ).append(sample)
    decisions: dict[str, ScoreDecision] = {}
    scored: list[str] = []
    for frame_samples in grouped.values():
        cached = _cached_for_sample(state, frame_samples[0], base_fingerprint)
        if cached is None:
            continue
        embeddings = XCalibBaseEmbeddings(
            rect_keys=cached.rect_keys,
            lidar_keys=cached.lidar_keys,
            image_embeddings=cached.image_embeddings,
            lidar_embeddings=cached.lidar_embeddings,
            kept_2d_indices=np.arange(len(cached.rect_keys), dtype=np.int64),
            kept_3d_indices=np.arange(len(cached.lidar_keys), dtype=np.int64),
            base_fingerprint=cached.base_fingerprint,
        )
        scores = ml_api.score_xcalib_embeddings(handle, embeddings)
        for sample in frame_samples:
            record = records_by_sample.get(sample.sample_key)
            if record is None:
                continue
            decisions[sample.sample_key] = _sample_decision(
                sample, record, cached, scores, train_config
            )
            scored.append(sample.sample_key)
    return decisions, tuple(scored)


def _chunk_frame_groups(
    samples: Sequence[SupervisionSample], batch_frames: int
) -> Iterable[list[SupervisionSample]]:
    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")
    groups: dict[tuple[str, str], list[SupervisionSample]] = {}
    order: list[tuple[str, str]] = []
    for sample in samples:
        key = (sample.frame, sample.feature_input_fingerprint)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(sample)
    for start in range(0, len(order), batch_frames):
        yield [sample for key in order[start:start + batch_frames] for sample in groups[key]]


def _balanced_samples(
    samples: Sequence[SupervisionSample],
    records_by_sample: Mapping[str, CorrectionRecord],
) -> list[SupervisionSample]:
    buckets: dict[str, list[SupervisionSample]] = {
        "corrected": [], "no_match": [], "positive": []
    }
    for sample in samples:
        record = records_by_sample.get(sample.sample_key)
        if record is None:
            continue
        if record.final_decision == "no_match":
            buckets["no_match"].append(sample)
        elif _replaced_key(record) is not None:
            buckets["corrected"].append(sample)
        else:
            buckets["positive"].append(sample)
    result: list[SupervisionSample] = []
    while any(buckets.values()):
        for name in ("corrected", "no_match", "positive"):
            if buckets[name]:
                result.append(buckets[name].pop(0))
    return result


def build_training_batches(
    samples: Sequence[SupervisionSample],
    state: ContinualState,
    batch_frames: int = 8,
    *,
    base_fingerprint: str | None = None,
    records_by_sample: Mapping[str, CorrectionRecord] | None = None,
) -> list[XCalibTrainingBatch]:
    """Build batches whose row masks expose candidates from that row's frame only."""

    if records_by_sample is None:
        records_by_sample = {
            sample.sample_key: record
            for sample in samples
            if (record := state.store.get_correction(sample.sample_key)) is not None
        }

    unique: list[SupervisionSample] = []
    seen: set[str] = set()
    for sample in _balanced_samples(samples, records_by_sample):
        if sample.sample_key not in seen:
            seen.add(sample.sample_key)
            unique.append(sample)

    batches: list[XCalibTrainingBatch] = []
    for chunk in _chunk_frame_groups(unique, batch_frames):
        frame_groups: dict[tuple[str, str], list[SupervisionSample]] = {}
        for sample in chunk:
            frame_groups.setdefault(
                (sample.frame, sample.feature_input_fingerprint), []
            ).append(sample)

        blocks: list[tuple[tuple[str, str], CachedFrame, int]] = []
        all_lidar: list[np.ndarray] = []
        all_lidar_keys: list[str] = []
        offset = 0
        for key, frame_samples in frame_groups.items():
            cached = _cached_for_sample(state, frame_samples[0], base_fingerprint)
            if cached is None or not cached.lidar_keys:
                continue
            blocks.append((key, cached, offset))
            all_lidar.append(cached.lidar_embeddings)
            all_lidar_keys.extend(cached.lidar_keys)
            offset += len(cached.lidar_keys)
        if not blocks:
            continue

        rows: list[tuple[SupervisionSample, np.ndarray, list[int], int, int]] = []
        for key, cached, block_offset in blocks:
            rect_index = {value: index for index, value in enumerate(cached.rect_keys)}
            lidar_index = {value: index for index, value in enumerate(cached.lidar_keys)}
            for sample in frame_groups[key]:
                record = records_by_sample.get(sample.sample_key)
                image_row = rect_index.get(sample.sample_key)
                if record is None or image_row is None:
                    continue
                candidate_columns = [
                    block_offset + lidar_index[value]
                    for value in _candidate_keys(record)
                    if value in lidar_index
                ]
                if not candidate_columns:
                    continue
                positive = -1
                if record.final_decision == "positive":
                    target = lidar_index.get(record.final_lidar_key or "")
                    if target is None or block_offset + target not in candidate_columns:
                        continue
                    positive = block_offset + target
                replaced = -1
                # A replaced-proposal marker is only meaningful for a positive
                # correction (the model's original pick became a negative). For a
                # no_match verdict there is no positive, so `_batch_tensors` rejects
                # positive=-1 with replaced>=0; leave replaced=-1 (the rejection loss
                # already treats every displayed candidate as a negative).
                if positive >= 0:
                    replacement = _replaced_key(record)
                    if replacement is not None and replacement in lidar_index:
                        candidate = block_offset + lidar_index[replacement]
                        if candidate in candidate_columns and candidate != positive:
                            replaced = candidate
                rows.append(
                    (
                        sample,
                        cached.image_embeddings[image_row],
                        candidate_columns,
                        positive,
                        replaced,
                    )
                )
        if not rows:
            continue

        columns = len(all_lidar_keys)
        mask = np.zeros((len(rows), columns), dtype=bool)
        for row_index, (_, _, candidates, _, _) in enumerate(rows):
            mask[row_index, candidates] = True
        lidar_key_rows = np.tile(
            np.asarray(all_lidar_keys, dtype=object), (len(rows), 1)
        )
        batches.append(
            FrameLocalTrainingBatch(
                sample_keys=tuple(row[0].sample_key for row in rows),
                group_keys=tuple(row[0].group_key for row in rows),
                image_embeddings=np.stack([row[1] for row in rows]).astype(
                    np.float32, copy=False
                ),
                lidar_embeddings=np.concatenate(all_lidar, axis=0).astype(
                    np.float32, copy=False
                ),
                candidate_mask=mask,
                positive_index=np.asarray([row[3] for row in rows], dtype=np.int64),
                replaced_index=np.asarray([row[4] for row in rows], dtype=np.int64),
                lidar_keys=lidar_key_rows,
                frame_keys=tuple(row[0].frame for row in rows),
            )
        )
    return batches


def _split_samples(
    state: ContinualState,
    records: Sequence[CorrectionRecord],
    dataset_split: Literal["train", "validation", "audit"],
) -> list[SupervisionSample]:
    return [
        sample
        for sample in _active_samples(state, records)
        if sample.dataset_split == dataset_split and sample.feature_state == "ready"
    ]


def _lifecycle_payload(
    state: ContinualState,
    decisions: Mapping[str, ScoreDecision],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for sample_key, decision in decisions.items():
        sample = state.get_sample(sample_key)
        if sample is None:
            continue
        payload[sample_key] = {
            "mastered": decision.mastered,
            "positive_margin": decision.positive_margin,
            "max_negative_logit": decision.max_negative_logit,
            "label_revision": sample.label_revision,
            "label_input_fingerprint": sample.label_input_fingerprint,
        }
    return payload


def finalize_continual_publication(
    conn: sqlite3.Connection, operation: PublicationOperation
) -> None:
    """Idempotently commit candidate-derived state inside publication's TX."""

    payload = operation.commit_payload
    if payload.get("schema_version") != 1 or payload.get("kind") != "continual_refine":
        raise ValueError("continual publication payload is incompatible")
    decisions = payload.get("lifecycle")
    if not isinstance(decisions, dict):
        raise ValueError("continual publication lifecycle payload is incompatible")
    now = datetime.now(timezone.utc).isoformat()
    for sample_key, raw in decisions.items():
        if not isinstance(sample_key, str) or not isinstance(raw, dict):
            raise ValueError("continual publication lifecycle row is incompatible")
        row = conn.execute(
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
                raw.get("label_revision"),
                raw.get("label_input_fingerprint"),
                raw.get("label_revision"),
                raw.get("label_input_fingerprint"),
            ),
        ).fetchone()
        if row is None:
            continue
        mastered = raw.get("mastered") is True
        conn.execute(
            """UPDATE sample_state SET difficulty_state=?,official_revision=?,
               correct_streak=?,positive_margin=?,max_negative_logit=?,updated_at=?
               WHERE sample_key=? AND label_revision=? AND label_input_fingerprint=?""",
            (
                "mastered" if mastered else "hard",
                operation.artifact.revision,
                int(row[0]) + 1 if mastered else 0,
                float(raw.get("positive_margin", 0.0)),
                float(raw.get("max_negative_logit", 0.0)),
                now,
                sample_key,
                raw.get("label_revision"),
                raw.get("label_input_fingerprint"),
            ),
        )


def _result(
    outcome: RefineOutcome,
    parent_revision: str,
    state: ContinualState,
    records: Sequence[CorrectionRecord],
    **values,
) -> RefineResult:
    return RefineResult(
        outcome=outcome,
        parent_revision=parent_revision,
        remaining=_remaining(state, records),
        **values,
    )


def _publication_exists(conn: sqlite3.Connection, run_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM publication_operations WHERE run_id=?", (run_id,)
    ).fetchone() is not None


def _cleanup_candidate_temporary(model_root: str | Path, temporary: Path) -> None:
    root = Path(model_root).resolve()
    candidate = temporary.resolve()
    if (
        candidate.parent != root
        or not candidate.name.startswith(".continual-")
        or not candidate.name.endswith(".tmp")
    ):
        raise ContinualRefineUnavailable(
            f"refusing to clean unsafe candidate path: {candidate}"
        )
    if candidate.is_dir():
        shutil.rmtree(candidate)
    elif candidate.exists():
        candidate.unlink()


def run_continual_refine(
    data_root: str | Path,
    scene: str,
    camera: str,
    run_id: str,
    progress: ProgressCallback | None,
    *,
    config: RefineConfig = RefineConfig(),
    ml_api=None,
    publisher=None,
    model_resolver=None,
    store_factory=None,
    frame_loader: FrameLoader | None = None,
) -> RefineResult:
    """Run one finite continual-refine unit and retain the parent on any reject."""

    ml_api = _default_ml if ml_api is None else ml_api
    model_resolver = locations.resolve_model if model_resolver is None else model_resolver
    store_factory = open_store if store_factory is None else store_factory
    frame_loader = _default_frame_loader if frame_loader is None else frame_loader
    publisher = publish_revision if publisher is None else publisher

    model = model_resolver(data_root, scene, camera)
    if model is None:
        raise ContinualRefineUnavailable(f"no trained model at {scene}/{camera}")
    parent_ref: XCalibRevisionRef = model.revision
    parent = ml_api.load_xcalib_revision(parent_ref)
    train_config = _train_config()
    store = store_factory(model.model_root)
    try:
        records = [
            row for row in _all_corrections(store)
            if row.scene == scene and row.camera == camera
        ]
        records_by_sample = {record.sample_key: record for record in records}
        _emit(progress, RefineProgress("indexing", 0, len(records), "indexing review snapshots"))
        ordinals = _frame_ordinals(data_root, scene, records)
        state = ContinualState(store)
        for record in records:
            state.ensure_sample(
                record.sample_key,
                scene,
                camera,
                record.frame,
                ordinals[record.frame],
                record.camera_track,
                label_revision=record.label_revision,
                label_input_fingerprint=record.input_fingerprint,
                feature_input_fingerprint=record.input_fingerprint,
            )
        _emit(
            progress,
            RefineProgress("indexing", len(records), len(records), "review snapshot index ready"),
        )

        active_samples = _active_samples(state, records)
        work = _active_feature_work_page(
            state, parent.base_fingerprint, active_samples, config.embed_limit
        )
        total_feature_work = _remaining(state, records).missing_features
        _emit(
            progress,
            RefineProgress("caching", 0, total_feature_work, "caching frozen-base embeddings"),
        )
        records_by_frame: dict[str, list[CorrectionRecord]] = {}
        for record in records:
            records_by_frame.setdefault(record.frame, []).append(record)
        completed = 0
        failures: list[tuple[str, str]] = []
        for item in work:
            try:
                frame_input = frame_loader(
                    data_root, scene, camera, item, records_by_frame.get(item.frame, ())
                )
                extracted = ml_api.extract_xcalib_base_embeddings(parent, frame_input)
                if extracted.base_fingerprint != parent.base_fingerprint:
                    raise ContinualRefineUnavailable(
                        "extracted embeddings do not match the parent base fingerprint"
                    )
                state.store_feature(
                    CachedFrame(
                        frame=item.frame,
                        input_fingerprint=item.input_fingerprint,
                        base_fingerprint=extracted.base_fingerprint,
                        rect_keys=tuple(extracted.rect_keys),
                        lidar_keys=tuple(extracted.lidar_keys),
                        image_embeddings=np.asarray(
                            extracted.image_embeddings, dtype=np.float32
                        ),
                        lidar_embeddings=np.asarray(
                            extracted.lidar_embeddings, dtype=np.float32
                        ),
                    )
                )
            except Exception as exc:
                with store.transaction(immediate=True) as conn:
                    conn.execute(
                        """UPDATE sample_state SET feature_state=?
                           WHERE frame=? AND feature_input_fingerprint=?""",
                        (
                            "stale" if isinstance(exc, _FrameInputFingerprintMismatch)
                            else "error",
                            item.frame,
                            item.input_fingerprint,
                        ),
                    )
                # One unreadable/changed frame must not abort the whole run: with
                # validation-priority ordering a poisoned frame would sit first and
                # block every other frame forever. Record it, surface it, continue.
                # A systemic failure (nothing caches at all) still fails below.
                failures.append((item.frame, exc))
                _emit(
                    progress,
                    RefineProgress(
                        "caching",
                        completed,
                        total_feature_work,
                        f"skipped frame {item.frame}: {type(exc).__name__}",
                    ),
                )
                continue
            completed += 1
            _emit(
                progress,
                RefineProgress(
                    "caching",
                    completed,
                    total_feature_work,
                    f"cached {completed}/{total_feature_work} frame(s)",
                ),
            )

        if failures and completed == 0:
            frame, exc = failures[0]
            raise ContinualRefineUnavailable(
                f"no frames could be cached ({len(failures)} failed); first {frame}: {exc}"
            ) from exc
        remaining = _remaining(state, records)
        if remaining.validation_features:
            _emit(
                progress,
                RefineProgress(
                    "caching",
                    completed,
                    total_feature_work,
                    f"{remaining.validation_features} validation frame(s) remain uncached",
                ),
            )
            return _result("work_remaining", parent_ref.revision, state, records)

        new_samples = [
            sample for sample in _active_samples(state, records)
            if sample.difficulty_state == "new" and sample.feature_state == "ready"
        ]
        active_keys = {record.sample_key for record in records}
        probes = _active_rotating_page(
            state,
            state.mastered_probe_page,
            active_keys,
            config.mastered_probe_limit,
        )
        classify = list(dict.fromkeys(
            sample.sample_key for sample in [*new_samples, *probes]
        ))
        by_key = {sample.sample_key: sample for sample in [*new_samples, *probes]}
        classify_samples = [by_key[key] for key in classify]
        _emit(
            progress,
            RefineProgress("classifying", 0, len(classify_samples), "scoring corpus with parent"),
        )
        parent_decisions, scored = _score_samples(
            classify_samples,
            state,
            parent,
            ml_api,
            parent.base_fingerprint,
            train_config,
            records_by_sample,
        )
        state.apply_parent_scores(
            parent_ref.revision,
            parent_decisions,
            expected_labels={
                key: (record.label_revision, record.input_fingerprint)
                for key, record in records_by_sample.items()
            },
        )
        scored_probes = [key for key in scored if key in {p.sample_key for p in probes}]
        state.mark_served(run_id, scored_probes, page_kind="mastered_probe")
        _emit(
            progress,
            RefineProgress(
                "classifying", len(scored), len(classify_samples), "parent classification ready"
            ),
        )

        hard = _active_hard_page(state, active_keys, config.hard_limit)
        if not hard:
            remaining = _remaining(state, records)
            outcome: RefineOutcome = (
                "up_to_date"
                if not (
                    remaining.missing_features
                    or remaining.train_new_samples
                    or remaining.train_hard_samples
                )
                else "work_remaining"
            )
            return _result(outcome, parent_ref.revision, state, records)

        anchors = _active_rotating_page(
            state,
            state.mastered_anchor_page,
            active_keys,
            config.mastered_anchor_limit,
        )
        selected = [*hard, *anchors]
        train_batches = build_training_batches(
            selected,
            state,
            base_fingerprint=parent.base_fingerprint,
            records_by_sample=records_by_sample,
        )
        included = {
            key for batch in train_batches for key in batch.sample_keys
        }
        if not train_batches:
            return _result("work_remaining", parent_ref.revision, state, records)

        validation_samples = _split_samples(state, records, "validation")
        validation_batches = build_training_batches(
            validation_samples,
            state,
            base_fingerprint=parent.base_fingerprint,
            records_by_sample=records_by_sample,
        )
        validation_count = sum(len(batch.sample_keys) for batch in validation_batches)
        validation_groups = {
            key for batch in validation_batches for key in batch.group_keys
        }
        if (
            validation_count < config.min_validation_samples
            or len(validation_groups) < config.min_validation_groups
        ):
            reason = (
                f"insufficient grouped validation: {validation_count} sample(s), "
                f"{len(validation_groups)} group(s)"
            )
            _emit(progress, RefineProgress("validating", 0, 2, reason))
            return _result(
                "rejected", parent_ref.revision, state, records, gate_reasons=(reason,)
            )

        candidate = ml_api.clone_xcalib_adapter(parent)
        shuffled_batches = tuple(train_batches)

        def train_factory(epoch: int):
            rows = list(shuffled_batches)
            random.Random(train_config.seed + epoch).shuffle(rows)
            return rows

        def validation_factory():
            return list(validation_batches)

        _emit(
            progress,
            RefineProgress(
                "training", 0, train_config.max_epochs, "training warm-started candidate",
                0, train_config.max_epochs,
            ),
        )

        def training_progress(epoch: int, maximum: int, _metrics: dict) -> None:
            _emit(
                progress,
                RefineProgress(
                    "training", epoch, maximum, f"epoch {epoch}/{maximum}", epoch, maximum
                ),
            )

        report = ml_api.train_xcalib_adapter(
            candidate,
            parent,
            train_factory,
            validation_factory,
            train_config,
            progress=training_progress,
        )
        state.mark_served(
            run_id, [sample.sample_key for sample in hard if sample.sample_key in included]
        )
        state.mark_served(
            run_id,
            [sample.sample_key for sample in anchors if sample.sample_key in included],
            page_kind="mastered_anchor",
        )

        _emit(progress, RefineProgress("validating", 0, 2, "evaluating parent and candidate"))
        parent_metrics = ml_api.evaluate_xcalib_adapter(
            parent, validation_factory(), train_config
        )
        _emit(progress, RefineProgress("validating", 1, 2, "parent validation ready"))
        candidate_metrics = ml_api.evaluate_xcalib_adapter(
            candidate, validation_factory(), train_config
        )
        _emit(progress, RefineProgress("validating", 2, 2, "candidate validation ready"))
        gate = publication_gate(parent_metrics, candidate_metrics, config)
        if not gate.accepted:
            return _result(
                "rejected",
                parent_ref.revision,
                state,
                records,
                best_epoch=report.best_epoch,
                validation_parent=parent_metrics,
                validation_candidate=candidate_metrics,
                gate_reasons=gate.reasons,
            )

        audit_samples = _split_samples(state, records, "audit")
        audit_batches = build_training_batches(
            audit_samples,
            state,
            base_fingerprint=parent.base_fingerprint,
            records_by_sample=records_by_sample,
        )
        _emit(progress, RefineProgress("auditing", 0, 1, "evaluating immutable audit"))
        audit_metrics = ml_api.evaluate_xcalib_adapter(
            candidate, audit_batches, train_config
        )
        _emit(progress, RefineProgress("auditing", 1, 1, "audit report ready"))

        lifecycle_samples = [
            sample for sample in [*hard, *anchors, *validation_samples, *audit_samples]
            if sample.sample_key in included or sample.dataset_split != "train"
        ]
        candidate_decisions, _ = _score_samples(
            lifecycle_samples,
            state,
            candidate,
            ml_api,
            parent.base_fingerprint,
            train_config,
            records_by_sample,
        )
        revision = "continual-" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
        temporary = Path(model.model_root) / f".{revision}.tmp"
        migrate_publications(store.connection)
        if temporary.exists():
            if _publication_exists(store.connection, run_id):
                raise ContinualRefineUnavailable(
                    f"run {run_id} already has a recoverable publication operation"
                )
            _cleanup_candidate_temporary(model.model_root, temporary)
        _emit(progress, RefineProgress("publishing", 0, 1, "saving verified candidate"))
        try:
            artifact = ml_api.save_xcalib_revision(
                candidate,
                temporary,
                revision=revision,
                parent_revision=parent_ref.revision,
                train_config=train_config,
                metrics=candidate_metrics,
            )
            artifact = ml_api.verify_xcalib_revision(artifact, device="cpu")
            commit_payload = {
                "schema_version": 1,
                "kind": "continual_refine",
                "lifecycle": _lifecycle_payload(state, candidate_decisions),
                "train_config": asdict(train_config),
                "validation": {
                    "parent": asdict(parent_metrics),
                    "candidate": asdict(candidate_metrics),
                },
                "validation_parent": asdict(parent_metrics),
                "validation_candidate": asdict(candidate_metrics),
                "audit": asdict(audit_metrics),
                "best_epoch": report.best_epoch,
            }
            published = publisher(
                store.connection,
                run_id,
                parent_ref,
                artifact,
                commit_payload,
                finalize_continual_publication,
            )
        except Exception:
            if not _publication_exists(store.connection, run_id):
                _cleanup_candidate_temporary(model.model_root, temporary)
            raise
        _emit(progress, RefineProgress("publishing", 1, 1, "published accepted revision"))
        return _result(
            "accepted",
            parent_ref.revision,
            state,
            records,
            new_revision=published.revision,
            best_epoch=report.best_epoch,
            validation_parent=parent_metrics,
            validation_candidate=candidate_metrics,
            audit=audit_metrics,
        )
    finally:
        store.close()


__all__ = [
    "ContinualRefineUnavailable",
    "FrameLocalTrainingBatch",
    "GateDecision",
    "RefineConfig",
    "RefineProgress",
    "RefineResult",
    "RemainingWork",
    "build_training_batches",
    "finalize_continual_publication",
    "publication_gate",
    "run_continual_refine",
]
