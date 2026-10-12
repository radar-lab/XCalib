"""Corpus-write seam: record human-verified pairs into the durable corrections
corpus -- the lighter counterpart to the journalled apply
(:meth:`XCalibStore.begin_apply` / :meth:`XCalibStore.commit_apply`).

A front end without the apply journal (the Qt review) accumulates training this
way: verified pairs -> corpus -> ``run_continual_refine`` (gated, durable), instead
of the in-memory one-shot ``bank_pairs``/``adapt`` tier. Every writer to one corpus
MUST key observations + fingerprints by the same rule or the shared corpus splits,
so ``_observation_key`` (canonical-JSON sha256) and the v2 feature fingerprint are
deterministic functions of the observation: any other front end that follows the
same rules gets the same keys. This module writes through
:meth:`XCalibStore.upsert_corrections`, which shares one upsert helper with the
journalled apply.

The corpus location is the caller's choice: open the store at the model dir
(the review window uses the scene's ``calib/xcalib/<camera>_model`` folder)
and pass it in, along with the ``data_root`` the scene is read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Sequence

from .scene_loader import frame_input_fingerprint
from .store import CorrectionRecord, XCalibStore

__all__ = ["VerifiedPair", "record_verified_pairs"]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _observation_key(scene: str, camera: str, frame: str, rect: Sequence[float]) -> str:
    """Stable identity for a reviewed 2D observation: ``"observation:"`` plus the
    sha256 of the canonical JSON of scene, camera, frame and the rect rounded to 4
    decimals, so the same rect dedupes to one corpus row for any front end that
    follows this rule."""
    identity = {
        "scene": scene,
        "camera": camera,
        "frame": frame,
        "rect": [round(float(value), 4) for value in rect],
    }
    return "observation:" + hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()


@dataclass
class VerifiedPair:
    """One reviewed camera rect's outcome.

    ``decision`` is ``"positive"`` (accept ``lidar_key``) or ``"no_match"`` (the
    reviewer rejected every pairing). ``candidates`` is the ranked menu that was
    shown (``[{"lidar_key": ..., ...}]``); ``proposal`` is the model's original pick
    (``{"decision": ..., "lidar_key": ...}``) used only to label provenance.
    """

    rect: list[float]
    decision: str
    lidar_key: str | None = None
    track: str | None = None
    candidates: list[dict] = field(default_factory=list)
    proposal: dict | None = None


def record_verified_pairs(
    store: XCalibStore,
    data_root: str,
    scene: str,
    camera: str,
    frame: str,
    pairs: Sequence[VerifiedPair],
    *,
    model_revision: str = "",
) -> int:
    """Upsert human-verified pairs for one frame into the corpus. Returns the count.

    Re-recording the same rect upserts the same row (``sample_key`` is deterministic
    from the observation) and bumps ``label_revision`` only when the decision
    actually changed -- same idempotency as the journalled apply.
    """
    scene_dir = os.path.join(str(data_root), scene)
    fingerprint = frame_input_fingerprint(scene_dir, camera, frame)
    now = datetime.now(timezone.utc).isoformat()

    records: list[CorrectionRecord] = []
    for pair in pairs:
        # Enforce the corpus invariant up front with a clear error, instead of
        # letting a bad pair reach the store's CHECK constraint as a cryptic
        # IntegrityError that aborts the whole batch: a positive needs a target, a
        # rejection never carries one.
        decision, lidar_key = pair.decision, pair.lidar_key
        if decision == "no_match":
            lidar_key = None
        elif decision == "positive":
            if not lidar_key:
                raise ValueError(
                    f"a positive VerifiedPair (rect {pair.rect}) requires a lidar_key")
        else:
            raise ValueError(
                f"VerifiedPair.decision must be 'positive' or 'no_match', got {decision!r}")

        observation_key = _observation_key(scene, camera, frame, pair.rect)
        sample_key = "sample:" + hashlib.sha256(
            observation_key.encode("utf-8")
        ).hexdigest()

        candidates = [dict(option) for option in pair.candidates]
        # Keep an accepted target in the training candidate set even if the current
        # matcher ranked it out of the shown menu. The refine loader only needs the
        # stable key.
        if decision == "positive" and not any(
            option.get("lidar_key") == lidar_key for option in candidates
        ):
            candidates.append({"lidar_key": lidar_key, "source": "verified"})

        proposal = pair.proposal
        verified_correct = (
            proposal is not None
            and proposal.get("decision") == decision
            and proposal.get("lidar_key") == lidar_key
        )
        records.append(
            CorrectionRecord(
                sample_key=sample_key,
                observation_key=observation_key,
                scene=scene,
                camera=camera,
                frame=frame,
                review_id=None,  # a direct corpus write has no issued review
                input_fingerprint=fingerprint,
                model_revision=model_revision,
                camera_track=pair.track if (pair.track or "").isdigit() else None,
                rect_json=_canonical_json(list(pair.rect)),
                final_decision=decision,
                final_lidar_key=lidar_key,
                original_proposal_json=(
                    None if proposal is None else _canonical_json(proposal)
                ),
                candidates_json=_canonical_json(candidates),
                provenance=(
                    "verified_model_correct" if verified_correct else "human_corrected"
                ),
                label_revision=1,
                created_at=now,
                updated_at=now,
            )
        )
    return store.upsert_corrections(records)
