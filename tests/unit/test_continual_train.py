from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xcalib.continual.refine_model import (
    XCalibArtifact,
    XCalibBaseEmbeddings,
    XCalibFrameInput,
    XCalibMetrics,
    XCalibRevisionRef,
    XCalibScores,
    XCalibTrainConfig,
    XCalibTrainReport,
)
from xcalib.continual.train import (
    ContinualRefineUnavailable,
    build_training_batches,
    finalize_continual_publication,
    publication_gate,
    run_continual_refine,
)
from xcalib.continual.state import (
    CachedFrame,
    ContinualState,
    stable_split,
)
from xcalib.continual.store import (
    ApplyOperationRecord,
    CorrectionRecord,
    IssuedReviewRecord,
    ReviewFrameRecord,
    open_store,
)


NOW = "2026-07-12T00:00:00+00:00"


def metrics(
    primary=0.70,
    top1=0.80,
    hard_violation=0.10,
    no_match_fpr=0.05,
):
    return XCalibMetrics(
        primary_score=primary,
        positive_top1=top1,
        positive_mrr=top1,
        positive_margin=0.4,
        hard_negative_violation_rate=hard_violation,
        no_match_false_positive_rate=no_match_fpr,
        macro_group_score=primary,
        group_scores={"group": primary},
        positive_count=20,
        corrected_count=1,
        no_match_count=1,
    )


def _track_for(split: str, offset: int = 0) -> str:
    found = 0
    for index in range(100_000):
        track = f"{split}-{index}"
        if stable_split("S", "cam", f"track:{track}") == split:
            if found == offset:
                return track
            found += 1
    raise AssertionError(split)


def _seed_corrections(model_root: Path, rows: list[dict]) -> None:
    store = open_store(model_root)
    try:
        frames = []
        corrections = []
        request_frames = []
        for index, row in enumerate(rows):
            review_id = f"review-{index}"
            frame = row["frame"]
            sample_key = row["sample_key"]
            store.issue_review(
                IssuedReviewRecord(
                    review_id=review_id,
                    scene="S",
                    camera="cam",
                    frame=frame,
                    model_revision="rev-7",
                    checkpoint_sha256="c" * 64,
                    input_fingerprint=row.get("fingerprint", f"fp-{frame}"),
                    payload_hash=f"{index:064x}",
                    payload_json="{}",
                    created_at=NOW,
                )
            )
            request_frames.append({"review_id": review_id})
            frames.append(
                ReviewFrameRecord(
                    review_id=review_id,
                    operation_id="apply-1",
                    scene="S",
                    camera="cam",
                    frame=frame,
                    model_revision="rev-7",
                    input_fingerprint=row.get("fingerprint", f"fp-{frame}"),
                    verification_action="apply_all",
                    applied_at=NOW,
                )
            )
            candidates = row.get("candidates", ("id:a", "id:b"))
            corrections.append(
                CorrectionRecord(
                    sample_key=sample_key,
                    observation_key=f"observation-{index}",
                    scene="S",
                    camera="cam",
                    frame=frame,
                    review_id=review_id,
                    input_fingerprint=row.get("fingerprint", f"fp-{frame}"),
                    model_revision="rev-7",
                    camera_track=row["track"],
                    rect_json="[0,0,10,10]",
                    final_decision=row.get("decision", "positive"),
                    final_lidar_key=(
                        None if row.get("decision") == "no_match" else "id:a"
                    ),
                    original_proposal_json=json.dumps(
                        {
                            "sample_key": sample_key,
                            "decision": "positive",
                            "lidar_key": "id:b",
                        }
                    ),
                    candidates_json=json.dumps(
                        [{"lidar_key": key, "score": 0.5} for key in candidates]
                    ),
                    provenance="human_corrected",
                    label_revision=1,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        request_json = json.dumps({"frames": request_frames})
        store.begin_apply(
            ApplyOperationRecord(
                operation_id="apply-1",
                scene="S",
                camera="cam",
                payload_hash="a" * 64,
                request_json=request_json,
                state="pending",
                result_json=None,
                files=(),
                last_recovery_error=None,
            )
        )
        store.commit_apply("apply-1", frames, corrections, "{}")
    finally:
        store.close()


@dataclass
class Corpus:
    data_root: Path
    model_root: Path
    model_ref: SimpleNamespace


def _corpus(tmp_path: Path, rows: list[dict]) -> Corpus:
    model_root = tmp_path / "S" / "calib" / "xcalib" / "cam_model"
    model_root.mkdir(parents=True)
    weights = model_root / "parent.pth"
    config = model_root / "parent.yaml"
    weights.write_bytes(b"parent")
    config.write_text("model: fake", encoding="utf-8")
    revision = XCalibRevisionRef(
        model_dir=model_root,
        model_name="fake",
        revision="rev-7",
        parent_revision="rev-6",
        weights_path=weights,
        config_path=config,
        sha256="d" * 64,
        source="pointer",
    )
    _seed_corrections(model_root, rows)
    return Corpus(
        data_root=tmp_path,
        model_root=model_root,
        model_ref=SimpleNamespace(
            scene="S", camera="cam", model_root=model_root, revision=revision
        ),
    )


class FakeMl:
    def __init__(self):
        self.parent = SimpleNamespace(
            ref=SimpleNamespace(revision="rev-7"),
            base_fingerprint="base-A",
            kind="parent",
        )
        self.extract_calls = 0
        self.train_calls = 0
        self.save_calls = 0
        self.publish_calls = 0
        self.audit_evaluations = 0
        self.clone_parent_revision = None
        self.train_config = None
        self.training_sample_keys = ()
        self.master_all = False
        self.audit_metrics = metrics(primary=0.10)

    def parent_scores_master_everything(self):
        self.master_all = True

    def load_xcalib_revision(self, _ref):
        return self.parent

    def extract_xcalib_base_embeddings(self, _handle, frame_input):
        self.extract_calls += 1
        width = 4
        return XCalibBaseEmbeddings(
            rect_keys=frame_input.rect_keys,
            lidar_keys=frame_input.lidar_keys,
            image_embeddings=np.ones((len(frame_input.rect_keys), width), np.float32),
            lidar_embeddings=np.ones((len(frame_input.lidar_keys), width), np.float32),
            kept_2d_indices=np.arange(len(frame_input.rect_keys), dtype=np.int64),
            kept_3d_indices=np.arange(len(frame_input.lidar_keys), dtype=np.int64),
            base_fingerprint="base-A",
        )

    def score_xcalib_embeddings(self, handle, embeddings):
        rows, columns = len(embeddings.rect_keys), len(embeddings.lidar_keys)
        retrieval = np.zeros((rows, columns), np.float32)
        if columns:
            retrieval[:, 0] = 1.0 if handle.kind == "candidate" or self.master_all else 0.0
        if columns > 1:
            retrieval[:, 1] = 0.0 if handle.kind == "candidate" or self.master_all else 1.0
        logits = np.full((rows, columns), -2.0, np.float32)
        return XCalibScores(
            retrieval_scores=retrieval,
            acceptance_logits=logits,
            acceptance_probabilities=1.0 / (1.0 + np.exp(-logits)),
            uses_legacy_acceptance=False,
        )

    def clone_xcalib_adapter(self, parent):
        self.clone_parent_revision = parent.ref.revision
        return SimpleNamespace(
            ref=parent.ref, base_fingerprint=parent.base_fingerprint, kind="candidate"
        )

    def train_xcalib_adapter(
        self, candidate, _parent, train_factory, _validation_factory, config, progress
    ):
        self.train_calls += 1
        self.train_config = config
        batches = list(train_factory(1))
        self.training_sample_keys = tuple(
            key for batch in batches for key in batch.sample_keys
        )
        progress(1, config.max_epochs, {})
        return XCalibTrainReport(
            epochs_completed=1,
            optimizer_steps=len(batches),
            best_epoch=1,
            stopped_reason="early_stopping",
            best_metrics=metrics(primary=0.71),
            final_loss=0.2,
        )

    def evaluate_xcalib_adapter(self, handle, batches, _config):
        batches = list(batches)
        if any(key.startswith("audit") for batch in batches for key in batch.sample_keys):
            self.audit_evaluations += 1
            return self.audit_metrics
        return metrics(primary=0.70 if handle.kind == "parent" else 0.71)

    def save_xcalib_revision(
        self, candidate, target, *, revision, parent_revision, train_config, metrics
    ):
        self.save_calls += 1
        target = Path(target)
        return XCalibArtifact(
            revision=revision,
            parent_revision=parent_revision,
            model_name="fake",
            directory=target,
            weights_path=target / "fake.pth",
            config_path=target / "fake.yaml",
            sha256="e" * 64,
            base_fingerprint=candidate.base_fingerprint,
        )

    def verify_xcalib_revision(self, artifact, *, device):
        assert device == "cpu"
        return artifact


def _frame_loader(_root, _scene, _camera, work, records):
    return XCalibFrameInput(
        image=np.zeros((2, 2, 3), np.uint8),
        point_cloud=np.zeros((2, 3), np.float32),
        bboxes_2d=np.zeros((len(records), 4), np.float32),
        bboxes_3d=np.zeros((2, 6), np.float32),
        rect_keys=tuple(row.sample_key for row in records),
        lidar_keys=("id:a", "id:b"),
        input_fingerprint=work.input_fingerprint,
    )


def _publisher(fake_ml):
    def publish(conn, _run_id, parent, artifact, payload, finalize):
        fake_ml.publish_calls += 1
        finalize(conn, SimpleNamespace(commit_payload=payload, artifact=artifact))
        return XCalibRevisionRef(
            model_dir=parent.model_dir,
            model_name=artifact.model_name,
            revision=artifact.revision,
            parent_revision=artifact.parent_revision,
            weights_path=artifact.weights_path,
            config_path=artifact.config_path,
            sha256=artifact.sha256,
            source="pointer",
        )

    return publish


def _run(corpus, fake_ml, run_id="run-1"):
    return run_continual_refine(
        corpus.data_root,
        "S",
        "cam",
        run_id,
        progress=lambda _value: None,
        ml_api=fake_ml,
        publisher=_publisher(fake_ml),
        model_resolver=lambda *_args: corpus.model_ref,
        frame_loader=_frame_loader,
    )


def test_run_uses_budgets_and_warm_starts_from_parent(tmp_path):
    train_track = _track_for("train")
    validation_tracks = [_track_for("validation", index) for index in range(2)]
    rows = [
        {"frame": f"f{i:04d}", "sample_key": f"train-{i}", "track": train_track}
        for i in range(480)
    ]
    rows += [
        {
            "frame": f"v{i:04d}",
            "sample_key": f"validation-{i}",
            "track": validation_tracks[i % 2],
        }
        for i in range(20)
    ]
    corpus = _corpus(tmp_path, rows)
    fake_ml = FakeMl()
    result = _run(corpus, fake_ml)
    assert fake_ml.clone_parent_revision == "rev-7"
    assert fake_ml.extract_calls == 500
    assert fake_ml.train_config == XCalibTrainConfig(
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
    assert len(fake_ml.training_sample_keys) <= 4096 + 256
    assert result.parent_revision == "rev-7"


def test_batches_are_frame_local_and_mask_non_candidates(tmp_path):
    rows = [
        {
            "frame": f"f{i}",
            "sample_key": f"sample-{i}",
            "track": _track_for("train"),
            "candidates": ("id:a",) if i == 0 else ("id:b",),
            "decision": "positive" if i == 0 else "no_match",
        }
        for i in range(2)
    ]
    corpus = _corpus(tmp_path, rows)
    store = open_store(corpus.model_root)
    try:
        state = ContinualState(store)
        samples = []
        for index, row in enumerate(rows):
            sample = state.ensure_sample(
                row["sample_key"], "S", "cam", row["frame"], index, row["track"],
                feature_input_fingerprint=f"fp-{row['frame']}",
            )
            state.store_feature(
                CachedFrame(
                    frame=row["frame"],
                    input_fingerprint=f"fp-{row['frame']}",
                    base_fingerprint="base-A",
                    rect_keys=(row["sample_key"],),
                    lidar_keys=("id:a", "id:b"),
                    image_embeddings=np.ones((1, 4), np.float32),
                    lidar_embeddings=np.ones((2, 4), np.float32),
                )
            )
            samples.append(state.get_sample(sample.sample_key))
        batches = build_training_batches(
            samples, state, batch_frames=8, base_fingerprint="base-A"
        )
        assert len(batches) == 1
        batch = batches[0]
        assert set(batch.lidar_keys[0][batch.candidate_mask[0]]) == {"id:a"}
        assert set(batch.lidar_keys[1][batch.candidate_mask[1]]) == {"id:b"}
        # Regression: the no_match row (i=1) carries a positive original proposal
        # ("id:b"), which used to be recorded as a replaced-proposal marker even
        # though there is no positive. _batch_tensors (in refine_model) rejects
        # positive=-1 with replaced>=0, so that shape crashed training on the most
        # common correction type. Enforce the invariant for every row of every batch.
        assert batch.positive_index[1] == -1 and batch.replaced_index[1] == -1
        for produced in batches:
            no_positive = produced.positive_index < 0
            assert np.all(produced.replaced_index[no_positive] < 0)
    finally:
        store.close()


def test_strong_new_samples_can_finish_up_to_date_without_training(tmp_path):
    corpus = _corpus(
        tmp_path,
        [{"frame": "f0", "sample_key": "positive", "track": _track_for("train")}],
    )
    fake_ml = FakeMl()
    fake_ml.parent_scores_master_everything()
    result = _run(corpus, fake_ml, "run-2")
    assert result.outcome == "up_to_date"
    assert fake_ml.train_calls == 0


def test_cache_budget_with_unready_validation_finishes_work_remaining(tmp_path):
    tracks = [_track_for("validation", index) for index in range(2)]
    corpus = _corpus(
        tmp_path,
        [
            {
                "frame": f"v{i:04d}",
                "sample_key": f"validation-{i}",
                "track": tracks[i % 2],
            }
            for i in range(501)
        ],
    )
    fake_ml = FakeMl()
    result = _run(corpus, fake_ml, "run-budget")
    assert result.outcome == "work_remaining"
    assert result.remaining.validation_features == 1
    assert fake_ml.extract_calls == 500
    assert fake_ml.train_calls == fake_ml.save_calls == fake_ml.publish_calls == 0


def test_gate_requires_delta_and_all_guardrails():
    parent = metrics(primary=.70, top1=.80, hard_violation=.10, no_match_fpr=.05)
    assert publication_gate(
        parent,
        metrics(primary=.706, top1=.80, hard_violation=.10, no_match_fpr=.05),
    ).accepted
    assert not publication_gate(
        parent,
        metrics(primary=.704, top1=.90, hard_violation=.01, no_match_fpr=.01),
    ).accepted
    assert not publication_gate(
        parent,
        metrics(primary=.72, top1=.78, hard_violation=.10, no_match_fpr=.05),
    ).accepted
    # NaN/Inf candidate metrics must fail closed (a huge primary would otherwise
    # pass every "reject-if" comparison since NaN comparisons are all False).
    decision = publication_gate(
        parent, metrics(primary=float("nan"), top1=.90, hard_violation=.01, no_match_fpr=.01)
    )
    assert not decision.accepted
    assert any("not finite" in reason for reason in decision.reasons)


def test_audit_is_not_used_for_epoch_or_gate(tmp_path):
    train_track = _track_for("train")
    validation_tracks = [_track_for("validation", index) for index in range(2)]
    audit_track = _track_for("audit")
    rows = [{"frame": "train", "sample_key": "train", "track": train_track}]
    rows += [
        {
            "frame": f"v{i}",
            "sample_key": f"validation-{i}",
            "track": validation_tracks[i % 2],
        }
        for i in range(20)
    ]
    rows.append({"frame": "audit", "sample_key": "audit-1", "track": audit_track})
    corpus = _corpus(tmp_path, rows)
    fake_ml = FakeMl()
    fake_ml.audit_metrics = metrics(primary=.10)
    result = _run(corpus, fake_ml, "run-3")
    assert result.outcome == "accepted"
    assert fake_ml.audit_evaluations == 1
    assert fake_ml.publish_calls == 1


def test_batches_use_captured_label_and_publication_cas_skips_live_relabel(tmp_path):
    corpus = _corpus(
        tmp_path,
        [{"frame": "f0", "sample_key": "sample", "track": _track_for("train")}],
    )
    store = open_store(corpus.model_root)
    try:
        captured = store.get_correction("sample")
        assert captured is not None
        state = ContinualState(store)
        sample = state.ensure_sample(
            "sample", "S", "cam", "f0", 0, captured.camera_track,
            label_revision=captured.label_revision,
            label_input_fingerprint=captured.input_fingerprint,
            feature_input_fingerprint=captured.input_fingerprint,
        )
        state.store_feature(
            CachedFrame(
                frame="f0",
                input_fingerprint=captured.input_fingerprint,
                base_fingerprint="base-A",
                rect_keys=("sample",),
                lidar_keys=("id:a", "id:b"),
                image_embeddings=np.ones((1, 4), np.float32),
                lidar_embeddings=np.ones((2, 4), np.float32),
            )
        )
        with store.transaction(immediate=True) as conn:
            conn.execute(
                """UPDATE corrections SET final_decision='no_match',final_lidar_key=NULL,
                   label_revision=2 WHERE sample_key='sample'"""
            )

        batches = build_training_batches(
            [state.get_sample(sample.sample_key)],
            state,
            base_fingerprint="base-A",
            records_by_sample={"sample": captured},
        )
        assert len(batches) == 1
        assert int(batches[0].positive_index[0]) >= 0

        operation = SimpleNamespace(
            commit_payload={
                "schema_version": 1,
                "kind": "continual_refine",
                "lifecycle": {
                    "sample": {
                        "mastered": True,
                        "positive_margin": 1.0,
                        "max_negative_logit": -1.0,
                        "label_revision": captured.label_revision,
                        "label_input_fingerprint": captured.input_fingerprint,
                    }
                },
            },
            artifact=SimpleNamespace(revision="rev-8"),
        )
        with store.transaction(immediate=True) as conn:
            finalize_continual_publication(conn, operation)
        unchanged = state.get_sample("sample")
        assert unchanged.difficulty_state == "new"
        assert unchanged.official_revision is None
    finally:
        store.close()


def test_default_frame_loader_is_package_native(tmp_path):
    # The default frame_loader reads SUSTechPOINTS-style scenes via
    # xcalib.gui.scene -- it does not refuse up front for lack of an injected
    # loader. With a corpus whose data_root holds NO scene files, the run reaches
    # the package-native loader and simply fails to cache the frame.
    # (The loader's own behaviour is covered directly in test_scene_loader.py.)
    corpus = _corpus(
        tmp_path,
        [{"frame": "f0", "sample_key": "sample", "track": _track_for("train")}],
    )
    with pytest.raises(ContinualRefineUnavailable, match="no frames could be cached"):
        run_continual_refine(
            corpus.data_root,
            "S",
            "cam",
            "stale-run",
            progress=lambda _value: None,
            ml_api=FakeMl(),
            publisher=_publisher(FakeMl()),
            model_resolver=lambda *_args: corpus.model_ref,
        )



def test_one_poisoned_frame_does_not_block_the_others(tmp_path):
    # A single unreadable frame must not abort the whole run: the good frame still
    # caches, the poisoned one is marked, and the run keeps making progress.
    # (Validation-priority ordering used to sit the poisoned frame first and starve
    # every other frame; the whole run failed on the first bad frame.)
    corpus = _corpus(
        tmp_path,
        [
            {"frame": "f0", "sample_key": "s0", "track": _track_for("train")},
            {"frame": "f1", "sample_key": "s1", "track": _track_for("train", 1)},
        ],
    )
    fake = FakeMl()
    fake.parent_scores_master_everything()

    def poison(root, scene, camera, work, records):
        if work.frame == "f1":
            raise RuntimeError("f1 image is unreadable")
        return _frame_loader(root, scene, camera, work, records)

    # Must not raise even though f1 fails.
    run_continual_refine(
        corpus.data_root, "S", "cam", "poison-run",
        progress=lambda _v: None, ml_api=fake, publisher=_publisher(fake),
        model_resolver=lambda *_a: corpus.model_ref, frame_loader=poison,
    )
    store = open_store(corpus.model_root)
    try:
        state = ContinualState(store)
        assert state.get_sample("s0").feature_state == "ready"
        assert state.get_sample("s1").feature_state == "error"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM frame_features"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_verify_failure_removes_candidate_temp_and_same_run_can_retry(tmp_path):
    train_track = _track_for("train")
    validation_tracks = [_track_for("validation", index) for index in range(2)]
    rows = [{"frame": "train", "sample_key": "train", "track": train_track}]
    rows += [
        {
            "frame": f"v{i}",
            "sample_key": f"validation-{i}",
            "track": validation_tracks[i % 2],
        }
        for i in range(20)
    ]
    corpus = _corpus(tmp_path, rows)

    class FlakyVerifyMl(FakeMl):
        def __init__(self):
            super().__init__()
            self.verify_attempts = 0

        def save_xcalib_revision(self, *args, **kwargs):
            artifact = super().save_xcalib_revision(*args, **kwargs)
            artifact.directory.mkdir(parents=True)
            artifact.weights_path.write_bytes(b"candidate")
            artifact.config_path.write_text("model: fake", encoding="utf-8")
            return artifact

        def verify_xcalib_revision(self, artifact, *, device):
            self.verify_attempts += 1
            if self.verify_attempts == 1:
                raise RuntimeError("verification failed")
            return super().verify_xcalib_revision(artifact, device=device)

    fake_ml = FlakyVerifyMl()
    with pytest.raises(RuntimeError, match="verification failed"):
        _run(corpus, fake_ml, "retry-run")
    assert list(corpus.model_root.glob(".continual-*.tmp")) == []

    result = _run(corpus, fake_ml, "retry-run")
    assert result.outcome == "accepted"
    assert fake_ml.verify_attempts == 2
