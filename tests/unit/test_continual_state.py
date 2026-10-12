from __future__ import annotations

import dataclasses
import sqlite3

import numpy as np
import pytest

from xcalib.continual.state import (
    CachedFrame,
    ContinualState,
    ScoreDecision,
    stable_split,
)
from xcalib.continual.store import IssuedReviewRecord, open_store


NOW = "2026-07-12T00:00:00+00:00"


@pytest.fixture()
def store(tmp_path):
    result = open_store(tmp_path / "model")
    try:
        yield result
    finally:
        result.close()


def _cached(frame="f0", input_fingerprint="raw-A", base_fingerprint="base-A"):
    return CachedFrame(
        frame=frame,
        input_fingerprint=input_fingerprint,
        base_fingerprint=base_fingerprint,
        rect_keys=("r0", "r1"),
        lidar_keys=("l0", "l1", "l2"),
        image_embeddings=np.arange(8, dtype=np.float32).reshape(2, 4),
        lidar_embeddings=np.arange(12, dtype=np.float32).reshape(3, 4),
    )


def _track_for(split):
    for index in range(10_000):
        track = f"{split}-{index}"
        if stable_split("S", "cam", f"track:{track}") == split:
            return track
    raise AssertionError(split)


def test_split_is_label_independent_track_stable_and_birth_group_immutable(store):
    state = ContinualState(store)
    a = state.ensure_sample("s/a", "S", "cam", "000099", 99, "t7")
    b = state.ensure_sample("s/b", "S", "cam", "008000", 8000, "t7")
    assert a.dataset_split == b.dataset_split
    assert a.group_key == b.group_key == "track:t7"
    assert stable_split("S", "cam", "track:t7") == a.dataset_split

    born = state.ensure_sample("fixed", "S", "cam", "f100", 100, None)
    assert born.group_key == "time:1"
    same = state.ensure_sample("fixed", "S", "cam", "f100", 9999, "later")
    assert (same.group_key, same.dataset_split) == (born.group_key, born.dataset_split)


def test_untracked_birth_group_uses_fixed_100_frame_bucket(store):
    state = ContinualState(store)
    assert state.ensure_sample("a", "S", "cam", "f100", 100, None).group_key == "time:1"
    assert state.ensure_sample("b", "S", "cam", "f199", 199, None).group_key == "time:1"


def test_v1_migrates_to_v2_and_review_store_can_reopen_and_write(store):
    ContinualState(store)  # constructing migrates v1 -> v2 (side effect)
    assert store.connection.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()[0] == "2"
    path = store.path
    store.close()
    reopened = open_store(path.parent)
    try:
        reopened.issue_review(IssuedReviewRecord(
            review_id="review-after-v2", scene="S", camera="cam", frame="f0",
            model_revision="rev-1", checkpoint_sha256="c" * 64,
            input_fingerprint="labels-A", payload_hash="p" * 64,
            payload_json="{}", created_at=NOW,
        ))
        assert reopened.get_issued_review("review-after-v2") is not None
        assert ContinualState(reopened).get_sample("missing") is None
    finally:
        reopened.close()


def test_valid_cache_skips_raw_work_and_base_change_requeues(store):
    state = ContinualState(store)
    state.ensure_sample("sample", "S", "cam", "f0", 0, None,
                        feature_input_fingerprint="raw-A")
    state.store_feature(_cached())
    assert state.feature_work_page("base-A", 500) == []
    assert [x.frame for x in state.feature_work_page("base-B", 500)] == ["f0"]
    assert state.get_sample("sample").feature_state == "stale"


def test_cache_round_trip_validates_two_typed_modalities(store):
    state = ContinualState(store)
    state.ensure_sample("sample", "S", "cam", "f0", 0, None,
                        feature_input_fingerprint="raw-A")
    expected = _cached()
    state.store_feature(expected)
    actual = state.get_feature("f0", "raw-A", "base-A")
    assert actual is not None
    assert actual.rect_keys == expected.rect_keys and actual.lidar_keys == expected.lidar_keys
    np.testing.assert_array_equal(actual.image_embeddings, expected.image_embeddings)
    np.testing.assert_array_equal(actual.lidar_embeddings, expected.lidar_embeddings)
    assert state.get_sample("sample").feature_state == "ready"


@pytest.mark.parametrize("column", ["image_blob", "lidar_blob"])
def test_corrupt_cache_is_deleted_marked_stale_and_reenters_work(store, column):
    state = ContinualState(store)
    state.ensure_sample("sample", "S", "cam", "f0", 0, None,
                        feature_input_fingerprint="raw-A")
    state.store_feature(_cached())
    store.connection.execute(
        f"UPDATE frame_features SET {column}=? WHERE frame='f0'",
        (sqlite3.Binary(b"bad"),),
    )
    assert [x.frame for x in state.feature_work_page("base-A", 500)] == ["f0"]
    assert state.get_feature("f0", "raw-A", "base-A") is None
    assert state.get_sample("sample").feature_state == "stale"
    assert [x.frame for x in state.feature_work_page("base-A", 500)] == ["f0"]


def test_two_modality_write_and_ready_transition_are_atomic(store):
    state = ContinualState(store)
    state.ensure_sample("sample", "S", "cam", "f0", 0, None,
                        feature_input_fingerprint="raw-A")
    invalid = dataclasses.replace(
        _cached(), image_embeddings=np.ones((2, 4), dtype=np.float64)
    )
    with pytest.raises(ValueError, match="float32"):
        state.store_feature(invalid)
    assert state.get_sample("sample").feature_state == "missing"
    assert store.connection.execute("SELECT COUNT(*) FROM frame_features").fetchone()[0] == 0


def test_label_and_feature_lifecycle_are_independent(store):
    state = ContinualState(store)
    state.ensure_sample(
        "positive", "S", "cam", "f0", 0, None, label_revision=1,
        label_input_fingerprint="labels-A", feature_input_fingerprint="raw-A",
    )
    state.store_feature(_cached())
    state.apply_parent_scores("rev-2", {"positive": ScoreDecision(True, .42, -1.2)})
    state.ensure_sample(
        "positive", "S", "cam", "f0", 0, None, label_revision=2,
        label_input_fingerprint="labels-B",
    )
    relabelled = state.get_sample("positive")
    assert relabelled.difficulty_state == "new"
    assert relabelled.feature_state == "ready"
    assert relabelled.feature_input_fingerprint == "raw-A"

    state.apply_parent_scores("rev-3", {"positive": ScoreDecision(True, .5, -1.4)})
    state.ensure_sample(
        "positive", "S", "cam", "f0", 0, None, label_revision=2,
        feature_input_fingerprint="raw-B",
    )
    geometry_changed = state.get_sample("positive")
    assert geometry_changed.difficulty_state == "mastered"
    assert geometry_changed.feature_state == "stale"


def test_feature_work_is_distinct_by_frame_and_validation_first(store):
    state = ContinualState(store)
    state.ensure_sample("t1", "S", "cam", "train", 0, _track_for("train"))
    state.ensure_sample("t2", "S", "cam", "train", 0, _track_for("train"))
    state.ensure_sample("v", "S", "cam", "validation", 0, _track_for("validation"))
    assert [x.frame for x in state.feature_work_page("base-A", 500)] == [
        "validation", "train"
    ]


def test_parent_classification_and_forgotten_mastered_transition(store):
    state = ContinualState(store)
    state.ensure_sample("positive", "S", "cam", "f0", 0, None)
    state.apply_parent_scores("rev-2", {"positive": ScoreDecision(True, .42, -1.2)})
    assert state.get_sample("positive").difficulty_state == "mastered"
    state.apply_parent_scores("rev-3", {"positive": ScoreDecision(False, -.08, .7)})
    sample = state.get_sample("positive")
    assert sample.difficulty_state == "hard"
    assert sample.official_revision == "rev-3" and sample.correct_streak == 0


def test_hard_page_is_fair_and_mark_retry_is_idempotent(store):
    state = ContinualState(store)
    state.seed_hard([f"s{i:05d}" for i in range(5000)])
    first = state.hard_page(4096)
    served = [x.sample_key for x in first[:10]]
    state.mark_served("run-1", served)
    state.mark_served("run-1", served)
    second = state.hard_page(4096)
    assert [x.sample_key for x in second[:10]] == [x.sample_key for x in first[10:20]]
    assert all(state.get_sample(key).served_count == 1 for key in served)


def test_probe_anchor_cursors_are_isolated_and_advance_only_after_mark(store):
    state = ContinualState(store)
    state.seed_mastered([f"m{i:02d}" for i in range(6)])
    probe = state.mastered_probe_page(2)
    anchor = state.mastered_anchor_page(2)
    assert probe == anchor == state.mastered_probe_page(2)
    state.mark_served("run-1", [x.sample_key for x in probe], page_kind="mastered_probe")
    assert [x.sample_key for x in state.mastered_probe_page(2)] == ["m02", "m03"]
    assert state.mastered_anchor_page(2) == anchor
    state.mark_served("run-1", [x.sample_key for x in anchor], page_kind="mastered_anchor")
    assert [x.sample_key for x in state.mastered_anchor_page(2)] == ["m02", "m03"]


def test_validation_and_audit_never_enter_gradient_pages(store):
    state = ContinualState(store)
    state.seed_hard(["hard"])
    state.seed_mastered(["mastered"])
    with store.transaction(immediate=True) as db:
        db.execute("UPDATE sample_state SET dataset_split='validation' WHERE sample_key='hard'")
        db.execute("UPDATE sample_state SET dataset_split='audit' WHERE sample_key='mastered'")
    assert all(x.dataset_split == "train" for x in state.hard_page(4096))
    assert all(x.dataset_split == "train" for x in state.mastered_probe_page(512))
    assert all(x.dataset_split == "train" for x in state.mastered_anchor_page(256))
