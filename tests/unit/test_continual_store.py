"""Focused unit tests for the continual store (pure sqlite3).

These tests exercise the store's own durable-ledger contract (schema creation, the
immutable issued-review invariant, empty-corpus counts) using only its public API
-- enough to prove the module works standalone in the package environment.
"""
from __future__ import annotations

import pytest

from xcalib.continual import (
    CorpusCounts,
    IssuedReviewRecord,
    ReviewConflict,
    open_store,
)


def _issued(review_id: str = "r1", **override) -> IssuedReviewRecord:
    base = dict(
        review_id=review_id,
        scene="S",
        camera="cam",
        frame="f0",
        model_revision="rev-1",
        checkpoint_sha256="a" * 64,
        input_fingerprint="b" * 64,
        payload_hash="c" * 64,
        payload_json="{}",
        created_at="2026-07-13T00:00:00Z",
    )
    base.update(override)
    return IssuedReviewRecord(**base)


def test_open_store_creates_schema_and_empty_corpus(tmp_path):
    store = open_store(tmp_path)
    try:
        assert (tmp_path / "refine.sqlite").is_file()
        counts = store.corpus_counts()
        assert counts == CorpusCounts(frames=0, corrections=0, positives=0, no_matches=0)
        assert counts.total == 0
    finally:
        store.close()


def test_issued_review_round_trips_and_missing_is_none(tmp_path):
    store = open_store(tmp_path)
    try:
        assert store.get_issued_review("absent") is None
        record = _issued()
        store.issue_review(record)
        assert store.get_issued_review("r1") == record
    finally:
        store.close()


def test_issued_review_is_immutable_but_idempotent(tmp_path):
    store = open_store(tmp_path)
    try:
        store.issue_review(_issued())
        # Re-issuing the identical record is a no-op (idempotent retry).
        store.issue_review(_issued())
        assert store.get_issued_review("r1") == _issued()
        # Re-issuing the same id with a changed immutable field is rejected.
        with pytest.raises(ReviewConflict):
            store.issue_review(_issued(payload_hash="d" * 64))
    finally:
        store.close()
