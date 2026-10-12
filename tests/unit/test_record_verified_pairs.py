"""Corpus-write seam (``record_verified_pairs``) + the journal-free store upsert.

These keep the Qt-review corpus writes coherent with any other writer to the same
corpus: the same observation key, the same upsert/supersede/label-revision semantics.
"""

import hashlib
import json

import pytest

from xcalib.continual import VerifiedPair, record_verified_pairs
from xcalib.continual.record import _observation_key
from xcalib.continual.store import CorrectionRecord, open_store


def _make_scene(tmp_path, camera="cam", frame="f0"):
    root = tmp_path / "data" / "scene"
    (root / "camera" / camera).mkdir(parents=True)
    (root / "lidar").mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    (root / "label_fusion" / "camera" / camera).mkdir(parents=True)
    # frame_input_fingerprint only stat()s these two -- content is irrelevant
    (root / "camera" / camera / f"{frame}.jpg").write_bytes(b"img-bytes")
    (root / "lidar" / f"{frame}.pcd").write_bytes(b"pcd-bytes")
    boxes = [{"obj_id": "10", "obj_type": "Car",
              "psr": {"position": {"x": 1, "y": 2, "z": 3},
                      "scale": {"x": 2, "y": 4, "z": 6},
                      "rotation": {"x": 0, "y": 0, "z": 0}}}]
    (root / "label" / f"{frame}.json").write_text(json.dumps({"objs": boxes}))
    rects = [{"rect": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
              "obj_attr": "t1", "obj_type": "Car"}]
    (root / "label_fusion" / "camera" / camera / f"{frame}.json").write_text(
        json.dumps({"objs": rects}))
    return tmp_path / "data", "scene", camera, frame


def _pair(decision="positive", lidar_key="id:10", proposal=None):
    return VerifiedPair(
        rect=[10.0, 20.0, 30.0, 40.0], decision=decision, lidar_key=lidar_key,
        track="1", candidates=[{"lidar_key": "id:10", "score": 0.9}],
        proposal=proposal)


def test_observation_key_follows_the_documented_formula():
    # canonical-JSON sha256 over {scene,camera,frame,rect(rounded 4)} -- any other
    # writer must use this exact rule or the shared corpus won't dedupe the same rect.
    identity = {"camera": "cam", "frame": "f0",
                "rect": [10.0, 20.0, 30.0, 40.0], "scene": "S"}
    expected = "observation:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()
    assert _observation_key("S", "cam", "f0", [10.0, 20.0, 30.0, 40.0]) == expected


def test_record_verified_pairs_writes_a_coherent_correction(tmp_path):
    data_root, scene, camera, frame = _make_scene(tmp_path)
    store = open_store(tmp_path / "model")

    written = record_verified_pairs(
        store, str(data_root), scene, camera, frame,
        [_pair(proposal={"decision": "positive", "lidar_key": "id:10"})])

    assert written == 1
    rows = list(store.iter_corrections())
    assert len(rows) == 1
    row = rows[0]
    assert row.final_decision == "positive"
    assert row.final_lidar_key == "id:10"
    assert row.camera_track == "1"
    assert json.loads(row.rect_json) == [10.0, 20.0, 30.0, 40.0]
    assert row.observation_key == _observation_key(
        scene, camera, frame, [10, 20, 30, 40])
    assert row.provenance == "verified_model_correct"  # equalled the model proposal
    assert row.input_fingerprint  # non-empty v2 fingerprint
    assert any(c["lidar_key"] == "id:10"
               for c in json.loads(row.candidates_json))  # target kept


def test_record_verified_pairs_flags_human_corrections(tmp_path):
    data_root, scene, camera, frame = _make_scene(tmp_path)
    store = open_store(tmp_path / "model")
    # the model proposed a different id than the human accepted
    record_verified_pairs(
        store, str(data_root), scene, camera, frame,
        [_pair(proposal={"decision": "positive", "lidar_key": "id:99"})])
    assert list(store.iter_corrections())[0].provenance == "human_corrected"


def test_record_verified_pairs_is_idempotent_and_bumps_on_change(tmp_path):
    data_root, scene, camera, frame = _make_scene(tmp_path)
    store = open_store(tmp_path / "model")

    record_verified_pairs(store, str(data_root), scene, camera, frame, [_pair()])
    base = list(store.iter_corrections())[0]

    # re-record the identical decision -> same row, revision unchanged
    record_verified_pairs(store, str(data_root), scene, camera, frame, [_pair()])
    rows = list(store.iter_corrections())
    assert len(rows) == 1 and rows[0].label_revision == base.label_revision

    # change the decision -> same row, revision bumped
    record_verified_pairs(store, str(data_root), scene, camera, frame,
                          [_pair(decision="no_match", lidar_key=None)])
    rows = list(store.iter_corrections())
    assert len(rows) == 1
    assert rows[0].final_decision == "no_match"
    assert rows[0].label_revision == base.label_revision + 1


def _correction(sample_key, decision="positive", lidar="id:1"):
    return CorrectionRecord(
        sample_key=sample_key, observation_key="observation:X", scene="S",
        camera="cam", frame="f0", review_id=None, input_fingerprint="fp",
        model_revision="m", camera_track="1", rect_json="[0,0,1,1]",
        final_decision=decision, final_lidar_key=lidar, original_proposal_json=None,
        candidates_json="[]", provenance="p", label_revision=1,
        created_at="t", updated_at="t")


def test_record_verified_pairs_enforces_the_decision_lidar_key_invariant(tmp_path):
    data_root, scene, camera, frame = _make_scene(tmp_path)
    store = open_store(tmp_path / "model")

    # a rejection that still carries a (stale) id -> normalized to no target, so the
    # store CHECK (no_match => NULL) is satisfied instead of raising IntegrityError
    record_verified_pairs(
        store, str(data_root), scene, camera, frame,
        [VerifiedPair(rect=[10.0, 20.0, 30.0, 40.0], decision="no_match",
                      lidar_key="id:99")])
    row = list(store.iter_corrections())[0]
    assert row.final_decision == "no_match" and row.final_lidar_key is None

    # a positive without a target -> a clear ValueError, not a cryptic IntegrityError
    with pytest.raises(ValueError, match="requires a lidar_key"):
        record_verified_pairs(
            store, str(data_root), scene, camera, frame,
            [VerifiedPair(rect=[50.0, 60.0, 70.0, 80.0], decision="positive")])


def test_upsert_corrections_supersedes_older_sample_for_same_observation(tmp_path):
    store = open_store(tmp_path)
    store.upsert_corrections([_correction("s1")])
    store.upsert_corrections([_correction("s2")])  # same observation, new sample_key

    active = list(store.iter_corrections())
    assert [row.sample_key for row in active] == ["s2"]  # only the newest is active
    superseded = store.connection.execute(
        "SELECT superseded_by FROM corrections WHERE sample_key='s1'").fetchone()
    assert superseded["superseded_by"] == "s2"
