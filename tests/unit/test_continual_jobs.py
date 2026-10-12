import json
import os
from pathlib import Path
import time

import pytest

from xcalib.continual.train import RefineProgress
from xcalib.continual.state import ContinualState
from xcalib.continual.jobs import (
    CANCELLED_DETAIL,
    CancelUnavailable,
    ContinualJobs,
    LeaseLost,
    StartResult,
    _pid_live,
    cancel_refine_job,
    get_refine_job,
    get_refine_workload,
    recover_interrupted_runs,
    start_refine_job,
)
from xcalib.continual.store import open_store


def test_pid_live_probes_without_terminating_the_target():
    # os.kill(pid, 0) TERMINATES the target on Windows; probing our own live pid
    # must return True WITHOUT killing this process (if it were killed, the asserts
    # below never run and the test process dies).
    assert _pid_live(os.getpid()) is True
    assert _pid_live(2_000_000_000) is False  # cannot be a live pid
    assert _pid_live(0) is False
    assert _pid_live(None) is False
    # Prove we are still alive after probing ourselves.
    assert _pid_live(os.getpid()) is True


@pytest.fixture()
def model_root(tmp_path: Path) -> Path:
    root = tmp_path / "S" / "calib" / "xcalib" / "cam_model"
    root.mkdir(parents=True)
    return root


@pytest.fixture()
def fake_popen(monkeypatch, tmp_path):
    """Replace the launcher. Callers that pass repo_root="." run in tmp_path, which holds a
    stand-in worker at the default location, because start_refine_job refuses a missing script."""
    default_worker = tmp_path / "tools" / "xcalib" / "refine_continual.py"
    default_worker.parent.mkdir(parents=True)
    default_worker.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []

    class Process:
        pid = 998877

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr("xcalib.continual.jobs.subprocess.Popen", launch)
    return calls


def test_v3_jobs_store_reopens_through_continual_state(model_root):
    jobs = ContinualJobs(model_root, "S", "cam")
    jobs.close()
    store = open_store(model_root)
    try:
        ContinualState(store)
        assert store.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "3"
    finally:
        store.close()


def test_same_camera_click_reuses_active_run(model_root, fake_popen):
    first = start_refine_job(model_root, "S", "cam", "python", ".")
    second = start_refine_job(model_root, "S", "cam", "python", ".")
    assert first.disposition == "started"
    assert second == StartResult(first.run_id, "existing")
    assert len(fake_popen) == 1
    assert "--lease-token" in fake_popen[0][0]


def test_the_worker_script_defaults_to_the_repo_root_layout(model_root, fake_popen, tmp_path):
    started = start_refine_job(model_root, "S", "cam", "python", tmp_path)
    command, kwargs = fake_popen[0]
    assert command[:2] == ["python", str(tmp_path / "tools" / "xcalib" / "refine_continual.py")]
    assert command[command.index("--run-id") + 1] == started.run_id
    assert kwargs["cwd"] == str(tmp_path)


def test_a_relative_repo_root_gives_the_same_script_and_working_directory(model_root, fake_popen, tmp_path):
    start_refine_job(model_root, "S", "cam", "python", ".")
    command, kwargs = fake_popen[0]
    assert command[1] == str(tmp_path / "tools" / "xcalib" / "refine_continual.py")
    assert kwargs["cwd"] == str(tmp_path)


def test_a_missing_worker_script_fails_before_anything_is_queued(model_root, fake_popen, tmp_path):
    for worker in (tmp_path / "absent.py", "workers/absent.py", None):
        repo = tmp_path if worker is not None else tmp_path / "empty_repo"
        repo.mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError, match="refine worker script not found"):
            start_refine_job(model_root, "S", "cam", "python", repo, worker=worker)
    assert fake_popen == []
    assert not (model_root / "refine.sqlite").exists()


def test_an_explicit_worker_script_is_used_as_given(model_root, fake_popen, tmp_path):
    worker = tmp_path / "my_refine_worker.py"
    worker.write_text("", encoding="utf-8")
    start_refine_job(model_root, "S", "cam", "python", tmp_path, worker=worker)
    command, kwargs = fake_popen[0]
    assert command[:2] == ["python", str(worker)]
    # The data root is the folder that holds the scene folder: <data_root>/S/calib/xcalib/cam_model.
    assert command[2:6] == ["--data-root", str(model_root.parents[3]), "--scene", "S"]
    assert kwargs["cwd"] == str(tmp_path)
    # A relative name is passed through unchanged; it resolves against repo_root.
    (tmp_path / "workers").mkdir()
    (tmp_path / "workers" / "refine.py").write_text("", encoding="utf-8")
    other = model_root.parents[3] / "S2" / "calib" / "xcalib" / "cam_model"
    other.mkdir(parents=True)
    start_refine_job(other, "S2", "cam", "python", tmp_path, worker="workers/refine.py")
    assert fake_popen[1][0][:2] == ["python", str(Path("workers/refine.py"))]


def test_stale_lease_is_interrupted_and_next_click_starts(model_root, fake_popen, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("xcalib.continual.jobs._epoch", lambda: clock[0])
    first = start_refine_job(model_root, "S", "cam", "python", ".")
    clock[0] += 31
    recover_interrupted_runs(model_root, live_pids=set())
    assert get_refine_job(model_root, first.run_id).status == "interrupted"
    assert start_refine_job(model_root, "S", "cam", "python", ".").run_id != first.run_id


def test_committed_publication_reconciles_stale_run_to_accepted(model_root, fake_popen):
    # A publication can commit and THEN a crash (or a dying heartbeat) stamps the run
    # 'failed'/'interrupted' before jobs.finish records 'accepted'. recover_publications
    # never revisits an already-committed op, so recovery must independently reconcile the
    # run: the model revision is already live on disk and being served.
    first = start_refine_job(model_root, "S", "cam", "python", ".")
    stamp = "2026-07-13T00:00:00Z"
    store = open_store(model_root)
    try:
        with store.transaction(immediate=True) as db:
            db.execute(
                "UPDATE continual_runs SET status='failed',message='crash' WHERE run_id=?",
                (first.run_id,),
            )
            db.execute(
                """INSERT INTO publication_operations(run_id,model_dir,state,old_pointer_json,
                   new_pointer_json,artifact_json,commit_payload_json,created_at,updated_at)
                   VALUES(?,?, 'committed', NULL, ?, '{}', ?, ?, ?)""",
                (
                    first.run_id,
                    str(model_root),
                    json.dumps({"revision": "rev-9"}),
                    json.dumps({"validation": {"top1": 0.91}}),
                    stamp,
                    stamp,
                ),
            )
    finally:
        store.close()

    recovered = recover_interrupted_runs(model_root, live_pids=set())

    assert f"{first.run_id}:reconciled-accepted" in recovered
    job = get_refine_job(model_root, first.run_id)
    assert job.status == "accepted"
    assert job.new_revision == "rev-9"


def test_committed_publication_reconcile_is_idempotent(model_root, fake_popen):
    # A second recovery pass over an already-accepted run must be a no-op (status !=
    # 'accepted' guard), so re-running recovery never re-stamps or double-reports.
    first = start_refine_job(model_root, "S", "cam", "python", ".")
    stamp = "2026-07-13T00:00:00Z"
    store = open_store(model_root)
    try:
        with store.transaction(immediate=True) as db:
            db.execute(
                "UPDATE continual_runs SET status='interrupted' WHERE run_id=?",
                (first.run_id,),
            )
            db.execute(
                """INSERT INTO publication_operations(run_id,model_dir,state,old_pointer_json,
                   new_pointer_json,artifact_json,commit_payload_json,created_at,updated_at)
                   VALUES(?,?, 'committed', NULL, ?, '{}', '{}', ?, ?)""",
                (first.run_id, str(model_root), json.dumps({"revision": "rev-9"}), stamp, stamp),
            )
    finally:
        store.close()

    assert recover_interrupted_runs(model_root, live_pids=set())  # first pass reconciles
    second = recover_interrupted_runs(model_root, live_pids=set())
    assert f"{first.run_id}:reconciled-accepted" not in second
    assert get_refine_job(model_root, first.run_id).status == "accepted"


def test_progress_survives_service_reopen(model_root):
    jobs = ContinualJobs(model_root, "S", "cam")
    jobs.record_progress("run-1", RefineProgress("training", 3, 20, "epoch 3/20", 3, 20))
    jobs.close()
    reopened = ContinualJobs(model_root, "S", "cam")
    state = reopened.get("run-1")
    reopened.close()
    assert state.phase == "training" and state.percent == 15 and state.best_epoch == 3


def test_heartbeat_thread_uses_its_own_sqlite_connection(model_root, fake_popen):
    started = start_refine_job(model_root, "S", "cam", "python", ".")
    jobs = ContinualJobs(model_root, "S", "cam")
    token = jobs.store.connection.execute(
        "SELECT lease_token FROM continual_runs WHERE run_id=?", (started.run_id,)
    ).fetchone()[0]
    with jobs.lease_heartbeat(started.run_id, token, interval_seconds=0.01):
        time.sleep(0.04)
    jobs.close()


def test_full_train_writer_lock_blocks_refine(model_root, fake_popen):
    jobs = ContinualJobs(model_root, "S", "cam")
    with jobs.writer_lease("full_train", "train-1", ttl_seconds=60):
        result = start_refine_job(model_root, "S", "cam", "python", ".")
    jobs.close()
    assert result == StartResult(None, "locked", "full_train:train-1")
    assert fake_popen == []


def test_workload_new_count_reflects_the_corpus_before_any_run(model_root):
    # sample_state (hard/mastered) is only built while a refine run indexes the
    # corpus. Until then the footer must still show the corrections as 'new', not
    # 0/0/0 -- 'new' is derived from the live corrections minus hard/mastered.
    stamp = "2026-07-13T00:00:00Z"
    store = open_store(model_root)
    try:
        with store.transaction(immediate=True) as db:
            for i in range(3):
                db.execute(
                    """INSERT INTO corrections(sample_key,observation_key,scene,camera,
                       frame,review_id,input_fingerprint,model_revision,camera_track,
                       rect_json,final_decision,final_lidar_key,original_proposal_json,
                       candidates_json,provenance,label_revision,created_at,updated_at)
                       VALUES(?,?, 'S','cam','f0', NULL, ?, 'rev-1', NULL, '[]',
                       'positive','id:7', NULL, '[]', 'human_corrected', 1, ?, ?)""",
                    (f"s{i}", f"o{i}", "b" * 64, stamp, stamp),
                )
    finally:
        store.close()

    counts = get_refine_workload(model_root).counts
    assert counts["corrections"] == 3
    assert counts["new"] == 3  # derived from the corpus before any refine run




def _lease_token(model_root: Path, run_id: str) -> str:
    store = open_store(model_root)
    try:
        return store.connection.execute(
            "SELECT lease_token FROM continual_runs WHERE run_id=?", (run_id,)
        ).fetchone()["lease_token"]
    finally:
        store.close()


def test_cancel_makes_the_worker_stop_at_its_next_progress_tick(model_root, fake_popen):
    # THE mechanism. Cancel does not kill the process -- a refine writes a publication
    # journal and switches current.json, and killing it mid-write is exactly how a diverged
    # pointer is manufactured. Instead the run goes terminal, and the worker's own
    # record_progress (every phase calls it) raises LeaseLost, so it unwinds at a boundary
    # of its own choosing.
    started = start_refine_job(model_root, "S", "cam", "python", ".")
    token = _lease_token(model_root, started.run_id)
    with ContinualJobs(model_root, "S", "cam") as jobs:
        jobs.record_progress(started.run_id, token, RefineProgress("caching", 1, 10, "caching"))
        assert jobs.cancel(started.run_id) is True
        with pytest.raises(LeaseLost):
            jobs.record_progress(started.run_id, token, RefineProgress("caching", 2, 10, "caching"))
    state = get_refine_job(model_root, started.run_id)
    assert state.status == "interrupted"
    assert state.failure_detail == CANCELLED_DETAIL


def test_the_workers_fail_cannot_overwrite_a_cancel(model_root, fake_popen):
    # The worker catches the LeaseLost and calls fail(); that must NOT rewrite the user's
    # cancel into a generic 'failed', or the UI reports a crash the user deliberately caused.
    started = start_refine_job(model_root, "S", "cam", "python", ".")
    token = _lease_token(model_root, started.run_id)
    with ContinualJobs(model_root, "S", "cam") as jobs:
        jobs.cancel(started.run_id)
        jobs.fail(started.run_id, token, "LeaseLost: writer lease lost")
    state = get_refine_job(model_root, started.run_id)
    assert state.status == "interrupted"
    assert state.failure_detail == CANCELLED_DETAIL


def test_cancel_releases_the_lease_so_the_next_click_starts_a_fresh_run(model_root, fake_popen):
    # "terminate and go back to normal progress": a cancelled camera must be immediately
    # runnable again -- the writer lease must not linger until its TTL expires.
    first = start_refine_job(model_root, "S", "cam", "python", ".")
    assert cancel_refine_job(model_root, first.run_id) is True
    second = start_refine_job(model_root, "S", "cam", "python", ".")
    assert second.disposition == "started"
    assert second.run_id != first.run_id


def test_cancel_is_idempotent_and_never_clobbers_a_real_outcome(model_root, fake_popen):
    started = start_refine_job(model_root, "S", "cam", "python", ".")
    store = open_store(model_root)
    try:
        with store.transaction(immediate=True) as db:
            db.execute("UPDATE continual_runs SET status='accepted' WHERE run_id=?",
                       (started.run_id,))
    finally:
        store.close()
    assert cancel_refine_job(model_root, started.run_id) is False
    assert get_refine_job(model_root, started.run_id).status == "accepted"


def test_cancel_refuses_while_publishing(model_root, fake_popen):
    # 'publishing' IS the current.json pointer switch. It is short -- refuse and let the
    # caller retry rather than manufacture the divergence remediate_divergence repairs.
    started = start_refine_job(model_root, "S", "cam", "python", ".")
    token = _lease_token(model_root, started.run_id)
    with ContinualJobs(model_root, "S", "cam") as jobs:
        jobs.record_progress(started.run_id, token,
                             RefineProgress("publishing", 0, 1, "publishing"))
        with pytest.raises(CancelUnavailable):
            jobs.cancel(started.run_id)
    assert get_refine_job(model_root, started.run_id).status == "running"


def test_cancel_unknown_run_raises(model_root):
    with ContinualJobs(model_root, "S", "cam") as jobs:
        with pytest.raises(KeyError):
            jobs.cancel("no-such-run")
