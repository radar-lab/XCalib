"""Offline tests that the 3D box source reaches the public entry points that advertise it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from xcalib.data import hdf5_loader
from xcalib.hub import datasets as ds

A9 = "a9_dataset_r02_s01"
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "paper"


def _load_script(name: str):
    """Import a script of scripts/paper by path (it is not a package); main() is not run."""
    spec = importlib.util.spec_from_file_location("xcalib_script_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset_forwards_the_box_source(tmp_path, monkeypatch):
    path = tmp_path / "cache.h5"
    path.write_bytes(b"")
    seen = []

    class Spy:
        def __init__(self, cache, box_source="stored"):
            seen.append((cache, box_source))

    monkeypatch.setattr(hdf5_loader, "UTCFrameLoader", Spy)
    ds.load_dataset(A9, local=path, box_source="envelope")
    ds.load_dataset(A9, local=path)
    assert seen == [(path, "envelope"), (path, "stored")]


def test_load_dataset_rejects_a_bad_box_source_before_any_download(monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("the dataset was fetched before box_source was checked")

    monkeypatch.setattr(ds, "dataset_path", no_download)
    with pytest.raises(ValueError, match="box_source must be one of stored, envelope"):
        ds.load_dataset(A9, box_source="hull")


# ---------------------------------------------------------------------------
# scripts/paper/train_hdf5.py
# ---------------------------------------------------------------------------

def test_train_hdf5_script_forwards_the_box_source_flag(tmp_path, monkeypatch):
    script = _load_script("train_hdf5")
    calls = []

    def fake_train(*args, **kwargs):
        calls.append(kwargs)
        return tmp_path / "best.pth"

    monkeypatch.setattr(script, "train", fake_train)
    base = ["train_hdf5.py", "--model", "crlite", "--train-hdf5", "a.h5", "--val-hdf5", "b.h5"]
    for extra in (["--box-source", "envelope"], []):
        monkeypatch.setattr(sys, "argv", base + extra)
        script.main()
    # Without the flag nothing is forced: the config, then 'stored', decide inside train().
    assert [call["box_source"] for call in calls] == ["envelope", None]


# ---------------------------------------------------------------------------
# scripts/paper/validate_paper.py
# ---------------------------------------------------------------------------

def test_validate_paper_box_source_and_the_shape_of_its_report(tmp_path, monkeypatch):
    script = _load_script("validate_paper")
    cache = tmp_path / "a9.h5"
    cache.write_bytes(b"")
    used = []
    summary = {"top1": 0.9, "top3": 1.0, "mrr": 0.95, "latency_ms_mean": 1.0, "latency_ms_p95": 2.0, "n_frames": 3}

    def fake_evaluate(model, site, hdf5_path, device, limit_frames=None, box_source="stored"):
        used.append(box_source)
        return dict(summary)

    monkeypatch.setattr(script, "evaluate_one_model", fake_evaluate)
    common = ["validate_paper.py", "--site", A9, "--hdf5", str(cache), "--models", "crlite"]
    for name, extra in (("default.json", []), ("envelope.json", ["--box-source", "envelope"])):
        monkeypatch.setattr(sys, "argv", common + ["--output", str(tmp_path / name)] + extra)
        script.main()

    assert used == ["stored", "envelope"]
    default = json.loads((tmp_path / "default.json").read_text(encoding="utf-8"))
    envelope = json.loads((tmp_path / "envelope.json").read_text(encoding="utf-8"))
    # A default report keeps the shape the frozen evidence files have; only a non-default choice is recorded.
    assert set(default) == {"site", "hdf5", "device", "results"}
    assert envelope["box_source"] == "envelope"
