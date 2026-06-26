"""Offline tests for xcalib.hub.datasets (no network, no real Hub repos)."""

from __future__ import annotations

import pytest

from xcalib.hub import datasets as ds

A9 = "a9_dataset_r02_s01"


# ---------------------------------------------------------------------------
# Site registry / repo routing
# ---------------------------------------------------------------------------

def test_site_conventions():
    a9 = ds.dataset_spec(A9)
    assert a9.default_repo == ds.A9_DATASET_REPO
    assert a9.hub_path("test") == "a9_r02_s01_test.h5"
    assert a9.license == "cc-by-nc-nd-4.0"


def test_utc_sites_are_not_hub_distributed():
    # UTC caches are deliberately absent from the public Hub registry.
    for site in ("utc3", "utc4"):
        assert site not in ds.DATASETS
        with pytest.raises(KeyError, match=A9):
            ds.dataset_spec(site)


def test_repo_env_override(monkeypatch):
    monkeypatch.setenv(ds.ENV_A9_REPO, "acme/a9-mirror")
    assert ds.dataset_spec(A9).repo_id() == "acme/a9-mirror"

    monkeypatch.delenv(ds.ENV_A9_REPO)
    assert ds.dataset_spec(A9).repo_id() == ds.A9_DATASET_REPO


def test_bad_split_rejected():
    with pytest.raises(ValueError, match="split"):
        ds.dataset_path(A9, "validation")


# ---------------------------------------------------------------------------
# Local-first resolution
# ---------------------------------------------------------------------------

def test_dataset_path_prefers_local(monkeypatch, tmp_path):
    local = tmp_path / "datasets" / A9 / "hdf5_cache" / "a9_r02_s01_test.h5"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"\x89HDF")
    monkeypatch.chdir(tmp_path)

    resolved = ds.dataset_path(A9, "test")
    assert resolved == local.resolve()


def test_dataset_path_falls_back_to_hub(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no local caches here
    calls = {}

    def fake_download(repo_id, filename, *, revision=None, token=None, repo_type="model"):
        calls.update(repo=repo_id, file=filename, type=repo_type)
        return tmp_path / "cached.h5"

    monkeypatch.setattr(ds, "download_file", fake_download)
    out = ds.dataset_path(A9, "val")
    assert out == tmp_path / "cached.h5"
    assert calls == {
        "repo": ds.A9_DATASET_REPO, "file": "a9_r02_s01_val.h5", "type": "dataset",
    }


def test_load_dataset_local_missing():
    with pytest.raises(FileNotFoundError):
        ds.load_dataset(A9, local="does/not/exist.h5")

