"""
HuggingFace Hub integration for released dataset caches.

Only approved public dataset caches are distributed through the public Hub
workflow::

    UArizona/xcalib-a9    — A9 r02_s01 HDF5 caches
                            (a9_r02_s01_{train,val,test}.h5 at repo root)

Env override: ``XCALIB_HF_A9_REPO``.

Other caches can still be used from local ``.h5`` paths through
:func:`xcalib.train` or `load_dataset(..., local=...)`.

Usage::

    from xcalib import load_dataset

    loader = load_dataset("a9_dataset_r02_s01", split="test")
    for frame in loader:
        ...

    # Pre-fetch a public split:
    #   xcalib pull-dataset --site a9_dataset_r02_s01 --split test

Licensing note: the A9 caches derive from the TUM Traffic (A9) dataset,
which is distributed under CC BY-NC-ND 4.0 — confirm redistribution
permission before flipping the A9 repo public.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from .weights import download_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.hdf5_loader import UTCFrameLoader

# ---------------------------------------------------------------------------
# Site registry
# ---------------------------------------------------------------------------

ENV_A9_REPO = "XCALIB_HF_A9_REPO"

A9_DATASET_REPO = "UArizona/xcalib-a9"

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """Where one site's HDF5 caches live, locally and on the Hub."""

    site: str
    repo_env: str            # env var that overrides the repo id
    default_repo: str        # Hub dataset repo id
    hub_template: str        # path inside the repo, with {split}
    local_template: str      # default checkout-relative path, with {split}
    license: str             # SPDX-ish id for the dataset card

    def repo_id(self) -> str:
        return os.environ.get(self.repo_env, self.default_repo)

    def hub_path(self, split: str) -> str:
        return self.hub_template.format(split=split)

    def local_path(self, split: str) -> Path:
        return Path(self.local_template.format(split=split))


# Non-public caches are deliberately absent; pass those as local paths.
DATASETS: dict[str, DatasetSpec] = {
    "a9_dataset_r02_s01": DatasetSpec(
        site="a9_dataset_r02_s01",
        repo_env=ENV_A9_REPO,
        default_repo=A9_DATASET_REPO,
        hub_template="a9_r02_s01_{split}.h5",
        local_template="datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_{split}.h5",
        license="cc-by-nc-nd-4.0",
    ),
}


def dataset_spec(site: str) -> DatasetSpec:
    """Look up a site spec; raises with the known site names on a typo."""
    try:
        return DATASETS[site]
    except KeyError:
        raise KeyError(
            f"Unknown dataset site {site!r}. Known sites: {', '.join(sorted(DATASETS))}."
        ) from None


def _check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return split


# ---------------------------------------------------------------------------
# Download / resolve
# ---------------------------------------------------------------------------

def dataset_path(
    site: str,
    split: str = "test",
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    prefer_local: bool = True,
) -> Path:
    """Resolve one split of a site's cache to a local HDF5 path.

    A checkout-local copy (``datasets/...``) wins when present, so lab boxes
    never re-download data they already have; otherwise the file is fetched
    from the site's Hub dataset repo into the huggingface cache.
    """
    spec = dataset_spec(site)
    _check_split(split)

    if prefer_local:
        local = spec.local_path(split)
        if local.exists():
            logger.info(f"dataset {site}/{split}: using local cache {local}")
            return local.resolve()

    repo = repo_id or spec.repo_id()
    logger.info(f"dataset {site}/{split}: downloading from {repo}")
    return download_file(
        repo,
        spec.hub_path(split),
        revision=revision,
        token=token,
        repo_type="dataset",
    )


def load_dataset(
    site: str,
    split: str = "test",
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    local: Optional[str | Path] = None,
) -> "UTCFrameLoader":
    """Return an HDF5 frame loader for one split of a site's cache.

    Resolution order: explicit ``local`` path → checkout-local default →
    Hub download (cached by huggingface_hub). Requires ``h5py`` (the
    ``[train]`` extra).
    """
    from ..data import UTCFrameLoader  # lazy: h5py is an optional extra

    if local is not None:
        path = Path(local)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        path = dataset_path(
            site, split, repo_id=repo_id, revision=revision, token=token
        )
    return UTCFrameLoader(path)

