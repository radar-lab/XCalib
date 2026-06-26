"""
pytest collection hook — imports torch *before* any other heavy native
dependency so that on Windows the torch DLLs (cuDNN, cuBLAS, OpenMP,
shm.dll) win the DLL-resolution race against HDF5 / OpenCV.

Without this, `pixi run smoke` on Windows fails at collection time with
`OSError: [WinError 127] ... torch\\lib\\shm.dll` because pytest's
plugin loader pulls in `h5py` (via `xcalib.data`) which locks an
incompatible MSVC runtime before torch ever gets a chance to load. The
package itself does the imports in the right order at runtime, so this
is purely a test-harness ordering fix.

On Linux (Thor) the DLL resolution model is different and this file is
a harmless no-op.
"""

from __future__ import annotations

import os
from pathlib import Path

# Match the pixi activation env; cheap and consistent across CI runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: F401, E402  -- import-order matters; do not move

import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def vit_exp1_matcher(tmp_path_factory):
    """A CPU crlite_vit_exp1 matcher with random (but loadable) weights.

    Shared across test modules: building the ViT takes a couple of seconds
    and none of the protocol / build / one-shot machinery cares whether
    the weights are trained.
    """
    from xcalib import Matcher
    from xcalib.models.registry import build_model
    from xcalib.utils.config import load_yaml
    from xcalib.utils.io import default_config_path

    cfg = load_yaml(default_config_path("crlite_vit_exp1", "utc4"))
    cfg.set("device", "cpu")
    model = build_model("crlite_vit_exp1", cfg)
    weights = tmp_path_factory.mktemp("weights") / "crlite_vit_exp1.pth"
    torch.save({"model_state_dict": model.state_dict()}, weights)

    return Matcher.from_pretrained(
        model="crlite_vit_exp1", weights=weights, config=cfg, device="cpu"
    )
