"""
HuggingFace Hub integration for released model weights.

The public package only downloads released pretrained weights. Hugging Face
handles authentication and access checks for artifacts that are not public.

Usage::

    from xcalib import Matcher
    matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Repo-id constants (override via env or pass repo_id= explicitly)
# ---------------------------------------------------------------------------

ENV_PUBLIC_REPO = "XCALIB_HF_PUBLIC_REPO"

PUBLIC_MODEL_REPO = "UArizona/xcalib"

#: Sites whose artifacts are fully public. Other released sites are attempted
#: through the same default repo and may require Hugging Face access.
PUBLIC_SITES: frozenset[str] = frozenset({"a9_dataset_r02_s01"})

#: (model, site) pairs available as released checkpoints (see docs/hub.md).
SHIPPED_MODELS: Tuple[str, ...] = (
    "calibrefine",
    "crlite",
    "crlite_2dpe",
    "crlite_vit_exp1",
    "crlite_vit_exp3",
    "crlite_vit_exp4",
)


def is_public_site(site: str) -> bool:
    """True when the site belongs to the public (paper-facing) repo."""
    return site in PUBLIC_SITES


def default_model_repo(site: Optional[str] = None) -> str:
    """Return the model repo for a given site.

    The default repo may contain both public and access-controlled artifacts;
    Hugging Face surfaces the auth/permission error when needed.
    """
    return os.environ.get(ENV_PUBLIC_REPO, PUBLIC_MODEL_REPO)


def weights_filename(model: str, site: str) -> str:
    """Path of a checkpoint inside the model repo."""
    return f"checkpoints/{model}_{site}_best.pth"


def config_filename(model: str, site: str) -> str:
    """Path of a reference YAML inside the model repo."""
    return f"configs/{model}_{site}.yaml"


# ---------------------------------------------------------------------------
# Lazy import — huggingface_hub is a wheel dependency, but keep the error
# actionable for source checkouts that have not re-run `pixi install`.
# ---------------------------------------------------------------------------

def _hf_api():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "huggingface_hub is required for Hub downloads. "
            "Install it with `pip install huggingface_hub` (it is part of "
            "the xcalib wheel dependencies — in a source checkout, re-run "
            "`pixi install`)."
        ) from exc


# ---------------------------------------------------------------------------
# hf:// URIs
# ---------------------------------------------------------------------------

def is_hf_uri(spec: object) -> bool:
    """True when `spec` is a string of the form hf://org/repo/path..."""
    return isinstance(spec, str) and spec.startswith("hf://")


def parse_hf_uri(uri: str) -> Tuple[str, str, Optional[str]]:
    """Split ``hf://org/repo[@revision]/path/in/repo`` into its parts.

    Returns:
        (repo_id, path_in_repo, revision) — revision is None when omitted.
    """
    if not is_hf_uri(uri):
        raise ValueError(f"Not an hf:// URI: {uri!r}")
    rest = uri[len("hf://"):]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Malformed hf:// URI {uri!r}; expected "
            "hf://<org>/<repo>[@revision]/<path/in/repo>"
        )
    org, repo = parts[0], parts[1]
    revision: Optional[str] = None
    if "@" in repo:
        repo, revision = repo.split("@", 1)
    path_in_repo = "/".join(parts[2:])
    if not path_in_repo:
        raise ValueError(f"hf:// URI {uri!r} is missing the file path")
    return f"{org}/{repo}", path_in_repo, revision


def download_file(
    repo_id: str,
    filename: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    repo_type: str = "model",
) -> Path:
    """`hf_hub_download` wrapper returning a local cached Path."""
    hub = _hf_api()
    local = hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        repo_type=repo_type,
    )
    return Path(local)


def resolve_uri(uri: str, *, token: Optional[str] = None) -> Path:
    """Download the file behind an hf:// URI and return its local path."""
    repo_id, path_in_repo, revision = parse_hf_uri(uri)
    return download_file(repo_id, path_in_repo, revision=revision, token=token)


def resolve_pretrained(
    model: str,
    site: str = "a9_dataset_r02_s01",
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Download (weights, config) for a (model, site) pair from the Hub.

    Uses the default released-weights repo unless ``repo_id`` overrides it.
    Hugging Face handles auth/permission checks for access-controlled files.

    Files follow the repo convention `checkpoints/{model}_{site}_best.pth`
    + `configs/{model}_{site}.yaml`. Both are cached by huggingface_hub,
    so repeat calls are free.
    """
    repo = repo_id or default_model_repo(site)
    scope = "public" if is_public_site(site) else "access-controlled"
    logger.info(f"Resolving {model}/{site} from HF Hub repo '{repo}' ({scope})"
                + (f"@{revision}" if revision else ""))
    weights = download_file(
        repo, weights_filename(model, site), revision=revision, token=token
    )
    config = download_file(
        repo, config_filename(model, site), revision=revision, token=token
    )
    return weights, config

