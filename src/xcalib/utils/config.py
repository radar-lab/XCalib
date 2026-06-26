"""
Tiny dict-backed Config used by every standalone model.

Replaces `src.utils.config.Config` so the edge package has zero dependency on
the lab training stack. Supports dotted-path access (`cfg.get('crlite.top_k')`)
identical to the original Config's contract.

Author: Lihao Guo (leolihao@arizona.edu)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class EdgeConfig:
    """Minimal Config with dotted-path get / set, populated from a YAML or dict.

    Only the subset of behaviour used at inference time is provided. There is
    no training section, no jinja substitution, and no active-model switch.
    """

    def __init__(self, data: Mapping[str, Any] | None = None):
        self._data: dict = dict(data or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EdgeConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Dotted-path lookup. Falls back to `default` if any key is missing."""
        node: Any = self._data
        for part in key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, key: str, value: Any) -> None:
        """Dotted-path set. Creates intermediate dicts as needed."""
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return self.get(key, _SENTINEL) is not _SENTINEL

    def __repr__(self) -> str:
        return f"EdgeConfig({self._data!r})"


_SENTINEL = object()


def load_yaml(path: str | Path) -> EdgeConfig:
    """Convenience wrapper for `EdgeConfig.from_yaml`."""
    return EdgeConfig.from_yaml(path)
