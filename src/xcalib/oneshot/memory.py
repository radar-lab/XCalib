"""
FeatureBank — growing memory of geometry-confirmed object features.

Every accepted pseudo-pair contributes its (image embedding, LiDAR
embedding, geometric confidence). The bank is what lets the adapter keep
learning *new objects over time* without forgetting old ones: adaptation
batches are replayed from the whole bank, not just the latest frame.

Embeddings stored here are the **base-model** outputs (pre-adapter), so
the adapter can be re-trained from scratch at any point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

__all__ = ["FeatureBank"]


class FeatureBank:
    """Capped reservoir of (img_embed, lid_embed, confidence, frame_id).

    Up to `capacity` rows are kept. Once full, new rows displace random old
    ones (reservoir sampling), keeping the bank an unbiased sample of
    everything observed so far.
    """

    def __init__(self, capacity: int = 4096, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._img: Optional[np.ndarray] = None     # [N, D] float32
        self._lid: Optional[np.ndarray] = None     # [N, D] float32
        self._conf: Optional[np.ndarray] = None    # [N] float32
        self._frame: Optional[np.ndarray] = None   # [N] int64
        self._seen = 0  # total rows ever offered (for reservoir math)

    # ----- properties -----

    def __len__(self) -> int:
        return 0 if self._conf is None else int(self._conf.shape[0])

    @property
    def dim(self) -> Optional[int]:
        return None if self._img is None else int(self._img.shape[1])

    @property
    def total_seen(self) -> int:
        return self._seen

    def stats(self) -> Dict[str, float]:
        if len(self) == 0:
            return {"size": 0, "total_seen": self._seen, "mean_confidence": 0.0,
                    "n_frames": 0}
        return {
            "size": len(self),
            "total_seen": self._seen,
            "mean_confidence": float(self._conf.mean()),
            "n_frames": int(np.unique(self._frame).size),
        }

    # ----- mutation -----

    def add(
        self,
        img_embeds: np.ndarray,
        lid_embeds: np.ndarray,
        confidences: np.ndarray,
        frame_id: int,
    ) -> int:
        """Offer a batch of pairs; returns how many were stored."""
        img = np.asarray(img_embeds, dtype=np.float32)
        lid = np.asarray(lid_embeds, dtype=np.float32)
        conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
        if img.ndim != 2 or lid.ndim != 2:
            raise ValueError("embeddings must be [B, D]")
        if not (img.shape[0] == lid.shape[0] == conf.shape[0]):
            raise ValueError("img/lid/confidence batch sizes disagree")
        if img.shape[0] == 0:
            return 0

        if self._img is None:
            self._img = np.zeros((0, img.shape[1]), dtype=np.float32)
            self._lid = np.zeros((0, lid.shape[1]), dtype=np.float32)
            self._conf = np.zeros((0,), dtype=np.float32)
            self._frame = np.zeros((0,), dtype=np.int64)
        elif img.shape[1] != self.dim:
            raise ValueError(f"embedding dim changed: bank={self.dim}, got {img.shape[1]}")

        stored = 0
        for k in range(img.shape[0]):
            self._seen += 1
            if len(self) < self.capacity:
                self._img = np.vstack([self._img, img[k:k + 1]])
                self._lid = np.vstack([self._lid, lid[k:k + 1]])
                self._conf = np.append(self._conf, conf[k])
                self._frame = np.append(self._frame, np.int64(frame_id))
                stored += 1
            else:
                # Reservoir sampling (Algorithm R): keep with prob cap/seen.
                slot = int(self._rng.integers(0, self._seen))
                if slot < self.capacity:
                    self._img[slot] = img[k]
                    self._lid[slot] = lid[k]
                    self._conf[slot] = conf[k]
                    self._frame[slot] = frame_id
                    stored += 1
        return stored

    # ----- access -----

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniform sample (without replacement when possible)."""
        n = len(self)
        if n == 0:
            raise RuntimeError("FeatureBank is empty — observe frames first")
        take = min(int(batch_size), n)
        idx = self._rng.choice(n, size=take, replace=False)
        return {
            "img": self._img[idx],
            "lid": self._lid[idx],
            "confidence": self._conf[idx],
            "frame_id": self._frame[idx],
        }

    def all(self) -> Dict[str, np.ndarray]:
        if len(self) == 0:
            raise RuntimeError("FeatureBank is empty")
        return {
            "img": self._img.copy(),
            "lid": self._lid.copy(),
            "confidence": self._conf.copy(),
            "frame_id": self._frame.copy(),
        }

    # ----- persistence -----

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            img=self._img if self._img is not None else np.zeros((0, 0), np.float32),
            lid=self._lid if self._lid is not None else np.zeros((0, 0), np.float32),
            confidence=self._conf if self._conf is not None else np.zeros((0,), np.float32),
            frame_id=self._frame if self._frame is not None else np.zeros((0,), np.int64),
            capacity=self.capacity,
            total_seen=self._seen,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "FeatureBank":
        data = np.load(Path(path))
        bank = cls(capacity=int(data["capacity"]), seed=seed)
        if data["img"].size:
            bank._img = data["img"].astype(np.float32)
            bank._lid = data["lid"].astype(np.float32)
            bank._conf = data["confidence"].astype(np.float32)
            bank._frame = data["frame_id"].astype(np.int64)
        bank._seen = int(data["total_seen"])
        return bank
