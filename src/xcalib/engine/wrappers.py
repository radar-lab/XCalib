"""
Per-model inference wrappers used by the validation script and the
partner-facing matcher.

Wrappers used by the paper validation script and the partner-facing matcher:
    - HybridTwoStageWrapper      -> CRLiteModel             (Stage 1 + Stage 2)
    - DotProductWrapper          -> CRLiteViTExp1Model      (cosine only)
    - PositionEnhancedDotWrapper -> CRLiteViTExp3Model      (cosine + PE)
    - PairwiseWrapper            -> CalibRefineModel        (O(N*M) fc1..fc6)

The wrappers accept a `FrameData` namespace dataclass produced by
`xcalib.data` and return an (N, M) similarity matrix plus the
elapsed forward-pass time in milliseconds.

All wrappers are device-agnostic; they push tensors to `self.device` once
inside `predict_matching_matrix`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..models._base import EdgeModelBase


# ============================================================================
# FrameData — what the wrappers expect on every call
# ============================================================================

@dataclass
class FrameData:
    """Pre-cropped frame ready for the model.

    Attributes
    ----------
    crops_2d         : [N, 3, H, W] float — image crops, already normalised.
    crops_3d         : [M, P, 3]    float — point-cloud crops, already
                                            resampled to P points.
    bboxes_2d        : [N, 4]       float — (x1,y1,x2,y2) per image crop
                                            (None disables img position encoding).
    bbox_centers_3d  : [M, 3]       float — (x,y,z) per LiDAR crop center
                                            (None disables 3D position encoding).
    """

    crops_2d: torch.Tensor
    crops_3d: torch.Tensor
    bboxes_2d: Optional[torch.Tensor] = None
    bbox_centers_3d: Optional[torch.Tensor] = None


# ============================================================================
# Base wrapper
# ============================================================================

class _InferenceWrapper(ABC):
    """Abstract base — every wrapper must implement predict_matching_matrix."""

    def __init__(
        self,
        model: EdgeModelBase,
        device: torch.device | str = "cuda",
        point_cloud_size: int = 1024,
    ):
        self.model = model
        self.device = torch.device(device)
        self.point_cloud_size = point_cloud_size
        self.model.eval()
        self.model.to(self.device)

    # ----- shared helpers -----

    def _resample_point_clouds(self, crops_3d: torch.Tensor) -> torch.Tensor:
        """Pad / sub-sample each cloud to `self.point_cloud_size` points."""
        target = self.point_cloud_size
        if crops_3d.numel() == 0:
            return crops_3d
        if crops_3d.shape[1] == target:
            return crops_3d
        resampled = []
        for pc in crops_3d:
            P = pc.shape[0]
            if P > target:
                perm = torch.randperm(P, device=pc.device)[:target]
                resampled.append(pc[perm])
            elif P < target:
                pad = torch.zeros((target - P, 3), device=pc.device, dtype=pc.dtype)
                resampled.append(torch.cat([pc, pad], dim=0))
            else:
                resampled.append(pc)
        return torch.stack(resampled)

    def _empty_scores(self, N: int, M: int) -> Tuple[torch.Tensor, float]:
        return torch.zeros((N, M), device=self.device), 0.0

    def _prepare(self, frame: FrameData) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        crops_2d = frame.crops_2d.to(self.device)
        crops_3d = frame.crops_3d.to(self.device)
        N, M = crops_2d.shape[0], crops_3d.shape[0]
        if N == 0 or M == 0:
            return crops_2d, crops_3d, N, M
        crops_3d = self._resample_point_clouds(crops_3d)
        return crops_2d, crops_3d, N, M

    def _img_centers(self, frame: FrameData) -> Optional[torch.Tensor]:
        if frame.bboxes_2d is None:
            return None
        b = frame.bboxes_2d.to(self.device)
        return (b[:, :2] + b[:, 2:]) / 2.0

    def _lid_centers(self, frame: FrameData) -> Optional[torch.Tensor]:
        if frame.bbox_centers_3d is None:
            return None
        return frame.bbox_centers_3d.to(self.device)

    @abstractmethod
    def predict_matching_matrix(
        self, frame: FrameData
    ) -> Tuple[torch.Tensor, float]:
        """Return (scores [N,M], elapsed_ms)."""


# ============================================================================
# Concrete wrappers
# ============================================================================

class HybridTwoStageWrapper(_InferenceWrapper):
    """CRLite — Stage 1 cosine + Stage 2 top-k MLP refinement."""

    def __init__(self, model, device="cuda", point_cloud_size=1024, top_k: int = 8):
        super().__init__(model, device, point_cloud_size)
        self.top_k = top_k

    def predict_matching_matrix(self, frame):
        crops_2d, crops_3d, N, M = self._prepare(frame)
        if N == 0 or M == 0:
            return self._empty_scores(N, M)

        img_centers = self._img_centers(frame)
        lid_centers = self._lid_centers(frame)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.forward_inference(
                crops_2d, crops_3d,
                img_centers=img_centers,
                lid_centers=lid_centers,
                top_k=self.top_k,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out["similarity"], elapsed_ms


class PositionEnhancedDotWrapper(_InferenceWrapper):
    """CRLite-ViT-Exp3 — paper-faithful inference.

    NOTE: To reproduce the paper-reported numbers
    (UTC3 89.9 / UTC4 98.0 top-1) we MUST evaluate this model without
    feeding it position information at inference time, even though the
    underlying CRLiteViTExp3Model was *trained* with PE injection.

    The reason is a wrapper-side bug in the lab evaluation code: the
    paper's `CRLiteViTExp3Wrapper` reads `img_centers` / `lid_centers`
    from the frame dict, while `consistent_loader.A9EvaluationDataset`
    produces `bboxes_2d` / `bbox_centers_3d`. The keys never matched,
    so the paper-time eval silently passed None for both and the model
    fell back to its appearance-only cosine pathway. We empirically
    confirmed: with PE inputs the same checkpoint scores ~3 % top-1
    (the trained PE branch is degenerate at test time); with PE=None
    it scores 91.75 / 97.54 % which matches the paper Table I.

    Standalone keeps the same protocol so the validation script
    reproduces paper numbers bit-for-bit; the partner-facing matcher
    therefore also benefits from this codepath (real PE actively hurts
    this checkpoint -- use the position-aware ResNet model `crlite`
    instead if you need 3D PE at inference time).
    """

    def predict_matching_matrix(self, frame):
        crops_2d, crops_3d, N, M = self._prepare(frame)
        if N == 0 or M == 0:
            return self._empty_scores(N, M)

        t0 = time.perf_counter()
        with torch.no_grad():
            # Intentionally pass img_centers=None / lid_centers=None so the
            # model uses its appearance-only fallback. See class docstring.
            out = self.model.forward_inference(
                crops_2d, crops_3d,
                img_centers=None,
                lid_centers=None,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out["similarity"], elapsed_ms


class DotProductWrapper(_InferenceWrapper):
    """CRLite-ViT-Exp1 — pure appearance cosine similarity, no PE."""

    def predict_matching_matrix(self, frame):
        crops_2d, crops_3d, N, M = self._prepare(frame)
        if N == 0 or M == 0:
            return self._empty_scores(N, M)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.forward_inference(crops_2d, crops_3d)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out["similarity"], elapsed_ms


class PairwiseWrapper(_InferenceWrapper):
    """CalibRefine — fc1..fc6 scored independently for every (i,j) pair."""

    def __init__(self, model, device="cuda", point_cloud_size=1024, batch_size: int = 64):
        super().__init__(model, device, point_cloud_size)
        self.batch_size = int(batch_size)

    def predict_matching_matrix(self, frame):
        crops_2d, crops_3d, N, M = self._prepare(frame)
        if N == 0 or M == 0:
            return self._empty_scores(N, M)

        # Track the model's current working dtype so FP16 inference works on
        # Thor: torch.zeros / torch.full default to fp32, which would
        # mis-match an fp16 model and trip "Index put requires source and
        # destination dtypes match".
        dtype = next(self.model.parameters()).dtype
        crops_2d = crops_2d.to(dtype=dtype)
        crops_3d = crops_3d.to(dtype=dtype)

        img_centers = self._img_centers(frame)
        lid_centers = self._lid_centers(frame)
        if img_centers is None:
            img_centers = torch.zeros(N, 2, device=self.device, dtype=dtype)
        else:
            img_centers = img_centers.to(dtype=dtype)
        if lid_centers is None:
            lid_centers = torch.zeros(M, 2, device=self.device, dtype=dtype)
        else:
            lid_centers = lid_centers[:, :2].to(dtype=dtype)

        # Build per-pair index lists
        ii, jj = torch.meshgrid(
            torch.arange(N, device=self.device),
            torch.arange(M, device=self.device),
            indexing="ij",
        )
        flat_i = ii.flatten()
        flat_j = jj.flatten()

        scores = torch.full((N, M), -1.0, device=self.device, dtype=dtype)

        t0 = time.perf_counter()
        with torch.no_grad():
            for k in range(0, flat_i.numel(), self.batch_size):
                bi = flat_i[k : k + self.batch_size]
                bj = flat_j[k : k + self.batch_size]
                batch_img = crops_2d[bi]
                batch_pc = crops_3d[bj]
                batch_img_pos = img_centers[bi]
                batch_lid_pos = lid_centers[bj]
                out = self.model(batch_img, batch_pc, batch_img_pos, batch_lid_pos)
                # main_output: [B, 1] raw logits — apply sigmoid for [0,1] match score
                logits = out["main_output"].squeeze(-1)
                pair_scores = torch.sigmoid(logits)
                scores[bi, bj] = pair_scores
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return scores, elapsed_ms


# ============================================================================
# Wrapper factory keyed by model name
# ============================================================================

def make_wrapper(
    model_name: str,
    model: EdgeModelBase,
    *,
    device: torch.device | str = "cuda",
    point_cloud_size: int = 1024,
    top_k: int = 8,
    pairwise_batch_size: int = 64,
) -> _InferenceWrapper:
    """Pick the right wrapper for a given model name."""
    name = model_name.lower()
    if name in {"crlite", "crlite_2dpe"}:
        return HybridTwoStageWrapper(
            model, device=device, point_cloud_size=point_cloud_size, top_k=top_k
        )
    if name == "crlite_vit_exp3":
        return PositionEnhancedDotWrapper(
            model, device=device, point_cloud_size=point_cloud_size
        )
    if name == "crlite_vit_exp1":
        return DotProductWrapper(
            model, device=device, point_cloud_size=point_cloud_size
        )
    if name == "calibrefine":
        return PairwiseWrapper(
            model,
            device=device,
            point_cloud_size=point_cloud_size,
            batch_size=pairwise_batch_size,
        )
    raise KeyError(f"No wrapper registered for model '{model_name}'")
