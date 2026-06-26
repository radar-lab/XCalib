"""
Embedding adapters — the small trainable surface for one-shot updates.

Design constraints:

1. The backbone stays frozen by default, so a few hundred pseudo-labeled
   pairs cannot destroy paper-grade weights.
2. The adapter is a residual linear layer initialised to the identity
   (zero weight/bias), so attaching it changes nothing until `adapt()`
   actually trains it.
3. `AdaptedModel` exposes the same `extract_features` / stage-2 surface as
   the base models, so the existing inference wrappers *and* the ONNX
   exporters consume it unchanged — an adapted matcher exports through
   `matcher.build("onnx")` with the adapters folded into the graph.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EmbeddingAdapter", "AdaptedModel", "weighted_infonce"]


class EmbeddingAdapter(nn.Module):
    """Residual linear adapter: ``y = x + W x + b`` with W, b zero-init."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class AdaptedModel(nn.Module):
    """Base matching model + per-modality embedding adapters.

    Supports the embedding-style families (`crlite`, `crlite_2dpe`,
    `crlite_vit_exp1`, `crlite_vit_exp3`): anything that exposes
    ``extract_features(images, pcds, img_centers=None, lid_centers=None)
    -> {"img_embed", "lid_embed"}``. The pairwise `calibrefine` baseline
    has no shared embedding space and is not adaptable.
    """

    def __init__(self, base: nn.Module, img_adapter: EmbeddingAdapter,
                 lid_adapter: EmbeddingAdapter):
        super().__init__()
        if not hasattr(base, "extract_features"):
            raise TypeError(
                f"{type(base).__name__} has no extract_features(); one-shot "
                "adaptation supports the embedding models only (crlite, "
                "crlite_2dpe, crlite_vit_exp1, crlite_vit_exp3)."
            )
        self.base = base
        self.img_adapter = img_adapter
        self.lid_adapter = lid_adapter

    # ----- construction helpers -----

    @classmethod
    def wrap(cls, base: nn.Module) -> "AdaptedModel":
        """Attach fresh identity adapters sized from the base embed dim."""
        dim = int(getattr(base, "embed_dim", 256))
        wrapped = cls(base, EmbeddingAdapter(dim), EmbeddingAdapter(dim))
        # Match the base model's device/dtype (e.g. fp16 on Thor).
        p = next(base.parameters(), None)
        if p is not None:
            wrapped.img_adapter.to(device=p.device, dtype=p.dtype)
            wrapped.lid_adapter.to(device=p.device, dtype=p.dtype)
        return wrapped

    @classmethod
    def from_state(
        cls,
        base: nn.Module,
        adapters_state: Mapping[str, torch.Tensor],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "AdaptedModel":
        """Rebuild a wrapped model from `save_pretrained` checkpoint parts."""
        dim = int(meta["dim"]) if meta and "dim" in meta else int(
            adapters_state["img_adapter.linear.weight"].shape[0]
        )
        wrapped = cls(base, EmbeddingAdapter(dim), EmbeddingAdapter(dim))
        missing, unexpected = wrapped.load_state_dict(
            {f"{k}": v for k, v in adapters_state.items()}, strict=False
        )
        # adapters_state only carries adapter keys; "missing" base keys are
        # expected (the base was loaded separately).
        unexpected_real = [k for k in unexpected]
        if unexpected_real:
            raise KeyError(f"Unexpected adapter keys: {unexpected_real}")
        p = next(base.parameters(), None)
        if p is not None:
            wrapped.img_adapter.to(device=p.device, dtype=p.dtype)
            wrapped.lid_adapter.to(device=p.device, dtype=p.dtype)
        return wrapped

    # ----- persistence helpers (consumed by Matcher.save_pretrained) --

    def adapters_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, tensor in self.img_adapter.state_dict().items():
            out[f"img_adapter.{name}"] = tensor
        for name, tensor in self.lid_adapter.state_dict().items():
            out[f"lid_adapter.{name}"] = tensor
        return out

    def adapters_meta(self) -> Dict[str, Any]:
        return {"dim": self.img_adapter.dim, "type": "residual_linear"}

    # ----- model surface (mirrors the base families) -----

    def extract_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feats = self.base.extract_features(images, pcds, img_centers, lid_centers)
        return {
            "img_embed": self.img_adapter(feats["img_embed"]),
            "lid_embed": self.lid_adapter(feats["lid_embed"]),
        }

    def base_features(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pre-adapter embeddings (what the FeatureBank stores)."""
        return self.base.extract_features(images, pcds, img_centers, lid_centers)

    @staticmethod
    def _cosine(img_embed: torch.Tensor, lid_embed: torch.Tensor) -> torch.Tensor:
        img = F.normalize(img_embed, p=2, dim=1)
        lid = F.normalize(lid_embed, p=2, dim=1)
        return img @ lid.t()

    def forward_inference(
        self,
        images: torch.Tensor,
        pcds: torch.Tensor,
        img_centers: Optional[torch.Tensor] = None,
        lid_centers: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.extract_features(images, pcds, img_centers, lid_centers)
        stage1_sim = self._cosine(features["img_embed"], features["lid_embed"])

        if not hasattr(self.base, "stage2_forward_candidates"):
            # Cosine families (vit_exp1 / vit_exp3-as-evaluated).
            return {"similarity": stage1_sim, "stage1_similarity": stage1_sim}

        # Two-stage family — replicate CRLiteModel.forward_inference with
        # adapted embeddings feeding both the retrieval and the pair MLP.
        if top_k is None:
            top_k = int(getattr(self.base, "top_k", 3))
        N, M = stage1_sim.shape
        k = min(top_k, M)
        _, top_indices = stage1_sim.topk(k, dim=1)
        stage2_scores = self.base.stage2_forward_candidates(features, top_indices)

        final_sim = torch.full(
            (N, M), -1.0, device=images.device, dtype=stage1_sim.dtype
        )
        for i in range(N):
            row = stage2_scores[i]
            rmin, rmax = row.min(), row.max()
            if rmax > rmin:
                normalized = (row - rmin) / (rmax - rmin)
            else:
                normalized = torch.full_like(row, 0.5)
            final_sim[i, top_indices[i]] = normalized

        return {
            "similarity": final_sim,
            "stage1_similarity": stage1_sim,
            "stage2_scores": stage2_scores,
            "top_indices": top_indices,
        }

    def forward(self, *args, **kwargs):
        return self.forward_inference(*args, **kwargs)

    # ----- attribute passthrough -----

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ resolves _parameters/_buffers/_modules first;
        # anything else (embed_fusion, similarity_head, top_k, config, ...)
        # falls through to the wrapped base model so existing wrappers and
        # exporters keep working.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)


def weighted_infonce(
    img_embed: torch.Tensor,
    lid_embed: torch.Tensor,
    confidence: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Confidence-weighted symmetric InfoNCE over in-batch pairs.

    Row i of the batch is a geometry-confirmed (image, LiDAR) pair; all
    other rows act as negatives. Each pair's loss contribution is scaled
    by its geometric confidence — the "reward" from the projection-matrix
    supervisor — so borderline pseudo-labels barely move the weights.
    Mirrors the row-wise CE shape used by scripts/paper/train_hdf5.py.
    """
    if img_embed.shape != lid_embed.shape:
        raise ValueError("img/lid embedding shapes disagree")
    B = img_embed.shape[0]
    if B < 2:
        raise ValueError("need >= 2 pairs for in-batch contrastive loss")

    img = F.normalize(img_embed, p=2, dim=1)
    lid = F.normalize(lid_embed, p=2, dim=1)
    sim = img @ lid.t() / temperature
    targets = torch.arange(B, device=sim.device)

    w = confidence.to(sim.dtype).clamp(min=0)
    w = w / w.sum().clamp(min=1e-8)

    ce_img = F.cross_entropy(sim, targets, reduction="none")
    ce_lid = F.cross_entropy(sim.t(), targets, reduction="none")
    return 0.5 * ((w * ce_img).sum() + (w * ce_lid).sum())
