"""
OneShotSession — the projection-matrix-supervised continual-learning loop.

    frame -> match -> confident pairs -> calibrate (PnP/RANSAC -> P = K[R|t])
          -> project 3D boxes -> pseudo match matrix + geometric confidence
          -> confidence gate (reward) -> FeatureBank
          -> adapt(): confidence-weighted InfoNCE on the adapters (+ replay)
          -> updated weights -> save_pretrained() / build("onnx")

Typical usage::

    session = matcher.oneshot(intrinsics=K)

    for image, pc, b2, b3 in stream:
        report = session.observe(image, pc, b2, b3)
        if not session.is_calibrated and len(session.calib_session) >= 12:
            session.calibrate()

    session.adapt(steps=100)            # confidence-gated adapter update
    matcher.save_pretrained("adapted/") # round-trips through from_pretrained
    matcher.build("onnx")               # adapters folded into the graph
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ..engine.wrappers import FrameData
from ..data.crops import prepare_frame
from ..protocol import CameraIntrinsics
from .adapter import AdaptedModel, weighted_infonce
from .calibration import CalibrationResult, CalibrationSession, bbox3d_centers
from .memory import FeatureBank
from .pseudo_labels import pseudo_labels_for_frame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..engine.matcher import Matcher

__all__ = ["OneShotSession", "ObserveReport", "AdaptReport"]

#: Embedding-style models that support adapter-based one-shot updates.
SUPPORTED_MODELS = ("crlite", "crlite_2dpe", "crlite_vit_exp1", "crlite_vit_exp3")

#: Named head modules unfrozen by adapt(mode="head"), when the model has them.
_HEAD_ATTRS = (
    "img_fusion", "lid_fusion",
    "img_position_proj", "lid_position_proj",
    "img_proj", "img_backbone_proj",
)


@dataclass
class ObserveReport:
    """What one observe() call contributed."""

    frame_id: int
    n_detections_2d: int = 0
    n_detections_3d: int = 0
    n_confident_matches: int = 0     # harvested into the calibration buffer
    calibrated: bool = False         # was a projection matrix available?
    n_pseudo_pairs: int = 0          # geometry-confirmed pairs this frame
    n_banked: int = 0                # pairs accepted into the FeatureBank
    mean_geometric_confidence: float = 0.0
    bank_size: int = 0


@dataclass
class AdaptReport:
    """What one adapt() call did to the weights."""

    mode: str
    steps: int
    n_bank_pairs: int
    loss_first: float
    loss_last: float
    top1_before: float               # in-bank retrieval top-1, pre-update
    top1_after: float                # same, post-update
    elapsed_s: float = 0.0
    losses: List[float] = field(default_factory=list)


class OneShotSession:
    """Continual adaptation driven by the camera-LiDAR projection matrix.

    Creating a session wraps ``matcher.model`` with identity-initialised
    embedding adapters (`AdaptedModel`) — a behavioural no-op until
    :meth:`adapt` trains them. The base backbone stays frozen in the
    default mode, so the session can never destroy paper-grade weights;
    everything it learns lives in the adapters (plus, optionally, the
    fusion heads with ``mode="head"``).
    """

    def __init__(
        self,
        matcher: "Matcher",
        intrinsics: CameraIntrinsics,
        *,
        match_threshold: float = 0.7,
        accept_confidence: float = 0.5,
        max_center_dist_px: float = 64.0,
        bank_capacity: int = 4096,
        calib_buffer: int = 500,
        calib_reproj_px: float = 8.0,
        replay_frames: int = 16,
        seed: int = 0,
    ):
        if matcher.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"One-shot adaptation supports {SUPPORTED_MODELS}; "
                f"'{matcher.model_name}' has no shared embedding space."
            )
        self.matcher = matcher
        self.intrinsics = intrinsics
        self.match_threshold = float(match_threshold)
        self.accept_confidence = float(accept_confidence)
        self.max_center_dist_px = float(max_center_dist_px)

        # Attach identity adapters (no behaviour change until adapt()).
        if not isinstance(matcher.model, AdaptedModel):
            matcher.model = AdaptedModel.wrap(matcher.model)
            matcher.model.to(matcher.device)
            matcher._rebuild_wrapper()

        self.calib_session = CalibrationSession(
            min_score=match_threshold,
            max_pairs=calib_buffer,
            ransac_reproj_px=calib_reproj_px,
        )
        self.bank = FeatureBank(capacity=bank_capacity, seed=seed)
        self.calibration: Optional[CalibrationResult] = None
        #: Lowest full-buffer reprojection error (px) seen across solves — the
        #: gate reference for the degenerate-pose rejection in `calibrate`.
        self.best_reproj_px: float = float("inf")

        self._replay: Deque[Tuple[FrameData, List[Tuple[int, int, float]]]] = deque(
            maxlen=int(replay_frames)
        )
        self._rng = np.random.default_rng(seed)
        self._frame_count = 0

    # ----- convenience -----

    @property
    def model(self) -> AdaptedModel:
        return self.matcher.model

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None and self.calibration.success

    # ----- the loop -----

    def observe(
        self,
        image: np.ndarray,
        point_cloud: np.ndarray,
        bboxes_2d: np.ndarray,
        bboxes_3d: np.ndarray,
        *,
        validate: str = "warn",
    ) -> ObserveReport:
        """Ingest one frame: match, buffer calibration pairs, harvest
        geometry-confirmed features into the bank (when calibrated)."""
        b2 = np.asarray(bboxes_2d, dtype=np.float32).reshape(-1, 4)
        b3 = np.asarray(bboxes_3d, dtype=np.float32).reshape(-1, 6)
        report = ObserveReport(
            frame_id=self._frame_count,
            n_detections_2d=int(b2.shape[0]),
            n_detections_3d=int(b3.shape[0]),
            calibrated=self.is_calibrated,
        )

        result = self.matcher.match(
            image, point_cloud, b2, b3,
            top_k=1, match_threshold=0.0, validate=validate,
        )
        centers_3d = bbox3d_centers(b3)
        report.n_confident_matches = self.calib_session.add_matches(
            b2, centers_3d, result.matches, min_score=self.match_threshold
        )

        if self.is_calibrated and b2.shape[0] and b3.shape[0]:
            h, w = image.shape[:2]
            labels = pseudo_labels_for_frame(
                self.calibration, b2, centers_3d,
                image_size=(int(w), int(h)),
                max_center_dist_px=self.max_center_dist_px,
            )
            accepted = [
                (i, j, c) for (i, j, c) in labels.pairs
                if c >= self.accept_confidence
            ]
            report.n_pseudo_pairs = len(accepted)
            if accepted:
                report.n_banked, report.mean_geometric_confidence = (
                    self._bank_frame(image, point_cloud, b2, b3, accepted)
                )

        report.bank_size = len(self.bank)
        self._frame_count += 1
        return report

    def calibrate(
        self,
        *,
        min_pairs: int = 12,
        disambiguate: bool = True,
        gate_factor: float = 2.0,
        min_frames: int = 3,
    ) -> CalibrationResult:
        """Solve / refresh the projection matrix from the buffered pairs.

        With ``disambiguate=True`` (default since 0.2), the solve is scored over
        *all* buffered pairs and only adopted as ``self.calibration`` if it is
        not a degenerate planar-pose "clump". Roadside scenes are nearly planar,
        so PnP is bistable: the clump pose has a deceptively low *inlier* error
        but a high full-buffer error (:meth:`CalibrationSession.reprojection_error`).
        The gate is **accept-latest within ``gate_factor`` × the best full-buffer
        error seen**, after a ``min_frames`` warm-up — so the calibration keeps
        refining as objects sweep the scene while rejecting the clump. The
        warm-up only applies to the streaming :meth:`observe` flow; a buffer fed
        directly (e.g. via ``calib_session.add_correspondences``) is trusted.
        Pass ``disambiguate=False`` for the pre-0.2 behaviour (adopt any
        successful solve).

        Returns the solve result. ``result.accepted`` reports whether it became
        the session calibration and ``result.buffer_reproj_px`` its full-buffer
        score; a rejected solve leaves the previous ``self.calibration`` intact.
        """
        result = self.calib_session.solve(self.intrinsics, min_pairs=min_pairs)
        if not result.success:
            return result

        if not disambiguate:
            self.calibration = result
            return result

        buffer_px = self.calib_session.reprojection_error(result, reduce="median")
        result.buffer_reproj_px = buffer_px
        # Track the running best (the clump never holds the minimum), then
        # accept the latest solve unless it is a gross outlier versus it. The
        # warm-up applies only to the streaming observe() flow (where the first
        # frame or two is degenerate); when the buffer is fed directly — no
        # frames observed — we trust the caller and let the gate alone decide.
        self.best_reproj_px = min(self.best_reproj_px, buffer_px)
        warming_up = 0 < self._frame_count < min_frames
        accept = (not warming_up) and buffer_px <= gate_factor * self.best_reproj_px
        result.accepted = bool(accept)
        if accept:
            self.calibration = result
        else:
            logger.info(
                f"calibrate: rejected degenerate pose "
                f"(buffer-median {buffer_px:.1f}px vs best {self.best_reproj_px:.1f}px)"
            )
        return result

    def set_calibration(self, calibration: CalibrationResult) -> None:
        """Inject a known-good calibration (e.g. surveyed extrinsics)."""
        if not calibration.success:
            raise ValueError("calibration.success is False")
        self.calibration = calibration

    def adapt(
        self,
        steps: int = 50,
        *,
        lr: float = 1e-3,
        batch_size: int = 256,
        temperature: float = 0.07,
        mode: str = "adapter",
        replay_weight: float = 1.0,
        min_bank: int = 8,
    ) -> AdaptReport:
        """Confidence-gated weight update from the FeatureBank.

        - ``mode="adapter"`` (default): trains only the residual linear
          adapters on banked base embeddings — fast (no backbone forward),
          safe (backbone untouched), and ONNX-exportable.
        - ``mode="head"``: additionally unfreezes the base model's fusion /
          projection heads and replays full forwards over recent frames
          (`replay_frames`) so the heads receive gradients. Stronger, but
          slower and slightly riskier — keep the paper checkpoint backed up.
        """
        if mode not in ("adapter", "head"):
            raise ValueError(f"mode must be 'adapter' or 'head', got {mode!r}")
        if len(self.bank) < max(2, min_bank):
            raise RuntimeError(
                f"FeatureBank has {len(self.bank)} pair(s); need >= "
                f"{max(2, min_bank)}. observe() more frames after calibrate()."
            )

        model = self.model
        device = self.matcher.device
        dtype = next(model.parameters()).dtype

        params: List[nn.Parameter] = list(model.img_adapter.parameters()) + list(
            model.lid_adapter.parameters()
        )
        head_modules: List[nn.Module] = []
        if mode == "head":
            for attr in _HEAD_ATTRS:
                m = getattr(model.base, attr, None)
                if isinstance(m, nn.Module):
                    head_modules.append(m)
            if head_modules:
                params += [p for m in head_modules for p in m.parameters()]
            else:
                logger.warning(
                    f"{self.matcher.model_name} has no unfreezable head "
                    "modules; mode='head' degrades to adapter+replay."
                )

        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW(params, lr=lr)

        top1_before = self._bank_top1()
        losses: List[float] = []
        t0 = time.perf_counter()

        for _ in range(int(steps)):
            batch = self.bank.sample(batch_size)
            img = torch.from_numpy(batch["img"]).to(device=device, dtype=dtype)
            lid = torch.from_numpy(batch["lid"]).to(device=device, dtype=dtype)
            conf = torch.from_numpy(batch["confidence"]).to(device=device, dtype=dtype)

            with torch.enable_grad():
                loss = weighted_infonce(
                    model.img_adapter(img), model.lid_adapter(lid),
                    conf, temperature=temperature,
                )
                if mode == "head" and self._replay:
                    replay_loss = self._replay_loss(temperature)
                    if replay_loss is not None:
                        loss = loss + replay_weight * replay_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))

        for p in params:
            p.requires_grad_(False)
        model.eval()

        top1_after = self._bank_top1()
        report = AdaptReport(
            mode=mode,
            steps=int(steps),
            n_bank_pairs=len(self.bank),
            loss_first=losses[0],
            loss_last=losses[-1],
            top1_before=top1_before,
            top1_after=top1_after,
            elapsed_s=time.perf_counter() - t0,
            losses=losses,
        )
        logger.info(
            f"adapt[{mode}] {steps} steps on {len(self.bank)} banked pairs: "
            f"loss {report.loss_first:.4f} -> {report.loss_last:.4f}, "
            f"bank top-1 {top1_before:.3f} -> {top1_after:.3f}"
        )
        return report

    # ----- persistence -----

    def save(self, save_dir: str | Path) -> Dict[str, Path]:
        """Persist weights (+adapters), feature bank, and calibration."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self.matcher.save_pretrained(save_dir)
        paths["feature_bank"] = self.bank.save(save_dir / "feature_bank.npz")
        if self.is_calibrated:
            calib_path = save_dir / "calibration.npz"
            self.calibration.save(calib_path)
            paths["calibration"] = calib_path
        return paths

    # ----- internals -----

    def _bank_frame(
        self,
        image: np.ndarray,
        point_cloud: np.ndarray,
        b2: np.ndarray,
        b3: np.ndarray,
        accepted: List[Tuple[int, int, float]],
    ) -> Tuple[int, float]:
        """Embed the frame's crops and store the accepted pseudo-pairs."""
        frame, kept2, kept3 = prepare_frame(
            image=image,
            point_cloud=np.asarray(point_cloud, dtype=np.float32),
            bboxes_2d=b2,
            bboxes_3d=b3,
            cfg=self.matcher.prepare_cfg,
        )
        if frame.crops_2d.shape[0] == 0 or frame.crops_3d.shape[0] == 0:
            return 0, 0.0

        pos2 = {int(orig): k for k, orig in enumerate(kept2)}
        pos3 = {int(orig): k for k, orig in enumerate(kept3)}
        pairs_kept = [
            (pos2[i], pos3[j], c)
            for (i, j, c) in accepted
            if i in pos2 and j in pos3
        ]
        if not pairs_kept:
            return 0, 0.0

        img_emb, lid_emb = self._base_embeddings(frame)
        ii = np.array([p[0] for p in pairs_kept], dtype=np.int64)
        jj = np.array([p[1] for p in pairs_kept], dtype=np.int64)
        cc = np.array([p[2] for p in pairs_kept], dtype=np.float32)

        n_banked = self.bank.add(img_emb[ii], lid_emb[jj], cc, frame_id=self._frame_count)
        self._replay.append((frame, pairs_kept))
        return n_banked, float(cc.mean())

    def _model_inputs(
        self, frame: FrameData
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Family-faithful extract_features inputs (mirrors make_wrapper)."""
        device = self.matcher.device
        dtype = next(self.model.parameters()).dtype
        crops_2d = frame.crops_2d.to(device=device, dtype=dtype)
        crops_3d = frame.crops_3d.to(device=device, dtype=dtype)
        if self.matcher.model_name in ("crlite", "crlite_2dpe"):
            b = frame.bboxes_2d.to(device=device, dtype=dtype)
            img_centers = (b[:, :2] + b[:, 2:]) / 2.0
            lid_centers = frame.bbox_centers_3d.to(device=device, dtype=dtype)
            return crops_2d, crops_3d, img_centers, lid_centers
        # ViT cosine families run appearance-only at inference (the exp3 PE
        # branch is bypassed to match the paper protocol — see inference.py).
        return crops_2d, crops_3d, None, None

    @torch.no_grad()
    def _base_embeddings(self, frame: FrameData) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-adapter embeddings of every kept crop (what the bank stores)."""
        crops_2d, crops_3d, img_centers, lid_centers = self._model_inputs(frame)
        feats = self.model.base_features(crops_2d, crops_3d, img_centers, lid_centers)
        return (
            feats["img_embed"].detach().float().cpu().numpy(),
            feats["lid_embed"].detach().float().cpu().numpy(),
        )

    def _replay_loss(self, temperature: float) -> Optional[torch.Tensor]:
        """Full-forward InfoNCE on one stored frame (gradients flow into the
        unfrozen heads AND the adapters)."""
        frame, pairs = self._replay[int(self._rng.integers(0, len(self._replay)))]
        if len(pairs) < 2:
            return None
        crops_2d, crops_3d, img_centers, lid_centers = self._model_inputs(frame)
        feats = self.model.extract_features(crops_2d, crops_3d, img_centers, lid_centers)
        ii = torch.as_tensor([p[0] for p in pairs], device=crops_2d.device)
        jj = torch.as_tensor([p[1] for p in pairs], device=crops_2d.device)
        cc = torch.as_tensor(
            [p[2] for p in pairs], device=crops_2d.device, dtype=feats["img_embed"].dtype
        )
        return weighted_infonce(
            feats["img_embed"][ii], feats["lid_embed"][jj], cc,
            temperature=temperature,
        )

    @torch.no_grad()
    def _bank_top1(self, max_rows: int = 512) -> float:
        """In-bank retrieval top-1 with the current adapters."""
        n = len(self.bank)
        if n < 2:
            return 0.0
        data = self.bank.sample(min(n, max_rows))
        device = self.matcher.device
        dtype = next(self.model.parameters()).dtype
        img = self.model.img_adapter(
            torch.from_numpy(data["img"]).to(device=device, dtype=dtype)
        )
        lid = self.model.lid_adapter(
            torch.from_numpy(data["lid"]).to(device=device, dtype=dtype)
        )
        img = torch.nn.functional.normalize(img.float(), dim=1)
        lid = torch.nn.functional.normalize(lid.float(), dim=1)
        sim = img @ lid.t()
        pred = sim.argmax(dim=1)
        target = torch.arange(sim.shape[0], device=sim.device)
        return float((pred == target).float().mean())
