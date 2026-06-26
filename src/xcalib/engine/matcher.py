"""
Partner-facing API.

A single class — `Matcher` — that hides cropping, point-cloud
resampling, two-stage inference, and device placement. It is what the
edge-device integration target imports from `xcalib`.

Typical usage::

    from xcalib import Matcher

    # From Hugging Face Hub — released weights resolve by (model, site)
    # convention; Hugging Face handles access checks when needed:
    matcher = Matcher.from_pretrained(model="crlite", site="a9_dataset_r02_s01")

    # Or from local files:
    matcher = Matcher.from_pretrained(
        model="crlite",
        weights="checkpoints/crlite_utc4_best.pth",
        config="configs/crlite_utc4.yaml",
        device="cuda",
    )

    result = matcher.match(image, point_cloud, bboxes_2d, bboxes_3d, top_k=5)
    print(result.similarity)    # K x M float32 in [0, 1]
    print(result.top_indices)   # K   int64

    matcher.build("onnx", output_dir="onnx/crlite_utc4")   # any weights
    calib = matcher.calibrate(image, point_cloud, bboxes_2d, bboxes_3d,
                              intrinsics=K)                # PnP/RANSAC
    session = matcher.oneshot(intrinsics=K)                # continual loop
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from ..data.crops import PrepareConfig, prepare_frame
from ..models.registry import build_model, list_models
from ..protocol import CameraIntrinsics, enforce, validate_frame_inputs
from ..utils.config import EdgeConfig, load_yaml
from ..utils.io import (
    default_config_path,
    extract_state_dict,
    load_checkpoint,
    resolve_device,
)
from .wrappers import FrameData, make_wrapper

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..oneshot.calibration import CalibrationResult
    from ..oneshot.session import OneShotSession
    from .exporter import BuildResult


@dataclass
class MatchResult:
    """Container for matcher.match() output."""
    similarity: np.ndarray           # [K, M] float32 in [-1, 1] or [0, 1] (model-dependent)
    top_indices: np.ndarray          # [K] int64 — argmax LiDAR per image bbox
    matches: List[Tuple[int, int, float]] = field(default_factory=list)
    kept_2d_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    kept_3d_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    latency_ms: float = 0.0
    model: str = ""
    device: str = ""


class Matcher:
    """High-level wrapper around any of the xcalib models."""

    def __init__(
        self,
        model_name: str,
        config: EdgeConfig,
        weights_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device, None] = None,
    ):
        if model_name not in list_models():
            raise KeyError(
                f"Unknown model '{model_name}'. Known: {', '.join(list_models())}"
            )

        if device is not None:
            config.set("device", str(device))
        self.device = resolve_device(config.get("device", "auto"))

        self.model_name = model_name
        self.config = config
        self.model = build_model(model_name, config)
        self.model.to(self.device)

        if weights_path is None:
            weights_path = config.get("weights_path")
        if weights_path is None:
            raise ValueError(
                "weights_path must be supplied (either as an argument or via config)"
            )
        self.weights_path = Path(weights_path)
        self._load_checkpoint_with_adapters(self.weights_path)
        self.model.to(self.device)

        self.prepare_cfg = PrepareConfig(
            crop_size=int(config.get("crop_size", 32)),
            point_cloud_size=int(config.get("point_cloud_size", 1024)),
            bbox_expansion=float(config.get("bbox_expansion", 1.25)),
        )
        self._rebuild_wrapper()

        # Warmup pass on synthetic inputs to JIT/cuDNN-init the model.
        self._warmup()

    # ----- factory helper -----

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        weights: Union[str, Path, None] = None,
        config: Union[str, Path, EdgeConfig, dict, None] = None,
        device: Union[str, torch.device, None] = None,
        *,
        site: str = "a9_dataset_r02_s01",
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> "Matcher":
        """Construct a matcher from local files or the HuggingFace Hub.

        Resolution rules for ``weights``:

        - ``None``         — download ``checkpoints/{model}_{site}_best.pth``
          (+ the matching YAML when ``config`` is None) from the configured
          Hub repo. Use ``repo_id`` / ``revision`` / ``token`` only when you
          need to override the released defaults.
        - ``hf://org/repo[@rev]/path`` — explicit Hub file.
        - directory       — output of :meth:`save_pretrained`; loads
          ``<dir>/{model}.pth`` (+ ``<dir>/{model}.yaml`` if config is None).
        - anything else   — local checkpoint path (original behaviour).

        ``config`` may likewise be an ``hf://`` URI, a local path outside the
        package, an EdgeConfig, or a plain dict. If omitted for local weights,
        the packaged ``xcalib/cfg/{model}_{site}.yaml`` is used when present.
        """
        from .. import hub

        if weights is None:
            weights, hub_cfg = hub.resolve_pretrained(
                model, site, repo_id=repo_id, revision=revision, token=token
            )
            if config is None:
                config = hub_cfg
        elif hub.is_hf_uri(weights):
            weights = hub.resolve_uri(str(weights), token=token)

        weights = Path(weights)
        if weights.is_dir():
            # save_pretrained() layout: <dir>/{model}.pth + <dir>/{model}.yaml
            candidate = weights / f"{model}.pth"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"'{weights}' is a directory but does not contain "
                    f"{model}.pth (expected save_pretrained() layout)"
                )
            if config is None:
                cfg_candidate = weights / f"{model}.yaml"
                if cfg_candidate.exists():
                    config = cfg_candidate
            weights = candidate

        if isinstance(config, str) and hub.is_hf_uri(config):
            config = hub.resolve_uri(config, token=token)

        if config is None:
            try:
                cfg = load_yaml(default_config_path(model, site))
            except FileNotFoundError:
                cfg = EdgeConfig({"model": model})
        elif isinstance(config, EdgeConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = EdgeConfig(config)
        else:
            cfg = load_yaml(config)
        cfg.set("weights_path", str(weights))
        return cls(model_name=model, config=cfg, weights_path=weights, device=device)

    # ----- inference -----

    @torch.no_grad()
    def match(
        self,
        image: np.ndarray,
        point_cloud: np.ndarray,
        bboxes_2d: np.ndarray,
        bboxes_3d: np.ndarray,
        top_k: int = 1,
        match_threshold: float = 0.0,
        return_latency: bool = False,
        validate: str = "warn",
    ) -> MatchResult:
        """Run the full pipeline on a single frame.

        Args:
            image:         [H, W, 3] uint8 RGB.
            point_cloud:   [P, 3+] float32 (X, Y, Z[, intensity ...]).
            bboxes_2d:     [K, 4] image bboxes (x1, y1, x2, y2).
            bboxes_3d:     [M, 6] LiDAR bboxes either
                           (xmin,ymin,zmin,xmax,ymax,zmax) or
                           (cx, cy, cz, dx, dy, dz).
            top_k:         How many top matches to keep in `matches` per image.
            match_threshold: Minimum similarity to include in `matches`.
            return_latency: Force the forward-pass timing into the result.
            validate:      Input-protocol policy — "strict" raises on any
                           violation, "warn" (default) raises on hard errors
                           and logs soft warnings once, "off" skips checks.
                           See docs/protocol.md.
        """
        if validate != "off":
            enforce(
                validate_frame_inputs(image, point_cloud, bboxes_2d, bboxes_3d),
                mode=validate,
            )

        bboxes_2d = np.asarray(bboxes_2d, dtype=np.float32)
        bboxes_3d = np.asarray(bboxes_3d, dtype=np.float32)

        t_total = time.perf_counter()
        frame, kept_2d, kept_3d = prepare_frame(
            image=image,
            point_cloud=np.asarray(point_cloud, dtype=np.float32),
            bboxes_2d=bboxes_2d,
            bboxes_3d=bboxes_3d,
            cfg=self.prepare_cfg,
        )

        scores_t, fwd_ms = self._wrapper.predict_matching_matrix(frame)
        scores = scores_t.detach().to("cpu").numpy()

        # top indices (argmax per image row)
        if scores.size == 0:
            top_idx = np.zeros((0,), dtype=np.int64)
        else:
            top_idx = scores.argmax(axis=1).astype(np.int64)

        # explicit (img, lid, score) triples above threshold
        matches: List[Tuple[int, int, float]] = []
        if scores.size:
            for i in range(scores.shape[0]):
                row = scores[i]
                order = np.argsort(-row)[: max(1, top_k)]
                for j in order:
                    s = float(row[j])
                    if s >= match_threshold:
                        # Map cropping-survived indices back to the user's
                        # original bbox numbering.
                        orig_i = int(kept_2d[i]) if kept_2d.size else i
                        orig_j = int(kept_3d[j]) if kept_3d.size else int(j)
                        matches.append((orig_i, orig_j, s))

        total_ms = (time.perf_counter() - t_total) * 1000.0
        return MatchResult(
            similarity=scores,
            top_indices=top_idx,
            matches=matches,
            kept_2d_indices=kept_2d,
            kept_3d_indices=kept_3d,
            latency_ms=fwd_ms if return_latency else total_ms,
            model=self.model_name,
            device=str(self.device),
        )

    def pair(self, *args, **kwargs) -> MatchResult:
        """Alias of :meth:`match` — "pairing" in the partner's vocabulary."""
        return self.match(*args, **kwargs)

    # ----- frame-level entry (for paper validation) -----

    @torch.no_grad()
    def match_frame(self, frame: FrameData) -> Tuple[np.ndarray, float]:
        """Skip cropping, score a pre-prepared FrameData.

        Used by `scripts/paper/validate_paper.py` where the UTC HDF5 loader has
        already produced FrameData. Returns (scores [N,M], elapsed_ms).
        """
        scores_t, fwd_ms = self._wrapper.predict_matching_matrix(frame)
        return scores_t.detach().to("cpu").numpy(), fwd_ms

    # ----- calibration (camera-LiDAR extrinsics from matches) -----

    def calibrate(
        self,
        image: np.ndarray,
        point_cloud: np.ndarray,
        bboxes_2d: np.ndarray,
        bboxes_3d: np.ndarray,
        intrinsics: CameraIntrinsics,
        *,
        match_threshold: float = 0.5,
        ransac_reproj_px: float = 8.0,
        validate: str = "warn",
    ) -> "CalibrationResult":
        """Estimate the camera-LiDAR projection matrix from one frame.

        Runs :meth:`match`, keeps each image detection's best LiDAR match
        above ``match_threshold``, and solves PnP/RANSAC over the
        (2D bbox center, 3D bbox center) correspondences. For better
        conditioning across multiple frames use :meth:`oneshot` /
        `CalibrationSession` instead.
        """
        from ..oneshot.calibration import (
            CalibrationSession,
            bbox3d_centers,
        )

        result = self.match(
            image, point_cloud, bboxes_2d, bboxes_3d,
            top_k=1, match_threshold=match_threshold, validate=validate,
        )
        session = CalibrationSession(ransac_reproj_px=ransac_reproj_px)
        session.add_matches(
            np.asarray(bboxes_2d, dtype=np.float32),
            bbox3d_centers(np.asarray(bboxes_3d, dtype=np.float32)),
            result.matches,
            min_score=match_threshold,
        )
        return session.solve(intrinsics)

    # ----- one-shot / continual adaptation -----

    def oneshot(self, intrinsics: CameraIntrinsics, **kwargs: Any) -> "OneShotSession":
        """Open a projection-matrix-supervised one-shot learning session.

        The session observes frames, refines the camera-LiDAR projection
        matrix, harvests geometry-confirmed pseudo-pairs into a feature
        bank, and applies confidence-gated adapter updates. See
        `xcalib.oneshot.session.OneShotSession`.
        """
        from ..oneshot.session import OneShotSession

        return OneShotSession(self, intrinsics, **kwargs)

    # ----- regular training / fine-tuning -----

    def train(
        self,
        train_data: Union[str, Path],
        val_data: Union[str, Path],
        *,
        output_dir: Union[str, Path] = "runs/finetune",
        **kwargs: Any,
    ) -> Path:
        """Fine-tune the currently loaded weights on HDF5 caches.

        Thin wrapper over :func:`xcalib.engine.trainer.train` that trains
        this matcher's model in place, then reloads the best (val-MRR)
        checkpoint so the matcher immediately serves the improved weights.
        ``train_data`` / ``val_data`` accept local ``.h5`` paths or Hub
        site names. Extra keyword arguments (``epochs``, ``lr``,
        ``scheduler``, ...) are forwarded to the trainer.

        Returns:
            Path to ``best.pth`` inside *output_dir*.
        """
        from .trainer import train as run_train

        # One-shot adapters wrap the base model; the regular trainer
        # fine-tunes the base weights (the adapters stay attached and are
        # re-applied at inference time).
        base_model = self.model
        if hasattr(base_model, "base") and hasattr(base_model, "adapters_state_dict"):
            base_model = base_model.base

        best_path = run_train(
            base_model,
            train_data,
            val_data,
            model_name=self.model_name,
            config=self.config,
            output_dir=output_dir,
            **kwargs,
        )

        # Training leaves last-epoch weights in the module; load the best
        # epoch back and restore inference state on the matcher's device.
        state = extract_state_dict(load_checkpoint(best_path, device=self.device))
        base_model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        return best_path

    # ----- deployment artifacts -----

    def build(
        self,
        target: str = "onnx",
        output_dir: Union[str, Path, None] = None,
        *,
        precision: str = "fp16",
        device: Union[str, torch.device, None] = None,
        onnx_dir: Union[str, Path, None] = None,
        trtexec: Optional[str] = None,
        extra_trt_args: Tuple[str, ...] = (),
        verify: bool = True,
    ) -> "BuildResult":
        """Export the currently loaded weights to a deployment artifact.

        Works for *any* weights — pretrained, fine-tuned, or one-shot
        adapted — because it traces ``self.model`` as it currently is.

        Args:
            target:      "onnx" (any host) or "trt" (needs trtexec, i.e.
                         Jetson/JetPack or a TensorRT install).
            output_dir:  Where artifacts land. Defaults to
                         ``./onnx/<model>/`` or ``./engines/<model>/``.
            precision:   TensorRT precision ("fp32" | "fp16" | "best").
            device:      Device used for ONNX tracing (default cpu).
            onnx_dir:    For target="trt": where to look for / export the
                         intermediate ONNX graphs (default ./onnx/<model>/).
            trtexec:     Explicit trtexec path override.
            extra_trt_args: Extra flags forwarded verbatim to trtexec.
            verify:      Run the onnxruntime parity check after export.

        Returns:
            `xcalib.engine.exporter.BuildResult` with artifact paths and
            (for ONNX) the max |torch - onnx| parity per output.
        """
        from .exporter import export_onnx

        target = target.lower()
        if target == "onnx":
            out_dir = Path(output_dir) if output_dir else Path.cwd() / "onnx" / self.model_name
            export_device = resolve_device(device or "cpu")
            try:
                result = export_onnx(
                    self.model_name, self.model, self.config, out_dir,
                    device=export_device, verify=verify,
                )
            finally:
                # Export may move the model; restore the matcher's device.
                self.model.to(self.device)
                self.model.eval()
            return result

        if target in ("trt", "tensorrt", "engine"):
            from .trt import build_engines

            onnx_out = Path(onnx_dir) if onnx_dir else Path.cwd() / "onnx" / self.model_name
            engine_out = Path(output_dir) if output_dir else Path.cwd() / "engines" / self.model_name
            return build_engines(
                self.model_name,
                self.config,
                onnx_dir=onnx_out,
                engine_dir=engine_out,
                precision=precision,
                trtexec=trtexec,
                extra_args=list(extra_trt_args),
                export_if_missing=lambda: self.build(
                    "onnx", onnx_out, device=device, verify=verify
                ),
            )

        raise ValueError(f"Unknown build target {target!r}; use 'onnx' or 'trt'")

    # ----- persistence -----

    def save_pretrained(self, save_dir: Union[str, Path]) -> Dict[str, Path]:
        """Write ``<dir>/{model}.pth`` + ``<dir>/{model}.yaml``.

        The checkpoint round-trips through :meth:`from_pretrained` (pass the
        directory as ``weights``) and can be exported via :meth:`build`.
        One-shot adapter weights, when present, are stored under
        ``adapters_state_dict`` and re-attached automatically on load.
        """
        from .. import __version__

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_path = save_dir / f"{self.model_name}.pth"
        config_path = save_dir / f"{self.model_name}.yaml"

        payload: Dict[str, Any] = {
            "model_name": self.model_name,
            "package_version": __version__,
        }

        # Unwrap one-shot adapters so the base state dict keeps the original
        # key namespace (loadable by every existing script).
        base_model = self.model
        adapters_state: Optional[Mapping[str, torch.Tensor]] = None
        adapters_meta: Optional[Dict[str, Any]] = None
        if hasattr(base_model, "base") and hasattr(base_model, "adapters_state_dict"):
            adapters_state = base_model.adapters_state_dict()
            adapters_meta = base_model.adapters_meta()
            base_model = base_model.base
        payload["model_state_dict"] = base_model.state_dict()
        if adapters_state is not None:
            payload["adapters_state_dict"] = adapters_state
            payload["adapters_meta"] = adapters_meta

        torch.save(payload, weights_path)

        cfg = dict(self.config.to_dict())
        cfg["model"] = self.model_name
        cfg["weights_path"] = weights_path.name
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return {"weights": weights_path, "config": config_path}

    # ----- internals -----

    def _load_checkpoint_with_adapters(self, weights_path: Path) -> None:
        """Load model weights; re-attach one-shot adapters when present."""
        ckpt = load_checkpoint(weights_path)
        self.model.load_weights(extract_state_dict(ckpt), strict=False)

        adapters_state = ckpt.get("adapters_state_dict")
        if adapters_state:
            from ..oneshot.adapter import AdaptedModel

            self.model = AdaptedModel.from_state(
                self.model, adapters_state, meta=ckpt.get("adapters_meta")
            )

    def _rebuild_wrapper(self) -> None:
        """(Re)create the inference wrapper around the current self.model."""
        self._wrapper = make_wrapper(
            self.model_name,
            self.model,
            device=self.device,
            point_cloud_size=self.prepare_cfg.point_cloud_size,
            top_k=int(self.config.get("top_k", 5)),
        )

    def _warmup(self, runs: int = 2) -> None:
        """One synthetic forward pass to absorb CUDA / cuDNN init cost."""
        try:
            cs = self.prepare_cfg.crop_size
            ps = self.prepare_cfg.point_cloud_size
            crops_2d = torch.zeros((2, 3, cs, cs), dtype=torch.float32, device=self.device)
            crops_3d = torch.zeros((2, ps, 3), dtype=torch.float32, device=self.device)
            bbox2 = torch.tensor([[0, 0, cs, cs], [0, 0, cs, cs]], dtype=torch.float32, device=self.device)
            centers3 = torch.zeros((2, 3), dtype=torch.float32, device=self.device)
            frame = FrameData(
                crops_2d=crops_2d,
                crops_3d=crops_3d,
                bboxes_2d=bbox2,
                bbox_centers_3d=centers3,
            )
            for _ in range(runs):
                self._wrapper.predict_matching_matrix(frame)
        except Exception:
            # Warmup failures should not block usage on the partner's side.
            pass
