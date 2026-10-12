from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Literal
import hashlib
import json

import copy
import os
import numpy as np
import re
import time
import uuid

PREPROCESSING_VERSION = "xcalib.prepare_frame.v1"
GEO_GATE_MIN_PX = 150.0
GEO_GATE_DIAG = 3.0


class XCalibUnavailable(RuntimeError):
    pass


class XCalibRevisionError(XCalibUnavailable):
    pass


class XCalibLegacyInsufficientPairs(XCalibUnavailable):
    def __init__(self, pairs_banked: int, min_pairs: int):
        self.pairs_banked = pairs_banked
        self.min_pairs = min_pairs
        super().__init__(
            f"only {pairs_banked} legacy pair(s) banked (need >= {min_pairs})")


@dataclass(frozen=True)
class XCalibRevisionRef:
    model_dir: Path
    model_name: str
    revision: str
    parent_revision: str | None
    weights_path: Path
    config_path: Path | None
    sha256: str
    source: Literal["pointer", "legacy"]


@dataclass
class XCalibHandle:
    ref: XCalibRevisionRef
    base_fingerprint: str
    uses_legacy_acceptance: bool
    _matcher: Any
    _acceptance: Any


@dataclass(frozen=True)
class XCalibFrameInput:
    image: np.ndarray
    point_cloud: np.ndarray
    bboxes_2d: np.ndarray
    bboxes_3d: np.ndarray
    rect_keys: tuple[str, ...]
    lidar_keys: tuple[str, ...]
    input_fingerprint: str


@dataclass(frozen=True)
class XCalibCalibration:
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class XCalibLegacyFrame:
    frame_input: XCalibFrameInput
    known_pairs: tuple[tuple[int, int, float], ...] | None


@dataclass(frozen=True)
class XCalibLegacyAdaptReport:
    frames_seen: int
    pairs_banked: int
    steps: int
    top1_before: float
    top1_after: float


@dataclass(frozen=True)
class XCalibBaseEmbeddings:
    rect_keys: tuple[str, ...]
    lidar_keys: tuple[str, ...]
    image_embeddings: np.ndarray
    lidar_embeddings: np.ndarray
    kept_2d_indices: np.ndarray
    kept_3d_indices: np.ndarray
    base_fingerprint: str


@dataclass(frozen=True)
class XCalibScores:
    retrieval_scores: np.ndarray
    acceptance_logits: np.ndarray | None
    acceptance_probabilities: np.ndarray
    uses_legacy_acceptance: bool


@dataclass(frozen=True)
class XCalibMatchResult:
    scores: XCalibScores
    top_indices: np.ndarray
    assignment: tuple[tuple[int, int, float], ...]
    kept_2d_indices: np.ndarray
    kept_3d_indices: np.ndarray
    geo_compatible: np.ndarray | None
    proj_px: np.ndarray | None


@dataclass(frozen=True)
class XCalibArtifact:
    revision: str
    parent_revision: str | None
    model_name: str
    directory: Path
    weights_path: Path
    config_path: Path
    sha256: str
    base_fingerprint: str


@dataclass(frozen=True)
class XCalibTrainingBatch:
    sample_keys: tuple[str, ...]
    group_keys: tuple[str, ...]
    image_embeddings: np.ndarray
    lidar_embeddings: np.ndarray
    candidate_mask: np.ndarray
    positive_index: np.ndarray
    replaced_index: np.ndarray


@dataclass(frozen=True)
class XCalibTrainConfig:
    learning_rate: float = 1e-4
    # The acceptance head is TWO scalars (log_scale, bias) that place a decision
    # threshold, not a matrix that wants gentle nudges -- it must be able to TRAVEL.
    # A legacy parent starts at log_scale=0/bias=0, i.e. "accept iff raw > 0", while
    # real raw scores sit near ~2, so bias must reach about -2 before even one no_match
    # is rejected. Two things then conspire at the adapters' 1e-4: AdamW steps ~lr, so
    # the reachable travel is ~0.2 (10x short); and `no_match_false_positive_rate` is a
    # STEP function -- pinned at exactly 1.0 until bias crosses that gap -- so the
    # validation score is dead flat while the head is still walking toward the cliff.
    # Early stopping cannot tell "converged" from "not there yet", quits on the plateau,
    # and restores the first epoch. Result: candidate == parent, "primary delta below
    # 0.005", rejected forever on any corpus size.
    # So this rate must cross a typical gap INSIDE the patience window (~18 steps/epoch
    # x patience) -- the metric jump then resets patience and training proceeds normally.
    calibration_learning_rate: float = 5e-2
    temperature: float = 0.07
    hard_margin: float = 0.2
    positive_ce_weight: float = 1.0
    positive_acceptance_weight: float = 0.25
    hard_margin_weight: float = 1.0
    no_match_weight: float = 1.0
    parent_anchor_weight: float = 1e-4
    acceptance_threshold: float = 0.5
    max_epochs: int = 20
    max_steps: int = 2000
    wall_time_seconds: float = 1800.0
    patience: int = 4
    min_delta: float = 1e-4
    seed: int = 0


@dataclass(frozen=True)
class XCalibMetrics:
    primary_score: float
    positive_top1: float
    positive_mrr: float
    positive_margin: float
    hard_negative_violation_rate: float
    no_match_false_positive_rate: float
    macro_group_score: float
    group_scores: dict[str, float]
    positive_count: int
    corrected_count: int
    no_match_count: int


@dataclass(frozen=True)
class XCalibTrainReport:
    epochs_completed: int
    optimizer_steps: int
    best_epoch: int
    stopped_reason: Literal[
        "early_stopping", "max_epochs", "max_steps", "wall_time"
    ]
    best_metrics: XCalibMetrics
    final_loss: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA_CACHE_LOCK = Lock()
_SHA_CACHE: "OrderedDict[tuple[str, int, int], str]" = OrderedDict()
_SHA_CACHE_MAX = 64


def _sha256_cached(path: Path) -> str:
    """`_sha256` memoised by (path, mtime_ns, size). `resolve_xcalib_revision` can be
    called many times for one unchanged checkpoint (for example once per request by
    a long-running caller) and would otherwise re-hash the whole checkpoint each
    call. A published revision writes a NEW weights path, so its first read misses
    and recomputes; overwriting a file changes mtime/size and also misses. The
    uncached `_sha256` is kept for publish-time integrity verification."""
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return _sha256(path)  # let the caller's own open() surface the real error
    with _SHA_CACHE_LOCK:
        cached = _SHA_CACHE.get(key)
        if cached is not None:
            _SHA_CACHE.move_to_end(key)
            return cached
    digest = _sha256(path)
    with _SHA_CACHE_LOCK:
        _SHA_CACHE[key] = digest
        while len(_SHA_CACHE) > _SHA_CACHE_MAX:
            _SHA_CACHE.popitem(last=False)
    return digest


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise XCalibRevisionError(f"revision path escapes model directory: {relative}") from exc
    if not candidate.is_file():
        raise XCalibRevisionError(f"revision artifact does not exist: {relative}")
    return candidate


def resolve_xcalib_revision(model_dir: str | Path) -> XCalibRevisionRef | None:
    try:
        root = Path(model_dir).resolve()
        pointer_path = root / "current.json"
        if pointer_path.is_file():
            try:
                payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise XCalibRevisionError(f"invalid current.json: {exc}") from exc
            if not isinstance(payload, dict):
                raise XCalibRevisionError("current.json must contain a JSON object")
            required = {
                "schema_version", "revision", "parent_revision", "model_name",
                "weights", "config", "sha256", "published_at",
            }
            if set(payload) != required:
                raise XCalibRevisionError("current.json must contain exactly the version 1 fields")
            if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
                raise XCalibRevisionError("current.json schema_version must be integer 1")
            parent_revision = payload["parent_revision"]
            if parent_revision is not None and not isinstance(parent_revision, str):
                raise XCalibRevisionError("current.json parent_revision must be a string or null")
            strings: dict[str, str] = {}
            for field in ("revision", "model_name", "weights", "config", "sha256",
                          "published_at"):
                value = payload[field]
                if not isinstance(value, str) or not value:
                    raise XCalibRevisionError(
                        f"current.json {field} must be a nonempty string")
                strings[field] = value
            weights = _inside(root, strings["weights"])
            config = _inside(root, strings["config"])
            actual = _sha256_cached(weights)
            if actual != strings["sha256"]:
                raise XCalibRevisionError(
                    f"current.json sha256 mismatch: expected {strings['sha256']}, got {actual}"
                )
            return XCalibRevisionRef(
                model_dir=root,
                model_name=strings["model_name"],
                revision=strings["revision"],
                parent_revision=parent_revision,
                weights_path=weights,
                config_path=config,
                sha256=actual,
                source="pointer",
            )
        checkpoints = sorted(root.glob("*.pth")) if root.is_dir() else []
        if not checkpoints:
            return None
        weights = checkpoints[0].resolve()
        digest = _sha256_cached(weights)
        model_name = weights.stem
        config = root / f"{model_name}.yaml"
        return XCalibRevisionRef(
            model_dir=root,
            model_name=model_name,
            revision=f"legacy-{digest[:12]}",
            parent_revision=None,
            weights_path=weights,
            config_path=config.resolve() if config.is_file() else None,
            sha256=digest,
            source="legacy",
        )
    except XCalibRevisionError:
        raise
    except Exception as exc:
        raise XCalibRevisionError(f"cannot resolve XCalib revision: {exc}") from exc


class _AcceptanceCalibration:
    def __init__(self, log_scale=None, bias=None):
        import torch
        self.log_scale = torch.nn.Parameter(
            torch.tensor(0.0) if log_scale is None else log_scale.detach().clone()
        )
        self.bias = torch.nn.Parameter(
            torch.tensor(0.0) if bias is None else bias.detach().clone()
        )

    @classmethod
    def from_checkpoint(cls, state):
        if not state:
            return cls()
        return cls(state["acceptance.log_scale"], state["acceptance.bias"])

    def state_dict(self):
        return {
            "acceptance.log_scale": self.log_scale.detach().cpu(),
            "acceptance.bias": self.bias.detach().cpu(),
        }


def _base_fingerprint(matcher, checkpoint) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(checkpoint["model_state_dict"].items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    prepare = matcher.prepare_cfg
    preprocessing = {
        "version": PREPROCESSING_VERSION,
        "crop_size": int(prepare.crop_size),
        "point_cloud_size": int(prepare.point_cloud_size),
        "bbox_expansion": float(prepare.bbox_expansion),
    }
    digest.update(json.dumps(preprocessing, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def load_xcalib_revision(
    ref: XCalibRevisionRef,
    *,
    device: str | None = None,
) -> XCalibHandle:
    try:
        from xcalib import Matcher
        import torch

        checkpoint = torch.load(ref.weights_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise XCalibRevisionError("checkpoint lacks model_state_dict")
        if checkpoint.get("model_name", ref.model_name) != ref.model_name:
            raise XCalibRevisionError("checkpoint model_name does not match current.json")
        has_state = "continual_state_dict" in checkpoint
        has_meta = "continual_meta" in checkpoint
        if has_state != has_meta:
            raise XCalibRevisionError(
                "continual_state_dict and continual_meta must appear together")
        state = None
        meta = None
        if has_state:
            state = checkpoint["continual_state_dict"]
            meta = checkpoint["continual_meta"]
            try:
                _validate_calibration_payload(state)
            except XCalibRevisionError as exc:
                raise XCalibRevisionError(f"continual {exc}") from exc
            required_meta = {"schema_version", "parent_revision", "base_fingerprint"}
            if not isinstance(meta, dict) or not required_meta.issubset(meta):
                raise XCalibRevisionError("continual metadata schema is incompatible")
            if type(meta["schema_version"]) is not int or meta["schema_version"] != 1:
                raise XCalibRevisionError("continual metadata schema is incompatible")
            if (meta["parent_revision"] is not None
                    and not isinstance(meta["parent_revision"], str)):
                raise XCalibRevisionError(
                    "continual metadata parent_revision must be a string or null")
            if meta["parent_revision"] != ref.parent_revision:
                raise XCalibRevisionError("continual parent revision mismatch")
            if (not isinstance(meta["base_fingerprint"], str)
                    or not meta["base_fingerprint"]):
                raise XCalibRevisionError(
                    "continual metadata base_fingerprint must be a nonempty string")
            adapters = checkpoint.get("adapters_state_dict")
            _validate_adapter_payload(adapters)
        matcher = Matcher.from_pretrained(
            ref.model_name, weights=ref.weights_path, config=ref.config_path,
            device=device,
        )
        base_fingerprint = _base_fingerprint(matcher, checkpoint)
        if meta is not None and meta.get("base_fingerprint") != base_fingerprint:
            raise XCalibRevisionError("continual base fingerprint mismatch")
        return XCalibHandle(
            ref=ref, base_fingerprint=base_fingerprint,
            uses_legacy_acceptance=state is None, _matcher=matcher,
            _acceptance=_AcceptanceCalibration.from_checkpoint(state),
        )
    except XCalibRevisionError:
        raise
    except Exception as exc:
        raise XCalibRevisionError(f"cannot load compatible XCalib revision: {exc}") from exc


def extract_xcalib_base_embeddings(
    handle: XCalibHandle,
    frame_input: XCalibFrameInput,
) -> XCalibBaseEmbeddings:
    """Extract deterministic, pre-adapter embeddings from the frozen base."""
    import torch
    from xcalib.data.crops import prepare_frame

    seed = int.from_bytes(
        hashlib.sha256(
            f"{PREPROCESSING_VERSION}:{frame_input.input_fingerprint}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    frame, kept_2d, kept_3d = prepare_frame(
        frame_input.image,
        frame_input.point_cloud,
        np.asarray(frame_input.bboxes_2d, dtype=np.float32),
        np.asarray(frame_input.bboxes_3d, dtype=np.float32),
        cfg=handle._matcher.prepare_cfg,
        rng=np.random.default_rng(seed),
    )
    kept_2d = np.asarray(kept_2d, dtype=np.int64)
    kept_3d = np.asarray(kept_3d, dtype=np.int64)
    model = handle._matcher.model
    base = model.base if hasattr(model, "base") else model
    device = handle._matcher.device
    with torch.no_grad():
        features = base.extract_features(
            frame.crops_2d.to(device),
            frame.crops_3d.to(device),
            img_centers=(
                (frame.bboxes_2d[:, :2] + frame.bboxes_2d[:, 2:]) / 2
            ).to(device),
            lid_centers=frame.bbox_centers_3d.to(device),
        )
    return XCalibBaseEmbeddings(
        rect_keys=tuple(frame_input.rect_keys[int(i)] for i in kept_2d),
        lidar_keys=tuple(frame_input.lidar_keys[int(i)] for i in kept_3d),
        image_embeddings=features["img_embed"].float().cpu().numpy(),
        lidar_embeddings=features["lid_embed"].float().cpu().numpy(),
        kept_2d_indices=kept_2d,
        kept_3d_indices=kept_3d,
        base_fingerprint=handle.base_fingerprint,
    )


def _score_tensors(handle: XCalibHandle, cached: XCalibBaseEmbeddings):
    import torch
    import torch.nn.functional as functional

    if cached.base_fingerprint != handle.base_fingerprint:
        raise ValueError("cached base_fingerprint does not match loaded revision")
    model = handle._matcher.model
    device = handle._matcher.device
    image = torch.as_tensor(cached.image_embeddings, device=device)
    lidar = torch.as_tensor(cached.lidar_embeddings, device=device)
    if hasattr(model, "img_adapter"):
        image = model.img_adapter(image)
        lidar = model.lid_adapter(lidar)
    features = {"img_embed": image, "lid_embed": lidar}
    base = model.base if hasattr(model, "base") else model
    if hasattr(base, "stage2_forward_all"):
        raw = base.stage2_forward_all(features)
    else:
        raw = (
            functional.normalize(image, dim=1)
            @ functional.normalize(lidar, dim=1).t()
        )
    logits = (
        handle._acceptance.log_scale.exp().to(raw.device) * raw
        + handle._acceptance.bias.to(raw.device)
    )
    return raw, logits


def score_xcalib_embeddings(
    handle: XCalibHandle,
    embeddings: XCalibBaseEmbeddings,
) -> XCalibScores:
    import torch

    with torch.no_grad():
        raw, logits = _score_tensors(handle, embeddings)
        return XCalibScores(
            retrieval_scores=raw.cpu().numpy(),
            acceptance_logits=logits.cpu().numpy(),
            acceptance_probabilities=torch.sigmoid(logits).cpu().numpy(),
            uses_legacy_acceptance=handle.uses_legacy_acceptance,
        )


def _build_wheel_calibration(
    calibration: XCalibCalibration,
    camera_intrinsics_type,
    calibration_result_type,
):
    """Build wheel values separately so matrix conversion stays unit-testable."""
    intrinsic = np.asarray(calibration.intrinsic, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(calibration.extrinsic, dtype=np.float64).reshape(4, 4)
    intrinsics = camera_intrinsics_type(
        fx=float(intrinsic[0, 0]),
        fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
    )
    # Avoid NumPy's BLAS-backed ``@`` here. On Windows the pinned torch and
    # NumPy/OpenCV OpenMP runtimes can otherwise abort during late calibration
    # construction even though this fixed 3x3-by-3x4 product is tiny.
    projection = np.empty((3, 4), dtype=np.float64)
    for row in range(3):
        for column in range(4):
            projection[row, column] = sum(
                float(intrinsic[row, inner]) * float(extrinsic[inner, column])
                for inner in range(3)
            )
    return calibration_result_type(
        success=True,
        intrinsics=intrinsics,
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        projection=projection,
        accepted=True,
    )


def _wheel_calibration(calibration: XCalibCalibration | None):
    if calibration is None:
        return None
    from xcalib import CalibrationResult, CameraIntrinsics

    return _build_wheel_calibration(
        calibration, CameraIntrinsics, CalibrationResult,
    )


def _adapt_legacy_match_result(result) -> XCalibMatchResult:
    display = np.asarray(result.similarity, dtype=np.float32)
    return XCalibMatchResult(
        scores=XCalibScores(display, None, display, True),
        top_indices=np.asarray(result.top_indices, dtype=np.int64),
        assignment=tuple(
            (int(i), int(j), float(score))
            for i, j, score in (result.assignment or ())
        ),
        kept_2d_indices=np.asarray(result.kept_2d_indices, dtype=np.int64),
        kept_3d_indices=np.asarray(result.kept_3d_indices, dtype=np.int64),
        geo_compatible=getattr(result, "geo_compatible", None),
        proj_px=getattr(result, "proj_px", None),
    )


def match_xcalib(
    handle: XCalibHandle,
    frame_input: XCalibFrameInput,
    *,
    calibration: XCalibCalibration | None = None,
    threshold: float = 0.5,
    geo_gate_min_px: float = GEO_GATE_MIN_PX,
    geo_gate_diag: float = GEO_GATE_DIAG,
) -> XCalibMatchResult:
    """Match a live frame while preserving exact legacy wheel behaviour."""
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite probability in [0, 1]") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability in [0, 1]")
    wheel_calibration = _wheel_calibration(calibration)
    if handle.uses_legacy_acceptance:
        result = handle._matcher.match(
            frame_input.image,
            frame_input.point_cloud,
            frame_input.bboxes_2d,
            frame_input.bboxes_3d,
            top_k=1,
            match_threshold=threshold,
            one_to_one=True,
            calibration=wheel_calibration,
            geo_gate_min_px=geo_gate_min_px,
            geo_gate_diag=geo_gate_diag,
        )
        return _adapt_legacy_match_result(result)

    from xcalib.assign import geometric_gate, one_to_one_assign
    from xcalib.oneshot.calibration import bbox3d_centers

    cached = extract_xcalib_base_embeddings(handle, frame_input)
    scores = score_xcalib_embeddings(handle, cached)
    raw = np.asarray(scores.retrieval_scores, dtype=np.float32)
    probabilities = np.asarray(scores.acceptance_probabilities, dtype=np.float32)
    geo_compatible = None
    proj_px = None
    if calibration is not None and raw.size:
        kept_boxes_2d = np.asarray(frame_input.bboxes_2d, dtype=np.float32)[
            cached.kept_2d_indices
        ]
        kept_centers_3d = bbox3d_centers(frame_input.bboxes_3d)[
            cached.kept_3d_indices
        ]
        geo_compatible, proj_px = geometric_gate(
            kept_boxes_2d,
            kept_centers_3d,
            calibration=wheel_calibration,
            min_px=geo_gate_min_px,
            diag_factor=geo_gate_diag,
        )

    assignment_scores = probabilities
    if geo_compatible is not None:
        assignment_scores = np.where(geo_compatible, assignment_scores, -1e9)
    kept_assignment = (
        one_to_one_assign(assignment_scores, threshold=threshold)
        if assignment_scores.size
        else []
    )
    assignment = tuple(
        (
            int(cached.kept_2d_indices[int(i)]),
            int(cached.kept_3d_indices[int(j)]),
            float(probabilities[int(i), int(j)]),
        )
        for i, j, _score in kept_assignment
        if geo_compatible is None or bool(geo_compatible[int(i), int(j)])
    )
    if raw.ndim == 2 and raw.shape[1]:
        top_indices = np.argmax(raw, axis=1).astype(np.int64)
    else:
        top_indices = np.full((raw.shape[0] if raw.ndim else 0,), -1, dtype=np.int64)
    return XCalibMatchResult(
        scores=scores,
        top_indices=top_indices,
        assignment=assignment,
        kept_2d_indices=cached.kept_2d_indices,
        kept_3d_indices=cached.kept_3d_indices,
        geo_compatible=geo_compatible,
        proj_px=proj_px,
    )


def adapt_xcalib_legacy(
    handle: XCalibHandle,
    frames: Iterable[XCalibLegacyFrame],
    *,
    calibration: XCalibCalibration,
    steps: int,
    progress: Callable[[int, int], None] | None = None,
    min_pairs: int = 0,
) -> XCalibLegacyAdaptReport:
    wheel_calibration = _wheel_calibration(calibration)
    session = handle._matcher.oneshot(intrinsics=wheel_calibration.intrinsics)
    session.set_calibration(wheel_calibration)
    frames_seen = pairs_banked = 0
    for item in frames:
        value = item.frame_input
        if item.known_pairs is None:
            report = session.observe(
                value.image,
                value.point_cloud,
                value.bboxes_2d,
                value.bboxes_3d,
            )
            bank_size = getattr(report, "bank_size", None)
            if bank_size is not None:
                pairs_banked = int(bank_size)
            else:
                pairs_banked += int(getattr(report, "n_banked", 0) or 0)
        else:
            pairs_banked += int(session.bank_pairs(
                value.image,
                value.point_cloud,
                value.bboxes_2d,
                value.bboxes_3d,
                list(item.known_pairs),
            ))
        frames_seen += 1
        if progress is not None:
            progress(frames_seen, pairs_banked)
    if pairs_banked < min_pairs:
        raise XCalibLegacyInsufficientPairs(pairs_banked, min_pairs)
    report = session.adapt(steps=steps)
    return XCalibLegacyAdaptReport(
        frames_seen,
        pairs_banked,
        report.steps,
        report.top1_before,
        report.top1_after,
    )


_ADAPTER_STATE_KEYS = {
    "img_adapter.linear.weight",
    "img_adapter.linear.bias",
    "lid_adapter.linear.weight",
    "lid_adapter.linear.bias",
}
_CALIBRATION_STATE_KEYS = {"acceptance.log_scale", "acceptance.bias"}


def _safe_artifact_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise XCalibRevisionError(f"{field} must be one safe path component")
    if value in {".", ".."}:
        raise XCalibRevisionError(f"{field} must be one safe path component")
    return value


def _named_trainable_parameters(handle: XCalibHandle) -> dict[str, Any]:
    model = handle._matcher.model
    if not hasattr(model, "img_adapter") or not hasattr(model, "lid_adapter"):
        raise XCalibRevisionError("candidate matcher does not have embedding adapters")
    named: dict[str, Any] = {}
    for prefix in ("img_adapter", "lid_adapter"):
        adapter = getattr(model, prefix)
        for name, parameter in adapter.named_parameters():
            named[f"{prefix}.{name}"] = parameter
    named["acceptance.log_scale"] = handle._acceptance.log_scale
    named["acceptance.bias"] = handle._acceptance.bias
    return named


def _build_optimizer(candidate: XCalibHandle, config: XCalibTrainConfig):
    """Two param groups on purpose (see `calibration_learning_rate`).

    The adapters are matrices warm-started at identity and want small nudges; the
    acceptance head is two scalars siting a decision threshold and has to travel. One
    shared rate serves the matrices and starves the scalars, which is what pinned
    no_match_false_positive_rate at 1.0 and made every refine reject.
    """
    import torch

    named = _named_trainable_parameters(candidate)
    adapters = [
        parameter for name, parameter in named.items()
        if name not in _CALIBRATION_STATE_KEYS
    ]
    calibration = [
        named[name] for name in sorted(_CALIBRATION_STATE_KEYS) if name in named
    ]
    # The explicit parent-anchor term is the continual regularizer.  Disabling
    # AdamW's implicit decay avoids pulling the identity-initialized residual
    # adapters away from their exact warm start for reasons outside the stated
    # objective.
    return torch.optim.AdamW(
        [
            {"params": adapters, "lr": config.learning_rate},
            {"params": calibration, "lr": config.calibration_learning_rate},
        ],
        lr=config.learning_rate,
        weight_decay=0.0,
    )


def _freeze_candidate_surface(candidate: XCalibHandle) -> None:
    model = candidate._matcher.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = model.base if hasattr(model, "base") else model
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    for adapter_name in ("img_adapter", "lid_adapter"):
        adapter = getattr(model, adapter_name)
        adapter.train()
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)
    device = candidate._matcher.device
    for name in ("log_scale", "bias"):
        parameter = getattr(candidate._acceptance, name)
        parameter.data = parameter.detach().to(device=device).clone()
        parameter.requires_grad_(True)


def clone_xcalib_adapter(parent: XCalibHandle) -> XCalibHandle:
    """Deep-clone the published handle and expose only its continual surface."""
    candidate = copy.deepcopy(parent)
    model = candidate._matcher.model
    if not hasattr(model, "img_adapter") or not hasattr(model, "lid_adapter"):
        from xcalib.oneshot.adapter import AdaptedModel

        candidate._matcher.model = AdaptedModel.wrap(model)
        rebuild = getattr(candidate._matcher, "_rebuild_wrapper", None)
        if callable(rebuild):
            rebuild()
    _freeze_candidate_surface(candidate)
    candidate.uses_legacy_acceptance = False
    return candidate


def _batch_tensors(handle: XCalibHandle, batch: XCalibTrainingBatch):
    import torch

    image = np.asarray(batch.image_embeddings, dtype=np.float32)
    lidar = np.asarray(batch.lidar_embeddings, dtype=np.float32)
    mask = np.asarray(batch.candidate_mask, dtype=bool)
    positive = np.asarray(batch.positive_index, dtype=np.int64)
    replaced = np.asarray(batch.replaced_index, dtype=np.int64)
    if image.ndim != 2 or lidar.ndim != 2 or image.shape[1:] != lidar.shape[1:]:
        raise ValueError("training embeddings must be compatible 2D matrices")
    rows, columns = image.shape[0], lidar.shape[0]
    if len(batch.sample_keys) != rows or len(batch.group_keys) != rows:
        raise ValueError("sample_keys and group_keys must match image rows")
    if mask.shape != (rows, columns):
        raise ValueError("candidate_mask shape must be [image rows, lidar rows]")
    if positive.shape != (rows,) or replaced.shape != (rows,):
        raise ValueError("positive_index and replaced_index must have one value per row")
    if np.any(positive < -1) or np.any(positive >= columns):
        raise ValueError("positive_index contains an out-of-range candidate")
    if np.any(replaced < -1) or np.any(replaced >= columns):
        raise ValueError("replaced_index contains an out-of-range candidate")
    positive_rows = np.flatnonzero(positive >= 0)
    if positive_rows.size and not np.all(mask[positive_rows, positive[positive_rows]]):
        raise ValueError("positive_index must identify a displayed candidate")
    replaced_rows = np.flatnonzero(replaced >= 0)
    if replaced_rows.size:
        if np.any(positive[replaced_rows] < 0):
            raise ValueError("a corrected proposal must also have a positive_index")
        if np.any(replaced[replaced_rows] == positive[replaced_rows]):
            raise ValueError("replaced_index must differ from positive_index")
        if not np.all(mask[replaced_rows, replaced[replaced_rows]]):
            raise ValueError("replaced_index must identify a displayed candidate")
    if not np.isfinite(image).all() or not np.isfinite(lidar).all():
        raise ValueError("training embeddings must be finite")
    device = handle._matcher.device
    return (
        torch.as_tensor(image, device=device),
        torch.as_tensor(lidar, device=device),
        torch.as_tensor(mask, device=device),
        torch.as_tensor(positive, device=device),
        torch.as_tensor(replaced, device=device),
    )


def _anchor_loss(candidate: XCalibHandle, parent: XCalibHandle, zero):
    candidate_parameters = _named_trainable_parameters(candidate)
    parent_parameters: dict[str, Any] = {}
    parent_model = parent._matcher.model
    if hasattr(parent_model, "img_adapter") and hasattr(parent_model, "lid_adapter"):
        parent_parameters.update(_named_trainable_parameters(parent))
    else:
        parent_parameters["acceptance.log_scale"] = parent._acceptance.log_scale
        parent_parameters["acceptance.bias"] = parent._acceptance.bias
    loss = zero
    for name, parameter in candidate_parameters.items():
        reference = parent_parameters.get(name)
        if reference is None:
            reference = parameter.detach().new_zeros(parameter.shape)
        else:
            reference = reference.detach().to(
                device=parameter.device, dtype=parameter.dtype
            )
        loss = loss + (parameter - reference).pow(2).mean()
    return loss


def _training_loss(
    candidate: XCalibHandle,
    parent: XCalibHandle,
    batch: XCalibTrainingBatch,
    config: XCalibTrainConfig,
):
    import torch
    import torch.nn.functional as functional

    image, lidar, candidate_mask, positive_index, replaced_index = _batch_tensors(
        candidate, batch
    )
    cached = XCalibBaseEmbeddings(
        rect_keys=batch.sample_keys,
        lidar_keys=tuple(f"candidate:{index}" for index in range(lidar.shape[0])),
        image_embeddings=image,
        lidar_embeddings=lidar,
        kept_2d_indices=np.arange(image.shape[0], dtype=np.int64),
        kept_3d_indices=np.arange(lidar.shape[0], dtype=np.int64),
        base_fingerprint=candidate.base_fingerprint,
    )
    raw, acceptance_logits = _score_tensors(candidate, cached)
    zero = raw.sum() * 0.0

    positive_rows = torch.nonzero(positive_index >= 0, as_tuple=False).flatten()
    if positive_rows.numel():
        masked_logits = (raw[positive_rows] / config.temperature).masked_fill(
            ~candidate_mask[positive_rows], torch.finfo(raw.dtype).min
        )
        positive_ce_loss = functional.cross_entropy(
            masked_logits, positive_index[positive_rows]
        )
        positive_acceptance_loss = functional.softplus(
            -acceptance_logits[positive_rows, positive_index[positive_rows]]
        ).mean()
    else:
        positive_ce_loss = zero
        positive_acceptance_loss = zero

    corrected_rows = torch.nonzero(replaced_index >= 0, as_tuple=False).flatten()
    if corrected_rows.numel():
        corrected_loss = functional.relu(
            config.hard_margin
            - raw[corrected_rows, positive_index[corrected_rows]]
            + raw[corrected_rows, replaced_index[corrected_rows]]
        ).mean()
    else:
        corrected_loss = zero

    no_match_rows = torch.nonzero(positive_index < 0, as_tuple=False).flatten()
    if no_match_rows.numel():
        no_match_values = acceptance_logits[no_match_rows][
            candidate_mask[no_match_rows]
        ]
        no_match_loss = (
            functional.softplus(no_match_values).mean()
            if no_match_values.numel()
            else zero
        )
    else:
        no_match_loss = zero

    anchor_loss = _anchor_loss(candidate, parent, zero)
    return (
        config.positive_ce_weight * positive_ce_loss
        + config.positive_acceptance_weight * positive_acceptance_loss
        + config.hard_margin_weight * corrected_loss
        + config.no_match_weight * no_match_loss
        + config.parent_anchor_weight * anchor_loss
    )


def evaluate_xcalib_adapter(
    candidate: XCalibHandle,
    batches: Iterable[XCalibTrainingBatch],
    config: XCalibTrainConfig,
) -> XCalibMetrics:
    import torch

    positive_correct: list[float] = []
    positive_reciprocal_ranks: list[float] = []
    positive_margins: list[float] = []
    corrected_violations: list[float] = []
    no_match_false_positives: list[float] = []
    group_correctness: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in batches:
            image, lidar, mask, positive, replaced = _batch_tensors(candidate, batch)
            cached = XCalibBaseEmbeddings(
                rect_keys=batch.sample_keys,
                lidar_keys=tuple(
                    f"candidate:{index}" for index in range(lidar.shape[0])
                ),
                image_embeddings=image,
                lidar_embeddings=lidar,
                kept_2d_indices=np.arange(image.shape[0], dtype=np.int64),
                kept_3d_indices=np.arange(lidar.shape[0], dtype=np.int64),
                base_fingerprint=candidate.base_fingerprint,
            )
            raw, acceptance_logits = _score_tensors(candidate, cached)
            probabilities = torch.sigmoid(acceptance_logits)
            for row, group_key in enumerate(batch.group_keys):
                valid = torch.nonzero(mask[row], as_tuple=False).flatten()
                target = int(positive[row].item())
                if target >= 0:
                    valid_scores = raw[row, valid]
                    top = int(valid[torch.argmax(valid_scores)].item())
                    correct = float(top == target)
                    positive_correct.append(correct)
                    target_score = raw[row, target]
                    rank = 1 + int((valid_scores > target_score).sum().item())
                    positive_reciprocal_ranks.append(1.0 / rank)
                    other = valid[valid != target]
                    margin = (
                        float((target_score - raw[row, other].max()).item())
                        if other.numel()
                        else 0.0
                    )
                    positive_margins.append(margin)
                    replacement = int(replaced[row].item())
                    if replacement >= 0:
                        corrected_violations.append(
                            float(
                                target_score - raw[row, replacement]
                                < config.hard_margin
                            )
                        )
                else:
                    false_positive = float(
                        bool(
                            valid.numel()
                            and torch.any(
                                probabilities[row, valid]
                                >= config.acceptance_threshold
                            ).item()
                        )
                    )
                    no_match_false_positives.append(false_positive)
                    correct = 1.0 - false_positive
                group_correctness.setdefault(group_key, []).append(correct)

    positive_top1 = (
        float(np.mean(positive_correct)) if positive_correct else 0.0
    )
    positive_mrr = (
        float(np.mean(positive_reciprocal_ranks))
        if positive_reciprocal_ranks
        else 0.0
    )
    no_match_false_positive_rate = (
        float(np.mean(no_match_false_positives))
        if no_match_false_positives
        else 0.0
    )
    primary_components: list[float] = []
    if positive_correct:
        primary_components.extend((positive_top1, positive_mrr))
    if no_match_false_positives:
        primary_components.append(1.0 - no_match_false_positive_rate)
    group_scores = {
        key: float(np.mean(values))
        for key, values in sorted(group_correctness.items())
    }
    return XCalibMetrics(
        primary_score=(
            float(np.mean(primary_components)) if primary_components else 0.0
        ),
        positive_top1=positive_top1,
        positive_mrr=positive_mrr,
        positive_margin=(
            float(np.mean(positive_margins)) if positive_margins else 0.0
        ),
        hard_negative_violation_rate=(
            float(np.mean(corrected_violations)) if corrected_violations else 0.0
        ),
        no_match_false_positive_rate=no_match_false_positive_rate,
        macro_group_score=(
            float(np.mean(list(group_scores.values()))) if group_scores else 0.0
        ),
        group_scores=group_scores,
        positive_count=len(positive_correct),
        corrected_count=len(corrected_violations),
        no_match_count=len(no_match_false_positives),
    )


def _validate_train_config(config: XCalibTrainConfig) -> None:
    numeric = asdict(config)
    for name, value in numeric.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.learning_rate < 0:
        raise ValueError("learning_rate must be nonnegative")
    if config.calibration_learning_rate < 0:
        raise ValueError("calibration_learning_rate must be nonnegative")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if config.hard_margin < 0:
        raise ValueError("hard_margin must be nonnegative")
    if not 0.0 <= config.acceptance_threshold <= 1.0:
        raise ValueError("acceptance_threshold must be in [0, 1]")
    for name in ("max_epochs", "max_steps", "patience", "seed"):
        if type(getattr(config, name)) is not int:
            raise ValueError(f"{name} must be an integer")
    if config.max_epochs < 1 or config.max_steps < 1:
        raise ValueError("max_epochs and max_steps must be positive")
    if config.wall_time_seconds < 0:
        raise ValueError("wall_time_seconds must be nonnegative")
    if config.patience < 1 or config.min_delta < 0:
        raise ValueError("patience must be positive and min_delta nonnegative")
    for name in (
        "positive_ce_weight",
        "positive_acceptance_weight",
        "hard_margin_weight",
        "no_match_weight",
        "parent_anchor_weight",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")


def _snapshot_trainable(handle: XCalibHandle) -> dict[str, Any]:
    return {
        name: parameter.detach().clone()
        for name, parameter in _named_trainable_parameters(handle).items()
    }


def _restore_trainable(handle: XCalibHandle, state: dict[str, Any]) -> None:
    parameters = _named_trainable_parameters(handle)
    if parameters.keys() != state.keys():
        raise XCalibRevisionError("best-state trainable keys changed during training")
    for name, parameter in parameters.items():
        parameter.data.copy_(state[name].to(parameter.device, parameter.dtype))


def train_xcalib_adapter(
    candidate: XCalibHandle,
    parent: XCalibHandle,
    train_batches: Callable[[int], Iterable[XCalibTrainingBatch]],
    validation_batches: Callable[[], Iterable[XCalibTrainingBatch]],
    config: XCalibTrainConfig,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> XCalibTrainReport:
    import torch

    _validate_train_config(config)
    if candidate is parent:
        raise ValueError("candidate and parent handles must be distinct")
    if candidate.base_fingerprint != parent.base_fingerprint:
        raise ValueError("candidate and parent base_fingerprint must match")
    _freeze_candidate_surface(candidate)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    optimizer = _build_optimizer(candidate, config)
    started = time.monotonic()
    optimizer_steps = 0
    epochs_completed = 0
    non_improving = 0
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_metrics: XCalibMetrics | None = None
    final_loss = 0.0
    stopped_reason: Literal[
        "early_stopping", "max_epochs", "max_steps", "wall_time"
    ] | None = None

    for epoch in range(1, config.max_epochs + 1):
        _freeze_candidate_surface(candidate)
        budget_reason: Literal["max_steps", "wall_time"] | None = None
        for batch in train_batches(epoch):
            if optimizer_steps >= config.max_steps:
                budget_reason = "max_steps"
                break
            if time.monotonic() - started >= config.wall_time_seconds:
                budget_reason = "wall_time"
                break
            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss(candidate, parent, batch, config)
            if not bool(torch.isfinite(loss).item()):
                raise XCalibRevisionError("continual training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            final_loss = float(loss.detach().cpu().item())
        epochs_completed = epoch
        metrics = evaluate_xcalib_adapter(
            candidate, validation_batches(), config
        )
        if not np.isfinite(metrics.primary_score):
            raise XCalibRevisionError("continual validation produced a non-finite score")
        if progress is not None:
            progress(epoch, config.max_epochs, asdict(metrics))
        if metrics.primary_score > best_score + config.min_delta:
            best_score = metrics.primary_score
            best_epoch = epoch
            best_metrics = metrics
            best_state = _snapshot_trainable(candidate)
            non_improving = 0
        else:
            non_improving += 1

        if budget_reason is None and optimizer_steps >= config.max_steps:
            budget_reason = "max_steps"
        if budget_reason is None and (
            time.monotonic() - started >= config.wall_time_seconds
        ):
            budget_reason = "wall_time"
        if budget_reason is not None:
            stopped_reason = budget_reason
            break
        if non_improving >= config.patience:
            stopped_reason = "early_stopping"
            break

    if stopped_reason is None:
        stopped_reason = "max_epochs"
    if best_state is None or best_metrics is None:
        best_state = _snapshot_trainable(candidate)
        best_metrics = evaluate_xcalib_adapter(
            candidate, validation_batches(), config
        )
    _restore_trainable(candidate, best_state)
    _freeze_candidate_surface(candidate)
    return XCalibTrainReport(
        epochs_completed=epochs_completed,
        optimizer_steps=optimizer_steps,
        best_epoch=best_epoch,
        stopped_reason=stopped_reason,
        best_metrics=best_metrics,
        final_loss=final_loss,
    )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def save_xcalib_revision(
    candidate: XCalibHandle,
    target_dir: str | Path,
    *,
    revision: str,
    parent_revision: str | None,
    train_config: XCalibTrainConfig,
    metrics: XCalibMetrics,
) -> XCalibArtifact:
    import torch

    _safe_artifact_component(revision, field="revision")
    model_name = _safe_artifact_component(
        str(candidate._matcher.model_name), field="model_name"
    )
    target = Path(target_dir)
    if target.exists():
        raise XCalibRevisionError(
            f"temporary revision target already exists: {target}"
        )
    candidate._matcher.save_pretrained(target)
    weights_path = target / f"{model_name}.pth"
    config_path = target / f"{model_name}.yaml"
    if not weights_path.is_file() or not config_path.is_file():
        raise XCalibRevisionError("save_pretrained did not create its wheel payload")
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise XCalibRevisionError("wheel checkpoint must be a mapping")
    saved_adapters = _validate_adapter_payload(
        payload.get("adapters_state_dict")
    )
    model_adapters = candidate._matcher.model.adapters_state_dict()
    if set(model_adapters) != _ADAPTER_STATE_KEYS:
        raise XCalibRevisionError("candidate adapter keys are incompatible")
    for key in sorted(_ADAPTER_STATE_KEYS):
        if not _tensors_match(saved_adapters[key], model_adapters[key]):
            raise XCalibRevisionError(
                f"wheel-saved adapter tensor does not match candidate: {key}"
            )
    model = candidate._matcher.model
    base = model.base if hasattr(model, "base") else model
    acceptance_source = (
        "stage2_raw" if hasattr(base, "stage2_forward_all") else "cosine"
    )
    payload["continual_state_dict"] = candidate._acceptance.state_dict()
    payload["continual_meta"] = _json_safe(
        {
            "schema_version": 1,
            "revision": revision,
            "parent_revision": parent_revision,
            "base_fingerprint": candidate.base_fingerprint,
            "acceptance_source": acceptance_source,
            "train_config": train_config,
            "metrics": metrics,
        }
    )
    json.dumps(payload["continual_meta"], sort_keys=True, allow_nan=False)
    temporary_weights = weights_path.with_name(
        f"{weights_path.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        with temporary_weights.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_weights, weights_path)
    finally:
        temporary_weights.unlink(missing_ok=True)
    return XCalibArtifact(
        revision=revision,
        parent_revision=parent_revision,
        model_name=model_name,
        directory=target,
        weights_path=weights_path,
        config_path=config_path,
        sha256=_sha256(weights_path),
        base_fingerprint=candidate.base_fingerprint,
    )


def _require_tensor_payload(state: Any, keys: set[str], label: str):
    import torch

    if not isinstance(state, dict) or set(state) != keys:
        raise XCalibRevisionError(f"{label} keys are incompatible")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise XCalibRevisionError(f"{label} value {key} is not a tensor")
        if not bool(torch.isfinite(tensor).all().item()):
            raise XCalibRevisionError(f"{label} value {key} is non-finite")
    return state


def _validate_adapter_payload(state: Any):

    state = _require_tensor_payload(state, _ADAPTER_STATE_KEYS, "adapter")
    img_weight = state["img_adapter.linear.weight"]
    img_bias = state["img_adapter.linear.bias"]
    lid_weight = state["lid_adapter.linear.weight"]
    lid_bias = state["lid_adapter.linear.bias"]
    if (
        img_weight.ndim != 2
        or img_weight.shape[0] != img_weight.shape[1]
        or img_bias.shape != (img_weight.shape[0],)
        or lid_weight.shape != img_weight.shape
        or lid_bias.shape != img_bias.shape
    ):
        raise XCalibRevisionError("adapter tensor shapes are incompatible")
    dtypes = {tensor.dtype for tensor in state.values()}
    if len(dtypes) != 1 or not all(
        tensor.dtype.is_floating_point for tensor in state.values()
    ):
        raise XCalibRevisionError("adapter tensor dtypes are incompatible")
    return state


def _validate_calibration_payload(state: Any):
    state = _require_tensor_payload(
        state, _CALIBRATION_STATE_KEYS, "calibration"
    )
    if any(tensor.shape != () for tensor in state.values()):
        raise XCalibRevisionError("calibration tensors must be scalar")
    dtypes = {tensor.dtype for tensor in state.values()}
    if len(dtypes) != 1 or not all(
        tensor.dtype.is_floating_point for tensor in state.values()
    ):
        raise XCalibRevisionError("calibration tensor dtypes are incompatible")
    return state


def _tensors_match(left: Any, right: Any) -> bool:
    import torch

    left_cpu = left.detach().contiguous().cpu()
    right_cpu = right.detach().contiguous().cpu()
    if left_cpu.shape != right_cpu.shape or left_cpu.dtype != right_cpu.dtype:
        return False
    left_bytes = left_cpu.reshape(-1).view(torch.uint8)
    right_bytes = right_cpu.reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def verify_xcalib_revision(
    artifact: XCalibArtifact, *, device: str = "cpu"
) -> XCalibArtifact:
    import torch

    try:
        _safe_artifact_component(artifact.revision, field="revision")
        _safe_artifact_component(artifact.model_name, field="model_name")
        directory = artifact.directory.resolve()
        weights_path = artifact.weights_path.resolve()
        config_path = artifact.config_path.resolve()
        if weights_path.parent != directory or config_path.parent != directory:
            raise XCalibRevisionError("artifact paths must stay inside its directory")
        if weights_path.name != f"{artifact.model_name}.pth":
            raise XCalibRevisionError("artifact weights filename is incompatible")
        if config_path.name != f"{artifact.model_name}.yaml":
            raise XCalibRevisionError("artifact config filename is incompatible")
        if not weights_path.is_file() or not config_path.is_file():
            raise XCalibRevisionError("artifact weights/config are missing")
        actual_digest = _sha256(weights_path)
        if actual_digest != artifact.sha256:
            raise XCalibRevisionError(
                f"artifact sha256 mismatch: expected {artifact.sha256}, got {actual_digest}"
            )
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise XCalibRevisionError("artifact checkpoint must be a mapping")
        if payload.get("model_name") != artifact.model_name:
            raise XCalibRevisionError("artifact model_name does not match checkpoint")
        adapters = _validate_adapter_payload(
            payload.get("adapters_state_dict")
        )
        calibration = _validate_calibration_payload(
            payload.get("continual_state_dict")
        )
        meta = payload.get("continual_meta")
        required_meta = {
            "schema_version",
            "revision",
            "parent_revision",
            "base_fingerprint",
            "acceptance_source",
            "train_config",
            "metrics",
        }
        if not isinstance(meta, dict) or not required_meta.issubset(meta):
            raise XCalibRevisionError("continual metadata schema is incompatible")
        if type(meta["schema_version"]) is not int or meta["schema_version"] != 1:
            raise XCalibRevisionError("continual metadata schema must be integer 1")
        if meta["revision"] != artifact.revision:
            raise XCalibRevisionError("continual revision does not match artifact")
        if meta["parent_revision"] != artifact.parent_revision:
            raise XCalibRevisionError("continual parent does not match artifact")
        if meta["base_fingerprint"] != artifact.base_fingerprint:
            raise XCalibRevisionError("continual base fingerprint does not match artifact")
        if meta["acceptance_source"] not in {"stage2_raw", "cosine"}:
            raise XCalibRevisionError("continual acceptance source is incompatible")
        json.dumps(_json_safe(meta), sort_keys=True, allow_nan=False)
        model_dir = (
            directory.parent.parent
            if directory.parent.name == "revisions"
            else directory.parent
        )
        ref = XCalibRevisionRef(
            model_dir=model_dir,
            model_name=artifact.model_name,
            revision=artifact.revision,
            parent_revision=artifact.parent_revision,
            weights_path=weights_path,
            config_path=config_path,
            sha256=artifact.sha256,
            source="pointer",
        )
        loaded = load_xcalib_revision(ref, device=device)
        if loaded.uses_legacy_acceptance:
            raise XCalibRevisionError("continual artifact reloaded as legacy")
        if loaded.base_fingerprint != artifact.base_fingerprint:
            raise XCalibRevisionError("reloaded base fingerprint mismatch")
        loaded_model = loaded._matcher.model
        loaded_base = (
            loaded_model.base if hasattr(loaded_model, "base") else loaded_model
        )
        actual_acceptance_source = (
            "stage2_raw"
            if hasattr(loaded_base, "stage2_forward_all")
            else "cosine"
        )
        if meta["acceptance_source"] != actual_acceptance_source:
            raise XCalibRevisionError(
                "continual acceptance source does not match reloaded base"
            )
        if not hasattr(loaded_model, "adapters_state_dict"):
            raise XCalibRevisionError("reloaded artifact has no adapters")
        reloaded_adapters = loaded_model.adapters_state_dict()
        if set(reloaded_adapters) != _ADAPTER_STATE_KEYS:
            raise XCalibRevisionError("reloaded adapter keys are incompatible")
        for key in sorted(_ADAPTER_STATE_KEYS):
            if not _tensors_match(adapters[key], reloaded_adapters[key]):
                raise XCalibRevisionError(f"reloaded adapter tensor mismatch: {key}")
        reloaded_calibration = loaded._acceptance.state_dict()
        for key in sorted(_CALIBRATION_STATE_KEYS):
            if not _tensors_match(calibration[key], reloaded_calibration[key]):
                raise XCalibRevisionError(
                    f"reloaded calibration tensor mismatch: {key}"
                )

        dimension = int(adapters["img_adapter.linear.weight"].shape[1])
        if dimension < 1:
            raise XCalibRevisionError("adapter embedding dimension must be positive")
        image = np.zeros((2, dimension), dtype=np.float32)
        lidar = np.zeros((2, dimension), dtype=np.float32)
        image[0, 0] = lidar[0, 0] = 1.0
        image[1, min(1, dimension - 1)] = 1.0
        lidar[1, min(1, dimension - 1)] = 1.0
        scores = score_xcalib_embeddings(
            loaded,
            XCalibBaseEmbeddings(
                rect_keys=("verify:image:0", "verify:image:1"),
                lidar_keys=("verify:lidar:0", "verify:lidar:1"),
                image_embeddings=image,
                lidar_embeddings=lidar,
                kept_2d_indices=np.arange(2, dtype=np.int64),
                kept_3d_indices=np.arange(2, dtype=np.int64),
                base_fingerprint=loaded.base_fingerprint,
            ),
        )
        finite_values = [scores.retrieval_scores, scores.acceptance_probabilities]
        if scores.acceptance_logits is not None:
            finite_values.append(scores.acceptance_logits)
        if not all(np.isfinite(value).all() for value in finite_values):
            raise XCalibRevisionError("reloaded artifact produced a non-finite score")
        return artifact
    except XCalibRevisionError:
        raise
    except Exception as exc:
        raise XCalibRevisionError(f"cannot verify XCalib artifact: {exc}") from exc


def save_xcalib_legacy(handle: XCalibHandle, destination: str | Path) -> None:
    handle._matcher.save_pretrained(destination)
