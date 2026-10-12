"""Continual-refine subsystem: turn human-verified 2D<->3D corrections into
metric-gated model revisions.

The calibration *library* owns the continual loop: a front end (such as the Qt
review in :mod:`xcalib.gui.review`, or a host application) calls into this package
instead of reimplementing it. The modules are layered bottom-up by dependency;
``store`` (pure sqlite3) is the foundation.
"""
from __future__ import annotations

from .jobs import (
    CANCELLED_DETAIL,
    CancelUnavailable,
    ContinualJobs,
    DurableJobState,
    LeaseLost,
    RefineWorkload,
    StartResult,
    WriterLeaseConflict,
    cancel_refine_job,
    get_refine_job,
    get_refine_workload,
    recover_interrupted_runs,
    start_refine_job,
)
from .locations import ModelRef, model_dir, resolve_model
from .publication import (
    PublicationConflict,
    PublicationDivergence,
    PublicationError,
    PublicationOperation,
    commit_publication,
    list_divergent_operations,
    migrate_publications,
    prepare_publication,
    prune_revision_artifacts,
    publish_revision,
    read_operation,
    recover_publications,
    remediate_divergence,
    rollback_revision,
    switch_publication_pointer,
)
from .train import (
    ContinualRefineUnavailable,
    GateDecision,
    RefineConfig,
    RefineProgress,
    RefineResult,
    RemainingWork,
    build_training_batches,
    finalize_continual_publication,
    publication_gate,
    run_continual_refine,
)
from .refine_model import (
    XCalibArtifact,
    XCalibRevisionError,
    XCalibRevisionRef,
    resolve_xcalib_revision,
    verify_xcalib_revision,
)
from .state import (
    CachedFrame,
    ContinualState,
    FeatureWork,
    ScoreDecision,
    SupervisionSample,
    stable_split,
)
from .store import (
    ApplyBlocked,
    ApplyOperationRecord,
    CorpusCounts,
    CorrectionRecord,
    IssuedReviewRecord,
    PendingFileRecord,
    ReviewConflict,
    ReviewFrameRecord,
    ReviewNotFound,
    XCalibStore,
    open_store,
)
from .scene_loader import frame_input_fingerprint, scene_frame_loader
from .record import VerifiedPair, record_verified_pairs

__all__ = [
    # store
    "ApplyBlocked",
    "ApplyOperationRecord",
    "CorpusCounts",
    "CorrectionRecord",
    "IssuedReviewRecord",
    "PendingFileRecord",
    "ReviewConflict",
    "ReviewFrameRecord",
    "ReviewNotFound",
    "XCalibStore",
    "open_store",
    # revision seam (publication builds on these)
    "XCalibArtifact",
    "XCalibRevisionError",
    "XCalibRevisionRef",
    "resolve_xcalib_revision",
    "verify_xcalib_revision",
    # continual state (corpus schema + dataset split over the store)
    "CachedFrame",
    "ContinualState",
    "FeatureWork",
    "ScoreDecision",
    "SupervisionSample",
    "stable_split",
    # model-dir resolution
    "ModelRef",
    "model_dir",
    "resolve_model",
    # training orchestration + gates
    "ContinualRefineUnavailable",
    "GateDecision",
    "RefineConfig",
    "RefineProgress",
    "RefineResult",
    "RemainingWork",
    "build_training_batches",
    "finalize_continual_publication",
    "publication_gate",
    "run_continual_refine",
    # package-native frame loader (default scene reader for refine)
    "frame_input_fingerprint",
    "scene_frame_loader",
    # corpus-write seam (front ends without the apply journal accumulate through this)
    "VerifiedPair",
    "record_verified_pairs",
    # durable refine jobs (leases, recovery, cooperative cancel)
    "CANCELLED_DETAIL",
    "CancelUnavailable",
    "ContinualJobs",
    "DurableJobState",
    "LeaseLost",
    "RefineWorkload",
    "StartResult",
    "WriterLeaseConflict",
    "cancel_refine_job",
    "get_refine_job",
    "get_refine_workload",
    "recover_interrupted_runs",
    "start_refine_job",
    # publication journal
    "PublicationConflict",
    "PublicationDivergence",
    "PublicationError",
    "PublicationOperation",
    "commit_publication",
    "list_divergent_operations",
    "migrate_publications",
    "prepare_publication",
    "prune_revision_artifacts",
    "publish_revision",
    "read_operation",
    "recover_publications",
    "remediate_divergence",
    "rollback_revision",
    "switch_publication_pointer",
]
