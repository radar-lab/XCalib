"""
Matching metrics — Top-1, Top-3, MRR, and latency aggregates.

These mirror the definitions used in
`src/paper_experiments/evaluate_matching.py::calculate_metrics` so the
numbers produced by the standalone validation script can be compared
directly against the paper's main table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np
import torch

__all__ = [
    "MatchingMetricsAccumulator",
    "format_summary_table",
    "format_latency_table",
    "_latency_stats",
]


@dataclass
class MatchingMetricsAccumulator:
    """Streaming accumulator for Top-1 / Top-3 / MRR / latency.

    Call `update(scores, match_matrix, latency_ms)` for every frame and
    `summary()` once at the end.
    """

    correct_top1: int = 0
    correct_top3: int = 0
    rr_sum: float = 0.0
    n_cameras_with_match: int = 0
    n_frames: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def update(
        self,
        scores: torch.Tensor | np.ndarray,
        match_matrix: torch.Tensor | np.ndarray,
        latency_ms: float = 0.0,
    ) -> None:
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(match_matrix, torch.Tensor):
            match_matrix = match_matrix.detach().cpu().numpy()

        if scores.size == 0 or match_matrix.size == 0:
            return

        N, M = scores.shape
        # Descending argsort -> ranking per row
        ranking = np.argsort(-scores, axis=1)  # [N, M]

        for i in range(N):
            gt_indices = np.nonzero(match_matrix[i])[0]
            if gt_indices.size == 0:
                continue
            self.n_cameras_with_match += 1
            r = ranking[i]

            if r[0] in gt_indices:
                self.correct_top1 += 1
            if any(p in gt_indices for p in r[: min(3, M)]):
                self.correct_top3 += 1

            best_rank = M + 1
            for gt in gt_indices:
                pos = int(np.where(r == gt)[0][0]) + 1
                if pos < best_rank:
                    best_rank = pos
            if best_rank <= M:
                self.rr_sum += 1.0 / best_rank

        self.n_frames += 1
        self.latency_ms.append(float(latency_ms))

    def summary(self) -> dict:
        n = self.n_cameras_with_match
        lat_stats = _latency_stats(self.latency_ms)
        if n == 0:
            return {
                "top1": 0.0,
                "top3": 0.0,
                "mrr": 0.0,
                "n_frames": self.n_frames,
                "n_cameras_with_match": 0,
                **lat_stats,
            }
        return {
            "top1": self.correct_top1 / n,
            "top3": self.correct_top3 / n,
            "mrr": self.rr_sum / n,
            "n_frames": self.n_frames,
            "n_cameras_with_match": n,
            **lat_stats,
        }


def _latency_stats(latencies_ms: list[float] | np.ndarray) -> dict:
    """Mean/p50/p95/p99/std/throughput summary for a list of per-frame times."""
    if isinstance(latencies_ms, list):
        lat = np.array(latencies_ms, dtype=np.float64) if latencies_ms else np.zeros(1)
    else:
        lat = np.asarray(latencies_ms, dtype=np.float64)
        if lat.size == 0:
            lat = np.zeros(1)
    mean = float(np.mean(lat))
    return {
        "latency_ms_mean": mean,
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
        "latency_ms_std": float(np.std(lat, ddof=0)),
        "latency_ms_min": float(np.min(lat)),
        "latency_ms_max": float(np.max(lat)),
        "throughput_fps_mean": float(1000.0 / mean) if mean > 0 else 0.0,
        "n_iters": int(lat.size),
    }


def format_summary_table(rows: Iterable[Tuple[str, dict]]) -> str:
    """Format a list of (model_name, summary_dict) pairs as a fixed-width table."""
    header = (
        f"{'model':<22} {'top1':>7} {'top3':>7} {'mrr':>7} "
        f"{'lat_mean_ms':>12} {'lat_p95_ms':>11} {'frames':>7}"
    )
    bar = "-" * len(header)
    out = [header, bar]
    for name, s in rows:
        out.append(
            f"{name:<22} "
            f"{s['top1']*100:7.2f} {s['top3']*100:7.2f} {s['mrr']:7.4f} "
            f"{s['latency_ms_mean']:12.3f} {s['latency_ms_p95']:11.3f} "
            f"{s['n_frames']:7d}"
        )
    return "\n".join(out)


def format_latency_table(rows: Iterable[Tuple[str, dict]]) -> str:
    """Latency-focused fixed-width table for latency-only benchmarks."""
    header = (
        f"{'model':<22} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} "
        f"{'std':>8} {'min':>8} {'max':>8} {'fps':>8} {'iters':>7}"
    )
    bar = "-" * len(header)
    out = [header, bar]
    for name, s in rows:
        out.append(
            f"{name:<22} "
            f"{s['latency_ms_mean']:8.3f} {s['latency_ms_p50']:8.3f} "
            f"{s['latency_ms_p95']:8.3f} {s['latency_ms_p99']:8.3f} "
            f"{s['latency_ms_std']:8.3f} {s['latency_ms_min']:8.3f} "
            f"{s['latency_ms_max']:8.3f} {s['throughput_fps_mean']:8.1f} "
            f"{s['n_iters']:7d}"
        )
    return "\n".join(out)
