"""
Structure-Aware Edge-Aware GAT modules for CRLite-ViT Exp4.

Verbatim port of `src/core/models/crlite_vit_exp4/gnn.py`. State-dict keys
are preserved exactly so trained weights load without renaming. All ops are
pure PyTorch (no torch-geometric, no custom CUDA), which is what we need
for Jetson AGX Thor + TensorRT.

Two graphs are built dynamically from the detection geometry of each frame:

- `CameraGNN`: 2D detections, edges over (螖u, 螖v, log_scale_ratio).
- `LiDARGNN`: 3D detections, edges over (螖x, 螖y, 螖z, distance, log_size_ratio).

Both run the same `EdgeAwareGATLayer` block stacked `num_layers` deep.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

EPS = 1e-6
CLAMP_MIN = -50.0
CLAMP_MAX = 50.0


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance with no `torch.cdist`.

    `torch.cdist` has a buggy ONNX symbolic in the legacy TorchScript
    exporter (it asserts a static batch dim). We use the broadcast form
    sqrt((a[:,None]-b[None,:])^2 summed), which traces cleanly to a
    `Sub -> Pow -> ReduceSum -> Sqrt` chain that TensorRT 10 handles.
    """
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    return torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)


class EdgeAwareGATLayer(nn.Module):
    """Edge-aware graph attention with safe numerical guards.

    Message:    m_ij = MLP([h_i, h_j, e_ij])
    Attention:  alpha_ij = softmax_j(MLP(m_ij))
    Aggregate:  h_i' = LayerNorm(h_i + sum_j alpha_ij * m_ij)
    """

    def __init__(
        self,
        node_dim: int = 256,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        self.attention_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        N = node_features.size(0)
        E = edge_index.size(1)

        if E == 0:
            return node_features

        node_features = torch.clamp(node_features, min=CLAMP_MIN, max=CLAMP_MAX)
        edge_features = torch.clamp(edge_features, min=CLAMP_MIN, max=CLAMP_MAX)

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        h_src = node_features[src_idx]
        h_tgt = node_features[tgt_idx]

        message_input = torch.cat([h_src, h_tgt, edge_features], dim=-1)
        messages = self.message_mlp(message_input)
        messages = torch.clamp(messages, min=CLAMP_MIN, max=CLAMP_MAX)

        attention_logits = self.attention_mlp(messages).squeeze(-1)
        attention_weights = self._stable_scatter_softmax(attention_logits, tgt_idx, N)

        weighted_messages = messages * attention_weights.unsqueeze(-1)

        aggregated = torch.zeros(
            N,
            self.node_dim,
            device=node_features.device,
            dtype=node_features.dtype,
        )
        aggregated.scatter_add_(
            0,
            tgt_idx.unsqueeze(-1).expand(-1, self.node_dim),
            weighted_messages,
        )

        output = self.norm(node_features + self.dropout(aggregated))
        return torch.clamp(output, min=CLAMP_MIN, max=CLAMP_MAX)

    def _stable_scatter_softmax(
        self,
        logits: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        index = index.long()
        logits_clamped = torch.clamp(logits, min=-20.0, max=20.0)
        exp_logits = torch.exp(logits_clamped)

        sum_exp = torch.zeros(
            num_nodes, device=logits.device, dtype=logits.dtype
        )
        sum_exp.scatter_add_(0, index, exp_logits)

        weights = exp_logits / (sum_exp[index] + EPS)
        weights = torch.clamp(weights, min=0.0, max=1.0)

        if torch.isnan(weights).any():
            weights = torch.where(
                torch.isnan(weights),
                torch.ones_like(weights) / max(num_nodes, 1),
                weights,
            )
        return weights


class CameraGNN(nn.Module):
    """Camera-side GNN. kNN graph over normalised 2D centres."""

    def __init__(
        self,
        embed_dim: int = 256,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        k_neighbors: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.k_neighbors = k_neighbors
        self.num_layers = num_layers

        # Raw edge features: (螖u, 螖v, log_scale_ratio)
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList(
            [
                EdgeAwareGATLayer(
                    node_dim=embed_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        bboxes: torch.Tensor,
        image_size: Tuple[int, int] = (1920, 1080),
    ) -> torch.Tensor:
        N = node_features.size(0)
        if N <= 1:
            return node_features

        edge_index, edge_features = self._build_camera_graph(bboxes, image_size)

        h = node_features
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
        return h

    def _build_camera_graph(
        self,
        bboxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = bboxes.size(0)
        device = bboxes.device
        dtype = bboxes.dtype

        W = max(float(image_size[0]), 1.0)
        H = max(float(image_size[1]), 1.0)

        bboxes_safe = torch.clamp(bboxes.clone(), min=0.0, max=max(W, H) * 2)
        centers = (bboxes_safe[:, :2] + bboxes_safe[:, 2:]) / 2
        widths = torch.abs(bboxes_safe[:, 2] - bboxes_safe[:, 0]).clamp(min=1.0)
        heights = torch.abs(bboxes_safe[:, 3] - bboxes_safe[:, 1]).clamp(min=1.0)
        areas = widths * heights

        centers_norm = torch.zeros_like(centers)
        centers_norm[:, 0] = centers[:, 0] / W
        centers_norm[:, 1] = centers[:, 1] / H
        centers_norm = torch.clamp(centers_norm, min=-1.0, max=2.0)

        dist_matrix = _pairwise_l2(centers_norm, centers_norm)

        K = min(self.k_neighbors, N - 1)
        if K == 0:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, self.edge_dim, device=device, dtype=dtype),
            )

        # Excludes self-loops via additive mask (ONNX-traceable; the in-place
        # `fill_diagonal_` has no ONNX kernel).
        eye_mask = torch.eye(N, device=device, dtype=dist_matrix.dtype) * 1e9
        dist_matrix = dist_matrix + eye_mask
        _, knn_indices = dist_matrix.topk(K, dim=1, largest=False)

        src = (
            torch.arange(N, device=device)
            .unsqueeze(1)
            .expand(-1, K)
            .flatten()
        )
        tgt = knn_indices.flatten()
        edge_index = torch.stack(
            [torch.cat([src, tgt]), torch.cat([tgt, src])],
            dim=0,
        )

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        src_centers = centers[src_idx]
        tgt_centers = centers[tgt_idx]
        src_areas = areas[src_idx]
        tgt_areas = areas[tgt_idx]

        delta_u = (tgt_centers[:, 0] - src_centers[:, 0]) / W
        delta_v = (tgt_centers[:, 1] - src_centers[:, 1]) / H
        delta_u = torch.clamp(delta_u, min=-2.0, max=2.0)
        delta_v = torch.clamp(delta_v, min=-2.0, max=2.0)

        scale_ratio = tgt_areas / src_areas
        log_scale = torch.log(scale_ratio).clamp(min=-5.0, max=5.0)

        raw_edge_feat = torch.stack([delta_u, delta_v, log_scale], dim=-1)
        raw_edge_feat = torch.where(
            torch.isfinite(raw_edge_feat),
            raw_edge_feat,
            torch.zeros_like(raw_edge_feat),
        )

        edge_features = self.edge_encoder(raw_edge_feat)
        return edge_index, edge_features


class LiDARGNN(nn.Module):
    """LiDAR-side GNN. kNN graph over normalised 3D centres."""

    def __init__(
        self,
        embed_dim: int = 256,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        k_neighbors: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.k_neighbors = k_neighbors
        self.num_layers = num_layers

        # Raw edge features: (螖x, 螖y, 螖z, distance, log_size_ratio)
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, edge_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList(
            [
                EdgeAwareGATLayer(
                    node_dim=embed_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        centers_3d: torch.Tensor,
        sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = node_features.size(0)
        if M <= 1:
            return node_features

        edge_index, edge_features = self._build_lidar_graph(centers_3d, sizes)

        h = node_features
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
        return h

    def _build_lidar_graph(
        self,
        centers_3d: torch.Tensor,
        sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = centers_3d.size(0)
        device = centers_3d.device
        dtype = centers_3d.dtype

        if sizes is None:
            sizes = torch.ones(M, device=device, dtype=dtype)
        sizes = torch.clamp(sizes, min=0.1)

        center_mean = centers_3d.mean(dim=0, keepdim=True)
        centered = centers_3d - center_mean
        max_dist = torch.norm(centered, dim=-1).max().clamp(min=1.0)
        centers_norm = centered / max_dist

        dist_matrix = _pairwise_l2(centers_norm, centers_norm)

        K = min(self.k_neighbors, M - 1)
        if K == 0:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, self.edge_dim, device=device, dtype=dtype),
            )

        # See note in CameraGNN: `fill_diagonal_` isn't ONNX-exportable.
        eye_mask = torch.eye(M, device=device, dtype=dist_matrix.dtype) * 1e9
        dist_matrix = dist_matrix + eye_mask
        _, knn_indices = dist_matrix.topk(K, dim=1, largest=False)

        src = (
            torch.arange(M, device=device)
            .unsqueeze(1)
            .expand(-1, K)
            .flatten()
        )
        tgt = knn_indices.flatten()
        edge_index = torch.stack(
            [torch.cat([src, tgt]), torch.cat([tgt, src])],
            dim=0,
        )

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        src_centers = centers_3d[src_idx]
        tgt_centers = centers_3d[tgt_idx]
        src_sizes = sizes[src_idx]
        tgt_sizes = sizes[tgt_idx]

        delta_xyz = (tgt_centers - src_centers) / max_dist
        delta_xyz = torch.clamp(delta_xyz, min=-5.0, max=5.0)

        distance = torch.norm(delta_xyz, dim=-1, keepdim=True).clamp(min=0.0, max=10.0)

        size_ratio = tgt_sizes / src_sizes
        log_size_ratio = torch.log(size_ratio).unsqueeze(-1).clamp(min=-5.0, max=5.0)

        raw_edge_feat = torch.cat([delta_xyz, distance, log_size_ratio], dim=-1)
        raw_edge_feat = torch.where(
            torch.isfinite(raw_edge_feat),
            raw_edge_feat,
            torch.zeros_like(raw_edge_feat),
        )

        edge_features = self.edge_encoder(raw_edge_feat)
        return edge_index, edge_features
