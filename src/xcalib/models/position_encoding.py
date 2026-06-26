"""
Position-encoding module used by CRLite, CRLite-2DPE, and CRLite-ViT Exp3.

Behaviour-preserving port of the originals under src/core/models/. The
only deliberate change is that the 2D sinusoidal computation is rewritten
in pure tensor ops (was numpy-on-CPU). This keeps numerical parity with
the trained checkpoints to fp32 precision and removes a CPU<->GPU sync
that would otherwise break ONNX tracing on the edge device.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# 2D sinusoidal kernel — shared helper
# ============================================================================

def _get_2d_position_encoding(
    x: torch.Tensor, y: torch.Tensor, token_len: int
) -> torch.Tensor:
    """Tensor-native 2D sinusoidal PE.

    Matches the lab's numpy implementation:
        - For j in [0, token_len//2):
            denom[j]   = 10000 ** (2 * (j // 2) / (token_len // 2))
            x_enc[:, j] = x / denom[j]
            y_enc[:, j] = y / denom[j]
        - x_sin = sin(x_enc), y_cos = cos(y_enc)
        - out[:, 0::2] = x_sin, out[:, 1::2] = y_cos

    Args:
        x, y: [B] coordinates (must be float tensors)
        token_len: total output dimension (must be even)
    Returns:
        [B, token_len] tensor on the same device/dtype as x.
    """
    if token_len % 2 != 0:
        raise ValueError("token_len must be even")

    half = token_len // 2
    device, dtype = x.device, x.dtype

    # j_indices: 0, 1, 2, ..., half-1
    j = torch.arange(half, device=device, dtype=dtype)
    # denom[j] = 10000 ** (2 * (j // 2) / half)
    exponent = (2.0 * torch.floor(j / 2.0)) / float(half)
    denom = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), exponent)

    # [B, half]
    x_enc = x.unsqueeze(1) / denom.unsqueeze(0)
    y_enc = y.unsqueeze(1) / denom.unsqueeze(0)

    x_sin = torch.sin(x_enc)
    y_cos = torch.cos(y_enc)

    # Interleave sin/cos
    out = torch.empty(x.size(0), token_len, device=device, dtype=dtype)
    out[:, 0::2] = x_sin
    out[:, 1::2] = y_cos
    return out


# ============================================================================
# Enhanced 3D PE (CRLite, CRLite-2DPE, CRLite-ViT Exp3)
# ============================================================================

class Enhanced3DPositionEncoding(nn.Module):
    """2D sinusoidal + depth MLP + 3D spatial MLP, fused to `token_len`.

    `pe_mode`:
        - "full"     (default)  — depth MLP and 3D MLP contribute their features.
        - "2d_only"             — depth/3D branches zeroed at every forward pass.
                                   Matches the R1 #6 ablation in the manuscript.

    State-dict keys are identical regardless of `pe_mode` (the same layers
    exist), so `crlite_2dpe` checkpoints load into this module with
    `pe_mode="2d_only"`.

    The ViT Exp3 variant of the lab code does not expose `pe_mode`, but its
    parameter names are identical to this class.
    """

    def __init__(self, token_len: int = 256, max_depth: float = 100.0, pe_mode: str = "full"):
        super().__init__()
        if pe_mode not in {"full", "2d_only"}:
            raise ValueError(f"Unknown pe_mode: {pe_mode}")
        self.token_len = token_len
        self.max_depth = max_depth
        self.pe_mode = pe_mode

        self.depth_proj = nn.Sequential(
            nn.Linear(1, token_len // 4),
            nn.ReLU(),
            nn.Linear(token_len // 4, token_len // 4),
        )
        self.spatial_3d_proj = nn.Sequential(
            nn.Linear(3, token_len // 4),
            nn.ReLU(),
            nn.Linear(token_len // 4, token_len // 4),
        )
        self.fusion = nn.Sequential(
            nn.Linear(token_len + token_len // 2, token_len),
            nn.LayerNorm(token_len),
            nn.ReLU(),
        )

    def forward(
        self,
        pos_2d: torch.Tensor,
        depth: torch.Tensor | None = None,
        lidar_points: torch.Tensor | None = None,
        pos_3d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = pos_2d.size(0)
        device = pos_2d.device
        dtype = pos_2d.dtype

        # 2D sinusoidal branch
        pos_feat_2d = _get_2d_position_encoding(
            pos_2d[:, 0].to(dtype), pos_2d[:, 1].to(dtype), self.token_len
        )

        # Depth feature
        if depth is not None:
            depth_norm = torch.clamp(depth / self.max_depth, 0.0, 1.0)
            depth_feat = self.depth_proj(depth_norm)
        elif lidar_points is not None:
            mean_z = lidar_points[:, :, 2].mean(dim=1, keepdim=True)
            depth_feat = self.depth_proj(torch.clamp(mean_z / self.max_depth, 0.0, 1.0))
        elif pos_3d is not None:
            depth_norm = torch.clamp(pos_3d[:, 2:3] / self.max_depth, 0.0, 1.0)
            depth_feat = self.depth_proj(depth_norm)
        else:
            depth_feat = torch.zeros(batch_size, self.token_len // 4, device=device, dtype=dtype)

        # 3D spatial feature
        if pos_3d is not None:
            spatial_3d_feat = self.spatial_3d_proj(pos_3d)
        elif lidar_points is not None:
            center_3d = lidar_points.mean(dim=1)
            spatial_3d_feat = self.spatial_3d_proj(center_3d)
        else:
            pos_3d_fake = torch.cat(
                [pos_2d, torch.zeros(batch_size, 1, device=device, dtype=dtype)], dim=1
            )
            spatial_3d_feat = self.spatial_3d_proj(pos_3d_fake)

        # 2D-PE-only ablation: zero the learned branches.
        if self.pe_mode == "2d_only":
            depth_feat = torch.zeros_like(depth_feat)
            spatial_3d_feat = torch.zeros_like(spatial_3d_feat)

        combined = torch.cat([pos_feat_2d, depth_feat, spatial_3d_feat], dim=1)
        return self.fusion(combined)
