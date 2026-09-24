from __future__ import annotations

import torch
from torch import nn


# Sequential chains used by the current graph/bone losses. This is NOT the official
# MDM t2m_kinematic_chain ([[0,2,5,8,11], [0,1,4,7,10], ...]). Skating uses
# HML22_FOOT_JOINTS / KIT21_FOOT_JOINTS below instead of these chain tips.
HML22_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
    (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (8, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21),
)

KIT21_EDGES = (
    (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
)


def skeleton_edges(num_joints: int):
    return KIT21_EDGES if num_joints == 21 else HML22_EDGES


# Five anatomical groups used by part-level language–motion alignment.
# Indices follow the same trees as HML22_EDGES / KIT21_EDGES.
HML22_PARTS = (
    (0, 7, 8, 9, 10),          # torso / head
    (11, 12, 13, 14, 15),      # left arm
    (16, 17, 18, 19, 20, 21),  # right arm
    (1, 2, 3),                 # left leg
    (4, 5, 6),                 # right leg
)
PART_NAMES = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")

# Official Guo/MDM contact-joint order (feet_l then feet_r in motion_process.py).
# These are dataset-convention indices, not the simplified HML22_EDGES chain tips.
# HumanML3D: left ankle/foot, right ankle/foot. KIT-ML: left toe/foot, right toe/foot.
HML22_FOOT_JOINTS = (7, 10, 8, 11)
HML22_FOOT_NAMES = ("left_ankle", "left_foot", "right_ankle", "right_foot")
KIT21_FOOT_JOINTS = (19, 20, 14, 15)
KIT21_FOOT_NAMES = ("left_toe", "left_foot", "right_toe", "right_foot")
KIT21_PARTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7),
    (8, 9, 10),
    (11, 12, 13, 14, 15),
    (16, 17, 18, 19, 20),
)


def body_parts(num_joints: int):
    return KIT21_PARTS if num_joints == 21 else HML22_PARTS


def foot_joints(num_joints: int):
    return KIT21_FOOT_JOINTS if num_joints == 21 else HML22_FOOT_JOINTS


def foot_names(num_joints: int):
    return KIT21_FOOT_NAMES if num_joints == 21 else HML22_FOOT_NAMES


def pool_body_parts(joint_feat: torch.Tensor, frame_mask: torch.Tensor, parts) -> torch.Tensor:
    """Pool (B, T, J, D) joint features into (B, P, D) part latents."""
    pooled = []
    weights = frame_mask.to(joint_feat.dtype).unsqueeze(-1)
    for joints in parts:
        part = joint_feat[:, :, list(joints)]
        num = (part * weights.unsqueeze(2)).sum(dim=(1, 2))
        den = weights.sum(dim=1).clamp_min(1) * len(joints)
        pooled.append(num / den)
    return torch.stack(pooled, dim=1)


def normalized_adjacency(num_joints: int, edges=None) -> torch.Tensor:
    if edges is None:
        edges = skeleton_edges(num_joints)
    a = torch.eye(num_joints)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            a[i, j] = a[j, i] = 1
    degree = a.sum(-1).clamp_min(1).rsqrt()
    return degree[:, None] * a * degree[None, :]


class STGCNBlock(nn.Module):
    """Static skeleton GCN (no sample-dependent topology)."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        del heads
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("ij,btjd->btid", adjacency, self.norm(x))
        x = x + self.proj(h)
        return x + self.ffn(x)


class CTRGraphBlock(nn.Module):
    """Static anatomy plus sample-dependent channel-wise topology refinement."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.topology_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, t, j, d = x.shape
        h = self.heads
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q = q.view(b, t, j, h, d // h).transpose(2, 3)
        k = k.view(b, t, j, h, d // h).transpose(2, 3)
        v = v.view(b, t, j, h, d // h).transpose(2, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (d // h) ** -0.5
        prior = adjacency.clamp_min(1e-6).log()[None, None, None]
        weights = (logits + self.topology_gate.sigmoid() * prior).softmax(-1)
        out = torch.matmul(weights, v).transpose(2, 3).reshape(b, t, j, d)
        x = x + self.proj(out)
        return x + self.ffn(x)


class STCTRGraphBlock(nn.Module):
    """CTR spatial graph + temporal self-attention per joint.

    Spatial branch reuses CTRGraphBlock. Temporal branch does per-joint
    self-attention across frames with learnable relative-position bias and
    frame-mask awareness.  Gates start at α_s=1, α_t=0 so the model begins
    from the original spatial-only behaviour.
    """

    def __init__(self, dim: int, heads: int = 4, temporal_heads: int = 4,
                 temporal_max_distance: int = 32, temporal_dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.temporal_heads = temporal_heads
        self.temporal_max_distance = temporal_max_distance
        # --- spatial branch (CTR) ---
        self.spatial = CTRGraphBlock(dim, heads)
        self.spatial_gate = nn.Parameter(torch.tensor(1.0))  # α_s
        # --- temporal branch ---
        self.temporal_norm = nn.LayerNorm(dim)
        self.t_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.t_proj = nn.Linear(dim, dim)
        self.temporal_gate = nn.Parameter(torch.tensor(0.0))  # α_t
        self.t_ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )
        # Learnable relative-position bias table: (2*max_dist+1, heads)
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * temporal_max_distance + 1, temporal_heads))
        self.temporal_dropout = nn.Dropout(temporal_dropout)

    def _temporal_attention(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, J, D)  frame_mask: (B, T) bool."""
        b, t, j, d = x.shape
        h = self.temporal_heads
        dh = d // h
        # Reshape to (B*J, T, D) so each joint attends across time.
        flat = x.permute(0, 2, 1, 3).reshape(b * j, t, d)
        q, k, v = self.t_qkv(self.temporal_norm(flat)).chunk(3, dim=-1)
        q = q.view(b * j, t, h, dh).transpose(1, 2)  # (B*J, h, T, dh)
        k = k.view(b * j, t, h, dh).transpose(1, 2)
        v = v.view(b * j, t, h, dh).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * dh ** -0.5  # (B*J, h, T, T)
        # Relative position bias
        rel = self.rel_pos_bias[self.rel_pos_bias.shape[0] // 2]  # center = 0
        positions = torch.arange(t, device=x.device)
        dist = positions[None] - positions[:, None]  # (T, T)
        dist = dist.clamp(-self.temporal_max_distance, self.temporal_max_distance)
        bias_idx = dist + self.temporal_max_distance  # (T, T)
        logits = logits + self.rel_pos_bias[bias_idx].permute(2, 0, 1)  # (h, T, T) -> broadcast
        # Frame mask: (B, T) -> (B*J, 1, 1, T)
        pad_mask = ~frame_mask[:, None].expand(b, j, t).reshape(b * j, t)
        logits = logits.masked_fill(pad_mask[:, None, None], torch.finfo(logits.dtype).min)
        attn = logits.softmax(dim=-1)
        attn = self.temporal_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b * j, t, d)
        out = out.reshape(b, j, t, d).permute(0, 2, 1, 3)  # back to (B, T, J, D)
        return self.t_proj(out)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                 frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        if frame_mask is None:
            frame_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        x = x + self.spatial_gate * self.spatial(x, adjacency)
        x = x + self.temporal_gate * self._temporal_attention(x, frame_mask)
        return x + self.t_ffn(x)


class SkeletonGraphEncoder(nn.Module):
    def __init__(self, num_joints: int, dim: int, layers: int, heads: int, kind: str = "ctr",
                 temporal_heads: int = 4, temporal_max_distance: int = 32, temporal_dropout: float = 0.1):
        super().__init__()
        self.kind = kind
        self.register_buffer("adjacency", normalized_adjacency(num_joints), persistent=False)
        self.input = nn.Linear(6, dim)
        if kind == "st_ctr":
            block_cls = STCTRGraphBlock
            self.blocks = nn.ModuleList([
                block_cls(dim, heads, temporal_heads, temporal_max_distance, temporal_dropout)
                for _ in range(layers)
            ])
        else:
            block_cls = STGCNBlock if kind == "stgcn" else CTRGraphBlock
            self.blocks = nn.ModuleList([block_cls(dim, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, positions: torch.Tensor, frame_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = torch.diff(positions, dim=1, prepend=positions[:, :1])
        x = self.input(torch.cat((positions, velocity), dim=-1))
        for block in self.blocks:
            if self.kind == "st_ctr":
                x = block(x, self.adjacency, frame_mask)
            else:
                x = block(x, self.adjacency)
        x = self.output_norm(x)
        return x.mean(dim=2), x
