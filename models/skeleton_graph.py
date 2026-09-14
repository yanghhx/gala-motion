from __future__ import annotations

import torch
from torch import nn


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
KIT21_PARTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7),
    (8, 9, 10),
    (11, 12, 13, 14, 15),
    (16, 17, 18, 19, 20),
)


def body_parts(num_joints: int):
    return KIT21_PARTS if num_joints == 21 else HML22_PARTS


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


class SkeletonGraphEncoder(nn.Module):
    def __init__(self, num_joints: int, dim: int, layers: int, heads: int, kind: str = "ctr"):
        super().__init__()
        self.kind = kind
        self.register_buffer("adjacency", normalized_adjacency(num_joints), persistent=False)
        self.input = nn.Linear(6, dim)
        block = STGCNBlock if kind == "stgcn" else CTRGraphBlock
        self.blocks = nn.ModuleList([block(dim, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = torch.diff(positions, dim=1, prepend=positions[:, :1])
        x = self.input(torch.cat((positions, velocity), dim=-1))
        for block in self.blocks:
            x = block(x, self.adjacency)
        x = self.output_norm(x)
        return x.mean(dim=2), x
