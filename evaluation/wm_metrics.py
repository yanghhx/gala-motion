from __future__ import annotations

import numpy as np
import torch


def local_joints(motion: torch.Tensor, num_joints: int = 22) -> torch.Tensor:
    """HumanML3D RIC local joints with a zero root, matching GALAMotion."""
    body = motion[..., 4: 4 + (num_joints - 1) * 3]
    body = body.reshape(*motion.shape[:-1], num_joints - 1, 3)
    root = torch.zeros(*motion.shape[:-1], 1, 3, device=motion.device, dtype=motion.dtype)
    return torch.cat((root, body), dim=-2)


def integrate_root_xy(motion: torch.Tensor) -> torch.Tensor:
    """Integrate denormalized root velocities into a world-frame XY path."""
    heading = torch.cumsum(motion[..., 0], dim=-1)
    heading = torch.cat((torch.zeros_like(heading[..., :1]), heading[..., :-1]), dim=-1)
    cos_h, sin_h = heading.cos(), heading.sin()
    vx, vz = motion[..., 1], motion[..., 2]
    world_x = cos_h * vx - sin_h * vz
    world_z = sin_h * vx + cos_h * vz
    x = torch.cumsum(world_x, dim=-1)
    z = torch.cumsum(world_z, dim=-1)
    return torch.stack((x, z), dim=-1)


def mpjpe_mm(pred_joints: torch.Tensor, true_joints: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    dist = (pred_joints - true_joints).norm(dim=-1)
    if mask is None:
        return dist.mean() * 1000
    weight = mask.to(dist.dtype)
    while weight.ndim < dist.ndim:
        weight = weight.unsqueeze(-1)
    return (dist * weight).sum() / weight.expand_as(dist).sum().clamp_min(1) * 1000


def root_xy_error_m(pred_motion: torch.Tensor, true_motion: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred_xy = integrate_root_xy(pred_motion)
    true_xy = integrate_root_xy(true_motion)
    dist = (pred_xy - true_xy).norm(dim=-1)
    if mask is None:
        return dist.mean()
    return (dist * mask).sum() / mask.sum().clamp_min(1)


def latent_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean()


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
