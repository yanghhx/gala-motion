from __future__ import annotations

import torch

from models.gala_motion import sequence_mask
from models.skeleton_graph import skeleton_edges


def reconstruction_errors(model, motion: torch.Tensor, lengths: torch.Tensor) -> dict[str, float]:
    """Kinematic reconstruction diagnostics in HumanML3D / KIT feature space."""
    frame_mask = sequence_mask(lengths, motion.shape[1]).to(motion.dtype)
    with torch.no_grad():
        _, mu, _, _, _, _, _ = model.encode(motion, lengths)
        recon = model.decode(mu, motion.shape[1]) * frame_mask.unsqueeze(-1)
        pred_pos = model.joint_positions(recon)
        true_pos = model.joint_positions(motion)
    valid = frame_mask.unsqueeze(-1)
    mpjpe = ((pred_pos - true_pos).norm(dim=-1) * frame_mask.unsqueeze(-1)).sum()
    mpjpe = mpjpe / (frame_mask.sum() * pred_pos.shape[2]).clamp_min(1)
    bone = []
    for i, j in skeleton_edges(model.cfg.num_joints):
        pred = (pred_pos[:, :, i] - pred_pos[:, :, j]).norm(dim=-1)
        true = (true_pos[:, :, i] - true_pos[:, :, j]).norm(dim=-1)
        bone.append((pred - true).abs())
    bone_err = (torch.stack(bone, -1) * frame_mask.unsqueeze(-1)).sum() / frame_mask.sum().clamp_min(1) / len(bone)
    pred_vel = torch.diff(recon, dim=1)
    true_vel = torch.diff(motion, dim=1)
    vel_mask = frame_mask[:, 1:] * frame_mask[:, :-1]
    vel_valid = vel_mask.unsqueeze(-1).expand_as(pred_vel)
    vel = ((pred_vel - true_vel).abs() * vel_valid).sum() / vel_valid.sum().clamp_min(1)
    foot_dim = min(model.cfg.foot_dim, motion.shape[-1])
    foot_valid = frame_mask.unsqueeze(-1).expand_as(recon[..., -foot_dim:])
    foot = ((recon[..., -foot_dim:] - motion[..., -foot_dim:]).abs() * foot_valid).sum()
    foot = foot / foot_valid.sum().clamp_min(1)
    mse = ((recon - motion).square() * valid).sum() / valid.sum().clamp_min(1)
    return {
        "mse": float(mse),
        "mpjpe": float(mpjpe),
        "bone": float(bone_err),
        "velocity": float(vel),
        "foot": float(foot),
    }
