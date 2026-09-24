from __future__ import annotations

import torch

from models.gala_motion import sequence_mask
from models.motion_repr import (
    acceleration_magnitude,
    bone_length_error,
    bone_length_variance,
    contact_mask_for_source,
    denormalize_motion,
    paired_acceleration_error,
    recover_from_ric,
    skating_error,
)
from models.skeleton_graph import skeleton_edges


def reconstruction_errors(model, motion: torch.Tensor, lengths: torch.Tensor, mean=None, std=None) -> dict[str, float]:
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
    stats = {
        "mse": float(mse),
        "mpjpe": float(mpjpe),
        "bone": float(bone_err),
        "velocity": float(vel),
        "foot": float(foot),
    }
    stats.update(
        physical_metrics(
            recon, motion, lengths, model.cfg.num_joints,
            mean=mean, std=std,
            contact_source=getattr(model.cfg, "contact_source", "gt_contact_feature"),
            height_threshold=getattr(model.cfg, "foot_height_threshold", 0.05),
            velocity_threshold=getattr(model.cfg, "foot_velocity_threshold", 0.01),
        )
    )
    return stats


def physical_metrics(
    pred_motion: torch.Tensor,
    true_motion: torch.Tensor | None,
    lengths: torch.Tensor,
    num_joints: int,
    mean=None,
    std=None,
    contact_source: str = "gt_contact_feature",
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.01,
) -> dict[str, float]:
    """Physical diagnostics, named separately from the official Guo T2M metrics.

    Paired velocity/acceleration errors are only reported when ``true_motion`` is
    given. Unpaired generation should use skating, bone variance, and acceleration
    magnitude rather than paired GT errors.
    """
    frame_mask = sequence_mask(lengths, pred_motion.shape[1])
    pred_world = recover_from_ric(denormalize_motion(pred_motion, mean, std), num_joints)
    out = {
        "physical/bone_length_variance": float(bone_length_variance(pred_world, frame_mask, num_joints)),
        "physical/acceleration_magnitude": float(acceleration_magnitude(pred_world, frame_mask)),
    }
    gt_for_contact = true_motion if true_motion is not None else pred_motion
    true_world = (
        recover_from_ric(denormalize_motion(true_motion, mean, std), num_joints)
        if true_motion is not None else pred_world
    )
    contact = contact_mask_for_source(
        contact_source, gt_for_contact, true_world, frame_mask, num_joints,
        mean=mean, std=std, height_threshold=height_threshold, velocity_threshold=velocity_threshold,
    )
    out["physical/foot_skating"] = float(skating_error(pred_world, contact, frame_mask, num_joints))
    trans = (frame_mask[:, 1:] & frame_mask[:, :-1]).unsqueeze(-1).to(pred_world.dtype)
    active = contact[:, :-1].to(pred_world.dtype) * trans
    out["physical/contact_ratio"] = float(active.sum() / trans.expand_as(active).sum().clamp_min(1))
    if true_motion is not None:
        out["physical/bone_error"] = float(bone_length_error(pred_world, true_world, frame_mask, num_joints))
        out["physical/acceleration"] = float(paired_acceleration_error(pred_world, true_world, frame_mask))
        pred_vel = pred_world[:, 1:] - pred_world[:, :-1]
        true_vel = true_world[:, 1:] - true_world[:, :-1]
        vel_mask = (frame_mask[:, 1:] & frame_mask[:, :-1]).unsqueeze(-1).unsqueeze(-1)
        out["physical/velocity"] = float(
            ((pred_vel - true_vel).abs() * vel_mask).sum() / vel_mask.expand_as(pred_vel).sum().clamp_min(1)
        )
    return out
