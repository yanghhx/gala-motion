"""HumanML3D / KIT-ML vector layout and world-frame recovery.

Feature layout (Guo / MDM ``motion_process.py``), last axis of a clip:

- ``[0]`` root yaw velocity
- ``[1:3]`` root XZ velocity in the heading frame
- ``[3]`` root height
- ``[4 : 4+(J-1)*3]`` RIC local joint positions (joints 1..J-1)
- next ``(J-1)*6`` continuous 6D rotations
- next ``J*3`` local joint velocities
- last 4 channels: binary foot-contact labels, **not** foot positions

``GALAMotion.joint_positions`` keeps the existing RIC-local convention used by
the graph tokenizer and bone loss (root at the origin, no heading/XZ
integration). Skating and world-frame physical metrics use ``recover_from_ric``
on **denormalized** features instead.
"""
from __future__ import annotations

import torch

from models.skeleton_graph import foot_joints, skeleton_edges


CONTACT_DIM = 4
ROOT_Y_INDEX = 3
RIC_START = 4


def denormalize_motion(motion: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    if mean is None or std is None:
        return motion
    mean_t = torch.as_tensor(mean, device=motion.device, dtype=motion.dtype)
    std_t = torch.as_tensor(std, device=motion.device, dtype=motion.dtype)
    return motion * std_t + mean_t


def _qinv(q: torch.Tensor) -> torch.Tensor:
    scale = q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return q * scale


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover heading quaternions and root XYZ from Guo features.

    Matches MDM ``recover_root_rot_pos`` (yaw-only, Y-up).
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., ROOT_Y_INDEX]
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, num_joints: int) -> torch.Tensor:
    """World-frame joints ``(B, T, J, 3)`` from denormalized Guo features."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    ric_end = RIC_START + (num_joints - 1) * 3
    positions = data[..., RIC_START:ric_end].reshape(*data.shape[:-1], num_joints - 1, 3)
    heading = _qinv(r_rot_quat).unsqueeze(-2).expand(*positions.shape[:-1], 4)
    positions = _qrot(heading, positions)
    positions = positions.clone()
    positions[..., 0] = positions[..., 0] + r_pos[..., 0:1]
    positions[..., 2] = positions[..., 2] + r_pos[..., 2:3]
    return torch.cat((r_pos.unsqueeze(-2), positions), dim=-2)


def contact_from_features(motion: torch.Tensor, mean=None, std=None, threshold: float = 0.5) -> torch.Tensor:
    """Binary contact ``(B, T, 4)`` from the last four Guo channels."""
    contact = denormalize_motion(motion, mean, std)[..., -CONTACT_DIM:]
    return contact > threshold


def kinematic_contact_mask(
    positions: torch.Tensor,
    num_joints: int,
    frame_mask: torch.Tensor,
    height_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Contact from GT world positions: low height and low speed.

    ``c[t]`` labels the transition ``t -> t+1``. The last frame is always False.
    """
    feet_idx = list(foot_joints(num_joints))
    feet = positions[:, :, feet_idx]
    height = feet[..., 1]
    delta = feet[:, 1:] - feet[:, :-1]
    speed = delta.norm(dim=-1)
    trans_mask = (frame_mask[:, 1:] & frame_mask[:, :-1]).unsqueeze(-1)
    contact = torch.zeros(*feet.shape[:3], device=positions.device, dtype=torch.bool)
    contact[:, :-1] = (height[:, :-1] < height_threshold) & (speed < velocity_threshold) & trans_mask
    return contact


def contact_mask_for_source(
    source: str,
    true_motion: torch.Tensor,
    true_positions: torch.Tensor,
    frame_mask: torch.Tensor,
    num_joints: int,
    mean=None,
    std=None,
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.01,
) -> torch.Tensor:
    if source == "gt_contact_feature":
        contact = contact_from_features(true_motion, mean, std)
        contact = contact & frame_mask.unsqueeze(-1)
        contact = contact.clone()
        contact[:, -1] = False
        return contact
    if source == "kinematic_threshold":
        return kinematic_contact_mask(
            true_positions, num_joints, frame_mask, height_threshold, velocity_threshold,
        )
    raise ValueError(f"Unknown contact_source={source!r}")


def skating_error(
    pred_positions: torch.Tensor,
    contact: torch.Tensor,
    frame_mask: torch.Tensor,
    num_joints: int,
) -> torch.Tensor:
    """Mean squared world-frame foot travel on contact transitions.

    Returns a 0 tensor (not NaN) when no contact frames are present.
    """
    feet_idx = list(foot_joints(num_joints))
    pred_feet = pred_positions[:, :, feet_idx]
    travel = (pred_feet[:, 1:] - pred_feet[:, :-1]).pow(2).sum(dim=-1)
    trans_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
    weights = contact[:, :-1].to(travel.dtype) * trans_mask.unsqueeze(-1).to(travel.dtype)
    numer = (travel * weights).sum()
    return numer / weights.sum().clamp_min(1)


def foot_transition_speed(positions: torch.Tensor, num_joints: int) -> torch.Tensor:
    feet_idx = list(foot_joints(num_joints))
    feet = positions[:, :, feet_idx]
    return (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)


def contact_velocity_valid_ratio(
    positions: torch.Tensor,
    contact: torch.Tensor,
    frame_mask: torch.Tensor,
    num_joints: int,
    vmax: float,
) -> torch.Tensor:
    """Fraction of GT-contact transitions whose world foot speed is below ``vmax``.

    r_valid = Σ c 1(v < vmax) / Σ c
    """
    speed = foot_transition_speed(positions, num_joints)
    trans = frame_mask[:, 1:] & frame_mask[:, :-1]
    weights = contact[:, :-1].to(speed.dtype) * trans.unsqueeze(-1).to(speed.dtype)
    valid = weights * (speed < vmax).to(speed.dtype)
    return valid.sum() / weights.sum().clamp_min(1)


def paired_acceleration_error(
    pred_positions: torch.Tensor,
    true_positions: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    pred_acc = pred_positions[:, 2:] - 2 * pred_positions[:, 1:-1] + pred_positions[:, :-2]
    true_acc = true_positions[:, 2:] - 2 * true_positions[:, 1:-1] + true_positions[:, :-2]
    acc_mask = frame_mask[:, 2:] & frame_mask[:, 1:-1] & frame_mask[:, :-2]
    valid = acc_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_acc)
    return ((pred_acc - true_acc).abs() * valid).sum() / valid.sum().clamp_min(1)


def bone_length_error(
    pred_positions: torch.Tensor,
    true_positions: torch.Tensor,
    frame_mask: torch.Tensor,
    num_joints: int,
) -> torch.Tensor:
    terms = []
    for i, j in skeleton_edges(num_joints):
        pred = (pred_positions[:, :, i] - pred_positions[:, :, j]).norm(dim=-1)
        true = (true_positions[:, :, i] - true_positions[:, :, j]).norm(dim=-1)
        terms.append((pred - true).abs())
    stacked = torch.stack(terms, -1) * frame_mask.unsqueeze(-1)
    return stacked.sum() / frame_mask.sum().clamp_min(1) / len(terms)


def bone_length_variance(positions: torch.Tensor, frame_mask: torch.Tensor, num_joints: int) -> torch.Tensor:
    variances = []
    weights = frame_mask.to(positions.dtype)
    count = weights.sum(dim=1).clamp_min(1)
    for i, j in skeleton_edges(num_joints):
        length = (positions[:, :, i] - positions[:, :, j]).norm(dim=-1)
        mean = (length * weights).sum(dim=1) / count
        var = ((length - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
        variances.append(var.mean())
    return torch.stack(variances).mean()


def acceleration_magnitude(positions: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    acc = (positions[:, 2:] - 2 * positions[:, 1:-1] + positions[:, :-2]).norm(dim=-1)
    acc_mask = frame_mask[:, 2:] & frame_mask[:, 1:-1] & frame_mask[:, :-2]
    valid = acc_mask.unsqueeze(-1).expand_as(acc)
    return (acc * valid).sum() / valid.sum().clamp_min(1)
