import torch

from evaluation.kinematics import physical_metrics
from models.gala_motion import GALAMotion, GALAMotionConfig, sequence_mask
from models.motion_repr import recover_from_ric, skating_error
from models.skeleton_graph import HML22_FOOT_JOINTS, KIT21_FOOT_JOINTS, foot_joints


def tiny_model(**kwargs):
    cfg = GALAMotionConfig(
        motion_dim=kwargs.pop("motion_dim", 263),
        num_joints=kwargs.pop("num_joints", 22),
        latent_dim=64,
        model_dim=64,
        text_dim=48,
        num_heads=4,
        graph_layers=1,
        dit_layers=1,
        max_frames=40,
        train_stage="flow",
        use_kinematic_flow=True,
        **kwargs,
    )
    return GALAMotion(cfg)


def _write_joint(motion, joint, xyz):
    if joint == 0:
        motion[..., 3] = xyz[..., 1]
        return
    start = 4 + (joint - 1) * 3
    motion[..., start: start + 3] = xyz


def test_recover_from_ric_root_height():
    data = torch.zeros(2, 8, 263)
    data[..., 3] = 1.25
    joints = recover_from_ric(data, 22)
    assert joints.shape == (2, 8, 22, 3)
    torch.testing.assert_close(joints[:, :, 0, 1], torch.full((2, 8), 1.25))
    torch.testing.assert_close(joints[:, :, 0, 0], torch.zeros(2, 8))


def test_foot_joint_indices_match_dataset_convention():
    assert foot_joints(22) == HML22_FOOT_JOINTS == (7, 10, 8, 11)
    assert foot_joints(21) == KIT21_FOOT_JOINTS == (19, 20, 14, 15)


def test_constant_velocity_acceleration_near_zero():
    model = tiny_model()
    frames = 16
    motion = torch.zeros(2, frames, 263)
    time = torch.arange(frames, dtype=torch.float32)
    for joint in range(1, 22):
        _write_joint(motion, joint, torch.stack((0.1 * time, torch.zeros(frames), torch.zeros(frames)), -1))
    mask = torch.ones(2, frames, dtype=torch.bool)
    loss = model._acceleration_loss(motion, motion, mask)
    assert torch.isfinite(loss)
    assert float(loss) < 1e-6


def test_padding_does_not_change_acceleration_loss():
    model = tiny_model()
    frames = 16
    valid = 10
    motion = torch.zeros(2, frames, 263)
    time = torch.arange(frames, dtype=torch.float32)
    for joint in range(1, 22):
        _write_joint(motion, joint, torch.stack((0.1 * time, torch.zeros(frames), torch.zeros(frames)), -1))
    padded = motion.clone()
    padded[:, valid:] = torch.randn(2, frames - valid, 263) * 50
    lengths = torch.tensor([valid, valid])
    mask = sequence_mask(lengths, frames)
    a = model._acceleration_loss(motion, motion, mask)
    b = model._acceleration_loss(padded, padded, mask)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_stationary_contact_skating_near_zero():
    model = tiny_model()
    frames = 12
    motion = torch.zeros(2, frames, 263)
    motion[..., -4:] = 1.0
    mask = torch.ones(2, frames, dtype=torch.bool)
    loss, ratio = model._contact_skating_loss(motion, motion, mask)[:2]
    assert torch.isfinite(loss)
    assert float(loss) < 1e-6
    assert float(ratio) > 0.9


def test_contact_with_moving_feet_increases_skating():
    model = tiny_model()
    frames = 12
    still = torch.zeros(1, frames, 263)
    still[..., -4:] = 1.0
    moving = still.clone()
    time = torch.arange(frames, dtype=torch.float32)
    for joint in HML22_FOOT_JOINTS:
        _write_joint(moving, joint, torch.stack((0.2 * time, torch.zeros(frames), torch.zeros(frames)), -1))
    mask = torch.ones(1, frames, dtype=torch.bool)
    still_loss, _ = model._contact_skating_loss(still, still, mask)[:2]
    moving_loss, _ = model._contact_skating_loss(moving, still, mask)[:2]
    assert float(moving_loss) > float(still_loss) + 0.01


def test_zero_contact_skating_is_zero_not_nan():
    model = tiny_model()
    frames = 8
    pred = torch.randn(2, frames, 263)
    true = torch.randn(2, frames, 263)
    true[..., -4:] = 0
    mask = torch.ones(2, frames, dtype=torch.bool)
    loss, ratio = model._contact_skating_loss(pred, true, mask)[:2]
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert float(ratio) == 0.0


def test_padding_does_not_change_skating_loss():
    model = tiny_model()
    frames = 16
    valid = 10
    motion = torch.zeros(2, frames, 263)
    motion[..., -4:] = 1.0
    padded = motion.clone()
    padded[:, valid:] = torch.randn(2, frames - valid, 263) * 40
    padded[:, valid:, -4:] = 1.0
    mask = sequence_mask(torch.tensor([valid, valid]), frames)
    a, _ = model._contact_skating_loss(motion, motion, mask)[:2]
    b, _ = model._contact_skating_loss(padded, padded, mask)[:2]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_kinematic_threshold_contact_source():
    model = tiny_model(contact_source="kinematic_threshold", foot_height_threshold=0.05, foot_velocity_threshold=0.01)
    frames = 10
    motion = torch.zeros(1, frames, 263)
    for joint in HML22_FOOT_JOINTS:
        xyz = torch.zeros(frames, 3)
        xyz[:, 1] = 0.01
        _write_joint(motion, joint, xyz)
    mask = torch.ones(1, frames, dtype=torch.bool)
    loss, ratio = model._contact_skating_loss(motion, motion, mask)[:2]
    assert torch.isfinite(loss)
    assert float(ratio) > 0.5


def test_new_losses_off_do_not_change_total():
    torch.manual_seed(0)
    model = tiny_model(lambda_kin_acceleration=0.0, lambda_kin_skating=0.0)
    motion = torch.randn(2, 32, 263)
    lengths = torch.tensor([32, 20])
    text = torch.randn(2, 6, 48)
    text_mask = torch.ones(2, 6, dtype=torch.bool)
    losses = model.compute_losses(motion, lengths, text, text_mask)
    expected = (
        losses["flow"] + model.cfg.lambda_alignment * losses["alignment"]
        + model.cfg.lambda_part * losses["part"]
        + losses["weighted_kin_velocity"] + losses["weighted_kin_bone"] + losses["weighted_kin_foot"]
    )
    torch.testing.assert_close(losses["total"], expected)
    torch.testing.assert_close(losses["kin_acceleration"], torch.zeros(()))
    torch.testing.assert_close(losses["kin_skating"], torch.zeros(()))
    assert losses["weighted_kin_acceleration"].abs() == 0
    assert losses["weighted_kin_skating"].abs() == 0


def test_enabled_acceleration_is_finite_and_weighted():
    torch.manual_seed(1)
    model = tiny_model(lambda_kin_acceleration=0.05, lambda_kin_skating=0.0)
    motion = torch.randn(2, 32, 263)
    lengths = torch.tensor([32, 24])
    text = torch.randn(2, 6, 48)
    text_mask = torch.ones(2, 6, dtype=torch.bool)
    losses = model.compute_losses(motion, lengths, text, text_mask)
    assert torch.isfinite(losses["kin_acceleration"])
    torch.testing.assert_close(losses["weighted_kin_acceleration"], 0.05 * losses["kin_acceleration"])
    losses["total"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.flow.parameters())


def test_kit_skating_uses_kit_foot_indices():
    model = tiny_model(motion_dim=251, num_joints=21)
    frames = 10
    still = torch.zeros(1, frames, 251)
    still[..., -4:] = 1.0
    moving = still.clone()
    time = torch.arange(frames, dtype=torch.float32)
    for joint in KIT21_FOOT_JOINTS:
        _write_joint(moving, joint, torch.stack((0.2 * time, torch.zeros(frames), torch.zeros(frames)), -1))
    mask = torch.ones(1, frames, dtype=torch.bool)
    still_loss, _ = model._contact_skating_loss(still, still, mask)[:2]
    moving_loss, _ = model._contact_skating_loss(moving, still, mask)[:2]
    assert float(moving_loss) > float(still_loss) + 0.01


def test_physical_metrics_keys_are_not_guo_names():
    motion = torch.zeros(2, 12, 263)
    motion[..., -4:] = 1.0
    lengths = torch.tensor([12, 8])
    stats = physical_metrics(motion, motion, lengths, 22)
    assert "physical/foot_skating" in stats
    assert "physical/bone_error" in stats
    assert "physical/acceleration" in stats
    assert "FID" not in stats
    assert stats["physical/foot_skating"] < 1e-6


def test_skating_error_zero_contact():
    pos = torch.randn(2, 6, 22, 3)
    contact = torch.zeros(2, 6, 4, dtype=torch.bool)
    mask = torch.ones(2, 6, dtype=torch.bool)
    value = skating_error(pos, contact, mask, 22)
    assert torch.isfinite(value)
    assert float(value) == 0.0


def test_contact_valid_ratio_drops_when_contact_has_fast_feet():
    from models.motion_repr import contact_velocity_valid_ratio

    frames = 8
    pos = torch.zeros(1, frames, 22, 3)
    for joint in HML22_FOOT_JOINTS:
        pos[:, :, joint, 0] = torch.arange(frames, dtype=torch.float32) * 0.2
    contact = torch.ones(1, frames, 4, dtype=torch.bool)
    mask = torch.ones(1, frames, dtype=torch.bool)
    r_valid = contact_velocity_valid_ratio(pos, contact, mask, 22, vmax=0.01)
    assert float(r_valid) < 0.5
    still = torch.zeros(1, frames, 22, 3)
    r_still = contact_velocity_valid_ratio(still, contact, mask, 22, vmax=0.01)
    assert float(r_still) > 0.99
