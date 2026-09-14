import torch
import pytest

from models.gala_motion import GALAMotion, GALAMotionConfig
from models.skeleton_graph import HML22_PARTS, KIT21_PARTS, PART_NAMES


def tiny_model():
    cfg = GALAMotionConfig(
        motion_dim=263,
        num_joints=22,
        latent_dim=64,
        model_dim=64,
        text_dim=48,
        num_heads=4,
        graph_layers=2,
        dit_layers=2,
        max_frames=40,
    )
    return GALAMotion(cfg)


def test_anatomical_query_order_is_fixed():
    assert PART_NAMES == ("torso", "left_arm", "right_arm", "left_leg", "right_leg")
    assert len(HML22_PARTS) == len(KIT21_PARTS) == len(PART_NAMES) == 5
    with pytest.raises(ValueError, match="query order is fixed"):
        GALAMotion(GALAMotionConfig(num_parts=4))


def test_part_attention_is_normalized_and_ordered():
    model = tiny_model()
    text = torch.randn(2, 7, 48)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    tokens, attention = model.part_alignment.part_text(text, mask, return_attention=True)
    assert tokens.shape == (2, len(PART_NAMES), 48)
    assert attention.shape == (2, len(PART_NAMES), 7)
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, len(PART_NAMES)))
    assert torch.count_nonzero(attention[0, :, 5:]) == 0


def test_joint_part_contrast_penalizes_slot_collapse():
    model = tiny_model()
    alignment = model.part_alignment
    text = torch.randn(3, 8, 48)
    mask = torch.ones(3, 8, dtype=torch.bool)
    distinct_motion = torch.randn(3, len(PART_NAMES), 64)
    loss, _ = alignment(distinct_motion, text, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert alignment.part_queries.grad is not None
    assert torch.count_nonzero(alignment.part_queries.grad) > 0


def test_forward_losses_are_finite_and_differentiable():
    model = tiny_model()
    motion = torch.randn(2, 40, 263)
    lengths = torch.tensor([40, 28])
    text = torch.randn(2, 12, 48)
    text_mask = torch.arange(12)[None] < torch.tensor([12, 8])[:, None]
    losses = model.compute_losses(motion, lengths, text, text_mask)
    assert {"total", "reconstruction", "kl", "alignment", "flow", "bone", "velocity"} <= losses.keys()
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_padding_does_not_change_motion_embedding():
    model = tiny_model().eval()
    motion = torch.randn(1, 40, 263)
    lengths = torch.tensor([24])
    changed = motion.clone()
    changed[:, 24:] = torch.randn_like(changed[:, 24:]) * 100
    with torch.no_grad():
        a = model.encode_motion_embedding(motion, lengths)
        b = model.encode_motion_embedding(changed, lengths)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_kit_forward_uses_21_joints():
    cfg = GALAMotionConfig(
        motion_dim=251,
        num_joints=21,
        latent_dim=64,
        model_dim=64,
        text_dim=48,
        num_heads=4,
        graph_layers=2,
        dit_layers=2,
        max_frames=40,
    )
    model = GALAMotion(cfg)
    motion = torch.randn(2, 32, 251)
    lengths = torch.tensor([32, 20])
    text = torch.randn(2, 4, 48)
    text_mask = torch.ones(2, 4, dtype=torch.bool)
    losses = model.compute_losses(motion, lengths, text, text_mask)
    assert torch.isfinite(losses["total"])


def test_vae_stage_skips_flow_gradients():
    cfg = GALAMotionConfig(
        motion_dim=263, num_joints=22, latent_dim=64, model_dim=64,
        text_dim=48, num_heads=4, graph_layers=2, dit_layers=2,
        max_frames=40, train_stage="vae",
    )
    model = GALAMotion(cfg)
    motion = torch.randn(2, 40, 263)
    lengths = torch.tensor([40, 28])
    text = torch.randn(2, 8, 48)
    text_mask = torch.ones(2, 8, dtype=torch.bool)
    losses = model.compute_losses(motion, lengths, text, text_mask)
    losses["total"].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.flow.parameters())
    assert any(p.grad is not None for p in model.vae.parameters())


def test_flow_stage_freezes_vae():
    cfg = GALAMotionConfig(
        motion_dim=263, num_joints=22, latent_dim=64, model_dim=64,
        text_dim=48, num_heads=4, graph_layers=2, dit_layers=2,
        max_frames=40, train_stage="flow",
    )
    model = GALAMotion(cfg)
    motion = torch.randn(2, 40, 263)
    lengths = torch.tensor([40, 28])
    text = torch.randn(2, 8, 48)
    text_mask = torch.ones(2, 8, dtype=torch.bool)
    losses = model.compute_losses(motion, lengths, text, text_mask)
    losses["total"].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.vae.parameters())
    assert any(p.grad is not None for p in model.flow.parameters())
    assert not any(p.requires_grad for p in model.vae.parameters())


def test_part_align_and_kinematic_flow_are_finite():
    cfg = GALAMotionConfig(
        motion_dim=263, num_joints=22, latent_dim=64, model_dim=64,
        text_dim=48, num_heads=4, graph_layers=2, dit_layers=2,
        max_frames=40, train_stage="flow", use_part_align=True, use_kinematic_flow=True,
    )
    model = GALAMotion(cfg)
    motion = torch.randn(2, 40, 263)
    lengths = torch.tensor([40, 28])
    text = torch.randn(2, 8, 48)
    text_mask = torch.ones(2, 8, dtype=torch.bool)
    losses = model.compute_losses(motion, lengths, text, text_mask)
    assert losses["part"].abs() > 0
    assert losses["kin_velocity"].abs() >= 0
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.part_alignment.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.flow.parameters())
    with torch.no_grad():
        sample = model.sample(text, text_mask, lengths, steps=2, guidance_scale=1.5)
    assert sample.shape == (2, 40, 263)


def test_conv_and_stgcn_tokenizers_run():
    motion = torch.randn(2, 32, 263)
    lengths = torch.tensor([32, 20])
    text = torch.randn(2, 4, 48)
    text_mask = torch.ones(2, 4, dtype=torch.bool)
    for graph_type, use_graph in (("none", False), ("stgcn", True)):
        cfg = GALAMotionConfig(
            motion_dim=263, num_joints=22, latent_dim=64, model_dim=64,
            text_dim=48, num_heads=4, graph_layers=2, dit_layers=1,
            max_frames=32, train_stage="vae", graph_type=graph_type, use_graph=use_graph,
        )
        model = GALAMotion(cfg)
        losses = model.compute_losses(motion, lengths, text, text_mask)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        assert any(p.grad is not None for p in model.vae.parameters())


def test_sampling_shape_and_padding():
    model = tiny_model().eval()
    text = torch.randn(2, 10, 48)
    text_mask = torch.ones(2, 10, dtype=torch.bool)
    lengths = torch.tensor([40, 21])
    with torch.no_grad():
        sample = model.sample(text, text_mask, lengths, steps=3, guidance_scale=1.5)
    assert sample.shape == (2, 40, 263)
    assert torch.count_nonzero(sample[1, 21:]) == 0


def test_ema_apply_restore_roundtrip():
    from trainers.ema import ModelEMA

    model = tiny_model()
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    after = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name in ema.shadow}
    ema.update(model)
    ema.apply(model)
    assert any(not torch.allclose(parameter, after[name]) for name, parameter in model.named_parameters() if name in after)
    ema.restore(model)
    for name, parameter in model.named_parameters():
        if name in after:
            torch.testing.assert_close(parameter, after[name])
