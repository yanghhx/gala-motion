import torch

from models.gala_motion import GALAMotion, GALAMotionConfig


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

