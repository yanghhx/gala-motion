import torch

from models.gala_motion import GALAMotion, GALAMotionConfig
from models.gala_wm import GALAWorldConfig, GALAWorldModel


def tiny_wm(action_type="root4"):
    gala = GALAMotionConfig(
        motion_dim=263, num_joints=22, latent_dim=64, model_dim=64,
        text_dim=48, num_heads=4, graph_layers=1, dit_layers=1,
        max_frames=40, stride=4, train_stage="vae",
    )
    wm = GALAWorldConfig(
        history_latents=2, action_type=action_type, action_dim=4, idm_dim=8,
        use_flow=True, lambda_flow=0.1, lambda_idm=0.1, lambda_cycle=0.1 if action_type == "idm" else 0.0,
    )
    return GALAWorldModel(GALAMotion(gala), wm)


def test_wm_losses_finite_and_trainable():
    model = tiny_wm()
    motion = torch.randn(2, 40, 263)
    lengths = torch.tensor([40, 32])
    losses = model.compute_losses(motion, lengths)
    assert {"total", "mse", "rollout", "flow", "idm", "pose"} <= losses.keys()
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.backbone.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_rollout_shape_and_swap_changes_prediction():
    model = tiny_wm().eval()
    motion = torch.randn(2, 40, 263)
    lengths = torch.tensor([40, 40])
    with torch.no_grad():
        mu, mask, _ = model.encode_latents(motion, lengths)
        z_hist = mu[:, :2]
        actions = torch.randn(2, 3, model.action_dim)
        pred = model.rollout(z_hist, actions)
        swapped = actions.clone()
        swapped[..., 1] *= -1
        other = model.rollout(z_hist, swapped)
    assert pred.shape == (2, 3, 64)
    assert not torch.allclose(pred, other, atol=1e-5)


def test_film_and_gate_change_action_sensitivity():
    wm = GALAWorldConfig(
        history_latents=2, action_type="root4", action_dim=4, idm_dim=8,
        use_flow=False, residual=True, use_film=True, use_action_gate=True, rollout_horizon=2,
    )
    gala = GALAMotionConfig(
        motion_dim=263, num_joints=22, latent_dim=64, model_dim=64,
        text_dim=48, num_heads=4, graph_layers=1, dit_layers=1,
        max_frames=40, stride=4, train_stage="vae",
    )
    model = GALAWorldModel(GALAMotion(gala), wm)
    motion = torch.randn(2, 32, 263)
    lengths = torch.tensor([32, 32])
    losses = model.compute_losses(motion, lengths)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in model.predictor.film.parameters())
    assert any(p.grad is not None for p in model.predictor.gate.parameters())


def test_idm_variant_uses_latent_actions():
    model = tiny_wm("idm")
    assert model.action_dim == 8
    motion = torch.randn(2, 36, 263)
    lengths = torch.tensor([36, 36])
    losses = model.compute_losses(motion, lengths)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in model.idm.parameters())
