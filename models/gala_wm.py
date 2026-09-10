from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from diffusion.rectified_flow import RectifiedFlowDiT
from evaluation.wm_metrics import local_joints
from models.gala_motion import GALAMotion, GALAMotionConfig, sequence_mask
from models.skeleton_graph import skeleton_edges


@dataclass
class GALAWorldConfig:
    history_latents: int = 4
    action_type: str = "root4"
    action_dim: int = 4
    idm_dim: int = 16
    cond_drop_prob: float = 0.1
    lambda_mse: float = 1.0
    lambda_rollout: float = 1.0
    lambda_flow: float = 0.1
    lambda_idm: float = 0.1
    lambda_cycle: float = 0.1
    lambda_bone: float = 0.01
    lambda_velocity: float = 0.01
    lambda_pose: float = 0.0
    use_flow: bool = True
    residual: bool = True
    rollout_horizon: int = 4
    predictor_layers: int = 2
    use_film: bool = False
    use_action_gate: bool = False


class HistActionPredictor(nn.Module):
    """Deterministic next-latent predictor used for rollout and CEM."""

    def __init__(self, latent_dim: int, model_dim: int, action_dim: int, heads: int, layers: int,
                 use_film: bool = False, use_gate: bool = False):
        super().__init__()
        self.use_film = use_film
        self.use_gate = use_gate
        self.z_proj = nn.Linear(latent_dim, model_dim)
        self.a_proj = nn.Linear(action_dim, model_dim)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, model_dim * 2)) if use_film else None
        self.gate = nn.Sequential(nn.Linear(action_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, 1)) if use_gate else None
        if self.gate is not None:
            nn.init.constant_(self.gate[-1].bias, -1.5)
        encoder = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=model_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=max(layers, 1))
        self.out = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim))

    def forward(self, z_hist: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = self.z_proj(z_hist)
        if self.film is not None:
            scale, shift = self.film(action).chunk(2, dim=-1)
            hidden = hidden * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tokens = torch.cat((hidden, self.a_proj(action).unsqueeze(1)), dim=1)
        delta = self.out(self.encoder(tokens)[:, -1])
        if self.gate is not None:
            delta = torch.sigmoid(self.gate(action)) * delta
        return delta


class InverseDynamics(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((z_t, z_next), dim=-1))


class GALAWorldModel(nn.Module):
    def __init__(self, backbone: GALAMotion, wm: GALAWorldConfig):
        super().__init__()
        self.backbone = backbone
        self.wm = wm
        self.cfg = backbone.cfg
        action_dim = wm.idm_dim if wm.action_type == "idm" else wm.action_dim
        self.action_dim = action_dim
        self.predictor = HistActionPredictor(
            self.cfg.latent_dim, self.cfg.model_dim, action_dim, self.cfg.num_heads,
            wm.predictor_layers, use_film=wm.use_film, use_gate=wm.use_action_gate,
        )
        self.idm = InverseDynamics(self.cfg.latent_dim, self.cfg.model_dim, action_dim)
        self.action_embed = nn.Linear(action_dim, self.cfg.model_dim)
        self.hist_proj = nn.Linear(self.cfg.latent_dim, self.cfg.model_dim)
        self.flow = RectifiedFlowDiT(
            self.cfg.latent_dim, self.cfg.model_dim, self.cfg.model_dim,
            self.cfg.num_heads, self.cfg.dit_layers, max_tokens=1,
        ) if wm.use_flow else None
        self.freeze_backbone()

    def freeze_backbone(self):
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_latents(self, motion: torch.Tensor, lengths: torch.Tensor):
        z, mu, logvar, frame_mask, latent_mask, _, _ = self.backbone.encode(motion, lengths)
        mu = mu.detach() * latent_mask.unsqueeze(-1)
        return mu, latent_mask, frame_mask

    def root4_actions(self, motion: torch.Tensor, latent_len: int) -> torch.Tensor:
        stride = self.cfg.stride
        need = latent_len * stride
        if motion.shape[1] < need:
            motion = torch.nn.functional.pad(motion, (0, 0, 0, need - motion.shape[1]))
        root = motion[:, :need, :4]
        return root.reshape(motion.shape[0], latent_len, stride, 4).mean(dim=2)

    def sample_windows(self, mu: torch.Tensor, latent_mask: torch.Tensor, actions: torch.Tensor):
        history = self.wm.history_latents
        horizon = max(int(self.wm.rollout_horizon), 1)
        lengths = latent_mask.sum(dim=1).long()
        low = torch.full_like(lengths, history)
        high = (lengths - horizon + 1).clamp_min(history + 1)
        span = (high - low).clamp_min(1).float()
        nxt = (low.float() + torch.rand(mu.shape[0], device=mu.device) * span).long()
        nxt = torch.minimum(nxt, (lengths - horizon).clamp_min(history)).clamp_min(history)
        nxt = nxt.clamp(max=mu.shape[1] - horizon)
        batch = torch.arange(mu.shape[0], device=mu.device)
        hist_off = torch.arange(history, device=mu.device)
        fut_off = torch.arange(horizon, device=mu.device)
        z_hist = mu[batch[:, None], nxt[:, None] - history + hist_off]
        z_future = mu[batch[:, None], nxt[:, None] + fut_off]
        z_prev = mu[batch, nxt - 1]
        act = actions[batch[:, None], nxt[:, None] + fut_off - 1]
        return z_hist, z_prev, z_future, act

    def resolve_action(self, z_prev: torch.Tensor, z_next: torch.Tensor, explicit: torch.Tensor):
        if self.wm.action_type == "idm":
            predicted = self.idm(z_prev, z_next)
            return predicted, predicted, explicit
        predicted = self.idm(z_prev, z_next)
        return explicit, predicted, explicit

    def predict_next(self, z_hist: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.predictor(z_hist, action)
        if self.wm.residual:
            return z_hist[:, -1] + delta
        return delta

    def rollout(self, z_hist: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """actions: [B, H, A] -> latents [B, H, D] starting after z_hist."""
        hist = z_hist
        preds = []
        for step in range(actions.shape[1]):
            nxt = self.predict_next(hist, actions[:, step])
            preds.append(nxt)
            hist = torch.cat((hist[:, 1:], nxt.unsqueeze(1)), dim=1)
        return torch.stack(preds, dim=1)

    def compute_losses(self, motion: torch.Tensor, lengths: torch.Tensor):
        mu, latent_mask, frame_mask = self.encode_latents(motion, lengths)
        latent_len = mu.shape[1]
        explicit = self.root4_actions(motion, latent_len)
        z_hist, z_prev, z_future, act_seq = self.sample_windows(mu, latent_mask, explicit)
        z_next = z_future[:, 0]
        if self.wm.action_type == "idm":
            prevs = torch.cat((z_prev.unsqueeze(1), z_future[:, :-1]), dim=1)
            idm_seq = self.idm(prevs.reshape(-1, prevs.shape[-1]), z_future.reshape(-1, z_future.shape[-1]))
            idm_seq = idm_seq.reshape(z_future.shape[0], z_future.shape[1], -1)
            action, idm_pred, idm_target = idm_seq[:, 0], idm_seq[:, 0], act_seq[:, 0]
            driven = action.detach()
            driven_seq = idm_seq.detach()
        else:
            explicit_a = act_seq[:, 0]
            action, idm_pred, idm_target = self.resolve_action(z_prev, z_next.detach(), explicit_a)
            driven = action
            driven_seq = act_seq
        if self.training and self.wm.cond_drop_prob > 0:
            drop = torch.rand(driven.shape[0], device=driven.device) < self.wm.cond_drop_prob
            driven = driven.clone()
            driven[drop] = 0
        pred = self.predict_next(z_hist, driven)
        mse = (pred - z_next).square().mean()
        preds = self.rollout(z_hist, driven_seq)
        rollout_mse = (preds - z_future).square().mean()
        idm_loss = (idm_pred - idm_target.detach()).square().mean() if self.wm.action_type != "idm" else pred.new_zeros(())
        if self.wm.action_type == "idm":
            cycle = (self.predict_next(z_hist, action.detach()) - z_next).square().mean()
            idm_loss = action.square().mean() * 1e-4
        else:
            cycle = pred.new_zeros(())
        flow_loss = pred.new_zeros(())
        if self.flow is not None and self.wm.lambda_flow > 0:
            cond = self.action_embed(driven)
            hist_tokens = self.hist_proj(z_hist)
            hist_mask = torch.ones(z_hist.shape[:2], device=z_hist.device, dtype=torch.bool)
            pad = torch.zeros(z_next.shape[0], 1, device=z_next.device, dtype=torch.bool)
            flow_loss = self.flow.loss(
                z_next.unsqueeze(1), cond, pad,
                text_tokens=hist_tokens, text_mask=hist_mask,
                cond_drop_prob=0.0,
            )
        recon = self.backbone.decode(pred.unsqueeze(1), self.cfg.stride)
        target = self.backbone.decode(z_next.unsqueeze(1), self.cfg.stride)
        pred_pos, true_pos = local_joints(recon, self.cfg.num_joints), local_joints(target, self.cfg.num_joints)
        bone_terms = []
        for i, j in skeleton_edges(self.cfg.num_joints):
            bone_terms.append(
                ((pred_pos[:, :, i] - pred_pos[:, :, j]).norm(dim=-1)
                 - (true_pos[:, :, i] - true_pos[:, :, j]).norm(dim=-1)).abs()
            )
        bone = torch.stack(bone_terms, -1).mean()
        velocity = (recon.diff(dim=1) - target.diff(dim=1)).abs().mean() if recon.shape[1] > 1 else pred.new_zeros(())
        pose = (recon - target).square().mean()
        total = (
            self.wm.lambda_mse * mse
            + self.wm.lambda_rollout * rollout_mse
            + self.wm.lambda_flow * flow_loss
            + self.wm.lambda_idm * idm_loss
            + self.wm.lambda_cycle * cycle
            + self.wm.lambda_bone * bone
            + self.wm.lambda_velocity * velocity
            + self.wm.lambda_pose * pose
        )
        return {
            "total": total, "mse": mse, "rollout": rollout_mse, "flow": flow_loss, "idm": idm_loss,
            "cycle": cycle, "bone": bone, "velocity": velocity, "pose": pose,
        }

    def forward(self, motion, lengths, *args, **kwargs):
        return self.compute_losses(motion, lengths)


def build_world_model(gala_cfg: GALAMotionConfig, wm_cfg: GALAWorldConfig, checkpoint: str | None, device):
    backbone = GALAMotion(gala_cfg)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        raw = state["model"] if isinstance(state, dict) and "model" in state else state
        backbone.load_state_dict(raw, strict=False)
    backbone.to(device)
    return GALAWorldModel(backbone, wm_cfg).to(device)
