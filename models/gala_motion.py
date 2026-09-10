from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from diffusion.rectified_flow import RectifiedFlowDiT
from models.alignment import LanguageMotionAlignment, masked_mean
from models.motion_vae import TopologyMotionVAE
from models.skeleton_graph import skeleton_edges, SkeletonGraphEncoder
from models.text_encoder import TokenTextEncoder


@dataclass
class GALAMotionConfig:
    motion_dim: int = 263
    num_joints: int = 22
    latent_dim: int = 256
    model_dim: int = 512
    text_dim: int = 512
    num_heads: int = 8
    graph_layers: int = 4
    dit_layers: int = 12
    text_layers: int = 2
    text_encoder_type: str = "token"
    train_stage: str = "joint"
    max_frames: int = 196
    stride: int = 4
    use_graph: bool = True
    use_alignment: bool = True
    use_tokenizer: bool = True
    cond_drop_prob: float = 0.15
    lambda_reconstruction: float = 1.0
    lambda_kl: float = 1.0e-4
    lambda_alignment: float = 0.25
    lambda_bone: float = 0.1
    lambda_velocity: float = 0.1


def sequence_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


class GALAMotion(nn.Module):
    def __init__(self, cfg: GALAMotionConfig):
        super().__init__()
        self.cfg = cfg
        self.graph = SkeletonGraphEncoder(cfg.num_joints, cfg.latent_dim, cfg.graph_layers, cfg.num_heads)
        self.flat_graph = nn.Linear(cfg.motion_dim, cfg.latent_dim)
        self.vae = TopologyMotionVAE(cfg.motion_dim, cfg.latent_dim, cfg.latent_dim, cfg.stride)
        self.direct_encoder = nn.Conv1d(cfg.motion_dim + cfg.latent_dim, cfg.latent_dim, cfg.stride, stride=cfg.stride)
        self.decoder = self.vae.decoder
        self.text_encoder = TokenTextEncoder(cfg.text_dim, cfg.text_layers, min(cfg.num_heads, 8))
        self.alignment = LanguageMotionAlignment(cfg.text_dim, cfg.latent_dim, cfg.latent_dim)
        self.flow = RectifiedFlowDiT(
            cfg.latent_dim, cfg.model_dim, cfg.text_dim, cfg.num_heads, cfg.dit_layers,
            max_tokens=(cfg.max_frames + cfg.stride - 1) // cfg.stride,
        )
        self.apply_train_stage()

    def apply_train_stage(self):
        stage = self.cfg.train_stage
        if stage == "vae":
            for module in (self.flow, self.alignment, self.text_encoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        elif stage == "flow":
            for module in (self.vae, self.graph, self.flat_graph, self.direct_encoder, self.decoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def joint_positions(self, motion: torch.Tensor) -> torch.Tensor:
        b, t, _ = motion.shape
        local = motion[..., 4: 4 + (self.cfg.num_joints - 1) * 3]
        local = local.reshape(b, t, self.cfg.num_joints - 1, 3)
        root = torch.zeros(b, t, 1, 3, device=motion.device, dtype=motion.dtype)
        return torch.cat((root, local), dim=2)

    def forward(self, motion, lengths, text, text_mask):
        return self.compute_losses(motion, lengths, text, text_mask)

    def _features(self, motion, lengths):
        frame_mask = sequence_mask(lengths, motion.shape[1])
        clean_motion = motion * frame_mask.unsqueeze(-1)
        if self.cfg.use_graph:
            graph_frame, graph_joint = self.graph(self.joint_positions(clean_motion))
        else:
            graph_frame = self.flat_graph(clean_motion)
            graph_joint = graph_frame.unsqueeze(2).expand(-1, -1, self.cfg.num_joints, -1)
        return clean_motion, frame_mask, graph_frame, graph_joint

    def encode(self, motion, lengths):
        clean_motion, frame_mask, graph_frame, graph_joint = self._features(motion, lengths)
        if self.cfg.use_tokenizer:
            z, mu, logvar = self.vae.encode(clean_motion, graph_frame, sample=self.training)
        else:
            z = self.direct_encoder(torch.cat((clean_motion, graph_frame), -1).transpose(1, 2)).transpose(1, 2)
            mu, logvar = z, torch.zeros_like(z)
        latent_lengths = torch.div(lengths + self.cfg.stride - 1, self.cfg.stride, rounding_mode="floor")
        latent_mask = sequence_mask(latent_lengths, z.shape[1])
        z = z * latent_mask.unsqueeze(-1)
        return z, mu, logvar, frame_mask, latent_mask, graph_frame, graph_joint

    def decode(self, z, frames):
        if self.cfg.use_tokenizer:
            return self.vae.decode(z, frames)
        x = F.interpolate(z.transpose(1, 2), size=frames, mode="linear", align_corners=False)
        return self.decoder(x).transpose(1, 2)

    def encode_motion_embedding(self, motion, lengths):
        z, _, _, _, latent_mask, _, _ = self.encode(motion, lengths)
        dummy_text = torch.zeros(z.shape[0], 1, self.cfg.text_dim, device=z.device, dtype=z.dtype)
        dummy_mask = torch.ones(z.shape[0], 1, device=z.device, dtype=torch.bool)
        _, embedding = self.alignment.embeddings(dummy_text, dummy_mask, z, latent_mask)
        return embedding

    def encode_text(self, text, text_mask):
        return self.text_encoder(text, text_mask)

    def compute_losses(self, motion, lengths, text, text_mask):
        z, mu, logvar, frame_mask, latent_mask, _, _ = self.encode(motion, lengths)
        reconstruction = self.decode(z, motion.shape[1]) * frame_mask.unsqueeze(-1)
        valid_motion = frame_mask.unsqueeze(-1).expand_as(motion)
        rec_loss = ((reconstruction - motion).square() * valid_motion).sum() / valid_motion.sum().clamp_min(1)
        kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
        if self.cfg.train_stage == "vae":
            text_tokens = text
            alignment_loss = rec_loss.new_zeros(())
            flow_loss = rec_loss.new_zeros(())
        else:
            text_tokens = self.encode_text(text, text_mask)
            text_condition = masked_mean(text_tokens, text_mask, 1)
            if self.cfg.use_alignment:
                alignment_loss, _, _ = self.alignment(text_tokens, text_mask, z, latent_mask)
            else:
                alignment_loss = rec_loss.new_zeros(())
            flow_clean = mu.detach() if self.cfg.use_tokenizer else z
            flow_loss = self.flow.loss(
                flow_clean, text_condition, ~latent_mask,
                text_tokens=text_tokens, text_mask=text_mask,
                cond_drop_prob=self.cfg.cond_drop_prob,
            )
        pred_pos, true_pos = self.joint_positions(reconstruction), self.joint_positions(motion)
        bone_terms = []
        for i, j in skeleton_edges(self.cfg.num_joints):
            pred_length = (pred_pos[:, :, i] - pred_pos[:, :, j]).norm(dim=-1)
            true_length = (true_pos[:, :, i] - true_pos[:, :, j]).norm(dim=-1)
            bone_terms.append((pred_length - true_length).abs())
        bone_loss = (torch.stack(bone_terms, -1) * frame_mask.unsqueeze(-1)).sum() / frame_mask.sum().clamp_min(1) / len(bone_terms)
        pred_velocity = torch.diff(reconstruction, dim=1)
        true_velocity = torch.diff(motion, dim=1)
        velocity_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
        valid_velocity = velocity_mask.unsqueeze(-1).expand_as(pred_velocity)
        velocity_loss = ((pred_velocity - true_velocity).abs() * valid_velocity).sum() / valid_velocity.sum().clamp_min(1)
        if self.cfg.train_stage == "vae":
            total = (
                self.cfg.lambda_reconstruction * rec_loss + self.cfg.lambda_kl * kl_loss
                + self.cfg.lambda_bone * bone_loss + self.cfg.lambda_velocity * velocity_loss
            )
        elif self.cfg.train_stage == "flow":
            total = flow_loss + self.cfg.lambda_alignment * alignment_loss
        else:
            total = (
                flow_loss + self.cfg.lambda_reconstruction * rec_loss + self.cfg.lambda_kl * kl_loss
                + self.cfg.lambda_alignment * alignment_loss + self.cfg.lambda_bone * bone_loss
                + self.cfg.lambda_velocity * velocity_loss
            )
        return {
            "total": total, "reconstruction": rec_loss, "kl": kl_loss,
            "alignment": alignment_loss, "flow": flow_loss, "bone": bone_loss,
            "velocity": velocity_loss,
        }

    def sample(self, text, text_mask, lengths, steps=30, guidance_scale=2.5):
        text_tokens = self.encode_text(text, text_mask)
        text_condition = masked_mean(text_tokens, text_mask, 1)
        max_frames = int(lengths.max().item())
        tokens = (max_frames + self.cfg.stride - 1) // self.cfg.stride
        latent_lengths = torch.div(lengths + self.cfg.stride - 1, self.cfg.stride, rounding_mode="floor")
        latent_mask = sequence_mask(latent_lengths, tokens)
        z = self.flow.sample(
            (text.shape[0], tokens, self.cfg.latent_dim), text_condition, ~latent_mask,
            steps, guidance_scale, text_tokens=text_tokens, text_mask=text_mask,
        )
        motion = self.decode(z, max_frames)
        return motion * sequence_mask(lengths, max_frames).unsqueeze(-1)
