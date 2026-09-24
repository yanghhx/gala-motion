from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from diffusion.rectified_flow import RectifiedFlowDiT
from models.alignment import LanguageMotionAlignment, PartLanguageAlignment, masked_mean
from models.motion_repr import (
    contact_mask_for_source,
    contact_velocity_valid_ratio,
    denormalize_motion,
    paired_acceleration_error,
    recover_from_ric,
    skating_error,
)
from models.motion_vae import TopologyMotionVAE
from models.skeleton_graph import PART_NAMES, body_parts, pool_body_parts, skeleton_edges, SkeletonGraphEncoder
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
    graph_type: str = "ctr"
    use_temporal_graph: bool = False
    temporal_heads: int = 4
    temporal_max_distance: int = 32
    temporal_dropout: float = 0.1
    use_alignment: bool = True
    use_part_align: bool = False
    use_anatomical_anchor: bool = False
    anchor_weight: float = 0.05
    use_kinematic_flow: bool = False
    use_tokenizer: bool = True
    cond_drop_prob: float = 0.15
    lambda_reconstruction: float = 1.0
    lambda_kl: float = 1.0e-4
    lambda_alignment: float = 0.25
    lambda_part: float = 0.25
    part_diversity_weight: float = 0.05
    query_diversity_weight: float = 0.01
    lambda_bone: float = 0.1
    lambda_velocity: float = 0.1
    lambda_kin_velocity: float = 0.05
    lambda_kin_bone: float = 0.05
    lambda_kin_foot: float = 0.05
    lambda_kin_acceleration: float = 0.0
    lambda_kin_skating: float = 0.0
    adaptive_aux_weight: bool = False
    aux_target_ratio: float = 0.15
    aux_ratio_decay: bool = False
    aux_ratio_start: float = 0.03
    aux_ratio_end: float = 0.01
    aux_decay_steps: int = 40000
    contact_source: str = "gt_contact_feature"
    foot_height_threshold: float = 0.05
    foot_velocity_threshold: float = 0.01
    num_parts: int = 5
    foot_dim: int = 4


def sequence_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


class GALAMotion(nn.Module):
    def __init__(self, cfg: GALAMotionConfig):
        super().__init__()
        self.cfg = cfg
        self.parts = body_parts(cfg.num_joints)
        if cfg.num_parts != len(PART_NAMES) or len(self.parts) != len(PART_NAMES):
            raise ValueError(
                f"Anatomical query order is fixed to {PART_NAMES}; "
                f"got num_parts={cfg.num_parts} and {len(self.parts)} graph parts"
            )
        graph_kind = "none" if not cfg.use_graph else cfg.graph_type
        if cfg.use_temporal_graph and graph_kind == "ctr":
            graph_kind = "st_ctr"
        self.graph = SkeletonGraphEncoder(
            cfg.num_joints, cfg.latent_dim, cfg.graph_layers, cfg.num_heads,
            kind="stgcn" if graph_kind == "stgcn" else graph_kind,
            temporal_heads=cfg.temporal_heads,
            temporal_max_distance=cfg.temporal_max_distance,
            temporal_dropout=cfg.temporal_dropout,
        )
        self.flat_graph = nn.Linear(cfg.motion_dim, cfg.latent_dim)
        self.vae = TopologyMotionVAE(cfg.motion_dim, cfg.latent_dim, cfg.latent_dim, cfg.stride)
        self.direct_encoder = nn.Conv1d(cfg.motion_dim + cfg.latent_dim, cfg.latent_dim, cfg.stride, stride=cfg.stride)
        self.decoder = self.vae.decoder
        self.text_encoder = TokenTextEncoder(cfg.text_dim, cfg.text_layers, min(cfg.num_heads, 8))
        self.alignment = LanguageMotionAlignment(cfg.text_dim, cfg.latent_dim, cfg.latent_dim)
        self.part_alignment = PartLanguageAlignment(
            cfg.text_dim, cfg.latent_dim, cfg.latent_dim, num_parts=cfg.num_parts,
            diversity_weight=cfg.part_diversity_weight,
            query_diversity_weight=cfg.query_diversity_weight,
            use_anatomical_anchor=cfg.use_anatomical_anchor,
            anchor_weight=cfg.anchor_weight,
        )
        self.flow = RectifiedFlowDiT(
            cfg.latent_dim, cfg.model_dim, cfg.text_dim, cfg.num_heads, cfg.dit_layers,
            max_tokens=(cfg.max_frames + cfg.stride - 1) // cfg.stride,
        )
        # Dataset Mean/Std are attributes, not buffers, so old checkpoints still load strictly.
        self.motion_mean = None
        self.motion_std = None
        self.apply_train_stage()

    def set_motion_stats(self, mean, std):
        self.motion_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.motion_std = torch.as_tensor(std, dtype=torch.float32)

    @property
    def uses_graph(self) -> bool:
        return self.cfg.use_graph and self.cfg.graph_type != "none"

    def apply_train_stage(self):
        stage = self.cfg.train_stage
        if stage == "vae":
            for module in (self.flow, self.alignment, self.part_alignment, self.text_encoder):
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
        if self.uses_graph:
            graph_frame, graph_joint = self.graph(self.joint_positions(clean_motion), frame_mask)
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

    def _condition_text(self, text, text_mask, graph_joint=None, frame_mask=None):
        text_tokens = self.encode_text(text, text_mask)
        part_loss = text_tokens.new_zeros(())
        part_tokens = None
        if self.cfg.use_part_align:
            part_motion = None
            if graph_joint is not None and frame_mask is not None:
                part_motion = pool_body_parts(graph_joint, frame_mask, self.parts).detach()
            if part_motion is not None and self.training:
                part_loss, part_tokens = self.part_alignment(part_motion, text_tokens, text_mask)
            else:
                part_tokens = self.part_alignment.part_text(text_tokens, text_mask)
            part_mask = torch.ones(text_tokens.shape[0], part_tokens.shape[1], device=text.device, dtype=torch.bool)
            text_tokens = torch.cat((text_tokens, part_tokens), dim=1)
            text_mask = torch.cat((text_mask, part_mask), dim=1)
        return text_tokens, text_mask, part_loss

    def _bone_loss(self, pred_motion, true_motion, frame_mask):
        pred_pos, true_pos = self.joint_positions(pred_motion), self.joint_positions(true_motion)
        bone_terms = []
        for i, j in skeleton_edges(self.cfg.num_joints):
            pred_length = (pred_pos[:, :, i] - pred_pos[:, :, j]).norm(dim=-1)
            true_length = (true_pos[:, :, i] - true_pos[:, :, j]).norm(dim=-1)
            bone_terms.append((pred_length - true_length).abs())
        stacked = torch.stack(bone_terms, -1) * frame_mask.unsqueeze(-1)
        return stacked.sum() / frame_mask.sum().clamp_min(1) / len(bone_terms)

    def _velocity_loss(self, pred_motion, true_motion, frame_mask):
        # Feature-space L1 on t -> t+1. Both endpoints of a transition must be valid,
        # so padding and the padded/valid boundary are excluded. First frame has no
        # incoming difference (torch.diff); last valid pair is included.
        pred_velocity = torch.diff(pred_motion, dim=1)
        true_velocity = torch.diff(true_motion, dim=1)
        velocity_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
        valid_velocity = velocity_mask.unsqueeze(-1).expand_as(pred_velocity)
        return ((pred_velocity - true_velocity).abs() * valid_velocity).sum() / valid_velocity.sum().clamp_min(1)

    def _acceleration_loss(self, pred_motion, true_motion, frame_mask):
        """L1 on RIC-local joint acceleration. Same joint convention as bone loss.

        a_t = p_{t+1} - 2 p_t + p_{t-1}, only on three consecutive valid frames.
        """
        pred_pos = self.joint_positions(pred_motion)
        true_pos = self.joint_positions(true_motion)
        return paired_acceleration_error(pred_pos, true_pos, frame_mask)

    def _foot_feature_loss(self, pred_motion, true_motion, frame_mask):
        """L1 on the last Guo foot-contact channels. Not a skating / physics loss."""
        dim = min(self.cfg.foot_dim, pred_motion.shape[-1])
        pred_foot, true_foot = pred_motion[..., -dim:], true_motion[..., -dim:]
        valid = frame_mask.unsqueeze(-1).expand_as(pred_foot)
        return ((pred_foot - true_foot).abs() * valid).sum() / valid.sum().clamp_min(1)

    def _foot_loss(self, pred_motion, true_motion, frame_mask):
        return self._foot_feature_loss(pred_motion, true_motion, frame_mask)

    def _world_positions(self, motion):
        denorm = denormalize_motion(motion, self.motion_mean, self.motion_std)
        return recover_from_ric(denorm, self.cfg.num_joints)

    def _contact_skating_loss(self, pred_motion, true_motion, frame_mask):
        pred_pos = self._world_positions(pred_motion)
        true_pos = self._world_positions(true_motion)
        contact = contact_mask_for_source(
            self.cfg.contact_source, true_motion, true_pos, frame_mask, self.cfg.num_joints,
            mean=self.motion_mean, std=self.motion_std,
            height_threshold=self.cfg.foot_height_threshold,
            velocity_threshold=self.cfg.foot_velocity_threshold,
        )
        loss = skating_error(pred_pos, contact, frame_mask, self.cfg.num_joints)
        trans = (frame_mask[:, 1:] & frame_mask[:, :-1]).unsqueeze(-1).to(loss.dtype)
        active = contact[:, :-1].to(loss.dtype) * trans
        ratio = active.sum() / trans.expand_as(active).sum().clamp_min(1)
        r_valid = contact_velocity_valid_ratio(
            true_pos, contact, frame_mask, self.cfg.num_joints, self.cfg.foot_velocity_threshold,
        )
        return loss, ratio.detach(), r_valid.detach()

    def compute_losses(self, motion, lengths, text, text_mask):
        z, mu, logvar, frame_mask, latent_mask, _, graph_joint = self.encode(motion, lengths)
        reconstruction = self.decode(z, motion.shape[1]) * frame_mask.unsqueeze(-1)
        valid_motion = frame_mask.unsqueeze(-1).expand_as(motion)
        rec_loss = ((reconstruction - motion).square() * valid_motion).sum() / valid_motion.sum().clamp_min(1)
        kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
        bone_loss = self._bone_loss(reconstruction, motion, frame_mask)
        velocity_loss = self._velocity_loss(reconstruction, motion, frame_mask)
        part_loss = rec_loss.new_zeros(())
        kin_velocity = rec_loss.new_zeros(())
        kin_bone = rec_loss.new_zeros(())
        kin_foot = rec_loss.new_zeros(())
        kin_acceleration = rec_loss.new_zeros(())
        kin_skating = rec_loss.new_zeros(())
        contact_ratio = rec_loss.new_zeros(())
        contact_valid_ratio = rec_loss.new_zeros(())
        if self.cfg.train_stage == "vae":
            alignment_loss = rec_loss.new_zeros(())
            flow_loss = rec_loss.new_zeros(())
        else:
            text_tokens, cond_mask, part_loss = self._condition_text(text, text_mask, graph_joint, frame_mask)
            text_condition = masked_mean(text_tokens, cond_mask, 1)
            if self.cfg.use_alignment:
                alignment_loss, _, _ = self.alignment(text_tokens[:, : text.shape[1]], text_mask, z, latent_mask)
            else:
                alignment_loss = rec_loss.new_zeros(())
            flow_clean = mu.detach() if self.cfg.use_tokenizer else z
            flow_out = self.flow.loss(
                flow_clean, text_condition, ~latent_mask,
                text_tokens=text_tokens, text_mask=cond_mask,
                cond_drop_prob=self.cfg.cond_drop_prob,
                return_endpoint=self.cfg.use_kinematic_flow,
            )
            if self.cfg.use_kinematic_flow:
                flow_loss, endpoint = flow_out
                pred_motion = self.decode(endpoint, motion.shape[1]) * frame_mask.unsqueeze(-1)
                kin_velocity = self._velocity_loss(pred_motion, motion, frame_mask)
                kin_bone = self._bone_loss(pred_motion, motion, frame_mask)
                kin_foot = self._foot_feature_loss(pred_motion, motion, frame_mask)
                if self.cfg.lambda_kin_acceleration > 0:
                    kin_acceleration = self._acceleration_loss(pred_motion, motion, frame_mask)
                if self.cfg.lambda_kin_skating > 0:
                    kin_skating, contact_ratio, contact_valid_ratio = self._contact_skating_loss(
                        pred_motion, motion, frame_mask,
                    )
            else:
                flow_loss = flow_out
        weighted_kin_velocity = self.cfg.lambda_kin_velocity * kin_velocity
        weighted_kin_bone = self.cfg.lambda_kin_bone * kin_bone
        weighted_kin_foot = self.cfg.lambda_kin_foot * kin_foot
        flow_scale = flow_loss.detach().abs().clamp_min(1e-8)
        eff_lambda_acc = self.cfg.lambda_kin_acceleration
        eff_lambda_skate = self.cfg.lambda_kin_skating
        if self.cfg.adaptive_aux_weight and self.cfg.train_stage != "vae":
            eta = self.cfg.aux_target_ratio
            if self.cfg.aux_ratio_decay:
                progress = min(getattr(self, "current_opt_step", 0) / max(self.cfg.aux_decay_steps, 1), 1.0)
                eta = self.cfg.aux_ratio_start + (self.cfg.aux_ratio_end - self.cfg.aux_ratio_start) * progress
            aux_scale_acc = kin_acceleration.detach().abs().clamp_min(1e-8)
            aux_scale_skate = kin_skating.detach().abs().clamp_min(1e-8)
            eff_lambda_acc = eta * flow_scale / aux_scale_acc if self.cfg.lambda_kin_acceleration > 0 else 0.0
            eff_lambda_skate = eta * flow_scale / aux_scale_skate if self.cfg.lambda_kin_skating > 0 else 0.0
        weighted_kin_acceleration = eff_lambda_acc * kin_acceleration
        weighted_kin_skating = eff_lambda_skate * kin_skating
        if self.cfg.train_stage == "vae":
            total = (
                self.cfg.lambda_reconstruction * rec_loss + self.cfg.lambda_kl * kl_loss
                + self.cfg.lambda_bone * bone_loss + self.cfg.lambda_velocity * velocity_loss
            )
        elif self.cfg.train_stage == "flow":
            total = (
                flow_loss + self.cfg.lambda_alignment * alignment_loss
                + self.cfg.lambda_part * part_loss
                + weighted_kin_velocity + weighted_kin_bone + weighted_kin_foot
                + weighted_kin_acceleration + weighted_kin_skating
            )
        else:
            total = (
                flow_loss + self.cfg.lambda_reconstruction * rec_loss + self.cfg.lambda_kl * kl_loss
                + self.cfg.lambda_alignment * alignment_loss + self.cfg.lambda_part * part_loss
                + self.cfg.lambda_bone * bone_loss + self.cfg.lambda_velocity * velocity_loss
                + weighted_kin_velocity + weighted_kin_bone + weighted_kin_foot
                + weighted_kin_acceleration + weighted_kin_skating
            )
        return {
            "total": total, "reconstruction": rec_loss, "kl": kl_loss,
            "alignment": alignment_loss, "part": part_loss, "flow": flow_loss,
            "bone": bone_loss, "velocity": velocity_loss,
            "kin_velocity": kin_velocity, "kin_bone": kin_bone, "kin_foot": kin_foot,
            "kin_acceleration": kin_acceleration, "kin_skating": kin_skating,
            "contact_ratio": contact_ratio,
            "contact_valid_ratio": contact_valid_ratio,
            "weighted_kin_velocity": weighted_kin_velocity,
            "weighted_kin_bone": weighted_kin_bone,
            "weighted_kin_foot": weighted_kin_foot,
            "weighted_kin_acceleration": weighted_kin_acceleration,
            "weighted_kin_skating": weighted_kin_skating,
            "acc_over_flow": (weighted_kin_acceleration.detach() / flow_scale),
            "skate_over_flow": (weighted_kin_skating.detach() / flow_scale),
            "eff_lambda_acc": weighted_kin_acceleration.new_tensor(eff_lambda_acc) if not isinstance(eff_lambda_acc, torch.Tensor) else eff_lambda_acc.detach(),
            "eff_lambda_skate": weighted_kin_skating.new_tensor(eff_lambda_skate) if not isinstance(eff_lambda_skate, torch.Tensor) else eff_lambda_skate.detach(),
        }

    def sample(self, text, text_mask, lengths, steps=30, guidance_scale=2.5):
        text_tokens, cond_mask, _ = self._condition_text(text, text_mask)
        text_condition = masked_mean(text_tokens, cond_mask, 1)
        max_frames = int(lengths.max().item())
        tokens = (max_frames + self.cfg.stride - 1) // self.cfg.stride
        latent_lengths = torch.div(lengths + self.cfg.stride - 1, self.cfg.stride, rounding_mode="floor")
        latent_mask = sequence_mask(latent_lengths, tokens)
        z = self.flow.sample(
            (text.shape[0], tokens, self.cfg.latent_dim), text_condition, ~latent_mask,
            steps, guidance_scale, text_tokens=text_tokens, text_mask=cond_mask,
        )
        motion = self.decode(z, max_frames)
        return motion * sequence_mask(lengths, max_frames).unsqueeze(-1)
