import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1)


class LanguageMotionAlignment(nn.Module):
    def __init__(self, text_dim: int, motion_dim: int, embed_dim: int):
        super().__init__()
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, embed_dim))
        self.motion_proj = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def embeddings(self, text, text_mask, motion, motion_mask):
        text_emb = F.normalize(self.text_proj(masked_mean(text, text_mask, 1)), dim=-1)
        motion_emb = F.normalize(self.motion_proj(masked_mean(motion, motion_mask, 1)), dim=-1)
        return text_emb, motion_emb

    def forward(self, text, text_mask, motion, motion_mask):
        text_emb, motion_emb = self.embeddings(text, text_mask, motion, motion_mask)
        logits = self.logit_scale.exp().clamp(max=100) * text_emb @ motion_emb.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, text_emb, motion_emb

