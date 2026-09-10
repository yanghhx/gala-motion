import torch
import torch.nn.functional as F
from torch import nn


class TopologyMotionVAE(nn.Module):
    def __init__(self, motion_dim: int, graph_dim: int, latent_dim: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        hidden = latent_dim * 2
        self.encoder = nn.Sequential(
            nn.Conv1d(motion_dim + graph_dim, hidden, 5, stride=stride, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.mu = nn.Conv1d(hidden, latent_dim, 1)
        self.logvar = nn.Conv1d(hidden, latent_dim, 1)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, motion_dim, 3, padding=1),
        )

    def encode(self, motion, graph_features, sample=True):
        h = self.encoder(torch.cat((motion, graph_features), dim=-1).transpose(1, 2))
        mu, logvar = self.mu(h).transpose(1, 2), self.logvar(h).transpose(1, 2).clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return z, mu, logvar

    def decode(self, z, frames):
        x = F.interpolate(z.transpose(1, 2), size=frames, mode="linear", align_corners=False)
        return self.decoder(x).transpose(1, 2)

