import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1)


def _symmetric_nce(a: torch.Tensor, b: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp().clamp(max=100) * a @ b.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


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
        return _symmetric_nce(text_emb, motion_emb, self.logit_scale), text_emb, motion_emb


ANATOMICAL_PROMPTS = (
    "torso, body and head movement",
    "left arm, left hand and left shoulder movement",
    "right arm, right hand and right shoulder movement",
    "left leg and left foot movement",
    "right leg and right foot movement",
)


class PartLanguageAlignment(nn.Module):
    """Learnable anatomical queries attend to text tokens and match body-part latents.

    With ``use_anatomical_anchor=True``, learnable queries are regularised toward
    precomputed embeddings of anatomical prompts, and a gated anchor signal is
    mixed into the query at attention time:

        q_p^anchor = q_p^learn + sigmoid(g_p) * W_a * a_p
        L_anchor   = mean_p [1 - cos(W_q q_p, W_a a_p)]
    """

    def __init__(
        self, text_dim: int, motion_dim: int, embed_dim: int, num_parts: int = 5,
        diversity_weight: float = 0.05, query_diversity_weight: float = 0.01,
        use_anatomical_anchor: bool = False, anchor_weight: float = 0.05,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.diversity_weight = diversity_weight
        self.query_diversity_weight = query_diversity_weight
        self.use_anatomical_anchor = use_anatomical_anchor
        self.anchor_weight = anchor_weight
        self.part_queries = nn.Parameter(torch.randn(num_parts, text_dim) * 0.02)
        self.query_proj = nn.Linear(text_dim, embed_dim)
        self.key_proj = nn.Linear(text_dim, embed_dim)
        self.value_proj = nn.Linear(text_dim, embed_dim)
        self.part_out = nn.Linear(embed_dim, text_dim)
        self.motion_proj = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, embed_dim))
        self.lang_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        if use_anatomical_anchor:
            self.anchor_proj = nn.Linear(text_dim, text_dim)
            self.anchor_to_embed = nn.Linear(text_dim, embed_dim)
            self.anchor_gate = nn.Parameter(torch.zeros(num_parts))
            self.register_buffer("anchor_embeddings", torch.zeros(num_parts, text_dim))

    def set_anchor_embeddings(self, embeddings: torch.Tensor):
        if not self.use_anatomical_anchor:
            return
        self.anchor_embeddings = embeddings.clone().detach()

    def _anchored_queries(self) -> torch.Tensor:
        """Part queries with gated anchor signal mixed in."""
        if not self.use_anatomical_anchor:
            return self.part_queries
        gate = torch.sigmoid(self.anchor_gate)
        return self.part_queries + gate[:, None] * self.anchor_proj(self.anchor_embeddings)

    def part_text(
        self, text: torch.Tensor, text_mask: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = text.shape[0]
        query = self.query_proj(self._anchored_queries()).unsqueeze(0).expand(batch, -1, -1)
        key = self.key_proj(text)
        value = self.value_proj(text)
        logits = torch.matmul(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
        logits = logits.masked_fill(~text_mask[:, None], torch.finfo(logits.dtype).min)
        attn = logits.softmax(dim=-1)
        part_tokens = self.part_out(torch.matmul(attn, value))
        return (part_tokens, attn) if return_attention else part_tokens

    def forward(self, part_motion: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor):
        part_tokens, attention = self.part_text(text, text_mask, return_attention=True)
        motion_emb = F.normalize(self.motion_proj(part_motion), dim=-1)
        lang_emb = F.normalize(self.lang_proj(part_tokens), dim=-1)
        # Flatten B x P so both other examples and other anatomical parts are
        # negatives. The positive at flat index b*P+p keeps the fixed mapping.
        contrast = _symmetric_nce(
            motion_emb.reshape(-1, motion_emb.shape[-1]),
            lang_emb.reshape(-1, lang_emb.shape[-1]),
            self.logit_scale,
        )
        normalized_attention = F.normalize(attention, dim=-1)
        similarity = normalized_attention @ normalized_attention.transpose(-1, -2)
        off_diagonal = ~torch.eye(self.num_parts, device=similarity.device, dtype=torch.bool)
        attention_diversity = similarity[:, off_diagonal].mean()
        normalized_queries = F.normalize(self.part_queries, dim=-1)
        query_similarity = normalized_queries @ normalized_queries.T
        query_diversity = query_similarity[off_diagonal].square().mean()
        loss = (
            contrast
            + self.diversity_weight * attention_diversity
            + self.query_diversity_weight * query_diversity
        )
        if self.use_anatomical_anchor:
            q_bar = F.normalize(self.query_proj(self.part_queries), dim=-1)
            a_bar = F.normalize(self.anchor_to_embed(self.anchor_embeddings), dim=-1)
            anchor_loss = (1 - (q_bar * a_bar).sum(-1)).mean()
            loss = loss + self.anchor_weight * anchor_loss
        return loss, part_tokens
