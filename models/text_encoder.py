from torch import nn


class TokenTextEncoder(nn.Module):
    """Contextualize hashed word tokens before they condition the motion DiT."""

    def __init__(self, dim: int, layers: int = 2, heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, heads, dim * 4, dropout=0.0, activation="gelu",
                    batch_first=True, norm_first=True,
                )
                for _ in range(max(layers, 0))
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, mask):
        hidden = tokens
        padding = ~mask
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=padding)
        return self.norm(hidden)
