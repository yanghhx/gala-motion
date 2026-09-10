import torch
from torch import nn


class FlowBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.cross_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.cross_modulation[-1].weight)
        nn.init.zeros_(self.cross_modulation[-1].bias)

    def forward(self, x, condition, padding_mask=None, text_tokens=None, text_pad=None):
        s1, b1, g1, s2, b2, g2 = self.modulation(condition).chunk(6, -1)
        h = self.norm1(x) * (1 + s1[:, None]) + b1[:, None]
        h = self.attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + g1[:, None] * h
        if text_tokens is not None:
            cs, cb, cg = self.cross_modulation(condition).chunk(3, -1)
            query = self.cross_norm(x) * (1 + cs[:, None]) + cb[:, None]
            cross = self.cross_attn(
                query, text_tokens, text_tokens,
                key_padding_mask=text_pad, need_weights=False,
            )[0]
            x = x + cg[:, None] * cross
        h = self.norm2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.ffn(h)


class RectifiedFlowDiT(nn.Module):
    def __init__(self, latent_dim, model_dim, text_dim, heads, layers, max_tokens=128):
        super().__init__()
        self.input = nn.Linear(latent_dim, model_dim)
        self.output = nn.Linear(model_dim, latent_dim)
        self.text = nn.Linear(text_dim, model_dim)
        self.text_tokens = nn.Linear(text_dim, model_dim)
        self.time = nn.Sequential(nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.position = nn.Parameter(torch.randn(1, max_tokens, model_dim) * 0.01)
        self.blocks = nn.ModuleList([FlowBlock(model_dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(model_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z_t, time, text_condition, padding_mask=None, text_tokens=None, text_mask=None):
        x = self.input(z_t) + self.position[:, : z_t.shape[1]]
        condition = self.text(text_condition) + self.time(time[:, None])
        tokens = self.text_tokens(text_tokens) if text_tokens is not None else None
        text_pad = ~text_mask if text_mask is not None else None
        for block in self.blocks:
            x = block(x, condition, padding_mask, tokens, text_pad)
        return self.output(self.norm(x))

    def _drop_condition(self, text_condition, text_tokens, text_mask, cond_drop_prob):
        if cond_drop_prob <= 0:
            return text_condition, text_tokens, text_mask
        drop = torch.rand(text_condition.shape[0], device=text_condition.device) < cond_drop_prob
        text_condition = text_condition.clone()
        text_condition[drop] = 0
        if text_tokens is not None:
            text_tokens = text_tokens.clone()
            text_tokens[drop] = 0
        return text_condition, text_tokens, text_mask

    def loss(self, clean, text_condition, padding_mask=None, text_tokens=None, text_mask=None, cond_drop_prob=0.0):
        b = clean.shape[0]
        time = torch.rand(b, device=clean.device, dtype=clean.dtype)
        noise = torch.randn_like(clean)
        z_t = (1 - time[:, None, None]) * noise + time[:, None, None] * clean
        target = clean - noise
        text_condition, text_tokens, text_mask = self._drop_condition(
            text_condition, text_tokens, text_mask, cond_drop_prob
        )
        prediction = self(z_t, time, text_condition, padding_mask, text_tokens, text_mask)
        valid = (~padding_mask).unsqueeze(-1) if padding_mask is not None else torch.ones_like(prediction, dtype=torch.bool)
        return ((prediction - target).square() * valid).sum() / valid.sum().clamp_min(1)

    def sample(self, shape, text_condition, padding_mask, steps, guidance_scale=1.0, text_tokens=None, text_mask=None):
        z = torch.randn(shape, device=text_condition.device, dtype=text_condition.dtype)
        dt = 1.0 / steps
        null = torch.zeros_like(text_condition)
        null_tokens = torch.zeros_like(text_tokens) if text_tokens is not None else None
        for index in range(steps):
            time = torch.full((shape[0],), index / steps, device=z.device, dtype=z.dtype)
            conditional = self(z, time, text_condition, padding_mask, text_tokens, text_mask)
            if guidance_scale != 1:
                unconditional = self(z, time, null, padding_mask, null_tokens, text_mask)
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            else:
                velocity = conditional
            z = z + dt * velocity
            if padding_mask is not None:
                z = z.masked_fill(padding_mask.unsqueeze(-1), 0)
        return z
