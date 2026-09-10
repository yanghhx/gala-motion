from __future__ import annotations

import torch
from torch import nn


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 text encoder. Returns token features (B, L, 512)."""

    def __init__(self, model_name: str | None = None, max_length: int = 77):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        local = "/home/qinyang/桌面/project/checkpoints/clip-vit-base-patch32"
        model_name = model_name or (local if os.path.exists(os.path.join(local, "pytorch_model.bin")) else "openai/clip-vit-base-patch32")
        kwargs = {"local_files_only": os.path.isdir(model_name)}
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, **kwargs)
        self.model = CLIPTextModel.from_pretrained(model_name, **kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.max_length = max_length

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, captions: list[str], device: torch.device):
        tokens = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].bool()
        return hidden, mask
