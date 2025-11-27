# train/model_codesearch.py
# -*- coding: utf-8 -*-
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ProjectionHead(nn.Module):
    """Kicsi projekciós fej a CLS embeddingre (stabilabb tanulás)."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.ln(x)
        return x

class CodeSearchEncoder(nn.Module):
    """
    Közös encoder NL-hez és Code-hoz. A CLS ([:,0,:]) embeddinget vesszük,
    opcionális projection head-del és L2 normalizálással (cosine = dot).
    """
    def __init__(self, base_checkpoint: str, use_projection: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_checkpoint)
        self.encoder = AutoModel.from_pretrained(base_checkpoint)
        self.hidden = self.config.hidden_size
        self.proj = ProjectionHead(self.hidden) if use_projection else nn.Identity()

    @staticmethod
    def _cls(hiddens: torch.Tensor) -> torch.Tensor:
        # (B, L, H) -> (B, H)
        return hiddens[:, 0, :]

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def encode_nl(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self._cls(out.last_hidden_state)
        return self._l2norm(self.proj(cls))

    def encode_code(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self._cls(out.last_hidden_state)
        return self._l2norm(self.proj(cls))
