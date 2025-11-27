# train/model_clone.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class CloneClassifier(nn.Module):
    """
    Cross-encoder: [CLS] reprezentáció -> egyretes Linear -> 2 logit.
    Ugyanaz a bázis encoder, mint a code-searchnél (GraphCodeBERT/ErlangBERT).
    """
    def __init__(self, base_checkpoint: str, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_checkpoint)
        self.encoder = AutoModel.from_pretrained(base_checkpoint)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    @staticmethod
    def _cls(last_hidden_state: torch.Tensor) -> torch.Tensor:
        return last_hidden_state[:, 0, :]  # (B,H)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self._cls(out.last_hidden_state)
        x = self.dropout(cls)
        logits = self.classifier(x)  # (B,2)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
