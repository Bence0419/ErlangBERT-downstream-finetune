# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

class CodeSearchDataset(Dataset):
    """
    Betölti a CodeSearchNet-szerű jsonl fájlokat:
      {"nl": "...", "code": "...", "idx": "...", ...}
    és csak a nyers szövegeket adja vissza; a tokenizálás a collatorban történik.
    """
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        assert self.path.exists(), f"Missing dataset file: {self.path}"
        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # minimal check
                if "nl" in obj and "code" in obj:
                    self.rows.append(obj)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        return {
            "nl": r["nl"],
            "code": r["code"],
            "idx": r.get("idx", str(i))
        }

class CodeSearchCollator:
    """
    Tokenizál NL-t és Code-ot külön, padeli a két csatornát.
    """
    def __init__(self, tokenizer, max_nl_len: int = 128, max_code_len: int = 256):
        self.tok = tokenizer
        self.max_nl_len = max_nl_len
        self.max_code_len = max_code_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        nl_texts = [b["nl"] for b in batch]
        code_texts = [b["code"] for b in batch]

        nl = self.tok(
            nl_texts,
            padding=True,
            truncation=True,
            max_length=self.max_nl_len,
            return_tensors="pt"
        )
        code = self.tok(
            code_texts,
            padding=True,
            truncation=True,
            max_length=self.max_code_len,
            return_tensors="pt"
        )

        return {
            "nl_input_ids": nl["input_ids"],
            "nl_attention_mask": nl["attention_mask"],
            "code_input_ids": code["input_ids"],
            "code_attention_mask": code["attention_mask"],
        }
