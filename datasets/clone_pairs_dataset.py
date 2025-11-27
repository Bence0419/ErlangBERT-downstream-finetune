# datasets/clone_pairs_dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class ClonePairsDataset:
    """
    JSONL formátum: soronként
      {
        "code_a": "...",
        "code_b": "...",
        "label": 0|1,
        "idx_a": "...", "idx_b": "...",
        ...
      }
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if "code_a" in obj and "code_b" in obj and "label" in obj:
                    self.rows.append(obj)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.rows[i]


class ClonePairsCollator:
    """
    Tokenizál két kódrészletet 'text_pair' (pair) módban a HF tokenizerrel.
    """
    def __init__(self, tokenizer, max_code_len: int = 256):
        self.tok = tokenizer
        self.max_code_len = max_code_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_a = [b["code_a"] for b in batch]
        texts_b = [b["code_b"] for b in batch]
        labels = [int(b["label"]) for b in batch]
        toks = self.tok(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=self.max_code_len,
            return_tensors="pt",
        )
        toks["labels"] = __import__("torch").tensor(labels, dtype=__import__("torch").long)
        return toks
