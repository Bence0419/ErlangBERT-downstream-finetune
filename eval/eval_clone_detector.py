# eval/eval_clone_detector.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, json
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# path
FILE_DIR = Path(__file__).resolve()
PROJECT_ROOT = FILE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.clone_pairs_dataset import ClonePairsDataset, ClonePairsCollator
from train.model_clone import CloneClassifier

def load_model(ckpt_dir: Path, device: torch.device) -> tuple[CloneClassifier, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    model = CloneClassifier(base_checkpoint=str(ckpt_dir)).to(device)
    # osztályozó fej visszatöltése
    head = ckpt_dir / "classifier.pt"
    if head.exists():
        state = torch.load(head, map_location="cpu")
        model.classifier.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-checkpoint", required=True, help="models\\erlang_clone_detector\\best")
    ap.add_argument("--data-file", required=True, help="output\\clone_pairs\\test_pairs.jsonl")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-code-len", type=int, default=256)
    ap.add_argument("--save-report", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.model_checkpoint)

    model, tokenizer = load_model(ckpt_dir, device)
    collator = ClonePairsCollator(tokenizer, max_code_len=args.max_code_len)

    ds = ClonePairsDataset(args.data_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    # evaluate
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)[:, 1]
            pred = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    import numpy as np
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    print("=== Clone Detection (test) ===")
    print(f"acc={acc:.4f}  f1={f1:.4f}  prec={prec:.4f}  rec={rec:.4f}  auc={auc:.4f}")
    print(f"confmat: TP={tp} FP={fp} FN={fn} TN={tn}")

    if args.save_report:
        report = {
            "model": str(ckpt_dir),
            "data_file": args.data_file,
            "metrics": {"acc": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc, "TP": tp, "FP": fp, "FN": fn, "TN": tn},
        }
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] report mentve: {args.save_report}")

if __name__ == "__main__":
    main()
