# train/train_clone_detector.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, json, math, time, random
from pathlib import Path
from typing import Dict, Any, Tuple
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# path hack
FILE_DIR = Path(__file__).resolve()
PROJECT_ROOT = FILE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.clone_pairs_dataset import ClonePairsDataset, ClonePairsCollator
from train.model_clone import CloneClassifier

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_loop(model, dl, device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
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

    # metrics
    y_true_np = np.array(y_true); y_pred_np = np.array(y_pred); y_prob_np = np.array(y_prob)
    acc = float((y_true_np == y_pred_np).mean())
    # precision/recall/f1 in binary (pos=1)
    tp = int(((y_true_np == 1) & (y_pred_np == 1)).sum())
    fp = int(((y_true_np == 0) & (y_pred_np == 1)).sum())
    fn = int(((y_true_np == 1) & (y_pred_np == 0)).sum())
    tn = int(((y_true_np == 0) & (y_pred_np == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true_np, y_prob_np))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def train(
    base_checkpoint: str,
    train_file: str,
    valid_file: str,
    out_dir: str,
    batch_size: int = 16,
    max_code_len: int = 256,
    lr: float = 2e-5,
    epochs: int = 3,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    seed: int = 42,
    grad_clip: float = 1.0
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, use_fast=True)
    collator = ClonePairsCollator(tokenizer, max_code_len=max_code_len)

    ds_train = ClonePairsDataset(train_file)
    ds_valid = ClonePairsDataset(valid_file)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    model = CloneClassifier(base_checkpoint=base_checkpoint).to(device)

    # optimizer & scheduler
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(dl_train) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1 = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(dl_train, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            out = model(**batch, labels=labels)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()

            running += float(loss.item())
            if step % 100 == 0:
                print(f"[epoch {ep} step {step}] loss={loss.item():.4f}")

        avg_loss = running / max(1, len(dl_train))
        print(f"[epoch {ep}] avg_train_loss={avg_loss:.4f}")

        # valid
        metrics = eval_loop(model, dl_valid, device)
        print(f"[epoch {ep}] valid: acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} auc={metrics['auc']:.4f}")
        print(f"          confmat: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")

        # save best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_dir = out_path / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"  new best F1={best_f1:.4f} → saving to {save_dir}")
            # HF-mentés
            model.encoder.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            torch.save(model.classifier.state_dict(), save_dir / "classifier.pt")
            with (save_dir / "clone_config.json").open("w", encoding="utf-8") as f:
                json.dump({"max_code_len": max_code_len}, f)

    print(f"[done] best valid F1 = {best_f1:.4f}  (saved in {out_path/'best'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True, help="Erlang-specializált (GraphCodeBERT) checkpoint könyvtár")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--valid-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-code-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(
        base_checkpoint=args.base_checkpoint,
        train_file=args.train_file,
        valid_file=args.valid_file,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_code_len=args.max_code_len,
        lr=args.lr,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
