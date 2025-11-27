import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve()
PROJECT_ROOT = FILE_DIR.parents[1]  # ...\erlang_corpus_scraper
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import math
from pathlib import Path
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from datasets.code_search_dataset import CodeSearchDataset, CodeSearchCollator
from train.model_codesearch import CodeSearchEncoder

def info_nce_symmetric(nl_vecs: torch.Tensor, code_vecs: torch.Tensor, temperature: float = 0.07):
    """
    Szimmetrikus InfoNCE:
      L = (L_nl2code + L_code2nl) / 2
    A bemenetek már L2-normalizált vektorok (cosine = dot).
    """
    logits = nl_vecs @ code_vecs.t()  # (B,B)
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_i + loss_t)
    return loss

def compute_mrr_at_k(nl_vecs: torch.Tensor, code_vecs: torch.Tensor, k: int = 10) -> float:
    """
    Egyszerű MRR@K ugyanazon batch-en (csak validációs mintára, kis batch-eknél hasznos).
    Teljes értékelést külön script számol majd (összes kód ellen).
    """
    sim = nl_vecs @ code_vecs.t()  # (B,B) cosine
    ranks = []
    for i in range(sim.size(0)):
        # nagyobb jobb; argsort desc
        scores = sim[i]
        topk = torch.topk(scores, k=k, largest=True).indices
        # az igaz pár indexe i
        if i in topk:
            rank = (topk == i).nonzero(as_tuple=False).item() + 1  # 1-indexed
            ranks.append(1.0 / rank)
        else:
            ranks.append(0.0)
    return float(sum(ranks) / len(ranks))

def train(
    base_checkpoint: str,
    train_file: str,
    valid_file: str,
    out_dir: str,
    batch_size: int = 32,
    max_nl_len: int = 128,
    max_code_len: int = 256,
    lr: float = 2e-5,
    epochs: int = 4,
    warmup_ratio: float = 0.05,
    temperature: float = 0.07,
    use_projection: bool = True,
    seed: int = 42,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, use_fast=True)
    collator = CodeSearchCollator(tokenizer, max_nl_len=max_nl_len, max_code_len=max_code_len)

    ds_train = CodeSearchDataset(train_file)
    ds_valid = CodeSearchDataset(valid_file)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    model = CodeSearchEncoder(base_checkpoint=base_checkpoint, use_projection=use_projection).to(device)

    optim = AdamW(model.parameters(), lr=lr)
    total_steps = len(dl_train) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_mrr = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dl_train, 1):
            nl_ids = batch["nl_input_ids"].to(device)
            nl_mask = batch["nl_attention_mask"].to(device)
            code_ids = batch["code_input_ids"].to(device)
            code_mask = batch["code_attention_mask"].to(device)

            nl_vecs = model.encode_nl(nl_ids, nl_mask)
            code_vecs = model.encode_code(code_ids, code_mask)

            loss = info_nce_symmetric(nl_vecs, code_vecs, temperature=temperature)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            total_loss += float(loss.item())

            if step % 100 == 0:
                mrr = compute_mrr_at_k(nl_vecs.detach(), code_vecs.detach(), k=10)
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f} mrr@10(batch)={mrr:.4f}")

        avg_loss = total_loss / max(1, len(dl_train))
        print(f"[epoch {epoch}] avg_train_loss={avg_loss:.4f}")

        # ---- quick validation (batch-local MRR) ----
        model.eval()
        with torch.no_grad():
            val_mrrs = []
            for batch in dl_valid:
                nl_ids = batch["nl_input_ids"].to(device)
                nl_mask = batch["nl_attention_mask"].to(device)
                code_ids = batch["code_input_ids"].to(device)
                code_mask = batch["code_attention_mask"].to(device)

                nl_vecs = model.encode_nl(nl_ids, nl_mask)
                code_vecs = model.encode_code(code_ids, code_mask)
                mrr = compute_mrr_at_k(nl_vecs, code_vecs, k=10)
                val_mrrs.append(mrr)
            mean_mrr = float(sum(val_mrrs) / max(1, len(val_mrrs)))
            print(f"[epoch {epoch}] valid_mrr@10(batch-wise)={mean_mrr:.4f}")

                # ---- save best ----
        if mean_mrr > best_mrr:
            best_mrr = mean_mrr
            save_dir = out_path / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"  new best mrr={best_mrr:.4f} → saving to {save_dir}")
            # 1) mentjük a teljes encodert HF-formátumban
            model.encoder.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            # 2) mentjük a projection head súlyait külön
            torch.save(model.proj.state_dict(), save_dir / "head.pt")
            # 3) kis config, hogy az eval tudja, kell-e head
            import json
            with (save_dir / "codesearch_config.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "use_projection": not isinstance(model.proj, torch.nn.Identity),
                    "temperature": temperature
                }, f)


    print(f"[done] best valid (batch-wise) MRR@10 = {best_mrr:.4f} saved in {out_path/'best'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True, help="Erlang-specializált GraphCodeBERT checkpoint (mappa)")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--valid-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-nl-len", type=int, default=128)
    ap.add_argument("--max-code-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--no-projection", action="store_true")
    args = ap.parse_args()

    train(
        base_checkpoint=args.base_checkpoint,
        train_file=args.train_file,
        valid_file=args.valid_file,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_nl_len=args.max_nl_len,
        max_code_len=args.max_code_len,
        lr=args.lr,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        use_projection=not args.no_projection,
    )
