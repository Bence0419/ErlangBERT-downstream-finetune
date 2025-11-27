# -*- coding: utf-8 -*-
"""
Full-corpus code search evaluation for ErlangBERT.

- Loads the (shared) encoder from a checkpoint dir.
- Loads optional projection head weights (head.pt) if present.
- Builds code embeddings for the whole dataset (e.g., test.jsonl).
- For each NL query, retrieves over ALL code embeddings.
- Computes MRR and Recall@K metrics.
- Can cache embeddings to speed up repeated runs.

Usage (PowerShell):
  python eval\eval_code_search.py `
    --model-checkpoint models\erlang_graphcodebert_codesearch\best `
    --data-file output\code_search_data\test.jsonl `
    --batch-size 64 `
    --k 1 3 5 10 `
    --save-report eval\codesearch_report.json `
    --save-embeddings output\code_search_index\test_code_embeds.npz

Author: you :)
"""
import sys, json, math, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# --- make sure local packages resolve ---
FILE_DIR = Path(__file__).resolve()
PROJECT_ROOT = FILE_DIR.parents[1]  # ...\erlang_corpus_scraper
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.code_search_dataset import CodeSearchDataset, CodeSearchCollator
from train.model_codesearch import CodeSearchEncoder

def load_codesearch_encoder(
    ckpt_dir: Path,
    device: torch.device
) -> Tuple[CodeSearchEncoder, AutoTokenizer, Dict]:
    """
    Load encoder (HF folder) and optional projection head.
    Returns (model, tokenizer, meta_config).
    """
    ckpt_dir = ckpt_dir.resolve()
    # tokenizer + base encoder weights are saved here during training
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    # default: assume projection is used if head.pt exists
    head_path = ckpt_dir / "head.pt"
    cfg_path  = ckpt_dir / "codesearch_config.json"
    use_projection = head_path.exists()
    meta = {"use_projection": use_projection, "temperature": 0.07}

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            meta.update(json.load(f))

    model = CodeSearchEncoder(base_checkpoint=str(ckpt_dir), use_projection=use_projection).to(device)
    if head_path.exists():
        state = torch.load(head_path, map_location="cpu")
        model.proj.load_state_dict(state, strict=True)

    model.eval()
    return model, tokenizer, meta

def embed_codes(
    model: CodeSearchEncoder,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_len: int,
    device: torch.device
) -> np.ndarray:
    """
    Compute embeddings for the 'code' field of rows.
    Returns numpy array of shape (N, H) with L2-normalized vectors.
    """
    vecs: List[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        code_texts = [r["code"] for r in batch]
        toks = tokenizer(code_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            v = model.encode_code(toks["input_ids"].to(device), toks["attention_mask"].to(device))
        vecs.append(v.cpu().numpy())
    return np.vstack(vecs)

def embed_queries(
    model: CodeSearchEncoder,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_len: int,
    device: torch.device
) -> np.ndarray:
    """
    Compute embeddings for the 'nl' field of rows.
    """
    vecs: List[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        nl_texts = [r["nl"] for r in batch]
        toks = tokenizer(nl_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            v = model.encode_nl(toks["input_ids"].to(device), toks["attention_mask"].to(device))
        vecs.append(v.cpu().numpy())
    return np.vstack(vecs)

def cosine_topk(query_vecs: np.ndarray, code_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarity (dot product on L2-normalized vecs) and return
    top-k indices and scores for each query.
    Returns (indices, scores) with shapes (Q, k).
    """
    # (Q,H) @ (H,N) -> (Q,N)
    sims = query_vecs @ code_vecs.T
    # argsort descending, take top-k
    topk_idx = np.argpartition(-sims, kth=range(k), axis=1)[:, :k]
    # but argpartition doesn't sort those top-k; sort within k
    row_idx = np.arange(sims.shape[0])[:, None]
    topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
    order_in_k = np.argsort(-topk_scores, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order_in_k, axis=1)
    topk_scores = np.take_along_axis(topk_scores, order_in_k, axis=1)
    return topk_idx, topk_scores

def compute_metrics(
    gold_indices: List[int],
    ranked_indices: np.ndarray,
    ks: List[int]
) -> Dict[str, float]:
    """
    gold_indices: length Q, gold code index (position in code list)
    ranked_indices: (Q, Kmax) top-ranked code indices for each query
    """
    Q, Kmax = ranked_indices.shape
    metrics: Dict[str, float] = {}
    # MRR
    rr_sum = 0.0
    for i in range(Q):
        gold = gold_indices[i]
        row = ranked_indices[i]
        # find position if present
        pos = np.where(row == gold)[0]
        if len(pos) > 0:
            rr_sum += 1.0 / (pos[0] + 1)  # 1-indexed
    metrics["MRR"] = rr_sum / Q

    # Recall@K
    for k in ks:
        hit = 0
        for i in range(Q):
            if gold_indices[i] in ranked_indices[i, :k]:
                hit += 1
        metrics[f"Recall@{k}"] = hit / Q
    return metrics

def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "nl" in obj and "code" in obj:
                rows.append(obj)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-checkpoint", required=True, help="Path to the trained code-search model folder (the 'best' dir).")
    ap.add_argument("--data-file", required=True, help="JSONL with fields: nl, code, idx (test split recommended).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-nl-len", type=int, default=128)
    ap.add_argument("--max-code-len", type=int, default=256)
    ap.add_argument("--k", type=int, nargs="+", default=[1,3,5,10], help="K values for Recall@K.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only on the first N items (debug).")
    ap.add_argument("--save-report", type=str, default="", help="Where to write a JSON report with metrics + examples.")
    ap.add_argument("--save-embeddings", type=str, default="", help="Cache code embeddings to an .npz file for reuse.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.model_checkpoint)
    data_path = Path(args.data_file)

    print(f"[eval] loading model from: {ckpt_dir}")
    model, tokenizer, meta = load_codesearch_encoder(ckpt_dir, device)
    print(f"[eval] model loaded. use_projection={meta.get('use_projection')} temperature={meta.get('temperature')}")

    print(f"[eval] loading dataset: {data_path}")
    rows = load_jsonl(data_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
        print(f"[eval] using only first {len(rows)} rows (debug mode)")

    # we will use the same list for codes and queries and use 'idx' equality as the gold mapping
    # Build a map from idx -> index in code list
    print("[eval] building code list")
    codes = rows
    idx2pos: Dict[str, int] = {r.get("idx", str(i)): i for i, r in enumerate(codes)}
    gold_indices: List[int] = []
    for i, r in enumerate(rows):
        idx = r.get("idx", str(i))
        gold_indices.append(idx2pos[idx])

    # ---- code embeddings (cache-aware) ----
    if args.save_embeddings:
        emb_path = Path(args.save_embeddings)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        emb_path = None

    code_vecs: Optional[np.ndarray] = None
    if emb_path and emb_path.exists():
        print(f"[eval] loading cached code embeddings from {emb_path}")
        cached = np.load(str(emb_path))
        code_vecs = cached["vecs"]
    else:
        print("[eval] computing code embeddings ...")
        t0 = time.time()
        code_vecs = embed_codes(model, tokenizer, codes, batch_size=args.batch_size, max_len=args.max_code_len, device=device)
        print(f"[eval] code embeddings shape: {code_vecs.shape}  (took {time.time()-t0:.1f}s)")
        if emb_path:
            np.savez_compressed(str(emb_path), vecs=code_vecs)

    # ---- query embeddings ----
    print("[eval] computing query embeddings ...")
    t0 = time.time()
    nl_vecs = embed_queries(model, tokenizer, rows, batch_size=args.batch_size, max_len=args.max_nl_len, device=device)
    print(f"[eval] query embeddings shape: {nl_vecs.shape}  (took {time.time()-t0:.1f}s)")

    # ---- retrieval ----
    Kmax = max(args.k)
    print(f"[eval] retrieving top-{Kmax} for each query ...")
    topk_idx, topk_scores = cosine_topk(nl_vecs, code_vecs, k=Kmax)

    # ---- metrics ----
    print("[eval] computing metrics ...")
    metrics = compute_metrics(gold_indices, topk_idx, ks=args.k)

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ---- examples ----
    examples = []
    for i in range(min(5, len(rows))):
        ex = {
            "query_nl": rows[i]["nl"],
            "gold_idx": gold_indices[i],
            "top1_idx": int(topk_idx[i,0]),
            "hit@1": bool(gold_indices[i] == topk_idx[i,0]),
            "top1_score": float(topk_scores[i,0]),
            "gold_code_snippet": rows[gold_indices[i]]["code"][:200].replace("\n"," "),
            "top1_code_snippet": rows[topk_idx[i,0]]["code"][:200].replace("\n"," ")
        }
        examples.append(ex)

    if args.save_report:
        out = {
            "model_checkpoint": str(ckpt_dir),
            "data_file": str(data_path),
            "num_items": len(rows),
            "metrics": metrics,
            "examples": examples
        }
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[eval] report written to: {args.save_report}")

if __name__ == "__main__":
    main()
