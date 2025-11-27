#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_rank_functions.py

Cél:
- Egy JSONL korpuszt ipari gyakorlat szerint train/valid/test szeletekre bont.
- Garantálja, hogy ugyanaz a funkció/kód NE kerüljön több split-be (grouping).
- Minden kimeneti sorban szerepel a "split": "train|valid|test" kulcs.

Fő opciók:
  --group-by {idx,function,code_hash,nl_code}
      idx       : (ALAPÉRT.) ha van 'idx', az alapján csoportosít, különben visszaesik code_hash-re
      function  : repo|path|func_name hármas alapján
      code_hash : normalizált kód SHA1 hash-e alapján
      nl_code   : (nl, code) pár alapján

  --splits "80,10,10"         : arányok train/valid/test (összesen 100)
  --seed 1337                  : reprodukálható keverés
  --out-dir output/splits      : ide írja a train/valid/test .jsonl fájlokat
  --combined output/all.jsonl  : egyben is kiírja az egészet split mezővel
  --require nl,code            : eldobja, amelyiknél hiányzik bármely megadott mező

Használat példák:
  python scripts/split_rank_functions.py output/rank_functions_fixed.jsonl --out-dir output/splits
  python scripts/split_rank_functions.py output/rank_functions_fixed.jsonl --out-dir output/splits --splits "90,5,5" --seed 42
  python scripts/split_rank_functions.py output/rank_functions_fixed.jsonl --out-dir output/splits --group-by function --combined output/all_with_split.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Train/valid/test szétvágó script szivárgásvédelemmel.")
    p.add_argument("in_path", help="Bemeneti JSONL")
    p.add_argument("--out-dir", required=True, help="Kimeneti mappa (train/valid/test.jsonl ide kerül)")
    p.add_argument("--combined", default=None, help="Opcionális: egyben is kiírja ide az összes rekordot split mezővel (JSONL)")
    p.add_argument("--splits", default="80,10,10", help="Train,valid,test arányok százalékban. Pl.: '80,10,10'")
    p.add_argument("--seed", type=int, default=1337, help="Véletlen sorrend seed (reprodukálható felosztás)")
    p.add_argument("--group-by", choices=["idx", "function", "code_hash", "nl_code"], default="idx",
                   help="Csoportosítás a szivárgás elkerüléséhez (alapértelmezés: idx)")
    p.add_argument("--require", default="", help="Vesszővel elválasztott kötelező mezők (pl.: 'nl,code')")
    return p.parse_args()

def load_jsonl(path):
    total = 0
    data = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
                data.append(obj)
            except Exception:
                # ipari gyakorlatban itt logolnánk/eldobnánk:
                continue
    return total, data

_ws_re = re.compile(r"\s+", re.MULTILINE)

def norm_code(code):
    if code is None:
        return ""
    return _ws_re.sub("", str(code))

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def group_key(rec, mode):
    """
    Csoportkulcs, ami meghatározza, hogy mi NEM keveredhet több split-be.
    """
    if mode == "idx":
        idx = str(rec.get("idx") or "").strip()
        if idx:
            return ("idx", idx)
        # ha nincs idx, essünk vissza code_hash-re
        code = rec.get("code")
        return ("code_hash", sha1_str(norm_code(code)))

    if mode == "function":
        repo = str(rec.get("repo") or "").strip()
        path = str(rec.get("path") or "").strip()
        func = str(rec.get("func_name") or "").strip()
        base = f"{repo}|{path}|{func}"
        if base != "||":
            return ("function", base)
        # fallback
        code = rec.get("code")
        return ("code_hash", sha1_str(norm_code(code)))

    if mode == "code_hash":
        code = rec.get("code")
        return ("code_hash", sha1_str(norm_code(code)))

    if mode == "nl_code":
        nl = str(rec.get("nl") or "").strip()
        code = str(rec.get("code") or "")
        return ("nl_code", nl + "<<<NL>>>"+ code)

    # biztos fallback
    code = rec.get("code")
    return ("code_hash", sha1_str(norm_code(code)))

def parse_splits(spec: str):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("A --splits értéke 'a,b,c' formátumú legyen, pl. '80,10,10'.")
    vals = [float(x) for x in parts]
    s = sum(vals)
    if s <= 0:
        raise ValueError("A split arányok összege > 0 legyen.")
    # normalizáljuk 1.0-re
    return [v / s for v in vals]  # train, valid, test arány

def compute_targets(n_items, ratios):
    base = [int(n_items * r) for r in ratios]
    diff = n_items - sum(base)
    # a maradékot a legnagyobb arányú (train->valid->test) irányban töltsük fel
    order = sorted(range(3), key=lambda i: ratios[i], reverse=True)
    i = 0
    while diff > 0:
        base[order[i % 3]] += 1
        diff -= 1
        i += 1
    return base  # [train_target, valid_target, test_target]

def greedy_assign(groups, sizes, targets):
    """
    Greedy kiosztás: mindig abba a splitbe rakjuk a következő csoportot,
    ahol a legtöbb szabad kapacitás van (target - current).
    Ha átlóg, nem gond – a csoport integritása a fontos.
    """
    assigned = [[], [], []]  # list of group keys
    used = [0, 0, 0]         # elemek darabszáma per split

    for g in groups:
        # szabad kapacitások:
        caps = [targets[i] - used[i] for i in range(3)]
        # melyikbe menjen? ahol a legnagyobb a 'cap'
        dest = max(range(3), key=lambda i: caps[i])
        assigned[dest].append(g)
        used[dest] += sizes[g]

    return assigned, used

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    random.seed(args.seed)

    # betöltés
    total_lines, records = load_jsonl(args.in_path)

    # kötelező mezők szűrése (ha kérték)
    required_fields = [x.strip() for x in args.require.split(",") if x.strip()]
    if required_fields:
        filtered = []
        dropped_req = 0
        for r in records:
            if all(r.get(k) not in (None, "") for k in required_fields):
                filtered.append(r)
            else:
                dropped_req += 1
        records = filtered
    else:
        dropped_req = 0

    if not records:
        print("[HIBA] Nincsenek feldolgozható rekordok.")
        return

    # csoportok felépítése
    groups = defaultdict(list)
    for r in records:
        k = group_key(r, args.group_by)
        groups[k].append(r)

    # csoportok listája és mérete
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    group_sizes = {g: len(groups[g]) for g in group_keys}

    # arányok és célok
    ratios = parse_splits(args.splits)
    n_total = sum(group_sizes.values())
    targets = compute_targets(n_total, ratios)

    # kiosztás
    assigned, used = greedy_assign(group_keys, group_sizes, targets)
    # 0=train, 1=valid, 2=test
    split_names = ["train", "valid", "test"]
    split_bins = {name: [] for name in split_names}

    # rekordok beírása a split-ekbe, split mező felülírása/beillesztése
    for i, name in enumerate(split_names):
        for g in assigned[i]:
            for rec in groups[g]:
                rec_out = dict(rec)
                rec_out["split"] = name
                split_bins[name].append(rec_out)

    # opcionális combined
    if args.combined:
        combined = split_bins["train"] + split_bins["valid"] + split_bins["test"]
        write_jsonl(args.combined, combined)

    # 3 fájl
    out_dir = args.out_dir
    write_jsonl(os.path.join(out_dir, "train.jsonl"), split_bins["train"])
    write_jsonl(os.path.join(out_dir, "valid.jsonl"), split_bins["valid"])
    write_jsonl(os.path.join(out_dir, "test.jsonl"),  split_bins["test"])

    # szivárgás ellenőrzés (diagnosztika)
    def code_hash_set(items):
        return {sha1_str(norm_code(x.get("code"))) for x in items if x.get("code")}

    def nl_set(items):
        return {str(x.get("nl") or "").strip() for x in items if x.get("nl")}

    def pair_set(items):
        return {(str(x.get("nl") or "").strip(), str(x.get("code") or "")) for x in items}

    tr, va, te = split_bins["train"], split_bins["valid"], split_bins["test"]
    tr_h, va_h, te_h = code_hash_set(tr), code_hash_set(va), code_hash_set(te)
    tr_nl, va_nl, te_nl = nl_set(tr), nl_set(va), nl_set(te)
    tr_p, va_p, te_p = pair_set(tr), pair_set(va), pair_set(te)

    # összegzés
    print("=== Összegzés ===")
    print(f"[OK] Bemeneti sorok (raw): {total_lines}")
    print(f"[OK] Feldolgozott rekordok: {len(records)}  (eldobva kötelező mező miatt: {dropped_req})")
    print(f"[OK] Csoportok száma: {len(group_keys)} (mód: {args.group_by})")
    print(f"[OK] Célok (train,valid,test): {targets} (összes: {sum(targets)})")
    print(f"[OK] Eredmény elemszám (train,valid,test): {len(tr)}, {len(va)}, {len(te)}  (összes: {len(tr)+len(va)+len(te)})")
    print("=== Szivárgás-ellenőrzés (metszetek) ===")
    print(f"code_hash   train∩valid: {len(tr_h & va_h)}, train∩test: {len(tr_h & te_h)}, valid∩test: {len(va_h & te_h)}")
    print(f"nl          train∩valid: {len(tr_nl & va_nl)}, train∩test: {len(tr_nl & te_nl)}, valid∩test: {len(va_nl & te_nl)}")
    print(f"(nl,code)   train∩valid: {len(tr_p & va_p)}, train∩test: {len(tr_p & te_p)}, valid∩test: {len(va_p & te_p)}")
    print(f"[OK] Kimenetek: {os.path.abspath(out_dir)}")
    if args.combined:
        print(f"[OK] Combined: {os.path.abspath(args.combined)}")

if __name__ == "__main__":
    main()
