#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_rank_functions.py

Feladatok:
- JSONL --> egységesített kimenet (kulcssorrend: nl, code, repo, path, func_name, lang, idx, split)
- Duplikátum szűrés: CSAK akkor dobjuk el, ha nl ÉS code is egyezik (AND logika)
- Részletes audit kiírás:
    * Összes sor
    * JSON hibás sorok
    * NL kulcs szerinti duplikátum rekordok száma és a külön NL értékek száma
    * CODE kulcs szerinti duplikátum rekordok száma és a külön CODE értékek száma
    * Ténylegesen eldobott sorok (duplikátum NL+CODE alapján)

Opcionális kimenetek:
  --rejects <path>   : ide menti a hibás JSON sorokat
  --dup-nl <path>    : ide menti az NL szerinti duplikátum rekordokat
  --dup-code <path>  : ide menti a CODE szerinti duplikátum rekordokat

Használat:
  python scripts/convert_rank_functions.py input.jsonl output.jsonl
  python scripts/convert_rank_functions.py input.jsonl output.jsonl --limit 50 --rejects output/rejects.jsonl --dup-nl output/dups_nl.jsonl --dup-code output/dups_code.jsonl
"""

import argparse
import json
import os
from collections import OrderedDict, defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Rank functions JSONL konvertálása és auditálása (duplikátum AND-logikával).")
    p.add_argument("in_path", help="Bemeneti JSONL fájl")
    p.add_argument("out_path", help="Kimeneti JSONL fájl")
    p.add_argument("--limit", type=int, default=None, help="Legfeljebb ennyi rekordot írjon ki (opcionális)")
    p.add_argument("--split", default="train", help="split mező értéke (alapértelmezés: train)")
    p.add_argument("--lang", default="erlang", help="lang mező értéke (alapértelmezés: erlang)")
    # Opcionális audit kimenetek:
    p.add_argument("--rejects", default=None, help="JSON hibás sorok mentése ide (JSONL)")
    p.add_argument("--dup-nl", dest="dup_nl", default=None, help="NL duplikátum rekordok mentése ide (JSONL)")
    p.add_argument("--dup-code", dest="dup_code", default=None, help="CODE duplikátum rekordok mentése ide (JSONL)")
    return p.parse_args()

def load_jsonl(path):
    total = 0
    valid = []
    rejects = []
    if not os.path.exists(path):
        print(f"Hiba: nem találom az input fájlt: {path}")
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                valid.append(obj)
            except Exception as e:
                rejects.append({"error": str(e), "raw": s})
    return total, valid, rejects

def to_key(text):
    """Normalizált kulcs a duplikáció-ellenőrzéshez."""
    if text is None:
        return ""
    return str(text).strip()

def build_idx(repo, func_name, arity):
    repo = repo or "unknown-repo"
    func_name = func_name or "unknown"
    if arity is None or arity == "":
        return f"{repo}::{func_name}::0"
    return f"{repo}::{func_name}/{arity}::0"

def standardize_record(obj, lang_default="erlang", split_default="train"):
    """
    Bemeneti rekordból egységes kimeneti rekord.
    Kulcssorrend: nl, code, repo, path, func_name, lang, idx, split
    """
    nl = obj.get("nl")
    code = obj.get("code")
    repo = obj.get("repo") or obj.get("repository") or ""
    path = obj.get("path") or ""
    func_name = obj.get("func_name") or obj.get("function") or ""
    arity = obj.get("arity")
    lang = obj.get("lang") or lang_default
    split = obj.get("split") or split_default

    # idx generálás
    idx = obj.get("idx")
    if not idx:
        idx = build_idx(repo, func_name, arity)

    # OrderedDict a kívánt kulcssorrendhez
    return OrderedDict([
        ("nl", nl),
        ("code", code),
        ("repo", repo),
        ("path", path),
        ("func_name", func_name),
        ("lang", lang),
        ("idx", idx),
        ("split", split),
    ])

def compute_group_dups(records, key_name):
    """
    Csoportos duplikátum statisztika egy kulcsra (nl vagy code).
    Visszaad:
      total_dup_recs: hány rekord esik olyan csoportba, ahol legalább 2 elem van
      distinct_dup_keys: hány külön kulcs okozott duplikációt
      dup_records_flat: a duplikációs csoportok összes rekordja (listában), standardizált formában
    """
    groups = defaultdict(list)
    for rec in records:
        key = to_key(rec.get(key_name))
        if key:
            groups[key].append(rec)

    dup_records_flat = []
    for key, lst in groups.items():
        if len(lst) > 1:
            dup_records_flat.extend(lst)

    total_dup_recs = len(dup_records_flat)
    distinct_dup_keys = sum(1 for _k, lst in groups.items() if len(lst) > 1)
    return total_dup_recs, distinct_dup_keys, dup_records_flat

def write_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    args = parse_args()

    # 1) Beolvasás
    total_lines, valid_raw, rejects = load_jsonl(args.in_path)

    # 2) JSON hibák (opcionális mentés)
    if args.rejects:
        write_jsonl(args.rejects, rejects)

    # 3) Standardizálás (csak a JSON szempontból érvényes rekordokra)
    valid_std = [standardize_record(v, lang_default=args.lang, split_default=args.split) for v in valid_raw]

    # 4) AUDIT: NL/CODE duplikátum statisztika (az ÖSSZES érvényes rekordon)
    nl_dup_recs, nl_dup_keys, nl_dup_list = compute_group_dups(valid_std, "nl")
    code_dup_recs, code_dup_keys, code_dup_list = compute_group_dups(valid_std, "code")

    # Opcionális mentések
    if args.dup_nl:
        write_jsonl(args.dup_nl, nl_dup_list)
    if args.dup_code:
        write_jsonl(args.dup_code, code_dup_list)

    # 5) Kimenet építése: duplikátumszűrés AND-logikával (nl+code pár alapján)
    seen_pairs = set()
    kept = []
    dup_and_count = 0

    for rec in valid_std:
        nl_key = to_key(rec.get("nl"))
        code_key = to_key(rec.get("code"))
        pair_key = (nl_key, code_key) if nl_key and code_key else None

        is_dup_and = pair_key is not None and pair_key in seen_pairs
        if is_dup_and:
            dup_and_count += 1
            continue

        if pair_key is not None:
            seen_pairs.add(pair_key)

        kept.append(rec)
        if args.limit is not None and len(kept) >= args.limit:
            break

    # 6) Kiírás
    out_abs = os.path.abspath(args.out_path)
    in_abs = os.path.abspath(args.in_path)
    if kept:
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w", encoding="utf-8") as fout:
            for rec in kept:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 7) Összegzés
    print(f"[OK] Bemeneti sorok: {total_lines}")
    print(f"[OK] Kimenetre írt sorok: {len(kept)}")
    print(f"[OK] Kihagyva (JSON hiba): {len(rejects)}")
    print(f"[OK] Kihagyva (duplikátum NL+CODE alapján): {dup_and_count}")
    print(f"[AUDIT] Duplikátum NL kulcs alapján: {nl_dup_recs} rekord, {nl_dup_keys} külön NL érték")
    print(f"[AUDIT] Duplikátum CODE kulcs alapján: {code_dup_recs} rekord, {code_dup_keys} külön CODE érték")

    if args.rejects:
        print(f"[AUDIT] Rejects mentve: {args.rejects}")
    if args.dup_nl:
        print(f"[AUDIT] NL duplikátumok mentve: {args.dup_nl}")
    if args.dup_code:
        print(f"[AUDIT] CODE duplikátumok mentve: {args.dup_code}")

    print(f"[OK] Input:  {os.path.relpath(in_abs)}")
    print(f"[OK] Output: {out_abs}")

if __name__ == "__main__":
    main()
