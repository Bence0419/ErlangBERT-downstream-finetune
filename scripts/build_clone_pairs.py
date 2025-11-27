# scripts/build_clone_pairs.py
# -*- coding: utf-8 -*-
"""
Klón-detektáláshoz tanító párok előállítása.
V2 FRISSÍTÉS: Tartalmazza a Jaccard-alapú hasonlósági szűrést (min-pos-sim).

Bemenet:
  --in-dir:   mappa, benne {train,valid,test}.jsonl
Kimenet:
  --out-dir:  mappa, benne {train,valid,test}_pairs.jsonl

Működés:
1. Pozitív párok: Azonos 'idx' csoporton belüli kódok, DE csak ha
   a szöveges hasonlóságuk eléri a --min-pos-sim küszöböt.
2. Negatív párok: Különböző csoportokból mintázva (opcionálisan hossz-egyeztetéssel).
3. Memória-optimalizálás: Streamelt írás a kimenetre.
"""
from __future__ import annotations
import argparse, json, random, re, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set, Any
from collections import defaultdict

# Erlang kommentek (% ...) és tokenek szűrése
RE_ERLANG_LINE_COMMENT = re.compile(r"(.*?)(%.*)$")
RE_TOKEN = re.compile(r"\w+")

def get_tokens(text: str) -> Set[str]:
    """Szavakra bontás (bag-of-words) a Jaccard számításhoz."""
    return set(RE_TOKEN.findall(text.lower()))

def jaccard_similarity(code_a: str, code_b: str) -> float:
    """
    Kiszámolja a Jaccard indexet: (Metszet) / (Unió).
    0.0 = teljesen más szavak, 1.0 = ugyanaz a szókészlet.
    """
    toks_a = get_tokens(code_a)
    toks_b = get_tokens(code_b)
    if not toks_a or not toks_b:
        return 0.0
    intersection = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return intersection / union

def strip_erlang_comments(src: str) -> str:
    """Erlang sorvégi kommentek eltávolítása."""
    out_lines = []
    for line in src.splitlines():
        m = RE_ERLANG_LINE_COMMENT.match(line)
        if m:
            line = m.group(1)
        out_lines.append(line.rstrip())
    return "\n".join(l for l in out_lines if l.strip() != "")

def normalize_code(src: str, lang: str, strip_comments: bool) -> str:
    if strip_comments and lang.lower() == "erlang":
        src = strip_erlang_comments(src)
    lines = [ln.rstrip() for ln in src.splitlines()]
    # Üres sorok eltávolítása az elejéről/végéről
    while lines and lines[0].strip() == "": lines.pop(0)
    while lines and lines[-1].strip() == "": lines.pop()
    return "\n".join(lines)

def code_hash(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()

def load_jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    """Generátor a JSONL fájl soronkénti olvasásához (memória-kímélő)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                yield json.loads(s)
            except:
                continue

def make_groups(path: Path, strip_comments: bool, dedup_exact: bool) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """
    Betölti a fájlt és 'idx' szerint csoportosítja a kódokat.
    Visszatér: (groups dict, összes sor száma)
    """
    groups = defaultdict(list)
    count = 0
    print(f" -> Csoportosítás és betöltés: {path.name} ...")
    
    for r in load_jsonl_iter(path):
        if "code" not in r or "idx" not in r: continue
        
        lang = str(r.get("lang", "erlang"))
        code = str(r.get("code", ""))
        code_norm = normalize_code(code, lang, strip_comments=strip_comments)
        
        # Metaadatok hozzáadása a memóriában lévő objektumhoz
        r["_code_norm"] = code_norm
        r["_code_hash"] = code_hash(code_norm)
        
        groups[str(r["idx"])].append(r)
        count += 1

    # Csoporton belüli pontos (byte-level) egyezések szűrése
    if dedup_exact:
        dedup_count = 0
        for gid, items in list(groups.items()):
            uniq = {}
            kept = []
            for it in items:
                h = it["_code_hash"]
                if h not in uniq:
                    uniq[h] = it
                    kept.append(it)
                else:
                    dedup_count += 1
            groups[gid] = kept
        if dedup_count > 0:
            print(f"    (Deduplikálva {dedup_count} teljesen azonos kódrészlet.)")
            
    return groups, count

def all_combinations(items: List[Dict[str, Any]], max_pairs: int | None) -> List[Tuple[int, int]]:
    """Visszaadja a lehetséges pár-indexeket (i, j)."""
    n = len(items)
    idx_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if (max_pairs is not None) and (len(idx_pairs) > max_pairs):
        random.shuffle(idx_pairs)
        idx_pairs = idx_pairs[:max_pairs]
    return idx_pairs

def build_for_split(
    in_path: Path,
    out_path: Path,
    neg_per_pos: float,
    max_pos_per_group: Optional[int],
    min_pos_sim: float,
    length_match: Optional[Tuple[float, float]],
    strip_comments: bool,
    dedup_exact: bool,
    seed: int
):
    rng = random.Random(seed)
    groups, total_rows = make_groups(in_path, strip_comments, dedup_exact)
    
    # Kimeneti mappa létrehozása
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    pos_pairs_buffer = []
    pair_keys = set() # Hogy ne írjuk ki ugyanazt a párt kétszer
    
    skipped_low_sim = 0
    
    # --- 1. POZITÍV PÁROK GYŰJTÉSE ---
    for gid, items in groups.items():
        if len(items) < 2: continue
        
        idx_pairs = all_combinations(items, max_pos_per_group)
        
        for i, j in idx_pairs:
            a, b = items[i], items[j]
            
            # JACCARD SZŰRÉS: Ha a kódok nagyon különböznek, nem klónok!
            if min_pos_sim > 0:
                sim = jaccard_similarity(a["_code_norm"], b["_code_norm"])
                if sim < min_pos_sim:
                    skipped_low_sim += 1
                    continue

            ha, hb = a["_code_hash"], b["_code_hash"]
            key = f"{min(ha,hb)}::{max(ha,hb)}"
            if key in pair_keys: continue
            pair_keys.add(key)
            
            # Megtartjuk a párt
            pos_pairs_buffer.append({
                "code_a": a["_code_norm"], "code_b": b["_code_norm"],
                "label": 1,
                "idx_a": a.get("idx"), "idx_b": b.get("idx"),
                "split": in_path.stem
            })

    pos_count = len(pos_pairs_buffer)
    print(f" -> Pozitív párok: {pos_count}")
    if skipped_low_sim > 0:
        print(f"    [SZŰRÉS] {skipped_low_sim} pár eldobva, mert a hasonlóság < {min_pos_sim}")

    # --- 2. NEGATÍV PÁROK GYŰJTÉSE ---
    target_neg = int(round(pos_count * neg_per_pos))
    neg_pairs_buffer = []
    
    # Gyorsítótár a véletlen választáshoz
    all_items = [item for sublist in groups.values() for item in sublist]
    
    tries = 0
    max_tries = max(10000, target_neg * 20)
    
    while len(neg_pairs_buffer) < target_neg and tries < max_tries:
        tries += 1
        a = rng.choice(all_items)
        b = rng.choice(all_items)
        
        # Ha ugyanaz a csoport (idx), nem lehet negatív példa
        if str(a["idx"]) == str(b["idx"]): continue
        
        # Hossz-egyeztetés (Hard Negative Mining)
        if length_match:
            len_a = len(a["_code_norm"])
            len_b = len(b["_code_norm"])
            # Kerüljük a 0-val osztást
            ratio = len_b / max(1, len_a)
            if not (length_match[0] <= ratio <= length_match[1]):
                continue

        ha, hb = a["_code_hash"], b["_code_hash"]
        key = f"{min(ha,hb)}::{max(ha,hb)}"
        if key in pair_keys: continue
        pair_keys.add(key)
        
        neg_pairs_buffer.append({
            "code_a": a["_code_norm"], "code_b": b["_code_norm"],
            "label": 0,
            "idx_a": a.get("idx"), "idx_b": b.get("idx"),
            "split": in_path.stem
        })

    print(f" -> Negatív párok: {len(neg_pairs_buffer)}")

    # --- 3. ÍRÁS (Streamelve) ---
    all_pairs = pos_pairs_buffer + neg_pairs_buffer
    rng.shuffle(all_pairs)
    
    with out_path.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f" -> [KÉSZ] Mentve: {out_path} (Összesen: {len(all_pairs)} pár)\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True, help="Mappa a forrás jsonl fájlokkal")
    ap.add_argument("--out-dir", type=str, required=True, help="Kimeneti mappa")
    ap.add_argument("--neg-per-pos", type=float, default=1.0, help="Hány negatív jut egy pozitívra")
    ap.add_argument("--max-pos-per-group", type=int, default=50, help="Max pozitív pár csoportonként")
    
    # ÚJ PARAMÉTER:
    ap.add_argument("--min-pos-sim", type=float, default=0.4, 
                    help="Minimális Jaccard hasonlóság (0.0-1.0), hogy egy azonos idx párt elfogadjunk pozitívnak.")
    
    ap.add_argument("--length-match", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="Negatív mintavételnél a hosszak aránya (pl. 0.5 2.0)")
    ap.add_argument("--strip-erlang-comments", action="store_true")
    ap.add_argument("--no-dedup-exact", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    
    # Mindhárom splitre lefuttatjuk
    for split in ["train", "valid", "test"]:
        p = in_dir / f"{split}.jsonl"
        if p.exists():
            build_for_split(
                p, out_dir / f"{split}_pairs.jsonl",
                args.neg_per_pos, args.max_pos_per_group, args.min_pos_sim,
                tuple(args.length_match) if args.length_match else None,
                args.strip_erlang_comments, not args.no_dedup_exact, args.seed
            )

if __name__ == "__main__":
    main()