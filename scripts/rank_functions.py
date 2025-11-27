#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Erlang függvény-rangsoroló V2 (Top-K kiválasztás manuális címkézéshez)

Fő javítások:
- A függvény-fej felismerése robusztus: tree-sitter csomópontok mellett regex fallback ("name(...) ->").
- Támogatja mind a "function", mind a "function_clause" csomópontokat.
- Multisoros -export([...]) felismerés (DOTALL).
- Pontosabb repo-név kivétel és bővebb debug.
- Rugalmas hossz-szűrés (min/max LOC), útvonal-súlyozás, dokumentáció-keresés (-doc / @doc / plain comment).

Kimenet: JSONL (Top-K), soronként egy objektum a legjobb függvényekről.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tree_sitter import Language, Parser

LOG = logging.getLogger("ranker_v2")

# ----------------- logging -----------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------- tree-sitter load -----------------
def find_erlang_lib(grammar_dir: Path) -> Path:
    candidates = [
        grammar_dir / "erlang.dll",
        grammar_dir / "erlang.so",
        grammar_dir / "tree-sitter-erlang.dll",
        grammar_dir / "tree-sitter-erlang.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Nem található a tree-sitter-erlang lib itt: {grammar_dir}")

def load_language(grammar_dir: Path) -> Language:
    lib_path = find_erlang_lib(grammar_dir)
    LOG.info(f"tree-sitter-erlang betöltve: {lib_path}")
    return Language(str(lib_path), "erlang")

# ----------------- util -----------------
EDOC_TAG_RE = re.compile(r"\{@[^}]+\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_WS_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = t.strip()
    t = EDOC_TAG_RE.sub(" ", t)
    t = HTML_TAG_RE.sub(" ", t)
    t = t.replace("`", " ").replace("*", " ").replace("_", " ")
    t = re.sub(r"^\s*%+\s*@doc\s*", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^\s*%+\s*@end\s*", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^\s*%+\s?", "", t, flags=re.MULTILINE)
    t = MULTI_WS_RE.sub(" ", t).strip()
    if t and not t.endswith((".", "!", "?")):
        t += "."
    return t

def parse(source: str, parser: Parser):
    return parser.parse(source.encode("utf-8"))

def get_text(source: str, node) -> str:
    return source[node.start_byte:node.end_byte]

def find_module_name(source: str) -> Optional[str]:
    m = re.search(r"-\s*module\s*\(\s*([a-zA-Z0-9_]+)\s*\)\s*\.\s*", source)
    if m:
        return m.group(1)
    return None

def parse_exports(source: str) -> set:
    out = set()
    # -export([foo/1, bar/2]).
    for m in re.finditer(r"-\s*export\s*\(\s*\[([^\]]*)\]\s*\)\s*\.", source, flags=re.DOTALL):
        body = m.group(1)
        for item in body.split(","):
            item = item.strip()
            mm = re.match(r"^([a-zA-Z0-9_]+)\s*/\s*([0-9]+)$", item)
            if mm:
                out.add((mm.group(1), int(mm.group(2))))
    return out

def strip_line_comments(source: str) -> str:
    out_lines = []
    for ln in source.splitlines():
        if "%" in ln:
            ln = re.sub(r"%.*", "", ln)
        out_lines.append(ln)
    return "\n".join(out_lines)

def path_weight_for(path_str: str, exclude_tests: bool) -> Tuple[float, bool]:
    p = path_str.replace("\\", "/").lower()
    w = 0.0
    is_test = False
    if "/src/" in p or p.endswith("/src") or p.startswith("src/"):
        w += 1.5
    if "test" in p or "eunit" in p:
        w -= 1.0
        is_test = True
    if "bench" in p:
        w -= 0.5
    if "example" in p or "examples" in p:
        w += 0.2
    if exclude_tests and is_test:
        return -999.0, True
    return w, is_test

def length_weight(loc: int) -> float:
    return max(0.0, 1.0 - abs(loc - 25) / 25.0)

def doc_weight_for(doc_type: str) -> float:
    return {"doc/attr": 2.0, "edoc": 1.5, "comment": 1.0, "none": 0.0}.get(doc_type, 0.0)

# ----------------- AST helpers -----------------
def find_nodes_of_types(root, types: Tuple[str, ...]) -> List:
    res = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in types:
            res.append(n)
        stack.extend(n.children)
    return res

HEAD_REGEX = re.compile(
    r"^\s*(?P<name>[a-z][a-zA-Z0-9_]*)\s*\((?P<args>[^)]*)\)\s*->",
    flags=re.MULTILINE | re.DOTALL,
)

def head_from_text(text: str) -> Tuple[Optional[str], Optional[int]]:
    m = HEAD_REGEX.search(text)
    if not m:
        return None, None
    name = m.group("name")
    args = m.group("args").strip()
    if args == "":
        arity = 0
    else:
        # durva, de működő arg-számolás: split top-level vesszőkön (nincs itt mintaillesztés)
        arity = len([a for a in args.split(",") if a.strip() != ""])
    return name, arity

def clause_name_arity(source: str, node) -> Tuple[Optional[str], Optional[int]]:
    # próbáljuk először node-on belüli slice-ból (→ gyors és pontos)
    head_slice = source[node.start_byte:min(node.end_byte, node.start_byte + 400)]
    name, arity = head_from_text(head_slice)
    if name is not None:
        return name, arity

    # általánosabb: keressük visszafelé egy kicsit (ha a '->' a következő soron van)
    ext_slice = source[max(0, node.start_byte - 100):min(len(source), node.start_byte + 600)]
    return head_from_text(ext_slice)

DOC_ATTR_LINE_RE = re.compile(r"^\s*-\s*doc\s*\((.+?)\)\s*\.\s*$", re.IGNORECASE)

def extract_doc_attr_line(line: str) -> Optional[str]:
    mm = DOC_ATTR_LINE_RE.match(line)
    if not mm:
        return None
    inner = mm.group(1).strip()
    if inner.startswith('<<"') and inner.endswith('">>'):
        return clean_text(inner[3:-3])
    if inner.startswith('"') and inner.endswith('"'):
        return clean_text(inner.strip('"'))
    parts = re.findall(r'"([^"\\]|\\.)*"', inner)
    if parts:
        return clean_text(" ".join([p.strip('"') for p in parts]))
    return None

def collect_comment_block_above(lines: List[str], start_line: int, lookback: int) -> List[str]:
    block = []
    for i in range(start_line - 1, max(-1, start_line - lookback - 1), -1):
        line = lines[i].rstrip("\n")
        if re.match(r"^\s*%+", line):
            block.append(line)
        elif line.strip() == "":
            if block:
                break
            else:
                continue
        else:
            break
    return list(reversed(block))

def extract_doc_for_function(source: str, clause, lookback: int) -> Tuple[str, str]:
    """
    doc_text, doc_type ('doc/attr'|'edoc'|'comment'|'none')
    """
    lines = source.splitlines()
    start_line = clause.start_point[0]

    # 1) try -doc(...) attribute within a few lines above
    for i in range(max(0, start_line - 10), start_line):
        txt = extract_doc_attr_line(lines[i]) if i < len(lines) else None
        if txt:
            return txt, "doc/attr"

    # 2) EDoc / comment block
    block = collect_comment_block_above(lines, start_line, lookback)
    if block:
        stripped = [re.sub(r"^\s*%+\s?", "", ln).rstrip() for ln in block]
        joined = "\n".join(stripped).strip()
        # EDoc?
        if any(re.search(r"(?i)@doc", ln) for ln in stripped):
            taking, buf = False, []
            for ln in stripped:
                if re.search(r"(?i)@doc", ln):
                    ln = re.sub(r"(?i)@doc\s*", "", ln).strip()
                    taking = True
                    if ln:
                        buf.append(ln)
                    continue
                if taking and re.search(r"(?i)@end", ln):
                    taking = False
                    continue
                if taking:
                    buf.append(ln)
            t = clean_text("\n".join(buf))
            if t:
                return t, "edoc"
        # Plain comment
        t = clean_text(joined)
        if t:
            return t, "comment"

    return "", "none"

def snippet(source: str, start_byte: int, end_byte: int, max_chars: int = 3000) -> str:
    s = source[start_byte:end_byte]
    s = s.replace("\r\n", "\n")
    if len(s) > max_chars:
        s = s[:max_chars] + "\n... (truncated)"
    return s

def approx_intra_module_calls(clean_src_wo_comments: str, func_name: str) -> int:
    tmp = re.sub(r'"([^"\\]|\\.)*"', '""', clean_src_wo_comments)
    occ = len(re.findall(rf"\b{re.escape(func_name)}\s*\(", tmp))
    return max(0, occ - 1)

# ----------------- main ranking -----------------
def rank_functions(
    repos_root: Path,
    grammar_dir: Path,
    output_path: Path,
    top_k: int,
    exclude_tests: bool,
    min_loc: int,
    max_loc: int,
    lookback: int,
    debug: int,
) -> Dict[str, int]:
    lang = load_language(grammar_dir)
    parser = Parser()
    parser.set_language(lang)

    results: List[Dict[str, Any]] = []
    files = list(repos_root.rglob("*.erl"))
    LOG.info(f".erl fájlok száma: {len(files)}")

    debug_printed = 0

    for fpath in files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        rel_parts = fpath.relative_to(repos_root).parts
        path_rel = "/".join(rel_parts)
        repo_name = rel_parts[0] if len(rel_parts) > 0 else ""
        module_name = find_module_name(source) or fpath.stem

        pw, is_test = path_weight_for(path_rel, exclude_tests)
        if pw <= -999.0:
            continue

        tree = parse(source, parser)
        root = tree.root_node

        # Gyűjtsünk "function" és "function_clause" csomópontokat is
        nodes = find_nodes_of_types(root, ("function", "function_clause"))
        if not nodes:
            continue

        # Csoportosítsuk name/arity szerint → összevont extents
        funcs: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for nd in nodes:
            name, arity = clause_name_arity(source, nd)
            if not name or arity is None:
                continue
            key = (name, arity)
            entry = funcs.get(key)
            if not entry:
                funcs[key] = {
                    "start_byte": nd.start_byte,
                    "end_byte": nd.end_byte,
                    "start_line": nd.start_point[0],
                    "end_line": nd.end_point[0],
                    "any_clause": nd,
                }
            else:
                entry["start_byte"] = min(entry["start_byte"], nd.start_byte)
                entry["end_byte"] = max(entry["end_byte"], nd.end_byte)
                entry["start_line"] = min(entry["start_line"], nd.start_point[0])
                entry["end_line"] = max(entry["end_line"], nd.end_point[0])
                if nd.start_point[0] < entry["any_clause"].start_point[0]:
                    entry["any_clause"] = nd

        if not funcs:
            continue

        exports = parse_exports(source)
        src_no_comments = strip_line_comments(source)

        for (name, arity), ext in funcs.items():
            start_line = ext["start_line"]
            end_line = ext["end_line"]
            loc = end_line - start_line + 1
            if loc < min_loc or loc > max_loc:
                continue

            doc_text, doc_type = extract_doc_for_function(source, ext["any_clause"], lookback)
            doc_prev = (doc_text or "")[:200]
            intra_calls = approx_intra_module_calls(src_no_comments, name)
            exported = (name, arity) in exports

            lw = length_weight(loc)
            dw = doc_weight_for(doc_type)
            ew = 2.5 if exported else 0.0
            calls_w = min(intra_calls, 50) * 0.05  # max +2.5
            score = pw + lw * 0.8 + dw + ew + calls_w

            item = {
                "repo": repo_name,
                "path": path_rel,
                "module": module_name,
                "func_name": name,
                "arity": arity,
                "exported": exported,
                "start_line": start_line + 1,
                "end_line": end_line + 1,
                "loc": loc,
                "doc_type": doc_type or "none",
                "doc_preview": doc_prev,
                "intra_module_calls": intra_calls,
                "path_weight": pw,
                "len_weight": round(lw, 3),
                "doc_weight": dw,
                "export_weight": ew,
                "score": round(float(score), 3),
                "code": snippet(source, ext["start_byte"], ext["end_byte"], max_chars=3000),
            }
            results.append(item)

            # minimál debug dump
            if debug_printed < debug:
                LOG.info(f"[DBG] {repo_name}:{path_rel}:{name}/{arity} "
                         f"loc={loc} exp={exported} doc={item['doc_type']} score={item['score']}")
                debug_printed += 1

    # sort & top-k
    results.sort(key=lambda x: (x["score"], x["exported"], x["loc"]), reverse=True)
    if top_k > 0:
        results = results[:top_k]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOG.info(f"Kiírva: {len(results)} → {output_path}")
    stats = {
        "files": len(files),
        "ranked": len(results),
    }
    return stats

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-root", default="./cloned_repos")
    ap.add_argument("--erlang-grammar-dir", default="./parsers")
    ap.add_argument("--output", default="output/ranked_functions.jsonl")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--exclude-tests", action="store_true")
    ap.add_argument("--min-loc", type=int, default=3)
    ap.add_argument("--max-loc", type=int, default=400)
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--debug", type=int, default=0, help="Ennyi találatról ír minimál debug sort")
    args = ap.parse_args()

    repos_root = Path(args.repos_root).resolve()
    grammar_dir = Path(args.erlang_grammar_dir).resolve()
    output_path = Path(args.output)

    stats = rank_functions(
        repos_root=repos_root,
        grammar_dir=grammar_dir,
        output_path=output_path,
        top_k=args.top_k,
        exclude_tests=args.exclude_tests,
        min_loc=args.min_loc,
        max_loc=args.max_loc,
        lookback=args.lookback,
        debug=args.debug,
    )
    LOG.info(stats)

if __name__ == "__main__":
    main()
