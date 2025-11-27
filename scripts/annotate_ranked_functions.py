#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate ranked Erlang functions with a one-sentence English NL description.

Input:  ranked_functions.jsonl (from rank_functions.py), fields at least:
        repo, path, module, func_name, arity, start_line, code
Output: JSONL lines in CodeSearchNet-like format:
        {"nl": "...", "code": "...", "repo": "...", "path": "...",
         "func_name": "...", "lang": "erlang",
         "idx": "<repo>::<func>/<arity>::<start_line>", "split": "ranked"}

Quality controls:
- Style constraints (8–18 words, start with a verb, no "This function..." etc.)
- Prompt-echo filter
- Code-echo filter
- Length & punctuation checks
- PLUS mode: multiple candidates + best valid pick

Caching: simple SHA1(code) -> nl JSONL cache to avoid re-generation on reruns.
"""

import argparse
import json
import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# ----------------- logging -----------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

LOG = logging.getLogger("annotate")

# ----------------- constants -----------------
MIN_WORDS = 8
MAX_WORDS = 18
CODE_ECHO_RATIO = 0.50
SIMILARITY_THRESHOLD = 0.90

DEFAULT_MODELS = [
    "google/gemma-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

CHAT_PROMPT_USER = """
You are a code-to-text annotator.

TASK
Write exactly ONE short, self-contained English sentence (8–18 words) describing what the given Erlang code DOES.

STYLE
- Start with a verb: Returns, Retrieves, Checks, Converts, Validates, Adds, Opens, Sends, Computes.
- Do NOT start with: "Sure", "Here is/Here's", "This function", "The function", "In Erlang", "This code", "The code".
- Do NOT mention variable names, module/file names, or code syntax.
- Base your sentence ONLY on the visible snippet (no guessing about hidden context).
- If behavior is conditional, summarize both outcomes concisely.
- End with a period.
- Output ONLY the sentence, nothing else.

Code:
{code}
""".strip()

BAD_STARTS = [
    "Sure", "Sure,", "Sure.", "Here is", "Here's", "Here’s",
    "This function", "The function", "In Erlang", "This code", "The code",
    "Okay", "Ok", "Alright", "Well",
    "Sure, here", "Sure here", "Sure here is",
]

PROMPT_ECHO_KEYWORDS = [
    "erlang code",
    "self-contained english sentence",
    "8–18 words",
    "8-18 words",
    "describe what the code does",
]

# ----------------- helpers -----------------
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def fixup_nl(nl: str) -> str:
    nl = (nl or "").strip()
    nl = re.sub(r"^(Sure|Here\s+is|Here’s|Here's|This\s+function|The\s+function|In\s+Erlang|This\s+code|The\s+code)[^A-Za-z0-9]+", "", nl, flags=re.IGNORECASE)
    nl = re.sub(r"\s+", " ", nl).strip()
    if nl:
        nl = nl[0].upper() + nl[1:]
    if nl and not nl.endswith((".", "!", "?")):
        nl += "."
    return nl

def post_filter(nl: str) -> Tuple[bool, str]:
    if not nl:
        return False, "empty"
    words = nl.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False, f"bad_length({len(words)})"
    if not nl[0].isupper():
        return False, "no_uppercase_start"
    if not nl.endswith((".", "!", "?")):
        return False, "no_terminal"
    if re.search(r"[{}\[\]$;`]|->|//|/\*|\*/", nl):
        return False, "code_symbols"
    return True, "OK"

def is_prompt_echo(nl: str) -> bool:
    low = (nl or "").lower()
    return any(k in low for k in PROMPT_ECHO_KEYWORDS)

def is_code_echo(nl: str, code: str) -> bool:
    from difflib import SequenceMatcher
    low = (nl or "").lower()
    ratio = SequenceMatcher(None, low, (code or "").lower()).ratio()
    return ratio > CODE_ECHO_RATIO

def similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() > SIMILARITY_THRESHOLD

def dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items:
        if not it:
            continue
        if any(similar(it, u) for u in out):
            continue
        out.append(it)
    return out

def safe_max_tokens(tok, fallback: int) -> int:
    try:
        ml = getattr(tok, "model_max_length", None)
        if ml is None:
            return fallback
        ml_int = int(ml)
    except Exception:
        return fallback
    # nagyon nagy sentinel értékek levágása
    if ml_int <= 0 or ml_int > 100_000:
        return fallback
    return min(ml_int, fallback)

# ----------------- generator -----------------
class TextGenerator:
    def __init__(self, model_name: Optional[str], device: torch.device, dtype: torch.dtype, max_tokens: int):
        self.device = device
        self.dtype = dtype
        self.kind = "causal"
        self.tok = None
        self.model = None
        self.bad_words_ids = None
        self.max_tokens = max_tokens
        self._load(model_name)

    def _load(self, model_name: Optional[str]):
        names = [model_name] if model_name else DEFAULT_MODELS
        for name in names:
            try:
                LOG.info(f"Loading generator: {name}")
                tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                # biztonságos max length clamp
                tok.model_max_length = safe_max_tokens(tok, self.max_tokens)

                if "mistral" in name.lower() or "gemma" in name.lower():
                    model_cls = AutoModelForCausalLM
                    self.kind = "causal"
                else:
                    model_cls = AutoModelForSeq2SeqLM
                    self.kind = "seq2seq"

                kwargs = {"torch_dtype": self.dtype, "trust_remote_code": True}
                if self.device.type == "cuda":
                    kwargs["device_map"] = "auto"
                    try:
                        import flash_attn  # noqa
                        kwargs["attn_implementation"] = "flash_attention_2"
                    except Exception:
                        kwargs["attn_implementation"] = "sdpa"

                model = model_cls.from_pretrained(name, **kwargs)
                model.eval()

                self.tok = tok
                self.model = model
                self.bad_words_ids = self._build_bad_starts(tok)
                LOG.info(f"Loaded: {name} ({self.kind}), max_tokens={tok.model_max_length}")
                return
            except Exception as e:
                LOG.warning(f"Failed to load {name}: {e}")
        raise RuntimeError("No generator model available.")

    @staticmethod
    def _build_bad_starts(tokenizer):
        enc = tokenizer(BAD_STARTS, add_special_tokens=False)
        return enc["input_ids"]

    def _prompt(self, code: str) -> str:
        prompt = CHAT_PROMPT_USER.format(code=code)
        if self.kind == "causal":
            try:
                chat = [{"role": "user", "content": prompt}]
                return self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                return prompt
        return prompt

    @torch.no_grad()
    def gen_fast(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tok.model_max_length,
        ).to(self.model.device)
        if self.kind == "causal":
            in_len = enc.input_ids.shape[1]
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                max_new_tokens=32,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=self.bad_words_ids,
            )
            txts = self.tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
        else:
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                max_length=64,
                min_length=10,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=self.bad_words_ids,
            )
            txts = self.tok.batch_decode(out, skip_special_tokens=True)
        return [t.strip() for t in txts]

    @torch.no_grad()
    def gen_plus(self, prompts: List[str], beams: int = 4, samples: int = 2) -> List[List[str]]:
        """
        Stabil PLUS: mintavételezésnél num_beams=1, num_return_sequences=beams+samples,
        így nincs num_return_sequences <= num_beams hiba.
        """
        if not prompts:
            return []
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tok.model_max_length,
        ).to(self.model.device)

        nret = max(1, beams) + max(0, samples)

        if self.kind == "causal":
            in_len = enc.input_ids.shape[1]
            out = self.model.generate(
                **enc,
                num_beams=1,                 # <<< fontos
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=nret,   # <<< több jelölt
                max_new_tokens=32,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=self.bad_words_ids,
            )
            dec = self.tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
        else:
            out = self.model.generate(
                **enc,
                num_beams=1,                 # <<< fontos
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=nret,   # <<< több jelölt
                max_length=64,
                min_length=10,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=self.bad_words_ids,
            )
            dec = self.tok.batch_decode(out, skip_special_tokens=True)

        grouped: List[List[str]] = []
        for i in range(len(prompts)):
            grouped.append([d.strip() for d in dec[i*nret:(i+1)*nret]])
        return grouped

# ----------------- IO -----------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------- pipeline -----------------
def annotate(
    input_path: Path,
    output_path: Path,
    mode: str,
    batch_size: int,
    limit: int,
    model_name: Optional[str],
    max_code_chars: int,
    resume: bool,
    max_tokens: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    LOG.info(f"Device: {device.type} | dtype={dtype}")

    gen = TextGenerator(model_name=model_name, device=device, dtype=dtype, max_tokens=max_tokens)

    # cache
    cache_dir = output_path.parent / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "nl_cache.jsonl"
    cache: Dict[str, str] = {}
    if cache_path.exists():
        for obj in load_jsonl(cache_path):
            if "k" in obj and "v" in obj:
                cache[obj["k"]] = obj["v"]

    seen_idx = set()
    if resume and output_path.exists():
        LOG.info(f"Resume mode: scanning existing output: {output_path}")
        for obj in load_jsonl(output_path):
            seen_idx.add(obj.get("idx"))

    rows = load_jsonl(input_path)
    if limit > 0:
        rows = rows[:limit]
    LOG.info(f"Loaded {len(rows)} ranked items")

    if output_path.exists() and not resume:
        output_path.unlink()

    i = 0
    processed = 0
    skipped = 0

    while i < len(rows):
        batch = rows[i:i+batch_size]
        i += batch_size

        prompts: List[str] = []
        ctx: List[Dict[str, Any]] = []

        for r in batch:
            repo = r.get("repo")
            path = r.get("path")
            func = r.get("func_name")
            arity = r.get("arity")
            start_line = r.get("start_line")
            code = (r.get("code") or "").strip()
            if not code:
                skipped += 1
                continue

            idx = f"{repo}::{func}/{arity}::{start_line}"
            if resume and idx in seen_idx:
                continue

            code_for_prompt = code
            if max_code_chars > 0 and len(code_for_prompt) > max_code_chars:
                code_for_prompt = code_for_prompt[:max_code_chars] + "\n... (truncated)"

            key = sha1_text(code_for_prompt)
            ctx.append({
                "repo": repo, "path": path, "func_name": func, "arity": arity,
                "start_line": start_line, "code": code, "idx": idx, "key": key
            })
            prompts.append(gen._prompt(code_for_prompt))

        if not prompts:
            continue

        outs: List[Optional[str]] = []
        to_gen_prompts: List[str] = []
        to_gen_ctx_idx: List[int] = []
        for j, c in enumerate(ctx):
            if c["key"] in cache:
                outs.append(cache[c["key"]])
            else:
                outs.append(None)
                to_gen_prompts.append(prompts[j])
                to_gen_ctx_idx.append(j)

        gen_outs: List[str] = []
        if to_gen_prompts:
            if mode == "fast":
                gen_outs = gen.gen_fast(to_gen_prompts)
            else:
                grouped = gen.gen_plus(to_gen_prompts, beams=4, samples=2)
                for cand_list in grouped:
                    fixeds = [fixup_nl(c) for c in cand_list if c.strip()]
                    fixeds = dedup_keep_order(fixeds)
                    fixeds = [f for f in fixeds if not is_prompt_echo(f)]
                    picked = ""
                    for f in fixeds:
                        ok, _ = post_filter(f)
                        if ok:
                            picked = f; break
                    if not picked:
                        picked = fixeds[0] if fixeds else ""
                    gen_outs.append(picked)

        cursor = 0
        for j, c in enumerate(ctx):
            nl = outs[j]
            if nl is None:
                raw = gen_outs[cursor] if cursor < len(gen_outs) else ""
                cursor += 1
                fixed = fixup_nl(raw)

                fail = None
                if is_prompt_echo(fixed):
                    fail = "prompt_echo"
                elif is_code_echo(fixed, c["code"]):
                    fail = "code_echo"
                else:
                    ok, why = post_filter(fixed)
                    if not ok:
                        fail = why

                if fail:
                    if mode == "fast":
                        g = gen.gen_plus([gen._prompt(c["code"][:max_code_chars])], beams=4, samples=2)
                        cand_list = g[0] if g else []
                        fixeds = [fixup_nl(x) for x in cand_list if x.strip()]
                        fixeds = dedup_keep_order(fixeds)
                        fixeds = [f for f in fixeds if not is_prompt_echo(f)]
                        picked = ""
                        for f in fixeds:
                            ok, _ = post_filter(f)
                            if ok and not is_code_echo(f, c["code"]):
                                picked = f; break
                        if not picked:
                            skipped += 1
                            continue
                        nl = picked
                    else:
                        skipped += 1
                        continue
                else:
                    nl = fixed

                cache[c["key"]] = nl
                append_jsonl(cache_path, {"k": c["key"], "v": nl})

            final = {
                "nl": nl,
                "code": c["code"],
                "repo": c["repo"],
                "path": c["path"],
                "func_name": c["func_name"],
                "lang": "erlang",
                "idx": c["idx"],
                "split": "ranked"
            }
            append_jsonl(output_path, final)
            processed += 1

        LOG.info(f"progress: {processed} ok, {skipped} skipped (out: {output_path.name})")

    LOG.info(f"Done. ok={processed} skipped={skipped} → {output_path}")

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ranked_functions.jsonl")
    ap.add_argument("--output", required=True, help="annotated JSONL output")
    ap.add_argument("--mode", choices=["fast", "plus"], default="plus")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", type=str, default="google/gemma-2b-it")
    ap.add_argument("--max-code-chars", type=int, default=1600)
    ap.add_argument("--max-tokens", type=int, default=2048, help="safe tokenizer/generation context length")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    annotate(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        mode=args.mode,
        batch_size=max(1, args.batch_size),
        limit=max(0, args.limit),
        model_name=args.model,
        max_code_chars=max(0, args.max_code_chars),
        resume=bool(args.resume),
        max_tokens=max(256, args.max_tokens),
    )

if __name__ == "__main__":
    main()
