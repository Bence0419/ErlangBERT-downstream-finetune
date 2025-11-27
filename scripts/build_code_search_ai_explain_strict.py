#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Erlang kódmagyarázat generátor (build_code_search_ai_explain_strict.py)

VERZIÓ 7.0: "No-Sure • Neutral Few-shot • Deterministic FAST • Safe PLUS"

- Semleges (domainfüggetlen) példák a promptban — nem ragad be a "virtual hosts/view row" szókészlet.
- Tiltólista (BANNED_PHRASES) + bad_words_ids a makacs panelek ellen.
- FAST módban determinisztikus (beam-only), kevesebb blabla.
- PLUS módban kétfázisú generálás (beams + mintavétel), így elkerüljük a
  `num_return_sequences <= num_beams` hibát.
- Stabil, simított ETA, batch feldolgozás, cache (SHA1 a megtisztított kódra).
"""

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import argparse
import json
import logging
import re
import time
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from tqdm.auto import tqdm

# ===== ETA =====

def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

class ETA:
    """Stabil ETA: teljesített / eltelt idő → throughput, majd simított becslés."""
    def __init__(self, total: int, smooth: float = 0.25):
        self.total = int(total)
        self.start = time.perf_counter()
        self.done = 0
        self.smooth = smooth
        self.smoothed_eta = None

    def update(self, inc: int = 1) -> None:
        self.done += inc
        now = time.perf_counter()
        elapsed = max(1e-6, now - self.start)
        if self.done <= 0:
            return
        throughput = self.done / elapsed  # items/sec
        remaining = max(0, self.total - self.done)
        eta_now = remaining / max(1e-6, throughput)
        if self.smoothed_eta is None:
            self.smoothed_eta = eta_now
        else:
            self.smoothed_eta = self.smooth * eta_now + (1 - self.smooth) * self.smoothed_eta

    def render(self) -> Tuple[str, str]:
        now = time.perf_counter()
        elapsed = now - self.start
        eta_str = _fmt_hms(self.smoothed_eta) if self.smoothed_eta is not None else "?"
        return _fmt_hms(elapsed), eta_str

# ===== Globális beállítások =====

GENERATOR_MODELS_TO_TRY = [
    "google/gemma-2b-it",                 # elsődleges
    "mistralai/Mistral-7B-Instruct-v0.2", # fallback
    "Salesforce/codet5p-770m"             # seq2seq fallback
]

# Szűrési paraméterek (egymondatos magyarázat)
MIN_WORDS = 8
MAX_WORDS = 18
SIMILARITY_THRESHOLD = 0.90   # duplikátum-küszöb
CODE_ECHO_RATIO = 0.50        # kód-echo küszöb

# Regexek
CODE_SYMBOLS_RE = re.compile(r"""
    [\[\]\$\{\}\;] | \-\> | \/\/ | \/\* | \*\/
""", re.VERBOSE)
FIXUP_SYMBOLS_RE = re.compile(r"""
    [\[\]\$\{\}\;\(\)\:\'\"`_] | \-\> | \/\/ | \/\* | \*\/ | ~
""", re.VERBOSE)
BOILERPLATE_RE = re.compile(r"""
    (
        copyright | licensed | apache | mit | gpl | lgpl |
        chromium | android\sopen\ssource |
        namespace | using\ssystem | class\s | \#include | import\s | package\s |
        def\s | public\sstatic | enum\s | template |
        std:: | printf\( | erlang | otp\s| \s(e\.g\.|i\.e\.)\s
    )
""", re.IGNORECASE | re.VERBOSE)

# Prompt-echo kulcsszavak
PROMPT_ECHO_KEYWORDS = [
    "erlang function", "concise english sentence", "high-level purpose",
    "variable names", "code syntax", "1-30 words", "function name",
    "explain this", "6-30 words", "english sentence"
]

# Kifejezetten tiltott panelek (utó-szűrés)
BANNED_PHRASES = [
    "virtual host", "virtual hosts", "vhosts",
    "view row", "view rows",
    "returns undefined if it is missing",
    "or returns undefined if it is missing",
    "the view row is missing",
    "from a view row",
]

# Chat prompt sablon (CausalLM-hez) — SEMLEGES példákkal!
CHAT_PROMPT_USER_CONTENT = """
You are a code-to-text annotator.

TASK
Write exactly ONE short, self-contained English sentence (8–18 words) describing what the given Erlang code DOES.

STYLE
- Start with a verb: Returns, Retrieves, Checks, Converts, Validates, Adds, Opens, Sends, Computes.
- Do NOT start with: "Sure", "Here is/Here's", "This function", "The function", "In Erlang", "This code", "The code".
- Do NOT mention variable names, module/file names, or code syntax.
- Base your sentence ONLY on the visible snippet (no guessing about hidden context).
- If behavior is conditional, summarize both outcomes (e.g., "… if …; otherwise …").
- End with a period.
- Output ONLY the sentence, nothing else.

EXAMPLES
Code:
handle_close(Sock) -> gen_tcp:close(Sock).
Answer:
Closes the TCP socket.

Code:
incr(N) when is_integer(N) -> N + 1.
Answer:
Returns the input integer increased by one.

Code:
maybe_put(Map, K, V) ->
  case maps:is_key(K, Map) of
    true  -> Map;
    false -> maps:put(K, V, Map)
  end.
Answer:
Adds the key-value pair only if missing; otherwise returns the original map.

Code:
{clean_code}
""".strip()

# ===== Log =====

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

# ===== I/O =====

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"JSONL hiba ({path.name}): {e}")
    return data

def save_jsonl_entry(entry: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ===== Tisztítás / fixup =====

def clean_erlang_code(code: str, max_lines: int, max_chars: int) -> str:
    # sorvégi kommentek törlése
    code = re.sub(r'%.*?($|\n)', '\n', code)
    lines = [ln.strip() for ln in code.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    clean = "\n".join(lines)
    if max_chars > 0 and len(clean) > max_chars:
        clean = clean[:max_chars] + "\n... (truncated)"
    if not clean and code.strip():
        raw_lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
        clean = raw_lines[0][:120] if raw_lines else ""
    return clean.strip()

def fixup_nl(nl: str) -> str:
    """Erős utó-tisztítás."""
    nl = (nl or "").strip()
    nl = re.sub(r"^(Explanation|Rewritten)\s*:\s*", "", nl).strip()
    nl = nl.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")
    nl = FIXUP_SYMBOLS_RE.sub(" ", nl)
    # Kezdő “Sure/Here/This function/The function/The code/In Erlang” eltávolítása
    nl = re.sub(
        r"^(Sure|Here\s+is|Here’s|Here's|This\s+function|The\s+function|In\s+Erlang|This\s+code|The\s+code)[^A-Za-z0-9]+",
        "", nl, flags=re.IGNORECASE
    )
    nl = re.sub(r"\s+", " ", nl).strip()
    if nl:
        nl = nl[0].upper() + nl[1:]
        if not nl.endswith(('.', '?', '!')):
            nl += '.'
    return nl

def post_filter_candidate(nl: str) -> Tuple[bool, str]:
    if not nl or not isinstance(nl, str):
        return False, "Empty/invalid"
    low = nl.lower()
    for p in BANNED_PHRASES:
        if p in low:
            return False, f"BANNED_PHRASE: {p}"
    w = nl.strip().split()
    if not (MIN_WORDS <= len(w) <= MAX_WORDS):
        return False, f"Bad length ({len(w)})"
    if not nl[0].isupper():
        return False, "No uppercase start"
    if not nl.endswith(('.', '?', '!')):
        return False, "No terminal punctuation"
    if BOILERPLATE_RE.search(nl):
        return False, "Boilerplate terms"
    if CODE_SYMBOLS_RE.search(nl):
        return False, "Code symbols"
    return True, "OK"

def is_prompt_echo(nl: str) -> bool:
    low = (nl or "").lower()
    for k in PROMPT_ECHO_KEYWORDS:
        if k in low:
            return True
    return False

def is_code_echo(nl: str, code: str, func_name: str) -> bool:
    low = (nl or "").lower()
    fn = (func_name or "").lower().strip()
    if fn and len(fn) >= 5:
        if re.search(rf"\b{re.escape(fn)}\b", low):
            return True
    ratio = SequenceMatcher(None, low, (code or "").lower()).ratio()
    return ratio > CODE_ECHO_RATIO

def deduplicate_texts(items: List[str]) -> List[str]:
    uniq: List[str] = []
    for it in items:
        if not it:
            continue
        dup = False
        for u in uniq:
            if SequenceMatcher(None, it, u).ratio() > SIMILARITY_THRESHOLD:
                dup = True
                break
        if not dup:
            uniq.append(it)
    return uniq

# ===== Modell betöltés =====

def _build_bad_starts(tokenizer) -> List[List[int]]:
    phrases = [
        "Sure", "Sure,", "Sure.", "Sure:", "Sure here", "Sure, here",
        "Here is", "Here’s", "Here's", "Here is the sentence", "Here is a sentence",
        "This function", "The function", "In Erlang", "This code", "The code",
        "Okay", "Ok", "Alright", "Well",
        # gyakran felbukkanó panelek kezdőmagjai
        "virtual host", "virtual hosts", "vhosts",
        "view row", "view rows",
        "returns undefined if it is missing",
    ]
    enc = tokenizer(phrases, add_special_tokens=False)
    return enc["input_ids"]

class TextGenerator:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tok = None
        self.kind = None  # "causal" | "seq2seq"
        self.bad_starts_ids: Optional[List[List[int]]] = None
        self._load()

    def _load(self):
        if torch.cuda.is_available():
            logging.info(f"CUDA: {torch.cuda.get_device_name(0)} | dtype={str(self.dtype).split('.')[-1]}")
        else:
            logging.info("CPU futás | dtype=float32")

        for name in GENERATOR_MODELS_TO_TRY:
            try:
                logging.info(f"Generátor betöltése: {name}")
                tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

                # explicit max_length
                if getattr(tok, "model_max_length", None) in (None, float("inf"), 1000000000000000019884624838656):
                    tok.model_max_length = 4096

                # modell típus
                if "gemma" in name.lower() or "mistral" in name.lower():
                    klass = AutoModelForCausalLM
                    self.kind = "causal"
                else:
                    klass = AutoModelForSeq2SeqLM
                    self.kind = "seq2seq"

                kwargs = {
                    "torch_dtype": self.dtype,
                    "trust_remote_code": True
                }
                if self.device.type == "cuda":
                    kwargs["device_map"] = "auto"
                    try:
                        import flash_attn  # noqa: F401
                        kwargs["attn_implementation"] = "flash_attention_2"
                    except Exception:
                        kwargs["attn_implementation"] = "sdpa"

                model = klass.from_pretrained(name, **kwargs)
                model.eval()

                self.model = model
                self.tok = tok
                self.bad_starts_ids = _build_bad_starts(self.tok)
                logging.info(f"Siker: {name} ({self.kind})")
                return
            except Exception as e:
                logging.warning(f"Betöltés sikertelen ({name}): {e}")
        raise RuntimeError("Egyik generátor modellt sem sikerült betölteni.")

    def _make_prompt(self, clean_code: str) -> str:
        if self.kind == "causal":
            chat = [
                {"role": "user", "content": CHAT_PROMPT_USER_CONTENT.format(clean_code=clean_code)}
            ]
            try:
                return self.tok.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                return CHAT_PROMPT_USER_CONTENT.format(clean_code=clean_code)
        else:
            return CHAT_PROMPT_USER_CONTENT.format(clean_code=clean_code)

    @torch.no_grad()
    def gen_batch_fast(self, prompts: List[str]) -> List[str]:
        """
        FAST mód: determinisztikus, beam-only (kevesebb "duma"), 1 jelölt / input.
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

        bad_words_ids = self.bad_starts_ids

        if self.kind == "causal":
            input_len = enc.input_ids.shape[1]
            out = self.model.generate(
                **enc,
                do_sample=False,
                num_beams=4,
                num_return_sequences=1,
                max_new_tokens=32,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
            txts = self.tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
        else:
            out = self.model.generate(
                **enc,
                do_sample=False,
                num_beams=4,
                num_return_sequences=1,
                max_length=64,
                min_length=10,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
            txts = self.tok.batch_decode(out, skip_special_tokens=True)

        return [t.strip() for t in txts]

    @torch.no_grad()
    def gen_batch_plus(self, prompts: List[str], beams: int = 4, samples: int = 2) -> List[List[str]]:
        """
        PLUS mód: két fázis — (A) beam-only determinisztikus, (B) sampling-only.
        Így elkerüljük a `num_return_sequences <= num_beams` korlát ütközéseit,
        és mégis kapunk több jelöltet inputonként.
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

        bad_words_ids = self.bad_starts_ids

        grouped: List[List[str]] = [[] for _ in range(len(prompts))]

        # --- (A) Beam-only (deterministic), num_return_sequences = beams ---
        if self.kind == "causal":
            input_len = enc.input_ids.shape[1]
            out_a = self.model.generate(
                **enc,
                do_sample=False,
                num_beams=max(1, beams),
                num_return_sequences=max(1, beams),
                max_new_tokens=32,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
            dec_a = self.tok.batch_decode(out_a[:, input_len:], skip_special_tokens=True)
        else:
            out_a = self.model.generate(
                **enc,
                do_sample=False,
                num_beams=max(1, beams),
                num_return_sequences=max(1, beams),
                max_length=64,
                min_length=10,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
            dec_a = self.tok.batch_decode(out_a, skip_special_tokens=True)

        # szétbontás beam-jelöltekre
        B = max(1, beams)
        for i in range(len(prompts)):
            grouped[i].extend([t.strip() for t in dec_a[i*B:(i+1)*B]])

        # --- (B) Sampling-only, num_beams=1, num_return_sequences=samples ---
        if samples > 0:
            if self.kind == "causal":
                out_b = self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_beams=1,
                    num_return_sequences=samples,
                    max_new_tokens=32,
                    pad_token_id=self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                    bad_words_ids=bad_words_ids,
                )
                dec_b = self.tok.batch_decode(out_b[:, input_len:], skip_special_tokens=True)
            else:
                out_b = self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_beams=1,
                    num_return_sequences=samples,
                    max_length=64,
                    min_length=10,
                    pad_token_id=self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                    bad_words_ids=bad_words_ids,
                )
                dec_b = self.tok.batch_decode(out_b, skip_special_tokens=True)

            for i in range(len(prompts)):
                grouped[i].extend([t.strip() for t in dec_b[i*samples:(i+1)*samples]])

        return grouped

# ===== Pipeline =====

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def retry_plus_single(gen: "TextGenerator", clean_code: str) -> Optional[str]:
    """Egyetlen inputra kis költségű PLUS újrapróba, elfogadható jelölt visszaadása vagy None."""
    prompt = gen._make_prompt(clean_code)
    grouped = gen.gen_batch_plus([prompt], beams=4, samples=2)
    if not grouped:
        return None
    cand_list = grouped[0]
    fixed = [fixup_nl(c) for c in cand_list if c.strip()]
    fixed = deduplicate_texts(fixed)
    fixed = [f for f in fixed if not is_prompt_echo(f)]
    final = [f for f in fixed if post_filter_candidate(f)[0]]
    return final[0] if final else None

def process_split(
    split_name: str,
    data: List[Dict[str, Any]],
    out_path: Path,
    raw_path: Path,
    gen: TextGenerator,
    mode: str,
    batch_size: int,
    truncate_lines: int,
    max_chars: int,
    limit: int,
    use_cache: bool,
    beams_plus: int,
    samples_plus: int,
    eta_every_sec: int,
):
    if out_path.exists():
        out_path.unlink()
    if raw_path.exists():
        raw_path.unlink()

    # Egyszerű cache: kód-SHA1 → NL
    cache_path = out_path.parent / "_raw" / f"{split_name}_cache.jsonl"
    seen_cache: Dict[str, str] = {}
    if use_cache and cache_path.exists():
        for obj in load_jsonl(cache_path):
            if "k" in obj and "v" in obj:
                seen_cache[obj["k"]] = obj["v"]

    n_total = len(data) if limit <= 0 else min(limit, len(data))
    logging.info(f"{split_name}: {n_total} rekord feldolgozása (mód={mode}, batch={batch_size})")

    i = 0
    processed = 0
    skipped = 0

    eta = ETA(total=n_total, smooth=0.25)
    last_log = time.perf_counter()
    log_interval_sec = max(5, eta_every_sec or 60)

    while i < n_total:
        batch = data[i: i + batch_size]
        i += batch_size

        # előkészítés: clean code + prompt
        prompts: List[str] = []
        ctx: List[Dict[str, Any]] = []

        for entry in batch:
            code = entry.get("code", "")
            idx = entry.get("idx", "unknown")
            func_name = entry.get("func_name", "")
            clean = clean_erlang_code(code, truncate_lines, max_chars)
            if not clean:
                save_jsonl_entry({
                    "idx": idx,
                    "func_name": func_name,
                    "reason": "EMPTY_AFTER_CLEAN"
                }, raw_path)
                skipped += 1
                continue

            key = sha1_text(clean)
            ctx.append({
                "entry": entry,
                "clean": clean,
                "key": key
            })
            prompts.append(gen._make_prompt(clean))

        if not prompts:
            continue

        # cache ellenőrzés
        cached_results: List[Optional[str]] = []
        to_generate_prompts: List[str] = []
        to_generate_ctx_idx: List[int] = []

        for j, c in enumerate(ctx):
            if use_cache and c["key"] in seen_cache:
                cached_results.append(seen_cache[c["key"]])
            else:
                cached_results.append(None)
                to_generate_prompts.append(prompts[j])
                to_generate_ctx_idx.append(j)

        # generálás (csak a hiányzókra)
        gen_outputs: List[str] = []
        if to_generate_prompts:
            if mode == "fast":
                gen_outputs = gen.gen_batch_fast(to_generate_prompts)
            else:
                grouped = gen.gen_batch_plus(to_generate_prompts, beams=beams_plus, samples=samples_plus)
                for cand_list in grouped:
                    fixed = [fixup_nl(c) for c in cand_list if c.strip()]
                    fixed = deduplicate_texts(fixed)
                    fixed = [f for f in fixed if not is_prompt_echo(f)]
                    final = [f for f in fixed if post_filter_candidate(f)[0]]
                    picked = final[0] if final else (fixed[0] if fixed else "")
                    gen_outputs.append(picked)

        # visszavetítés a batch helyeire
        out_cursor = 0
        for j, c in enumerate(ctx):
            idx = c["entry"].get("idx", "unknown")
            func_name = c["entry"].get("func_name", "")
            code = c["entry"].get("code", "")

            nl_text = cached_results[j]
            debug_info = {"idx": idx, "func_name": func_name}

            if nl_text is None:
                # újonnan generált
                raw = gen_outputs[out_cursor] if out_cursor < len(gen_outputs) else ""
                out_cursor += 1

                fixed = fixup_nl(raw)

                failed_reason = None
                if is_prompt_echo(fixed):
                    failed_reason = "PROMPT_ECHO"
                elif is_code_echo(fixed, code, func_name):
                    failed_reason = "CODE_ECHO"
                else:
                    ok, why = post_filter_candidate(fixed)
                    if not ok:
                        failed_reason = f"FORMAT_FAIL: {why}"

                # ha bukott: egyszeri plus-retry
                if failed_reason:
                    picked = retry_plus_single(gen, c["clean"])
                    if picked:
                        nl_text = picked
                        if use_cache:
                            seen_cache[c["key"]] = nl_text
                            save_jsonl_entry({"k": c["key"], "v": nl_text}, cache_path)
                    else:
                        debug_info["reason"] = failed_reason
                        save_jsonl_entry(debug_info, raw_path)
                        skipped += 1
                        continue
                else:
                    nl_text = fixed
                    if use_cache:
                        seen_cache[c["key"]] = nl_text
                        save_jsonl_entry({"k": c["key"], "v": nl_text}, cache_path)

            # végső CodeSearchNet-formátumú sor
            final_entry = {
                "nl": nl_text,
                "code": c["entry"].get("code"),
                "repo": c["entry"].get("repo"),
                "path": c["entry"].get("path"),
                "func_name": c["entry"].get("func_name"),
                "lang": "erlang",
                "idx": c["entry"].get("idx"),
                "split": split_name
            }
            save_jsonl_entry(final_entry, out_path)
            processed += 1

        # progress + ETA
        eta.update(inc=len(ctx))
        now = time.perf_counter()
        if (now - last_log) >= log_interval_sec:
            elapsed_str, eta_str = eta.render()
            logging.info(f"[{split_name}] {min(i, n_total)}/{n_total} (ok: {processed}, skip: {skipped}) | elapsed {elapsed_str} | ETA {eta_str}")
            last_log = now

        # időnként is log
        if (min(i, n_total)) % (batch_size * 5) == 0:
            elapsed_str, eta_str = eta.render()
            logging.info(f"[{split_name}] {min(i, n_total)}/{n_total} kész (ok: {processed}, skip: {skipped}) | elapsed {elapsed_str} | ETA {eta_str}")

    logging.info(f"[{split_name}] Kész. Sikeres: {processed}, Kihagyva: {skipped}, Össz: {n_total}")

# ===== Main =====

def detect_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32

def main():
    parser = argparse.ArgumentParser(
        description="Egymondatos NL magyarázatok generálása Erlang függvényekhez (CodeSearchNet formátum)"
    )
    parser.add_argument("--splits-dir", type=str, default="output/graphcodebert_data",
                        help="Bemeneti {train,valid,test}.jsonl mappa.")
    parser.add_argument("--out-dir", type=str, default="output/code_search_data_ai",
                        help="Kimeneti mappa (ide írja a {train,valid,test}.jsonl-t és a _raw debugot).")
    parser.add_argument("--mode", type=str, choices=["fast", "plus"], default="fast",
                        help="FAST: determinisztikus 1 jelölt; PLUS: több jelölt (beams+samples).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch-méret generáláskor.")
    parser.add_argument("--truncate-lines", type=int, default=80,
                        help="Forráskód max sorainak száma (0 = nincs limit).")
    parser.add_argument("--max-chars", type=int, default=1600,
                        help="Forráskód max karakterszáma (0 = nincs limit).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Rekordlimit splittenként (0 = összes).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Cache kikapcsolása.")
    parser.add_argument("--beams", type=int, default=4,
                        help="[plus] beam search sugár.")
    parser.add_argument("--samples", type=int, default=2,
                        help="[plus] mintavételes jelöltek száma (beam mellett külön fut).")
    parser.add_argument("--eta-every-sec", type=int, default=60,
                        help="ETA kiírás időköze másodpercben (0 = auto).")

    args = parser.parse_args()
    setup_logging()

    device, dtype = detect_device_and_dtype()
    logging.info(f"{'CUDA' if device.type=='cuda' else 'CPU'}: "
                 f"{torch.cuda.get_device_name(0) if device.type=='cuda' else ''} | dtype={str(dtype).split('.')[-1]}")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # modell
    gen = TextGenerator(device, dtype)

    # Splitek
    splits = ["train", "valid", "test"]
    for split in splits:
        in_path = Path(args.splits_dir) / f"{split}.jsonl"
        if not in_path.exists():
            logging.warning(f"Bemeneti fájl hiányzik: {in_path} — kihagyom.")
            continue

        data = load_jsonl(in_path)
        if not data:
            logging.warning(f"Nincs adat: {in_path}")
            continue

        # limit debughoz
        if args.limit and args.limit > 0:
            data = data[:args.limit]

        out_path = out_dir / f"{split}.jsonl"
        raw_path = raw_dir / f"{split}_raw.jsonl"
        process_split(
            split_name=split,
            data=data,
            out_path=out_path,
            raw_path=raw_path,
            gen=gen,
            mode=args.mode,
            batch_size=max(1, args.batch_size),
            truncate_lines=max(0, args.truncate_lines),
            max_chars=max(0, args.max_chars),
            limit=max(0, args.limit),
            use_cache=not args.no_cache,
            beams_plus=max(1, args.beams),
            samples_plus=max(0, args.samples),
            eta_every_sec=args.eta_every_sec,
        )

    logging.info("=== Befejezve ===")
    logging.info(f"Kimenet: {out_dir}")
    logging.info(f"Debug:   {raw_dir}")

if __name__ == "__main__":
    main()
