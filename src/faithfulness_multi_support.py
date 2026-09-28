"""
faithfulness_multi_support.py
=============================
Benchmark adapters, prompts and IO helpers for faithfulness_multi.

Kept separate so the study logic reads as one file: everything here is about
reconciling three benchmarks that store their documents, passages and labels
differently, without that bookkeeping obscuring the measurements.
"""

import os
import re
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

FINDVER = os.environ.get("FINDVER_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "findver"))
COVERBENCH = os.environ.get("COVERBENCH_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "coverbench", "coverbench.json"))

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def ntok(s):
    if not s:
        return 0
    return len(_ENC.encode(str(s))) if _ENC else len(str(s).split())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


LABELSETS = {"findver": ["True", "False"], "coverbench": ["True", "False"]}

PROMPTS = {
    "findver": {
        "agentic": (
            "You are verifying a statement against a company's SEC filing.\n"
            "You cannot see the full filing. You can SEARCH for relevant "
            "excerpts by issuing queries. On each turn, respond in ONE of these "
            "two formats only:\n\n"
            "SEARCH: <a short query to find relevant filing paragraphs>\n"
            "or\n"
            "ANSWER: <one of: True, False>\n\n"
            "Issue SEARCH queries until you have enough evidence, then give "
            "ANSWER.\nYou have a maximum of {max_steps} searches."
        ),
        "cot": (
            "You are analyzing excerpts from a company's SEC filing.\n\n"
            "Decide whether the excerpts support the STATEMENT. Think step by "
            "step, quoting the specific figures or sentences you rely on, then "
            "finish with exactly one word: True or False.\n\n"
            "EXCERPTS:\n{context}\n\nSTATEMENT:\n{claim}\n\n"
            "Reasoning, then your one-word answer:"
        ),
    },
    "coverbench": {
        "agentic": (
            "You are verifying a claim against a source document.\n"
            "You cannot see the full document. You can SEARCH for relevant "
            "excerpts by issuing queries. On each turn, respond in ONE of these "
            "two formats only:\n\n"
            "SEARCH: <a short query to find relevant passages>\n"
            "or\n"
            "ANSWER: <one of: True, False>\n\n"
            "Issue SEARCH queries until you have enough evidence, then give "
            "ANSWER.\nYou have a maximum of {max_steps} searches."
        ),
        "cot": (
            "You are analyzing excerpts from a source document, which may "
            "contain prose, tables, or both.\n\n"
            "Decide whether the excerpts support the CLAIM. Think step by step, "
            "quoting the specific values or sentences you rely on, then finish "
            "with exactly one word: True or False.\n\n"
            "EXCERPTS:\n{context}\n\nCLAIM:\n{claim}\n\n"
            "Reasoning, then your one-word answer:"
        ),
    },
}


def normalize(raw, bench):
    """Last standalone True/False wins: the model reasons before committing, so
    an early mention inside the reasoning must not be read as the answer."""
    if not raw:
        return "False"
    s = str(raw)
    if "</think>" in s:
        s = s.split("</think>")[-1]
    hits = re.findall(r"\b(true|false)\b", s, re.IGNORECASE)
    return hits[-1].capitalize() if hits else "False"


# ----------------------------------------------------------------------
# Passage segmentation
# ----------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?])\s+")


def _chunk(text, target=400):
    """Sentence-pack CoverBench contexts into ~400-char passages.

    CoverBench ships a rendered blob with no segmentation, so retrieval needs
    units invented for it. Packing whole sentences keeps each passage readable
    and keeps table rows (which arrive as their own lines) intact.
    """
    out, buf = [], ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _SENT.split(line):
            s = s.strip()
            if not s:
                continue
            if len(buf) + len(s) + 1 > target and buf:
                out.append(buf)
                buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf and len(buf) >= target * 0.6:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out or [text[:target]]


def passages_of(bench, it):
    return it["passages"]


def gold_of(bench, it):
    return it.get("gold_idx") or []


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def _stratified(items, n, seed):
    if n is None or n >= len(items):
        return list(items)
    from collections import defaultdict
    by = defaultdict(list)
    for it in items:
        by[it["gold"]].append(it)
    rng = random.Random(seed)
    picked = []
    for lab in sorted(by):
        pool = sorted(by[lab], key=lambda x: str(x["uid"]))
        rng.shuffle(pool)
        picked.extend(pool[:max(1, round(n * len(by[lab]) / len(items)))])
    rng.shuffle(picked)
    return picked[:n]


def load_benchmark(bench, split, n, seed):
    if bench == "findver":
        split = split or "testmini"
        ex = json.load(open(os.path.join(FINDVER, "data", f"{split}.json"),
                            encoding="utf-8"))
        cache = {}
        items = []
        for e in ex:
            rep = e["report"]
            if rep not in cache:
                with open(os.path.join(FINDVER, "financial_reports", rep),
                          encoding="utf-8") as f:
                    d = json.load(f)
                # Paragraph ids are contiguous 0..n-1 and relevant_context
                # indexes them directly -- verified across the split -- so the
                # list position IS the gold id.
                cache[rep] = [c["context"] for c in d["context"]]
            g = e.get("relevant_context") or []
            if isinstance(g, str):
                g = json.loads(g)
            ps = cache[rep]
            items.append({"uid": e["example_id"], "claim": e["statement"],
                          "gold": str(e["entailment_label"]),
                          "passages": ps,
                          "gold_idx": [i for i in g if 0 <= i < len(ps)],
                          "doc_key": rep,
                          "meta": {"subset": e.get("subset"), "report": rep}})
        return _stratified(items, n, seed)

    items = []
    with open(COVERBENCH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ps = _chunk(r["context"])
            items.append({"uid": r["id"], "claim": r["claim"],
                          "gold": str(r["label"]), "passages": ps,
                          # Only Feverous carries evidence, and its ids index
                          # wiki sentences rather than this rendered context,
                          # so there is no sound mapping -- left empty, and the
                          # report states which metrics are unavailable.
                          "gold_idx": [],
                          "doc_key": "cb:" + r["id"],
                          "meta": {"source_dataset": r.get("source_dataset"),
                                   "domain": r.get("domain")}})
    return _stratified(items, n, seed)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def load_done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if not r.get("error"):
                        keys.add(r["uid"])
    return keys


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_concurrent(path, todo, work, concurrency, lock):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_err = 0
    with open(path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(work, t) for t in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                if rec.get("error"):
                    n_err += 1
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"    {i}/{len(todo)}  ({n_err} errored)")
