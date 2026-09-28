"""
rlm_multi.py
============
The RLM arm across all three claim-verification benchmarks, on a subset.

Why a subset: one FinDVer instance took 440s on OpenRouter (14 model calls,
20 rate-limit retries absorbed), and sub-call attribution requires concurrency
1 -- llm_query calls arrive over a socket on the LMHandler thread, so parallel
workers would interleave in the shared ledger. Full splits are a multi-day job;
this samples each benchmark so the recursion behaviour can be measured now.

Benchmarks
----------
  contractnli  3-way (Entailment / Contradiction / NotMentioned), ~11K chars
  findver      binary True/False, ~209K chars  -> recursion expected
  coverbench   binary True/False, median 4K chars, heterogeneous sources

Each is sampled stratified by gold label at a fixed seed, so the subset is
reproducible and class-balanced rather than whatever the file order gives.

Decoding
--------
The rlm library sends NO temperature and NO reasoning field unless sampling_args
carries them, so the RLM arm has been running at the provider default (~1.0,
thinking per provider) while every other condition in the study runs greedy at
temperature 0.0. That is an uncontrolled difference, not a design choice, so
this driver sets both explicitly. Default is temperature 0.0 with thinking OFF,
matching the main matrix; --thinking flips the reasoning flag.

Per-call accounting
-------------------
Every model call is recorded via the run_rlm_traced tracer: role (root vs
sub-call), prompt/response tokens from both tiktoken and the provider, latency,
and each call's prompt as a fraction of the document in tokens -- the number
that says whether a sub-call actually carried evidence.

Usage
-----
    python rlm_multi.py --benchmark findver --n 20
    python rlm_multi.py --benchmark contractnli --n 20
    python rlm_multi.py --benchmark coverbench --n 20 --thinking
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("CONTRACTNLI_RESULTS_DIR", os.path.join(_ROOT, "results"))
import re
import json
import time
import random
import argparse
import statistics

import run_rlm_traced as T
import rlm_utf8_patch              # UTF-8 fix for the rlm REPL context file (Windows)

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
    if _ENC is not None:
        return len(_ENC.encode(str(s)))
    return len(str(s).split())


# ----------------------------------------------------------------------
# Root prompts. Same shape across benchmarks -- the document lives in the
# REPL, the root model receives only the question -- differing only in the
# document type and the label set, which is what makes the arms comparable.
# ----------------------------------------------------------------------
PROMPTS = {
    "contractnli": (
        "The `context` variable in your REPL holds the full text of a "
        "non-disclosure agreement (NDA).\n\n"
        "Decide the relationship between that contract and the following "
        "hypothesis:\n\nHYPOTHESIS: {claim}\n\n"
        "Answer Entailment if the contract implies the hypothesis, "
        "Contradiction if the contract implies its negation, and NotMentioned "
        "if the contract does not address it.\n"
        "Inspect the contract in the REPL before answering. Finish with "
        "exactly one of these labels: Entailment, Contradiction, NotMentioned."
    ),
    "findver": (
        "The `context` variable in your REPL holds the full text of a "
        "company's SEC filing (a 10-K or 10-Q report). It is a long document: "
        "paragraphs are numbered one per line as `[<id>] <text>`.\n\n"
        "Decide whether the filing supports the following statement:\n\n"
        "STATEMENT: {claim}\n\n"
        "Answer True if the filing entails the statement, and False if it "
        "contradicts it or the figures do not match.\n"
        "Inspect the filing in the REPL before answering. Finish with exactly "
        "one word: True or False."
    ),
    "coverbench": (
        "The `context` variable in your REPL holds a source document, which "
        "may contain prose, tables, or both.\n\n"
        "Decide whether the document supports the following claim:\n\n"
        "CLAIM: {claim}\n\n"
        "Answer True if the document entails the claim, and False if it "
        "contradicts it or the values do not match.\n"
        "Inspect the document in the REPL before answering. Finish with "
        "exactly one word: True or False."
    ),
}

LABELS = {
    "contractnli": ["Entailment", "Contradiction", "NotMentioned"],
    "findver": ["True", "False"],
    "coverbench": ["True", "False"],
}


# ----------------------------------------------------------------------
# Loaders -> [(uid, claim, document, gold, meta)]
# ----------------------------------------------------------------------
def load_contractnli(split):
    from run_rlm_contractnli import load_split, iter_instances
    data = load_split(split)
    out = []
    for t in iter_instances(data):
        doc_id, nda_key, hyp, text, gold = t[:5]
        out.append((f"{doc_id}:{nda_key}", hyp, text, gold,
                    {"doc_id": doc_id, "nda_key": nda_key}))
    return out


def load_findver(split):
    ex = json.load(open(os.path.join(FINDVER, "data", f"{split}.json"),
                        encoding="utf-8"))
    cache = {}
    out = []
    for e in ex:
        rep = e["report"]
        if rep not in cache:
            with open(os.path.join(FINDVER, "financial_reports", rep),
                      encoding="utf-8") as f:
                d = json.load(f)
            cache[rep] = "\n".join(f"[{c['id']}] {c['context']}"
                                   for c in d["context"])
        out.append((e["example_id"], e["statement"], cache[rep],
                    str(e["entailment_label"]),
                    {"report": rep, "subset": e.get("subset"),
                     "relevant_context": e.get("relevant_context")}))
    return out


def load_coverbench(split=None):
    out = []
    with open(COVERBENCH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append((r["id"], r["claim"], r["context"], str(r["label"]),
                        {"source_dataset": r.get("source_dataset"),
                         "domain": r.get("domain"),
                         "complexity_tags": r.get("complexity_tags")}))
    return out


LOADERS = {"contractnli": load_contractnli, "findver": load_findver,
           "coverbench": load_coverbench}


def stratified(items, n, seed):
    """Label-stratified subsample, deterministic for a given (n, seed).

    Sorting the per-label pool before shuffling makes the draw independent of
    the order the loader happened to yield, so the same subset comes back on
    any machine.
    """
    if n is None or n >= len(items):
        return list(items)
    from collections import defaultdict
    by = defaultdict(list)
    for it in items:
        by[it[3]].append(it)
    rng = random.Random(seed)
    picked = []
    for lab in sorted(by):
        pool = sorted(by[lab], key=lambda x: str(x[0]))
        rng.shuffle(pool)
        share = max(1, round(n * len(by[lab]) / len(items)))
        picked.extend(pool[:share])
    rng.shuffle(picked)
    return picked[:n]


def parse_label(raw, benchmark):
    """Last matching label token wins -- the model reasons before committing,
    so an early mention inside the reasoning is not the answer."""
    if not raw:
        return None
    s = str(raw)
    if "</think>" in s:
        s = s.split("</think>")[-1]
    if benchmark == "contractnli":
        hits = re.findall(r"contradiction|entailment|notmentioned|not mentioned",
                          s, re.IGNORECASE)
        if not hits:
            return None
        h = hits[-1].lower().replace(" ", "")
        return {"contradiction": "Contradiction", "entailment": "Entailment",
                "notmentioned": "NotMentioned"}[h]
    hits = re.findall(r"\b(true|false)\b", s, re.IGNORECASE)
    return hits[-1].capitalize() if hits else None


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add(r["uid"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(LOADERS))
    ap.add_argument("--split", default=None,
                    help="contractnli: dev/test; findver: testmini/test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL",
                                                      "qwen/qwen3-8b"))
    ap.add_argument("--backend", default="openrouter")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--thinking", action="store_true",
                    help="enable the provider reasoning flag (default: off, "
                         "matching the main matrix)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--max-errors", type=int, default=5)
    ap.add_argument("--max-budget", type=float, default=0.30)
    ap.add_argument("--max-timeout", type=float, default=900.0)
    ap.add_argument("--max-concurrent-subcalls", type=int, default=4)
    ap.add_argument("--environment", default="local")
    args = ap.parse_args()

    split = args.split or {"contractnli": "dev", "findver": "testmini",
                           "coverbench": None}[args.benchmark]
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and args.backend == "openrouter":
        raise SystemExit("OPENROUTER_API_KEY not set")
    T.API_KEY = key
    rlm_utf8_patch.apply()
    T.install_tracing()

    items = LOADERS[args.benchmark](split)
    sub = stratified(items, args.n, args.seed)

    tag = "think" if args.thinking else "nothink"
    out_dir = os.path.join(RESULTS_DIR, "rlm-multi", args.benchmark)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rlm_{tag}_n{args.n}_seed{args.seed}.jsonl")
    done = load_done(path)
    todo = [s for s in sub if s[0] not in done]

    from rlm import RLM
    # The fix: the library forwards sampling_args straight to the chat
    # completion, and merges `extra_body`, so both temperature and the
    # OpenRouter reasoning flag land on every call -- root and sub-call alike.
    sampling = {"temperature": args.temperature}
    if args.backend == "openrouter" or (args.base_url
                                        and "openrouter" in args.base_url):
        sampling["extra_body"] = {"reasoning": {"enabled": bool(args.thinking)},
                                  "usage": {"include": True}}
    bk = {"model_name": args.model, "api_key": key or "EMPTY",
          "sampling_args": sampling}
    if args.base_url:
        bk["base_url"] = args.base_url
    rlm = RLM(backend=args.backend, backend_kwargs=bk,
              environment=args.environment, max_depth=args.max_depth,
              max_iterations=args.max_iterations, max_errors=args.max_errors,
              max_budget=args.max_budget, max_timeout=args.max_timeout,
              max_concurrent_subcalls=args.max_concurrent_subcalls,
              verbose=False)

    print(f"=== RLM | {args.benchmark} | split={split} | n={len(sub)} "
          f"({len(todo)} to run, {len(done)} cached)")
    print(f"    model {args.model} | temperature {args.temperature} | "
          f"thinking {'ON' if args.thinking else 'OFF'} (both set explicitly)")
    doclens = [ntok(s[2]) for s in sub]
    print(f"    doc tokens: mean {statistics.mean(doclens):,.0f} | "
          f"median {statistics.median(doclens):,.0f} | "
          f"max {max(doclens):,} | over 20K advisory: "
          f"{sum(1 for t in doclens if t*4 > 20000)}/{len(doclens)}\n")

    with open(path, "a", encoding="utf-8") as fh:
        for i, (uid, claim, doc, gold, meta) in enumerate(todo, 1):
            doc_chars, doc_tok = len(doc), ntok(doc)
            # Per-instance ledger; sound only at concurrency 1, which this
            # single-threaded loop enforces.
            T._tl.ledger = []
            T._tl.seq = 0
            with T._global_lock:
                T._global_ledger["rows"] = []
            rec = {"uid": uid, "benchmark": args.benchmark, "split": split,
                   "gold": gold, "doc_chars": doc_chars, "doc_tiktoken": doc_tok,
                   "thinking": bool(args.thinking),
                   "temperature": args.temperature, "meta": meta, "error": None}
            t0 = time.perf_counter()
            try:
                r = rlm.completion(prompt=doc,
                                   root_prompt=PROMPTS[args.benchmark].format(
                                       claim=claim))
                wall = time.perf_counter() - t0
                calls = list(T._tl.ledger or [])
                with T._global_lock:
                    calls += (T._global_ledger["rows"] or [])
                    T._global_ledger["rows"] = None
                T._tl.ledger = None

                root = [c for c in calls if c["role"] == "root"]
                subs = [c for c in calls if c["role"] == "subcall"]
                S = lambda rows, f: sum((c.get(f) or 0) for c in rows)
                resp = getattr(r, "response", "")
                us = getattr(r, "usage_summary", None)
                pred = parse_label(resp, args.benchmark)
                rec.update({
                    "pred": pred, "parsed_ok": pred is not None,
                    "correct": (pred == gold) if pred else False,
                    "n_calls": len(calls), "n_root": len(root),
                    "n_subcalls": len(subs), "recursed": len(subs) > 0,
                    # tiktoken, one tokenizer across every benchmark
                    "root_prompt_tik": S(root, "prompt_tiktoken"),
                    "root_resp_tik": S(root, "response_tiktoken"),
                    "sub_prompt_tik": S(subs, "prompt_tiktoken"),
                    "sub_resp_tik": S(subs, "response_tiktoken"),
                    "max_root_prompt_tik": max([c["prompt_tiktoken"]
                                                for c in root], default=0),
                    "max_sub_prompt_tik": max([c["prompt_tiktoken"]
                                               for c in subs], default=0),
                    # share of the document each layer carried, in tokens
                    "root_docfrac_peak": round(
                        max([c["prompt_tiktoken"] for c in root], default=0)
                        / doc_tok, 4) if doc_tok else None,
                    "sub_docfrac_sum": round(S(subs, "prompt_tiktoken")
                                             / doc_tok, 4) if doc_tok else None,
                    # provider's own numbers, kept beside tiktoken
                    "prov_in": getattr(us, "total_input_tokens", None),
                    "prov_out": getattr(us, "total_output_tokens", None),
                    "prov_cost": getattr(us, "total_cost", None),
                    "any_think_block": any(c.get("has_think") for c in calls),
                    "wall_s": round(wall, 1),
                    "calls": calls,
                    "response_tail": str(resp)[-240:],
                })
                print(f"  {i}/{len(todo)} {uid[:28]:28} "
                      f"calls={len(calls):2} (r{len(root)}/s{len(subs)}) "
                      f"{'REC' if subs else '---'} "
                      f"pred={str(pred)[:13]:13} gold={gold[:13]:13} "
                      f"{'OK ' if rec['correct'] else 'x  '}"
                      f"{rec['wall_s']:6.0f}s")
            except Exception as exc:
                rec.update({"error": f"{type(exc).__name__}: {str(exc)[:250]}",
                            "wall_s": round(time.perf_counter() - t0, 1)})
                print(f"  {i}/{len(todo)} {uid[:28]:28} ERROR {rec['error'][:90]}")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print(f"\nSaved -> {path}")
    print(f"retries absorbed by the tracer: {T._retries['n']}")


if __name__ == "__main__":
    main()
