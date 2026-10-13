"""
rag_multi.py
============
RAG and agentic RAG on FinDVer and CoverBench, with the same settings as the
ContractNLI conditions in run_matrix.py:

  rag          top-8 passages for the claim (all-MiniLM-L6-v2), one call
  agentic_rag  the model issues SEARCH queries (top-5 passages each, at most
               4 searches) and then ANSWERs

Passages are FinDVer's numbered report paragraphs and CoverBench sentence
chunks (faithfulness_multi_support.load_benchmark). Prompts are the benchmark's
own True/False prompts from faithfulness_multi_support.PROMPTS.

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...
    python rag_multi.py --benchmark findver --model meta-llama/llama-3.2-3b-instruct
    python rag_multi.py --benchmark coverbench --model meta-llama/llama-3.2-3b-instruct
    python rag_multi.py --benchmark findver --model ... --report-only

Resumable: finished claims are skipped.
"""

import os
import re
import json
import argparse
import threading
import statistics

from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_score, recall_score, accuracy_score)

import contractnli_zeroshot as cz
from faithfulness_multi import retrieve_idx
from faithfulness_multi_support import (
    load_benchmark, passages_of, PROMPTS, LABELSETS, normalize,
    append_concurrent, read_jsonl, load_done_keys,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("CONTRACTNLI_RESULTS_DIR",
                             os.path.join(_ROOT, "results"))
OUT_DIR = os.path.join(RESULTS_DIR, "rag-multi")
CONDITIONS = ["rag", "agentic_rag"]


def _usage(data, acc):
    u = data.get("usage") or {}
    acc["prompt"] += u.get("prompt_tokens") or 0
    acc["completion"] += u.get("completion_tokens") or 0
    return acc


def _answer_of(content, bench):
    body = (content or "").split("</think>")[-1]
    parsed = bool(re.search(r"\b(true|false)\b", body, re.IGNORECASE))
    return normalize(body, bench), parsed


def run_rag(it, bench, thinking):
    ps = passages_of(bench, it)
    idx = retrieve_idx(it["claim"], ps, 8, it.get("doc_key"))
    ctx = "\n".join(ps[i] for i in sorted(idx))
    prompt = PROMPTS[bench]["direct"].format(context=ctx, claim=it["claim"])
    data = cz._post_chat(cz._chat_payload(
        [{"role": "user", "content": prompt}], thinking), timeout=180)
    content = data["choices"][0]["message"]["content"] or ""
    pred, parsed = _answer_of(content, bench)
    return pred, parsed, _usage(data, {"prompt": 0, "completion": 0}), {
        "retrieved": len(idx)}


def run_agentic(it, bench, thinking, k=5, max_steps=4):
    """Same loop as the agentic stopping probe (faithfulness_multi study 3)."""
    ps = passages_of(bench, it)
    labels = " | ".join(LABELSETS[bench])
    msgs = [{"role": "system",
             "content": PROMPTS[bench]["agentic"].format(max_steps=max_steps)},
            {"role": "user", "content": f"Claim: {it['claim']}\n\nBegin."}]
    usage = {"prompt": 0, "completion": 0}
    seen, steps, parsed = set(), 0, False
    pred = LABELSETS[bench][-1]
    for step in range(max_steps + 1):
        data = cz._post_chat(cz._chat_payload(msgs, thinking), timeout=180)
        _usage(data, usage)
        content = data["choices"][0]["message"]["content"] or ""
        body = content.split("</think>")[-1]
        m = re.search(r"ANSWER\s*:\s*(.+)", body, re.IGNORECASE)
        if m:
            pred, parsed = _answer_of(m.group(1), bench)
            steps = step
            break
        q = re.search(r"SEARCH\s*:\s*(.+)", body, re.IGNORECASE)
        query = q.group(1).strip() if q else it["claim"]
        idx = retrieve_idx(query, ps, k, it.get("doc_key"))
        seen |= set(idx)
        exc = "\n".join(f"- {ps[i]}" for i in sorted(idx)) or "(no matches)"
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content":
                     f"Search results:\n{exc}\n\nIssue another SEARCH "
                     f"or give your ANSWER."})
    else:
        msgs.append({"role": "user", "content":
                     f"You are out of searches. Reply with ANSWER: <{labels}>"})
        data = cz._post_chat(cz._chat_payload(msgs, thinking), timeout=180)
        _usage(data, usage)
        pred, parsed = _answer_of(data["choices"][0]["message"]["content"], bench)
        steps = max_steps + 1
    return pred, parsed, usage, {"steps": steps, "unique_seen": len(seen)}


def out_path(bench, cond, model, n, thinking):
    slug = model.split("/")[-1].replace(":", "-")
    tag = "think" if thinking else "nothink"
    return os.path.join(OUT_DIR, bench, slug, f"{cond}_{tag}_n{n}.jsonl")


def report(bench, model, n, thinking):
    print(f"\n=== {bench} | {model} | thinking {'on' if thinking else 'off'}")
    for cond in CONDITIONS:
        rows = [r for r in read_jsonl(out_path(bench, cond, model, n, thinking))
                if not r.get("error")]
        if not rows:
            continue
        g = [r["gold"] for r in rows]
        p = [r["pred"] for r in rows]
        pos = "True"
        gen = statistics.mean(r["usage"]["completion"] for r in rows)
        tin = statistics.mean(r["usage"]["prompt"] for r in rows)
        print(f"  {cond:12} n={len(rows):4} acc={accuracy_score(g, p):.4f} "
              f"BA={balanced_accuracy_score(g, p):.4f} "
              f"P={precision_score(g, p, pos_label=pos, zero_division=0):.4f} "
              f"R={recall_score(g, p, pos_label=pos, zero_division=0):.4f} "
              f"F1={f1_score(g, p, pos_label=pos, zero_division=0):.4f} "
              f"macroF1={f1_score(g, p, average='macro', zero_division=0):.4f} "
              f"gen={gen:.1f} in={tin:.0f} "
              f"unparsed={sum(1 for r in rows if not r['parsed'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["findver", "coverbench"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--n", type=int, default=None, help="default: full set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cz.MODEL = args.model
    items = load_benchmark(args.benchmark, None, args.n, args.seed)
    n = args.n or len(items)
    if not args.report_only:
        if not cz.API_KEY and "openrouter.ai" in cz.BASE_URL:
            raise SystemExit("OPENROUTER_API_KEY not set")
        fn = {"rag": run_rag, "agentic_rag": run_agentic}
        for cond in args.conditions:
            path = out_path(args.benchmark, cond, args.model, n, args.thinking)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            done = load_done_keys(path)
            todo = [it for it in items if it["uid"] not in done]
            print(f"[{cond}] {len(todo)} to run ({len(done)} cached)")

            def work(it, cond=cond):
                rec = {"uid": it["uid"], "gold": it["gold"], "error": None}
                try:
                    pred, parsed, usage, extra = fn[cond](
                        it, args.benchmark, args.thinking)
                    rec.update({"pred": pred, "parsed": parsed,
                                "correct": pred == it["gold"], "usage": usage,
                                **extra})
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                return rec

            append_concurrent(path, todo, work, args.concurrency, threading.Lock())
    report(args.benchmark, args.model, n, args.thinking)


if __name__ == "__main__":
    main()
