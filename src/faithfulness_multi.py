"""
faithfulness_multi.py
=====================
The faithfulness sub-study on FinDVer and CoverBench, mirroring the
ContractNLI study in faithfulness_study.py.

What each benchmark can actually support differs, and the difference is not
cosmetic -- it decides which studies are meaningful:

  ContractNLI  gold evidence on 59% of instances (Entailment + Contradiction),
               annotated spans, documents fit the context window -> all four
               studies run. Already done at n=150.

  FinDVer      gold evidence on 100% of instances (`relevant_context`, mean 2.8
               paragraphs), documents pre-segmented into numbered paragraphs.
               BUT 52% of documents exceed Qwen3-8B's 40,960-token window, so
               the ContractNLI form of study 1 -- ablate the full document -- is
               impossible on half the data, and the half that fits is short
               10-Qs rather than a random subset.

               Study 1 is therefore run in ORACLE form: gold evidence alone vs
               gold evidence with the gold paragraphs removed, against a
               length-matched control drawn from non-evidence paragraphs. Both
               arms fit any window, and it tests the same causal claim -- does
               removing the evidence change the answer -- without the truncation
               confound. Study 4 likewise uses retrieved context rather than the
               full filing.

  CoverBench   gold evidence on only 121/733 instances (Feverous alone; the
               other eight sources carry none), and its context is a rendered
               blob with no segmentation. Every gold-dependent metric is
               therefore unavailable: study 1, study 2, study 3's
               "answered without gold", and study 4's overlap-lift.
               What remains and is reported: study 3's stopping DYNAMICS and
               study 4's quote groundedness, both of which need no annotation.
               Passages are sentence-chunked so retrieval has units to work on.

Reporting the unavailable metrics as if they were comparable would be the easy
mistake here; this module omits them per benchmark and says so in the report.

Usage
-----
    python faithfulness_multi.py --benchmark findver --studies 1,2,3,4 --n 150
    python faithfulness_multi.py --benchmark coverbench --studies 3,4 --n 150
    python faithfulness_multi.py --benchmark findver --report-only
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("CONTRACTNLI_RESULTS_DIR", os.path.join(_ROOT, "results"))
import re
import json
import random
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import contractnli_zeroshot as cz
from faithfulness_multi_support import (
    load_benchmark, passages_of, gold_of, PROMPTS, LABELSETS,
    normalize, append_concurrent, read_jsonl, load_done_keys, write_jsonl,
    mean, ntok,
)

OUT_DIR = os.path.join(RESULTS_DIR, "faithfulness-multi")


# ----------------------------------------------------------------------
# Context builders. All operate on the passage list so they work identically
# whether passages come from annotated spans, numbered paragraphs, or
# sentence chunking.
# ----------------------------------------------------------------------
def ctx_join(passages, idx):
    return "\n".join(passages[i] for i in sorted(idx))


def random_control_idx(passages, gold_idx, rng):
    """Same count as gold, matched on characters, drawn from non-evidence
    passages. Without this a drop under ablation is confounded with simply
    having less text."""
    gold_chars = sum(len(passages[i]) for i in gold_idx)
    pool = [i for i in range(len(passages)) if i not in set(gold_idx)]
    rng.shuffle(pool)
    picked, chars = [], 0
    for i in pool:
        if len(picked) >= len(gold_idx) and chars >= gold_chars:
            break
        picked.append(i)
        chars += len(passages[i])
    return picked


def study1_variants(passages, gold_idx, rng, pad=6):
    """Oracle-form ablation.

    `base` is the gold evidence plus a few neighbouring passages, which gives
    the ablated arm something to be wrong about rather than an empty prompt.
    `ablated` removes the gold passages from that base; `random_ctrl` removes an
    equivalent amount of non-evidence text from it instead.
    """
    n = len(passages)
    near = set(gold_idx)
    for i in gold_idx:
        for j in range(max(0, i - 2), min(n, i + 3)):
            near.add(j)
    extra = [i for i in range(n) if i not in near]
    rng.shuffle(extra)
    base_idx = sorted(near | set(extra[:pad]))

    gold = set(gold_idx)
    ablated_idx = [i for i in base_idx if i not in gold]
    ctrl_pool = [i for i in base_idx if i not in gold]
    rng.shuffle(ctrl_pool)
    drop, chars, want = set(), 0, sum(len(passages[i]) for i in gold_idx)
    for i in ctrl_pool:
        if len(drop) >= len(gold_idx) and chars >= want:
            break
        drop.add(i)
        chars += len(passages[i])
    ctrl_idx = [i for i in base_idx if i not in drop]

    return {
        "base": ctx_join(passages, base_idx),
        "ablated": ctx_join(passages, ablated_idx),
        "random_ctrl": ctx_join(passages, ctrl_idx),
        "oracle": ctx_join(passages, gold_idx),
    }


# ----------------------------------------------------------------------
# Study 1
# ----------------------------------------------------------------------
def run_study1(bench, items, args):
    path = os.path.join(OUT_DIR, bench, f"study1_ablation_n{args.n}.jsonl")
    done = load_done_keys(path)
    todo = [it for it in items if gold_of(bench, it) and it["uid"] not in done]
    print(f"[study1] {len(todo)} to run ({len(done)} cached)")
    if not todo:
        return
    lock = threading.Lock()

    def work(it):
        ps = passages_of(bench, it)
        gi = gold_of(bench, it)
        rng = random.Random(f"{it['uid']}:{args.seed}")
        rec = {"uid": it["uid"], "gold": it["gold"], "n_gold": len(gi),
               "n_passages": len(ps), "preds": {}, "ctx_tok": {}, "error": None}
        try:
            for name, ctx in study1_variants(ps, gi, rng).items():
                label, usage = cz.predict_raw(it["claim"], ctx, thinking_on=False)
                rec["preds"][name] = normalize(label, bench)
                rec["ctx_tok"][name] = ntok(ctx)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    append_concurrent(path, todo, work, args.concurrency, lock)


# ----------------------------------------------------------------------
# Study 2 -- deterministic, no API
# ----------------------------------------------------------------------
def retrieve_idx(query, passages, k, key):
    emb = cz._span_embeddings(passages, key)
    q = cz._get_embedder().encode([query], convert_to_numpy=True,
                                  normalize_embeddings=True)[0]
    return np.argsort(-(emb @ q))[:k].tolist()


def run_study2(bench, items, args):
    path = os.path.join(OUT_DIR, bench, f"study2_retrieval_n{args.n}.jsonl")
    ks = [1, 3, 5, 8, 10, 20]
    rows = []
    ev = [it for it in items if gold_of(bench, it)]
    print(f"[study2] {len(ev)} with gold evidence (offline)")
    for it in ev:
        ps = passages_of(bench, it)
        gs = set(gold_of(bench, it))
        top = retrieve_idx(it["claim"], ps, max(ks), it.get("doc_key"))
        rec = {"uid": it["uid"], "gold": it["gold"], "n_gold": len(gs),
               "n_passages": len(ps), "at_k": {}}
        for k in ks:
            hit = gs & set(top[:k])
            rec["at_k"][str(k)] = {"recall": len(hit) / len(gs),
                                   "precision": len(hit) / k,
                                   "any_hit": bool(hit), "all_hit": hit == gs}
        rows.append(rec)
    write_jsonl(path, rows)
    print(f"[study2] wrote {len(rows)}")


# ----------------------------------------------------------------------
# Study 3 -- agentic stopping
# ----------------------------------------------------------------------
def run_study3(bench, items, args):
    path = os.path.join(OUT_DIR, bench, f"study3_stopping_n{args.n}.jsonl")
    done = load_done_keys(path)
    todo = [it for it in items if it["uid"] not in done]
    print(f"[study3] {len(todo)} to run ({len(done)} cached)")
    if not todo:
        return
    lock = threading.Lock()
    labels = " | ".join(LABELSETS[bench])

    def work(it):
        ps = passages_of(bench, it)
        gs = set(gold_of(bench, it))
        rec = {"uid": it["uid"], "gold": it["gold"], "n_gold": len(gs),
               "error": None}
        try:
            sysmsg = PROMPTS[bench]["agentic"].format(max_steps=4)
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user",
                     "content": f"Claim: {it['claim']}\n\nBegin."}]
            seen, trace = set(), []
            first_gold, answer_step, label = None, None, LABELSETS[bench][-1]
            for step in range(5):
                data = cz._post_chat(cz._chat_payload(msgs, False), timeout=180)
                content = data["choices"][0]["message"]["content"] or ""
                body = content.split("</think>")[-1] if "</think>" in content else content
                m = re.search(r"ANSWER\s*:\s*(.+)", body, re.IGNORECASE)
                if m:
                    label = normalize(m.group(1), bench)
                    answer_step = step
                    trace.append({"step": step, "action": "answer"})
                    break
                q = re.search(r"SEARCH\s*:\s*(.+)", body, re.IGNORECASE)
                query = q.group(1).strip() if q else it["claim"]
                idx = retrieve_idx(query, ps, 5, it.get("doc_key"))
                seen |= set(idx)
                if first_gold is None and (gs & seen):
                    first_gold = step
                trace.append({"step": step, "action": "search",
                              "fellback": q is None, "retrieved": idx,
                              "cum_gold_seen": len(gs & seen) if gs else None})
                exc = "\n".join(f"- {ps[i]}" for i in sorted(idx)) or "(no matches)"
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content":
                             f"Search results:\n{exc}\n\nIssue another SEARCH "
                             f"or give your ANSWER."})
            else:
                msgs.append({"role": "user", "content":
                             f"You are out of searches. Reply with ANSWER: "
                             f"<{labels}>"})
                data = cz._post_chat(cz._chat_payload(msgs, False), timeout=180)
                label = normalize(
                    data["choices"][0]["message"]["content"] or "", bench)
                answer_step = 5
                trace.append({"step": 5, "action": "forced_answer"})

            slots = sum(len(t["retrieved"]) for t in trace if t.get("retrieved"))
            rec.update({
                "pred": label, "correct": label == it["gold"],
                "answer_step": answer_step,
                "first_gold_step": first_gold,
                "n_unique_seen": len(seen), "slots": slots,
                "dup_frac": round(1 - len(seen) / slots, 4) if slots else None,
                # gold-dependent, therefore None where the benchmark has none
                "gold_recall_at_stop": (len(gs & seen) / len(gs)) if gs else None,
                "answered_without_gold": (not (gs & seen)) if gs else None,
                "fallbacks": sum(1 for t in trace if t.get("fellback")),
                "trace": trace,
            })
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    append_concurrent(path, todo, work, args.concurrency, lock)


# ----------------------------------------------------------------------
# Study 4 -- citation faithfulness
# ----------------------------------------------------------------------
_QUOTE = re.compile(r'"([^"]{12,400})"')
_FOLD = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-",
         0x2014: "-", 0x2212: "-", 0x00A0: " ", 0x2026: "...", 0x00AD: ""}


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").translate(_FOLD)).strip().lower()


def check_citations(rationale, passages, gold_idx, claim):
    """Same metric as the ContractNLI study, including the two corrections it
    needed: quotes of the CLAIM are separated out rather than scored as
    fabricated contract citations, and punctuation is folded so a typographic
    apostrophe does not turn a genuine quote into a miss."""
    text = _norm("\n".join(passages))
    nclaim = _norm(claim)
    raw = _QUOTE.findall(rationale or "")
    quotes, claim_quotes = [], 0
    for q in raw:
        nq = _norm(q)
        if nq and nclaim and (nq in nclaim or nclaim in nq):
            claim_quotes += 1
        else:
            quotes.append(q)

    def grounded(q):
        nq = _norm(q)
        if nq in text:
            return True
        w = nq.split()
        return len(w) >= 12 and " ".join(w[:12]) in text

    ok = [q for q in quotes if grounded(q)]
    out = {"n_quotes_raw": len(raw), "n_quotes_of_claim": claim_quotes,
           "n_quotes": len(quotes), "n_grounded": len(ok),
           "quote_groundedness": (len(ok) / len(quotes)) if quotes else None,
           "rationale_chars": len(rationale or "")}

    if gold_idx:
        words = lambda s: set(re.findall(r"[a-z]{4,}", _norm(s)))
        gw = set().union(*[words(passages[i]) for i in gold_idx])
        rw = words(rationale or "")
        other = [i for i in range(len(passages)) if i not in set(gold_idx)]
        rng = random.Random(0)
        rng.shuffle(other)
        cw = set().union(*[words(passages[i])
                           for i in other[:max(1, len(gold_idx))]]) or {"_"}
        out["gold_word_overlap"] = len(rw & gw) / len(gw) if gw else None
        out["ctrl_word_overlap"] = len(rw & cw) / len(cw) if cw else None
    else:
        out["gold_word_overlap"] = None
        out["ctrl_word_overlap"] = None
    return out


def run_study4(bench, items, args):
    path = os.path.join(OUT_DIR, bench, f"study4_citation_n{args.n}.jsonl")
    done = load_done_keys(path)
    todo = [it for it in items if it["uid"] not in done]
    print(f"[study4] {len(todo)} to run ({len(done)} cached)")
    if not todo:
        return
    lock = threading.Lock()

    def work(it):
        ps = passages_of(bench, it)
        gi = gold_of(bench, it)
        rec = {"uid": it["uid"], "gold": it["gold"], "n_gold": len(gi),
               "error": None}
        try:
            # Retrieved context rather than the full document: on FinDVer the
            # full filing exceeds the model's window on half the instances, and
            # a metric that silently changes its context source between
            # instances is not comparable.
            idx = retrieve_idx(it["claim"], ps, min(12, len(ps)),
                               it.get("doc_key"))
            ctx = ctx_join(ps, idx)
            prompt = PROMPTS[bench]["cot"].format(claim=it["claim"], context=ctx)
            data = cz._post_chat(cz._chat_payload(
                [{"role": "user", "content": prompt}], False), timeout=180)
            raw = data["choices"][0]["message"]["content"] or ""
            rec["pred"] = normalize(raw, bench)
            rec["correct"] = rec["pred"] == it["gold"]
            rec["rationale"] = raw[:4000]
            rec["retrieved_k"] = len(idx)
            rec["gold_in_retrieved"] = len(set(gi) & set(idx)) if gi else None
            rec.update(check_citations(raw, ps, gi, it["claim"]))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    append_concurrent(path, todo, work, args.concurrency, lock)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def report(bench, args):
    d = os.path.join(OUT_DIR, bench)
    L = ["=" * 92, f"Faithfulness sub-study - {bench} - n={args.n} seed={args.seed}",
         "=" * 92, ""]

    rows = [r for r in read_jsonl(os.path.join(d, f"study1_ablation_n{args.n}.jsonl"))
            if not r.get("error")]
    if rows:
        acc = defaultdict(list)
        for r in rows:
            for k, p in r["preds"].items():
                acc[k].append(p == r["gold"])
        L += ["STUDY 1 - EVIDENCE ABLATION (oracle form)", "-" * 92,
              "  base = gold evidence + neighbouring passages.",
              "  ablated = base minus the gold passages.",
              "  random_ctrl = base minus an equal amount of NON-evidence text.",
              f"  n={len(rows)}", ""]
        for k in ["base", "random_ctrl", "ablated", "oracle"]:
            if k in acc:
                L.append(f"  {k:14} accuracy {mean(acc[k]):.3f}")
        if acc.get("ablated") and acc.get("random_ctrl"):
            L += ["", f"  Evidence-specific effect = "
                  f"{mean(acc['random_ctrl']) - mean(acc['ablated']):+.3f}"]
        L.append("")

    rows = read_jsonl(os.path.join(d, f"study2_retrieval_n{args.n}.jsonl"))
    if rows:
        L += ["STUDY 2 - RETRIEVAL FAITHFULNESS", "-" * 92,
              f"  n={len(rows)} | mean gold {mean([r['n_gold'] for r in rows]):.1f} "
              f"of {mean([r['n_passages'] for r in rows]):.0f} passages", "",
              f"  {'k':>4} {'recall':>9} {'precision':>11} {'any hit':>9} {'all hits':>9}"]
        for k in sorted({int(k) for r in rows for k in r["at_k"]}):
            s = str(k)
            L.append(f"  {k:>4} {mean([r['at_k'][s]['recall'] for r in rows]):9.3f} "
                     f"{mean([r['at_k'][s]['precision'] for r in rows]):11.3f} "
                     f"{mean([r['at_k'][s]['any_hit'] for r in rows]):9.3f} "
                     f"{mean([r['at_k'][s]['all_hit'] for r in rows]):9.3f}")
        L.append("")

    rows = [r for r in read_jsonl(os.path.join(d, f"study3_stopping_n{args.n}.jsonl"))
            if not r.get("error")]
    if rows:
        ev = [r for r in rows if r.get("n_gold")]
        L += ["STUDY 3 - AGENTIC STOPPING", "-" * 92,
              f"  n={len(rows)} trajectories ({len(ev)} with gold evidence)",
              f"  mean answer step        : {mean([r['answer_step'] for r in rows]):.2f}",
              f"  mean unique passages    : {mean([r['n_unique_seen'] for r in rows]):.1f}",
              f"  duplicate slot fraction : {mean([r['dup_frac'] for r in rows]):.3f}",
              f"  accuracy                : {mean([r['correct'] for r in rows]):.3f}"]
        if ev:
            nv = [r for r in ev if r["answered_without_gold"]]
            L += [f"  gold recall at stop     : "
                  f"{mean([r['gold_recall_at_stop'] for r in ev]):.3f}",
                  f"  ANSWERED WITHOUT GOLD   : {len(nv)}/{len(ev)} = "
                  f"{len(nv)/len(ev):.1%}"]
            if nv:
                c = sum(1 for r in nv if r["correct"])
                L.append(f"    ...still correct      : {c}/{len(nv)}")
        else:
            L.append("  gold-dependent metrics  : UNAVAILABLE "
                     "(no evidence annotation for this benchmark)")
        L.append("")

    rows = [r for r in read_jsonl(os.path.join(d, f"study4_citation_n{args.n}.jsonl"))
            if not r.get("error")]
    if rows:
        q = [r for r in rows if r.get("n_quotes")]
        L += ["STUDY 4 - CITATION FAITHFULNESS", "-" * 92,
              f"  n={len(rows)} | with contract-directed quotes: {len(q)}"]
        if q:
            L.append(f"  quote groundedness      : "
                     f"{mean([r['quote_groundedness'] for r in q]):.3f}")
        L.append(f"  quotes/rationale        : raw "
                 f"{mean([r['n_quotes_raw'] for r in rows]):.2f} "
                 f"(of the claim {mean([r['n_quotes_of_claim'] for r in rows]):.2f})")
        ev = [r for r in rows if r.get("gold_word_overlap") is not None]
        if ev:
            g = mean([r["gold_word_overlap"] for r in ev])
            c = mean([r["ctrl_word_overlap"] for r in ev])
            L += [f"  overlap with gold       : {g:.3f}",
                  f"  overlap with control    : {c:.3f}",
                  f"  lift over chance        : {g - c:+.3f}"]
        else:
            L.append("  overlap metrics         : UNAVAILABLE (no gold evidence)")
        L.append("")

    text = "\n".join(L)
    print(text)
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, f"report_n{args.n}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=["findver", "coverbench"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--studies", default="2")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cz.MODEL = args.model
    os.makedirs(os.path.join(OUT_DIR, args.benchmark), exist_ok=True)

    if not args.report_only:
        items = load_benchmark(args.benchmark, args.split, args.n, args.seed)
        ng = sum(1 for it in items if gold_of(args.benchmark, it))
        print(f"=== {args.benchmark} | n={len(items)} | "
              f"{ng} with gold evidence ===")
        want = {s.strip() for s in args.studies.split(",") if s.strip()}
        if "1" in want:
            run_study1(args.benchmark, items, args)
        if "2" in want:
            run_study2(args.benchmark, items, args)
        if "3" in want:
            run_study3(args.benchmark, items, args)
        if "4" in want:
            run_study4(args.benchmark, items, args)

    report(args.benchmark, args)


if __name__ == "__main__":
    main()
