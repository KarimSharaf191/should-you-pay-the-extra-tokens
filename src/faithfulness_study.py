"""
faithfulness_study.py
=====================
Faithfulness sub-study for the ContractNLI leg, on the n=150 stratified sample.

Faithfulness asks whether a scaffold's stated evidence and reasoning actually
drive its prediction -- as opposed to being plausible decoration over a label
the model would have produced anyway. Accuracy cannot answer that; intervention
can. ContractNLI is well suited because every Entailment and Contradiction
annotation carries gold evidence span indices (89 of the 150 sampled
instances; NotMentioned has none by definition).

Four studies:

  1 ablation   Remove the gold evidence from the context and see whether the
               prediction moves. Includes a LENGTH-MATCHED RANDOM CONTROL --
               without it, any accuracy drop is confounded with simply having
               a shorter context. Also runs an oracle (evidence only) arm as
               the upper bound.

  2 retrieval  Does retrieval actually surface the gold evidence? Deterministic
               and offline for regular RAG (no API calls): recall@k, precision,
               and hit-rate across a sweep of k.

  3 stopping   Agentic only, and the study the agentic scaffold uniquely
               permits: at which search step does gold evidence first enter the
               context, and does the model commit to an ANSWER before it ever
               arrives? An answer given before any gold evidence was retrieved
               is confident with no evidential basis.

  4 citation   Do CoT rationales quote text that is actually in the contract,
               and do they overlap the gold evidence more than chance?

Outputs per-instance JSONL under results/faithfulness/ plus a summary report.

Usage
-----
    python faithfulness_study.py --studies 2            # offline, instant
    python faithfulness_study.py --studies 1,3,4 --concurrency 3
    python faithfulness_study.py --report-only
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
from run_rlm_contractnli import load_split, iter_instances, stratified_sample

OUT_DIR = os.path.join(RESULTS_DIR, "faithfulness")
LABELS = ["Entailment", "Contradiction", "NotMentioned"]


# ----------------------------------------------------------------------
# Sample + gold evidence
# ----------------------------------------------------------------------
def build_sample(split, n, seed):
    """Return [(doc_id, nda_key, hypothesis, gold, doc, gold_span_idx)]."""
    data = load_split(split)
    by_id = {d["id"]: d for d in data["documents"]}
    inst = stratified_sample(list(iter_instances(data)), n, seed)
    out = []
    for t in inst:
        doc_id, nda_key, hyp, _text, gold = t[:5]
        doc = by_id[doc_id]
        ann = doc["annotation_sets"][0]["annotations"][nda_key]
        out.append((doc_id, nda_key, hyp, gold, doc,
                    list(ann.get("spans") or [])))
    return out


def span_texts(doc):
    return [doc["text"][s:e] for s, e in doc["spans"]]


def context_without(doc, drop_idx):
    """Document text with the character ranges of `drop_idx` spans removed.

    Removal is done back-to-front so earlier offsets stay valid. Everything
    outside the listed spans -- headers, connective text -- is preserved, so
    the ablated document still reads as a contract rather than a fragment list.
    """
    text = doc["text"]
    ranges = sorted((doc["spans"][i] for i in drop_idx), key=lambda r: -r[0])
    for s, e in ranges:
        text = text[:s] + text[e:]
    return text


def context_only(doc, keep_idx):
    """Just the listed spans, in document order (the oracle arm)."""
    st = span_texts(doc)
    return "\n".join(st[i] for i in sorted(keep_idx))


def random_control_idx(doc, gold_idx, rng):
    """Same NUMBER of spans as gold, drawn from the non-evidence spans, and
    matched on total character length as closely as greedy selection allows.

    This is the control that makes study 1 interpretable: it removes an
    equivalent amount of contract text that is known NOT to be the evidence.
    """
    n_spans = len(doc["spans"])
    gold_chars = sum(doc["spans"][i][1] - doc["spans"][i][0] for i in gold_idx)
    pool = [i for i in range(n_spans) if i not in set(gold_idx)]
    rng.shuffle(pool)
    picked, chars = [], 0
    for i in pool:
        if len(picked) >= len(gold_idx) and chars >= gold_chars:
            break
        picked.append(i)
        chars += doc["spans"][i][1] - doc["spans"][i][0]
    return picked


# ----------------------------------------------------------------------
# Study 1: evidence ablation
# ----------------------------------------------------------------------
def study1_variants(doc, gold_idx, rng):
    return {
        "ablated": context_without(doc, gold_idx),
        "random_ctrl": context_without(doc, random_control_idx(doc, gold_idx, rng)),
        "oracle": context_only(doc, gold_idx),
    }


def run_study1(sample, args):
    path = os.path.join(OUT_DIR, f"study1_ablation_n{args.n}.jsonl")
    done = load_done_keys(path)
    todo = [s for s in sample if s[5] and (str(s[0]), s[1]) not in done]
    print(f"[study1] {len(todo)} instances to run ({len(done)} cached)")
    if not todo:
        return

    rng_master = random.Random(args.seed)
    lock = threading.Lock()

    def work(item):
        doc_id, nda_key, hyp, gold, doc, gold_idx = item
        # Per-instance RNG seeded from the instance key: the control arm is
        # then reproducible rather than depending on thread scheduling.
        rng = random.Random(f"{doc_id}:{nda_key}:{args.seed}")
        variants = study1_variants(doc, gold_idx, rng)
        rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
               "n_gold_spans": len(gold_idx), "n_doc_spans": len(doc["spans"]),
               "preds": {}, "usage": {}, "error": None}
        try:
            for name, ctx in variants.items():
                label, usage = cz.predict_raw(hyp, ctx, thinking_on=False)
                rec["preds"][name] = label
                rec["usage"][name] = usage
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    append_concurrent(path, todo, work, args.concurrency, lock)


# ----------------------------------------------------------------------
# Study 2: retrieval faithfulness (deterministic, no API)
# ----------------------------------------------------------------------
def retrieve_idx(hypothesis, doc, k):
    """Top-k span INDICES for a query -- the same scoring retrieve_top_k uses,
    but returning identities so they can be compared against gold."""
    st = span_texts(doc)
    if not st:
        return []
    emb = cz._span_embeddings(st, doc.get("id"))
    q = cz._get_embedder().encode([hypothesis], convert_to_numpy=True,
                                  normalize_embeddings=True)[0]
    scores = emb @ q
    return np.argsort(-scores)[:k].tolist()


def run_study2(sample, args):
    path = os.path.join(OUT_DIR, f"study2_retrieval_n{args.n}.jsonl")
    ks = [1, 3, 5, 8, 10, 20]
    rows = []
    ev = [s for s in sample if s[5]]
    print(f"[study2] {len(ev)} instances with gold evidence (offline)")
    for doc_id, nda_key, hyp, gold, doc, gold_idx in ev:
        gs = set(gold_idx)
        rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
               "n_gold": len(gs), "n_spans": len(doc["spans"]), "at_k": {}}
        top = retrieve_idx(hyp, doc, max(ks))
        for k in ks:
            got = set(top[:k])
            hit = gs & got
            rec["at_k"][str(k)] = {
                "recall": len(hit) / len(gs),
                "precision": len(hit) / k,
                "any_hit": bool(hit),
                "all_hit": hit == gs,
            }
        rows.append(rec)
    write_jsonl(path, rows)
    print(f"[study2] wrote {len(rows)} rows")


# ----------------------------------------------------------------------
# Study 3: agentic stopping faithfulness (instrumented re-run)
# ----------------------------------------------------------------------
def instrumented_agentic(hypothesis, doc, gold_idx, k=5, max_steps=4):
    """Agentic RAG with per-step logging.

    Prompts and control flow mirror cz.predict_agentic_rag exactly; the only
    additions are the trace and the gold-evidence bookkeeping. Keeping them in
    step matters -- if this diverged, study 3 would describe a scaffold the
    main results never ran.
    """
    st = span_texts(doc)
    gs = set(gold_idx)
    messages = [
        {"role": "system",
         "content": cz.AGENTIC_SYSTEM.format(max_steps=max_steps)},
        {"role": "user", "content": f"Hypothesis: {hypothesis}\n\nBegin."},
    ]
    trace = []
    seen = set()              # every span index shown so far
    first_gold_step = None
    label = "NotMentioned"
    answer_step = None

    for step in range(max_steps + 1):
        data = cz._post_chat(cz._chat_payload(messages, False), timeout=180)
        content = data["choices"][0]["message"]["content"] or ""
        body = content.split("</think>")[-1] if "</think>" in content else content

        m = re.search(r"ANSWER\s*:\s*(.+)", body, re.IGNORECASE)
        if m:
            label = cz.normalize_label(m.group(1))
            answer_step = step
            trace.append({"step": step, "action": "answer", "raw": body[:300]})
            break

        q = re.search(r"SEARCH\s*:\s*(.+)", body, re.IGNORECASE)
        query = q.group(1).strip() if q else hypothesis
        fellback = q is None
        idx = retrieve_idx(query, doc, k)
        seen |= set(idx)
        if first_gold_step is None and (gs & seen):
            first_gold_step = step
        trace.append({"step": step, "action": "search", "query": query[:200],
                      "fellback_to_hypothesis": fellback,
                      "retrieved": idx, "gold_in_retrieved": sorted(gs & set(idx)),
                      "cum_gold_seen": len(gs & seen)})
        excerpts = "\n".join(f"- {st[i]}" for i in sorted(idx)) or "(no matches)"
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user",
                         "content": f"Search results:\n{excerpts}\n\n"
                                    f"Issue another SEARCH or give your ANSWER."})
    else:
        messages.append({"role": "user",
                         "content": "You are out of searches. Reply with "
                                    "ANSWER: <Entailment|Contradiction|NotMentioned>"})
        data = cz._post_chat(cz._chat_payload(messages, False), timeout=180)
        content = data["choices"][0]["message"]["content"] or ""
        label = cz.normalize_label(content)
        answer_step = max_steps + 1
        trace.append({"step": answer_step, "action": "forced_answer",
                      "raw": content[:300]})

    return {
        "pred": label,
        "answer_step": answer_step,
        "first_gold_step": first_gold_step,
        "n_gold": len(gs),
        "gold_seen": len(gs & seen),
        "gold_recall_at_stop": (len(gs & seen) / len(gs)) if gs else None,
        # The headline: did it commit to an answer having never seen the
        # evidence the annotation says the label depends on?
        "answered_without_gold": bool(gs) and not (gs & seen),
        "n_unique_spans_seen": len(seen),
        "trace": trace,
    }


def run_study3(sample, args):
    path = os.path.join(OUT_DIR, f"study3_stopping_n{args.n}.jsonl")
    done = load_done_keys(path)
    todo = [s for s in sample if (str(s[0]), s[1]) not in done]
    print(f"[study3] {len(todo)} instances to run ({len(done)} cached)")
    if not todo:
        return
    lock = threading.Lock()

    def work(item):
        doc_id, nda_key, hyp, gold, doc, gold_idx = item
        rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
               "error": None}
        try:
            rec.update(instrumented_agentic(hyp, doc, gold_idx))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    append_concurrent(path, todo, work, args.concurrency, lock)


# ----------------------------------------------------------------------
# Study 4: citation faithfulness in CoT rationales
# ----------------------------------------------------------------------
_QUOTE_RE = re.compile(r'"([^"]{12,400})"')


# Contracts are full of typographic punctuation (curly quotes, en/em dashes,
# non-breaking spaces) that a model reproduces as ASCII when it quotes. Folding
# both sides to ASCII first is the difference between measuring fabrication and
# measuring an encoding mismatch -- before this, genuine verbatim quotes were
# being scored as unfounded.
_PUNCT_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2013: "-", 0x2014: "-", 0x2012: "-", 0x2015: "-", 0x2212: "-",
    0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x2009: " ",
    0x2026: "...", 0x00AD: "",
}


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").translate(_PUNCT_FOLD)).strip().lower()


def check_citations(rationale, doc, gold_idx, hypothesis=""):
    """Quoted-segment verification plus gold-evidence overlap.

    A quote counts as grounded if its normalised text appears in the normalised
    contract. Overlap with gold is measured on content words, which is coarse
    but does not depend on the model choosing to use quotation marks.
    """
    ntext = _norm(doc["text"])
    nhyp = _norm(hypothesis)
    raw_quotes = _QUOTE_RE.findall(rationale or "")

    # A rationale that quotes the HYPOTHESIS back is not making a claim about
    # the contract, so it cannot be a fabricated citation. Scoring those as
    # unfounded was the single biggest distortion in the first version of this
    # metric -- ContractNLI hypotheses read exactly like contract clauses.
    quotes, hyp_quotes = [], 0
    for q in raw_quotes:
        nq = _norm(q)
        if nq and nhyp and (nq in nhyp or nhyp in nq):
            hyp_quotes += 1
        else:
            quotes.append(q)

    def _grounded(q):
        nq = _norm(q)
        if nq in ntext:
            return True
        # A long quote that is verbatim for its first 12 content words is a real
        # citation that drifted or was elided, not an invention. Shorter quotes
        # must match outright.
        w = nq.split()
        return len(w) >= 12 and " ".join(w[:12]) in ntext

    grounded = [q for q in quotes if _grounded(q)]

    st = span_texts(doc)
    gold_words = set()
    for i in gold_idx:
        gold_words |= {w for w in re.findall(r"[a-z]{4,}", _norm(st[i]))}
    rat_words = {w for w in re.findall(r"[a-z]{4,}", _norm(rationale or ""))}
    # Chance baseline: overlap with a same-sized set of NON-evidence spans.
    other = [i for i in range(len(st)) if i not in set(gold_idx)]
    rng = random.Random(0)
    rng.shuffle(other)
    ctrl_words = set()
    for i in other[:max(1, len(gold_idx))]:
        ctrl_words |= {w for w in re.findall(r"[a-z]{4,}", _norm(st[i]))}

    return {
        "n_quotes_raw": len(raw_quotes),
        "n_quotes_of_hypothesis": hyp_quotes,
        "n_quotes": len(quotes),          # contract-directed quotes only
        "n_quotes_grounded": len(grounded),
        "quote_groundedness": (len(grounded) / len(quotes)) if quotes else None,
        "gold_word_overlap": (len(rat_words & gold_words) / len(gold_words))
                             if gold_words else None,
        "ctrl_word_overlap": (len(rat_words & ctrl_words) / len(ctrl_words))
                             if ctrl_words else None,
        "rationale_chars": len(rationale or ""),
    }


def run_study4(sample, args):
    path = os.path.join(OUT_DIR, f"study4_citation_n{args.n}.jsonl")
    done = load_done_keys(path)
    todo = [s for s in sample if (str(s[0]), s[1]) not in done]
    print(f"[study4] {len(todo)} instances to run ({len(done)} cached)")
    if not todo:
        return

    # Built once on the main thread: dspy.configure() has thread affinity.
    program = cz.build_dspy_program(condition="cot")
    lock = threading.Lock()

    def work(item):
        doc_id, nda_key, hyp, gold, doc, gold_idx = item
        rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
               "n_gold_spans": len(gold_idx), "error": None}
        try:
            out = program(contract=doc["text"], hypothesis=hyp)
            rationale = (getattr(out, "reasoning", None)
                         or getattr(out, "rationale", "") or "")
            rec["pred"] = cz.normalize_label(getattr(out, "label", ""))
            rec["rationale"] = rationale[:4000]
            rec.update(check_citations(rationale, doc, gold_idx, hyp))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    append_concurrent(path, todo, work, args.concurrency, lock)


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def load_done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("error"):
                    keys.add((str(r["doc_id"]), r["nda_key"]))
    return keys


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


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def baseline_preds(n, seed, split):
    """Full-context zeroshot predictions from the main matrix, for study 1."""
    p = os.path.join(RESULTS_DIR, "matrix-zeroshot", "qwen3-8b",
                     f"records_{split}_n{n}_seed{seed}.jsonl")
    out = {}
    for r in read_jsonl(p):
        if not r.get("error"):
            out[(str(r["doc_id"]), r["nda_key"])] = r["pred"]
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def report(args):
    L = ["=" * 92,
         f"Faithfulness sub-study - {args.split} n={args.n} seed={args.seed}",
         "=" * 92, ""]

    # ---- study 1 -------------------------------------------------------
    rows = [r for r in read_jsonl(
        os.path.join(OUT_DIR, f"study1_ablation_n{args.n}.jsonl"))
        if not r.get("error")]
    if rows:
        base = baseline_preds(args.n, args.seed, args.split)
        L += ["STUDY 1 - EVIDENCE ABLATION", "-" * 92,
              "  Does removing the gold evidence change the prediction?",
              "  random_ctrl removes an equal amount of NON-evidence text, so a",
              "  drop under 'ablated' beyond 'random_ctrl' is evidence-specific.",
              ""]
        acc = defaultdict(list)
        flips = defaultdict(list)
        n_base = 0
        for r in rows:
            key = (str(r["doc_id"]), r["nda_key"])
            b = base.get(key)
            if b is not None:
                n_base += 1
                acc["full(baseline)"].append(b == r["gold"])
            for name, pred in r["preds"].items():
                acc[name].append(pred == r["gold"])
                if b is not None:
                    flips[name].append(pred != b)
        L.append(f"  n={len(rows)} instances with gold evidence "
                 f"({n_base} matched to a full-context baseline)")
        L.append("")
        L.append(f"  {'variant':16} {'accuracy':>9} {'flip vs full':>14}")
        for name in ["full(baseline)", "ablated", "random_ctrl", "oracle"]:
            if name not in acc:
                continue
            fl = mean(flips[name]) if flips.get(name) else float("nan")
            L.append(f"  {name:16} {mean(acc[name]):9.3f} "
                     f"{fl:14.3f}")
        if acc.get("ablated") and acc.get("random_ctrl"):
            d = mean(acc["random_ctrl"]) - mean(acc["ablated"])
            L += ["",
                  f"  Evidence-specific effect = acc(random_ctrl) - acc(ablated)"
                  f" = {d:+.3f}",
                  "  Near zero means the model was not relying on the gold "
                  "evidence:",
                  "  removing it cost no more than removing arbitrary text of "
                  "the same size."]
        L.append("")

    # ---- study 2 -------------------------------------------------------
    rows = read_jsonl(os.path.join(OUT_DIR, f"study2_retrieval_n{args.n}.jsonl"))
    if rows:
        L += ["STUDY 2 - RETRIEVAL FAITHFULNESS (deterministic)", "-" * 92,
              "  Does embedding retrieval surface the annotated evidence?", ""]
        L.append(f"  n={len(rows)}  |  mean gold spans/instance "
                 f"{mean([r['n_gold'] for r in rows]):.1f}  "
                 f"of {mean([r['n_spans'] for r in rows]):.0f} spans/doc")
        L.append("")
        L.append(f"  {'k':>4} {'recall':>9} {'precision':>11} "
                 f"{'any hit':>9} {'all hits':>9}")
        ks = sorted({int(k) for r in rows for k in r["at_k"]})
        for k in ks:
            sk = str(k)
            L.append(f"  {k:>4} {mean([r['at_k'][sk]['recall'] for r in rows]):9.3f} "
                     f"{mean([r['at_k'][sk]['precision'] for r in rows]):11.3f} "
                     f"{mean([r['at_k'][sk]['any_hit'] for r in rows]):9.3f} "
                     f"{mean([r['at_k'][sk]['all_hit'] for r in rows]):9.3f}")
        L += ["", "  k=8 is what the regular RAG condition uses; k=5 is the "
              "agentic per-search k.", ""]

    # ---- study 3 -------------------------------------------------------
    rows = [r for r in read_jsonl(
        os.path.join(OUT_DIR, f"study3_stopping_n{args.n}.jsonl"))
        if not r.get("error")]
    if rows:
        ev = [r for r in rows if r.get("n_gold")]
        L += ["STUDY 3 - AGENTIC STOPPING FAITHFULNESS", "-" * 92,
              "  When does gold evidence arrive, and does the model answer "
              "before it does?", ""]
        L.append(f"  n={len(rows)} trajectories ({len(ev)} with gold evidence)")
        L.append(f"  mean answer step            : "
                 f"{mean([r['answer_step'] for r in rows]):.2f}")
        L.append(f"  mean unique spans seen      : "
                 f"{mean([r['n_unique_spans_seen'] for r in rows]):.1f}")
        if ev:
            never = [r for r in ev if r["answered_without_gold"]]
            L.append(f"  gold recall at stop         : "
                     f"{mean([r['gold_recall_at_stop'] for r in ev]):.3f}")
            L.append(f"  ANSWERED WITHOUT ANY GOLD   : "
                     f"{len(never)}/{len(ev)} = {len(never)/len(ev):.1%}")
            correct_never = sum(1 for r in never if r["pred"] == r["gold"])
            if never:
                L.append(f"    ...of those, {correct_never}/{len(never)} were "
                         f"still scored CORRECT "
                         f"({correct_never/len(never):.1%}) - right answer, "
                         f"no evidence")
            got = [r for r in ev if r["first_gold_step"] is not None]
            if got:
                L.append(f"  mean step gold first appears: "
                         f"{mean([r['first_gold_step'] for r in got]):.2f}")
        fell = sum(1 for r in rows for t in r.get("trace", [])
                   if t.get("fellback_to_hypothesis"))
        L.append(f"  turns where the model emitted neither SEARCH nor ANSWER "
                 f"(fell back to hypothesis as query): {fell}")
        L.append("")

    # ---- study 4 -------------------------------------------------------
    rows = [r for r in read_jsonl(
        os.path.join(OUT_DIR, f"study4_citation_n{args.n}.jsonl"))
        if not r.get("error")]
    if rows:
        L += ["STUDY 4 - CITATION FAITHFULNESS IN CoT RATIONALES", "-" * 92,
              "  Are quoted passages real, and does the rationale track the "
              "gold evidence?", ""]
        q = [r for r in rows if r.get("n_quotes")]
        L.append(f"  n={len(rows)}  |  rationales containing quotes: {len(q)}")
        if q:
            L.append(f"  quote groundedness (quoted text found verbatim in "
                     f"contract): {mean([r['quote_groundedness'] for r in q]):.3f}")
        ev = [r for r in rows if r.get("n_gold_spans")]
        if ev:
            g = mean([r["gold_word_overlap"] for r in ev])
            c = mean([r["ctrl_word_overlap"] for r in ev])
            L.append(f"  content-word overlap with GOLD evidence   : {g:.3f}")
            L.append(f"  content-word overlap with control spans   : {c:.3f}")
            L.append(f"  lift over chance                          : {g - c:+.3f}")
            L.append("  A lift near zero means the rationale is no more "
                     "anchored to the")
            L.append("  operative clause than to arbitrary contract text.")
        L.append("")

    if len(L) <= 4:
        L.append("No study outputs found yet. Run with --studies first.")

    text = "\n".join(L)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(OUT_DIR, f"faithfulness_report_n{args.n}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--studies", default="2",
                    help="comma list from 1,2,3,4")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cz.MODEL = args.model
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.report_only:
        sample = build_sample(args.split, args.n, args.seed)
        want = {s.strip() for s in args.studies.split(",") if s.strip()}
        if "1" in want:
            run_study1(sample, args)
        if "2" in want:
            run_study2(sample, args)
        if "3" in want:
            run_study3(sample, args)
        if "4" in want:
            run_study4(sample, args)

    report(args)


if __name__ == "__main__":
    main()
