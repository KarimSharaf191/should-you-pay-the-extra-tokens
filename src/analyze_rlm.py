"""
analyze_rlm.py
==============
Turns the JSONL records from run_rlm_contractnli.py into the numbers you
actually want on a slide: does the recursion engage, does engaging help, and
what does it cost.

Deliberately stdlib-only (no sklearn/scipy) so it can run alongside a live
eval without tripping Windows paging limits.

    python analyze_rlm.py --n 150
"""

import os
import json
import argparse
import statistics
from collections import Counter, defaultdict

# Repo layout: src/ (this file), data/contractnli/, results/.
# CONTRACTNLI_DATA_DIR / CONTRACTNLI_RESULTS_DIR override both. When only
# CONTRACTNLI_DATA_DIR is set (flat layout, e.g. the Kaggle notebook), results
# go next to the data, as before.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("CONTRACTNLI_DATA_DIR",
                          os.path.join(_ROOT, "data", "contractnli"))
RESULTS_DIR = os.environ.get(
    "CONTRACTNLI_RESULTS_DIR",
    os.path.join(DATA_DIR, 'results') if "CONTRACTNLI_DATA_DIR" in os.environ
    else os.path.join(_ROOT, 'results'))
LABELS = ["Entailment", "Contradiction", "NotMentioned"]


def load(condition, split, n, seed, model_slug):
    path = os.path.join(RESULTS_DIR, f"rlm-study-{condition}",
                        model_slug, f"records_{split}_n{n}_seed{seed}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def bal_acc(pairs):
    """Balanced accuracy = mean per-class recall. Stdlib reimplementation."""
    per = defaultdict(lambda: [0, 0])
    for gold, pred in pairs:
        per[gold][1] += 1
        if gold == pred:
            per[gold][0] += 1
    recalls = [hit / tot for hit, tot in per.values() if tot]
    return sum(recalls) / len(recalls) if recalls else float("nan")


def contract_lengths(split):
    with open(os.path.join(DATA_DIR, f"{split}.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: len(d["text"]) for d in data["documents"]}


def pct(x, n):
    return f"{x} ({100.0 * x / n:.0f}%)" if n else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-slug", default="qwen3-8b")
    args = ap.parse_args()

    recs = load("rlm", args.split, args.n, args.seed, args.model_slug)
    if not recs:
        raise SystemExit("No RLM records found yet.")

    ok = [r for r in recs if not r.get("error") and r.get("pred")]
    errored = [r for r in recs if r.get("error")]
    L = []

    L.append("=" * 66)
    L.append("RLM TRAJECTORY ANALYSIS — ContractNLI")
    L.append("=" * 66)
    L.append(f"Records: {len(recs)}   scoreable: {len(ok)}   "
             f"hard errors: {len(errored)}")
    L.append(f"Parse failures: {pct(sum(1 for r in ok if not r['parsed_ok']), len(ok))}")

    if errored:
        L.append("")
        L.append("Error taxonomy:")
        for kind, c in Counter(
                r["error"].split(":")[0] for r in errored).most_common():
            L.append(f"  {kind:40} {c}")

    # ---- Does the recursion actually engage? -------------------------
    # A 1-call trajectory means the root LM answered without ever running
    # code against `context` — the scaffold collapsed to a zero-shot call
    # on a prompt that does not even contain the contract.
    calls = [(r["usage"] or {}).get("calls", 0) for r in ok]
    L.append("")
    L.append("-" * 66)
    L.append("DOES THE RECURSION ENGAGE?")
    L.append("-" * 66)
    dist = Counter(calls)
    for c in sorted(dist):
        bar = "#" * min(50, dist[c])
        L.append(f"  {c:3} LLM call(s) : {dist[c]:4}  {bar}")
    n_single = sum(1 for c in calls if c <= 1)
    L.append(f"\n  Trajectories with <=1 call (never inspected context): "
             f"{pct(n_single, len(ok))}")
    if calls:
        L.append(f"  Median calls: {statistics.median(calls):.1f}   "
                 f"mean: {statistics.mean(calls):.2f}   max: {max(calls)}")

    # ---- Accuracy conditioned on engagement ---------------------------
    engaged = [r for r in ok if (r["usage"] or {}).get("calls", 0) > 1]
    lazy = [r for r in ok if (r["usage"] or {}).get("calls", 0) <= 1]
    L.append("")
    L.append("-" * 66)
    L.append("ACCURACY BY ENGAGEMENT")
    L.append("-" * 66)
    for name, group in (("inspected context (>1 call)", engaged),
                        ("answered blind (<=1 call)", lazy)):
        if group:
            pairs = [(r["gold"], r["pred"]) for r in group]
            acc = sum(g == p for g, p in pairs) / len(pairs)
            L.append(f"  {name:32} n={len(group):4}  "
                     f"acc={acc:.3f}  bal_acc={bal_acc(pairs):.3f}")
        else:
            L.append(f"  {name:32} n=0")

    # ---- Cost ---------------------------------------------------------
    toks = [(r["usage"] or {}).get("total", 0) for r in ok]
    costs = [(r["usage"] or {}).get("cost") for r in ok]
    costs = [c for c in costs if c is not None]
    secs = [r["elapsed_s"] for r in ok]
    L.append("")
    L.append("-" * 66)
    L.append("COST / EFFICIENCY (whole recursion tree)")
    L.append("-" * 66)
    if toks:
        st = sorted(toks)
        L.append(f"  Tokens  mean {statistics.mean(toks):8.0f}   "
                 f"median {statistics.median(toks):8.0f}   "
                 f"p90 {st[int(.9*len(st))-1]:8.0f}   max {max(toks):8.0f}")
    if secs:
        L.append(f"  Seconds mean {statistics.mean(secs):8.1f}   "
                 f"median {statistics.median(secs):8.1f}   max {max(secs):8.1f}")
    if costs:
        L.append(f"  Cost    mean ${statistics.mean(costs):.5f}   "
                 f"total ${sum(costs):.4f}")
        # The paper's own finding: median RLM run is cheap, the mean is
        # dragged up by a few runaway trajectories. Check it here.
        L.append(f"  Cost    median ${statistics.median(costs):.5f}  "
                 f"(mean/median = {statistics.mean(costs)/statistics.median(costs):.2f}x "
                 f"-> outlier skew)")

    # ---- Per-class confusion ------------------------------------------
    L.append("")
    L.append("-" * 66)
    L.append("CONFUSION (rows = gold, cols = predicted)")
    L.append("-" * 66)
    conf = defaultdict(Counter)
    for r in ok:
        conf[r["gold"]][r["pred"]] += 1
    L.append("  " + " " * 15 + "".join(l[:12].rjust(14) for l in LABELS))
    for g in LABELS:
        L.append("  " + g.ljust(15)
                 + "".join(str(conf[g][p]).rjust(14) for p in LABELS))
    pred_dist = Counter(r["pred"] for r in ok)
    L.append(f"\n  Prediction distribution: {dict(pred_dist)}")
    L.append(f"  Gold distribution      : {dict(Counter(r['gold'] for r in ok))}")

    # ---- Context length effect ----------------------------------------
    lens = contract_lengths(args.split)
    withlen = [(lens.get(r["doc_id"], 0), r) for r in ok if r["doc_id"] in lens]
    if withlen:
        withlen.sort(key=lambda x: x[0])
        L.append("")
        L.append("-" * 66)
        L.append("EFFECT OF CONTRACT LENGTH (tercile split)")
        L.append("-" * 66)
        third = max(1, len(withlen) // 3)
        for name, chunk in (("short", withlen[:third]),
                            ("medium", withlen[third:2 * third]),
                            ("long", withlen[2 * third:])):
            if not chunk:
                continue
            pairs = [(r["gold"], r["pred"]) for _, r in chunk]
            acc = sum(g == p for g, p in pairs) / len(pairs)
            mt = statistics.mean((r["usage"] or {}).get("total", 0)
                                 for _, r in chunk)
            mc = statistics.mean((r["usage"] or {}).get("calls", 0)
                                 for _, r in chunk)
            L.append(f"  {name:7} n={len(chunk):3}  "
                     f"chars {chunk[0][0]:6}-{chunk[-1][0]:6}  "
                     f"acc={acc:.3f}  bal_acc={bal_acc(pairs):.3f}  "
                     f"avg_tokens={mt:7.0f}  avg_calls={mc:.1f}")

    text = "\n".join(L)
    print(text)
    dest = os.path.join(RESULTS_DIR, "rlm-study-rlm", args.model_slug,
                        f"analysis_{args.split}_n{args.n}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()
