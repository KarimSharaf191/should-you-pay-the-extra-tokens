"""
compare_thinking.py
===================
Side-by-side table of the thinking-OFF and thinking-ON arms of the ContractNLI
matrix, plus the RLM row (which exists thinking-off only).

Both arms score the identical stratified sample, so every OFF/ON pair is
paired and the delta column is a within-instance comparison.

    python compare_thinking.py --n 150
    python compare_thinking.py --n 1037 --out results/full_dev_comparison.txt
"""

import os
import json
import argparse
from collections import defaultdict

from sklearn.metrics import balanced_accuracy_score, f1_score

import contractnli_zeroshot as cz

CONDITIONS = ["zeroshot", "dspy", "cot", "rag", "agentic_rag"]
MODELS = ["qwen3-8b", "llama-3.2-3b-instruct"]
LABELS = ["Entailment", "Contradiction", "NotMentioned"]


def load(path):
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["doc_id"], r["nda_key"])] = r   # last wins
    return rows


def stats(rows):
    if not rows:
        return None
    ok = [r for r in rows.values() if not r.get("error") and r.get("pred")]
    if not ok:
        return None
    yt = [r["gold"] for r in ok]
    yp = [r["pred"] for r in ok]
    us = [r["usage"] for r in ok if r.get("usage")]

    def avg(f):
        v = [(u.get(f) or 0) for u in us]
        return sum(v) / len(v) if v else 0

    return {
        "n": len(ok),
        "bal": balanced_accuracy_score(yt, yp),
        "f1": f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
        "per_class": dict(zip(LABELS, f1_score(yt, yp, labels=LABELS,
                                               average=None, zero_division=0))),
        "tok": avg("total"),
        "gen": avg("completion"),
        "rea": avg("reasoning"),
        "think": avg("think_tokens"),
        "sec": sum(r["elapsed_s"] for r in ok) / len(ok),
        "by_key": {k: r for k, r in rows.items()
                   if not r.get("error") and r.get("pred")},
    }


def paired_delta(off, on):
    """Instances the two arms disagree on, restricted to shared keys."""
    if not off or not on:
        return None
    shared = set(off["by_key"]) & set(on["by_key"])
    if not shared:
        return None
    on_right = sum(1 for k in shared
                   if on["by_key"][k]["pred"] == on["by_key"][k]["gold"]
                   and off["by_key"][k]["pred"] != off["by_key"][k]["gold"])
    off_right = sum(1 for k in shared
                    if off["by_key"][k]["pred"] == off["by_key"][k]["gold"]
                    and on["by_key"][k]["pred"] != on["by_key"][k]["gold"])
    return len(shared), on_right, off_right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fn = f"records_{args.split}_n{args.n}_seed{args.seed}.jsonl"
    data = {}
    for m in MODELS:
        for c in CONDITIONS:
            for arm, pref in (("off", "matrix"), ("on", "matrix-think")):
                data[(m, c, arm)] = stats(load(
                    os.path.join(cz.RESULTS_DIR, f"{pref}-{c}", m, fn)))
    rlm = stats(load(os.path.join(cz.RESULTS_DIR, "rlm-study-rlm",
                                  "qwen3-8b", fn)))

    L = []
    L.append("=" * 100)
    L.append(f"ContractNLI — thinking OFF vs ON   |   {args.split} "
             f"n={args.n} seed={args.seed}")
    L.append("=" * 100)
    L.append("Same instances in both arms, so OFF/ON is a paired comparison.")
    L.append("")

    # ---- main table -------------------------------------------------
    hdr = (f"{'model':<22}{'condition':<13}"
           f"{'bal OFF':>9}{'bal ON':>9}{'delta':>8}   "
           f"{'F1 OFF':>8}{'F1 ON':>8}   "
           f"{'tok OFF':>9}{'tok ON':>9}   {'gen ON':>8}{'reas ON':>8}")
    L.append(hdr)
    L.append("-" * len(hdr))

    for m in MODELS:
        for c in CONDITIONS:
            off, on = data[(m, c, "off")], data[(m, c, "on")]
            if not off and not on:
                continue
            d = (on["bal"] - off["bal"]) if (off and on) else None
            L.append(
                f"{m:<22}{c:<13}"
                f"{off['bal']:>9.3f}" if off else f"{m:<22}{c:<13}{'-':>9}")
            L[-1] += (f"{on['bal']:>9.3f}" if on else f"{'-':>9}")
            L[-1] += (f"{d:>+8.3f}" if d is not None else f"{'-':>8}")
            L[-1] += "   "
            L[-1] += (f"{off['f1']:>8.3f}" if off else f"{'-':>8}")
            L[-1] += (f"{on['f1']:>8.3f}" if on else f"{'-':>8}")
            L[-1] += "   "
            L[-1] += (f"{off['tok']:>9.0f}" if off else f"{'-':>9}")
            L[-1] += (f"{on['tok']:>9.0f}" if on else f"{'-':>9}")
            L[-1] += "   "
            L[-1] += (f"{on['gen']:>8.0f}" if on else f"{'-':>8}")
            L[-1] += (f"{on['rea']:>8.0f}" if on else f"{'-':>8}")
        L.append("")

    if rlm:
        L.append(f"{'qwen3-8b':<22}{'rlm':<13}{rlm['bal']:>9.3f}{'n/a':>9}"
                 f"{'-':>8}   {rlm['f1']:>8.3f}{'-':>8}   "
                 f"{rlm['tok']:>9.0f}{'-':>9}   {'-':>8}{'-':>8}")
        L.append("  (RLM was only run thinking-off; the rlm library reports no "
                 "reasoning breakdown)")
        L.append("")

    # ---- paired disagreement ---------------------------------------
    L.append("=" * 100)
    L.append("PAIRED DISAGREEMENT (thinking ON vs OFF, same instances)")
    L.append("=" * 100)
    L.append(f"{'model':<22}{'condition':<13}{'shared':>8}"
             f"{'ON right / OFF wrong':>22}{'OFF right / ON wrong':>22}")
    L.append("-" * 87)
    for m in MODELS:
        for c in CONDITIONS:
            pd = paired_delta(data[(m, c, "off")], data[(m, c, "on")])
            if pd:
                n, a, b = pd
                L.append(f"{m:<22}{c:<13}{n:>8}{a:>22}{b:>22}")
        L.append("")

    # ---- per-class F1 ----------------------------------------------
    L.append("=" * 100)
    L.append("PER-CLASS F1")
    L.append("=" * 100)
    L.append(f"{'model':<22}{'condition':<13}{'arm':>5}"
             + "".join(f"{l[:13]:>15}" for l in LABELS))
    L.append("-" * 90)
    for m in MODELS:
        for c in CONDITIONS:
            for arm in ("off", "on"):
                s = data[(m, c, arm)]
                if s:
                    L.append(f"{m:<22}{c:<13}{arm:>5}"
                             + "".join(f"{s['per_class'][l]:>15.3f}"
                                       for l in LABELS))
        L.append("")

    text = "\n".join(L)
    print(text)
    out = args.out or os.path.join(cz.RESULTS_DIR,
        f"thinking_comparison_{args.split}_n{args.n}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
