"""
bootstrap_test.py
=================
Paired bootstrap significance testing over the (model x condition x thinking)
matrix produced by run_matrix.py.

Every condition scores the SAME instances, so the comparisons are paired: we
resample INSTANCES (not predictions), recompute the metric for both arms on the
identical resample, and build the distribution of the DELTA. Pairing cancels
the per-instance difficulty that both arms share, which is what makes small
deltas detectable at this sample size.

Two families of comparison are reported:

  thinking   thinking ON vs OFF, within each scaffold   (5 tests / model)
  scaffold   each scaffold vs the zeroshot baseline,
             separately within the ON and OFF arms      (8 tests / model)

Both are corrected for multiple comparisons with Holm-Bonferroni, per model and
per family. RLM is excluded (it does not live in the matrix-* tree).

Usage
-----
    python bootstrap_test.py                       # dev, n=150, seed 0
    python bootstrap_test.py --n 1037              # full dev
    python bootstrap_test.py --metric macro_f1
    python bootstrap_test.py --models qwen3-8b --iters 20000
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("CONTRACTNLI_RESULTS_DIR", os.path.join(_ROOT, "results"))
import re
import json
import glob
import argparse
from collections import defaultdict

import numpy as np

LABELS = ["Entailment", "Contradiction", "NotMentioned"]
SCAFFOLDS = ["zeroshot", "dspy", "cot", "rag", "agentic_rag"]
BASELINE = "zeroshot"


# ----------------------------------------------------------------------
# Metrics. Implemented directly rather than via sklearn so that a class that
# vanishes from a bootstrap resample is handled explicitly (skipped) instead of
# silently becoming a nan that poisons the mean.
# ----------------------------------------------------------------------
def balanced_accuracy(gold, pred):
    """Mean per-class recall over the classes actually present in `gold`."""
    recalls = []
    for c in range(len(LABELS)):
        mask = gold == c
        n = mask.sum()
        if n == 0:
            continue                       # class absent from this resample
        recalls.append((pred[mask] == c).sum() / n)
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(gold, pred):
    """Unweighted mean F1 over classes present in gold or predicted."""
    f1s = []
    for c in range(len(LABELS)):
        tp = ((pred == c) & (gold == c)).sum()
        fp = ((pred == c) & (gold != c)).sum()
        fn = ((pred != c) & (gold == c)).sum()
        if tp + fp + fn == 0:
            continue                       # class neither present nor predicted
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


METRICS = {"bal_acc": balanced_accuracy, "macro_f1": macro_f1}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def records_path(cond, thinking, model, split, n, seed):
    prefix = "matrix-think-" if thinking else "matrix-"
    return os.path.join(RESULTS_DIR, prefix + cond, model,
                        f"records_{split}_n{n}_seed{seed}.jsonl")


def load_run(path):
    """Return {(doc_id, nda_key): (gold, pred)} for non-errored records."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            if r.get("gold") not in LABELS or r.get("pred") not in LABELS:
                continue
            out[(str(r["doc_id"]), r["nda_key"])] = (r["gold"], r["pred"])
    return out


def discover_models(split, n, seed):
    """Models that have at least one complete-looking run in the matrix tree."""
    found = set()
    for p in glob.glob(os.path.join(RESULTS_DIR, "matrix*", "*",
                                    f"records_{split}_n{n}_seed{seed}.jsonl")):
        found.add(p.split(os.sep)[2])
    return sorted(found)


# ----------------------------------------------------------------------
# Paired bootstrap
# ----------------------------------------------------------------------
def paired_bootstrap(gold, pred_a, pred_b, metric_fn, iters, rng):
    """Bootstrap the distribution of metric(A) - metric(B) over resampled
    instances. `gold`, `pred_a`, `pred_b` are aligned int arrays.

    Returns (delta_point, lo95, hi95, p_two_sided).
    """
    n = len(gold)
    obs = metric_fn(gold, pred_a) - metric_fn(gold, pred_b)

    deltas = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)      # SAME indices for both arms
        g = gold[idx]
        deltas[i] = metric_fn(g, pred_a[idx]) - metric_fn(g, pred_b[idx])

    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # Two-sided achieved significance level, centred on the observed delta:
    # how often does the recentred distribution reach as far from 0 as we did?
    centred = deltas - deltas.mean()
    p = float((np.abs(centred) >= abs(obs)).mean())
    p = min(1.0, max(p, 1.0 / (len(deltas) + 1)))   # floor at resolution
    return obs, float(lo), float(hi), p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)           # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def align(run_a, run_b):
    """Intersect two runs on instance key; return (gold, pred_a, pred_b) as
    int arrays in a deterministic order."""
    keys = sorted(set(run_a) & set(run_b))
    gold, pa, pb = [], [], []
    for k in keys:
        g_a, p_a = run_a[k]
        g_b, p_b = run_b[k]
        assert g_a == g_b, f"gold mismatch at {k}: {g_a} vs {g_b}"
        gold.append(LABELS.index(g_a))
        pa.append(LABELS.index(p_a))
        pb.append(LABELS.index(p_b))
    return (np.array(gold, dtype=np.int8),
            np.array(pa, dtype=np.int8),
            np.array(pb, dtype=np.int8), keys)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def fmt_row(name, n, a, b, delta, lo, hi, p, p_adj):
    star = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else \
           "*" if p_adj < 0.05 else ""
    return (f"  {name:34} n={n:4}  {b:.3f} -> {a:.3f}  "
            f"D={delta:+.3f}  [{lo:+.3f},{hi:+.3f}]  "
            f"p={p:.4f}  p_holm={p_adj:.4f} {star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=None,
                    help="model slugs as they appear under results/matrix-*/ "
                         "(default: auto-discover)")
    ap.add_argument("--metric", default="bal_acc", choices=list(METRICS))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    metric_fn = METRICS[args.metric]
    models = args.models or discover_models(args.split, args.n, args.seed)
    if not models:
        raise SystemExit(f"no runs found for split={args.split} n={args.n} "
                         f"seed={args.seed}")

    L = ["=" * 96,
         f"Paired bootstrap significance test - metric={args.metric}",
         "=" * 96,
         f"Split {args.split} | n={args.n} | seed={args.seed} | "
         f"{args.iters} resamples | boot-seed {args.boot_seed}",
         "Pairing: instances resampled jointly; both arms scored on the same "
         "resample.",
         "Holm-Bonferroni applied within each (model, family).",
         "Columns: <reference> -> <arm>  D=arm-reference  [95% CI]",
         ""]

    for model in models:
        # Load every available run once.
        runs = {}
        for cond in SCAFFOLDS:
            for think in (False, True):
                r = load_run(records_path(cond, think, model, args.split,
                                          args.n, args.seed))
                if r:
                    runs[(cond, think)] = r

        L.append("#" * 96)
        L.append(f"# {model}")
        L.append("#" * 96)
        have = sorted(f"{c}{'+think' if t else ''}" for (c, t) in runs)
        L.append(f"  runs present: {', '.join(have) if have else '(none)'}")
        missing = [f"{c}{'+think' if t else ''}"
                   for c in SCAFFOLDS for t in (False, True)
                   if (c, t) not in runs]
        if missing:
            L.append(f"  MISSING     : {', '.join(missing)}")
        L.append("")

        rng = np.random.default_rng(args.boot_seed)

        # ---- family 1: thinking ON vs OFF, within scaffold -----------------
        fam = []
        for cond in SCAFFOLDS:
            if (cond, True) not in runs or (cond, False) not in runs:
                continue
            gold, pa, pb, keys = align(runs[(cond, True)], runs[(cond, False)])
            if len(gold) == 0:
                continue
            d, lo, hi, p = paired_bootstrap(gold, pa, pb, metric_fn,
                                            args.iters, rng)
            fam.append((f"{cond}: think ON vs OFF", len(gold),
                        metric_fn(gold, pa), metric_fn(gold, pb),
                        d, lo, hi, p))
        if fam:
            adj = holm([f[-1] for f in fam])
            L.append("  THINKING ON vs OFF  (positive D = thinking helps)")
            L.append("  " + "-" * 92)
            for f, a in zip(fam, adj):
                L.append(fmt_row(*f, a))
            L.append("")

        # ---- family 2: scaffold vs zeroshot baseline, within arm -----------
        fam = []
        for think in (False, True):
            if (BASELINE, think) not in runs:
                continue
            tag = "think" if think else "nothink"
            for cond in SCAFFOLDS:
                if cond == BASELINE or (cond, think) not in runs:
                    continue
                gold, pa, pb, keys = align(runs[(cond, think)],
                                           runs[(BASELINE, think)])
                if len(gold) == 0:
                    continue
                d, lo, hi, p = paired_bootstrap(gold, pa, pb, metric_fn,
                                                args.iters, rng)
                fam.append((f"[{tag}] {cond} vs {BASELINE}", len(gold),
                            metric_fn(gold, pa), metric_fn(gold, pb),
                            d, lo, hi, p))
        if fam:
            adj = holm([f[-1] for f in fam])
            L.append(f"  SCAFFOLD vs {BASELINE.upper()}  "
                     f"(positive D = scaffold beats baseline)")
            L.append("  " + "-" * 92)
            for f, a in zip(fam, adj):
                L.append(fmt_row(*f, a))
            L.append("")

        # ---- per-class support, the real limit on power -------------------
        any_run = next(iter(runs.values()), None)
        if any_run:
            counts = defaultdict(int)
            for g, _ in any_run.values():
                counts[g] += 1
            L.append("  class support: " + "  ".join(
                f"{c}={counts.get(c, 0)}" for c in LABELS))
            rare = min(counts.values()) if counts else 0
            L.append(f"  -> balanced accuracy is bottlenecked by the rarest "
                     f"class (n={rare}); its recall has SE ~"
                     f"{(0.25 / max(rare, 1)) ** 0.5:.3f}, contributing "
                     f"~{(0.25 / max(rare, 1)) ** 0.5 / 3:.3f} to the metric.")
            L.append("")

    L.append("Significance marks are on Holm-adjusted p: * <.05  ** <.01  "
             "*** <.001")
    text = "\n".join(L)
    print(text)

    dest = args.out or os.path.join(RESULTS_DIR, f"bootstrap_{args.metric}_{args.split}_n{args.n}.txt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()
