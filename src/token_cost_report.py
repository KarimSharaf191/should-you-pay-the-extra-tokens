"""
token_cost_report.py
====================
Token-usage and dollar-cost table across the matrix runs.

Two sources of cost, in order of preference:

  1. `usage.cost_usd` recorded per instance, which is what OpenRouter actually
     charged. Exact, including any provider-side routing or discounts.
  2. Otherwise, tokens x the model's current OpenRouter list price. Runs made
     before cost logging existed (and any vLLM run) fall back to this. It is
     accurate as long as the model was served at list price, but it cannot see
     a different upstream provider having been routed to at the time.

Every figure is reported as mean / median / min / max, not just the mean: the
spread is the point. Agentic and thinking conditions have long right tails
where a handful of instances cost an order of magnitude more than the typical
one, and a mean alone hides that completely.

Usage
-----
    python token_cost_report.py --n 1037
    python token_cost_report.py --n 1037 --csv results/cost_table.csv
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("CONTRACTNLI_RESULTS_DIR", os.path.join(_ROOT, "results"))
import csv
import json
import glob
import argparse
import statistics

# OpenRouter list prices, USD per token. Update if you re-run much later --
# these move. Fetch current values from https://openrouter.ai/api/v1/models
PRICES = {
    "qwen3-8b":              {"prompt": 0.000000117, "completion": 0.000000455},
    "llama-3.2-3b-instruct": {"prompt": 0.000000050, "completion": 0.000000330},
    "qwen3.5-9b":            {"prompt": 0.000000117, "completion": 0.000000455},
}

FIELDS = [("prompt", "prompt tok"), ("completion", "compl tok"),
          ("reasoning", "reason tok"), ("total", "total tok"),
          ("think_tokens", "<think> tok"), ("steps", "steps"),
          ("retrieved_spans", "ret spans")]


def spread(vals):
    """mean / median / min / max, zero-safe."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (statistics.mean(vals), statistics.median(vals),
            min(vals), max(vals))


def instance_cost(u, model):
    """Cost of one instance: recorded if present, else priced from tokens."""
    rec = u.get("cost_usd")
    if rec:
        return float(rec), "recorded"
    price = PRICES.get(model)
    if not price:
        return None, "unpriced"
    # completion_tokens already includes reasoning tokens, and reasoning bills
    # at the completion rate -- so completion alone is the right multiplicand.
    # Adding reasoning separately would double-count it.
    return ((u.get("prompt") or 0) * price["prompt"]
            + (u.get("completion") or 0) * price["completion"]), "listprice"


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or not r.get("usage"):
                continue
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=1037)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    pattern = os.path.join(RESULTS_DIR, "matrix*", "*",
                           f"records_{args.split}_n{args.n}_seed{args.seed}.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no runs matching {pattern}")

    L = ["=" * 100,
         f"Token usage and cost - {args.split} n={args.n} seed={args.seed}",
         "=" * 100,
         "completion tokens INCLUDE reasoning tokens; cost uses completion "
         "alone to avoid double-counting.",
         ""]
    rows = []

    for p in paths:
        parts = p.split(os.sep)
        cond, model = parts[1], parts[2]
        recs = load(p)
        if not recs:
            continue
        us = [r["usage"] for r in recs]

        costs, sources = [], set()
        for u in us:
            c, src = instance_cost(u, model)
            sources.add(src)
            if c is not None:
                costs.append(c)

        L.append(f"{cond} | {model} | n={len(recs)}")
        for key, lab in FIELDS:
            s = spread([u.get(key) for u in us if u.get(key) is not None])
            if not s or s[3] == 0:
                continue
            L.append(f"    {lab:12} mean={s[0]:10.1f}  med={s[1]:9.1f}  "
                     f"min={s[2]:8.0f}  max={s[3]:9.0f}")
        if costs:
            c = spread(costs)
            L.append(f"    {'cost USD':12} mean={c[0]:10.6f}  med={c[1]:9.6f}  "
                     f"min={c[2]:8.6f}  max={c[3]:9.6f}")
            L.append(f"    {'':12} RUN TOTAL = ${sum(costs):.4f}   "
                     f"(source: {'/'.join(sorted(sources))})")
            rows.append({
                "condition": cond, "model": model, "n": len(recs),
                "mean_total_tok": round(statistics.mean(
                    [u.get("total") or 0 for u in us]), 1),
                "min_total_tok": min(u.get("total") or 0 for u in us),
                "max_total_tok": max(u.get("total") or 0 for u in us),
                "mean_cost_usd": round(c[0], 8),
                "min_cost_usd": round(c[2], 8),
                "max_cost_usd": round(c[3], 8),
                "run_total_usd": round(sum(costs), 6),
                "cost_source": "/".join(sorted(sources)),
            })
        L.append("")

    grand = sum(r["run_total_usd"] for r in rows)
    L.append("-" * 100)
    L.append(f"GRAND TOTAL across all listed runs: ${grand:.4f}")

    text = "\n".join(L)
    print(text)
    dest = os.path.join(RESULTS_DIR, f"token_cost_{args.split}_n{args.n}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {dest}")

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV   -> {args.csv}")


if __name__ == "__main__":
    main()
