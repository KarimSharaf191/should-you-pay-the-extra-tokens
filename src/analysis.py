"""Statistics behind the paper's appendix tables, computed from the saved
per-claim records in results/ (no API calls):

  * 95% bootstrap confidence intervals of balanced accuracy for every
    ContractNLI condition and model, and for the RLM and Llama retrieval runs
  * paired bootstrap tests with Holm-Bonferroni correction (thinking on vs off,
    each scaffold vs zero-shot) for Qwen3-8B, 4B and 1.7B on ContractNLI
  * token use by contract length (ContractNLI, Qwen3-8B)
  * the decoding check (greedy vs three sampled runs, Qwen3-8B zero-shot with thinking)

    python src/analysis.py            # writes results/summaries/analysis.txt
"""

import json
import os
import statistics as S

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
B = 10000
rng = np.random.default_rng(0)
SCAFFOLDS = ["zeroshot", "dspy", "cot", "rag", "agentic_rag"]
SIZES = ["qwen3-8b", "qwen3-4b", "qwen3-1.7b"]
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def matrix(prefix, cond, model):
    best = {}
    for r in read(os.path.join(RES, f"{prefix}-{cond}", model, "records_dev_n1037_seed0.jsonl")):
        k = (r["doc_id"], r["nda_key"])
        if k not in best or not r.get("error"):
            best[k] = r
    return best


def ba(g, p, labels):
    return float(np.mean([(p[g == l] == l).mean() for l in labels if (g == l).any()]))


def ci(gold, pred):
    g, p = np.array(gold), np.array(pred)
    labels, n = sorted(set(gold)), len(gold)
    vals = [ba(g[i], p[i], labels) for i in rng.integers(0, n, (B, n))]
    return ba(g, p, labels), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def paired(a, b):
    keys = sorted(set(a) & set(b))
    g = np.array([a[k]["gold"] for k in keys])
    pa = np.array([a[k].get("pred") or "NONE" for k in keys])
    pb = np.array([b[k].get("pred") or "NONE" for k in keys])
    labels, n = sorted(set(g)), len(g)
    d0 = ba(g, pb, labels) - ba(g, pa, labels)
    ds = np.array([ba(g[i], pb[i], labels) - ba(g[i], pa[i], labels)
                   for i in rng.integers(0, n, (B, n))])
    p = max(2 * min((ds <= 0).mean(), (ds >= 0).mean()), 1 / B)
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5), p


def holm(ps):
    order, m, out, run = np.argsort(ps), len(ps), [0] * len(ps), 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * ps[i]))
        out[i] = run
    return out


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main():
    say("== 95% bootstrap CIs of balanced accuracy, ContractNLI dev (1,037)")
    for m in SIZES:
        for prefix, think in (("matrix", "off"), ("matrix-think", "on")):
            row = [fmt(ci(*zip(*[(r["gold"], r.get("pred") or "NONE")
                                 for r in matrix(prefix, c, m).values()]))) for c in SCAFFOLDS]
            say(f"{m:11} {think:3} " + "  ".join(row))
    row = [fmt(ci(*zip(*[(r["gold"], r.get("pred") or "NONE")
                         for r in matrix("matrix", c, "llama-3.2-3b-instruct").values()])))
           for c in SCAFFOLDS]
    say(f"{'llama-3.2-3b':11} --  " + "  ".join(row))

    say("\n== Paired bootstrap tests, ContractNLI (Holm within family and model)")
    for m in SIZES:
        fams = {"thinking on vs off": [(c, paired(matrix("matrix", c, m), matrix("matrix-think", c, m)))
                                       for c in SCAFFOLDS],
                "vs zero-shot (off)": [(c, paired(matrix("matrix", "zeroshot", m), matrix("matrix", c, m)))
                                       for c in SCAFFOLDS[1:]]}
        for fam, rows in fams.items():
            for (c, (d, lo, hi, p)), h in zip(rows, holm([r[1][3] for r in rows])):
                say(f"{m:11} {fam:19} {c:12} d={d:+.3f} [{lo:+.3f}, {hi:+.3f}] p_holm={h:.4f}")

    say("\n== RLM (Qwen3-8B, thinking off) and Llama retrieval: BA with 95% CI")
    for bench, n in (("contractnli", 1037), ("findver", 700), ("coverbench", 733)):
        rs = read(os.path.join(RES, "rlm-multi", bench, f"rlm_nothink_n{n}_seed0.jsonl"))
        ok = [r for r in rs if not r.get("error")]
        say(f"rlm {bench:11} BA {fmt(ci([r['gold'] for r in rs], [r.get('pred') or 'NONE' for r in rs]))}"
            f"  failed={len(rs) - len(ok)}  tokens/claim={S.mean((r['prov_in'] or 0) + (r['prov_out'] or 0) for r in ok):,.0f}"
            f"  recursion rate={sum(r['recursed'] for r in ok) / len(ok):.1%}")
    for bench, n in (("findver", 700), ("coverbench", 733)):
        for c in ("rag", "agentic_rag"):
            rs = [r for r in read(os.path.join(RES, "rag-multi", bench, "llama-3.2-3b-instruct",
                                               f"{c}_nothink_n{n}.jsonl")) if not r.get("error")]
            say(f"llama {bench:10} {c:12} BA {fmt(ci([r['gold'] for r in rs], [r['pred'] for r in rs]))}")

    say("\n== Token use by contract length, ContractNLI, Qwen3-8B (input / generated per claim)")
    doclen = {k: r["usage"]["prompt"] for k, r in matrix("matrix", "zeroshot", "qwen3-8b").items() if r.get("usage")}
    edges = [0, 1500, 2500, 3500, 10 ** 9]
    for prefix in ("matrix", "matrix-think"):
        for c in SCAFFOLDS:
            d = matrix(prefix, c, "qwen3-8b")
            cells = []
            for lo, hi in zip(edges, edges[1:]):
                ks = [k for k in d if k in doclen and lo <= doclen[k] < hi and d[k].get("usage")]
                cells.append(f"{S.mean(d[k]['usage']['prompt'] for k in ks):5.0f}/"
                             f"{S.mean(d[k]['usage']['completion'] for k in ks):4.0f}")
            say(f"{prefix:12} {c:12} " + "  ".join(cells))

    say("\n== Decoding check: Qwen3-8B zero-shot with thinking, greedy vs sampled (T=0.6, top-p 0.95, top-k 20)")
    greedy = matrix("matrix-think", "zeroshot", "qwen3-8b")
    seeds = {s: matrix(f"matrix-think-s{s}", "zeroshot", "qwen3-8b") for s in (1, 2, 3)}
    common = sorted(set(greedy).intersection(*[{k for k, r in d.items() if not r.get("error")}
                                               for d in seeds.values()]))
    g = [greedy[k]["gold"] for k in common]
    labels = sorted(set(g))
    gb = ba(np.array(g), np.array([greedy[k]["pred"] or "NONE" for k in common]), labels)
    say(f"claims finished in all three sampled runs: {len(common)}")
    say(f"greedy   BA {gb:.3f}  generated tokens {S.mean(greedy[k]['usage']['completion'] for k in common):.0f}")
    vals = []
    for s, d in seeds.items():
        b = ba(np.array(g), np.array([d[k]["pred"] or "NONE" for k in common]), labels)
        vals.append(b)
        agree = np.mean([d[k]["pred"] == greedy[k]["pred"] for k in common])
        say(f"seed {s}   BA {b:.3f}  same label as greedy {agree:.1%}  "
            f"generated tokens {S.mean(d[k]['usage']['completion'] for k in common):.0f}")
    say(f"sampled  BA {np.mean(vals):.3f} +- {np.std(vals, ddof=1):.3f}")

    os.makedirs(os.path.join(RES, "summaries"), exist_ok=True)
    with open(os.path.join(RES, "summaries", "analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print("\nSaved -> results/summaries/analysis.txt")


if __name__ == "__main__":
    main()
