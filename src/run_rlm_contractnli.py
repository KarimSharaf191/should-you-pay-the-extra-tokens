"""
run_rlm_contractnli.py
======================
ContractNLI evaluation of Recursive Language Models (Zhang, Kraska & Khattab,
arXiv:2512.24601) against a budget-matched Chain-of-Thought baseline.

Both conditions run the SAME model (`qwen/qwen3-8b` via OpenRouter) over the
SAME stratified subsample, so any difference is attributable to the scaffold
rather than to the model or to sample drift.

  --condition rlm  : authors' `rlm` library scaffold (pip install rlms).
                     The contract is offloaded into the REPL as `context`;
                     the root LM only ever sees the hypothesis and can probe
                     the contract programmatically / via recursive sub-calls.
  --condition cot  : DSPy ChainOfThought with the whole contract in-prompt.
                     Reuses build_dspy_program("cot") from contractnli_zeroshot
                     so the baseline is literally the same program as the
                     existing full-dev CoT run.

NOTE ON HARDWARE: this runs the RLM *scaffold* on a stock instruction model,
which needs no GPU. The post-trained checkpoint (mit-oasys/rlm-qwen3-8b-v0.1)
is a separate condition that does require a GPU + vLLM; it is not served by any
HF Inference Provider.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python run_rlm_contractnli.py --condition cot --n 150
    python run_rlm_contractnli.py --condition rlm --n 150
    python run_rlm_contractnli.py --compare        # side-by-side report
"""

import os
import re
import json
import time
import random
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
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
MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-8b")
API_KEY = os.environ.get("OPENROUTER_API_KEY")

LABELS = ["Entailment", "Contradiction", "NotMentioned"]

# The root LM sees only this; the contract lives in the REPL.
ROOT_PROMPT = (
    "The `context` variable in your REPL holds the full text of a "
    "non-disclosure agreement (NDA).\n\n"
    "Decide the relationship between that contract and the following "
    "hypothesis:\n\n"
    "HYPOTHESIS: {hypothesis}\n\n"
    "Answer Entailment if the contract implies the hypothesis, Contradiction "
    "if the contract implies its negation, and NotMentioned if the contract "
    "does not address it.\n"
    "Inspect the contract in the REPL before answering. Finish with exactly "
    "one of these labels: Entailment, Contradiction, NotMentioned."
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_split(split):
    with open(os.path.join(DATA_DIR, f"{split}.json"), encoding="utf-8") as f:
        return json.load(f)


def iter_instances(data):
    """Yield (doc_id, nda_key, hypothesis, contract_text, gold_choice)."""
    labels = data["labels"]
    for doc in data["documents"]:
        text = doc["text"]
        annotations = doc["annotation_sets"][0]["annotations"]
        for nda_key, ann in annotations.items():
            yield (doc["id"], nda_key, labels[nda_key]["hypothesis"],
                   text, ann["choice"])


def stratified_sample(instances, n, seed=0):
    """Deterministic label-stratified subsample.

    Identical for a given (split, n, seed) regardless of condition, which is
    what makes the RLM-vs-CoT comparison paired rather than merely parallel.
    Class proportions are preserved so balanced accuracy stays comparable to
    the existing full-dev numbers.
    """
    if n is None or n >= len(instances):
        return list(instances)

    by_label = defaultdict(list)
    for inst in instances:
        by_label[inst[4]].append(inst)

    rng = random.Random(seed)
    picked = []
    for label in sorted(by_label):
        pool = sorted(by_label[label], key=lambda x: (x[0], x[1]))
        rng.shuffle(pool)
        share = max(1, round(n * len(by_label[label]) / len(instances)))
        picked.extend(pool[:share])

    rng.shuffle(picked)
    return picked[:n]


# ----------------------------------------------------------------------
# Label parsing
# ----------------------------------------------------------------------
def parse_label(raw):
    """Return (label, parsed_ok).

    Unlike the original script we do NOT silently fall back to NotMentioned:
    an unparseable RLM trajectory is a scaffold failure, and folding it into
    the majority-ish class would hide exactly the effect we are measuring.
    Callers record parsed_ok so the failure rate is reported separately.
    """
    if raw is None:
        return "NotMentioned", False

    text = str(raw)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    s = re.sub(r"[`'\"*]", "", text.strip().lower())
    if not s:
        return "NotMentioned", False

    # Prefer the LAST label mentioned: models restate the option list before
    # committing, so the final mention is the actual answer.
    hits = []
    for pat, lab in (
        (r"contradict\w*", "Contradiction"),
        (r"entail\w*", "Entailment"),
        (r"not[\s_-]*mentioned|notmentioned|neutral", "NotMentioned"),
    ):
        for m in re.finditer(pat, s):
            hits.append((m.start(), lab))

    if not hits:
        return "NotMentioned", False
    return max(hits)[1], True


# ----------------------------------------------------------------------
# Condition: RLM (authors' scaffold)
# ----------------------------------------------------------------------
_thread_local = threading.local()


def _get_rlm(args):
    """One RLM client per worker thread.

    `completion()` spawns its own environment per call, but we keep the client
    thread-local anyway so concurrent workers never share REPL state.
    """
    rlm = getattr(_thread_local, "rlm", None)
    if rlm is None:
        from rlm import RLM

        # backend="vllm" (or "openai" with --base-url) lets this same harness
        # drive a locally served checkpoint — e.g. mit-oasys/rlm-qwen3-8b-v0.1
        # on a Colab/RunPod GPU — instead of OpenRouter.
        backend_kwargs = {"model_name": MODEL, "api_key": API_KEY or "EMPTY"}
        if args.base_url:
            backend_kwargs["base_url"] = args.base_url

        rlm = RLM(
            backend=args.backend,
            backend_kwargs=backend_kwargs,
            environment=args.environment,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            max_errors=args.max_errors,
            # Hard per-instance ceiling. The paper flags "exploding sub-call
            # costs" as a real failure mode, so we cap rather than discover it
            # on the invoice.
            max_budget=args.max_budget,
            max_timeout=args.max_timeout,
            max_concurrent_subcalls=args.max_concurrent_subcalls,
            verbose=False,
        )
        _thread_local.rlm = rlm
    return rlm


def predict_rlm(args, hypothesis, contract):
    rlm = _get_rlm(args)

    # prompt = the context offloaded into the REPL.
    # root_prompt = the small question the root LM actually sees.
    # (The original script had these reversed and passed a `context=` kwarg
    # that does not exist in the library's signature.)
    result = rlm.completion(prompt=contract, root_prompt=ROOT_PROMPT.format(
        hypothesis=hypothesis))

    label, ok = parse_label(getattr(result, "response", None))

    summary = getattr(result, "usage_summary", None)
    usage = {
        "calls": 0,
        "prompt": 0,
        "completion": 0,
        "total": 0,
        "cost": None,
        "wall_s": getattr(result, "execution_time", None),
    }
    if summary is not None:
        # usage_summary aggregates the WHOLE recursion tree (root + every
        # sub-call), which is the number the efficiency axis needs.
        usage["prompt"] = summary.total_input_tokens
        usage["completion"] = summary.total_output_tokens
        usage["total"] = summary.total_input_tokens + summary.total_output_tokens
        usage["cost"] = summary.total_cost
        usage["calls"] = sum(
            m.total_calls for m in summary.model_usage_summaries.values())

    return label, ok, usage, getattr(result, "response", "")


# ----------------------------------------------------------------------
# Condition: CoT baseline (budget-matched, same subsample)
# ----------------------------------------------------------------------
_cot_program = None
_cot_lock = threading.Lock()


def _get_cot_program():
    global _cot_program
    with _cot_lock:
        if _cot_program is None:
            import contractnli_zeroshot as cz

            # Reuse the exact program from the existing harness so this
            # baseline is the same condition as the full-dev CoT run.
            _cot_program = cz.build_dspy_program(condition="cot")
    return _cot_program


def predict_cot(args, hypothesis, contract):
    program = _get_cot_program()
    out = program(contract=contract, hypothesis=hypothesis)

    label, ok = parse_label(getattr(out, "label", None))

    usage = {"calls": 1, "prompt": 0, "completion": 0, "total": 0,
             "cost": None, "wall_s": None}
    try:
        for entry in (out.get_lm_usage() or {}).values():
            usage["prompt"] += entry.get("prompt_tokens", 0) or 0
            usage["completion"] += entry.get("completion_tokens", 0) or 0
        usage["total"] = usage["prompt"] + usage["completion"]
    except Exception:
        pass

    return label, ok, usage, str(getattr(out, "label", ""))


PREDICTORS = {"rlm": predict_rlm, "cot": predict_cot}


# ----------------------------------------------------------------------
# Run loop (concurrent, resumable)
# ----------------------------------------------------------------------
def out_dir(condition):
    slug = MODEL.split("/")[-1]
    path = os.path.join(RESULTS_DIR, f"rlm-study-{condition}", slug)
    os.makedirs(path, exist_ok=True)
    return path


def jsonl_path(condition, split, n, seed):
    return os.path.join(out_dir(condition),
                        f"records_{split}_n{n}_seed{seed}.jsonl")


def load_done(path):
    """Resume support: an agentic run over hundreds of instances will be
    interrupted, and re-paying for completed instances is the one cost we can
    trivially avoid."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a hard kill
            done[(rec["doc_id"], rec["nda_key"])] = rec
    return done


def run(args):
    # A self-served vLLM endpoint needs no key; OpenRouter does.
    if not API_KEY and not args.base_url and args.backend == "openrouter":
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Export it before running:\n"
            "    export OPENROUTER_API_KEY=sk-or-...\n"
            "(or pass --backend vllm --base-url http://localhost:8000/v1)"
        )

    data = load_split(args.split)
    instances = stratified_sample(list(iter_instances(data)), args.n, args.seed)

    path = jsonl_path(args.condition, args.split, args.n, args.seed)
    done = load_done(path) if args.resume else {}

    # Errored records are NOT done. Transient infrastructure failures (REPL
    # temp-dir races on Windows, upstream 429s) would otherwise be frozen into
    # the results as if the model had failed. A later successful record for the
    # same key supersedes the error, since load_done keeps the last occurrence.
    if args.retry_errors:
        done = {k: v for k, v in done.items() if not v.get("error")}

    todo = [i for i in instances if (i[0], i[1]) not in done]

    print(f"condition={args.condition}  model={MODEL}  split={args.split}")
    print(f"sampled={len(instances)}  already done={len(done)}  to run={len(todo)}")
    print(f"records -> {path}\n")

    predict = PREDICTORS[args.condition]
    write_lock = threading.Lock()
    t0 = time.time()
    counter = {"n": 0}

    def work(inst):
        doc_id, nda_key, hyp, contract, gold = inst
        started = time.time()
        try:
            label, ok, usage, raw = predict(args, hyp, contract)
            rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
                   "pred": label, "parsed_ok": ok, "usage": usage,
                   "elapsed_s": round(time.time() - started, 2),
                   "error": None}
            if args.save_raw:
                rec["raw"] = raw[:4000]
        except Exception as e:
            rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
                   "pred": None, "parsed_ok": False, "usage": None,
                   "elapsed_s": round(time.time() - started, 2),
                   "error": f"{type(e).__name__}: {e}"}
        return rec

    with open(path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(work, i): i for i in todo}
            for fut in as_completed(futures):
                rec = fut.result()
                with write_lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    counter["n"] += 1
                    k = counter["n"]
                tag = rec["error"] or f"{rec['pred']} (gold {rec['gold']})"
                rate = (time.time() - t0) / max(k, 1)
                eta = rate * (len(todo) - k) / 60
                print(f"  [{k}/{len(todo)}] {rec['nda_key']:>7} {tag}"
                      f"   ~{eta:.0f}m left")

    report(args)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def summarize(records):
    """Metrics for one condition. Returns None if nothing is scoreable."""
    ok = [r for r in records if r.get("error") is None and r.get("pred")]
    if not ok:
        return None

    y_true = [r["gold"] for r in ok]
    y_pred = [r["pred"] for r in ok]
    usages = [r["usage"] for r in ok if r.get("usage")]
    n = len(ok)

    def avg(key):
        vals = [u.get(key) or 0 for u in usages]
        return sum(vals) / len(vals) if vals else 0.0

    costs = [u.get("cost") for u in usages if u.get("cost") is not None]

    return {
        "n_scored": n,
        "n_errored": len(records) - n,
        "n_parse_fail": sum(1 for r in ok if not r.get("parsed_ok")),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS,
                             average="macro", zero_division=0),
        "per_class_f1": dict(zip(LABELS, f1_score(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0))),
        "report": classification_report(y_true, y_pred, labels=LABELS,
                                        zero_division=0),
        "avg_calls": avg("calls"),
        "avg_prompt": avg("prompt"),
        "avg_completion": avg("completion"),
        "avg_total": avg("total"),
        "total_tokens": sum((u.get("total") or 0) for u in usages),
        "avg_cost": (sum(costs) / len(costs)) if costs else None,
        "total_cost": sum(costs) if costs else None,
        "avg_wall": sum(r["elapsed_s"] for r in ok) / n,
        "y_true": y_true,
        "y_pred": y_pred,
        "per_hyp": _per_hypothesis(ok),
    }


def _per_hypothesis(ok):
    per = defaultdict(lambda: {"t": [], "p": []})
    for r in ok:
        per[r["nda_key"]]["t"].append(r["gold"])
        per[r["nda_key"]]["p"].append(r["pred"])
    out = {}
    for k, v in per.items():
        try:
            out[k] = (len(v["t"]), balanced_accuracy_score(v["t"], v["p"]))
        except ValueError:
            out[k] = (len(v["t"]), float("nan"))
    return out


def _block(name, s):
    lines = [f"--- {name} " + "-" * (56 - len(name))]
    if s is None:
        lines.append("  no scoreable records")
        return lines
    lines += [
        f"  Scored / errored / parse-fail : {s['n_scored']} / "
        f"{s['n_errored']} / {s['n_parse_fail']}",
        f"  Balanced accuracy             : {s['bal_acc']:.4f}",
        f"  Macro F1                      : {s['macro_f1']:.4f}",
    ]
    for lab, f in s["per_class_f1"].items():
        lines.append(f"    F1 [{lab:13}]          : {f:.4f}")
    lines += [
        f"  Avg LLM calls / instance      : {s['avg_calls']:.2f}",
        f"  Avg prompt tokens             : {s['avg_prompt']:.1f}",
        f"  Avg completion tokens         : {s['avg_completion']:.1f}",
        f"  Avg TOTAL tokens (whole tree) : {s['avg_total']:.1f}",
        f"  Avg wall-clock                : {s['avg_wall']:.1f}s",
    ]
    if s["avg_cost"] is not None:
        lines.append(f"  Avg cost / instance           : ${s['avg_cost']:.5f}")
        lines.append(f"  Total cost                    : ${s['total_cost']:.4f}")
    return lines


def report(args):
    conditions = ["rlm", "cot"] if args.compare else [args.condition]
    loaded = {}
    for cond in conditions:
        recs = list(load_done(
            jsonl_path(cond, args.split, args.n, args.seed)).values())
        loaded[cond] = (recs, summarize(recs) if recs else None)

    L = ["=" * 64,
         "ContractNLI — RLM scaffold vs budget-matched CoT",
         "=" * 64,
         f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"Model     : {MODEL}  (identical across conditions)",
         f"Split     : {args.split} | stratified n={args.n} | seed={args.seed}",
         f"RLM caps  : depth={args.max_depth} iters={args.max_iterations} "
         f"budget=${args.max_budget}",
         ""]

    for cond in conditions:
        recs, s = loaded[cond]
        L += _block(cond.upper(), s) + [""]

    # Paired comparison on the intersection of completed instances.
    if args.compare and all(loaded[c][1] for c in ("rlm", "cot")):
        r, c = loaded["rlm"][1], loaded["cot"][1]
        rmap = {(x["doc_id"], x["nda_key"]): x for x in loaded["rlm"][0]
                if x.get("pred") and not x.get("error")}
        cmap = {(x["doc_id"], x["nda_key"]): x for x in loaded["cot"][0]
                if x.get("pred") and not x.get("error")}
        shared = sorted(set(rmap) & set(cmap))

        L += ["=" * 64, "HEAD-TO-HEAD", "=" * 64]
        L.append(f"  Balanced accuracy : RLM {r['bal_acc']:.4f}  vs  "
                 f"CoT {c['bal_acc']:.4f}   (Δ {r['bal_acc']-c['bal_acc']:+.4f})")
        L.append(f"  Macro F1          : RLM {r['macro_f1']:.4f}  vs  "
                 f"CoT {c['macro_f1']:.4f}   (Δ {r['macro_f1']-c['macro_f1']:+.4f})")
        if c["avg_total"]:
            L.append(f"  Tokens / instance : RLM {r['avg_total']:.0f}  vs  "
                     f"CoT {c['avg_total']:.0f}   "
                     f"({r['avg_total']/c['avg_total']:.1f}x)")
        L.append(f"  LLM calls / inst  : RLM {r['avg_calls']:.2f}  vs  "
                 f"CoT {c['avg_calls']:.2f}")
        L.append(f"  Wall-clock / inst : RLM {r['avg_wall']:.1f}s  vs  "
                 f"CoT {c['avg_wall']:.1f}s")

        if shared:
            # Paired accuracy on exactly the same instances, plus a McNemar
            # table: the b/c cells are what a significance test needs, and
            # they show whether the two scaffolds fail on the SAME items.
            rc = sum(rmap[k]["pred"] == rmap[k]["gold"] for k in shared)
            cc = sum(cmap[k]["pred"] == cmap[k]["gold"] for k in shared)
            b = sum(1 for k in shared
                    if rmap[k]["pred"] == rmap[k]["gold"]
                    and cmap[k]["pred"] != cmap[k]["gold"])
            cq = sum(1 for k in shared
                     if rmap[k]["pred"] != rmap[k]["gold"]
                     and cmap[k]["pred"] == cmap[k]["gold"])
            L += ["",
                  f"  Paired on {len(shared)} shared instances:",
                  f"    plain accuracy  : RLM {rc/len(shared):.4f}  vs  "
                  f"CoT {cc/len(shared):.4f}",
                  f"    RLM right / CoT wrong : {b}",
                  f"    RLM wrong / CoT right : {cq}"]
            try:
                from statsmodels.stats.contingency_tables import mcnemar
                p = mcnemar([[0, b], [cq, 0]], exact=True).pvalue
                L.append(f"    McNemar exact p       : {p:.4f}")
            except Exception:
                L.append("    (pip install statsmodels for the McNemar p-value)")

    # Per-hypothesis breakdown, side by side where available.
    L += ["", "=" * 64, "PER-HYPOTHESIS BALANCED ACCURACY", "=" * 64]
    keys = sorted({k for c in conditions if loaded[c][1]
                   for k in loaded[c][1]["per_hyp"]},
                  key=lambda x: int(x.split("-")[1]))
    L.append("  " + "hyp".rjust(8) + "".join(c.rjust(14) for c in conditions))
    for k in keys:
        row = "  " + k.rjust(8)
        for c in conditions:
            s = loaded[c][1]
            if s and k in s["per_hyp"]:
                n_k, ba = s["per_hyp"][k]
                row += f"{ba:.3f} (n={n_k})".rjust(14)
            else:
                row += "-".rjust(14)
        L.append(row)

    for cond in conditions:
        if loaded[cond][1]:
            L += ["", f"--- {cond.upper()} classification report ---",
                  loaded[cond][1]["report"]]

    text = "\n".join(L)
    print("\n" + text)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "compare" if args.compare else args.condition
    dest = os.path.join(out_dir(conditions[0]),
                        f"results_{tag}_{args.split}_n{args.n}_{stamp}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved report -> {dest}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["rlm", "cot"], default="rlm")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=150,
                    help="stratified subsample size (same sample per seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--retry-errors", action="store_true", default=True,
                    help="on resume, re-run instances whose last record errored")
    ap.add_argument("--no-retry-errors", dest="retry_errors",
                    action="store_false")
    ap.add_argument("--save-raw", action="store_true", default=True,
                    help="store the response text for qualitative analysis")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="report rlm and cot side by side")

    # RLM budget caps
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--max-errors", type=int, default=5)
    ap.add_argument("--max-budget", type=float, default=0.10,
                    help="USD ceiling per instance")
    ap.add_argument("--max-timeout", type=float, default=300.0)
    ap.add_argument("--max-concurrent-subcalls", type=int, default=4)
    ap.add_argument("--environment", default="local",
                    choices=["local", "ipython", "docker", "e2b"])
    ap.add_argument("--backend", default=os.environ.get("RLM_BACKEND", "openrouter"),
                    choices=["openrouter", "openai", "vllm"],
                    help="use vllm/openai + --base-url to drive a self-served "
                         "checkpoint instead of OpenRouter")
    ap.add_argument("--base-url", default=os.environ.get("RLM_BASE_URL"),
                    help="e.g. http://localhost:8000/v1 for a local vLLM server")

    args = ap.parse_args()

    if args.report_only or args.compare:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
