"""
run_matrix.py
=============
Runs the (model x condition) matrix for the ContractNLI leg of the long-context
claim-verification study, concurrently and resumably.

Conditions reuse the predictors in contractnli_zeroshot.py, so prompts are
identical to the existing single-model runs -- this file only adds a thread
pool, a shared stratified sample, and one combined report.

  zeroshot  raw single call, whole contract in prompt, thinking off
  dspy      DSPy Predict  (same signature, DSPy-formatted prompt)
  cot       DSPy ChainOfThought (prompt-elicited CoT, thinking off)
  rag       top-k span retrieval, only retrieved spans in prompt
  agentic_rag  model issues its own search queries over several turns

Every condition scores the SAME stratified sample (default n=150, seed 0) --
the same one run_rlm_contractnli.py uses -- so the RLM arm drops straight into
the table and every comparison is paired.

Endpoint is whatever contractnli_zeroshot.BASE_URL points at, so the same
command evaluates OpenRouter models or a local vLLM server:

    # OpenRouter
    export OPENROUTER_API_KEY=sk-or-...
    python run_matrix.py --models qwen/qwen3-8b meta-llama/llama-3.2-3b-instruct

    # self-hosted (Colab A100 + vLLM)
    python run_matrix.py --base-url http://localhost:8000/v1 --models Qwen/Qwen3-4B

    python run_matrix.py --report-only        # rebuild the matrix table
"""

import os
import sys
import csv
import json
import hashlib
import time
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import balanced_accuracy_score, f1_score

import contractnli_zeroshot as cz
from run_rlm_contractnli import (
    load_split, iter_instances, stratified_sample, parse_label, load_done,
)

CONDITIONS = ["zeroshot", "dspy", "cot", "rag", "agentic_rag"]
LABELS = ["Entailment", "Contradiction", "NotMentioned"]

# DSPy configures a single global LM, so programs cannot be built concurrently
# for different models. Serialise construction; the calls themselves still run
# in parallel.
_dspy_lock = threading.Lock()
_dspy_cache = {}


def slug(model):
    # Ollama tags ("qwen3:4b") and HF ids ("Qwen/Qwen3-4B") both have to become
    # a legal directory name -- ":" is illegal on Windows.
    return model.split("/")[-1].lower().replace(":", "-")


# Set from --thinking. Keeps the two arms in separate directories so a
# thinking-on run can never overwrite the thinking-off results.
THINKING = False


def _prefix():
    return "matrix-think" if THINKING else "matrix"


def out_dir(condition, model):
    p = os.path.join(cz.RESULTS_DIR, f"{_prefix()}-{condition}",
                     slug(model))
    os.makedirs(p, exist_ok=True)
    return p


def jsonl_path(condition, model, split, n, seed):
    return os.path.join(out_dir(condition, model),
                        f"records_{split}_n{n}_seed{seed}.jsonl")


def sample_fingerprint(instances):
    """Stable hash of a sample's (doc_id, nda_key) set.

    Printed on every run so a Colab/remote run can be proven to have scored the
    same instances as the local one, rather than merely assumed to.
    """
    keys = sorted((str(i[0]), i[1]) for i in instances)
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:16]


def load_manifest(path, data):
    """Pin the sample to an explicit manifest CSV instead of re-deriving it.

    Removes any dependence on Python version, dict ordering, or the RNG being
    identical across machines.
    """
    want = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            want.append((row["doc_id"], row["nda_key"]))
    wantset = set(want)

    by_key = {(str(i[0]), i[1]): i for i in iter_instances(data)}
    missing = [k for k in wantset if k not in by_key]
    if missing:
        raise SystemExit(f"manifest references {len(missing)} instances absent "
                         f"from this split, e.g. {missing[:3]}")
    return [by_key[k] for k in want]


def build_program(condition, model):
    """Build the DSPy program for a condition+model.

    MUST be called from the main thread: dspy.configure() binds settings to the
    calling thread, and a worker calling it raises
    "dspy.settings can only be changed by the thread that initially configured it".
    Calling an already-configured program from workers is fine.
    """
    cz.MODEL = model
    # The harness encodes thinking in the condition name:
    #   dspy -> Predict         thinking off | "reasoning"  -> Predict + thinking
    #   cot  -> ChainOfThought  thinking off | "cot_think"  -> CoT + thinking
    if condition == "dspy":
        dspy_cond = "reasoning" if THINKING else "zeroshot"
    else:
        dspy_cond = "cot_think" if THINKING else "cot"
    return cz.build_dspy_program(condition=dspy_cond)


def predict(condition, model, hyp, contract, doc, program=None):
    """Dispatch to the shared predictors, returning (label, ok, usage).

    All four conditions route through contractnli_zeroshot, so the usage dict
    always carries prompt / completion / reasoning / total / think_tokens --
    reasoning being what the provider reports separately, think_tokens being
    the provider-independent count of tokens inside <think>...</think>.
    """
    cz.MODEL = model            # module-level global the predictors read

    if condition == "zeroshot":
        label, usage = cz.predict_raw(hyp, contract, thinking_on=THINKING)
    elif condition == "rag":
        label, usage = cz.predict_rag(hyp, doc, thinking_on=THINKING)
    elif condition == "agentic_rag":
        # Model issues its own search queries over several turns instead of one
        # fixed top-k retrieval. Usage additionally carries `steps`.
        label, usage = cz.predict_agentic_rag(hyp, doc, thinking_on=THINKING)
    else:
        label, usage = cz.predict_dspy(program, hyp, contract)

    # cz.normalize_label has already collapsed the text to a label, so this
    # re-parse confirms the label is well-formed rather than detecting the
    # model's own failure to answer.
    label, ok = parse_label(label)
    return label, ok, usage


def run_one(condition, model, instances, args):
    path = jsonl_path(condition, model, args.split, args.n, args.seed)
    done = load_done(path) if args.resume else {}
    if args.retry_errors:
        done = {k: v for k, v in done.items() if not v.get("error")}
    todo = [i for i in instances if (i[0], i[1]) not in done]

    print(f"\n=== {condition:9} | {model:40} | {len(todo)} to run "
          f"({len(done)} cached)")
    if not todo:
        return

    # Built here, on the main thread, for the DSPy thread-affinity reason above.
    program = (build_program(condition, model)
               if condition in ("dspy", "cot") else None)

    lock = threading.Lock()
    t0 = time.time()
    count = {"n": 0}

    def work(inst):
        doc_id, nda_key, hyp, contract, gold = inst
        doc = DOC_BY_ID[doc_id]
        started = time.time()
        try:
            label, ok, usage = predict(condition, model, hyp, contract, doc,
                                       program)
            return {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
                    "pred": label, "parsed_ok": ok, "usage": usage,
                    "elapsed_s": round(time.time() - started, 2), "error": None}
        except Exception as e:
            return {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
                    "pred": None, "parsed_ok": False, "usage": None,
                    "elapsed_s": round(time.time() - started, 2),
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    with open(path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(work, i) for i in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    count["n"] += 1
                    k = count["n"]
                if k % 25 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"    {k}/{len(todo)}  {el:.0f}s elapsed  "
                          f"~{el/k*(len(todo)-k):.0f}s left")

    n_err = sum(1 for f in futs if f.result().get("error"))
    if n_err:
        print(f"    WARNING: {n_err} errored")


def summarize(recs):
    ok = [r for r in recs if not r.get("error") and r.get("pred")]
    if not ok:
        return None
    yt = [r["gold"] for r in ok]
    yp = [r["pred"] for r in ok]
    us = [r["usage"] for r in ok if r.get("usage")]

    def vals_of(field):
        return [(u.get(field) or 0) for u in us]

    def avg(field):
        vals = vals_of(field)
        return sum(vals) / len(vals) if vals else 0

    def spread(field):
        """(mean, min, max, median) — the mean alone hides the long tail that
        agentic and thinking conditions produce, where a handful of instances
        cost an order of magnitude more than the typical one."""
        vals = sorted(vals_of(field))
        if not vals:
            return (0, 0, 0, 0)
        mid = len(vals) // 2
        med = (vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)
        return (sum(vals) / len(vals), vals[0], vals[-1], med)

    out = {
        "n": len(ok),
        "n_err": len(recs) - len(ok),
        "n_parsefail": sum(1 for r in ok if not r.get("parsed_ok")),
        "bal_acc": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, labels=LABELS, average="macro",
                             zero_division=0),
        "f1": dict(zip(LABELS, f1_score(yt, yp, labels=LABELS, average=None,
                                        zero_division=0))),
        "avg_tokens": avg("total"),
        "avg_prompt": avg("prompt"),
        "avg_completion": avg("completion"),
        "avg_reasoning": avg("reasoning"),
        "avg_think": avg("think_tokens"),
        "n_with_think": sum(1 for u in us if (u.get("think_tokens") or 0) > 0),
        "avg_s": sum(r["elapsed_s"] for r in ok) / len(ok),
    }
    for field, key in (("total", "tokens"), ("prompt", "prompt"),
                       ("completion", "completion"), ("reasoning", "reasoning"),
                       ("think_tokens", "think"), ("steps", "steps"),
                       ("retrieved_spans", "retrieved"),
                       ("cost_usd", "cost"), ("cached_prompt", "cached")):
        m, lo, hi, med = spread(field)
        out[f"mean_{key}"] = m
        out[f"min_{key}"] = lo
        out[f"max_{key}"] = hi
        out[f"med_{key}"] = med
    return out


def report(args, models):
    rows = {}
    for m in models:
        for c in CONDITIONS:
            recs = list(load_done(
                jsonl_path(c, m, args.split, args.n, args.seed)).values())
            if recs:
                rows[(m, c)] = summarize(recs)

    # The RLM arm lives in its own directory but uses the same sample.
    # It was only ever run thinking-off, so it belongs to that table only.
    rlm_path = os.path.join(cz.RESULTS_DIR, "rlm-study-rlm", "qwen3-8b",
                            f"records_{args.split}_n{args.n}_seed{args.seed}.jsonl")
    if not THINKING and os.path.exists(rlm_path):
        recs = list(load_done(rlm_path).values())
        if recs:
            rows[("qwen/qwen3-8b", "rlm")] = summarize(recs)

    conds = CONDITIONS + (["rlm"] if any(c == "rlm" for _, c in rows) else [])
    L = ["=" * 78,
         "ContractNLI — model x condition matrix",
         "=" * 78,
         f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"Split     : {args.split} | stratified n={args.n} | seed={args.seed}",
         f"Endpoint  : {cz.BASE_URL}",
         f"Thinking  : {'ON (native reasoning enabled)' if THINKING else 'OFF'}",
         "",
         "BALANCED ACCURACY", "-" * 78,
         "  " + "model".ljust(34) + "".join(c.rjust(13) for c in conds)]

    for m in models:
        line = "  " + slug(m).ljust(34)
        for c in conds:
            s = rows.get((m, c))
            line += (f"{s['bal_acc']:.3f}".rjust(13) if s else "-".rjust(13))
        L.append(line)

    L += ["", "MACRO F1", "-" * 78,
          "  " + "model".ljust(34) + "".join(c.rjust(13) for c in conds)]
    for m in models:
        line = "  " + slug(m).ljust(34)
        for c in conds:
            s = rows.get((m, c))
            line += (f"{s['macro_f1']:.3f}".rjust(13) if s else "-".rjust(13))
        L.append(line)

    for title, field in (("AVG TOTAL TOKENS / INSTANCE", "avg_tokens"),
                         ("AVG GENERATED (COMPLETION) TOKENS / INSTANCE",
                          "avg_completion"),
                         ("AVG REASONING TOKENS / INSTANCE", "avg_reasoning")):
        L += ["", title, "-" * 78,
              "  " + "model".ljust(34) + "".join(c.rjust(13) for c in conds)]
        for m in models:
            line = "  " + slug(m).ljust(34)
            for c in conds:
                s = rows.get((m, c))
                line += (f"{s[field]:.0f}".rjust(13) if s else "-".rjust(13))
            L.append(line)

    L += ["", "PER-RUN DETAIL", "-" * 78]
    for m in models:
        for c in conds:
            s = rows.get((m, c))
            if not s:
                continue
            L.append(f"  {slug(m):22} {c:9} n={s['n']:4} err={s['n_err']:3} "
                     f"bal={s['bal_acc']:.3f} f1={s['macro_f1']:.3f} "
                     f"{s['avg_s']:5.1f}s")
            L.append("      " + "  ".join(
                f"F1[{k[:4]}]={v:.3f}" for k, v in s["f1"].items()))
            L.append(f"      tokens: prompt={s['avg_prompt']:.0f} "
                     f"completion={s['avg_completion']:.0f} "
                     f"reasoning={s['avg_reasoning']:.0f} "
                     f"think={s['avg_think']:.0f} "
                     f"(<think> in {s['n_with_think']}/{s['n']})")
            # mean / min / max / median per token field. The spread matters for
            # the cost story: a mean of 1,600 tokens reads very differently if
            # the worst instance cost 20,000.
            for key, lab in (("tokens", "total"), ("prompt", "prompt"),
                             ("completion", "completion"),
                             ("reasoning", "reasoning"), ("think", "<think>"),
                             ("steps", "steps"), ("retrieved", "ret.spans"),
                             ("cached", "cachedtok")):
                if s.get(f"max_{key}", 0) == 0 and s.get(f"mean_{key}", 0) == 0:
                    continue          # field absent for this condition
                L.append(f"        {lab:11} mean={s[f'mean_{key}']:9.1f} "
                         f"med={s[f'med_{key}']:8.1f} "
                         f"min={s[f'min_{key}']:7.0f} "
                         f"max={s[f'max_{key}']:8.0f}")
            # Actual USD charged, straight from the provider. Printed to 6dp
            # per instance plus the run total, which is what a cost table in
            # the paper actually needs.
            if s.get("mean_cost", 0) > 0:
                L.append(f"        {'cost USD':11} mean={s['mean_cost']:9.6f} "
                         f"med={s['med_cost']:8.6f} "
                         f"min={s['min_cost']:7.6f} "
                         f"max={s['max_cost']:8.6f}  "
                         f"RUN TOTAL=${s['mean_cost'] * s['n']:.4f}")

    text = "\n".join(L)
    print("\n" + text)
    dest = os.path.join(cz.RESULTS_DIR,
                        f"{_prefix()}_{args.split}_n{args.n}_"
                        f"{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen/qwen3-8b"])
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint (default: OpenRouter)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--retry-errors", action="store_true", default=True)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--manifest", default=None,
                    help="CSV of doc_id,nda_key to score instead of deriving "
                         "the sample from --n/--seed (guarantees an identical "
                         "sample across machines)")
    ap.add_argument("--thinking", action="store_true",
                    help="run the thinking-ON arm (native reasoning enabled); "
                         "writes to results/matrix-think-* so it never "
                         "overwrites the thinking-off results")
    args = ap.parse_args()

    global THINKING
    THINKING = args.thinking

    if args.base_url:
        cz.BASE_URL = args.base_url
    if args.api_key:
        cz.API_KEY = args.api_key
    if (not args.report_only and not cz.API_KEY
            and "openrouter.ai" in cz.BASE_URL):
        sys.exit("OPENROUTER_API_KEY not set (or pass --base-url for local vLLM)")

    data = load_split(args.split)
    global DOC_BY_ID
    DOC_BY_ID = {d["id"]: d for d in data["documents"]}
    if args.manifest:
        instances = load_manifest(args.manifest, data)
        print(f"sample pinned to manifest: {args.manifest}")
    else:
        instances = stratified_sample(list(iter_instances(data)),
                                      args.n, args.seed)
    print(f"sample: {len(instances)} instances | "
          f"fingerprint {sample_fingerprint(instances)}")

    if not args.report_only:
        for model in args.models:
            for cond in args.conditions:
                try:
                    run_one(cond, model, instances, args)
                except Exception as e:
                    print(f"  !! {cond}/{model} aborted: "
                          f"{type(e).__name__}: {e}")

    report(args, args.models)


if __name__ == "__main__":
    main()
