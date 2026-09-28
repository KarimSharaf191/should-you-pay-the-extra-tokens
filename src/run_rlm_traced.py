"""
run_rlm_traced.py
=================
Fully instrumented RLM evaluation on ContractNLI.

WHY THIS EXISTS
---------------
`result.usage_summary` from the rlm library does NOT include sub-call tokens.
`RLM._subcall` builds a brand-new client via `get_client()` for the plain-LM
path (the path always taken when max_depth=1), and the top-level summary only
aggregates the RLM's own `lm_handler` clients. So an uninstrumented run
reports ROOT-LEVEL TOKENS ONLY and understates the true cost.

This runner wraps `rlm.core.rlm.get_client` so every LLM call made anywhere in
the tree is recorded: role (root vs subcall), model, prompt/completion tokens,
cost, latency, and the size of the prompt each sub-call actually received.

Attribution: within one `completion()` the library creates the root client
first (rlm.py:234) and sub-call clients afterwards (rlm.py:739), so the first
client seen on a thread is the root and the rest are sub-calls. Calls that
arrive on an unexpected thread land in an `orphan` bucket which is reported --
if that count is non-zero the attribution is unreliable and the run says so.

    export OPENROUTER_API_KEY=sk-or-...
    python run_rlm_traced.py --model qwen/qwen3-8b --n 150
    python run_rlm_traced.py --model qwen/qwen3-coder-30b-a3b-instruct --n 150
    python run_rlm_traced.py --report-only
"""

import os
import json
import time
import random
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_rlm_contractnli import (
    load_split, iter_instances, stratified_sample, parse_label, load_done,
    ROOT_PROMPT, LABELS,
)

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
API_KEY = os.environ.get("OPENROUTER_API_KEY")


# ----------------------------------------------------------------------
# Call tracing
# ----------------------------------------------------------------------
_tl = threading.local()
_orphans = {"n": 0}
_orphan_lock = threading.Lock()
FORCED = False
ABLATED = False
_global_ledger = {"rows": None}
_global_lock = threading.Lock()
_retries = {"n": 0}
_retry_lock = threading.Lock()
MAX_RETRIES = 5

# One tokenizer across every benchmark and condition. cl100k_base is NOT Qwen's
# tokenizer, so these counts will not match the provider's exactly -- that is
# accepted: a single consistent proxy is what makes root-vs-subcall and
# cross-benchmark comparisons valid, and the provider's own numbers are recorded
# beside it so the discrepancy stays visible rather than hidden.
try:
    import tiktoken as _tiktoken
    _TRACE_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _TRACE_ENC = None


def trace_ntok(s):
    """tiktoken length of a prompt, which may be a str or a message list."""
    if not s:
        return 0
    if isinstance(s, list):
        s = chr(10).join(str(m.get("content", "")) if isinstance(m, dict) else str(m)
                         for m in s)
    s = str(s)
    if _TRACE_ENC is not None:
        return len(_TRACE_ENC.encode(s))
    return len(s.split())


def _ledger():
    return getattr(_tl, "ledger", None)


class _TracingClient:
    """Transparent proxy that records every completion() through this client."""

    def __init__(self, inner, seq):
        self._inner = inner
        self._seq = seq

    def __getattr__(self, name):          # forward everything else untouched
        return getattr(self._inner, name)

    def completion(self, prompt, *a, **kw):
        t0 = time.perf_counter()
        # The rlm library does not retry. A single upstream 429 anywhere in a
        # trajectory otherwise discards the whole verification, so absorb them
        # here with exponential backoff + jitter.
        attempt, resp = 0, None
        while True:
            try:
                resp = self._inner.completion(prompt, *a, **kw)
                break
            except Exception as e:
                msg = str(e)
                transient = ("429" in msg or "rate" in msg.lower()
                             or "502" in msg or "503" in msg
                             or "overloaded" in msg.lower())
                if not transient or attempt >= MAX_RETRIES:
                    raise
                delay = min(2.0 * (2 ** attempt), 30.0) + random.uniform(0, 1.5)
                with _retry_lock:
                    _retries["n"] += 1
                time.sleep(delay)
                attempt += 1
        dt = time.perf_counter() - t0

        # Some models (observed on qwen3-coder) return message.content = null.
        # The library regexes the response for code blocks and crashes on None,
        # discarding the whole verification, so normalise it to an empty string.
        if resp is None:
            resp = ""

        rec = {
            # seq 0 is the root's lm_handler client; later ones are sub-calls.
            "role": "root" if self._seq == 0 else "subcall",
            "seq": self._seq,
            "model": getattr(self._inner, "model_name", None),
            "latency_s": round(dt, 3),
            "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
            "response_chars": len(resp) if isinstance(resp, str) else None,
            # Counted here, at call time, because the prompt text is not kept.
            "prompt_tiktoken": trace_ntok(prompt),
            "response_tiktoken": trace_ntok(resp),
            "has_think": "</think>" in str(resp or ""),
            "prompt_tokens": 0, "completion_tokens": 0, "cost": None,
        }
        try:
            u = self._inner.get_last_usage()
            rec["prompt_tokens"] = getattr(u, "total_input_tokens", 0) or 0
            rec["completion_tokens"] = getattr(u, "total_output_tokens", 0) or 0
            rec["cost"] = getattr(u, "total_cost", None)
        except Exception:
            pass

        # llm_query does NOT go through _subcall. It is sent over a socket to
        # the LMHandler, which serves it on its own ThreadingTCPServer thread,
        # so a thread-local ledger never sees it. Fall back to the global
        # ledger and classify by prompt type: the root's turns carry a message
        # list, whereas llm_query passes a plain string.
        if not isinstance(prompt, str):
            rec["role"] = "root"
        else:
            rec["role"] = "subcall"

        lg = _ledger()
        if lg is None:
            with _global_lock:
                if _global_ledger["rows"] is None:
                    with _orphan_lock:
                        _orphans["n"] += 1
                else:
                    _global_ledger["rows"].append(rec)
        else:
            lg.append(rec)
        return resp


def install_tracing():
    import rlm.core.rlm as rlm_core
    if getattr(rlm_core, "_tracing_installed", False):
        return
    real = rlm_core.get_client

    def traced(backend, backend_kwargs):
        seq = getattr(_tl, "seq", 0)
        _tl.seq = seq + 1
        return _TracingClient(real(backend, backend_kwargs), seq)

    rlm_core.get_client = traced
    rlm_core._tracing_installed = True


# ----------------------------------------------------------------------
# RLM driver
# ----------------------------------------------------------------------
def forced_system_prompt(threshold_chars):
    """The stock RLM prompt with its own chunking threshold lowered.

    The library's prompt tells the root model:
        "REPL outputs over ~20K characters are truncated, so for longer
         payloads slice `context` and pass slices through `llm_query` rather
         than `print`-ing them whole."

    Nothing enforces that 20K -- it is advisory prose. ContractNLI contracts
    average ~12.1K characters, comfortably under it, so the scaffold's own
    policy sanctions `print(context)` and recursion is never invoked.

    Lowering the stated threshold below the mean contract length makes the
    scaffold's own rule mandate decomposition. This is a ONE-PARAMETER change
    to the authors' prompt -- everything else is untouched -- so the resulting
    condition isolates the cost of fragmentation rather than confounding it
    with a rewritten prompt.
    """
    from rlm.utils.prompts import RLM_SYSTEM_PROMPT
    old = "REPL outputs over ~20K characters are truncated"
    new = f"REPL outputs over ~{threshold_chars // 1000}K characters are truncated"
    if old not in RLM_SYSTEM_PROMPT:
        raise SystemExit("threshold sentence not found; library prompt changed")
    return RLM_SYSTEM_PROMPT.replace(old, new, 1)


def ablate_escape_clause():
    """Remove the orchestrator prompt's direct-read escape clause.

    The library appends ORCHESTRATOR_ADDENDUM to every system prompt. It pushes
    the root model to delegate ("act as an orchestrator, not a solver"; "push
    every long-context operation ... into llm_query") but then grants an
    explicit exemption:

        "(Conversely: if a Python keyword / regex search over `context` would
         already pin the answer, or if a single visible passage already
         contains it, just read it directly -- sub-LMs are for when the raw
         text won't fit ...)"

    Because ContractNLI contracts DO fit, that exemption applies and recursion
    is never invoked. Deleting this one sentence -- and nothing else -- keeps
    every instruction to delegate while removing the licence to skip it, so any
    change in sub-call behaviour is attributable to this clause alone.

    build_rlm_system_prompt resolves ORCHESTRATOR_ADDENDUM as a module global at
    call time, so rebinding it here takes effect for subsequent RLM builds.
    """
    import rlm.utils.prompts as P
    clause_start = "(Conversely:"
    i = P.ORCHESTRATOR_ADDENDUM.find(clause_start)
    if i == -1:
        # Already ablated -- this runs once per worker thread, so it must be
        # idempotent rather than treating a second call as a library change.
        if getattr(P, "_escape_clause_ablated", False):
            return P.ORCHESTRATOR_ADDENDUM
        raise SystemExit("escape clause not found; library prompt changed")
    P._escape_clause_ablated = True
    j = P.ORCHESTRATOR_ADDENDUM.find(")", P.ORCHESTRATOR_ADDENDUM.find(
        "semantic interpretation"))
    P.ORCHESTRATOR_ADDENDUM = (P.ORCHESTRATOR_ADDENDUM[:i]
                               + P.ORCHESTRATOR_ADDENDUM[j + 1:]).replace(
                                   "  ", " ")
    return P.ORCHESTRATOR_ADDENDUM


def build_rlm(args, model):
    from rlm import RLM
    from rlm.logger.rlm_logger import RLMLogger
    if args.ablate_escape:
        ablate_escape_clause()
        return RLM(
            backend="openrouter",
            backend_kwargs={"model_name": model, "api_key": API_KEY},
            environment="local",
            depth=0,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            max_errors=args.max_errors,
            max_budget=args.max_budget,
            max_timeout=args.max_timeout,
            max_concurrent_subcalls=1,
            logger=RLMLogger(),
            verbose=False,
        )
    if args.force_subcalls:
        return RLM(
            backend="openrouter",
            backend_kwargs={"model_name": model, "api_key": API_KEY},
            environment="local",
            depth=0,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            max_errors=args.max_errors,
            max_budget=args.max_budget,
            max_timeout=args.max_timeout,
            max_concurrent_subcalls=1,
            custom_system_prompt=forced_system_prompt(args.chunk_threshold),
            # Lowering the stated threshold alone is insufficient: the model
            # complies by printing SMALLER slices and still reads them itself,
            # because a print limit never requires delegation. An explicit
            # mandate is needed to actually exercise the recursion path.
            user_prologue=(
                f"MANDATORY STRATEGY for this task. Do not print the contract "
                f"and do not read it yourself. Instead:\n"
                f"1. Split `context` into consecutive chunks of at most "
                f"{args.chunk_threshold} characters.\n"
                f"2. Send EVERY chunk to a sub-LLM with llm_query (or "
                f"llm_query_batched), asking whether that chunk supports, "
                f"contradicts, or does not mention the hypothesis.\n"
                f"3. Combine the sub-LLM answers in code and only then decide "
                f"the final label.\n"
                f"You must call llm_query at least once per chunk."
            ),
            logger=RLMLogger(),
            verbose=False,
        )
    return RLM(
        backend="openrouter",
        backend_kwargs={"model_name": model, "api_key": API_KEY},
        environment="local",
        depth=0,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_errors=args.max_errors,
        max_budget=args.max_budget,
        max_timeout=args.max_timeout,
        # Keep sub-calls on the calling thread so ledger attribution holds.
        max_concurrent_subcalls=1,
        logger=RLMLogger(),
        verbose=False,
    )


def _thread_rlm(args, model):
    r = getattr(_tl, "rlm", None)
    if r is None:
        r = build_rlm(args, model)
        _tl.rlm = r
    return r


def run_instance(args, model, hypothesis, contract):
    rlm = _thread_rlm(args, model)

    _tl.ledger = []
    _tl.seq = 0                      # reset so this completion's root is seq 0
    # Socket-served llm_query calls land here. Safe only at concurrency 1,
    # which the caller must enforce when sub-call attribution matters.
    with _global_lock:
        _global_ledger["rows"] = []
    t0 = time.perf_counter()
    result = rlm.completion(prompt=contract,
                            root_prompt=ROOT_PROMPT.format(hypothesis=hypothesis))
    wall = time.perf_counter() - t0
    calls = list(_tl.ledger)
    _tl.ledger = None
    with _global_lock:
        calls += (_global_ledger["rows"] or [])
        _global_ledger["rows"] = None

    label, ok = parse_label(getattr(result, "response", None))

    root = [c for c in calls if c["role"] == "root"]
    subs = [c for c in calls if c["role"] == "subcall"]

    def tot(rows, field):
        return sum((r.get(field) or 0) for r in rows)

    def cost(rows):
        c = [r["cost"] for r in rows if r.get("cost") is not None]
        return sum(c) if c else None

    # Root iterations from the attached logger (code the model ran, REPL output)
    iters = []
    meta = getattr(result, "metadata", None)
    if isinstance(meta, dict):
        for it in (meta.get("iterations") or []):
            blocks = it.get("code_blocks") or []
            iters.append({
                "response_chars": len(it.get("response") or ""),
                "n_code_blocks": len(blocks),
                "code_chars": sum(len(b.get("code") or "") for b in blocks),
                "stdout_chars": sum(
                    len(((b.get("result") or {}).get("stdout")) or "")
                    for b in blocks),
                "stderr_chars": sum(
                    len(((b.get("result") or {}).get("stderr")) or "")
                    for b in blocks),
                "had_error": any(
                    bool(((b.get("result") or {}).get("stderr")) or "")
                    for b in blocks),
                "iteration_time": it.get("iteration_time"),
            })

    usage = {
        # --- the number an uninstrumented run would have reported ---
        "root_calls": len(root),
        "root_prompt": tot(root, "prompt_tokens"),
        "root_completion": tot(root, "completion_tokens"),
        "root_cost": cost(root),
        # --- what it was missing ---
        "subcall_calls": len(subs),
        "subcall_prompt": tot(subs, "prompt_tokens"),
        "subcall_completion": tot(subs, "completion_tokens"),
        "subcall_cost": cost(subs),
        # --- true whole-tree totals ---
        "tree_calls": len(calls),
        "tree_prompt": tot(calls, "prompt_tokens"),
        "tree_completion": tot(calls, "completion_tokens"),
        "tree_total": tot(calls, "prompt_tokens") + tot(calls, "completion_tokens"),
        "tree_cost": cost(calls),
        # --- evidence budget: how much text each sub-call actually saw ---
        "subcall_prompt_chars": tot(subs, "prompt_chars"),
        "n_iterations": len(iters),
        "repl_stdout_chars": sum(i["stdout_chars"] for i in iters),
        "repl_errors": sum(1 for i in iters if i["had_error"]),
        "wall_s": round(wall, 2),
    }
    return label, ok, usage, calls, iters, getattr(result, "response", "")


# ----------------------------------------------------------------------
def out_dir(model):
    tree = ("rlm-traced-ablated" if ABLATED else
            "rlm-traced-forced" if FORCED else "rlm-traced")
    p = os.path.join(RESULTS_DIR, tree,
                     model.split("/")[-1].lower().replace(":", "-"))
    os.makedirs(p, exist_ok=True)
    return p


def jsonl_path(model, split, n, seed):
    return os.path.join(out_dir(model), f"records_{split}_n{n}_seed{seed}.jsonl")


def run(args):
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set")
    install_tracing()

    data = load_split(args.split)
    instances = stratified_sample(list(iter_instances(data)), args.n, args.seed)
    path = jsonl_path(args.model, args.split, args.n, args.seed)
    done = {k: v for k, v in load_done(path).items() if not v.get("error")} \
        if args.resume else {}
    todo = [i for i in instances if (i[0], i[1]) not in done]

    print(f"model={args.model} | {len(todo)} to run ({len(done)} cached)")
    print(f"-> {path}\n")
    if not todo:
        report(args)
        return

    lock = threading.Lock()
    t0 = time.time()
    cnt = {"n": 0, "err": 0}

    def work(inst):
        doc_id, nda_key, hyp, contract, gold = inst
        try:
            label, ok, usage, calls, iters, raw = run_instance(
                args, args.model, hyp, contract)
            rec = {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
                   "pred": label, "parsed_ok": ok, "usage": usage,
                   "error": None}
            if args.save_trace:
                rec["calls"] = calls
                rec["iterations"] = iters
                rec["raw"] = (raw or "")[:2000]
            return rec
        except Exception as e:
            return {"doc_id": doc_id, "nda_key": nda_key, "gold": gold,
                    "pred": None, "parsed_ok": False, "usage": None,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    with open(path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(work, i) for i in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    cnt["n"] += 1
                    if rec.get("error"):
                        cnt["err"] += 1
                    k = cnt["n"]
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"  {k}/{len(todo)}  err={cnt['err']}  "
                          f"{el:.0f}s  ~{el/k*(len(todo)-k):.0f}s left")

    print("")
    print(f"  transient errors absorbed by retry: {_retries['n']}")
    if _orphans["n"]:
        print(f"\n  WARNING: {_orphans['n']} LLM calls could not be attributed "
              f"to a thread ledger; sub-call totals may be undercounted.")
    report(args)


# ----------------------------------------------------------------------
def report(args):
    import statistics
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 classification_report)

    path = jsonl_path(args.model, args.split, args.n, args.seed)
    recs = list(load_done(path).values())
    ok = [r for r in recs if not r.get("error") and r.get("pred")]
    if not ok:
        print("no scoreable records")
        return

    yt = [r["gold"] for r in ok]
    yp = [r["pred"] for r in ok]
    us = [r["usage"] for r in ok if r.get("usage")]

    def m(f):
        v = [(u.get(f) or 0) for u in us]
        return statistics.mean(v) if v else 0

    def s(f):
        return sum((u.get(f) or 0) for u in us)

    L = []
    L.append("=" * 74)
    L.append(f"RLM TRACED — {args.model}"
             + ("  [ESCAPE-CLAUSE ABLATED]" if ABLATED else
                "  [FORCED SUB-CALLS]" if FORCED else "  [natural behaviour]"))
    L.append("=" * 74)
    L.append(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Caps      : depth={args.max_depth} iters={args.max_iterations}")
    L.append(f"Scored    : {len(ok)}   errored: {len(recs)-len(ok)}   "
             f"parse-fail: {sum(1 for r in ok if not r.get('parsed_ok'))}")
    L.append("")
    L.append(f"Balanced accuracy : {balanced_accuracy_score(yt, yp):.4f}")
    L.append(f"Macro F1          : {f1_score(yt, yp, labels=LABELS, average='macro', zero_division=0):.4f}")
    for lab, f in zip(LABELS, f1_score(yt, yp, labels=LABELS, average=None,
                                       zero_division=0)):
        L.append(f"  F1 [{lab:13}] : {f:.4f}")

    L.append("")
    L.append("-" * 74)
    L.append("TOKEN TRACE — where the tokens actually go (mean per verification)")
    L.append("-" * 74)
    L.append(f"{'':22}{'calls':>9}{'prompt':>11}{'completion':>12}{'total':>11}")
    for tag, pre in (("Root (orchestration)", "root"),
                     ("Sub-calls (evidence)", "subcall")):
        c, p, comp = m(pre + "_calls"), m(pre + "_prompt"), m(pre + "_completion")
        L.append(f"{tag:22}{c:>9.2f}{p:>11.0f}{comp:>12.0f}{p+comp:>11.0f}")
    L.append(f"{'WHOLE TREE':22}{m('tree_calls'):>9.2f}{m('tree_prompt'):>11.0f}"
             f"{m('tree_completion'):>12.0f}{m('tree_total'):>11.0f}")
    L.append("")
    rp, sp = m("root_prompt") + m("root_completion"), m("subcall_prompt") + m("subcall_completion")
    if rp + sp:
        L.append(f"  Share of tokens spent on orchestration : {100*rp/(rp+sp):5.1f}%")
        L.append(f"  Share of tokens spent on sub-calls     : {100*sp/(rp+sp):5.1f}%")
    L.append("")
    L.append(f"  An uninstrumented run would report {m('root_prompt')+m('root_completion'):.0f} "
             f"tokens (root only)")
    L.append(f"  True whole-tree cost is             {m('tree_total'):.0f} tokens"
             + (f"  ({(m('tree_total')/(rp) if rp else 0):.2f}x higher)" if rp else ""))

    L.append("")
    L.append("-" * 74)
    L.append("EVIDENCE BUDGET")
    L.append("-" * 74)
    L.append(f"  Mean sub-calls per verification      : {m('subcall_calls'):.2f}")
    L.append(f"  Mean chars passed into sub-calls     : {m('subcall_prompt_chars'):.0f}")
    L.append(f"  Mean REPL stdout returned to root    : {m('repl_stdout_chars'):.0f} chars")
    L.append(f"  Mean root REPL iterations            : {m('n_iterations'):.2f}")
    L.append(f"  Verifications with a REPL error      : "
             f"{sum(1 for u in us if (u.get('repl_errors') or 0) > 0)}/{len(us)}")
    nosub = sum(1 for u in us if (u.get("subcall_calls") or 0) == 0)
    L.append(f"  Verifications with ZERO sub-calls    : {nosub}/{len(us)} "
             f"({100*nosub/len(us):.0f}%)")

    costs = [u.get("tree_cost") for u in us if u.get("tree_cost") is not None]
    if costs:
        L.append("")
        L.append("-" * 74)
        L.append("COST (USD)")
        L.append("-" * 74)
        L.append(f"  Mean / verification : ${statistics.mean(costs):.5f}")
        L.append(f"  Median              : ${statistics.median(costs):.5f}")
        L.append(f"  Max                 : ${max(costs):.5f}")
        L.append(f"  Total               : ${sum(costs):.4f}")

    L.append("")
    L.append(f"  Mean wall-clock / verification : {m('wall_s'):.1f}s")
    L.append("")
    L.append(classification_report(yt, yp, labels=LABELS, zero_division=0))

    text = "\n".join(L)
    print("\n" + text)
    dest = os.path.join(out_dir(args.model),
                        f"traced_{args.split}_n{args.n}_"
                        f"{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--max-errors", type=int, default=5)
    ap.add_argument("--max-budget", type=float, default=0.15)
    ap.add_argument("--max-timeout", type=float, default=300.0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--save-trace", action="store_true", default=True)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--ablate-escape", action="store_true",
                    help="delete the orchestrator prompt's direct-read escape "
                         "clause, keeping all delegation instructions")
    ap.add_argument("--force-subcalls", action="store_true",
                    help="lower the scaffold's own chunking threshold below "
                         "the mean contract length so recursion is mandated")
    ap.add_argument("--chunk-threshold", type=int, default=3000,
                    help="chars; stated REPL-output limit in the system prompt")
    args = ap.parse_args()

    global FORCED, ABLATED
    FORCED = args.force_subcalls
    ABLATED = args.ablate_escape

    if args.report_only:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
