"""Run scaffolds on FinDVer or CoverBench.

Examples (from the repository root):

    # Qwen3-8B through OpenRouter, or a vLLM server with --base-url
    python -m src.findver_coverbench.run --benchmark findver --models qwen/qwen3-8b \\
        --scaffolds zeroshot typed cot --thinking off on

    # Qwen3-4B / 1.7B locally (Hugging Face Transformers, one GPU)
    python -m src.findver_coverbench.run --benchmark coverbench \\
        --models Qwen/Qwen3-4B Qwen/Qwen3-1.7B --scaffolds rag agentic_rag --thinking off on

    # Llama-3.2-3B-Instruct (no thinking mode)
    python -m src.findver_coverbench.run --benchmark findver \\
        --models meta-llama/llama-3.2-3b-instruct --scaffolds zeroshot typed cot

Per-claim records go to results/<benchmark>/<model>/<scaffold>_<thinking>.jsonl
(resumable) and one summary row per run to results/<benchmark>/summary.csv.
Retrieval scaffolds need an index (python -m src.findver_coverbench.build_index).
"""

import argparse
import csv
import json
import os
import time

from tqdm import tqdm

from . import benchmarks as B
from . import metrics
from .backends import free_gpu, has_thinking, make_backend
from .scaffolds import RUNNERS

RESULTS = os.path.join(B.ROOT, "results")


def run_one(bench, backend, model, scaffold, thinking, examples, index_dir):
    tag = "na" if thinking is None else ("think" if thinking else "nothink")
    slug = model.split("/")[-1].lower()
    path = os.path.join(RESULTS, bench.name, slug, f"{scaffold}_{tag}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["uid"] for l in f if l.strip()}
    max_new = bench.max_new[(scaffold, bool(thinking))]
    parse = B.parser(bench.name, scaffold)

    with open(path, "a", encoding="utf-8") as fh:
        for ex in tqdm([e for e in examples if e.uid not in done],
                       desc=f"{bench.name} | {slug} | {scaffold} | {tag}"):
            t0 = time.time()
            try:
                r = RUNNERS[scaffold](bench, backend, thinking, max_new, ex, index_dir=index_dir)
            except Exception as e:                  # runners catch their own errors
                free_gpu()
                r = dict(raw=None, in_tok=0, out_tok=0, think_words=0, error=repr(e), extras={})
            row = {"uid": ex.uid, "model": model, "scaffold": scaffold, "thinking": tag,
                   "gold": ex.gold, "input_tokens": r["in_tok"], "output_tokens": r["out_tok"],
                   "total_tokens": r["in_tok"] + r["out_tok"], "think_words": r["think_words"],
                   "seconds": round(time.time() - t0, 2), "error": r["error"],
                   **ex.meta, **r["extras"]}
            if r["raw"] is None:
                row.update(prediction="ZERO_TOKEN", correct=False, quality="zero_token", raw_output="")
            else:
                pred, quality = parse(r["raw"])
                row.update(prediction=pred, correct=pred == ex.gold, quality=quality,
                           raw_output=r["raw"])
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    m = metrics.compute(rows, scaffold)
    print(f"  BA={m['balanced_accuracy']:.4f} F1={m['f1']:.4f} "
          f"in/out tok={m['avg_input_tokens']:.0f}/{m['avg_output_tokens']:.0f} "
          f"failed={m['zero_token_rows']}")
    return {"benchmark": bench.name, "model": model, "scaffold": scaffold, "thinking": tag, **m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(B.BENCHMARKS))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scaffolds", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    ap.add_argument("--thinking", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible server (e.g. vLLM) instead of OpenRouter")
    ap.add_argument("--index-dir", default=None,
                    help="retrieval index folder (default: data/<benchmark>_index)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bench = B.BENCHMARKS[args.benchmark]
    index_dir = args.index_dir or os.path.join(B.ROOT, "data", f"{bench.name}_index")
    examples = B.load(bench.name, args.limit)
    print(f"{bench.name}: {len(examples)} claims")

    summaries = []
    for model in args.models:
        backend = make_backend(model, base_url=args.base_url)
        modes = [None] if not has_thinking(model) else [t == "on" for t in args.thinking]
        for scaffold in args.scaffolds:
            for thinking in modes:
                summaries.append(run_one(bench, backend, model, scaffold, thinking,
                                         examples, index_dir))
        backend.close()
        free_gpu()

    out = os.path.join(RESULTS, bench.name, "summary.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for s in summaries for k in s}))
        if new:
            w.writeheader()
        w.writerows(summaries)
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
