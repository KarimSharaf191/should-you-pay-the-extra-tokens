"""Build the retrieval indexes used by RAG and agentic RAG.

    python -m src.findver_coverbench.build_index --benchmark findver

FinDVer: one index per report, with the report's paragraphs as chunks (the
units FinDVer's gold evidence ids refer to), embedded with Qwen3-Embedding-4B.

CoverBench: one index per claim over chunks of its source document. The
chunking script used for the paper is not yet in this repository; until it
is, CoverBench indexes cannot be rebuilt here.
"""

import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from . import benchmarks as B
from .retrieval import embed


def build_findver(out_dir, batch=64):
    with open(os.path.join(B.FINDVER_DIR, "data", "testmini.json"), encoding="utf-8") as f:
        reports = sorted({e["report"] for e in json.load(f)})
    for name in tqdm(reports, desc="findver reports"):
        path = os.path.join(out_dir, name.replace(".json", "") + ".pkl")
        if os.path.exists(path):
            continue
        with open(os.path.join(B.FINDVER_DIR, "financial_reports", name), encoding="utf-8") as f:
            items = [c for c in json.load(f).get("context", []) if "context" in c]
        ids = [str(c.get("id", i)) for i, c in enumerate(items)]
        texts = [c["context"] for c in items]
        vecs = np.concatenate([embed(texts[i:i + batch]) for i in range(0, len(texts), batch)])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": ids, "texts": texts, "embeddings": vecs}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["findver", "coverbench"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(B.ROOT, "data", f"{args.benchmark}_index")
    if args.benchmark == "findver":
        build_findver(out)
    else:
        raise SystemExit("CoverBench chunking script not included yet: place the "
                         "per-claim indexes (<sanitised id>.pkl with chunk_ids, texts, "
                         "embeddings) in data/coverbench_index/.")
    print("indexes ->", out)


if __name__ == "__main__":
    main()
