"""Benchmark loading and per-benchmark settings."""

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import parsing, prompts

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Example:
    uid: str
    claim: str
    gold: str                 # "TRUE" / "FALSE"
    document: str             # full text (FinDVer report or CoverBench context)
    index_key: str            # which retrieval index to use
    gold_chunks: set = field(default_factory=set)
    meta: dict = field(default_factory=dict)


@dataclass
class Benchmark:
    name: str
    max_new: dict             # (scaffold, thinking) -> max generated tokens per call
    max_chars: int = None     # cut the document to this many characters (API runs)
    hf_truncate: bool = False # also cut to the 32,768-token window on HF models
    rag_top_k: int = 10
    agent_k: int = 5
    agent_steps: int = 4
    embed_retries: int = 5


FINDVER = Benchmark(
    name="findver",
    max_new={("zeroshot", False): 5, ("zeroshot", True): 4096,
             ("typed", False): 300, ("typed", True): 4096,
             ("cot", False): 300, ("cot", True): 4096,
             ("rag", False): 300, ("rag", True): 4096,
             ("agentic_rag", False): 300, ("agentic_rag", True): 4096},
    max_chars=180000, hf_truncate=True, embed_retries=5,
)

COVERBENCH = Benchmark(
    name="coverbench",
    max_new={("zeroshot", False): 300, ("zeroshot", True): 4096,
             ("typed", False): 300, ("typed", True): 4096,
             ("cot", False): 1024, ("cot", True): 4096,
             ("rag", False): 300, ("rag", True): 4096,
             ("agentic_rag", False): 300, ("agentic_rag", True): 4096},
    max_chars=None, hf_truncate=False, embed_retries=2,
)

BENCHMARKS = {"findver": FINDVER, "coverbench": COVERBENCH}

FINDVER_DIR = os.environ.get("FINDVER_DIR", os.path.join(ROOT, "data", "findver"))
COVERBENCH_FILE = os.environ.get("COVERBENCH_FILE") or os.environ.get(
    "COVERBENCH_PATH", os.path.join(ROOT, "data", "coverbench", "coverbench.json"))


@lru_cache(maxsize=64)
def _findver_report(name):
    with open(os.path.join(FINDVER_DIR, "financial_reports", name), encoding="utf-8") as f:
        report = json.load(f)
    return "\n\n".join(item["context"] for item in report.get("context", []) if "context" in item)


def safe_id(sample_id):
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(sample_id))


def load(name, limit=None):
    """List of Example objects in benchmark order."""
    out = []
    if name == "findver":
        with open(os.path.join(FINDVER_DIR, "data", "testmini.json"), encoding="utf-8") as f:
            data = json.load(f)
        for e in data[:limit] if limit else data:
            out.append(Example(
                uid=e["example_id"], claim=e["statement"],
                gold="TRUE" if e["entailment_label"] else "FALSE",
                document=_findver_report(e["report"]),
                index_key=e["report"].replace(".json", ""),
                gold_chunks={str(x) for x in e.get("relevant_context") or []},
                meta={"report": e["report"], "subset": e.get("subset")}))
    elif name == "coverbench":
        with open(COVERBENCH_FILE, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for i, r in enumerate(rows[:limit] if limit else rows):
            sid = str(r.get("id", i))
            out.append(Example(
                uid=sid, claim=r["claim"], gold="TRUE" if r["label"] else "FALSE",
                document=r["context"], index_key=safe_id(sid),
                meta={"source_dataset": r.get("source_dataset", ""), "domain": r.get("domain", "")}))
    else:
        raise ValueError(name)
    return out


# ---------------------------------------------------------- prompt pieces
def zeroshot_messages(name, ex, document):
    if name == "findver":
        return [{"role": "system", "content": prompts.FINDVER_SYSTEM},
                {"role": "user", "content": prompts.findver_user(document, ex.claim)}]
    return [{"role": "user", "content": prompts.coverbench_zeroshot(ex.claim, document)}]


def rag_messages(name, ex, context):
    if name == "findver":
        return [{"role": "system", "content": prompts.FINDVER_SYSTEM},
                {"role": "user", "content": prompts.findver_user(context, ex.claim)}]
    return [{"role": "system", "content": prompts.COVERBENCH_REASONING},
            {"role": "user", "content": prompts.coverbench_rag_user(context, ex.claim)}]


def agent_prompts(name):
    return prompts.FINDVER_AGENT if name == "findver" else prompts.COVERBENCH_AGENT


def parser(name, scaffold):
    if name == "coverbench":
        return parsing.coverbench
    return parsing.findver_retrieval if scaffold in ("rag", "agentic_rag") else parsing.findver_fulldoc
