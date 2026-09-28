"""
ContractNLI zero-shot evaluation on Qwen3-8B via OpenRouter.

Two paths:
  1. Raw OpenRouter chat-completions call  (--mode raw)
  2. DSPy-orchestrated call                (--mode dspy)

Reports balanced accuracy + macro F1 (and per-class F1) against the
gold annotations.

SECURITY: set your key in the environment, do NOT hardcode it.
    Windows (PowerShell):  $env:OPENROUTER_API_KEY="sk-or-v1-..."
    Linux/Mac:             export OPENROUTER_API_KEY="sk-or-v1-..."
"""

import os
import re
import json
import time
import random
import argparse
import threading
from collections import defaultdict

from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL = "qwen/qwen3-8b"          # OpenRouter model slug
LABELS = ["Entailment", "Contradiction", "NotMentioned"]

# Set via environment variable — do NOT hardcode a real key here.
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# OpenRouter model slug, overridable via --model or OPENROUTER_MODEL env var.
MODEL = os.environ.get("OPENROUTER_MODEL", MODEL)

# Inference endpoint. Defaults to OpenRouter; point it at a local vLLM server
# (e.g. http://localhost:8000/v1) to evaluate self-hosted checkpoints with the
# exact same prompts. Both are OpenAI-compatible, so only the URL changes.
BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# Path to your split files
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

# Long-context study: pass the full contract text, no truncation.
MAX_CHARS = None

# Local tokenizer used ONLY to count tokens inside <think>...</think> blocks,
# as a provider-independent measure of how much the model actually reasoned.
# (cl100k_base is a reasonable proxy; it won't exactly match Qwen's tokenizer
#  but is consistent across all conditions, which is what matters for comparison.)
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def count_think_tokens(raw_text):
    """Count tokens inside <think>...</think> blocks of a raw model output.

    Returns 0 if there is no think block. Run on EVERY condition so we can
    detect a provider doing hidden reasoning even when thinking was 'off'.
    """
    if not raw_text:
        return 0
    blocks = _THINK_RE.findall(raw_text)
    if not blocks:
        return 0
    joined = "\n".join(blocks)
    if _ENC is not None:
        return len(_ENC.encode(joined))
    # fallback: rough word-count estimate if tiktoken unavailable
    return len(joined.split())



# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_split(split):
    path = os.path.join(DATA_DIR, f"{split}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_instances(data, max_chars=MAX_CHARS):
    """Yield (doc_id, nda_key, hypothesis, contract_text, gold_choice, doc)."""
    labels = data["labels"]
    for doc in data["documents"]:
        text = doc["text"]
        if max_chars:
            text = text[:max_chars]
        annotations = doc["annotation_sets"][0]["annotations"]
        for nda_key, ann in annotations.items():
            yield (
                doc["id"],
                nda_key,
                labels[nda_key]["hypothesis"],
                text,
                ann["choice"],
                doc,
            )


# ----------------------------------------------------------------------
# Label parsing
# ----------------------------------------------------------------------
def normalize_label(raw):
    """Map a free-form model answer to one of the three canonical labels."""
    if raw is None:
        return "NotMentioned"
    # Qwen3 may emit a <think>...</think> block before the answer; drop it
    # and keep what comes after the final closing tag.
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    s = raw.strip().lower()
    # strip code fences / quotes / punctuation
    s = re.sub(r"[`'\"*]", "", s)
    if "contradict" in s:
        return "Contradiction"
    if "entail" in s:
        return "Entailment"
    if "not mentioned" in s or "notmentioned" in s or "neutral" in s:
        return "NotMentioned"
    # fallback
    return "NotMentioned"


def _get(obj, key, default=0):
    """Read key from a dict OR an attribute from an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_usage(usage):
    """Normalize an OpenRouter/LiteLLM usage dict/object to a flat token record.

    Handles both plain dicts and LiteLLM wrapper objects (Usage,
    CompletionTokensDetailsWrapper). completion_tokens already INCLUDES
    reasoning tokens for thinking models; the reasoning portion is also
    exposed separately under completion_tokens_details.reasoning_tokens.
    """
    if not usage:
        return {"prompt": 0, "completion": 0, "reasoning": 0,
                "total": 0, "think_tokens": 0, "cost_usd": 0.0,
                "cached_prompt": 0}
    prompt = _get(usage, "prompt_tokens", 0) or 0
    completion = _get(usage, "completion_tokens", 0) or 0
    total = _get(usage, "total_tokens", 0) or (prompt + completion)
    reasoning = 0
    details = _get(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = _get(details, "reasoning_tokens", 0) or 0
    # OpenRouter reports the actual charge for the call under `cost` (USD).
    # Taking it from the provider avoids maintaining a price table and stays
    # correct when a model is served by several providers at different rates.
    # vLLM and other OpenAI-compatible servers omit it, leaving 0.0.
    cost = float(_get(usage, "cost", 0.0) or 0.0)
    # Cached prompt tokens are billed at a reduced rate. Relevant to agentic
    # RAG, which resends its whole history every turn: if the provider caches
    # that prefix the cost story differs sharply from the token story.
    cached = 0
    pdetails = _get(usage, "prompt_tokens_details", None)
    if pdetails is not None:
        cached = _get(pdetails, "cached_tokens", 0) or 0
    return {
        "prompt": prompt,
        "completion": completion,
        "reasoning": reasoning,
        "total": total,
        "think_tokens": 0,  # filled in by caller via count_think_tokens()
        "cost_usd": cost,
        "cached_prompt": cached,
    }


PROMPT_TEMPLATE = """You are analyzing a non-disclosure agreement (NDA).

Decide the relationship between the CONTRACT and the HYPOTHESIS. Answer with exactly one of these three labels and nothing else:
- Entailment   (the contract supports / implies the hypothesis)
- Contradiction (the contract states the opposite of the hypothesis)
- NotMentioned (the contract does not address the hypothesis)

CONTRACT:
{contract}

HYPOTHESIS:
{hypothesis}

Answer (one label only):{nothink}"""


# ----------------------------------------------------------------------
# Path 1: raw OpenRouter
# ----------------------------------------------------------------------
def _chat_payload(messages, thinking_on):
    """Build a chat-completions body for the configured endpoint.

    `reasoning` and `usage` are OpenRouter extensions. A strict OpenAI-compatible
    server (vLLM) rejects unknown top-level fields with 400, so they are only
    sent when actually talking to OpenRouter. vLLM returns standard `usage`
    on non-streaming responses anyway, so token accounting still works.
    """
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
    }
    if "openrouter.ai" in BASE_URL:
        body["reasoning"] = {"enabled": thinking_on}
        body["usage"] = {"include": True}
    else:
        # vLLM's Qwen3 chat template thinks by default, so thinking must be
        # switched explicitly in both directions; /no_think alone is not
        # sent on every path (DSPy never adds it).
        body["chat_template_kwargs"] = {"enable_thinking": thinking_on}
    return json.dumps(body)


def _timeout(default):
    """Per-request timeout, overridable via CONTRACTNLI_TIMEOUT.

    Thinking-on runs against a shared local GPU can exceed the defaults,
    which were sized for OpenRouter.
    """
    return int(os.environ.get("CONTRACTNLI_TIMEOUT", default))


# Retries: without these a 429 or a read timeout becomes a recorded error row,
# the instance is dropped from scoring, and because rate limits and timeouts hit
# the SLOWEST requests (longest contracts, deepest agentic trajectories) the
# dropout is not random -- conditions end up scored on slightly different, and
# slightly easier, instance sets. Retrying in-process keeps the sample intact.
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_RETRIES = int(os.environ.get("CONTRACTNLI_MAX_RETRIES", "5"))


def _post_chat(payload, timeout=180):
    """POST to /chat/completions with exponential backoff on transient errors.

    Honours Retry-After when the provider sends it. Raises the last error if
    every attempt fails, so a genuine failure still surfaces as an error row
    rather than being silently swallowed.
    """
    import requests

    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, data=payload,
                                 timeout=timeout)
            if resp.status_code in _RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                wait = (float(retry_after) if retry_after
                        and retry_after.replace(".", "", 1).isdigit()
                        else min(2 ** attempt, 30))
                last = requests.HTTPError(
                    f"{resp.status_code} from provider", response=resp)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait + random.uniform(0, 0.5))  # jitter
                    continue
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 0.5))
                continue
    raise last if last else RuntimeError("request failed with no exception")


def predict_raw(hypothesis, contract, thinking_on=False):
    import requests

    # Qwen3 honours an inline /no_think directive; other models ignore it and
    # rely on the API-level reasoning flag below. Both are set together.
    nothink = "" if thinking_on else " /no_think"
    prompt = PROMPT_TEMPLATE.format(contract=contract, hypothesis=hypothesis,
                                    nothink=nothink)
    data = _post_chat(
        _chat_payload([{"role": "user", "content": prompt}], thinking_on),
        timeout=_timeout(120),
    )
    content = data["choices"][0]["message"]["content"]
    usage = _extract_usage(data.get("usage", {}) or {})
    # Provider-independent reasoning measure: count tokens inside <think>.
    # If the provider reported reasoning_tokens, keep the larger of the two.
    think_tok = count_think_tokens(content)
    usage["think_tokens"] = think_tok
    if usage["reasoning"] == 0 and think_tok > 0:
        usage["reasoning"] = think_tok
    return normalize_label(content), usage


# ----------------------------------------------------------------------
# Path 2: DSPy
# ----------------------------------------------------------------------
def build_dspy_program(condition="zeroshot"):
    """
    condition:
      "zeroshot"  -> Predict,         thinking OFF  (neither mechanism)
      "cot"       -> ChainOfThought,  thinking OFF  (prompt CoT only)
      "reasoning" -> Predict,         thinking ON   (native reasoning only)
      "cot_think" -> ChainOfThought,  thinking ON   (prompt CoT + native)
    """
    import dspy

    # Native model thinking is ON for "reasoning" and "cot_think".
    #   - reasoning  : native thinking only (Predict)
    #   - cot_think  : native thinking + prompt-elicited CoT (ChainOfThought)
    #   - cot        : prompt-elicited CoT only, native thinking OFF
    #   - zeroshot   : neither
    thinking_on = condition in ("reasoning", "cot_think")

    _via_openrouter = "openrouter.ai" in BASE_URL
    lm_kwargs = dict(
        api_key=API_KEY or "EMPTY",
        api_base=BASE_URL,
        temperature=0.0,
        # Disable caching: cached responses don't carry usage metadata, which
        # is why tokens vanish on repeated identical runs. We need a live call
        # each time to get accurate token accounting.
        cache=False,
    )
    if _via_openrouter:
        # OpenRouter-only extensions. A strict OpenAI-compatible server (vLLM)
        # returns 400 for these, so they must not be sent to one.
        lm_kwargs["reasoning"] = {"enabled": thinking_on}
        lm_kwargs["extra_body"] = {"usage": {"include": True}}
    else:
        # vLLM exposes Qwen3 thinking through chat_template_kwargs instead.
        # Set it in both directions: the template's default is thinking ON,
        # and DSPy prompts carry no /no_think directive.
        lm_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": thinking_on}}
        lm_kwargs["timeout"] = _timeout(600)

    lm = dspy.LM(
        (f"openrouter/{MODEL}" if _via_openrouter else f"openai/{MODEL}"),
        **lm_kwargs,
    )
    # track_usage=True lets us read per-call tokens via out.get_lm_usage().
    dspy.configure(lm=lm, track_usage=True)

    class ContractNLI(dspy.Signature):
        """Classify the relationship between an NDA contract and a hypothesis.
        Answer with exactly one of: Entailment, Contradiction, NotMentioned."""

        contract: str = dspy.InputField(desc="full text of the NDA")
        hypothesis: str = dspy.InputField(desc="statement to verify against the contract")
        label: str = dspy.OutputField(desc="one of: Entailment, Contradiction, NotMentioned")

    if condition in ("cot", "cot_think"):
        # ChainOfThought adds a rationale field before the label (prompt CoT).
        # For cot_think, native thinking is additionally ON (set above).
        return dspy.ChainOfThought(ContractNLI)

    # zeroshot and reasoning both use Predict (direct label).
    # The difference between them is native thinking on/off, set above.
    return dspy.Predict(ContractNLI)


# ----------------------------------------------------------------------
# Path 3: RAG (Retrieval-Augmented Generation) — tool augmentation.
# ContractNLI already segments each document into spans (sentences/list
# items). RAG retrieves only the top-k spans most relevant to the hypothesis
# and feeds ONLY those to the model, instead of the whole contract. This
# tests the proposal's Strategy 3:
#   - performance with RAG
#   - context reduction (far fewer prompt tokens than full-context)
#   - reasoning vs instruction-following gap under RAG (thinking on/off)
#   - regular retrieval vs agentic retrieval (two retrieval styles)
# ----------------------------------------------------------------------

# Lazy global embedder so we load the model once per run.
_EMBEDDER = None


_EMBEDDER_LOCK = threading.Lock()


def _get_embedder():
    """Lazily load the span embedder, once per process.

    The lock matters under concurrency: without it every worker thread sees
    _EMBEDDER as None simultaneously and loads its own copy of the model,
    which exhausts memory and kills the run mid-condition.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:          # re-check inside the lock
                # Torch defaults to one intra-op thread per core, so N worker
                # threads each spawn a full thread pool and the machine ends up
                # many times oversubscribed -- encodes then contend with each
                # other instead of running faster. Cap it; with the span cache
                # below there is little encoding left to parallelise anyway.
                try:
                    import torch
                    torch.set_num_threads(
                        int(os.environ.get("CONTRACTNLI_TORCH_THREADS", "2")))
                except Exception:
                    pass
                from sentence_transformers import SentenceTransformer
                # Small, fast, good enough for span ranking.
                _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


# Span embeddings depend only on the document, never on the query, but
# retrieve_top_k used to re-encode every span on every call -- and agentic RAG
# calls it once per search step. Over a full dev run that is thousands of
# re-encodings of the same handful of documents, and it dominated wall clock on
# the retrieval conditions. Cache per document instead.
_SPAN_EMB_CACHE = {}
_SPAN_CACHE_LOCK = threading.Lock()


def _span_embeddings(span_texts, doc_key):
    """Encoded span matrix for a document, computed once per doc_key.

    With no doc_key the cache is bypassed and behaviour is exactly as before,
    so callers that do not have a document id keep working.
    """
    if doc_key is None:
        return _get_embedder().encode(span_texts, convert_to_numpy=True,
                                      normalize_embeddings=True)
    with _SPAN_CACHE_LOCK:
        hit = _SPAN_EMB_CACHE.get(doc_key)
    if hit is not None:
        return hit
    # Encoding outside the lock: two threads may briefly duplicate work for the
    # same document, which is harmless and far cheaper than serialising every
    # encode behind one global lock.
    emb = _get_embedder().encode(span_texts, convert_to_numpy=True,
                                 normalize_embeddings=True)
    with _SPAN_CACHE_LOCK:
        _SPAN_EMB_CACHE[doc_key] = emb
    return emb


def get_document_spans(doc):
    """Return the list of span texts for a document, sliced from char offsets."""
    text = doc["text"]
    spans = []
    for start, end in doc["spans"]:
        spans.append(text[start:end])
    return spans


def retrieve_top_k(hypothesis, span_texts, k=8, doc_key=None):
    """Embedding cosine-similarity retrieval: return the top-k spans (in
    original document order) most relevant to the hypothesis.

    `doc_key` (a document id) enables the span-embedding cache; omitting it
    falls back to encoding the spans on every call.
    """
    import numpy as np

    if not span_texts:
        return []
    embedder = _get_embedder()
    span_emb = _span_embeddings(span_texts, doc_key)
    hyp_emb = embedder.encode([hypothesis], convert_to_numpy=True,
                              normalize_embeddings=True)[0]
    scores = span_emb @ hyp_emb  # cosine sim (already normalized)
    top_idx = np.argsort(-scores)[:k]
    # keep original document order for readability/coherence
    top_idx = sorted(top_idx.tolist())
    return [span_texts[i] for i in top_idx]


RAG_PROMPT = """You are analyzing a non-disclosure agreement (NDA).

Below are the contract excerpts most relevant to the hypothesis. Decide the
relationship between these excerpts and the HYPOTHESIS. Answer with exactly one
of these three labels and nothing else:
- Entailment   (the excerpts support / imply the hypothesis)
- Contradiction (the excerpts state the opposite of the hypothesis)
- NotMentioned (the excerpts do not address the hypothesis)

CONTRACT EXCERPTS:
{excerpts}

HYPOTHESIS:
{hypothesis}

Answer (one label only):{nothink}"""


def predict_rag(hypothesis, doc, thinking_on=False, k=8):
    """Regular RAG: retrieve top-k spans once, then classify with one call."""
    import requests

    span_texts = get_document_spans(doc)
    retrieved = retrieve_top_k(hypothesis, span_texts, k=k,
                               doc_key=doc.get("id"))
    excerpts = "\n".join(f"- {s}" for s in retrieved)
    nothink = "" if thinking_on else " /no_think"
    prompt = RAG_PROMPT.format(excerpts=excerpts, hypothesis=hypothesis,
                               nothink=nothink)

    data = _post_chat(
        _chat_payload([{"role": "user", "content": prompt}], thinking_on),
        timeout=_timeout(180),
    )
    content = data["choices"][0]["message"]["content"]
    usage = _extract_usage(data.get("usage", {}) or {})
    think_tok = count_think_tokens(content)
    usage["think_tokens"] = think_tok
    if usage["reasoning"] == 0 and think_tok > 0:
        usage["reasoning"] = think_tok
    usage["retrieved_spans"] = len(retrieved)
    return normalize_label(content), usage


# ---- Agentic RAG --------------------------------------------------------
# Instead of a single fixed top-k retrieval, the model iteratively issues
# search queries over the spans and decides when it has enough evidence.
# This is the "agentic RAG vs regular retrieval" comparison.

AGENTIC_SYSTEM = """You are verifying a hypothesis against an NDA contract.
You cannot see the full contract. You can SEARCH for relevant excerpts by
issuing queries. On each turn, respond in ONE of these two formats only:

SEARCH: <a short query to find relevant contract spans>
or
ANSWER: <one of: Entailment, Contradiction, NotMentioned>

Issue SEARCH queries until you have enough evidence, then give ANSWER.
You have a maximum of {max_steps} searches."""


def predict_agentic_rag(hypothesis, doc, thinking_on=False, k=5, max_steps=4):
    """Agentic RAG: the model issues its own search queries over the spans
    across several turns, then commits to a label."""
    import requests

    span_texts = get_document_spans(doc)
    messages = [
        {"role": "system", "content": AGENTIC_SYSTEM.format(max_steps=max_steps)},
        {"role": "user", "content": f"Hypothesis: {hypothesis}\n\nBegin."},
    ]

    agg = {"prompt": 0, "completion": 0, "reasoning": 0,
           "total": 0, "think_tokens": 0, "retrieved_spans": 0, "steps": 0,
           "cost_usd": 0.0, "cached_prompt": 0}

    def call(msgs):
        return _post_chat(_chat_payload(msgs, thinking_on),
                          timeout=_timeout(180))

    label = "NotMentioned"
    for step in range(max_steps + 1):
        data = call(messages)
        content = data["choices"][0]["message"]["content"] or ""
        u = _extract_usage(data.get("usage", {}) or {})
        agg["prompt"] += u["prompt"]
        agg["completion"] += u["completion"]
        agg["reasoning"] += u["reasoning"]
        agg["total"] += u["total"]
        agg["cost_usd"] += u.get("cost_usd", 0.0)
        agg["cached_prompt"] += u.get("cached_prompt", 0)
        agg["think_tokens"] += count_think_tokens(content)
        agg["steps"] += 1

        body = content
        if "</think>" in body:
            body = body.split("</think>")[-1]

        # Did the model commit to an answer?
        m = re.search(r"ANSWER\s*:\s*(.+)", body, re.IGNORECASE)
        if m:
            label = normalize_label(m.group(1))
            break

        # Otherwise treat it as a search query
        q = re.search(r"SEARCH\s*:\s*(.+)", body, re.IGNORECASE)
        query = q.group(1).strip() if q else hypothesis
        retrieved = retrieve_top_k(query, span_texts, k=k,
                                   doc_key=doc.get("id"))
        agg["retrieved_spans"] += len(retrieved)
        excerpts = "\n".join(f"- {s}" for s in retrieved) or "(no matches)"

        # Feed results back and continue the loop
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user",
                         "content": f"Search results:\n{excerpts}\n\n"
                                    f"Issue another SEARCH or give your ANSWER."})
    else:
        # ran out of steps without ANSWER -> one final forced decision
        messages.append({"role": "user",
                         "content": "You are out of searches. Reply with "
                                    "ANSWER: <Entailment|Contradiction|NotMentioned>"})
        data = call(messages)
        content = data["choices"][0]["message"]["content"] or ""
        u = _extract_usage(data.get("usage", {}) or {})
        agg["prompt"] += u["prompt"]; agg["completion"] += u["completion"]
        agg["reasoning"] += u["reasoning"]; agg["total"] += u["total"]
        agg["cost_usd"] += u.get("cost_usd", 0.0)
        agg["cached_prompt"] += u.get("cached_prompt", 0)
        agg["steps"] += 1
        label = normalize_label(content)

    if agg["reasoning"] == 0 and agg["think_tokens"] > 0:
        agg["reasoning"] = agg["think_tokens"]
    return label, agg


def _raw_output_from_history(lm):
    """Best-effort extraction of the raw model completion text from DSPy history,
    so we can scan for <think> blocks regardless of how DSPy parsed the fields."""
    try:
        if not lm or not lm.history:
            return ""
        entry = lm.history[-1]
        # DSPy stores the raw provider response under "response"; the text may
        # also be available under "outputs".
        outputs = entry.get("outputs")
        if outputs:
            if isinstance(outputs, list):
                return "\n".join(str(o) for o in outputs)
            return str(outputs)
        resp = entry.get("response")
        if resp is not None:
            # LiteLLM ModelResponse-like object
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                return str(resp)
    except Exception:
        pass
    return ""


def _usage_from_history(lm):
    """Pull a usage dict from the most recent DSPy history entry.
    Used as a fallback when out.get_lm_usage() returns empty."""
    try:
        if not lm or not lm.history:
            return {}
        entry = lm.history[-1]
        # usage may be a top-level key, or nested in the response object
        u = entry.get("usage")
        if u:
            return u if isinstance(u, dict) else dict(u)
        resp = entry.get("response")
        if resp is not None:
            ru = getattr(resp, "usage", None)
            if ru is not None:
                # LiteLLM Usage object -> dict
                try:
                    return dict(ru)
                except Exception:
                    return {
                        "prompt_tokens": getattr(ru, "prompt_tokens", 0),
                        "completion_tokens": getattr(ru, "completion_tokens", 0),
                        "total_tokens": getattr(ru, "total_tokens", 0),
                        "completion_tokens_details":
                            getattr(ru, "completion_tokens_details", None),
                    }
    except Exception:
        pass
    return {}


def _usage_is_empty(u):
    return (u.get("prompt", 0) == 0
            and u.get("completion", 0) == 0
            and u.get("total", 0) == 0)


def predict_dspy(program, hypothesis, contract):
    import dspy

    out = program(contract=contract, hypothesis=hypothesis)

    # Source 1: official DSPy usage tracking (track_usage=True).
    # Structure is {model_name: usage_dict}. Sum across models (usually one),
    # passing each model's usage dict straight to _extract_usage so the
    # nested completion_tokens_details object is handled correctly.
    usage = {"prompt": 0, "completion": 0, "reasoning": 0,
             "total": 0, "think_tokens": 0}
    try:
        lm_usage = out.get_lm_usage() or {}
        acc = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
        any_model = False
        for _model, u in lm_usage.items():
            any_model = True
            e = _extract_usage(u)
            acc["prompt"] += e["prompt"]
            acc["completion"] += e["completion"]
            acc["reasoning"] += e["reasoning"]
            acc["total"] += e["total"]
        if any_model:
            acc["think_tokens"] = 0
            usage = acc
    except Exception:
        pass

    # Source 2 (fallback): read straight from lm.history if Source 1 was empty.
    if _usage_is_empty(usage):
        hist_usage = _usage_from_history(dspy.settings.lm)
        if hist_usage:
            usage = _extract_usage(hist_usage)

    # Provider-independent reasoning measure from the raw <think> block.
    raw_text = _raw_output_from_history(dspy.settings.lm)
    think_tok = count_think_tokens(raw_text)
    usage["think_tokens"] = think_tok
    if usage["reasoning"] == 0 and think_tok > 0:
        usage["reasoning"] = think_tok

    return normalize_label(out.label), usage


# ----------------------------------------------------------------------
# Output organization: each mode writes into results/<folder>/
# ----------------------------------------------------------------------
MODE_FOLDER = {
    "raw": "zeroshot-raw",
    "dspy": "zeroshot-dspy",
    "cot": "cot",
    "cot_think": "cot_think",
    "reasoning": "reasoning",
    "rag": "rag",
    "rag_think": "rag_think",
    "agentic_rag": "agentic_rag",
    "agentic_rag_think": "agentic_rag_think",
}


def _model_slug():
    """Filesystem-safe slug for the current MODEL, e.g. qwen/qwen3-8b -> qwen3-8b."""
    return MODEL.split("/")[-1].replace(":", "-")


def _mode_dir(mode):
    """results/<mode-folder>/<model-slug>/ , created if needed."""
    folder = MODE_FOLDER.get(mode, mode)
    path = os.path.join(RESULTS_DIR, folder, _model_slug())
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# Evaluation
def _write_report(mode, split, report):
    """Write the metrics report to a timestamped text file in the mode folder."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_mode_dir(mode), f"results_{mode}_{split}_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nResults saved to {path}")


# ----------------------------------------------------------------------
def evaluate(mode, split, limit=None):
    if not API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")

    data = load_split(split)
    instances = list(iter_instances(data))
    if limit:
        instances = instances[:limit]

    if mode == "raw":
        program = None
    elif mode == "dspy":
        program = build_dspy_program(condition="zeroshot")
    elif mode == "cot":
        program = build_dspy_program(condition="cot")
    elif mode == "reasoning":
        program = build_dspy_program(condition="reasoning")
    elif mode == "cot_think":
        program = build_dspy_program(condition="cot_think")
    elif mode in ("rag", "rag_think", "agentic_rag", "agentic_rag_think"):
        program = None  # RAG modes call OpenRouter directly + a retriever
    else:
        raise ValueError(f"unknown mode: {mode}")

    y_true, y_pred = [], []
    per_hyp = defaultdict(lambda: {"true": [], "pred": []})
    n_errors = 0
    usage_records = []  # one dict per successful instance

    for i, (doc_id, nda_key, hyp, contract, gold, doc) in enumerate(instances, 1):
        try:
            if mode == "raw":
                pred, usage = predict_raw(hyp, contract)
            elif mode == "rag":
                pred, usage = predict_rag(hyp, doc, thinking_on=False)
            elif mode == "rag_think":
                pred, usage = predict_rag(hyp, doc, thinking_on=True)
            elif mode == "agentic_rag":
                pred, usage = predict_agentic_rag(hyp, doc, thinking_on=False)
            elif mode == "agentic_rag_think":
                pred, usage = predict_agentic_rag(hyp, doc, thinking_on=True)
            else:  # dspy / cot / reasoning / cot_think use the DSPy program
                pred, usage = predict_dspy(program, hyp, contract)
        except Exception as e:
            print(f"  [error doc {doc_id} {nda_key}]: {e}")
            n_errors += 1
            continue  # do NOT score failed calls as a prediction

        y_true.append(gold)
        y_pred.append(pred)
        per_hyp[nda_key]["true"].append(gold)
        per_hyp[nda_key]["pred"].append(pred)
        usage_records.append(usage)

        if i % 25 == 0:
            print(f"  {i}/{len(instances)} done")
        time.sleep(0.2)  # gentle rate limiting

    # ------------------------------------------------------------------
    # Build the report as a string so we can both print and save it.
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 60)
    lines.append("ContractNLI zero-shot evaluation")
    lines.append("=" * 60)
    lines.append(f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model       : {MODEL}")
    lines.append(f"Mode        : {mode}")
    lines.append(f"Split       : {split}")
    lines.append(f"Limit       : {limit}")
    lines.append(f"Max chars   : {MAX_CHARS}")
    lines.append(f"Scored      : {len(y_true)} instances")
    lines.append(f"Errored     : {n_errors} (excluded from metrics)")
    if n_errors:
        lines.append("WARNING: some calls failed. Fix these before trusting the metrics.")

    if not y_true:
        lines.append("No successful predictions — nothing to score.")
        report = "\n".join(lines)
        print(report)
        _write_report(mode, split, report)
        return

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    lines.append("")
    lines.append("-" * 60)
    lines.append("Overall metrics")
    lines.append("-" * 60)
    lines.append(f"Balanced accuracy : {bal_acc:.4f}")
    lines.append(f"Macro F1          : {macro_f1:.4f}")
    for lab, f in zip(LABELS, f1_per):
        lines.append(f"  F1 [{lab:13}]: {f:.4f}")

    lines.append("")
    lines.append("Full classification report:")
    lines.append(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    # Per-hypothesis breakdown
    lines.append("-" * 60)
    lines.append("Per-hypothesis balanced accuracy")
    lines.append("-" * 60)
    for nda_key in sorted(per_hyp.keys(), key=lambda x: int(x.split("-")[1])):
        t = per_hyp[nda_key]["true"]
        p = per_hyp[nda_key]["pred"]
        try:
            ba = balanced_accuracy_score(t, p)
        except Exception:
            ba = float("nan")
        lines.append(f"  {nda_key:>7} | n={len(t):>3} | bal_acc={ba:.3f}")

    # ------------------------------------------------------------------
    # Efficiency (token usage) — proposal's second evaluation dimension.
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("-" * 60)
    lines.append("Efficiency (tokens per instance, averaged over scored)")
    lines.append("-" * 60)
    n = len(usage_records)
    if n:
        sum_prompt = sum(u["prompt"] for u in usage_records)
        sum_completion = sum(u["completion"] for u in usage_records)
        sum_reasoning = sum(u["reasoning"] for u in usage_records)
        sum_total = sum(u["total"] for u in usage_records)
        lines.append(f"  Avg prompt tokens     : {sum_prompt / n:.1f}")
        lines.append(f"  Avg completion tokens : {sum_completion / n:.1f}")
        lines.append(f"  Avg reasoning tokens  : {sum_reasoning / n:.1f}")
        lines.append(f"  Avg total tokens      : {sum_total / n:.1f}")
        sum_think = sum(u.get("think_tokens", 0) for u in usage_records)
        n_with_think = sum(1 for u in usage_records if u.get("think_tokens", 0) > 0)
        lines.append(f"  Avg <think> tokens    : {sum_think / n:.1f}")
        lines.append(f"  Instances with <think>: {n_with_think}/{n}")
        lines.append("")
        lines.append(f"  Total prompt tokens     : {sum_prompt}")
        lines.append(f"  Total completion tokens : {sum_completion}")
        lines.append(f"  Total reasoning tokens  : {sum_reasoning}")
        lines.append(f"  Total <think> tokens    : {sum_think}")
        lines.append(f"  Total tokens (all calls): {sum_total}")
        # RAG-specific: how many spans were retrieved (context reduction proxy)
        if any("retrieved_spans" in u for u in usage_records):
            sum_ret = sum(u.get("retrieved_spans", 0) for u in usage_records)
            lines.append("")
            lines.append(f"  Avg retrieved spans   : {sum_ret / n:.1f}")
        if any("steps" in u for u in usage_records):
            sum_steps = sum(u.get("steps", 0) for u in usage_records)
            lines.append(f"  Avg agentic steps     : {sum_steps / n:.1f}")
        if sum_total == 0:
            lines.append("  NOTE: usage came back empty — provider may not report tokens.")
        # 'Reasoning behind our back' check: thinking-off conditions should
        # have zero <think> tokens. *_think modes have thinking ON, excluded.
        if mode in ("raw", "dspy", "cot", "rag", "agentic_rag") and n_with_think > 0:
            lines.append("")
            lines.append(f"  ** WARNING: thinking was supposed to be OFF for mode "
                         f"'{mode}', but {n_with_think} instances contained a "
                         f"<think> block. The provider may be reasoning anyway. **")
    else:
        lines.append("  No usage records.")

    report = "\n".join(lines)
    print("\n" + report)

    # Save the text report
    _write_report(mode, split, report)

    # Save raw predictions + per-instance usage (JSON) into the mode folder
    preds_path = os.path.join(_mode_dir(mode), f"preds_{mode}_{split}.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"true": t, "pred": p, "usage": u}
             for t, p, u in zip(y_true, y_pred, usage_records)],
            f, indent=2,
        )
    print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["raw", "dspy", "cot", "reasoning", "cot_think",
                             "rag", "rag_think",
                             "agentic_rag", "agentic_rag_think"],
                    default="raw",
                    help="raw=direct OpenRouter instruct; dspy=DSPy zero-shot; "
                         "cot=ChainOfThought (thinking off); "
                         "reasoning=Predict (native thinking on); "
                         "cot_think=ChainOfThought + native thinking on; "
                         "rag=regular top-k retrieval (thinking off); "
                         "rag_think=regular RAG + native thinking on; "
                         "agentic_rag=model issues its own searches (thinking off); "
                         "agentic_rag_think=agentic RAG + native thinking on")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of instances (for quick smoke tests)")
    ap.add_argument("--model", default=None,
                    help="OpenRouter model slug, e.g. qwen/qwen3-8b "
                         "(overrides MODEL / OPENROUTER_MODEL env var)")
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
    evaluate(args.mode, args.split, args.limit)