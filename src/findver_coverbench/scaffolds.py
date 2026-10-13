"""The five scaffolds. Each runner returns a dict with ``raw`` (None on
failure), ``in_tok``, ``out_tok``, ``think_words``, ``error`` and ``extras``."""

import json
import os

import dspy

from . import benchmarks as B
from . import prompts, retrieval
from .backends import dspy_lm, free_gpu
from .parsing import ANSWER_RE, SEARCH_RE, strip_think


def _result(**kw):
    out = dict(raw=None, in_tok=0, out_tok=0, think_words=0, error="", extras={})
    out.update(kw)
    return out


def _document(bench, backend, ex, max_new):
    """FinDVer: first 180,000 characters, and on HF models also cut to the
    32,768-token window. CoverBench: the full document (prompts that do not
    fit fail and are left out of the score)."""
    if backend.kind == "hf" and bench.hf_truncate:
        return backend.truncate(ex.document, max_new, max_chars=bench.max_chars)
    return ex.document[:bench.max_chars] if bench.max_chars else ex.document


# ------------------------------------------------------------ full document
def zeroshot(bench, backend, thinking, max_new, ex, **_):
    doc = _document(bench, backend, ex, max_new)
    try:
        answer, reasoning, i, o = backend.generate(B.zeroshot_messages(bench.name, ex, doc),
                                                   thinking, max_new)
    except Exception as e:
        free_gpu()
        return _result(error=repr(e))
    if not answer.strip():
        return _result(in_tok=i, out_tok=o, error="empty answer (thinking may have hit max tokens)")
    return _result(raw=answer, in_tok=i, out_tok=o, think_words=len(reasoning.split()))


def _dspy_call(bench, backend, thinking, max_new, signature, chain_of_thought, ex):
    lm = dspy_lm(backend, thinking, max_new)
    doc = _document(bench, backend, ex, max_new)
    predictor = dspy.ChainOfThought(signature) if chain_of_thought else dspy.Predict(signature)
    inputs = ({"document": doc, "statement": ex.claim} if bench.name == "findver"
              else {"evidence": doc, "claim": ex.claim})
    ctx = dict(lm=lm) if bench.name == "findver" else dict(lm=lm, adapter=dspy.ChatAdapter())
    try:
        with dspy.context(**ctx):
            pred = predictor(**inputs)
    except Exception as e:
        free_gpu()
        return None, lm, repr(e)
    finally:
        lm.history.clear()
    return pred, lm, ""


def typed(bench, backend, thinking, max_new, ex, **_):
    sig = prompts.FinDVerTyped if bench.name == "findver" else prompts.CoverBenchTyped
    pred, lm, err = _dspy_call(bench, backend, thinking, max_new, sig, False, ex)
    i, o = lm.last_usage
    if pred is None:
        return _result(in_tok=i, out_tok=o, error=err)
    if bench.name == "findver":
        answer = (getattr(pred, "answer", "") or "").strip()
        if not answer:
            return _result(in_tok=i, out_tok=o, error="empty answer")
        low = answer.lower()
        norm = "TRUE" if "entail" in low else "FALSE" if "refut" in low else answer.upper()
        return _result(raw=f"Answer: {norm}", in_tok=i, out_tok=o)
    raw = lm.last_text or (getattr(pred, "answer", "") or "")
    if not raw.strip():
        return _result(in_tok=i, out_tok=o, error="empty answer")
    return _result(raw=raw, in_tok=i, out_tok=o)


def cot(bench, backend, thinking, max_new, ex, **_):
    sig = prompts.FinDVerCoT if bench.name == "findver" else prompts.CoverBenchCoT
    pred, lm, err = _dspy_call(bench, backend, thinking, max_new, sig, True, ex)
    i, o = lm.last_usage
    if pred is None:
        return _result(in_tok=i, out_tok=o, error=err)
    reasoning = getattr(pred, "reasoning", "") or ""
    if bench.name == "findver":
        answer = getattr(pred, "answer", "") or ""
        if not answer.strip():
            return _result(in_tok=i, out_tok=o, error="empty answer")
        return _result(raw=f"{reasoning}\nAnswer: {answer}".strip(), in_tok=i, out_tok=o)
    verdict = (getattr(pred, "verdict", "") or "").strip()
    if not verdict:
        return _result(in_tok=i, out_tok=o, error="empty verdict field")
    return _result(raw=f"{reasoning.strip()}\n\nFinal Answer: {verdict}".strip(), in_tok=i, out_tok=o)


# ---------------------------------------------------------------- retrieval
def _index(index_dir, ex):
    return retrieval.load_index(os.path.join(index_dir, ex.index_key + ".pkl"))


def rag(bench, backend, thinking, max_new, ex, index_dir, **_):
    try:
        idx = _index(index_dir, ex)
        ids, scores, texts, full = idx.retrieve(ex.claim, bench.rag_top_k, bench.embed_retries)
    except Exception as e:
        return _result(error=f"retrieval failed: {e!r}")
    if not texts:
        return _result(error="retrieved no context")
    ctx = "\n\n".join(texts)
    extras = {"retrieved_ids": json.dumps(ids), "retrieved_scores": json.dumps(scores),
              "context_count": len(texts), "context_chars": len(ctx)}
    if ex.gold_chunks:
        k = bench.rag_top_k
        extras.update({"gold_ids": json.dumps(sorted(ex.gold_chunks)),
                       f"hit@{k}": retrieval.hit_at_k(ids, ex.gold_chunks),
                       f"recall@{k}": retrieval.recall_at_k(ids, ex.gold_chunks),
                       "mrr": retrieval.mrr(full, ex.gold_chunks),
                       f"ndcg@{k}": retrieval.ndcg_at_k(ids, ex.gold_chunks, k)})
    try:
        answer, reasoning, i, o = backend.generate(B.rag_messages(bench.name, ex, ctx), thinking, max_new)
    except Exception as e:
        free_gpu()
        return _result(error=repr(e), extras=extras)
    if not answer.strip():
        return _result(in_tok=i, out_tok=o, extras=extras,
                       error="empty answer (thinking may have hit max tokens)")
    return _result(raw=answer, in_tok=i, out_tok=o, think_words=len(reasoning.split()), extras=extras)


def agentic_rag(bench, backend, thinking, max_new, ex, index_dir, **_):
    """The model never sees the document; it SEARCHes it, then ANSWERs."""
    P = B.agent_prompts(bench.name)
    out = _result()
    # Qwen3 soft switch, sent alongside the API / chat-template flag.
    suffix = " /no_think" if thinking is False else ""
    messages = [{"role": "system", "content": P["system"].format(max_steps=bench.agent_steps)},
                {"role": "user", "content": P["opening"].format(claim=ex.claim) + suffix}]
    steps, shown, seen, queries, fallbacks = 0, 0, [], [], 0
    final_text, forced = None, False

    def one_call():
        nonlocal steps
        content, reasoning, i, o = backend.generate(messages, thinking, max_new)
        steps += 1
        out["in_tok"] += i
        out["out_tok"] += o
        out["think_words"] += len(reasoning.split())
        return content

    try:
        idx = _index(index_dir, ex)
        for step in range(bench.agent_steps):
            content = one_call()
            body = strip_think(content)
            m = ANSWER_RE.search(body)                 # ANSWER is checked before SEARCH
            if m:
                final_text = m.group(1).strip()
                break
            s = SEARCH_RE.search(body)
            if s:
                query = s.group(1).strip()
            else:
                query = ex.claim                       # malformed turn falls back to the claim
                fallbacks += 1
            queries.append(query)
            ids, _sc, texts, _full = idx.retrieve(query, bench.agent_k, bench.embed_retries)
            ids, texts = idx.in_document_order(ids, texts)
            shown += len(texts)
            for cid in ids:
                if cid not in seen:
                    seen.append(cid)
            results = "Search results:\n" + ("\n".join(f"- {t}" for t in texts) if texts else "(no matches)")
            tail = P["forced"] if step == bench.agent_steps - 1 else P["continue"]
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"{results}\n\n{tail}{suffix}"}]
        else:
            forced = True
            body = strip_think(one_call())
            m = ANSWER_RE.search(body)
            final_text = m.group(1).strip() if m else body
    except Exception as e:
        free_gpu()
        out["error"] = repr(e)

    out["extras"] = {"queries": json.dumps(queries), "unique_ids": json.dumps(seen),
                     "steps": steps, "forced_decision": forced, "retrieved_spans": shown,
                     "unique_spans": len(seen), "query_fallbacks": fallbacks}
    if ex.gold_chunks:
        out["extras"].update({"gold_ids": json.dumps(sorted(ex.gold_chunks)),
                              "hit_union": retrieval.hit_at_k(seen, ex.gold_chunks) if seen else 0.0,
                              "recall_union": retrieval.recall_at_k(seen, ex.gold_chunks) if seen else 0.0})
    if out["error"] or not (final_text or "").strip():
        out["error"] = out["error"] or "no answer produced"
        return out
    out["raw"] = final_text
    return out


RUNNERS = {"zeroshot": zeroshot, "typed": typed, "cot": cot, "rag": rag, "agentic_rag": agentic_rag}
