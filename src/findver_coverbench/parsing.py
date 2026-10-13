"""Turning model output into a TRUE / FALSE prediction.

Each function returns (prediction, quality) with quality in
{"ok", "ambiguous", "malformed"}. Output with no recognisable label is
predicted FALSE; the ``quality`` column records how often that happens.
"""

import re

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
SEARCH_RE = re.compile(r"SEARCH\s*:\s*(.+)", re.IGNORECASE)


def strip_think(text):
    """Keep only what follows the LAST </think>; an unfinished <think> block yields nothing."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def _segment(raw_text):
    text = str(raw_text).strip().upper()
    return text.rsplit("ANSWER:", 1)[-1] if "ANSWER:" in text else text


def findver_fulldoc(raw_text):
    """Zero-shot / typed / CoT on FinDVer: FALSE wins if both labels appear."""
    seg = _segment(raw_text)
    has_true, has_false = "TRUE" in seg, "FALSE" in seg
    quality = "ambiguous" if (has_true and has_false) else ("malformed" if not (has_true or has_false) else "ok")
    pred = "FALSE" if has_false else ("TRUE" if has_true else "FALSE")
    return pred, quality


def findver_retrieval(raw_text):
    """RAG / agentic RAG on FinDVer: also accepts 'entailed' / 'refuted'."""
    seg = _segment(raw_text)
    has_true = "TRUE" in seg or "ENTAIL" in seg
    has_false = "FALSE" in seg or "REFUT" in seg
    quality = "ambiguous" if (has_true and has_false) else ("malformed" if not (has_true or has_false) else "ok")
    pred = "FALSE" if has_false else ("TRUE" if has_true else "FALSE")
    return pred, quality


_MARKERS = ["FINAL ANSWER:", "FINAL VERDICT:", "VERDICT:"]


def coverbench(raw_text):
    """Counts CORRECT / INCORRECT after the last verdict marker (or in the whole
    text). 'INCORRECT' contains 'CORRECT', so it is subtracted out."""
    full_text = str(raw_text).strip()
    upper = full_text.upper()
    split_idx = max((upper.rfind(m) for m in _MARKERS), default=-1)
    v = (full_text[split_idx:] if split_idx >= 0 else full_text).upper()
    incorrect = v.count("INCORRECT")
    correct = v.count("CORRECT") - incorrect
    pred = "FALSE" if incorrect > 0 else ("TRUE" if correct > 0 else "FALSE")
    quality = "ambiguous" if (correct > 0 and incorrect > 0) else ("malformed" if (correct == 0 and incorrect == 0) else "ok")
    return pred, quality
