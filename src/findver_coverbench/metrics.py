"""Scores for one run. Calls that produced no answer (``ZERO_TOKEN``) are left
out of the confusion matrix and the token averages, and counted separately."""

EXTRA_METRICS = {
    "rag": ["hit@10", "recall@10", "mrr", "ndcg@10", "context_count", "context_chars"],
    "agentic_rag": ["steps", "retrieved_spans", "unique_spans", "query_fallbacks",
                    "hit_union", "recall_union", "think_words"],
}


def compute(rows, scaffold):
    tp = sum(r["prediction"] == "TRUE" and r["gold"] == "TRUE" for r in rows)
    fp = sum(r["prediction"] == "TRUE" and r["gold"] == "FALSE" for r in rows)
    tn = sum(r["prediction"] == "FALSE" and r["gold"] == "FALSE" for r in rows)
    fn = sum(r["prediction"] == "FALSE" and r["gold"] == "TRUE" for r in rows)
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    valid = [r for r in rows if r["prediction"] != "ZERO_TOKEN"]

    def avg(k):
        vals = [r[k] for r in valid if isinstance(r.get(k), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0

    m = {"samples": total, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
         "accuracy": round((tp + tn) / total, 4) if total else 0,
         "balanced_accuracy": round((recall + specificity) / 2, 4),
         "precision": round(precision, 4), "recall": round(recall, 4),
         "specificity": round(specificity, 4), "f1": round(f1, 4),
         "avg_input_tokens": avg("input_tokens"), "avg_output_tokens": avg("output_tokens"),
         "avg_total_tokens": avg("total_tokens"),
         "zero_token_rows": sum(r["prediction"] == "ZERO_TOKEN" for r in rows),
         "ambiguous_rows": sum(r.get("quality") == "ambiguous" for r in rows),
         "malformed_rows": sum(r.get("quality") == "malformed" for r in rows)}
    for k in EXTRA_METRICS.get(scaffold, []):
        m[f"avg_{k}"] = avg(k)
    if scaffold == "agentic_rag":
        m["forced_decision_rate"] = avg("forced_decision")
    return m
