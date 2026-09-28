"""
ContractNLI benchmark analysis for LLM context-length / claim-difficulty study.

For each (document, hypothesis) pair in train/dev/test.json this script extracts:

  - Length:           hypothesis token length, evidence token length, document
                      (premise) token length.
  - Information source:
        * local vs dispersed   -> how spread out the evidence spans are across
                                   the document (char-distance heuristic).
        * single-hop vs multi-hop -> how many evidence spans support the label
                                   (NotMentioned examples have 0 spans and are
                                   marked "not_applicable").
        * structured vs unstructured (combining) -> whether the evidence spans
                                   look like enumerated/tabular text vs prose,
                                   and whether a multi-hop claim mixes both.

It also tokenizes every contract with tiktoken and cross-references document
length against the published context windows of common LLMs, to find the
point at which each model would start failing (i.e. truncating/missing
evidence) on this dataset.

ASSUMPTIONS / HEURISTICS (tune via CLI flags, see --help):
  - "Local vs dispersed" is based on the *gap* between evidence spans: the
    [first_start, last_end] range minus the spans' own character widths,
    as a fraction of document length. Using the raw outer range alone would
    conflate dispersion with span length (a single long enumerated clause
    would look "dispersed" purely because it's wide, not because it's far
    from other evidence). <= --locality-threshold (default 0.15 = 15% of
    doc) => local.
  - "Structured" spans are detected via regex: enumerated list markers
    (a., i., 1)...), or a high digit/punctuation density typical of
    tables, dates, money amounts, defined-term lists, etc.
  - These are heuristics over the *existing* ContractNLI span annotations;
    the dataset itself has no ground-truth label for this dimension.

Usage:
    python benchmark_analysis.py --data-dir . --out-dir analysis_out --plot
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import tiktoken

# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
# tiktoken only ships OpenAI encodings. There is no public tiktoken-compatible
# tokenizer for Claude/Gemini/Llama, so we report OpenAI encodings exactly and
# use them as a documented *approximation* for non-OpenAI models (the ratio of
# tokens/char is similar enough across modern BPE tokenizers for an order-of-
# magnitude context-window analysis).
ENCODINGS = {
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),   # GPT-3.5 / GPT-4
    "o200k_base": tiktoken.get_encoding("o200k_base"),     # GPT-4o / GPT-4.1
}
PRIMARY_ENCODING = "cl100k_base"

# Published max context window (input tokens) for models commonly compared
# against in NLI / legal-NLP papers. Edit freely.
MODEL_CONTEXT_WINDOWS = {
    "BERT/RoBERTa-base (orig. ContractNLI baseline)": 512,
    "Longformer-base (orig. ContractNLI baseline)": 4096,
    "GPT-3.5-turbo (4k)": 4096,
    "GPT-3.5-turbo-16k": 16384,
    "GPT-4 (8k)": 8192,
    "GPT-4-32k": 32768,
    "GPT-4o / GPT-4.1 / GPT-4-turbo (128k)": 128000,
    "Claude 3 / 3.5 / 4 (200k)": 200000,
    "Gemini 1.5/2.x Pro (1M)": 1_000_000,
}

ENUM_MARKER_RE = re.compile(
    r"(^|\n)\s*(\(?[a-zA-Z]\)|\(?[ivxlcdmIVXLCDM]{1,5}\)|\(?\d{1,3}\)|\d{1,3}\.)\s+"
)
DIGIT_PUNCT_RE = re.compile(r"[\d%$.,;:()/-]")


def is_structured_span(text: str) -> bool:
    """Heuristic: enumerated lists / tables / dates-amounts-heavy clauses."""
    if not text:
        return False
    if ENUM_MARKER_RE.search(text):
        return True
    if len(text) >= 8:
        density = len(DIGIT_PUNCT_RE.findall(text)) / len(text)
        if density > 0.20:
            return True
    return False


@dataclass
class ExampleRow:
    split: str
    doc_id: int
    file_name: str
    hypothesis_id: str
    hypothesis_text: str
    label: str
    doc_char_len: int
    doc_tokens: int
    hypothesis_tokens: int
    num_evidence_spans: int
    evidence_char_len: int
    evidence_tokens: int
    hop_type: str
    locality: str
    evidence_spread: float  # fraction of doc length spanned by *gaps* between evidence spans (span width excluded); NaN if no evidence
    structure_type: str
    combined_tokens: int  # doc_tokens + hypothesis_tokens, used for context-window checks


def count_tokens(text: str, encoding_name: str = PRIMARY_ENCODING) -> int:
    return len(ENCODINGS[encoding_name].encode(text, disallowed_special=()))


def load_split(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_hop(num_spans: int) -> str:
    if num_spans == 0:
        return "not_applicable"
    if num_spans == 1:
        return "single-hop"
    return "multi-hop"


def compute_spread(span_char_ranges: list[tuple[int, int]], doc_char_len: int) -> float | None:
    """Fraction of the document spanned by the *gaps* between consecutive
    evidence spans (sorted by position), not the outer [first_start, last_end]
    range. Using the outer range alone would conflate dispersion with span
    length: a single long enumerated clause spanning 400 chars would
    otherwise look "more dispersed" than two short spans separated by the
    same physical gap, purely because of its own width. Summing only the
    text between consecutive spans isolates the actual non-evidence text a
    reader/model has to skip over to connect the evidence."""
    if not span_char_ranges or doc_char_len == 0:
        return None
    ranges = sorted(span_char_ranges)
    total_gap = sum(
        max(next_start - prev_end, 0)
        for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:])
    )
    return total_gap / doc_char_len


def classify_locality(spread: float | None, threshold: float) -> str:
    if spread is None:
        return "not_applicable"
    return "local" if spread <= threshold else "dispersed"


def classify_structure(span_texts: list[str]) -> str:
    if not span_texts:
        return "not_applicable"
    flags = [is_structured_span(t) for t in span_texts]
    if all(flags):
        return "structured"
    if not any(flags):
        return "unstructured"
    return "combining_structured_unstructured"


def process_split(split_name: str, data: dict, locality_threshold: float) -> list[ExampleRow]:
    labels = data["labels"]
    rows: list[ExampleRow] = []

    for doc in data["documents"]:
        doc_text = doc["text"]
        doc_char_len = len(doc_text)
        doc_tokens = count_tokens(doc_text)
        span_ranges: list[tuple[int, int]] = [tuple(s) for s in doc["spans"]]

        for annot_set in doc["annotation_sets"]:
            for hyp_id, annot in annot_set["annotations"].items():
                choice = annot["choice"]
                evidence_idx = annot["spans"]
                evidence_ranges = [span_ranges[i] for i in evidence_idx]
                evidence_texts = [doc_text[s:e] for s, e in evidence_ranges]
                evidence_char_len = sum(e - s for s, e in evidence_ranges)
                evidence_tokens = count_tokens(" ".join(evidence_texts)) if evidence_texts else 0

                hyp_text = labels[hyp_id]["hypothesis"]
                hyp_tokens = count_tokens(hyp_text)
                spread = compute_spread(evidence_ranges, doc_char_len)

                rows.append(
                    ExampleRow(
                        split=split_name,
                        doc_id=doc["id"],
                        file_name=doc["file_name"],
                        hypothesis_id=hyp_id,
                        hypothesis_text=hyp_text,
                        label=choice,
                        doc_char_len=doc_char_len,
                        doc_tokens=doc_tokens,
                        hypothesis_tokens=hyp_tokens,
                        num_evidence_spans=len(evidence_idx),
                        evidence_char_len=evidence_char_len,
                        evidence_tokens=evidence_tokens,
                        hop_type=classify_hop(len(evidence_idx)),
                        locality=classify_locality(spread, locality_threshold),
                        evidence_spread=spread,
                        structure_type=classify_structure(evidence_texts),
                        combined_tokens=doc_tokens + hyp_tokens,
                    )
                )
    return rows


def build_per_claim_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hypothesis (17 rows): claim text/length plus the share of
    its tested examples falling into each claim-dimension bucket, aggregated
    over every document/split it was annotated on."""
    records = []
    for hyp_id, sub in df.groupby("hypothesis_id"):
        hyp_text = sub["hypothesis_text"].iloc[0]
        n = len(sub)

        def pct(col: str, value: str) -> float:
            return round(100 * (sub[col] == value).sum() / n, 1)

        records.append(
            {
                "hypothesis_id": hyp_id,
                "hypothesis_text": hyp_text,
                "hypothesis_word_count": len(hyp_text.split()),
                "hypothesis_token_count": count_tokens(hyp_text),
                "n_examples": n,
                "pct_entailment": pct("label", "Entailment"),
                "pct_contradiction": pct("label", "Contradiction"),
                "pct_not_mentioned": pct("label", "NotMentioned"),
                "pct_single_hop": pct("hop_type", "single-hop"),
                "pct_multi_hop": pct("hop_type", "multi-hop"),
                "pct_local": pct("locality", "local"),
                "pct_dispersed": pct("locality", "dispersed"),
                "pct_structured": pct("structure_type", "structured"),
                "pct_unstructured": pct("structure_type", "unstructured"),
                "pct_combining_structured_unstructured": pct("structure_type", "combining_structured_unstructured"),
                "mean_evidence_tokens": round(sub.loc[sub["num_evidence_spans"] > 0, "evidence_tokens"].mean(), 1),
                "mean_num_evidence_spans": round(sub.loc[sub["num_evidence_spans"] > 0, "num_evidence_spans"].mean(), 2),
            }
        )
    out = pd.DataFrame(records)
    out["hyp_num"] = out["hypothesis_id"].str.split("-").str[1].astype(int)
    return out.sort_values("hyp_num").drop(columns="hyp_num").reset_index(drop=True)


def describe_series(s: pd.Series) -> dict:
    """Descriptive stats for one numeric (or 0/1 indicator) column.
    std/variance use ddof=1 (sample statistics), the standard choice for
    reporting dataset stats in a paper."""
    s = s.dropna()
    return {
        "n": int(s.count()),
        "mean": s.mean(),
        "std": s.std(),
        "variance": s.var(),
        "min": s.min(),
        "p25": s.quantile(0.25),
        "median": s.median(),
        "p75": s.quantile(0.75),
        "max": s.max(),
        "skewness": s.skew(),
        "kurtosis": s.kurt(),
    }


def stats_rows(data: pd.DataFrame, col: str, metric_name: str) -> list[dict]:
    rows = []
    groups = {sp: g[col] for sp, g in data.groupby("split")}
    groups["overall"] = data[col]
    for split, series in groups.items():
        rows.append({"metric": metric_name, "split": split, **describe_series(series)})
    return rows


def indicator_rows(data: pd.DataFrame, col: str, value: str, metric_name: str) -> list[dict]:
    tmp = data[["split"]].copy()
    tmp["_ind"] = (data[col] == value).astype(int)
    return stats_rows(tmp, "_ind", metric_name)


def build_summary_statistics(df: pd.DataFrame, doc_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # Document-level length stats (one row per document, not per example)
    rows += stats_rows(doc_df, "doc_char_len", "document_char_length")
    rows += stats_rows(doc_df, "doc_tokens", "document_token_length (cl100k_base)")

    # Claim/example-level length stats
    df = df.copy()
    df["hypothesis_word_count"] = df["hypothesis_text"].str.split().str.len()
    rows += stats_rows(df, "hypothesis_word_count", "hypothesis_word_length")
    rows += stats_rows(df, "hypothesis_tokens", "hypothesis_token_length")
    rows += stats_rows(df, "combined_tokens", "combined_doc+hypothesis_token_length")

    # Evidence stats only make sense for examples that actually have evidence
    # (Entailment/Contradiction); NotMentioned examples have 0 spans by construction.
    ev = df[df["num_evidence_spans"] > 0]
    rows += stats_rows(ev, "evidence_char_len", "evidence_char_length (evidence-bearing examples only)")
    rows += stats_rows(ev, "evidence_tokens", "evidence_token_length (evidence-bearing examples only)")
    rows += stats_rows(ev, "num_evidence_spans", "num_evidence_spans (evidence-bearing examples only)")

    # Class-balance stats as binary indicators: mean == proportion,
    # std == sqrt(p*(1-p)) (population) approximated here with sample ddof=1.
    for value in ["Entailment", "Contradiction", "NotMentioned"]:
        rows += indicator_rows(df, "label", value, f"label_is_{value}")
    for value in ["single-hop", "multi-hop", "not_applicable"]:
        rows += indicator_rows(df, "hop_type", value, f"hop_type_is_{value}")
    for value in ["local", "dispersed", "not_applicable"]:
        rows += indicator_rows(df, "locality", value, f"locality_is_{value}")
    for value in ["structured", "unstructured", "combining_structured_unstructured", "not_applicable"]:
        rows += indicator_rows(df, "structure_type", value, f"structure_type_is_{value}")

    out = pd.DataFrame(rows)
    split_order = {"train": 0, "dev": 1, "test": 2, "overall": 3}
    out["_split_order"] = out["split"].map(split_order)
    out = out.sort_values(["metric", "_split_order"]).drop(columns="_split_order").reset_index(drop=True)
    return out


def build_context_window_table(doc_df: pd.DataFrame) -> pd.DataFrame:
    """For each split x model context window, % of *documents* that would not
    fit in a single forward pass (doc_tokens alone, ignoring prompt/hypothesis
    overhead -> a lower bound on failure rate)."""
    records = []
    for split, sub in doc_df.groupby("split"):
        n = len(sub)
        for model, window in MODEL_CONTEXT_WINDOWS.items():
            n_fail = int((sub["doc_tokens"] > window).sum())
            records.append(
                {
                    "split": split,
                    "model": model,
                    "context_window": window,
                    "n_docs": n,
                    "n_exceeding": n_fail,
                    "pct_exceeding": round(100 * n_fail / n, 2) if n else 0.0,
                }
            )
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("."), help="dir containing train/dev/test.json")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    ap.add_argument("--locality-threshold", type=float, default=0.15,
                     help="max fraction of doc length between first/last evidence span to call it 'local'")
    ap.add_argument("--plot", action="store_true", help="also save matplotlib figures")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[ExampleRow] = []
    for split in ["train", "dev", "test"]:
        path = args.data_dir / f"{split}.json"
        if not path.exists():
            print(f"  skip {split}: {path} not found")
            continue
        data = load_split(path)
        rows = process_split(split, data, args.locality_threshold)
        print(f"{split}: {len(data['documents'])} docs, {len(rows)} claim examples")
        all_rows.extend(rows)

    df = pd.DataFrame([asdict(r) for r in all_rows])
    df.to_csv(args.out_dir / "examples.csv", index=False)
    print(f"\nWrote {len(df)} rows -> {args.out_dir / 'examples.csv'}")

    # One row per document (token length is doc-level, not example-level)
    doc_df = df.drop_duplicates(subset=["split", "doc_id"])[["split", "doc_id", "file_name", "doc_char_len", "doc_tokens"]]
    doc_df.to_csv(args.out_dir / "documents.csv", index=False)

    # --- Summary tables -----------------------------------------------------
    print("\n=== Token length summary (per split) ===")
    print(doc_df.groupby("split")["doc_tokens"].describe()[["count", "mean", "min", "50%", "max"]])

    print("\n=== Hop type distribution ===")
    print(df.groupby("split")["hop_type"].value_counts(normalize=True).round(3))

    print("\n=== Locality distribution ===")
    print(df.groupby("split")["locality"].value_counts(normalize=True).round(3))

    print("\n=== Structure type distribution ===")
    print(df.groupby("split")["structure_type"].value_counts(normalize=True).round(3))

    print("\n=== Label distribution ===")
    print(df.groupby("split")["label"].value_counts(normalize=True).round(3))

    stats_df = build_summary_statistics(df, doc_df)
    stats_df.to_csv(args.out_dir / "summary_statistics.csv", index=False)
    print(f"\nWrote summary statistics table -> {args.out_dir / 'summary_statistics.csv'}")

    claims_df = build_per_claim_table(df)
    claims_df.to_csv(args.out_dir / "claims_dimensions.csv", index=False)
    print(f"\nWrote per-claim dimension table ({len(claims_df)} claims) -> {args.out_dir / 'claims_dimensions.csv'}")

    ctx_df = build_context_window_table(doc_df)
    ctx_df.to_csv(args.out_dir / "context_window_failures.csv", index=False)
    print("\n=== % of documents exceeding model context window (doc tokens only) ===")
    print(ctx_df.pivot(index="model", columns="split", values="pct_exceeding").reindex(MODEL_CONTEXT_WINDOWS.keys()))

    # Cross-tab: claim difficulty (hop_type x locality x structure_type) vs label,
    # useful for the "where do models fail" section of the paper.
    cross = df.groupby(["hop_type", "locality", "structure_type"])["label"].count().rename("n_examples")
    cross.to_csv(args.out_dir / "claim_dimension_crosstab.csv")
    print(f"\nWrote claim-dimension crosstab -> {args.out_dir / 'claim_dimension_crosstab.csv'}")

    if args.plot:
        save_plots(df, doc_df, ctx_df, args.out_dir, args.locality_threshold)


def save_plots(df: pd.DataFrame, doc_df: pd.DataFrame, ctx_df: pd.DataFrame, out_dir: Path, locality_threshold: float):
    import matplotlib.pyplot as plt

    # 0. Evidence spread histogram (local vs dispersed), continuous metric.
    # Only examples with >=2 evidence spans have a meaningful spread (single-hop
    # claims trivially score ~0; NotMentioned has none).
    spread_df = df[df["num_evidence_spans"] >= 2]
    fig, ax = plt.subplots(figsize=(8, 5))
    local_mask = spread_df["evidence_spread"] <= locality_threshold
    ax.hist(spread_df.loc[local_mask, "evidence_spread"], bins=30, range=(0, 1),
            color="#4c72b0", alpha=0.85, label=f"local (<= {locality_threshold:.0%})")
    ax.hist(spread_df.loc[~local_mask, "evidence_spread"], bins=30, range=(0, 1),
            color="#dd8452", alpha=0.85, label=f"dispersed (> {locality_threshold:.0%})")
    ax.axvline(locality_threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"locality threshold ({locality_threshold:.0%})")
    ax.set_xlabel("Evidence gap (fraction of document length between evidence spans, excluding span width)")
    ax.set_ylabel("# multi-hop examples")
    ax.set_title("Local vs. dispersed evidence: distribution of evidence gap")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "locality_spread_histogram.png", dpi=150)
    plt.close(fig)

    # 0b. Simple bar chart: local vs dispersed share per split (excludes not_applicable,
    # i.e. NotMentioned/single-hop examples where locality doesn't apply).
    loc_df = df[df["locality"].isin(["local", "dispersed"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ct = pd.crosstab(loc_df["split"], loc_df["locality"], normalize="index")
    ct = ct.reindex(columns=["local", "dispersed"])
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("fraction of evidence-bearing examples", fontsize=9)
    ax.set_title("Local vs. dispersed evidence share by split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "locality_local_vs_dispersed_bar.png", dpi=150)
    plt.close(fig)

    # 1. Document token-length histogram with model context windows overlaid
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(doc_df["doc_tokens"], bins=40, color="#4c72b0", alpha=0.85)
    colors = plt.cm.tab10.colors
    for i, (model, window) in enumerate(MODEL_CONTEXT_WINDOWS.items()):
        if window <= doc_df["doc_tokens"].max() * 1.05:
            ax.axvline(window, color=colors[i % len(colors)], linestyle="--", linewidth=1, label=f"{model} ({window:,})")
    ax.set_xlabel("Document tokens (cl100k_base)")
    ax.set_ylabel("# documents")
    ax.set_title("ContractNLI document length vs. model context windows")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "doc_token_hist_vs_context_windows.png", dpi=150)
    plt.close(fig)

    # 2. Claim dimension distributions (stacked bar per split)
    for dim in ["hop_type", "locality", "structure_type"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ct = pd.crosstab(df["split"], df[dim], normalize="index")
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("fraction of examples")
        ax.set_title(f"{dim} distribution by split")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dim}_distribution.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots -> {out_dir}")


if __name__ == "__main__":
    main()
