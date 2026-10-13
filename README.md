# Should You Pay the Extra Tokens?

**A Cost-Matched Evaluation of Inference-Time Scaffolds for Long-Context Claim Verification with Small LLMs**

Code, data and per-claim results for the paper. We compare inference-time
scaffolds on balanced accuracy *and* on the tokens they consume, with input
and generated tokens counted separately.

| | |
|---|---|
| **Benchmarks** | ContractNLI (dev, 1,037 claims), FinDVer (testmini, 700), CoverBench (733) |
| **Models** | Qwen3-1.7B, 4B, 8B with thinking off and on; Llama-3.2-3B-Instruct (no thinking mode) |
| **Scaffolds** | zero-shot (`raw`), typed DSPy prompt (`typed`, called `dspy` in file names), chain-of-thought (`cot`), RAG, agentic RAG, recursive language model (RLM) |
| **Metrics** | balanced accuracy, macro-F1, Cohen's κ; input and generated tokens per claim |

## Repository layout

```
src/                         ContractNLI scaffolds, RLM and faithfulness probes (all benchmarks)
src/findver_coverbench/      FinDVer and CoverBench scaffolds (package; python -m ...)
src/analysis.py              confidence intervals, paired tests, decoding check (from saved records)
scripts/                     drivers for the self-hosted (vLLM) runs on a single GPU
notebooks/                   Kaggle notebook for the Qwen3-1.7B / 4B ContractNLI runs
data/                        ContractNLI (included); download scripts for FinDVer and CoverBench
results/                     per-claim records and the report files behind every table
paper/                       LaTeX source of the paper
```

## Setup

Python 3.12.

```bash
pip install -r requirements.txt
bash data/get_findver.sh                  # FinDVer, MIT license
python data/get_coverbench.py             # CoverBench, gated: accept its terms on Hugging Face first
export OPENROUTER_API_KEY=...             # only for new API runs
```

Everything under *Reproducing the tables* works from the saved records, with no
API key and no GPU.

## Reproducing the tables and figures

| Paper element | How | Source |
|---|---|---|
| Table 2 (BA for every model, scaffold, benchmark), ContractNLI columns | `python src/run_matrix.py --report-only --n 1037 --models qwen/qwen3-8b Qwen/Qwen3-4B Qwen/Qwen3-1.7B meta-llama/llama-3.2-3b-instruct` (add `--thinking` for thinking on) | `results/matrix[-think]-<scaffold>/<model>/` |
| Table 2, FinDVer and CoverBench columns | `python -m src.findver_coverbench.run ...` (see below); report rows in `results/<benchmark>/summary.csv` | per-claim records produced by the run |
| Table 2, RLM row; RLM behaviour table | `python src/analysis.py` | `results/rlm-multi/` |
| Figure 3 (accuracy vs. tokens), Figure 4 (tokens by contract length) | `python src/analysis.py` and the token tables | `results/matrix*/`, `results/summaries/` |
| Confidence intervals and paired tests | `python src/analysis.py` | `results/summaries/analysis.txt` |
| Evidence ablation, agentic stopping, CoT quotes (RQ3) | `python src/faithfulness_study.py --n 1037 --report-only`; `python src/faithfulness_multi.py --benchmark findver --n 700 --report-only` (and `coverbench --n 733`) | `results/faithfulness*/` |
| Llama retrieval rows on FinDVer and CoverBench | `python src/rag_multi.py --benchmark findver --model meta-llama/llama-3.2-3b-instruct --report-only` | `results/rag-multi/` |
| Decoding check (greedy vs. sampling) | `python src/analysis.py` | `results/matrix-think-s{1,2,3}-zeroshot/` |

`results/summaries/` holds the report files the paper's numbers were taken from.

## Running experiments

All runs append one JSON line per claim and resume where they stopped.

### ContractNLI scaffolds (`src/run_matrix.py`)

```bash
# Any OpenAI-compatible endpoint: OpenRouter, or a local vLLM server
python src/run_matrix.py --models meta-llama/llama-3.2-3b-instruct --n 1037
python src/run_matrix.py --models Qwen/Qwen3-4B --n 1037 --thinking \
    --base-url http://localhost:8000/v1 --api-key EMPTY
```

Qwen3 models are no longer served by OpenRouter; serve them with vLLM
(`vllm serve Qwen/Qwen3-8B --enable-prefix-caching`) and pass `--base-url`.
`notebooks/kaggle_qwen_small_full_dev.ipynb` runs Qwen3-1.7B and 4B on a free
Kaggle GPU session; zip `src/*.py` and `data/contractnli/dev.json` as a Kaggle
dataset and follow the notebook.

### FinDVer and CoverBench scaffolds (`src/findver_coverbench/`)

```bash
# full-document scaffolds
python -m src.findver_coverbench.run --benchmark findver --models qwen/qwen3-8b \
    --scaffolds zeroshot typed cot --thinking off on --base-url http://localhost:8000/v1
# local Hugging Face models (one GPU)
python -m src.findver_coverbench.run --benchmark coverbench --models Qwen/Qwen3-4B Qwen/Qwen3-1.7B
# retrieval scaffolds need an index first
python -m src.findver_coverbench.build_index --benchmark findver
python -m src.findver_coverbench.run --benchmark findver --models qwen/qwen3-8b --scaffolds rag agentic_rag
# Llama-3.2-3B-Instruct (no thinking mode)
python -m src.findver_coverbench.run --benchmark coverbench --models meta-llama/llama-3.2-3b-instruct \
    --scaffolds zeroshot typed cot
```

| Module | Purpose |
|---|---|
| `prompts.py` | Prompts and DSPy signatures for both benchmarks |
| `benchmarks.py` | Data loading, document truncation, per-call output caps |
| `backends.py` | API backend (OpenRouter / vLLM) and Hugging Face backend; per-model decoding settings |
| `retrieval.py`, `build_index.py` | Hybrid BM25 + Qwen3-Embedding-4B retrieval with reciprocal rank fusion |
| `scaffolds.py` | zero-shot, typed, CoT, RAG and agentic RAG runners |
| `parsing.py`, `metrics.py` | Label parsing and scores |
| `run.py` | Command line entry point |

### RLM and faithfulness probes with Qwen3-8B

`scripts/run_qwen3_8b_rlm_faithfulness.py` serves Qwen3-8B with vLLM on one GPU
and runs RLM on all three benchmarks and the faithfulness probes, in parallel
and resumable. It needs the FinDVer and CoverBench data in `data/`.
`scripts/run_decoding_check.py` reruns thinking-on conditions with sampling.

| Script | Purpose |
|---|---|
| `contractnli_zeroshot.py` | Prompts, predictors and token accounting for the ContractNLI scaffolds |
| `run_matrix.py` | Model × scaffold matrix on ContractNLI (thinking off or on) and its report |
| `run_rlm_contractnli.py`, `rlm_multi.py`, `run_rlm_traced.py`, `rlm_utf8_patch.py` | RLM on the three benchmarks, with per-call tracing |
| `faithfulness_study.py`, `faithfulness_multi.py`, `faithfulness_multi_support.py` | Evidence ablation, agentic stopping and CoT quote probes |
| `rag_multi.py` | Llama retrieval runs on FinDVer and CoverBench |
| `analysis.py`, `bootstrap_test.py` | Confidence intervals and paired bootstrap tests |
| `run_driver.py` | vLLM driver for the Kaggle notebook |

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | none | API key for OpenRouter |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint for the ContractNLI scripts |
| `CONTRACTNLI_DATA_DIR`, `CONTRACTNLI_RESULTS_DIR` | `data/contractnli`, `results` | Data and output folders |
| `FINDVER_DIR`, `COVERBENCH_FILE` | `data/findver`, `data/coverbench/coverbench.json` | Downloaded benchmarks |
| `CONTRACTNLI_TEMPERATURE`, `CONTRACTNLI_TOP_P`, `CONTRACTNLI_TOP_K`, `CONTRACTNLI_SAMPLE_SEED` | greedy | Sampling for the decoding check |
| `CONTRACTNLI_MAX_TOKENS`, `RLM_MAX_TOKENS` | none | Per-call output cap for self-hosted runs |

## Decoding and serving

API and vLLM runs use greedy decoding (temperature 0). The Hugging Face runs of
Qwen3-4B and 1.7B on FinDVer and CoverBench sample with thinking on
(temperature 0.6, top-p 0.95, top-k 20, seed 42), as Qwen3 recommends.
Self-hosted runs used vLLM on one NVIDIA A100 80GB GPU.

## What is included

- **Included:** every per-claim record for ContractNLI (all models and conditions), RLM on the three benchmarks, the faithfulness probes, the Llama retrieval runs, and the decoding check.
- **Not included:** FinDVer and CoverBench data (download scripts in `data/`); per-claim records of the FinDVer and CoverBench scaffold runs reported in Table 2; the script that chunks CoverBench documents for retrieval.
- **CoverBench text is never stored.** CoverBench records keep only ids, labels, predictions and token counts; model rationales and answers, which may quote CoverBench documents, are removed.

## Licenses

Code: MIT (`LICENSE`). ContractNLI: CC BY 4.0, © Hitachi America, Ltd.
(`data/contractnli/LICENSE`). FinDVer: MIT, downloaded from its authors'
repository. CoverBench: CC BY-ND 4.0, gated; its terms ask that it not be
uploaded where web crawlers can reach it, so it is not redistributed here.

## Citation

```bibtex
@misc{shouldyoupay2026,
  title  = {Should You Pay the Extra Tokens? A Cost-Matched Evaluation of Inference-Time
            Scaffolds for Long-Context Claim Verification with Small LLMs},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
