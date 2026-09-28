# Should You Pay the Extra Tokens?

**A Cost-Matched Evaluation of Inference-Time Scaffolds for Long-Context Claim Verification with Small LLMs**

Code, data and per-instance results for the paper.

Inference-time scaffolds such as chain-of-thought, native reasoning, retrieval and agentic loops are usually compared on accuracy alone. We compare them on accuracy *per token*. For small open-weight LLMs (under 10B parameters), which scaffolds sit on the cost–accuracy frontier, and which spend extra tokens without earning them?

| | |
|---|---|
| **Benchmarks** | ContractNLI (legal), CoverBench (multi-domain), FinDVer (financial) |
| **Models** | Qwen3 1.7B, 4B, 8B; thinking mode off and on |
| **Scaffolds** | zero-shot, DSPy, chain-of-thought, RAG, agentic RAG, recursive LM (RLM) |
| **Metrics** | balanced accuracy, macro-F1, Cohen's κ, input and output tokens per instance |

## Repository layout

```
paper/          LaTeX source of the paper
src/            evaluation code (all scripts run from the repository root)
data/
  contractnli/  ContractNLI splits (CC BY 4.0) and the pinned n=150 sample
results/        per-instance predictions and the summary files behind each table
notebooks/      Kaggle notebook for the Qwen3-1.7B / 4B runs
```

## Setup

Python 3.12.

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # only needed to run new experiments
```

Rebuilding tables from the saved results needs no API key.

## Reproducing the paper's tables

Every command runs from the repository root and reads the saved records in `results/`.

| Paper element | Command | Output |
|---|---|---|
| ContractNLI main table (10 conditions) | `python src/run_matrix.py --report-only --n 1037` and again with `--thinking` | `results/matrix[-think]_dev_n1037_*.txt` |
| Token and USD cost per condition | `python src/token_cost_report.py` | `results/token_cost_dev_n1037.txt` |
| Paired bootstrap significance tests | `python src/bootstrap_test.py --n 1037 --metric bal_acc` (and `--metric macro_f1`) | `results/bootstrap_*_dev_n1037.txt` |
| RLM behaviour table | Aggregates saved in `results/rlm-multi/summary.json`; re-run with `python src/rlm_multi.py --benchmark <contractnli\|coverbench\|findver> --n 20` (makes API calls) | `results/rlm-multi/<benchmark>/` |
| Faithfulness sub-study | `python src/faithfulness_study.py --report-only` | `results/faithfulness/faithfulness_report_n150.txt` |

`results/summaries/` holds the exact report files the paper's numbers were taken from.

## Running experiments

Each run appends one JSON line per instance and resumes where it stopped.

```bash
# Qwen3-8B via OpenRouter, all 5 scaffolds, full ContractNLI dev split
python src/run_matrix.py --models qwen/qwen3-8b --n 1037
python src/run_matrix.py --models qwen/qwen3-8b --n 1037 --thinking

# Any OpenAI-compatible server, e.g. a local vLLM
python src/run_matrix.py --models Qwen/Qwen3-4B --base-url http://localhost:8000/v1 --api-key EMPTY \
    --n 150 --manifest data/contractnli/sample_manifest_dev_n150_seed0.csv
```

| Script | Purpose |
|---|---|
| `contractnli_zeroshot.py` | Prompts and predictors for every scaffold; token accounting |
| `run_matrix.py` | Runs the model × scaffold matrix (thinking off or on); builds the summary tables |
| `run_rlm_contractnli.py` | Data loading, stratified sampling, label parsing; RLM runner for ContractNLI |
| `rlm_multi.py`, `run_rlm_traced.py`, `rlm_utf8_patch.py`, `analyze_rlm.py` | Recursive LM runs across the three benchmarks, with sub-call tracing |
| `faithfulness_study.py` | Faithfulness sub-study on ContractNLI |
| `faithfulness_multi.py`, `faithfulness_multi_support.py` | Faithfulness sub-study on FinDVer and CoverBench |
| `bootstrap_test.py` | Paired bootstrap tests with Holm correction |
| `token_cost_report.py` | Token and cost tables |
| `compare_thinking.py` | Thinking on vs. off comparison |
| `benchmark_analysis.py` | Dataset statistics (length, hops, evidence locality) |
| `run_driver.py` | Serves Qwen3-1.7B / 4B with vLLM and runs the matrix (used by the Kaggle notebook) |

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | none | API key for OpenRouter |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `CONTRACTNLI_DATA_DIR` | `data/contractnli` | Location of the ContractNLI split files |
| `CONTRACTNLI_RESULTS_DIR` | `results` | Where records and reports are written |
| `FINDVER_DIR` | `data/findver` | FinDVer checkout (see `data/README.md`) |
| `COVERBENCH_PATH` | `data/coverbench/coverbench.json` | CoverBench file (see `data/README.md`) |

## Qwen3-1.7B and 4B

These models are no longer served by OpenRouter, so they are self-hosted with vLLM. `notebooks/kaggle_qwen_small.ipynb` runs them on a free Kaggle "GPU T4 ×2" session. Zip `src/*.py`, `data/contractnli/dev.json` and `data/contractnli/sample_manifest_dev_n150_seed0.csv`, upload the zip as a Kaggle Dataset, and follow the notebook.

## What is and is not included

- **Included:** all Qwen3-8B ContractNLI records (full dev split, n=1037, and the n=150 sample), faithfulness records, and RLM records for all three benchmarks.
- **Not included:** FinDVer and CoverBench data (see `data/README.md` for where to get them), and the FinDVer / CoverBench scaffold results reported in the appendix tables.

## Data licenses

ContractNLI is © Hitachi America, Ltd., released under CC BY 4.0 (`data/contractnli/LICENSE`, `data/contractnli/TERMS`). FinDVer and CoverBench are distributed by their authors under their own terms.

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
