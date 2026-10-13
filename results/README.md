# Results

One JSON line per claim. Every file can be re-scored offline (see the table in
the top-level README); `summaries/` holds the reports the paper's numbers come from.

| Folder | Contents |
|---|---|
| `matrix-<scaffold>/<model>/records_dev_n1037_seed0.jsonl` | ContractNLI, thinking off (Llama has no thinking mode). Scaffolds: `zeroshot`, `dspy` (typed), `cot`, `rag`, `agentic_rag` |
| `matrix-think-<scaffold>/<model>/` | ContractNLI, Qwen3 with thinking on (greedy) |
| `matrix-think-s{1,2,3}-zeroshot/qwen3-8b/` | Decoding check: Qwen3-8B zero-shot with thinking, sampled with seeds 1-3 |
| `matrix-zeroshot/qwen3-8b-vllm/` | Qwen3-8B zero-shot served with vLLM, as a check against the API run |
| `rlm-multi/<benchmark>/` | RLM with Qwen3-8B (thinking off) on the three benchmarks |
| `faithfulness/` | ContractNLI probes: evidence ablation (study 1), retrieval (2), agentic stopping (3), CoT quotes (4) |
| `faithfulness-multi/<benchmark>/` | The same probes on FinDVer and CoverBench (CoverBench has no gold evidence, so only studies 3 and 4) |
| `rag-multi/<benchmark>/llama-3.2-3b-instruct/` | Llama-3.2-3B-Instruct RAG and agentic RAG on FinDVer and CoverBench |
| `summaries/` | Matrix reports per model, bootstrap tests, and `analysis.txt` (CIs, paired tests, RLM, token use by length, decoding check) |

## Record fields (ContractNLI matrix)

`doc_id`, `nda_key` (hypothesis id), `gold`, `pred`, `parsed_ok`, `error`,
`elapsed_s`, and `usage` with `prompt` (input), `completion` (generated,
including thinking), `reasoning`/`think_tokens` (thinking part) and `total`.
Retrieval scaffolds add the retrieved spans and tool calls. Token counts are
those reported by the serving endpoint.

## CoverBench records

CoverBench records keep ids, gold labels, predictions and token counts only.
Model outputs (rationales, answers, raw text) were removed because they can
quote CoverBench documents, which may not be redistributed.
