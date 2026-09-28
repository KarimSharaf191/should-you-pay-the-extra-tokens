# Results

All records are for **Qwen3-8B**, greedy decoding (temperature 0), served through OpenRouter.

## Per-instance records

`matrix-<scaffold>/qwen3-8b/` holds thinking-off runs; `matrix-think-<scaffold>/qwen3-8b/` holds thinking-on runs. Scaffolds: `zeroshot`, `dspy`, `cot`, `rag`, `agentic_rag`.

| File | Instances |
|---|---|
| `records_dev_n1037_seed0.jsonl` | Full ContractNLI dev split (used for the main tables) |
| `records_dev_n150_seed0.jsonl` | The pinned 150-instance sample (`data/contractnli/sample_manifest_dev_n150_seed0.csv`) |

One JSON object per line:

| Field | Meaning |
|---|---|
| `doc_id`, `nda_key` | Contract id and hypothesis id |
| `gold`, `pred` | Gold and predicted label |
| `parsed_ok` | Whether the model's output parsed to a valid label |
| `usage` | `prompt`, `completion`, `reasoning`, `think_tokens`, `total` tokens; `cost_usd` (from OpenRouter); `steps` and `retrieved_spans` for the retrieval scaffolds |
| `elapsed_s`, `error` | Wall time; error message if the call failed |

## Other folders

| Folder | Contents |
|---|---|
| `faithfulness/` | Faithfulness sub-study (ContractNLI, n=150): per-instance records for each study and the report |
| `rlm-multi/` | Recursive LM runs on ContractNLI, CoverBench and FinDVer (label-stratified, about 20 instances each) |
| `summaries/` | The exact report files the paper's numbers were taken from: matrix tables, token cost, bootstrap tests, thinking comparison |

Re-running any report command in the top-level README writes a new, timestamped file here; the saved records are not modified.
