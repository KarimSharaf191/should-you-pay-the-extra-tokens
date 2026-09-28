# Data

## ContractNLI (included)

`contractnli/` contains the three official splits from
[stanfordnlp/contract-nli](https://stanfordnlp.github.io/contract-nli/) (Koreeda and Manning, Findings of EMNLP 2021), released under CC BY 4.0 by Hitachi America, Ltd. See `contractnli/LICENSE` and `contractnli/TERMS`.

| File | Contents |
|---|---|
| `dev.json` | 61 contracts, 1,037 (contract, hypothesis) instances; the evaluation split |
| `test.json` | 123 contracts, 2,091 instances |
| `train.json` | 423 contracts, 7,191 instances; used only for dataset statistics |
| `sample_manifest_dev_n150_seed0.csv` | The 150-instance label-stratified dev sample (seed 0) used for the faithfulness study and the smaller-model runs |

Each document has the contract `text`, its `spans` (character offsets), and `annotation_sets[0].annotations[<hypothesis id>]` with the gold `choice` (Entailment / Contradiction / NotMentioned) and gold evidence `spans` (indices into the document's span list). The 17 hypotheses are in the top-level `labels` object.

## FinDVer (not included)

Clone the authors' repository ([yilunzhao/FinDVer](https://github.com/yilunzhao/FinDVer)) to `data/findver/`, or set `FINDVER_DIR`. The code expects `data/<split>.json` and the `financial_reports/` folder inside it. The paper uses the `testmini` split (700 claims), the only split with public labels.

## CoverBench (not included)

Request access to the gated Hugging Face release ([Jacovi et al., 2024](https://arxiv.org/abs/2408.03325)) and save its `eval/coverbench.json` as `data/coverbench/coverbench.json`, or set `COVERBENCH_PATH`.

CoverBench is licensed CC BY-ND 4.0, and its terms ask that it not be uploaded anywhere web crawlers can reach. Keep it out of this repository; `.gitignore` already excludes `data/coverbench/`. The CoverBench records in `results/rlm-multi/` contain only instance ids, labels and token counts, no CoverBench text.
