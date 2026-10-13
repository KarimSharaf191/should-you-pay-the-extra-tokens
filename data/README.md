# Data

## ContractNLI (included)

`contractnli/` contains the three official splits from
[stanfordnlp/contract-nli](https://stanfordnlp.github.io/contract-nli/)
(Koreeda and Manning, Findings of EMNLP 2021), released under CC BY 4.0 by
Hitachi America, Ltd. See `contractnli/LICENSE` and `contractnli/TERMS`.

| File | Contents |
|---|---|
| `dev.json` | 61 contracts, 1,037 (contract, hypothesis) claims; the evaluation split |
| `test.json` | 123 contracts, 2,091 claims |
| `train.json` | 423 contracts, 7,191 claims; used only for dataset statistics |

Each document has the contract `text`, its `spans` (character offsets), and
`annotation_sets[0].annotations[<hypothesis id>]` with the gold `choice`
(Entailment / Contradiction / NotMentioned) and gold evidence `spans`.

## FinDVer (downloaded)

```bash
bash data/get_findver.sh
```

Clones [yilunzhao/FinDVer](https://github.com/yilunzhao/FinDVer) (MIT) at the
commit used for the paper into `data/findver/`. We use the `testmini` split
(700 claims), the only split with public labels.

## CoverBench (downloaded, not redistributed)

```bash
pip install huggingface_hub
huggingface-cli login            # after accepting the dataset terms on Hugging Face
python data/get_coverbench.py
```

Downloads `eval/coverbench.json` from the gated
[google/coverbench](https://huggingface.co/datasets/google/coverbench) release
into `data/coverbench/`. CoverBench is licensed CC BY-ND 4.0 and its terms ask
that it not be uploaded anywhere web crawlers can reach, so it is git-ignored
here, and CoverBench records in `results/` contain no CoverBench text.

## Retrieval indexes

RAG and agentic RAG on FinDVer and CoverBench use per-document indexes in
`data/findver_index/` and `data/coverbench_index/` (git-ignored). Build the
FinDVer index with `python -m src.findver_coverbench.build_index --benchmark findver`.
