# Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions

![Teaser](/assets/gae_teaser.jpg "Teaser")


This repository contains code for the paper [Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions](https://arxiv.org/pdf/2602.14279) by Ruomeng Ding*, Tianwei Gao*, Thomas P. Zollo, Eitan Bachmat, Richard Zemel, and Zhun Deng.

We study adaptive group elicitation, a framework that jointly selects which questions to ask and which respondents to query under limited budgets. Our approach combines an LLM-based expected information gain objective with heterogeneous graph neural network propagation to impute missing responses and guide respondent selection.

> 🚧 **This repository is currently under active development.** 🚧



## Overview
- **LLM**: Predictor pool for question-level predictions; used for query selection (e.g. information gain, MCTS lookahead) and optional imputation.
- **GNN**: Heterogeneous graph of users, question-option nodes, and demographic subgroups (RGCN). Used to score users by uncertainty (entropy, margin, etc.) for adaptive node selection.
- **Inference**: Two entrypoints—**baseline** (LLM-only query selection) and **GAE** (GNN + LLM hybrid with node and query selection strategies).

## Project structure

```
Group-Adaptive-Elicitation/
├── scripts/
│   ├── run_gnn_train.py      # Train GNN from config (YAML)
│   ├── run_meta_train.py     # Meta-train LLM (e.g. Llama) with accelerate
│   ├── run_inference_baseline.py  # Adaptive rounds, LLM query selection only
│   ├── run_inference_gae.py       # Adaptive rounds, GNN node + LLM query selection
│   ├── gnn_train.sh
│   └── meta_train.sh
└── src/
    ├── gnn/                  # Graph model and data
    │   ├── dataset.py        # QAGraph, splits, load_qa_graph, build_graph_from_raw
    │   ├── model.py          # GEMSModel (RGCN)
    │   ├── train.py          # Training loop
    │   └── utils.py          # Config, overrides, build_graph_from_raw
    ├── inference/
    │   ├── dataset.py        # Survey state / candidate questions
    │   ├── model.py          # LLM PredictorPool
    │   ├── evaluation.py    # Evaluate on held-out / hard groups
    │   ├── gnn_predictor.py  # GNNElicitationPredictor
    │   ├── select_node.py    # NodeSelector (entropy, margin, diversity, …)
    │   ├── select_query.py   # select_queries (info_gain, etc.)
    │   ├── select_query_lookahaed.py  # MCTS lookahead
    │   ├── utils_gnn.py      # GNN helpers, gold answers from edges
    │   └── utils_llm.py      # LLM helpers, imputation, codebook
    └── meta_train/           # LLM meta-training
        ├── args.py
        ├── dataset.py
        ├── train.py
        └── training_utils.py
```


## Set Up

1. Clone Group-Adaptive-Elicitation repository.
```bash
    git clone https://github.com/ZDCSlab/Group-Adaptive-Elicitation.git
    cd Group-Adaptive-Elicitation
```

2. Create the environment.

```bash
    conda create -n gae python=3.9
    conda activate gae
    pip install -r requirements.txt
```


## Datasets

This project utilizes three primary datasets. To ensure the preprocessing scripts function correctly, please download the raw data and organize it according to the structure specified below.

---

### 1. Download Sources

| Dataset | Focus | Source Link |
| :--- | :--- | :--- |
| **CES** | Cooperative Election Study | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CETPVT) |
| **OpinionQA** | Pew Research Public Opinion (W50, W54, W92) | [Pew Research Center](https://www.pewresearch.org/american-trends-panel-datasets/) |
| **Twin-2k** | LLM Digital Twin Dataset | [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) |

---

### 2. Setup Instructions

#### 2.1 Place the raw dataset files under: `dataset/{ces,opinionqa,twin}/raw_data`:
```
Group-Adaptive-Elicitation/
└── dataset/
    ├── ces/
    │   └── raw_data/
    │       └── merged_recontact_2024.dta
    │
    ├── opinionqa/
    │   └── raw_data/
    │       ├── ATP W50.sav
    │       ├── ATP W54.sav
    │       └── ATP W92.sav
    │
    └── twin/
        └── raw_data/
            └── Twin-2K-500
```
#### 2.2 Run the following command to preprocess all datasets and generate the processed features:
```
sh dataset/dataset.sh
```

This script will:
- Preprocess the CES, OpinionQA, and Twin-2K datasets
- Generate cleaned feature matrices
- Construct train/validation/test user splits
- Save data features under `dataset/{ces,opinionqa,twin}/data/`
- Prepare meta-training sequences under `dataset/{ces,opinionqa,twin}/processed_data/`


## Usage

### 1. Train the Heterogeneous GNNs

Requires a YAML config with `data`, `model`, `train`, `optim`, `split`, `logs`, `checkpoint` (see `scripts/args_gnn/config_{ces,opinionqa,twin}.yaml`). You can launch training via the provided shell script:

```bash
sh scripts/gnn_train.sh
```
Or run a specific dataset configuration manually. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_gnn_train.py --config scripts/args_gnn/config_ces.yaml
```

### 2. Meta-train the LLMs

Uses Hugging Face `accelerate` and model/dataset args (see `scripts/run_meta_train.py` and `src/meta_train/args.py`).

```bash
accelerate launch --config_file /path/to/accelerate_config.yaml \
  scripts/run_meta_train.py \
  --root_data_dir=/path/to/dataset \
  --option_dict_path=/path/to/codebook.jsonl \
  --dataset=opinionQA \
  --model_name=Llama-3.1-8B \
  --save_dir=/path/to/logs \
  --wandb
```

Adjust paths and `model_name` to your setup; `run_meta_train.py` loads `scripts/model_args/{model_name}.yaml` if present.

### 3. Run inference

**Baseline (LLM-only query selection):**

```bash
python scripts/run_inference_baseline.py \
  --llm_checkpoint /path/to/llm \
  --query_selection info_gain \
  --node_selection entropy \
  --node_selection_prec 0.1 \
  --runs /path/to/runs.csv \
  --dataset /path/to/dataset \
  --infer_data /path/to/infer_data.csv
```

**GAE (GNN + LLM):**

```bash
python scripts/run_inference_gae.py \
  --llm_checkpoint /path/to/llm \
  --gnn_config_path /path/to/gnn_config.yaml \
  --query_selection info_gain \
  --node_selection entropy \
  --node_selection_prec 0.1 \
  --runs /path/to/runs.csv \
  --dataset /path/to/dataset \
  --infer_data /path/to/infer_data.csv
```

Use `--imputation` and `--impute_thres` for LLM-based imputation. Query selection can be `random`, `mean_entropy`, `info_gain`, or `mcts_lookahead` (GAE). Node selection can be `random`, `entropy`, `margin`, `full`, `info_gain`, `entropy_diversity`, etc., depending on the script.


## Cite Our Work
```
@misc{ding2026querywhatadaptivegroup,
      title={Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions}, 
      author={Ruomeng Ding and Tianwei Gao and Thomas P. Zollo and Eitan Bachmat and Richard Zemel and Zhun Deng},
      year={2026},
      eprint={2602.14279},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.14279}, 
}
```