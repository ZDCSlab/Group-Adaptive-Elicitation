# Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions

![Teaser](/assets/gae_teaser.jpg "Teaser")


This repository contains code for the paper **Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions**.

We study adaptive group elicitation, a framework that jointly selects which questions to ask and which respondents to query under limited budgets. Our approach combines an LLM-based expected information gain objective with heterogeneous graph neural network propagation to impute missing responses and guide respondent selection.

> 🚧 **This repository is currently under active development.** 🚧



## Overview
- **LLM**: Predictor pool for question-level predictions; used for query selection (e.g. information gain, MCTS lookahead) and optional imputation.
- **GNN**: Heterogeneous graph of users, question-option nodes, and demographic subgroups (RGCN). Used to score users by uncertainty (entropy, margin, etc.) for adaptive node selection.
- **Inference**: Two entrypoints—**baseline** (LLM-only query selection) and **GAE** (GNN + LLM hybrid with node and query selection strategies).

## Project structure

TBD

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

## Setup

From the project root:

```bash
# Install PyTorch and PyTorch Geometric (see pytorch.org / pyg.org for your CUDA version).
# Then typical dependencies (infer from imports):
pip install torch torch-geometric pandas numpy pyyaml scikit-learn tqdm wandb accelerate
```

Ensure the package is on `PYTHONPATH` when running scripts:

```bash
export PYTHONPATH=/path/to/Group-Adaptive-Elicitation:$PYTHONPATH
```

## Data

TBD


## Usage

### 1. Train the GNN

Requires a YAML config with `data`, `model`, `train`, `optim`, `split`, `logs`, `checkpoint` (see `src/gnn/utils.py` and `src/gnn/dataset.py`).

```bash
python scripts/run_gnn_train.py --config /path/to/gnn_config.yaml
# Optional overrides:
python scripts/run_gnn_train.py --config /path/to/config.yaml --set train.epochs=100 optim.lr=0.001
```

### 2. Meta-train the LLM

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
TBD
```