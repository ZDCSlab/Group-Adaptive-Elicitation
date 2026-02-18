# Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions

![Teaser](/assets/gae_teaser.jpg "Teaser")


This repository contains code for the paper [Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions](https://arxiv.org/pdf/2602.14279) by Ruomeng Ding*, Tianwei Gao*, Thomas P. Zollo, Eitan Bachmat, Richard Zemel, and Zhun Deng.

We study *adaptive group elicitation*, a multi-round decision-making framework in which a system selects both which questions to ask and which respondents to query under explicit query and participation budgets. Unlike prior approaches that optimize question selection over a fixed respondent pool, our method integrates:

- an LLM-based expected information gain objective to score and select candidate questions, and

- a heterogeneous graph neural network (HGNN) to aggregate observed responses and participant attributes, impute missing responses, and guide respondent selection through learned population structure.

This closed-loop procedure queries a small, informative subset of individuals while recovering population-level response patterns from sparse and incomplete observations.

## Project Structure

```
Group-Adaptive-Elicitation/
├── scripts/
│   ├── run_gnn_train.py           # Train GNN from YAML config
│   ├── run_meta_train.py          # Meta-train LLM (e.g. Llama) with Accelerate
│   ├── run_inference_baseline.py  # Adaptive elicitation, LLM-only (no GNN)
│   ├── run_inference_gae.py       # Adaptive elicitation, Ours (GNN + LLM)
│   ├── run_gnn_train.sh
│   ├── run_meta_train.sh
│   ├── run_inference_baseline.sh
│   ├── run_inference_gae.sh
│   ├── args_gnn/                  # GNN configs
│   ├── args_llm/                  # LLM model configs
│   └── runs/                      # Query/eval splits per dataset
└── src/
    ├── gnn/                       # Heterogeneous GNN (user–question graph)
    │   ├── dataset.py             # QAGraph, build_loaders_for_epoch...
    │   ├── model.py               # HGNNModel (RGCN encoder + decoder)
    │   ├── train.py               # GNN training loop
    │   └── utils.py          
    ├── inference/                 # Adaptive elicitation 
    │   ├── dataset.py             # Dataset, SurveyGraph...
    │   ├── model.py               # PredictorPool (multi-GPU LLM inference)
    │   ├── evaluation.py          # Held-out evaluation (accuracy, Brier score, PPL)
    │   ├── gnn_predictor.py       # GNNElicitationPredictor...
    │   ├── select_query.py        # select_queries (info_gain), select_queries_mcts (MCTS lookahead)
    │   └── utils.py              
    └── meta_train/                # LLM meta-training
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

| Dataset | Description | Source Link |
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
sh scripts/run_gnn_train.sh
```
Or run a specific dataset configuration manually:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_gnn_train.py --config scripts/args_gnn/config_{dataset}.yaml
```

The trained model checkpoints will be saved under `checkpoints/gnn/`, while training logs will be written to `logs/gnn/`.

### 2. Meta-train the LLMs

Meta-training is implemented using Hugging Face `accelerate`. Model and dataset arguments are defined in `src/meta_train/args.py`. You can launch training via the provided shell script:

```bash
sh scripts/run_meta_train.sh
```
Or run a specific dataset configuration manually:

```bash
model_name='Llama-3.1-8B'
dataset='ces'

accelerate launch --config_file scripts/accelerate/default_config.yaml \
    scripts/run_meta_train.py \
    --root_data_dir="dataset"\
    --dataset="$dataset" \
    --option_dict_path="dataset/$dataset/codebook.jsonl"\
    --model_name="$model_name"\
    --save_dir="checkpoints/meta_train" \
    --wandb
```

- Configure model-specific arguments in `scripts/model_args/{model_name}.yaml`.
- The `--wandb` flag enables experiment tracking via Weights & Biases.

### 3. Run Inference (Adaptive Elicitation)

Multi-GPU inference is powered by **Ray**. You can execute the provided shell scripts for a quick start or run the Python entry points manually for custom configurations.
```bash
sh scripts/run_inference_baseline.sh   # baseline: meta model only
sh scripts/run_inference_gae.sh       # ours: GAE (GNN + meta model + imputation)
```

Or run a single configuration manually:

#### **Baseline**: Meta-trained model for query selection (no GNN integration).

```bash
log_path=results/${dataset}/meta_${model_name}-Q-${query_selection}-N-${node_selection}
CUDA_VISIBLE_DEVICES=$cuda python scripts/run_inference_baseline.py --cuda $cuda \
  --llm_batch_size $llm_batch_size --T $T \
  --dataset $dataset --infer_data $infer_data --runs $runs --runs_id $runs_id \
  --llm_checkpoint $checkpoint --log_path $log_path \
  --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec
```

#### **Ours**: Meta-trained model for query selection with GNN-guided node selection and feature imputation.

```bash
log_path=results/${dataset}/gae_${model_name}-Q-${query_selection}-N-${node_selection}
CUDA_VISIBLE_DEVICES=$cuda python scripts/run_inference_gae.py --cuda $cuda \
  --llm_batch_size $llm_batch_size --T $T \
  --dataset $dataset --infer_data $infer_data --runs $runs --runs_id $runs_id \
  --llm_checkpoint $checkpoint --log_path $log_path \
  --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
  --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec \
  --imputation
```

#### Key Parameter Reference

| Parameter | Type / Options | Description |
|-----------|-----------------|-------------|
| `--query_selection` | `random`, `info_gain`, `mcts_lookahead` | Strategy for selecting the next query. |
| `--node_selection` | `random`, `relational` | Method for selecting which respondent to query. |
| `--node_selection_prec` | float | The proportion of respondents to query per round (e.g., 0.5 = 50%). |
| `--T` | int | Number of elicitation rounds. |
| `--imputation` | flag | Enable GNN-based pseudo-label imputation. |

- The inference results, including logs and performance metrics, will be saved under the `results/` directory using the following naming convention: `results/${dataset}/${method}_${model_name}-Q-${query_selection}-N-${node_selection}`.

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