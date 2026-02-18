#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

model_name='Llama-3.1-8B'

for dataset in ces opinionqa twin; do
    accelerate launch --config_file scripts/accelerate/default_config.yaml \
        scripts/run_meta_train.py \
        --root_data_dir="dataset"\
        --dataset="$dataset" \
        --option_dict_path="dataset/$dataset/codebook.jsonl"\
        --model_name="$model_name"\
        --save_dir="checkpoints/meta_train" \
        --wandb
done