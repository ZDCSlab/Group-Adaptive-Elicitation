#!/bin/bash


accelerate launch --config_file /home/ruomeng/.cache/huggingface/accelerate/default_config.yaml \
    /home/ruomeng/gae_graph/scripts/run_meta_train.py \
    --root_data_dir="/home/ruomeng/gae_graph/dataset"\
    --option_dict_path='/home/ruomeng/gae_graph/dataset/opinionQA/codebook.jsonl'\
    --dataset="opinionQA" \
    --model_name='Llama-3.1-8B'\
    --save_dir="/home/ruomeng/gae_graph/logs" \
    --wandb