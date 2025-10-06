#!/usr/bin/env bash
export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH
# 333b 355a 330b 334e 331b 331d 334d 327a 330a 334a

for q in 327a; do
    for num_node in 0.5 0.1; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --selected-respondent "$num_node" \
        --mode group_entropy \
        --node_select random \
        --x-heldout "$q" \
        --imputation majority \
        --wandb
    done
done


for q in 333b 355a 330b 334e 331b; do
    for num_node in 1.0; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --selected-respondent "$num_node" \
        --mode group_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --imputation majority \
        --wandb
    done
done


for q in 330a 334a 333b 355a 330b 334e 331b; do
    for num_node in 0.5 0.1; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --selected-respondent "$num_node" \
        --mode group_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --imputation majority \
        --wandb

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --selected-respondent "$num_node" \
        --mode group_entropy \
        --node_select random \
        --x-heldout "$q" \
        --imputation majority \
        --wandb
    done
done


for q in 333b 355a 330b 334e 331b 331d 334d 327a 330a 334a; do
    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --selected-respondent "$num_node" \
        --mode group_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --MIG \
        --imputation majority \
        --wandb
done
