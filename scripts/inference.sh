#!/usr/bin/env bash
export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH
# 333b 355a 330b 334e 331b 331d 334d 327a 330a 334a


for q in 333b;
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --mode group_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --MIG \
        --wandb
done


for q in 333b; do
    for num_node in 1.0 0.5 0.1 0.9 0.7 0.5 0.3; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
        --year 24 \
        --selected-respondent "$num_node" \
        --mode group_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --wandb
    done
done


# for q in 333b 355a 330b 334e 331b 331d 334d 327a 330a 334a; do
#     for num_node in 1.0 0.9 0.7 0.5 0.3 0.1; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
#         --year 24 \
#         --selected-respondent "$num_node" \
#         --mode group_entropy \
#         --node_select entropy \
#         --x-heldout "$q" \
#         --wandb
#     done
# done


# for q in 333b 355a 330b 334e 331b 331d 334d 327a 330a 334a; do
#     for num_node in 1.0 0.9 0.7 0.5 0.3 0.1; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_inference_dist.py \
#         --year 24 \
#         --selected-respondent "$num_node" \
#         --mode group_entropy \
#         --node_select random \
#         --x-heldout "$q" \
#         --wandb
#     done
# done

