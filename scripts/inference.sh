#!/usr/bin/env bash
export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH
# 333b 355a 330b 334e 331b 331d 334d 327a 330a 334a
# 333b 355a 330b 331d 327a 334a


for q in 333b 355a 330b 331d 327a 334a; do

        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/run_inference_dist_demo.py \
        --year 24 \
        --mode group_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --selected-respondent 1.0 \
        --imputation majority \
        --top_k_nei 5 \
        --N 30

        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/run_inference_dist_demo.py \
        --year 24 \
        --mode group_entropy \
        --node_select occurence_out \
        --x-heldout "$q" \
        --selected-respondent 0.5 \
        --imputation majority \
        --top_k_nei 5 \
        --N 30

        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/run_inference_dist_demo.py \
        --year 24 \
        --mode group_entropy \
        --node_select random \
        --x-heldout "$q" \
        --selected-respondent 0.5 \
        --imputation majority \
        --top_k_nei 5 \
        --N 30

        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/run_inference_dist_demo.py \
        --year 24 \
        --mode iid_entropy \
        --node_select entropy \
        --x-heldout "$q" \
        --selected-respondent 1.0 \
        --imputation majority \
        --top_k_nei 5 \
        --N 30

        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/run_inference_dist_demo.py \
        --year 24 \
        --mode group_random \
        --node_select entropy \
        --x-heldout "$q" \
        --selected-respondent 1.0 \
        --imputation majority \
        --top_k_nei 5 \
        --N 30
done

