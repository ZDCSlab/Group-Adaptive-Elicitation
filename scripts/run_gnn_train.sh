#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH
gpus=0

CUDA_VISIBLE_DEVICES=$gpus python scripts/run_gnn_train.py --config scripts/args_gnn/config_ces.yaml

CUDA_VISIBLE_DEVICES=$gpus python scripts/run_gnn_train.py --config scripts/args_gnn/config_opinionqa.yaml

CUDA_VISIBLE_DEVICES=$gpus python scripts/run_gnn_train.py --config scripts/args_gnn/config_twin.yaml
