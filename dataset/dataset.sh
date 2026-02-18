# !/bin/bash

# After downloading the datasets into the dataset/raw_data folder, run this script to preprocess the datasets

cd dataset

# ces dataset
python preprocess.py --config configs/ces.yaml
python generate_feats.py --config configs/ces.yaml
python data_processing.py --config configs/ces.yaml

# opinionqa dataset
python preprocess.py --config configs/opinionqa.yaml
python generate_feats.py --config configs/opinionqa.yaml
python data_processing.py --config configs/opinionqa.yaml

# # twin dataset
python preprocess.py --config configs/twin.yaml
python generate_feats.py --config configs/twin.yaml
python data_processing.py --config configs/twin.yaml