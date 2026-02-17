import pandas as pd
import json
import numpy as np
from utils import load_jsonl_as_dict_of_dict
from sklearn.model_selection import train_test_split
import yaml
import argparse


def split_by_caseid(df, test_size=0.1, val_size=0.1, seed=42, id_col="caseid"):
    if isinstance(id_col, list):
        id_col = id_col[0]
    # unique ids (drop NaNs just in case)
    ids = df[id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)

    n = len(ids)
    n_test = int(round(n * test_size))
    n_val  = int(round(n * val_size))
    n_train = max(0, n - n_test - n_val)

    test_ids  = set(ids[:n_test])
    val_ids   = set(ids[n_test:n_test+n_val])
    train_ids = set(ids[n_test+n_val:])

    # masks
    is_test  = df[id_col].isin(test_ids)
    is_val   = df[id_col].isin(val_ids)
    is_train = df[id_col].isin(train_ids)

    # split DataFrames
    train_df = df[is_train].reset_index(drop=True)
    val_df   = df[is_val].reset_index(drop=True)
    test_df  = df[is_test].reset_index(drop=True)

    return train_df, val_df, test_df, (train_ids, val_ids, test_ids)



if __name__ == "__main__":
    # set the random seed
    np.random.seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/twin.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    

    IDENTITY_COL = [cfg["dataset"]["identity_col"]]

    REGION_COL = cfg["region"]["column"]
    REGION_DICT = cfg["region"]["mapping"]

    DEMOGRAPHIC_COLS = list(cfg["demographics"]["columns"].values())
    DEMOGRAPHIC_NAME_MAP = cfg["demographics"]["columns"]

    QUESTION_COLS = cfg["questions"]["columns"]
    IN_DIST = cfg["splits"]["in_distribution"]
    OOD = cfg["splits"]["out_of_distribution"]


    file_path = f'{cfg["dataset"]["name"]}/raw_data/{cfg["dataset"]["name"]}_responses.csv'
    survey_data = pd.read_csv(file_path)

    # Load Question code book
    jsonl_file = f'{cfg["dataset"]["name"]}/codebook.jsonl'  
    codebook = load_jsonl_as_dict_of_dict(jsonl_file)

    REGION_NAME_TO_CODE = {v: k for k, v in REGION_DICT.items()}
    target_region_codes = [REGION_NAME_TO_CODE[IN_DIST], REGION_NAME_TO_CODE[OOD]]

    for split_val in REGION_NAME_TO_CODE[IN_DIST]:
        split_df = survey_data[survey_data[REGION_COL]==split_val]
        train_df, val_df, test_df, (train_ids, val_ids, test_ids) = split_by_caseid(split_df, test_size=0.1, val_size=0.1, seed=42, id_col=IDENTITY_COL)
        train_df[IDENTITY_COL+DEMOGRAPHIC_COLS].to_csv(f'{cfg["dataset"]["name"]}/data/demo_{REGION_DICT[split_val]}_train.csv', index=None)
        train_df[IDENTITY_COL+QUESTION_COLS].to_csv(f'{cfg["dataset"]["name"]}/data/question_{REGION_DICT[split_val]}_train.csv', index=None)
        val_df[IDENTITY_COL+DEMOGRAPHIC_COLS].to_csv(f'{cfg["dataset"]["name"]}/data/demo_{REGION_DICT[split_val]}_val.csv', index=None)
        val_df[IDENTITY_COL+QUESTION_COLS].to_csv(f'{cfg["dataset"]["name"]}/data/question_{REGION_DICT[split_val]}_val.csv', index=None)
    

    for split_val in REGION_NAME_TO_CODE[OOD]:
        split_df = survey_data[survey_data[REGION_COL]==split_val]
        split_df[IDENTITY_COL+DEMOGRAPHIC_COLS].to_csv(f'{cfg["dataset"]["name"]}/data/demo_{REGION_DICT[split_val]}_test.csv', index=None)
        split_df[IDENTITY_COL+QUESTION_COLS].to_csv(f'{cfg["dataset"]["name"]}/data/question_{REGION_DICT[split_val]}_test.csv', index=None)