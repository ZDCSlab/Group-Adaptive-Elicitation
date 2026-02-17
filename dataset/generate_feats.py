import pandas as pd
import json
import numpy as np
from pathlib import Path
from utils import load_jsonl_as_dict_of_dict
import yaml
import os
import argparse


# Helper function to split the dataframe by caseid
def split_by_caseid(df, test_size=0.1, seed=42, id_col="caseid"):
    if isinstance(id_col, list):
        id_col = id_col[0]
    # unique ids (drop NaNs just in case)
    ids = df[id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)

    n = len(ids)
    n_test = int(round(n * test_size))
    n_train = max(0, n - n_test)

    test_ids  = set(ids[:n_test])
    train_ids = set(ids[n_test:])

    # masks
    is_test  = df[id_col].isin(test_ids)
    is_train = df[id_col].isin(train_ids)

    # split DataFrames
    train_df = df[is_train].reset_index(drop=True)
    test_df  = df[is_test].reset_index(drop=True)

    return train_df, test_df, (train_ids, test_ids)


# Helper function to extract unique caseids from a dataframe
def unique_caseids(df):
    # drop NaN
    ids = df["caseid"].dropna().tolist()
    # if they are numeric but read as float, cast to int
    ids = [int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in ids]
    # remove duplicates while preserving order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


if __name__ == "__main__":
    # set the random seed
    np.random.seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/twin.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loading config from: {args.config}")
    print(f"Dataset: {cfg['dataset']['name']}")
  
    IDENTITY_COL = [cfg["dataset"]["identity_col"]]

    REGION_COL = cfg["region"]["column"]
    REGION_DICT = cfg["region"]["mapping"]

    DEMOGRAPHIC_COLS = list(cfg["demographics"]["columns"].values())
    DEMOGRAPHIC_NAME_MAP = cfg["demographics"]["columns"]

    QUESTION_COLS = cfg["questions"]["columns"]
    IN_DIST_REGION = cfg["splits"]["in_distribution_region"]
    OOD_REGION = cfg["splits"]["out_of_distribution_region"]

    out_json = cfg["dataset"]["user_split_path"]
    survey_data = pd.read_csv(cfg["dataset"]["all_responses_path"])
    codebook = load_jsonl_as_dict_of_dict(cfg["dataset"]["codebook_path"])

    REGION_NAME_TO_CODE = {v: k for k, v in REGION_DICT.items()}
    target_region_codes = [REGION_NAME_TO_CODE[IN_DIST_REGION], REGION_NAME_TO_CODE[OOD_REGION]]

    base_path = f'{cfg["dataset"]["name"]}/data'
    # create base_path if it does not exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Split the data by region: 
    for split_val in REGION_NAME_TO_CODE[IN_DIST_REGION]:
        split_df = survey_data[survey_data[REGION_COL]==split_val]
        train_df, val_df, (train_ids, val_ids) = split_by_caseid(split_df, test_size=cfg["splits"]["val_size"], seed=cfg["splits"]["seed"], id_col=IDENTITY_COL)
        train_df[IDENTITY_COL+DEMOGRAPHIC_COLS].to_csv(f'{base_path}/demo_{REGION_DICT[split_val]}_train.csv', index=None)
        train_df[IDENTITY_COL+QUESTION_COLS].to_csv(f'{base_path}/question_{REGION_DICT[split_val]}_train.csv', index=None)
        val_df[IDENTITY_COL+DEMOGRAPHIC_COLS].to_csv(f'{base_path}/demo_{REGION_DICT[split_val]}_val.csv', index=None)
        val_df[IDENTITY_COL+QUESTION_COLS].to_csv(f'{base_path}/question_{REGION_DICT[split_val]}_val.csv', index=None)
    
    # Split the data by region
    for split_val in REGION_NAME_TO_CODE[OOD_REGION]:
        test_df = survey_data[survey_data[REGION_COL]==split_val]
        test_df[IDENTITY_COL+DEMOGRAPHIC_COLS].to_csv(f'{base_path}/demo_{REGION_DICT[split_val]}_test.csv', index=None)
        test_df[IDENTITY_COL+QUESTION_COLS].to_csv(f'{base_path}/question_{REGION_DICT[split_val]}_test.csv', index=None)

    print(f"Split data by region: {IN_DIST_REGION} and {OOD_REGION}")
    print(f"#train_users={len(train_df[IDENTITY_COL])}, #val_users={len(val_df[IDENTITY_COL])}, #test_users={len(test_df[IDENTITY_COL])}")

    # Extract unique caseids from the split data
    users_list = [unique_caseids(train_df), unique_caseids(val_df), unique_caseids(test_df)]
    split_obj = {
        "train_users": users_list[0],
        "val_users":   users_list[1],
        "test_users":  users_list[2],
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(split_obj, f)

    print(f"Saved user split to: {out_json}")
    print(f"#train_users={len(users_list[0])}, #val_users={len(users_list[1])}, #test_users={len(users_list[2])}")
