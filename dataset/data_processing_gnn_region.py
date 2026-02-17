import pandas as pd
import json
from pathlib import Path

for feature in ['demo', 'question']:

    # ----- 1. Paths -----
    train_csv = f"/home/ruomeng/gae_graph/dataset/twin/raw_region/{feature}_South_train.csv"
    val_csv   = f"/home/ruomeng/gae_graph/dataset/twin/raw_region/{feature}_South_val.csv"
    test_csv  = f"/home/ruomeng/gae_graph/dataset/twin/raw_region/{feature}_West_test.csv"

    out_json  = "/home/ruomeng/gae_graph/dataset/twin/raw_region/user_split.json"   # where you want to save the JSON

    # ----- 2. Load 3 CSVs -----
    df_train = pd.read_csv(train_csv)
    df_train["caseid"] = df_train["caseid"].astype(str)
    df_val   = pd.read_csv(val_csv)
    df_val["caseid"] = df_val["caseid"].astype(str)
    df_test  = pd.read_csv(test_csv)
    df_test["caseid"] = df_test["caseid"].astype(str)

    # (Optional) combine to one big dataframe if you ever need it
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_all.to_csv( f"/home/ruomeng/gae_graph/dataset/twin/raw_region/{feature}_all.csv", index=None)

    # ----- 3. Extract caseid lists -----
    # ensure no NaN, cast to int if they are numeric, and deduplicate while keeping order
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

    train_users = unique_caseids(df_train)
    val_users   = unique_caseids(df_val)
    test_users  = unique_caseids(df_test)

    split_obj = {
        "train_users": train_users,
        "val_users":   val_users,
        "test_users":  test_users,
    }

    # ----- 4. Save JSON -----
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(split_obj, f)

    print(f"Saved user split to: {out_json}")
    print(f"#train_users={len(train_users)}, #val_users={len(val_users)}, #test_users={len(test_users)}")
