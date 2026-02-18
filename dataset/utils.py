import pandas as pd
import json
from pathlib import Path

# Helper function to convert numeric values to letters
def number_to_letter(n):
    if pd.isna(n):
        return n
    try:
        n = int(n)
        return chr(ord('A') + n - 1)
    except:
        return n

# Helper function to load jsonl file as a dictionary of dictionaries
def load_jsonl_as_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj  
    return data

# Helper function to load codebook mappings
def load_codebook_mappings(codebook_path: str) -> dict:
    col_mappings = {}
    with open(codebook_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            options = entry.get("options", {})
            inv_map = {str(val): key for key, val in options.items()}
            col_mappings[entry["id"]] = inv_map
    return col_mappings


# Helper function to map dataframe to options
def map_dataframe_to_options(
    df: pd.DataFrame,
    col_mappings: dict,
    key_col: str = "QKEY",
) -> pd.DataFrame:
    df_mapped = df.copy()
    for col in df_mapped.columns:
        if col == key_col or col not in col_mappings:
            continue
        df_mapped[col] = df_mapped[col].astype(str).map(col_mappings[col])
    return df_mapped


# Helper function to load question strings
def load_question_strings(path: str) -> dict:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} question strings from {path}")
        return data
    except FileNotFoundError:
        print(f"Warning: Question strings not found at {path}")
        return {}


# Helper function to sample one respondent and shuffle their questions
def sample_and_shuffle(df, rng, case_col='caseid'):
    # 1. Sample one row (one user)
    # We use the RNG to pick a random index
    random_idx = rng.integers(0, len(df))
    row = df.iloc[random_idx]
    caseid = row[case_col]

    # 2. Extract columns and shuffle
    items = [(col, row[col]) for col in df.columns if col != case_col]
    
    # Shuffle the list of (question_id, answer) pairs
    rng.shuffle(items)
    
    return caseid, items

# Helper function to check the NaN statistics of the dataframe
def check_nan_stats(df):
    nan_stats = (
        df.isna()
        .mean()                # the ratio of NaN in each column
        .reset_index()
        .rename(columns={"index": "column", 0: "nan_ratio"})
    )
    nan_stats = nan_stats[nan_stats["nan_ratio"] > 0]
    print("Columns with NaN and their missing ratio:")
    print(nan_stats)


# Helper function to fill NaN with majority value
def fill_nan_with_majority(df):
    for col in df.columns:
        if col == "caseid":
            continue

        if not df[col].isna().any():
            continue

        non_na = df[col].dropna()
        if len(non_na) == 0:
            continue  
        
        majority_value = non_na.mode().iloc[0]
        df[col] = df[col].fillna(majority_value)
        print(f"Filled {col} with majority value: {majority_value}")

    return df


def filter_by_valid_values(df, check_cols, codebook):
    valid_map = {
        qid: {str(k) for k in codebook[qid]["options"].keys()}
        for qid in check_cols
        if qid in codebook and "options" in codebook[qid]
    }

    mask = pd.Series(True, index=df.index)

    for qid, valid in valid_map.items():
        if qid not in df.columns:
            continue
        mask &= df[qid].astype(str).isin(valid)

    return df[mask].copy()



# Helper function for opinionqa
def load_question_list(json_path: str) -> list:
    """Load question keys from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


# Helper function for opinionqa
def load_merge_fast(directory_path: str, key_col: str = "QKEY"):
    """
    Load all .sav files in a directory, concatenate, and take first row per key.
    Returns (master_df, user_stats, col_stats) or (None, None, None) if no data.
    """
    folder = Path(directory_path)
    sav_files = list(folder.glob("*.sav"))
    if not sav_files:
        print("No .sav files found.")
        return None, None, None

    print(f"Found {len(sav_files)} files. Loading...")
    all_dfs = []
    for file_path in sav_files:
        try:
            df = pd.read_spss(file_path)
            print(len(df), file_path)
            if key_col in df.columns:
                all_dfs.append(df)
            else:
                print(f"Skipped {file_path.name}: Key '{key_col}' not found.")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    if not all_dfs:
        return None, None, None

    print("Concatenating and grouping...")
    big_stack = pd.concat(all_dfs, ignore_index=True)
    master_df = big_stack.groupby(key_col, as_index=True).first()
    master_df.reset_index(inplace=True)
    user_stats = master_df.isnull().mean(axis=1) * 100
    col_stats = master_df.isnull().mean(axis=0) * 100
    return master_df, user_stats, col_stats


# Helper function for opinionqa
def filter_columns_by_questions(df: pd.DataFrame, q_list: list, key_col: str) -> pd.DataFrame:
    """Keep only columns in q_list, always including key_col."""
    cols_to_keep = [c for c in df.columns if c in q_list]
    if key_col not in cols_to_keep:
        print(f"Note: '{key_col}' was not in the JSON list. Adding it to preserve user IDs.")
        cols_to_keep.insert(0, key_col)
    return df[cols_to_keep]


# Helper function for opinionqa
def filter_users_then_questions(
    df: pd.DataFrame,
    user_thresh_pct: float = 0.5,
    question_thresh_pct: float = 0.9,
) -> pd.DataFrame:
    """
    Drop users who answered fewer than user_thresh_pct of all questions,
    then drop questions present for fewer than question_thresh_pct of remaining users.
    """
    print(f"--- Initial Shape: {df.shape} ---")
    answers_per_user = df.count(axis=1)
    max_answers = answers_per_user.max()
    avg_answers = answers_per_user.mean()
    print(f"Stats: Avg answers/user: {avg_answers:.1f} | Max answers/user: {max_answers}")
    print(f"Total Columns: {df.shape[1]}")

    required_answers = int(df.shape[1] * user_thresh_pct)
    if required_answers > max_answers:
        print(
            f"\n[WARNING] Requested users with {required_answers} answers, "
            f"but max is {max_answers}. Adjusting to 80% of max."
        )
        required_answers = int(max_answers * 0.8)

    print(f"\n1. Keeping users with >= {required_answers} answers...")
    df_users = df.dropna(axis=0, thresh=required_answers)
    print(f"   -> Users remaining: {df_users.shape[0]}")

    if df_users.empty:
        return df_users

    required_users = int(df_users.shape[0] * question_thresh_pct)
    print(f"2. Keeping questions present for >= {question_thresh_pct*100}% of remaining users...")
    df_final = df_users.dropna(axis=1, thresh=required_users)
    print(f"   -> Final Shape: {df_final.shape}")
    return df_final


# Helper function for opinionqa
def extract_demographics(df: pd.DataFrame, user_ids: list, target_cols: list, key_col: str = "QKEY") -> pd.DataFrame:
    """Extract demographic columns for given users; drop rows with any NaN in demographics."""
    existing = [c for c in target_cols if c in df.columns]
    missing = [c for c in target_cols if c not in df.columns]
    print(f"Requested {len(target_cols)} demographic columns.")
    print(f"Found {len(existing)}.")
    if missing:
        print("[WARNING] Not found in df:", missing)
    cols = [key_col] + existing
    df_demo = df.loc[df[key_col].isin(user_ids), cols].copy()
    return df_demo.dropna(), existing


# Helper function for opinionqa
def build_final_dataframe(
    df: pd.DataFrame,
    user_ids: list,
    question_cols: list,
    demo_cols: list,
    key_col: str = "QKEY",
) -> pd.DataFrame:
    """Build dataframe with QKEY + question columns + demographic columns (only existing)."""
    desired = [key_col] + question_cols + demo_cols
    final_cols = list(dict.fromkeys(c for c in desired if c in df.columns))
    df_subset = df.loc[df[key_col].isin(user_ids), final_cols].copy()
    print(f"Final shape: {df_subset.shape}")
    if not df_subset[key_col].is_unique:
        print("WARNING: Duplicate QKEYs found.")
    return df_subset


# Helper function for opinionqa
def report_missing(target_df: pd.DataFrame, top_n: int = 5, key_col: str = "QKEY") -> pd.DataFrame:
    """Print missing-value stats and return sorted report; show top_n dirty columns."""
    na_counts = target_df.isnull().sum()
    na_pct = target_df.isnull().mean() * 100
    report = pd.DataFrame({"Missing Count": na_counts, "Missing %": na_pct})
    report = report.sort_values(by="Missing Count", ascending=False)
    dirty = report[report["Missing Count"] > 0]
    print(f"Total columns: {target_df.shape[1]}")
    print(f"With no missing: {len(target_df.columns) - len(dirty)}")
    print(f"With missing: {len(dirty)}")
    print("\n--- Top columns with missing data ---")
    print(dirty.head(top_n))
    return report


# Helper function for opinionqa
def impute_by_demographics(
    df: pd.DataFrame,
    question_cols: list,
    grouping_keys: list,
    key_col: str = "QKEY",
) -> pd.DataFrame:
    """
    Forward- and backward-fill missing question answers within demographic groups.
    Demographics are temporarily filled with 'Unknown' or -1 so groupby includes all rows.
    """
    df_imputed = df.copy()
    for col in grouping_keys:
        if df_imputed[col].dtype == "object" or df_imputed[col].dtype.name == "category":
            if isinstance(df_imputed[col].dtype, pd.CategoricalDtype):
                if "Unknown" not in df_imputed[col].cat.categories:
                    df_imputed[col] = df_imputed[col].cat.add_categories("Unknown")
            df_imputed[col] = df_imputed[col].fillna("Unknown")
        else:
            df_imputed[col] = df_imputed[col].fillna(-1)

    df_imputed = df_imputed.sort_values(by=grouping_keys)
    print("Applying Forward Fill within groups...")
    df_imputed[question_cols] = df_imputed.groupby(grouping_keys)[question_cols].ffill()
    print("Applying Backward Fill within groups...")
    df_imputed[question_cols] = df_imputed.groupby(grouping_keys)[question_cols].bfill()
    df_imputed = df_imputed.sort_values(by=key_col)
    return df_imputed


# Helper function for opinionqa
def clean_features(df: pd.DataFrame, key_col: str = "QKEY") -> pd.DataFrame:
    """Drop any remaining NaNs, cast QKEY to int, save to CSV; return clean df."""
    clean = df.dropna()
    total_na = clean.isnull().sum().sum()
    if total_na == 0:
        print("SUCCESS: No missing values in cleaned DataFrame.")
    else:
        print(f"WARNING: {total_na} missing values remain.")
    
    clean = df.dropna().copy()
    clean.loc[:, key_col] = clean[key_col].astype(int)

    return clean

