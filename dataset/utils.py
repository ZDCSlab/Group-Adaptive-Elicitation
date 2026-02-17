import pandas as pd
import json

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
def load_jsonl_as_dict_of_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj  
    return data

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
