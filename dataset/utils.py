import pandas as pd
import json

# convert numeric values to letters
def number_to_letter(n):
    if pd.isna(n):
        return n
    try:
        n = int(n)
        return chr(ord('A') + n - 1)
    except:
        return n

# load jsonl file as a dictionary of dictionaries
def load_jsonl_as_dict_of_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj  
    return data
