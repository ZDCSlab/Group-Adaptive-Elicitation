import numpy as np
import pandas as pd
import json
from collections import defaultdict
from transformers import AutoTokenizer
############

import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_jsonl_as_dict_of_dict(path, key=None):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj[key]] = obj  
    return data


def sample_and_shuffle(df, rng, case_col='caseid'):
    # 1. Sample one row (one user)
    # We use the RNG to pick a random index
    random_idx = rng.integers(0, len(df))
    row = df.iloc[random_idx]
    caseid = row[case_col]

    # 2. Extract columns and shuffle
    # FIX: Removed int() around row[col]. 
    # The values are now strings ('A', 'B'...), so we take them as is.
    items = [(col, row[col]) for col in df.columns if col != case_col]
    
    # Shuffle the list of (question_id, answer) pairs
    rng.shuffle(items)
    
    return caseid, items

    # ---------- worker function (runs in subprocesses) ----------

def build_one_sample_no_neighbor(sample_id: int) -> dict:
    """
    Worker:
    - uses a per-sample RNG seed for reproducibility
    - builds one q_text_all using ONLY the target respondent's own answers
        (no neighbor prediction / neighbor weights).
    """
    # each sample has deterministic RNG
    rng = np.random.default_rng(GLOBAL_SEED + sample_id)

    # 1) sample one respondent and shuffle their questions
    caseid, shuffled_pairs = sample_and_shuffle(survey_data, rng=rng)

    # 2) build the concatenated prompt text q_text_all
    parts, qids = [], []
    for qid, target_raw in shuffled_pairs:
        # target_raw is the value in the DataFrame (likely 'A', 'B', etc. if mapped)
        target_ans = target_raw
        
        # Retrieve metadata from codebook
        question_text = codebook[qid]["question"]
        options_dict = codebook[qid]["options"]  # e.g. {'A': 'Favor', 'B': 'Oppose'}

        # Build the options string dynamically (A. Favor\nB. Oppose\n...)
        # We sort keys to ensure A comes before B, B before C
        option_lines = []
        for key in sorted(options_dict.keys()):
            val = options_dict[key]
            option_lines.append(f"{key}. {val}")
        
        # Join with newlines
        options_str = "\n".join(option_lines)

        q_text = (
            f"<Question>{question_text}\n"
            f"{options_str}\n"
            f"<Answer>{target_ans}"
        )
        parts.append(q_text)
        qids.append(qid)

    q_text_all = "".join(parts)

    return {
        "sample_id": sample_id,
        "caseid": caseid,
        "question_ids": qids,
        "q_text_all": q_text_all,
    }
    # ---------- parallel loop + progress bar ----------


for split in ['train', 'val', 'test']:
    # ---------- global data (loaded once in main, reused in workers) ----------
    
    if split == 'train':
        survey_data = pd.read_csv(
            f"/home/ruomeng/gae_graph/dataset/twin/raw_region/question_South_train.csv"
        )
        num_records = len(survey_data)
        num_samples = num_records * 50
    elif split == 'val':
        survey_data = pd.read_csv(
            f"/home/ruomeng/gae_graph/dataset/twin/raw_region/question_South_val.csv"
        )
        num_records = len(survey_data)
        num_samples = num_records * 10
    elif split == 'test':
        survey_data = pd.read_csv(
            f"/home/ruomeng/gae_graph/dataset/twin/raw_region/question_West_test.csv"
        )
        num_records = len(survey_data)
        num_samples = num_records * 10
    survey_data["caseid"] = survey_data["caseid"].astype(str)

    jsonl_file = "/home/ruomeng/gae_graph/dataset/twin/codebook.jsonl"
    codebook = load_jsonl_as_dict_of_dict(jsonl_file, key="id")

    GLOBAL_SEED = 42
    
   


    outputs = []
    max_workers = 8  # 按你机器 CPU 调整

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(build_one_sample_no_neighbor, i): i for i in range(num_samples)}
        for fut in tqdm(as_completed(futures), total=num_samples, desc="Building q_text samples"):
            rec = fut.result()
            outputs.append(rec)

    # 排序一下（as_completed 是乱序的）
    outputs.sort(key=lambda x: x["sample_id"])

    save_path = f"/home/ruomeng/gae_graph/dataset/twin/processed/nei0/{split}.jsonl"
    # create save_path if not exists
    import os
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(outputs)} samples to {save_path}")

