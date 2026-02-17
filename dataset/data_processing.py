import numpy as np
import pandas as pd
import os
import json
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from utils import load_jsonl_as_dict_of_dict, sample_and_shuffle


def build_data_for_meta_training(sample_id: int) -> dict:
    """
        Worker:
        - uses a per-sample RNG seed for reproducibility
        - builds one q_text_all using the target respondent's own answers
    """
    # each sample has deterministic RNG
    rng = np.random.default_rng(GLOBAL_SEED + sample_id)

    # sample one respondent and shuffle their questions
    caseid, shuffled_pairs = sample_and_shuffle(survey_data, rng=rng)

    # build the concatenated prompt text q_text_all
    parts, qids = [], []
    for qid, target_raw in shuffled_pairs:
        target_ans = target_raw
        
        # retrieve metadata from codebook
        question_text = codebook[qid]["question"]
        options_dict = codebook[qid]["options"]  # e.g. {'A': 'Favor', 'B': 'Oppose'}

        # build the options string dynamically (A. Favor\nB. Oppose\n...)
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
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/twin.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loading config from: {args.config}")
    print(f"Dataset: {cfg['dataset']['name']}")

    for split in ['train', 'val', 'test']:
        # ---------- global data (loaded once in main, reused in workers) ----------
        # Map splits to their file suffixes and multipliers
        split_config = {
            "train": {"region": cfg["splits"]["in_distribution_region"], "suffix": "train", "multiplier": 50},
            "val":   {"region": cfg["splits"]["in_distribution_region"], "suffix": "val",   "multiplier": 10},
            "test":  {"region": cfg["splits"]["out_of_distribution_region"],  "suffix": "test",  "multiplier": 10},
        }

        region = split_config[split]["region"]
        suffix = split_config[split]["suffix"]
        multiplier = split_config[split]["multiplier"]

        questions_base_path = f'{cfg["dataset"]["name"]}/data/question_{region}_{suffix}.csv'
        survey_data = pd.read_csv(questions_base_path)
        num_records = len(survey_data)
        num_samples = num_records * multiplier

        # Ensure caseid is string (robustly handle possible missing column)
        if "caseid" in survey_data.columns:
            survey_data["caseid"] = survey_data["caseid"].astype(str)
        else:
            raise ValueError(f"'caseid' column not found in {questions_base_path}")

        codebook = load_jsonl_as_dict_of_dict(cfg["dataset"]["codebook_path"])

        GLOBAL_SEED = cfg["splits"]["seed"]
        save_path = f'{cfg["dataset"]["name"]}/processed_data/{split}.jsonl'
        # create save_path if it does not exist
        if not os.path.exists(os.path.dirname(save_path)):  
            os.makedirs(os.path.dirname(save_path))

        outputs = []
        max_workers = 8  # adjust the number of workers based on your machine's CPU

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(build_data_for_meta_training, i): i for i in range(num_samples)}
            for fut in tqdm(as_completed(futures), total=num_samples, desc="Building Question samples"):
                rec = fut.result()
                outputs.append(rec)

        with open(save_path, "w") as f:
            for rec in outputs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Saved {len(outputs)} samples to {save_path}")

