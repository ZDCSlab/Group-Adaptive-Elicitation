import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm  # works in terminals & notebooks
from collections import Counter, defaultdict
import numpy as np
import os

def options_to_string(options: dict, prefix="Options: ", sep=", "):
    """
    Render {"1":"Yes","2":"No"} -> "Options: [1] Yes, [2] No"
    Sorts numerically if possible, else lexicographically.
    """
    def try_int(s):
        try: return int(s)
        except: return None

    # Normalize keys to strings for display, but sort by numeric value if possible
    items = []
    for k, v in options.items():
        ks = str(k).strip()
        kn = try_int(ks)
        items.append((ks, v, (0, kn) if kn is not None else (1, ks)))
    items.sort(key=lambda x: x[2])

    body = sep.join(f"[{ks}] {v}" for ks, v, _ in items)
    return f"{prefix}{body}"



def load_jsonl_as_dict_of_dict(path, key=None):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj[key]] = obj  # 用 id 作为 key，整行对象作为 value
    return data

def get_neighbor_ids(entry):
        """
        兼容不同文件字段名：优先 neighbors，其次 neighbor_ids/topk/ids
        确保最终为字符串列表，便于和 df_survey['caseid'] 匹配
        """
        for k in ['neighbors', 'neighbor_ids', 'topk', 'top30', 'ids']:
            if k in entry and entry[k] is not None:
                ids_raw = entry[k]
                break
        else:
            return []
        # 统一转为 str
        return [str(x) for x in ids_raw]

def compute_neighbor_stats(nei_df, question_id, codebook=None):
    if question_id not in nei_df.columns:
        return {
            "total": 0, "counts": {}, "proportions": {}, "percentages": {},
            "entropy_nats": 0.0, "entropy_bits": 0.0,
            "hhi": 0.0, "gini_impurity": 0.0,
            "majority_answer": None, "majority_pct": 0.0
        }

    series = pd.to_numeric(nei_df[question_id], errors='coerce')
    series = series.dropna().astype(int)
    series = series[series != -1]  # 去掉缺失
    total = int(series.shape[0])
    if total == 0:
        return {
            "total": 0, "counts": {}, "proportions": {}, "percentages": {},
            "entropy_nats": 0.0, "entropy_bits": 0.0,
            "hhi": 0.0, "gini_impurity": 0.0,
            "majority_answer": None, "majority_pct": 0.0
        }

    cnt = Counter(series.tolist())
    counts = {str(k): int(v) for k, v in cnt.items()}
    probs = {str(k): v / total for k, v in cnt.items()}
    percentages = {k: 100.0 * p for k, p in probs.items()}


    p = np.array(list(probs.values()), dtype=float)

    # entropy
    entropy_nats = float(-(p * (np.log(p + 1e-12))).sum())
    entropy_bits = float(entropy_nats / np.log(2.0))

    # HHI / Gini impurity
    hhi = float((p ** 2).sum())
    gini_impurity = float(1.0 - hhi)

    # 多数类
    maj_key, maj_cnt = max(counts.items(), key=lambda kv: kv[1])
    majority_pct = 100.0 * (maj_cnt / total)
    # ... 前面和之前一样（算 counts, probs, percentages, entropy 等）

    if total == 0:
        return {
            "total": 0, "counts": {}, "proportions": {}, "percentages": {},
            "entropy_nats": 0.0, "entropy_bits": 0.0,
            "hhi": 0.0, "gini_impurity": 0.0,
            "majority_answer": None, "majority_pct": 0.0,
            "summary_text": "No valid neighbor answers."
        }


    parts = []
    for k, pct in sorted(percentages.items(), key=lambda kv: int(kv[0])):
        if codebook and question_id in codebook:
            opts = codebook[question_id]["options"]

            meaning = opts.get(str(k), "")
            parts.append(f"Option {k} ({meaning}): {pct:.1f}%")
        else:
            parts.append(f"Option {k}: {pct:.1f}%")

    dist_text = "; ".join(parts)

    summary_text = (
        f"Among Top-{total} neighbors, {dist_text}. "
        f"Entropy = {entropy_bits:.2f} bits "
        f"(higher means more diverse). "
        # f"Majority answer is {maj_key} "
        # f"({codebook[question_id]['options'].get(str(maj_key), '')}, "
        # f"{majority_pct:.1f}%)."
    )

    return summary_text

def load_if_exists(out_path):
    """
    Check if out_path exists. If so, load and return JSONL content as a list of dicts.
    Otherwise, return None.
    """
    if os.path.exists(out_path):
        data = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        return None


def load_ces_dataset(year):
    out_path = Path(f'/home/ruomeng/gae/dataset/ces/processed/ces_{year}.jsonl')
    records = load_if_exists(out_path)

    if records is not None:
        print(f"Loaded {len(records)} records from {out_path}")
        return records
    else:

        jsonl_file = f"/home/ruomeng/gae/dataset/ces/raw/question_codebook.jsonl"  
        codebook = load_jsonl_as_dict_of_dict(jsonl_file, key='id')

        jsonl_file = f"/home/ruomeng/gae/dataset/ces/raw/{year}/semantic_identity_embedding_{year}.jsonl"  
        demograohic_info = load_jsonl_as_dict_of_dict(jsonl_file, key='caseid')

        jsonl_file = f"/home/ruomeng/gae/dataset/ces/raw/{year}/neighbors_top30_semantic_{year}.jsonl"  
        neighbors_info = load_jsonl_as_dict_of_dict(jsonl_file, key='caseid')
        caseid_lst = list(neighbors_info.keys())
        print('len(caseid_lst)', len(caseid_lst))

        df_survey = pd.read_csv(f"/home/ruomeng/gae/dataset/ces/raw/{year}/question_{year}.csv")

        # caseid_lst = caseid_lst[:10]

        dataset = []
        for respondent in tqdm(caseid_lst, total=len(caseid_lst), desc="Building dataset", dynamic_ncols=True):
            respondent_data = df_survey[df_survey['caseid'].astype(str) == str(respondent)].iloc[0]  # Series
            # print(respondent_data)
            # 取邻居ID并切出邻居子表
            nei_ids = get_neighbor_ids(neighbors_info[str(respondent)])
            nei_ids = [int(id) for id in nei_ids]
            
            nei_df = df_survey[df_survey['caseid'].isin(nei_ids)] 
        
            respondent_designs= []
            for question_id, val in respondent_data.items():
                if question_id in ["year", "caseid"]:  # skip metadata
                    continue
                # look up mapping, fallback to raw value if not in codebook
                # print(question_id, val)
                q_text = f'{codebook[question_id]["question"]} {options_to_string(codebook[question_id]["options"])} ?'
                if int(val) == -1:
                    a_text = ''
                else:
                    a_text = int(val) # f'[{int(val)}] {codebook[question_id]["options"][str(int(val))]}'

                summary_text = compute_neighbor_stats(nei_df, question_id, codebook)
                # print(summary_text)
    

                respondent_designs.append({"q_id": question_id, "question": q_text, "answer": a_text, "neighbor_stats": summary_text})
            dataset.append({'case_id': respondent, 'demograohic_info': demograohic_info[int(respondent)]['info'], 'label': neighbors_info[str(respondent)]['label'], 'survey': respondent_designs})

        
        with out_path.open("w", encoding="utf-8") as f:
            for rec in dataset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Saved {len(dataset)} records -> {out_path}")
        return dataset

import random

Q, A, N, EOS, EOP = "<Question>", "<Answer>", "<Neighbors>", "<EOS>", "<EOP>"

def _pack_entity(entity, q_key="question", a_key="answer",
                 neighbor_field="neighbor_stats",  # 指定哪个字段装的是邻居文本
                 include_neighbor=True):
    # 已预打包文本
    if "text" in entity and isinstance(entity["text"], str):
        text = entity["text"]
        return text if text.endswith(EOP) else (text + EOP)

    items = entity.get("survey") or entity.get("designs")
    if not items:
        raise ValueError("Entity must have 'survey' or 'designs' (list of Q/A items), or a 'text' field.")

    parts = []
    for it in items:
        # dictor (q,a) tuple
        if isinstance(it, dict):
            q = it.get(q_key, "")
            a = it.get(a_key, "")
            # include_neighbor 
            neighbor_txt = ""
            if include_neighbor:
                nt = it.get(neighbor_field, "")
                if isinstance(nt, str) and nt.strip():
                    neighbor_txt = " " + nt.strip() + " "
        else:
            q, a = it[0], it[1]
            neighbor_txt = " "

        a_str = "" if a is None else str(a)
        parts.append(f"{Q}{q}{N}{neighbor_txt}{A}{a_str}{EOS}")

    return "".join(parts) + EOP


def train_split_entities_from_json(entities, split_ratio=0.85, seed=0, include_neighbor=True, q_key="question", a_key="answer"):
    """
    Input:
      entities: List[dict] like
        {"case_id": ..., "label": ..., "survey":[{"question":..., "answer":...}, ...]}
      split_ratio: fraction for train; remainder goes to test
      seed: RNG seed for deterministic shuffling
    Output:
      train_str, test_str: packed strings with <EOP>/<EOS>/<Question>/<Answer>
    """
    rng = random.Random(seed)
    packed = [_pack_entity(e, q_key=q_key, a_key=a_key, neighbor_field="neighbor_stats", include_neighbor=include_neighbor) for e in entities]
    rng.shuffle(packed)

    n = len(packed)
    assert n > 0, "No entities provided."
    n_train = int(n * split_ratio)
    n_train = max(1, min(n - 1, n_train)) if n > 1 else 1  # keep both splits non-empty when possible

    train_str = "".join(packed[:n_train])
    test_str  = "".join(packed[n_train:])
    return train_str, test_str


if __name__ == "__main__":


    dataset_20 = load_ces_dataset(year='20')
    dataset_22 = load_ces_dataset(year='22')
    dataset_24 = load_ces_dataset(year='24')
    train_str_20, val_str_20 = train_split_entities_from_json(dataset_20, split_ratio=0.85, seed=42)
    train_str_22, val_str_22 = train_split_entities_from_json(dataset_22, split_ratio=0.85, seed=42)
    train_str_24, val_str_24 = train_split_entities_from_json(dataset_24, split_ratio=0.85, seed=42)

    for split in ['20-22', '22-24', '20-24']:
        if split == '20-22':
            data_dict = {'train': train_str_20, 'val': val_str_20, 'test': val_str_22}
        if split == '22-24':
            data_dict = {'train': train_str_22, 'val': val_str_22, 'test': val_str_24}
        if split == '20-24':
            data_dict = {'train': train_str_20, 'val': val_str_20, 'test': val_str_24}

        save_dir = Path("/home/ruomeng/gae/dataset/ces/processed")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"dataset_{split}.json", 'w') as f:
            json.dump(data_dict, f)


