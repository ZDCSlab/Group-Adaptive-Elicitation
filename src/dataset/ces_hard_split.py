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
            data[obj[key]] = obj  
    return data

def get_neighbor_ids(entry):
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


from collections import Counter
import numpy as np
import pandas as pd
import json

def compute_neighbor_stats_A(nei_df, question_id, codebook=None):
    """
    返回一个 dict，包含完整统计信息 + 按模板A生成的 summary_text：
    
    [Similar users' answers summary]
    - Top-{total} neighbors → counts: {1: c1, 2: c2, ...}, majority: {maj_id} ({meaning})
    
    说明：
    - counts 的 key 会按数字顺序排序再转为字符串，便于稳定展示
    - 若有 codebook 且命中该题，则 majority 后会附上语义 (meaning)
    """
    # 列不存在或无有效值，直接返回空模板
    if question_id not in nei_df.columns:
        return {
            "total": 0, "counts": {}, "proportions": {}, "percentages": {},
            "entropy_nats": 0.0, "entropy_bits": 0.0,
            "hhi": 0.0, "gini_impurity": 0.0,
            "majority_answer": None, "majority_pct": 0.0,
            "summary_text": (
                "[Similar users' answers summary]\n"
                "- Top-0 neighbors → counts: {}, majority: None"
            )
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
            "majority_answer": None, "majority_pct": 0.0,
            "summary_text": (
                "[Similar users' answers summary]\n"
                "- Top-0 neighbors → counts: {}, majority: None"
            )
        }

    # 基础统计
    cnt = Counter(series.tolist())
    # 为了展示稳定：按选项ID的数字顺序排序
    sorted_items = sorted(cnt.items(), key=lambda kv: int(kv[0]) if isinstance(kv[0], (int, str)) else kv[0])
    counts = {str(k): int(v) for k, v in sorted_items}
    probs = {k: v / total for k, v in counts.items()}
    percentages = {k: 100.0 * p for k, p in probs.items()}

    p = np.array(list(probs.values()), dtype=float)
    entropy_nats = float(-(p * (np.log(p + 1e-12))).sum())
    entropy_bits = float(entropy_nats / np.log(2.0))
    hhi = float((p ** 2).sum())
    gini_impurity = float(1.0 - hhi)

    # 多数类
    maj_key, maj_cnt = max(counts.items(), key=lambda kv: kv[1])
    majority_pct = 100.0 * (maj_cnt / total)

    # ---- 模板A的摘要行 ----
    # counts 用紧凑的 JSON 串展示，便于直接贴到 prompt
    counts_str = json.dumps({k: int(v) for k, v in counts.items()}, ensure_ascii=False)
    maj_meaning = ""
    if codebook and question_id in codebook:
        opts = codebook[question_id].get("options", {})
        if str(maj_key) in opts and opts[str(maj_key)]:
            maj_meaning = f" ({opts[str(maj_key)]})"

    # summary_text = (
    #     "[Similar users' answers summary] "
    #     f"- Top-{total} neighbors → counts: {counts_str}, "
    #     f"Majority Answer: {maj_key}{maj_meaning}"
    # )

    summary_text = (f"Majority Answer: {maj_key}{maj_meaning}")

    # print(summary_text)
    # input()

    return {
        "total": total,
        "counts": counts,
        "proportions": probs,
        "percentages": percentages,
        "entropy_nats": entropy_nats,
        "entropy_bits": entropy_bits,
        "hhi": hhi,
        "gini_impurity": gini_impurity,
        "majority_answer": maj_key,
        "majority_pct": majority_pct,
        "summary_text": summary_text
    }



def compute_neighbor_answers(nei_df, question_id, codebook=None, return_labels=False):
    """
    Return neighbors' answers for a given question.

    Args:
        nei_df (pd.DataFrame): rows = neighbors (index is neighbor/node id), columns = question ids
        question_id: column key to read (int/str)
        codebook (dict|None): optional; expect codebook[q]["options"][str(code)] -> label
        return_labels (bool): if True and codebook provided, include labels

    Returns:
        dict
          - if return_labels and codebook:
              { neighbor_id: {"code": int, "label": str} }
            else:
              { neighbor_id: int_code }
        (Neighbors with missing or -1 are excluded.)
    """
    if question_id not in nei_df.columns:
        return {}

    # to numeric, drop NaNs and -1 (missing)
    s = pd.to_numeric(nei_df[question_id], errors='coerce')
    s = s.dropna()
    # some frames store as float; cast after dropping NaN
    s = s.astype(int)
    s = s[s != -1]

    if s.empty:
        return {}

    if return_labels and codebook and question_id in codebook:
        opts = codebook[question_id].get("options", {})
        return {
            int(nid): {
                "code": int(val),
                "label": opts.get(str(int(val)), "")
            }
            for nid, val in s.items()
        }
    else:
        # raw codes only
        return {int(nid): int(val) for nid, val in s.items()}


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


def load_ces_dataset_partition(year, train_ids, test_ids):
    out_dir = Path(f'/home/ruomeng/gae/dataset/ces/processed')
    out_dir.mkdir(parents=True, exist_ok=True)  

    out_path_train = Path(f'/home/ruomeng/gae/dataset/ces/processed/ces_{year}_train.jsonl')
    out_path_test = Path(f'/home/ruomeng/gae/dataset/ces/processed/ces_{year}_test.jsonl')
    records_train = load_if_exists(out_path_train)
    records_test = load_if_exists(out_path_test)

  
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
    nei_dict, option_dict = dict(), dict()

    dataset_train, dataset_test = [], []
    for respondent in tqdm(caseid_lst, total=len(caseid_lst), desc="Building dataset", dynamic_ncols=True):
        respondent_data = df_survey[df_survey['caseid'].astype(str) == str(respondent)].iloc[0]  # Series
        # print('respondent_data', respondent_data)

        # 
        nei_ids = get_neighbor_ids(neighbors_info[str(respondent)])
        nei_ids_filtered = []
        for id in nei_ids:
            if id in train_ids:
                nei_ids_filtered.append(int(id))

        # print('nei_ids_filtered', len(nei_ids_filtered))
        nei_df = df_survey[df_survey['caseid'].isin(nei_ids_filtered)] 

        respondent_designs= []
        for question_id, val in respondent_data.items():
            if question_id in ["year", "caseid"]:  # skip metadata
                continue
            # look up mapping, fallback to raw value if not in codebook
            # print(question_id, val)
            option_dict[question_id] = list(codebook[question_id]["options"].keys())
            q_text = f'{codebook[question_id]["question"]} {options_to_string(codebook[question_id]["options"])} ?'
            if int(val) == -1:
                a_text = ''
            else:
                a_text = int(val) # f'[{int(val)}] {codebook[question_id]["options"][str(int(val))]}'

            summary_text = compute_neighbor_stats_A(nei_df, question_id, codebook)
            
            respondent_designs.append({"q_id": question_id, "question": q_text, "answer": a_text, "neighbor_stats": summary_text['summary_text']})
        
        if respondent in train_ids:
            # print('train_ids')
            dataset_train.append({'case_id': respondent, 'demograohic_info': demograohic_info[int(respondent)]['info'], 'label': neighbors_info[str(respondent)]['label'], 'survey': respondent_designs})
        else:
            # print('test_ids')
            dataset_test.append({'case_id': respondent, 'demograohic_info': demograohic_info[int(respondent)]['info'], 'label': neighbors_info[str(respondent)]['label'], 'survey': respondent_designs})
        # input()

        nei_dict[respondent] = nei_ids_filtered

    with out_path_train.open("w", encoding="utf-8") as f:
        for rec in dataset_train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset_train)} records -> {out_path_train}")

    with out_path_test.open("w", encoding="utf-8") as f:
        for rec in dataset_test:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset_test)} records -> {out_path_test}")
    return dataset_train, dataset_test, option_dict

import random

Q, A, N, EOS, EOP = "<Question>", "<Answer>", "<Neighbor>", "<EOS>", "<EOP>"

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


def train_split_entities_from_json(entities, seed=42, include_neighbor=True, q_key="question", a_key="answer"):
    rng = random.Random(seed)
    packed = [_pack_entity(e, q_key=q_key, a_key=a_key, neighbor_field="neighbor_stats", include_neighbor=include_neighbor) for e in entities]
    rng.shuffle(packed)

    n = len(packed)
    assert n > 0, "No entities provided."
  
    data_str = "".join(packed)
    return data_str



from sklearn.model_selection import train_test_split

def split_caseid(year, test_size=0.15, random_state=42):
    jsonl_file = f"/home/ruomeng/gae/dataset/ces/raw/{year}/neighbors_top30_semantic_{year}.jsonl"  
    neighbors_info = load_jsonl_as_dict_of_dict(jsonl_file, key='caseid')
    caseid_lst = list(neighbors_info.keys())
    # 85% train, 15% test
    train_ids, test_ids = train_test_split(caseid_lst, test_size=test_size, random_state=random_state)
   
    print(len(train_ids), len(test_ids))
    return train_ids, test_ids



if __name__ == "__main__":
    include_neighbor= True
    train_ids_20, test_ids_20 = split_caseid(year='20', test_size=0.15, random_state=42)
    dataset_train_20, dataset_test_20, option_dict_20 = load_ces_dataset_partition(year='20', train_ids=train_ids_20, test_ids=test_ids_20)
    train_str_20, test_str_20 = train_split_entities_from_json(dataset_train_20, include_neighbor=include_neighbor), train_split_entities_from_json(dataset_test_20, include_neighbor=include_neighbor)

    train_ids_22, test_ids_22 = split_caseid(year='22', test_size=0.15, random_state=42)
    dataset_train_22, dataset_test_22, option_dict_22 = load_ces_dataset_partition(year='22', train_ids=train_ids_22, test_ids=test_ids_22)
    train_str_22, test_str_22 = train_split_entities_from_json(dataset_train_22, include_neighbor=include_neighbor), train_split_entities_from_json(dataset_test_22, include_neighbor=include_neighbor)

    train_ids_24, test_ids_24 = split_caseid(year='24', test_size=0.15, random_state=42)
    dataset_train_24, dataset_test_24, option_dict_24 = load_ces_dataset_partition(year='24', train_ids=train_ids_24, test_ids=test_ids_24)
    train_str_24, test_str_24 = train_split_entities_from_json(dataset_train_24, include_neighbor=include_neighbor), train_split_entities_from_json(dataset_test_24, include_neighbor=include_neighbor)

    for split in ['20-22', '22-24', '20-24']:
        if split == '20-22':
            data_dict = {'train': train_str_20, 'val': test_str_20, 'test': test_str_22, 'option_dict': option_dict_20}
        if split == '22-24':
            data_dict = {'train': train_str_22, 'val': test_str_22, 'test': test_str_24, 'option_dict': option_dict_22}
        if split == '20-24':
            data_dict = {'train': train_str_20, 'val': test_str_20, 'test': test_str_24, 'option_dict': option_dict_24}

        save_dir = Path("/home/ruomeng/gae/dataset/ces/processed_new")
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"dataset_{split}_neighbor{int(include_neighbor)}.json"
        with open(save_dir / file_name, 'w') as f:
            json.dump(data_dict, f)


