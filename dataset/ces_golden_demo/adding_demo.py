from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import pandas as pd

def df_to_feature_map(df: pd.DataFrame, id_col: str = "caseid") -> dict:
    """
    Return {caseid: [feature_values...]} for each row.
    Uses all columns except `id_col` as features.
    """
    if id_col not in df.columns:
        raise KeyError(f"{id_col!r} not in DataFrame columns")

    # Ensure unique ids (optional: drop duplicates or raise)
    if df[id_col].duplicated().any():
        raise ValueError(f"{id_col!r} contains duplicates")

    feats = df.drop(columns=[id_col])
    idx = df[id_col].astype(str).tolist()
    vals = feats.to_numpy().tolist()   # fast, preserves column order
    return dict(zip(idx, vals))



def load_jsonl_as_dict_of_dict(path, key=None):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj[key]] = obj  
    return data


import numpy as np
import pandas as pd
from typing import Dict, List

def neighbors_by_hamming_from_com(
    com_dict: Dict[str, List[str]],
    dict_features: Dict[str, List],
    *,
    K: int = 20,
    min_overlap: int = 1,   # require at least this many shared (non-missing) features
) -> Dict[str, List[str]]:
    """
    For each caseid, pick up to K nearest neighbors within its (possibly noisy) community
    by Hamming distance over feature lists in `dict_features`.

    Hamming distance = (#mismatches) / (#compared positions), ignoring positions
    where either row is missing (NaN/None). Pairs with overlap < min_overlap are excluded.
    """
    neighbors = {}

    for com, ids in com_dict.items():
        # Filter to ids that actually have features
        ids = [cid for cid in ids if cid in dict_features]
      
        if len(ids) <= 1:
            for cid in ids:
                neighbors[cid] = []
            continue

        # Build a DataFrame of features: rows=ids, columns=0..d-1
        feat_df = pd.DataFrame([dict_features[cid] for cid in ids], index=ids)
        X = feat_df.to_numpy()                    # shape (n, d), dtype=object if mixed
        n, d = X.shape
        isna = pd.isna(X)

        # Pairwise overlap mask & counts: positions observed in both rows
        overlap = (~isna[:, None, :]) & (~isna[None, :, :])       # (n,n,d)
        overlap_cnt = overlap.sum(axis=2).astype(np.int32)        # (n,n)

        # Pairwise matches over overlapping positions
        eq = (X[:, None, :] == X[None, :, :]) & overlap           # (n,n,d)
        match_cnt = eq.sum(axis=2).astype(np.int32)
        mismatches = overlap_cnt - match_cnt

        # Hamming distance; undefined when overlap==0 → +inf
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = mismatches / overlap_cnt.astype(np.float32)
        dist[overlap_cnt < max(1, min_overlap)] = np.inf

        # Exclude self
        np.fill_diagonal(dist, np.inf)

        # For each row, get up to K smallest distances
        k_eff = max(0, min(K, n - 1))
        if k_eff == 0:
            for cid in ids:
                neighbors[cid] = []
            continue

        # argpartition then argsort within the K for stability
        part = np.argpartition(dist, kth=k_eff-1, axis=1)[:, :k_eff]  # (n,k_eff)
        row_idx = np.arange(n)[:, None]
        part_sorted = part[row_idx, np.argsort(dist[row_idx, part], axis=1)]

        # Optional deterministic tie-break by caseid when distances equal
        # (stable secondary sort)
        ids_arr = np.array(ids)
        for i in range(n):
            nbrs = part_sorted[i].tolist()
            # stable sort by (distance, neighbor_id) for deterministic output
            nbrs.sort(key=lambda j: (dist[i, j], ids_arr[j]))
            neighbors[ids_arr[i]] = ids_arr[nbrs].tolist()

    return neighbors

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any

def _build_feat_df(ids: List[str], dict_features: Dict[str, List[Any]]) -> pd.DataFrame:
    # Keep only ids with features; align rows by ids and columns by position
    rows = []
    kept = []
    for cid in ids:
        if cid in dict_features:
            rows.append(dict_features[cid])
            kept.append(cid)
    if not rows:  # empty
        return pd.DataFrame(index=[])
    # ragged safety: pad to same length if needed
    max_len = max(len(r) for r in rows)
    padded = [r + [np.nan] * (max_len - len(r)) for r in rows]
    return pd.DataFrame(padded, index=kept)

def intra_disagreement_pairwise(
    com_dict: Dict[str, List[str]],
    dict_features: Dict[str, List[Any]],
    *,
    min_overlap: int = 1,
) -> Dict[str, float]:
    """
    Pairwise Hamming disagreement per community.
    Returns: {community: mean_pairwise_hamming}
    """
    out = {}
    for com, ids in com_dict.items():
        feat_df = _build_feat_df(ids, dict_features)
        n, d = feat_df.shape[0], feat_df.shape[1] if not feat_df.empty else (0, 0)
        if n <= 1 or d == 0:
            out[com] = np.nan
            continue

        X = feat_df.to_numpy(object)
        isna = pd.isna(X)

        # overlap and matches for all pairs (n,n,d)
        overlap = (~isna[:, None, :]) & (~isna[None, :, :])
        overlap_cnt = overlap.sum(axis=2)
        eq = (X[:, None, :] == X[None, :, :]) & overlap
        match_cnt = eq.sum(axis=2)
        mismatches = overlap_cnt - match_cnt

        # hamming per pair; mask pairs with insufficient overlap or self-pairs
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = mismatches / overlap_cnt.astype(float)
        np.fill_diagonal(dist, np.nan)
        dist[overlap_cnt < max(1, min_overlap)] = np.nan

        # mean over valid pairs
        valid = ~np.isnan(dist)
        if valid.any():
            out[com] = float(np.nanmean(dist))
        else:
            out[com] = np.nan
    return out

def intra_disagreement_majority(
    com_dict: Dict[str, List[str]],
    dict_features: Dict[str, List[Any]],
    *,
    micro_weighted: bool = False,
) -> Dict[str, float]:
    """
    Majority-vote feature disagreement per community.
    For each feature j: disagr_j = 1 - max_class_count / valid_count.
    Then average disagr_j across features (macro), or micro-weighted if requested.
    Returns: {community: disagreement}
    """
    out = {}
    for com, ids in com_dict.items():
        feat_df = _build_feat_df(ids, dict_features)
        if feat_df.empty:
            out[com] = np.nan
            continue

        # column-wise disagreement
        disagrs = []
        counts = []
        for col in feat_df.columns:
            col_vals = feat_df[col]
            valid = col_vals[~pd.isna(col_vals)]
            m = int(valid.shape[0])
            if m <= 1:
                continue
            # majority count
            vc = valid.value_counts(dropna=True)
            maj = int(vc.iloc[0]) if len(vc) else 0
            disag = 1.0 - (maj / m if m > 0 else 0.0)
            disagrs.append(disag)
            counts.append(m)

        if not disagrs:
            out[com] = np.nan
        else:
            if micro_weighted:
                # weight by valid_count per feature
                w = np.array(counts, dtype=float)
                out[com] = float(np.average(np.array(disagrs, dtype=float), weights=w))
            else:
                out[com] = float(np.mean(disagrs))
    return out

def summarize_intra_disagreement(
    com_dict: Dict[str, List[str]],
    dict_features: Dict[str, List[Any]],
    *,
    min_overlap: int = 1,
    micro_weighted_majority: bool = False,
):
    pairwise = intra_disagreement_pairwise(com_dict, dict_features, min_overlap=min_overlap)
    majority = intra_disagreement_majority(com_dict, dict_features, micro_weighted=micro_weighted_majority)

    # Global summaries (mean over communities, skipping NaN)
    pair_vals = [v for v in pairwise.values() if v == v]  # filter NaN
    maj_vals  = [v for v in majority.values() if v == v]
    summary = {
        "pairwise_per_comm": pairwise,
        "majority_per_comm": majority,
        "global_pairwise_mean": float(np.mean(pair_vals)) if pair_vals else np.nan,
        "global_majority_mean": float(np.mean(maj_vals)) if maj_vals else np.nan,
    }
    return summary


def neighbors_by_hamming_from_com(
    com_dict: Dict[str, List[str]],
    dict_features: Dict[str, List],
    *,
    K: int = 20,
    min_similarity: float = 0.7,   # 需要 >= 这个匹配比例才算邻居
    min_compared: int = 1          # 至少需要这么多个“双方非缺失”的可比维度
) -> Dict[str, List[str]]:
    """
    对每个 caseid，在其（可能带噪）社区内，按“汉明相似度=匹配数/可比位数”筛选并排名：
      - 只保留 相似度 >= min_similarity 且 可比位数 >= min_compared 的配对
      - 按相似度从高到低取前 K 个邻居（相同相似度时按邻居ID字典序稳定排序）
    缺失值：当某一维任一方为 NaN/None 时，该维不参与比较。
    """
    neighbors: Dict[str, List[str]] = {}

    for com, ids in com_dict.items():
        ids = [cid for cid in ids if cid in dict_features]
        if len(ids) <= 1:
            for cid in ids:
                neighbors[cid] = []
            continue

        feat_df = pd.DataFrame([dict_features[cid] for cid in ids], index=ids)
        X = feat_df.to_numpy()
        n, d = X.shape
        isna = pd.isna(X)

        # 可比位（两边均非缺失）
        overlap = (~isna[:, None, :]) & (~isna[None, :, :])     # (n,n,d)
        overlap_cnt = overlap.sum(axis=2).astype(np.int32)      # (n,n)

        # 匹配位（相等且可比）
        eq = (X[:, None, :] == X[None, :, :]) & overlap
        match_cnt = eq.sum(axis=2).astype(np.int32)             # (n,n)

        # 相似度（匹配数 / 可比位数）
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = match_cnt / overlap_cnt.astype(np.float32)    # (n,n)

        # 过滤：可比位不足 或 相似度不足 的设为 -inf（不可选）
        min_compared_eff = max(1, int(min_compared))
        sim[overlap_cnt < min_compared_eff] = -np.inf
        sim[sim < float(min_similarity)] = -np.inf

        # 去掉自环
        np.fill_diagonal(sim, -np.inf)

        # 为每行选择前 K 个最高相似度（若不足 K，就返回能满足条件的全部）
        k_eff = max(0, min(K, n - 1))
        ids_arr = np.array(ids)

        if k_eff == 0:
            for cid in ids:
                neighbors[cid] = []
            continue

        # 先取前 k_eff 大的小顶堆分区，再在这 k_eff 内按 (相似度降序, 邻居id 升序) 稳定排序
        part = np.argpartition(sim, kth=n - k_eff, axis=1)[:, -k_eff:]   # 取最大的 k_eff 个索引（未排序）
        row_idx = np.arange(n)[:, None]
        # 对候选做稳定排序
        # 注意：需要把 -inf 过滤掉
        for i in range(n):
            cand = part[i].tolist()
            # 过滤 -inf（不满足阈值的配对）
            cand = [j for j in cand if sim[i, j] > -np.inf]
            # 排序：先按相似度降序，再按邻居ID升序
            cand.sort(key=lambda j: (-sim[i, j], ids_arr[j]))
            # 截断到 K
            cand = cand[:k_eff]
            neighbors[ids_arr[i]] = ids_arr[cand].tolist()

    return neighbors


from random import Random


# Load communities of testing set from /home/ruomeng/gae/dataset/ces_golden_demo/raw/24/neighbors_24.jsonl
year = '24'
jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/neighbors_{year}.jsonl"  


# Load features of each nodes from /home/ruomeng/gae/dataset/ces_golden_demo/raw/24/questions_test_24.csv
df_feature = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/questions_test_{year}.csv")

caseid_list = df_feature['caseid'].astype(str).tolist()


df_demo = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/identity_{year}.csv")
df_demo['caseid'] = df_demo['caseid'].astype(str)
caseid_set = {str(c) for c in caseid_list}  # set is faster for large lists
df_demo = df_demo[df_demo['caseid'].isin(caseid_set)].copy()

comms = {state: grp['caseid'].astype(str).tolist() for state, grp in df_demo.groupby('inputstate', dropna=False)}
caseids = {cid for cids in comms.values() for cid in cids}
for c in comms:
    print(c, len(comms[c]))
input()
dict_features = df_to_feature_map(df_feature, id_col="caseid")

# neighbors = neighbors_by_hamming_from_com(comms, dict_features, K=20, min_overlap=1)
neighbors = neighbors_by_hamming_from_com(
    comms,
    dict_features,
    K=20,
    min_similarity=0.8,  # 至少 70% 维度匹配
    min_compared=1       # 至少 1 个可比维度（可按需提高，比如 10/15）
)
print(neighbors)

for node in neighbors:
    if len(neighbors[node]) == 0:
        neighbors[node] = [node]

# Invert noisy_com_dict to {caseid -> community}
caseid2com = {str(cid): str(com)
              for com, ids in comms.items()
              for cid in ids}

# Union of all caseids we might care about (robust to missing keys)
all_caseids = set(caseid2com.keys()) | set(map(str, neighbors.keys()))

out_path = "/home/ruomeng/gae/dataset/ces_golden_demo/raw/24/neighbors_demo_24.jsonl"  # <- change if needed
with open(out_path, "w", encoding="utf-8") as f:
    for cid in sorted(all_caseids):
        rec = {
            "caseid": cid,
            "community": caseid2com.get(cid, ""),  # empty if not found (shouldn't happen)
            "neighbors": [str(x) for x in neighbors.get(cid, [])],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", out_path)



res = summarize_intra_disagreement(
    com_dict=comms,          # or original com_dict
    dict_features=dict_features,
    min_overlap=1,                    # raise for stricter comparisons
    micro_weighted_majority=False,    # True to weight features by #valid rows
)
summary_path = "/home/ruomeng/gae/dataset/ces_golden_demo/raw/24/intra_disagreement_summary_demo_com_dict.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Wrote:", summary_path)
