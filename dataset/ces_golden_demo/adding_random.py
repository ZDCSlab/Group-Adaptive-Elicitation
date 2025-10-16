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


def exchange_noise(
    com_dict: Dict[str, List[str]],
    *,
    frac: float = 0.1,
    min_per_comm: int = 0,
    seed: int | None = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, str]]]]:
    """
    Exchange nodes across communities to inject noise, preserving each community's size.

    Strategy:
      1) Select k_c nodes from each community c (k_c = max(min_per_comm, int(len(c)*frac))).
      2) Remove them from their communities.
      3) Reassign ALL removed nodes to (possibly many) other communities with per-community
         capacity equal to k_c, avoiding self-returns. This is a capacitated matching.

    Returns
    -------
    new_com_dict : dict[community] -> list[caseid]
    moves_log    : dict[community] -> list[(caseid, new_community)]  # moved OUT of that community
    """
    rng = random.Random(seed)

    # Work on copies so original input is not mutated
    new_com = {c: list(nodes) for c, nodes in com_dict.items()}

    # 1) Choose nodes to move from each community
    selected = {}
    caps = {}  # capacity = # we must receive back to preserve size
    for c, nodes in new_com.items():
        n = len(nodes)
        k = max(min_per_comm, int(n * frac))
        k = min(k, n)
        if k > 0:
            chosen = rng.sample(nodes, k)
            selected[c] = chosen
            caps[c] = k

    # If fewer than 2 communities have selections, nothing to exchange
    sel_comms = [c for c in selected]
    if len(sel_comms) < 2:
        return new_com, defaultdict(list)

    # 2) Remove selected nodes from their source communities
    for c, chosen in selected.items():
        remove = set(chosen)
        new_com[c] = [x for x in new_com[c] if x not in remove]

    # 3) Build donor pools and recipient capacities (avoid self-assign)
    donors: List[Tuple[str, str]] = []  # (src_comm, node)
    for c, nodes in selected.items():
        for u in nodes:
            donors.append((c, u))
    rng.shuffle(donors)

    # Recipients with remaining capacity
    remaining_cap = dict(caps)  # how many nodes each community must receive
    moves_log = defaultdict(list)

    # Helper: pick a recipient != src with available capacity
    def pick_recipient_not_src(src: str) -> str | None:
        # Randomized order to avoid bias
        keys = list(remaining_cap.keys())
        rng.shuffle(keys)
        for dst in keys:
            if dst != src and remaining_cap[dst] > 0:
                return dst
        return None

    # 4) First pass: greedy assignment while we can avoid self-assignments
    backlog: List[Tuple[str, str]] = []  # donors we couldn't place yet
    for src, u in donors:
        dst = pick_recipient_not_src(src)
        if dst is None:
            backlog.append((src, u))
            continue
        new_com[dst].append(u)
        remaining_cap[dst] -= 1
        moves_log[src].append((u, dst))

    # 5) Resolve backlog (only possible issue is everyone left wants to return to themselves)
    # We fix this by pair/3-cycle swaps among remaining capacities to avoid self-returns.
    # Build a list of open recipient slots (dst repeated by its remaining capacity)
    open_slots: List[str] = []
    for dst, cap in remaining_cap.items():
        open_slots.extend([dst] * cap)

    # Sanity: counts must match
    assert len(backlog) == len(open_slots), "Internal mismatch: backlog vs capacities."

    # Try to assign greedily, repairing self-assignments by swaps
    # Convert to list to allow swapping
    backlog_srcs = [src for (src, _) in backlog]
    backlog_nodes = [u for (_, u) in backlog]

    # Initial one-to-one pairing (may contain self-assignments)
    # We’ll fix self-assignments by swapping destinations among positions
    # to ensure each (src != assigned_dst)
    # Shuffle open_slots to reduce chance of many self-assignments
    rng.shuffle(open_slots)

    # Map positions where open_slots[i] == backlog_srcs[i] (self-assign) and repair
    for i in range(len(backlog_nodes)):
        if open_slots[i] == backlog_srcs[i]:
            # Find a j>i to swap with where swap fixes both positions (or at least i)
            swapped = False
            for j in range(i + 1, len(backlog_nodes)):
                # Try simple swap
                if open_slots[j] != backlog_srcs[i] and open_slots[i] != backlog_srcs[j]:
                    open_slots[i], open_slots[j] = open_slots[j], open_slots[i]
                    swapped = True
                    break
            if not swapped:
                # As last resort (rare when >=3 comms), try any j and accept that we fix i at least
                for j in range(i + 1, len(backlog_nodes)):
                    if open_slots[j] != backlog_srcs[i]:
                        open_slots[i], open_slots[j] = open_slots[j], open_slots[i]
                        swapped = True
                        break
            # If still not swapped, it means only two communities and both positions clash.
            # Handle the 2-comm corner by swapping node assignments between the two donors.
            if not swapped and len(backlog_nodes) == 2:
                open_slots[0], open_slots[1] = open_slots[1], open_slots[0]

    # Now apply the repaired assignments
    for (src, u), dst in zip(backlog, open_slots):
        # After repair, we should have src != dst
        if src == dst:
            # If this happens, we’re in a degenerate case; bail gracefully w/o corruption:
            # put the node back to its original community (no net change for that node).
            # (This should practically never trigger except pathological 1-comm cases.)
            new_com[src].append(u)
            moves_log[src].append((u, src))  # indicates no-op
        else:
            new_com[dst].append(u)
            moves_log[src].append((u, dst))

    # 6) Invariants: per-community sizes preserved
    for c in com_dict:
        assert len(new_com[c]) == len(com_dict[c]), f"Size drift in community {c}"

    # Global invariant
    assert sum(len(v) for v in new_com.values()) == sum(len(v) for v in com_dict.values())

    return new_com, moves_log

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

from random import Random


def random_split_into_N_groups(nodes, N, *, seed=None, strict_equal=False):
    """
    Randomly split `nodes` into N groups with sizes as equal as possible.

    Args:
        nodes (list): items to split
        N (int): number of groups
        seed (int|None): RNG seed for reproducibility
        strict_equal (bool): if True, require len(nodes) % N == 0 (exact equality)

    Returns:
        dict[int, list]: {0: [...], 1: [...], ..., N-1: [...]}
    """
    assert N > 0, "N must be positive"
    pool = list(nodes)
    if strict_equal:
        assert len(pool) % N == 0, "len(nodes) must be divisible by N for strict equality"
    assert N <= len(pool), "N cannot exceed number of nodes (no empty groups)"

    rng = Random(seed)
    rng.shuffle(pool)

    n = len(pool)
    base = n // N            # minimum size per group
    r = n % N                # first r groups get one extra

    comms = {}
    idx = 0
    for cid in range(N):
        sz = base + (1 if cid < r else 0)
        comms[cid] = pool[idx: idx + sz]
        idx += sz
    return comms



# Load communities of testing set from /home/ruomeng/gae/dataset/ces_golden_demo/raw/24/neighbors_24.jsonl
year = '24'
jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/neighbors_{year}.jsonl"  


# Load features of each nodes from /home/ruomeng/gae/dataset/ces_golden_demo/raw/24/questions_test_24.csv
df_feature = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/questions_test_{year}.csv")

caseid_list = df_feature['caseid'].astype(str).tolist()
print(caseid_list)
comms = random_split_into_N_groups(caseid_list, 20, seed=42)

dict_features = df_to_feature_map(df_feature, id_col="caseid")

neighbors_same_noisy = neighbors_by_hamming_from_com(comms, dict_features, K=20, min_overlap=1)

# print(neighbors_same_noisy)

# Invert noisy_com_dict to {caseid -> community}
caseid2com = {str(cid): str(com)
              for com, ids in comms.items()
              for cid in ids}

# Union of all caseids we might care about (robust to missing keys)
all_caseids = set(caseid2com.keys()) | set(map(str, neighbors_same_noisy.keys()))

out_path = "/home/ruomeng/gae/dataset/ces_golden_demo/raw/24/neighbors_random_24.jsonl"  # <- change if needed
with open(out_path, "w", encoding="utf-8") as f:
    for cid in sorted(all_caseids):
        rec = {
            "caseid": cid,
            "community": caseid2com.get(cid, ""),  # empty if not found (shouldn't happen)
            "neighbors": [str(x) for x in neighbors_same_noisy.get(cid, [])],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", out_path)



res = summarize_intra_disagreement(
    com_dict=comms,          # or original com_dict
    dict_features=dict_features,
    min_overlap=1,                    # raise for stricter comparisons
    micro_weighted_majority=False,    # True to weight features by #valid rows
)
summary_path = "/home/ruomeng/gae/dataset/ces_golden_demo/raw/24/intra_disagreement_summary_random_com_dict.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Wrote:", summary_path)
