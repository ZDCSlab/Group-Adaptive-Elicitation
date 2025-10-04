import numpy as np
import pandas as pd
from collections import Counter
import torch
import json
import random

def split_questions(question_set, train_ratio=0.8, seed=42):
    """
    Randomly split a set/list of questions into train and hold-out.

    Args:
        question_set (list): list of question IDs (or names).
        train_ratio (float): fraction to put in train split (default 0.8).
        seed (int): random seed for reproducibility.

    Returns:
        train_qs (list), holdout_qs (list)
    """
    random.seed(seed)
    qs = list(question_set)
    random.shuffle(qs)

    n_train = int(len(qs) * train_ratio)
    train_qs = qs[:n_train]
    holdout_qs = qs[n_train:]
    return train_qs, holdout_qs


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
from sklearn.metrics import pairwise_distances
import networkx as nx
from sklearn.cluster import SpectralClustering

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import networkx as nx

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
import networkx as nx

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
import networkx as nx
from collections import Counter

def knn_communities_sized(
    df: pd.DataFrame,
    id_col: str = "caseid",
    *,
    s_min: float = 0.85,          # keep edge if S >= s_min
    deg_cap: int | None = 20,     # per-node degree cap (None = no cap)
    mutual: bool = False,         # keep only reciprocal edges after thresholding
    min_size: int = 30,           # << 新增: 只保留 >= min_size 的社区
    max_size: int | None = 250,   # communities larger than this will be split
    neighbor_k: int = 30,         # how many top neighbors to return (within community)
    missing_vals = (-1, np.nan),
    random_state: int = 42,
):
    """
    Returns (只含 size >= min_size 的社区与成员):
      communities: DataFrame [caseid, community]  (community 从0开始连续编号)
      stats      : dict community_id -> {'size','avg_internal_sim','min_internal_sim'}
      G          : nx.Graph (仅含保留成员的诱导子图)
      S          : similarity matrix (m x m)  (m = 保留成员数)
      neighbors  : DataFrame [caseid, topk_ids, topk_sims, neighbor_modes(dict)]
                   其中 Top-K 与众数仅在“各自社区内部”计算
    """
    # ---------- prep ----------
    ids_all = df[id_col].astype(str).tolist()
    Qcols = [c for c in df.columns if c != id_col]
    X = df[Qcols].copy()
    for mv in missing_vals:
        X = X.replace(mv, np.nan)
    X = X.apply(lambda col: col.fillna(col.mode().iloc[0] if not col.mode().empty else col.dropna().iloc[0]))
    X = X.astype(int).to_numpy()
    n, q = X.shape

    # ---------- similarity ----------
    D = pairwise_distances(X, metric="hamming")
    S_full = 1.0 - D
    np.fill_diagonal(S_full, 1.0)

    # ---------- build ε-graph ----------
    adj = []
    for i in range(n):
        js = np.where(S_full[i] >= s_min)[0]
        js = js[js != i]
        if js.size == 0:
            continue
        order = js[np.argsort(-S_full[i, js])]
        if deg_cap is not None:
            order = order[:deg_cap]
        for j in order:
            if i < j:
                adj.append((i, j, float(S_full[i, j])))

    if mutual:
        nbrs = [set() for _ in range(n)]
        for i, j, _ in adj:
            nbrs[i].add(j)
            nbrs[j].add(i)
        adj = [(i, j, w) for i, j, w in adj if (i in nbrs[j] and j in nbrs[i])]

    G_full = nx.Graph()
    G_full.add_nodes_from(range(n))
    G_full.add_weighted_edges_from(adj)

    # ---------- components with size constraints (max_size) ----------
    comps = [set(c) for c in nx.connected_components(G_full)]

    def split_block(nodes: set[int], next_cid: int):
        size = len(nodes)
        if max_size is None or size <= max_size:
            return ({u: next_cid for u in nodes}, next_cid + 1)
        k = int(np.ceil(size / max_size))
        idx_nodes = np.array(list(nodes))
        subS = S_full[np.ix_(idx_nodes, idx_nodes)]
        try:
            sc = SpectralClustering(
                n_clusters=k, affinity="precomputed",
                assign_labels="kmeans", random_state=random_state
            )
            sub_labels = sc.fit_predict(subS)
        except Exception:
            s_mean = subS.mean(axis=1)
            order = idx_nodes[np.argsort(-s_mean)]
            sub_labels = np.zeros_like(idx_nodes)
            for g, start in enumerate(range(0, size, max_size)):
                sub_labels[np.isin(idx_nodes, order[start:start+max_size])] = g
        mapping, cid_cur = {}, next_cid
        for g in np.unique(sub_labels):
            subnodes = set(idx_nodes[sub_labels == g].tolist())
            sub_map, cid_cur = split_block(subnodes, cid_cur)
            mapping.update(sub_map)
        return mapping, cid_cur

    comp_ids, next_cid = {}, 0
    for comp in comps:
        mapping, next_cid = split_block(comp, next_cid)
        comp_ids.update(mapping)
    labels = np.array([comp_ids[i] for i in range(n)])

    # ---------- keep only communities with size >= min_size ----------
    # 计算每个社区规模
    unique_c = np.unique(labels)
    sizes = {c: int(np.sum(labels == c)) for c in unique_c}
    keep_c = sorted([int(c) for c in unique_c if sizes[int(c)] >= min_size])

    # 过滤成员索引
    keep_mask = np.isin(labels, keep_c)
    idx_keep = np.where(keep_mask).to_list()[0] if hasattr(np.where(keep_mask), "to_list") else np.where(keep_mask)[0]
    if len(idx_keep) == 0:
        # 没有任何满足 min_size 的社区
        # 返回空结果（按同样结构）
        empty_df = pd.DataFrame(columns=[id_col, "community"])
        return empty_df, {}, nx.Graph(), np.zeros((0, 0), dtype=np.float32), pd.DataFrame(columns=[id_col, "topk_ids", "topk_sims", "neighbor_modes"])

    # 压缩后的数据
    ids = [ids_all[i] for i in idx_keep]
    X_keep = X[idx_keep]
    labels_keep_old = labels[idx_keep]

    # 把社区ID重新映射为 0..C-1 连续编号
    c_map = {c_old: i_new for i_new, c_old in enumerate(keep_c)}
    labels_keep = np.array([c_map[int(c)] for c in labels_keep_old], dtype=int)

    # 相似度子矩阵、诱导子图
    S = S_full[np.ix_(idx_keep, idx_keep)].astype(np.float32)
    m = len(idx_keep)
    # 诱导子图：保留的节点重编号为 0..m-1
    # G = nx.Graph()
    # G.add_nodes_from(range(m))
    # # 从 S 与阈值再建一次边（与上面一致的规则）
    # for i in range(m):
    #     js = np.where(S[i] >= s_min)[0]
    #     js = js[js != i]
    #     if js.size == 0:
    #         continue
    #     order = js[np.argsort(-S[i, js])]
    #     if deg_cap is not None:
    #         order = order[:deg_cap]
    #     for j in order:
    #         if i < j:
    #             G.add_edge(i, j, weight=float(S[i, j]))
    # if mutual:
    #     # 再做一遍互惠裁剪
    #     edges = list(G.edges(data=True))
    #     nbrs = [set() for _ in range(m)]
    #     for u, v, _ in edges:
    #         nbrs[u].add(v)
    #         nbrs[v].add(u)
    #     for u, v, _ in edges:
    #         if not (u in nbrs[v] and v in nbrs[u]):
    #             if G.has_edge(u, v):
    #                 G.remove_edge(u, v)

    # ---------- 只在社区内部取 Top-K ----------
    topk_ids, topk_sims_list, neighbor_modes = [], [], []
    K_global = neighbor_k

    for i in range(m):
        c = labels_keep[i]
        members = np.where(labels_keep == c)[0]

        # 在社区内部取 Top-K（去掉自身）
        S_sub = S[i, members]                      # [size_c]
        idx_sub = members
        # 设置自身为 -inf，防止被选中
        self_pos = np.where(idx_sub == i)[0]
        S_sub = S_sub.copy()
        if self_pos.size > 0:
            S_sub[self_pos[0]] = -np.inf

        K_eff = int(min(max(1, K_global), max(1, len(idx_sub) - 1)))
        # 取前 K_eff
        cand_idx = np.argpartition(-S_sub, K_eff)[:K_eff]
        sims_topk = S_sub[cand_idx]
        # 排序
        order = np.argsort(-sims_topk)
        cand_idx = cand_idx[order]
        sims_topk = sims_topk[order]

        nbr_idx = idx_sub[cand_idx]               # 全局(压缩后)索引
        topk_ids.append([ids[j] for j in nbr_idx])
        topk_sims_list.append([float(x) for x in sims_topk])

        # 邻居众数答案（只在本社区邻居上）
        neigh_answers = X_keep[nbr_idx]           # [K, q]
        mode_dict = {}
        for qi, qname in enumerate(Qcols):
            cnts = Counter(neigh_answers[:, qi].tolist())
            mode_dict[qname] = cnts.most_common(1)[0][0] if cnts else None
        neighbor_modes.append(mode_dict)

    neighbors = pd.DataFrame({
        id_col: ids,
        "topk_ids": topk_ids,
        "topk_sims": topk_sims_list,
        "neighbor_modes": neighbor_modes
    })

    # ---------- 社区表（只含保留成员） ----------
    communities = pd.DataFrame({id_col: ids, "community": labels_keep})

    # ---------- stats（只对保留社区） ----------
    stats = {}
    for c in np.unique(labels_keep):
        members = np.where(labels_keep == c)[0]
        if len(members) <= 1:
            stats[int(c)] = {"size": int(len(members)),
                             "avg_internal_sim": None,
                             "min_internal_sim": None}
        else:
            subS = S[np.ix_(members, members)]
            triu = subS[np.triu_indices_from(subS, k=1)]
            stats[int(c)] = {"size": int(len(members)),
                             "avg_internal_sim": float(triu.mean()),
                             "min_internal_sim": float(triu.min())}

    return communities, stats, neighbors

import numpy as np
import pandas as pd

def split_by_community(df, id_col="id", community_col="community",
                       train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split respondents into train/val/test by communities.

    Args:
        df: pd.DataFrame with at least [id_col, community_col]
        id_col: column name for respondent IDs
        community_col: column name for community labels
        train_ratio: fraction of communities for training
        val_ratio: fraction of communities for validation
        seed: random seed for reproducibility

    Returns:
        dict with keys 'train', 'val', 'test',
        each a DataFrame containing respondents
    """
    rng = np.random.RandomState(seed)
    comms = df[community_col].unique().tolist()
    rng.shuffle(comms)

    n_train = int(len(comms) * train_ratio)
    n_val   = int(len(comms) * val_ratio)

    train_comms = comms[:n_train]
    val_comms   = comms[n_train:n_train+n_val]
    test_comms  = comms[n_train+n_val:]

    train_df = df[df[community_col].isin(train_comms)]
    val_df   = df[df[community_col].isin(val_comms)]
    test_df  = df[df[community_col].isin(test_comms)]

    return {"train": train_df, "val": val_df, "test": test_df}


def split_communities(stats, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split communities into train/val/test disjoint sets.

    Args:
        stats: dict[community_id -> dict with community info] 
               (at least the keys exist, values can be anything)
        train_ratio: fraction of communities for training
        val_ratio: fraction for validation (rest is test)
        seed: random seed for reproducibility

    Returns:
        dict with keys 'train', 'val', 'test',
        each a list of community IDs
    """
    rng = np.random.RandomState(seed)
    comms = list(stats.keys())
    rng.shuffle(comms)

    n_train = int(len(comms) * train_ratio)
    n_val   = int(len(comms) * val_ratio)

    train_comms = comms[:n_train]
    val_comms   = comms[n_train:n_train+n_val]
    test_comms  = comms[n_train+n_val:]

    return {
        "train": train_comms,
        "val": val_comms,
        "test": test_comms,
    }


def check_neighbor_majority(
    df: pd.DataFrame,
    neighbors: pd.DataFrame,
    communities: pd.DataFrame,
    *,
    id_col: str = "caseid",
    question_set: list[str] | None = None,
    missing_vals = (-1, np.nan),
):
    # --- safety: unify dtypes & columns ---
    df2 = df.copy()
    df2[id_col] = df2[id_col].astype(str)

    neighbors = neighbors.copy()
    neighbors[id_col] = neighbors[id_col].astype(str)

    communities = communities.copy()
    communities[id_col] = communities[id_col].astype(str)

    if question_set is None:
        question_set = [c for c in df2.columns if c != id_col]

    # treat custom missing codes as NaN
    for mv in missing_vals:
        df2[question_set] = df2[question_set].replace(mv, np.nan)
    # ensure numeric (coerce strange strings to NaN)
    df2[question_set] = df2[question_set].apply(pd.to_numeric, errors="coerce")

    # --- fast lookups ---
    mode_map = dict(zip(neighbors[id_col], neighbors["neighbor_modes"]))   # {caseid: {q: mode}}
    comm_map = dict(zip(communities[id_col], communities["community"]))    # {caseid: community}

    # keep only respondents that actually have neighbor modes
    df2 = df2[df2[id_col].isin(mode_map.keys())].reset_index(drop=True)

    # --- row-wise compare ---
    def compare_to_neighbor_mode(row):
        cid = row[id_col]
        modes = mode_map.get(cid, {}) or {}
        out = {}
        for q in question_set:
            v = row[q]
            m = modes.get(q, None)
            if pd.isna(v) or (m is None):
                out[q] = np.nan
            else:
                # compare as floats; if anything goes wrong, mark NaN
                try:
                    out[q] = float(v) == float(m)
                except Exception:
                    out[q] = np.nan
        return pd.Series(out, index=question_set)

    # Per-(caseid, question) agreement matrix (True/False/NaN)
    agree_matrix = df2.apply(compare_to_neighbor_mode, axis=1)
    agree_matrix.insert(0, id_col, df2[id_col].values)
    agree_matrix.insert(1, "community", df2[id_col].map(comm_map).values)

    # Per-respondent summary: mean over questions, skipping NaN
    match_rate = agree_matrix[question_set].astype(float).mean(axis=1, skipna=True)
    summary = pd.DataFrame({
        id_col: agree_matrix[id_col],
        "community": agree_matrix["community"],
        "match_rate": match_rate
    })

    # Community-level summary (drop rows without community)
    comm_summary = (
        summary.dropna(subset=["community"])
               .groupby("community", as_index=False)
               .agg(size=(id_col, "count"),
                    avg_match_rate=("match_rate", "mean"))
               .sort_values("size", ascending=False)
    )

    return agree_matrix, summary, comm_summary


if __name__ == "__main__":

    for year in ['20', '22', '24']:

        file_path = f'/home/ruomeng/gae/dataset/ces/raw/{year}/question_{year}.csv'
        survey_data = pd.read_csv(file_path)
        
        jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/question_codebook.jsonl"  
        codebook = load_jsonl_as_dict_of_dict(jsonl_file, key='id')
        question_set = []
        target = {"1": "Support", "2": "Oppose"}

        for caseid in codebook.keys():
            if codebook[caseid]["options"] == target:
                question_set.append(caseid)

        train_qs, holdout_qs = split_questions(question_set, train_ratio=0.67, seed=42)
        print('len(question_set)', len(question_set), len(train_qs), len(holdout_qs))

        # Assuming you already have train_qs and holdout_qs
        train_df = pd.DataFrame({"train_qs": train_qs})
        holdout_df = pd.DataFrame({"holdout_qs": holdout_qs})

        # Save to CSV
        train_df.to_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/train_qs.csv", index=False)
        holdout_df.to_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/holdout_qs.csv", index=False)

        print("Saved train_qs.csv and holdout_qs.csv")

        survey_data_filtered = survey_data[["caseid"] + question_set]

        # print(survey_data_filtered.head())

        communities, stats, neighbors = knn_communities_sized(
            survey_data_filtered,
            id_col="caseid",
            s_min=0.85,
            deg_cap=20,
            mutual=True,
            min_size=30,
            max_size=100,
            neighbor_k=20,
        )
        

        print('len(stats)', len(stats))
        sizes = [v["size"] for v in stats.values()]
        print("max size =", max(sizes))
        print("min size =", min(sizes))
        # for k, v in stats.items():
        #     print("community:", k, " stats:", v)
        #     # v 是一个字典，可以进一步访问
        #     print("size =", v["size"])

        splits = split_by_community(communities, id_col="id", community_col="community",
                            train_ratio=0.7, val_ratio=0.15, seed=42)

        print("Train respondents:\n", splits["train"])
        print("Val respondents:\n", splits["val"])
        print("Test respondents:\n", splits["test"])
        # print(survey_data["caseid"])

        # 
        for split in ['train', 'val', 'test']:
            survey_data_filtered_seen = survey_data[["caseid"] + question_set].copy()
            communities = communities.copy()  
            survey_data_filtered_seen["caseid"] = survey_data_filtered_seen["caseid"].astype(str)
            communities["caseid"] = communities["caseid"].astype(str)
            survey_data_filtered_seen = survey_data_filtered_seen[
                survey_data_filtered_seen["caseid"].isin(splits[f"{split}"]["caseid"])]
            survey_data_filtered_seen.to_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/questions_{split}_{year}.csv", index=False)


    
        id_col = "caseid"
        df_merged = pd.merge(neighbors, communities, on="caseid", how="inner")
        subset = df_merged[[id_col, "community", "topk_ids"]]
        records = subset.to_dict(orient="records")

        out_path = f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/neighbors_{year}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        neighbors.to_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/neighbors_info_{year}.csv")
        
        agree_matrix, summary, comm_summary = check_neighbor_majority(
            df=survey_data_filtered, neighbors=neighbors, communities=communities,
            id_col="caseid", question_set=question_set, missing_vals=(-1, np.nan)
        )

        summary.to_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/cse_golden_analysis_{year}.csv")
        print("summary['match_rate'].mean()", summary['match_rate'].mean())




            

        
    