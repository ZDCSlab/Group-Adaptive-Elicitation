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

def knn_communities_sized(
    df: pd.DataFrame,
    id_col: str = "caseid",
    *,
    # --- graph construction ---
    graph_mode: str = "mutual_knn",      # 'epsilon' | 'mutual_knn' | 'snn'
    s_min: float = 0.85,                  # used by 'epsilon' (similarity = 1 - hamming)
    k_core: int = 12,                     # neighbors used to form the graph
    deg_cap: int | None = 40,             # hard cap per-node degree (None = no cap)
    mutual: bool | None = None,           # if None: auto (True for 'mutual_knn'); else force
    snn_min_shared: int = 4,              # SNN: min shared neighbors to keep an edge
    # --- community sizing ---
    min_size: int = 60,                   # more conservative default to avoid tiny comms
    max_size: int | None = 400,           # split very large comps by BFS chunking
    ensure_min_degree: int = 3,           # drop edges to raise under-degree? we *prune nodes* below this
    # --- neighbor outputs ---
    neighbor_k: int = 30,                 # per-node top-K neighbors (within final community)
    # --- merging small communities ---
    merge_small: bool = True,
    merge_min_sim: float = 0.78,          # min centroid sim (1-hamming) to allow merge
    merge_max_passes: int = 3,
    # --- misc ---
    missing_vals = (-1, np.nan),
    random_state: int = 42,
):
    """
    Returns:
      communities: DataFrame [caseid, community]  (0..C-1)
      stats      : {cid: {'size','avg_internal_sim','min_internal_sim'}}
      neighbors  : DataFrame [caseid, topk_ids, topk_sims, neighbor_modes]
    """
    rng = np.random.default_rng(random_state)

    # ---------- prep & encoding to non-negative ints ----------
    ids_all = df[id_col].astype(str).tolist()
    Qcols = [c for c in df.columns if c != id_col]
    Xraw = df[Qcols].copy()

    repl_vals = [v for v in missing_vals if not (isinstance(v, float) and np.isnan(v))]
    if repl_vals:
        Xraw = Xraw.replace(repl_vals, np.nan)

    X_list = []
    for c in Qcols:
        col = Xraw[c]
        if col.isna().all():
            X_list.append(np.zeros(len(col), dtype=int))
            continue
        # detect non-numeric -> factorize to codes
        col_num = pd.to_numeric(col, errors="coerce")
        if col_num.isna().sum() and (col.dtype == object):
            mask = col.notna()
            codes, _ = pd.factorize(col[mask], sort=True)
            filled = np.full(len(col), np.nan)
            filled[mask] = codes
            vals = filled[~np.isnan(filled)]
            mode_val = int(pd.Series(vals).mode().iloc[0]) if len(vals) else 0
            filled = np.where(np.isnan(filled), mode_val, filled).astype(int)
            X_list.append(filled)
        else:
            vals = col_num
            if vals.isna().any():
                mv = vals.mode(dropna=True)
                fillv = (mv.iloc[0] if not mv.empty else vals.dropna().iloc[0])
                vals = vals.fillna(fillv)
            X_list.append(vals.astype(int).to_numpy())
    X = np.column_stack(X_list).astype(int, copy=False)
    n, q = X.shape
    if n == 0 or q == 0:
        empty_df = pd.DataFrame(columns=[id_col, "community"])
        return empty_df, {}, pd.DataFrame(columns=[id_col, "topk_ids", "topk_sims", "neighbor_modes"])

    # ---------- kNN search (shared for all modes) ----------
    # slight buffer > k_core to tolerate thresholding
    nn_k = max(k_core + 1, 2)
    nbrs = NearestNeighbors(metric="hamming", n_neighbors=min(nn_k, n), n_jobs=-1)
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)
    sims = 1.0 - dists  # similarity

    # ---------- build graph ----------
    if mutual is None:
        mutual_eff = (graph_mode == "mutual_knn")
    else:
        mutual_eff = bool(mutual)
    edges = []

    if graph_mode == "epsilon":
        for i in range(n):
            js = idxs[i]
            ss = sims[i]
            m = js != i
            js, ss = js[m], ss[m]
            keep = ss >= s_min
            js, ss = js[keep], ss[keep]
            if deg_cap is not None and js.size > deg_cap:
                order = np.argsort(-ss)[:deg_cap]
                js, ss = js[order], ss[order]
            for j, w in zip(js, ss):
                a, b = (i, j) if i < j else (j, i)
                edges.append((a, b, float(w)))

    elif graph_mode == "mutual_knn":
        # edge if i in top-k of j AND j in top-k of i
        topk = [set(row[1: min(k_core + 1, len(row))]) for row in idxs]  # drop self if present
        for i in range(n):
            for j in topk[i]:
                if i == j:
                    continue
                if i in topk[j]:
                    a, b = (i, j) if i < j else (j, i)
                    # use average sim for weight
                    w = (float(sims[i, np.where(idxs[i]==j)[0][0]])
                         + float(sims[j, np.where(idxs[j]==i)[0][0]])) / 2.0
                    edges.append((a, b, w))
        # degree cap here is usually unnecessary; mutual already limits degree

    elif graph_mode == "snn":
        # Shared Nearest Neighbor: weight = #shared in top-k
        topk = [set(row[1: min(k_core + 1, len(row))]) for row in idxs]
        for i in range(n):
            Ni = topk[i]
            for j in Ni:
                if i >= j:
                    continue
                Nj = topk[j]
                shared = len(Ni & Nj)
                if shared >= snn_min_shared:
                    edges.append((i, j, float(shared)))

    else:
        raise ValueError("graph_mode must be one of {'epsilon','mutual_knn','snn'}")

    # Undirected unique edges; average duplicate weights if any
    if not edges:
        empty_df = pd.DataFrame(columns=[id_col, "community"])
        return empty_df, {}, pd.DataFrame(columns=[id_col, "topk_ids", "topk_sims", "neighbor_modes"])

    agg = defaultdict(list)
    for i, j, w in edges:
        a, b = (i, j) if i < j else (j, i)
        agg[(a, b)].append(w)
    edges = [(i, j, float(np.mean(ws))) for (i, j), ws in agg.items()]

    G_full = nx.Graph()
    G_full.add_nodes_from(range(n))
    G_full.add_weighted_edges_from(edges)

    # prune nodes with degree < ensure_min_degree (they tend to form tiny components)
    if ensure_min_degree > 0:
        low_deg = [u for u, d in G_full.degree() if d < ensure_min_degree]
        if low_deg:
            G_full.remove_nodes_from(low_deg)

    if G_full.number_of_nodes() == 0:
        empty_df = pd.DataFrame(columns=[id_col, "community"])
        return empty_df, {}, pd.DataFrame(columns=[id_col, "topk_ids", "topk_sims", "neighbor_modes"])

    # ---------- split oversized components by BFS chunking ----------
    def split_component_greedy(G: nx.Graph, nodes_set: set[int], max_block: int | None):
        nodes = list(nodes_set)
        if max_block is None or len(nodes) <= max_block:
            return [nodes]
        chunks, visited = [], set()
        for start in nodes:
            if start in visited:
                continue
            cur = []
            dq = [start]
            visited.add(start)
            while dq:
                u = dq.pop()
                cur.append(u)
                if max_block is not None and len(cur) >= max_block:
                    chunks.append(cur)
                    cur = []
                for v in G.neighbors(u):
                    if v not in visited and v in nodes_set:
                        visited.add(v)
                        dq.append(v)
            if cur:
                chunks.append(cur)
        return chunks

    blocks = []
    for comp in nx.connected_components(G_full):
        blocks.extend(split_component_greedy(G_full, set(comp), max_size))

    # initial labels
    labels = np.empty(n, dtype=int)
    labels.fill(-1)  # -1 for nodes that were pruned
    cid = 0
    for block in blocks:
        for u in block:
            labels[u] = cid
        cid += 1

    # keep only labeled nodes
    keep_idx = np.flatnonzero(labels >= 0)
    if keep_idx.size == 0:
        empty_df = pd.DataFrame(columns=[id_col, "community"])
        return empty_df, {}, pd.DataFrame(columns=[id_col, "topk_ids", "topk_sims", "neighbor_modes"])

    labels0 = labels[keep_idx]
    ids0 = [ids_all[i] for i in keep_idx]
    X0 = X[keep_idx]
    G0 = G_full.subgraph(keep_idx).copy()

    # ---------- (optional) merge small communities ----------
    def community_centroid_modes(X_block: np.ndarray) -> np.ndarray:
        # per-column mode via bincount
        vmax = X_block.max(axis=0)
        out = []
        for j in range(X_block.shape[1]):
            counts = np.bincount(X_block[:, j], minlength=int(vmax[j]) + 1)
            out.append(int(np.argmax(counts)))
        return np.array(out, dtype=int)

    def centroid_sim(mode_a: np.ndarray, mode_b: np.ndarray) -> float:
        # similarity as 1 - hamming between centroid codes
        return 1.0 - np.mean(mode_a != mode_b)

    if merge_small:
        for _ in range(merge_max_passes):
            # sizes
            cids, counts = np.unique(labels0, return_counts=True)
            size_map = dict(zip(cids, counts))
            small = [c for c in cids if size_map[c] < min_size]
            large = [c for c in cids if size_map[c] >= min_size]
            if not small or not large:
                break

            # compute centroids for all cids once
            centroids = {}
            for c in cids:
                idx = np.flatnonzero(labels0 == c)
                centroids[c] = community_centroid_modes(X0[idx])

            merged_any = False
            for c in small:
                # find best large target
                sims = [(t, centroid_sim(centroids[c], centroids[t])) for t in large]
                if not sims:
                    continue
                t_best, s_best = max(sims, key=lambda x: x[1])
                if s_best >= merge_min_sim:
                    labels0[labels0 == c] = t_best
                    merged_any = True
            if not merged_any:
                break

        # reindex labels to 0..C-1
        uniq = np.unique(labels0)
        remap = {c: i for i, c in enumerate(uniq)}
        labels0 = np.array([remap[c] for c in labels0], dtype=int)

    # ---------- enforce min_size (final pass) ----------
    cids, counts = np.unique(labels0, return_counts=True)
    keep_c = [c for c, cnt in zip(cids, counts) if cnt >= min_size]
    keep_mask = np.isin(labels0, keep_c)
    if not np.any(keep_mask):
        empty_df = pd.DataFrame(columns=[id_col, "community"])
        return empty_df, {}, pd.DataFrame(columns=[id_col, "topk_ids", "topk_sims", "neighbor_modes"])

    labels_keep = labels0[keep_mask]
    idx_keep = np.flatnonzero(keep_mask)
    ids = [ids0[i] for i in np.flatnonzero(keep_mask)]
    X_keep = X0[idx_keep]
    # induced subgraph on kept nodes
    node_map = {int(keep_idx[i]): i for i in idx_keep}  # orig -> compact pos
    G = nx.Graph()
    G.add_nodes_from(range(len(idx_keep)))
    for u, v, d in G0.edges(data=True):
        if u in node_map and v in node_map:
            G.add_edge(node_map[u], node_map[v], **d)

    # ---------- neighbors (Top-K within community) ----------
    adj_w = defaultdict(list)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        adj_w[u].append((v, w))
        adj_w[v].append((u, w))

    topk_ids, topk_sims_list, neighbor_modes = [], [], []
    for pos in range(len(idx_keep)):
        c = labels_keep[pos]
        cand = [(nbr, w) for (nbr, w) in adj_w.get(pos, []) if labels_keep[nbr] == c and nbr != pos]
        if cand:
            cand.sort(key=lambda x: -x[1])
            cand = cand[:max(1, neighbor_k)]
            nbr_idx = [nbr for (nbr, _) in cand]
            sims_topk = [w for (_, w) in cand]
        else:
            nbr_idx, sims_topk = [], []

        # ids mapping back to df
        orig_ids = [ids[i] for i in nbr_idx]
        topk_ids.append(orig_ids)
        topk_sims_list.append([float(x) for x in sims_topk])

        if nbr_idx:
            neigh_answers = X_keep[nbr_idx]
            vmax = neigh_answers.max(axis=0)
            modes = []
            for j in range(neigh_answers.shape[1]):
                counts = np.bincount(neigh_answers[:, j], minlength=int(vmax[j]) + 1)
                modes.append(int(np.argmax(counts)))
            mode_dict = {qname: modes[qi] for qi, qname in enumerate(Qcols)}
        else:
            mode_dict = {qname: None for qname in Qcols}
        neighbor_modes.append(mode_dict)

    neighbors = pd.DataFrame({
        id_col: [ids[i] for i in range(len(idx_keep))],
        "topk_ids": topk_ids,
        "topk_sims": topk_sims_list,
        "neighbor_modes": neighbor_modes
    })

    # ---------- communities & stats ----------
    # reindex labels_keep to 0..C-1 for output cleanliness
    uniq = np.unique(labels_keep)
    remap = {c: i for i, c in enumerate(uniq)}
    labels_out = np.array([remap[c] for c in labels_keep], dtype=int)

    communities = pd.DataFrame({id_col: [ids[i] for i in range(len(idx_keep))],
                                "community": labels_out})

    stats = {}
    for c in np.unique(labels_out):
        members = np.flatnonzero(labels_out == c)
        size_c = int(len(members))
        if size_c <= 1:
            stats[int(c)] = {"size": size_c, "avg_internal_sim": None, "min_internal_sim": None}
            continue
        # edge weights within community (on G)
        ws = []
        mem_set = set(members)
        for u in members:
            for v, w in adj_w.get(u, []):
                if v in mem_set and v > u:
                    ws.append(float(w))
        if not ws:
            stats[int(c)] = {"size": size_c, "avg_internal_sim": None, "min_internal_sim": None}
        else:
            stats[int(c)] = {"size": size_c,
                             "avg_internal_sim": float(np.mean(ws)),
                             "min_internal_sim": float(np.min(ws))}
    return communities, stats, neighbors


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


import numpy as np
import pandas as pd
from collections import Counter

def cluster_communities_size_constrained(
    df: pd.DataFrame,
    id_col: str = "caseid",
    *,
    min_size: int = 60,
    max_size: int | None = 400,
    K: int | None = None,            # if None, inferred from n / target_size
    target_size: int | None = None,  # used when K=None; default=(min+max)/2
    max_iter: int = 10,
    missing_vals = (-1, np.nan),
    random_state: int = 42,
):
    """
    Hard-partition respondents into communities by answer similarity (Hamming/k-modes),
    enforcing per-cluster min_size and max_size.

    Returns
    -------
    communities : DataFrame [id_col, community]    # community is 0..K-1
    stats       : dict cid -> {'size', 'avg_intra_hamming', 'min_intra_hamming'}
    """
    rng = np.random.default_rng(random_state)

    # ---------- prep: encode to non-negative ints & fill NaN by column mode ----------
    ids = df[id_col].astype(str).to_numpy()
    Qcols = [c for c in df.columns if c != id_col]
    Xraw = df[Qcols].copy()

    # treat custom missings; (avoid passing NaN to replace)
    repl_vals = [v for v in missing_vals if not (isinstance(v, float) and np.isnan(v))]
    if repl_vals:
        Xraw = Xraw.replace(repl_vals, np.nan)

    X_list = []
    for c in Qcols:
        col = Xraw[c]
        if col.isna().all():
            X_list.append(np.zeros(len(col), dtype=int))
            continue
        # if non-numeric, factorize (categorical → codes)
        cnum = pd.to_numeric(col, errors="coerce")
        if col.dtype == object or cnum.isna().sum() and not col.isna().all():
            mask = col.notna()
            codes, _ = pd.factorize(col[mask], sort=True)
            tmp = np.full(len(col), np.nan)
            tmp[mask] = codes
            # fill NaN with mode code (or 0)
            vals = tmp[~np.isnan(tmp)]
            fillv = int(pd.Series(vals).mode().iloc[0]) if len(vals) else 0
            tmp = np.where(np.isnan(tmp), fillv, tmp).astype(int)
            X_list.append(tmp)
        else:
            vals = cnum
            if vals.isna().any():
                m = vals.mode(dropna=True)
                fillv = (m.iloc[0] if not m.empty else vals.dropna().iloc[0])
                vals = vals.fillna(fillv)
            X_list.append(vals.astype(int).to_numpy())
    X = np.column_stack(X_list).astype(int, copy=False)
    n, q = X.shape
    if n == 0 or q == 0:
        return pd.DataFrame(columns=[id_col, "community"]), {}

    # ---------- decide K from size bounds ----------
    lo = int(np.ceil(n / (max_size if max_size else n))) if min_size <= n else 1
    hi = int(np.floor(n / max(1, min_size)))
    if K is None:
        if target_size is None:
            if max_size:
                target_size = (min_size + max_size) // 2
            else:
                target_size = max(min_size, int(max(1, round(np.median([min_size, 100])))))
        K = int(np.clip(int(round(n / target_size)), max(1, lo), max(1, hi)))
    else:
        # sanity: adjust if K impossible
        K = int(np.clip(K, max(1, lo), max(1, hi)))

    # quotas
    cap_hi = max_size if max_size is not None else n
    cap_lo = min_size

    # ---------- init centroids (modes) via k-modes++ seeding ----------
    # pick first seed randomly
    seeds = [rng.integers(0, n)]
    def hamming_to_seed(x, S):
        # x: (n,q) ints; S: list of centroid vectors
        # return min Hamming distance to any centroid
        if not S:
            return np.ones(x.shape[0])
        D = np.stack([np.mean(x != c, axis=1) for c in S], axis=1)  # (n, |S|)
        return D.min(axis=1)

    for _ in range(1, K):
        d2 = hamming_to_seed(X, [X[i] for i in seeds])
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        seeds.append(int(rng.choice(n, p=probs)))
    centroids = np.stack([X[i] for i in seeds], axis=0)  # (K,q)

    # helpers
    def assign_with_caps(X, C, cap_hi, cap_lo):
        """
        Greedy size-constrained assignment.
        1) Fill clusters up to cap_hi by descending confidence (gap to 2nd-best).
        2) If any clusters < cap_lo, move in nearest points from oversized clusters.
        """
        K = C.shape[0]
        # cost: Hamming distance to each centroid
        costs = np.mean(X[:, None, :] != C[None, :, :], axis=2)  # (n,K)

        # best and second-best for confidence
        order = np.argsort(costs, axis=1)
        best = order[:, 0]
        second = order[:, 1] if K > 1 else np.full(n, 0)
        margin = costs[np.arange(n), second] - costs[np.arange(n), best]
        idx = np.argsort(-margin)  # assign confident points first

        sizes = np.zeros(K, dtype=int)
        assign = -np.ones(n, dtype=int)

        # pass 1: fill up to cap_hi
        for i in idx:
            prefs = order[i]  # sorted cluster preferences
            for k in prefs:
                if sizes[k] < cap_hi:
                    assign[i] = k
                    sizes[k] += 1
                    break
            if assign[i] == -1:
                # all full; park for later (we'll force into least-full)
                pass

        # any unassigned? put into clusters with remaining capacity (least cost)
        unassigned = np.flatnonzero(assign == -1)
        if unassigned.size:
            # build list of clusters with remaining capacity
            remain = np.where(sizes < cap_hi)[0]
            if remain.size == 0:
                # fallback: let them go to their best (will exceed cap_hi; fixed later)
                assign[unassigned] = best[unassigned]
            else:
                # greedily place into the available cluster that minimizes cost
                for i in unassigned:
                    # best among remain
                    k_best = remain[np.argmin(costs[i, remain])]
                    assign[i] = k_best
                    sizes[k_best] += 1
                    # update remain
                    remain = np.where(sizes < cap_hi)[0]

        # pass 2: lift clusters to cap_lo by borrowing from oversized ones
        need = [(k, cap_lo - sizes[k]) for k in range(K) if sizes[k] < cap_lo]
        have = [(k, sizes[k] - cap_lo) for k in range(K) if sizes[k] > cap_lo]
        # also allow borrowing from clusters > cap_lo (prefer those farthest above)
        have = sorted([(k, sizes[k] - cap_lo) for k in range(K) if sizes[k] > cap_lo],
                      key=lambda x: -x[1])

        if need and have:
            # precompute per-point current cluster and cost deltas
            for k_need, deficit in need:
                if deficit <= 0:
                    continue
                # candidates to steal: points currently not in k_need, ideally from oversized
                cand_idx = np.flatnonzero(assign != k_need)
                # compute delta cost if moved to k_need
                delta = costs[cand_idx, k_need] - costs[cand_idx, assign[cand_idx]]
                # prefer stealing from clusters well above cap_lo
                from_over = np.isin(assign[cand_idx], [k for k, _ in have])
                # rank: (not from_over → large), then by delta
                rank = np.lexsort((delta, ~from_over))
                moved = 0
                for j in cand_idx[rank]:
                    k_from = assign[j]
                    # check donor still safe to give
                    if sizes[k_from] <= cap_lo:
                        continue
                    assign[j] = k_need
                    sizes[k_from] -= 1
                    sizes[k_need] += 1
                    moved += 1
                    if moved >= deficit:
                        break

        return assign, sizes

    def update_centroids(X, assign, K):
        C = np.zeros((K, q), dtype=int)
        for k in range(K):
            idx = np.flatnonzero(assign == k)
            if idx.size == 0:
                # empty: re-seed randomly
                C[k] = X[rng.integers(0, n)]
            else:
                # column-wise mode via bincount
                block = X[idx]
                vmax = block.max(axis=0)
                modes = []
                for j in range(q):
                    counts = np.bincount(block[:, j], minlength=int(vmax[j]) + 1)
                    modes.append(int(np.argmax(counts)))
                C[k] = np.array(modes, dtype=int)
        return C

    # ---------- main loop ----------
    assign, sizes = assign_with_caps(X, centroids, cap_hi, cap_lo)
    for it in range(max_iter):
        centroids_new = update_centroids(X, assign, K)
        assign_new, sizes_new = assign_with_caps(X, centroids_new, cap_hi, cap_lo)

        if np.array_equal(assign_new, assign):
            break
        centroids, assign, sizes = centroids_new, assign_new, sizes_new

    # ---------- outputs ----------
    communities = pd.DataFrame({
        id_col: ids,
        "community": assign.astype(int)
    })

    # stats
    stats = {}
    for k in range(K):
        idx = np.flatnonzero(assign == k)
        size_k = int(idx.size)
        if size_k <= 1:
            stats[k] = {"size": size_k, "avg_intra_hamming": None, "min_intra_hamming": None}
            continue
        block = X[idx]
        # estimate intra distances: to centroid and pairwise min
        cvec = centroids[k]
        d_to_c = np.mean(block != cvec, axis=1)
        # avg intra ~ average to centroid (fast proxy)
        avg_intra = float(d_to_c.mean())
        # min intra: min pairwise distance via small sample if large
        if size_k <= 800:
            # exact (O(m^2 q)) but OK for moderate m
            B = block
            # compute pairwise hamming via broadcasting (memory-friendly chunked)
            mh = []
            step = 256
            for a in range(0, B.shape[0], step):
                Aa = B[a:a+step]
                # (step, m, q) compare then mean over q and take min across pairs
                d = np.mean(Aa[:, None, :] != B[None, :, :], axis=2)  # (s, m)
                # ignore self-dist by setting diag big
                for ii in range(Aa.shape[0]):
                    d[ii, a+ii] = 1.0
                mh.append(d.min(axis=1))
            min_intra = float(np.min(np.concatenate(mh)))
        else:
            # sample-based estimate
            samp = rng.choice(size_k, size=min(800, size_k), replace=False)
            B = block[samp]
            d = np.mean(B[:, None, :] != B[None, :, :], axis=2)
            np.fill_diagonal(d, 1.0)
            min_intra = float(np.min(d))
        stats[k] = {"size": size_k, "avg_intra_hamming": avg_intra, "min_intra_hamming": min_intra}

    return communities, stats

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def community_similarity_report(
    df: pd.DataFrame,
    communities: pd.DataFrame,
    *,
    id_col: str = "caseid",
    max_exact_pairs: int = 800,
    random_state: int = 42,
):
    """
    Compute (a) within-community similarity summaries and (b) between-community centroid similarities.
    Similarity = 1 - Hamming over column-wise categorical codes.

    Returns
    -------
    intra_summary : DataFrame [community, size, avg_to_centroid_sim, mean_pair_sim, min_pair_sim]
    centroid_sim  : DataFrame (K x K) similarity between community centroids
    """
    rng = np.random.default_rng(random_state)

    # ---- align ids ----
    df2 = df.copy()
    df2[id_col] = df2[id_col].astype(str)
    C = communities.copy()
    C[id_col] = C[id_col].astype(str)

    Qcols = [c for c in df2.columns if c != id_col]
    X_df = df2.set_index(id_col)[Qcols]

    # ---- encode each column to non-negative integer codes ----
    # Categorical coding preserves equality (Hamming cares only equal/unequal)
    X_cols = []
    for c in Qcols:
        cats = pd.Categorical(X_df[c], ordered=False)
        codes = cats.codes.astype(np.int64, copy=False)     # -1 for NaN
        if (codes == -1).any():
            # fill NaN codes with mode among non -1; fallback to 0
            nonneg = codes[codes != -1]
            fill = int(pd.Series(nonneg).mode().iloc[0]) if nonneg.size else 0
            codes = np.where(codes == -1, fill, codes)
        # safety: ensure non-negative
        if codes.min() < 0:
            # shift up just in case (shouldn't happen after fill)
            codes = codes - codes.min()
        X_cols.append(codes)
    X_enc = np.column_stack(X_cols)  # (n, q), dtype int64, all >= 0

    # keep only ids present in communities
    ids_all = X_df.index.to_numpy()
    mask_keep = np.isin(ids_all, C[id_col].to_numpy())
    ids_keep = ids_all[mask_keep]
    X_keep = X_enc[mask_keep]

    # map ids -> row positions
    pos = pd.Series(np.arange(len(ids_keep)), index=ids_keep)

    # ---- per-community member indices ----
    C_sorted = C[[id_col, "community"]].dropna().copy()
    C_sorted = C_sorted[C_sorted[id_col].isin(ids_keep)]
    C_sorted = C_sorted.sort_values("community").reset_index(drop=True)

    comm_ids = C_sorted["community"].to_numpy()
    uniq_comms = np.unique(comm_ids)
    member_pos = {
        c: pos[C_sorted.loc[C_sorted["community"] == c, id_col].to_numpy()].to_numpy()
        for c in uniq_comms
    }

    q = X_keep.shape[1]

    def mode_vector(block: np.ndarray) -> np.ndarray:
        """Column-wise mode using bincount (block: m x q, int, non-negative)."""
        if block.size == 0:
            return np.zeros(q, dtype=int)
        vmax = block.max(axis=0)
        modes = np.empty(q, dtype=int)
        for j in range(q):
            col = block[:, j]
            # safety: ensure non-negative before bincount
            if col.min() < 0:
                col = col - col.min()
            counts = np.bincount(col, minlength=int(vmax[j]) + 1)
            modes[j] = int(np.argmax(counts))
        return modes

    # ---- centroids and within-community stats ----
    centroids = {}
    sizes = {}
    rows = []

    for c in uniq_comms:
        idx = member_pos[c]
        sizes[c] = int(idx.size)
        block = X_keep[idx]
        cen = mode_vector(block)
        centroids[c] = cen

        size = sizes[c]
        if size == 0:
            rows.append(dict(community=int(c), size=0,
                             avg_to_centroid_sim=np.nan,
                             mean_pair_sim=np.nan,
                             min_pair_sim=np.nan))
            continue

        # avg similarity to centroid
        d_to_c = np.mean(block != cen[None, :], axis=1)  # Hamming
        avg_to_centroid_sim = float(1.0 - d_to_c.mean())

        # pairwise similarities: exact if small, sample if large
        if size <= max_exact_pairs:
            step = 256
            sims_sum = 0.0
            sims_min = 1.0
            count_pairs = 0
            for a in range(0, size, step):
                A = block[a:a+step]
                dist = np.mean(A[:, None, :] != block[None, :, :], axis=2)  # (sa, size)
                # mask self
                for i in range(A.shape[0]):
                    dist[i, a + i] = np.nan
                sims = 1.0 - dist[~np.isnan(dist)]
                if sims.size:
                    sims_sum += sims.sum()
                    sims_min = min(sims_min, float(sims.min()))
                    count_pairs += sims.size
            mean_pair_sim = float(sims_sum / count_pairs) if count_pairs else np.nan
            min_pair_sim = float(sims_min) if count_pairs else np.nan
        else:
            m = min(max_exact_pairs, size)
            samp = np.sort(np.unique(rng.choice(size, size=m, replace=False)))
            B = block[samp]
            dist = np.mean(B[:, None, :] != B[None, :, :], axis=2)
            np.fill_diagonal(dist, np.nan)
            sims = 1.0 - dist[~np.isnan(dist)]
            mean_pair_sim = float(np.nanmean(sims)) if sims.size else np.nan
            min_pair_sim = float(np.nanmin(sims)) if sims.size else np.nan

        rows.append(dict(community=int(c), size=size,
                         avg_to_centroid_sim=avg_to_centroid_sim,
                         mean_pair_sim=mean_pair_sim,
                         min_pair_sim=min_pair_sim))

    intra_summary = pd.DataFrame(rows).sort_values(["size", "community"], ascending=[False, True])

    # ---- centroid similarity (between communities) ----
    K = len(uniq_comms)
    cent_mat = np.zeros((K, q), dtype=int)
    for i, c in enumerate(uniq_comms):
        cent_mat[i] = centroids[c]
    D = np.mean(cent_mat[:, None, :] != cent_mat[None, :, :], axis=2)
    centroid_sim = 1.0 - D
    centroid_sim_df = pd.DataFrame(centroid_sim, index=uniq_comms, columns=uniq_comms)

    return intra_summary, centroid_sim_df

import json
import pandas as pd
from collections import defaultdict
from typing import Optional
import json
from collections import defaultdict
from typing import Optional, List, Dict, Tuple


import json
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


def save_cluster_jsonl_with_ordered_neighbors_wide(
    communities: pd.DataFrame,
    survey_data_filtered: pd.DataFrame,
    path: str,
    *,
    id_col: str = "caseid",
    comm_col: str = "community",
    include_self: bool = False,           # usually False
    neighbor_limit: Optional[int] = None, # cap to first K if desired
    min_overlap: int = 1                  # min shared answered questions to compute similarity
) -> None:
    """
    Writes JSONL lines like:
      {"caseid": "<str>", "community": <int>, "neighbors": ["idA","idB", ...]}

    Similarity = (# equal answers on commonly answered questions) / (# commonly answered questions).
    If overlap < min_overlap, similarity = 0.0.
    Ties are broken by lexicographic order of neighbor caseid to keep output deterministic.
    """

    # --- 1) sanitize/align inputs
    comm_df = communities[[id_col, comm_col]].copy()
    comm_df[id_col] = comm_df[id_col].astype(str)

    if id_col not in survey_data_filtered.columns:
        raise ValueError(f"Expected '{id_col}' in survey_data_filtered.")

    # set index to caseid (string), keep only answer columns after id
    ans_df = survey_data_filtered.copy()
    ans_df[id_col] = ans_df[id_col].astype(str)
    ans_df = ans_df.set_index(id_col)
    # ensure float with NaNs preserved (answers like 1.0/2.0); don't coerce to str
    ans_df = ans_df.apply(pd.to_numeric, errors="coerce")

    # roster per community (keep NaN communities if present)
    comm_to_ids = defaultdict(list)
    for cid, grp in comm_df.groupby(comm_col, dropna=False):
        comm_to_ids[cid] = grp[id_col].tolist()

    have_ans_ids = set(ans_df.index)

    # --- 2) helper to compute similarity vector from one row to a matrix (same columns)
    def sim_to_matrix(row_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """
        row_vec: shape (Q,)
        mat:     shape (N, Q)
        returns: shape (N,) similarity per row in mat
        """
        # masks for non-NaN overlap
        row_notnan = ~np.isnan(row_vec)           # (Q,)
        mat_notnan = ~np.isnan(mat)               # (N, Q)
        overlap_mask = mat_notnan & row_notnan    # (N, Q)
        overlap_cnt = overlap_mask.sum(axis=1)    # (N,)

        # equals on overlap
        equals_mask = (mat == row_vec) & overlap_mask
        eq_cnt = equals_mask.sum(axis=1)          # (N,)

        # avoid divide-by-zero; min_overlap gating
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(overlap_cnt >= min_overlap, eq_cnt / overlap_cnt, 0.0)

        # replace possible NaNs (shouldn't occur due to gating) with 0
        sim = np.nan_to_num(sim, nan=0.0)
        return sim.astype(float)

    # --- 3) write JSONL
    with open(path, "w", encoding="utf-8") as f:
        for _, row in comm_df.iterrows():
            cur_id = row[id_col]
            cid = row[comm_col]

            roster = comm_to_ids.get(cid, [])
            # include/exclude self
            candidates = roster if include_self else [rid for rid in roster if rid != cur_id]

            # filter to those with answers
            cand_with_ans: List[str] = [rid for rid in candidates if rid in have_ans_ids]

            # if current respondent has no answers, everyone gets sim=0; just deterministic sort
            if cur_id not in have_ans_ids or len(cand_with_ans) == 0:
                neigh_sorted = sorted(candidates)  # tie-break by id
            else:
                # build matrix for candidates (N x Q) and vector for current (Q,)
                cur_vec = ans_df.loc[cur_id].to_numpy(dtype=float)
                cand_mat = ans_df.loc[cand_with_ans].to_numpy(dtype=float)

                sims = sim_to_matrix(cur_vec, cand_mat)  # (N,)

                # pack (rid, sim) including candidates without answers (sim=0)
                sim_pairs: List[Tuple[str, float]] = list(zip(cand_with_ans, sims))
                # add any candidate missing from ans_df with sim=0
                missing = [(rid, 0.0) for rid in candidates if rid not in have_ans_ids]
                sim_pairs.extend(missing)

                # sort by similarity desc, then id asc for stability
                sim_pairs.sort(key=lambda x: (-x[1], x[0]))
                neigh_sorted = [rid for rid, _ in sim_pairs]

            if neighbor_limit is not None and neighbor_limit > 0:
                neigh_sorted = neigh_sorted[:neighbor_limit]

            rec = {
                "caseid": cur_id,
                "community": (int(cid) if pd.notna(cid) else None),
                "neighbors": neigh_sorted,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_centroid_sim(
    centroid_sim: pd.DataFrame,
    *,
    order: list | None = None,      # e.g., a list of community ids to reorder rows/cols
    topk: int | None = None,        # show only the top-k largest communities by id (or by order)
    title: str = "Centroid Similarity (1 − Hamming)",
    save_path: str | None = None,   # e.g., "centroid_sim.png"
    figsize=(8, 6),
    annotate: bool = False,         # write numbers on cells (can get busy)
    clip01: bool = True             # clamp values to [0,1] for safety
):
    """
    Draw a heatmap of centroid_sim (DataFrame K×K).
    """
    if not isinstance(centroid_sim, pd.DataFrame):
        raise TypeError("centroid_sim must be a pandas DataFrame (K×K).")

    # Optional: subset to a particular order or top-k
    cm = centroid_sim.copy()
    if order is not None:
        missing = [c for c in order if c not in cm.index]
        if missing:
            raise ValueError(f"IDs in 'order' not found in centroid_sim: {missing[:10]} ...")
        cm = cm.loc[order, order]
    elif topk is not None:
        # simple subset by the first topk communities (sorted by index)
        ids = list(cm.index)[:int(topk)]
        cm = cm.loc[ids, ids]

    M = cm.values.astype(float)
    if clip01:
        M = np.clip(M, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns.astype(str), rotation=90)
    ax.set_yticklabels(cm.index.astype(str))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate and M.size <= 80*80:  # avoid clutter on huge matrices
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def centroid_sim_top_pairs(centroid_sim: pd.DataFrame, k: int = 15):
    """
    Return the top-k most similar *distinct* community pairs and the bottom-k (most dissimilar).
    """
    cm = centroid_sim.copy()
    cm.values[np.diag_indices_from(cm.values)] = np.nan  # ignore self
    pairs = []
    idx = cm.index
    arr = cm.values
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            val = arr[i, j]
            if not np.isnan(val):
                pairs.append((idx[i], idx[j], float(val)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:k]
    bottom = pairs[-k:][::-1]
    return pd.DataFrame(top, columns=["c1","c2","sim"]), pd.DataFrame(bottom, columns=["c1","c2","sim"])

if __name__ == "__main__":

    for year in ['20', '22', '24']:

        file_path = f'/home/ruomeng/gae/dataset/ces/raw/{year}/question_{year}.csv'
        survey_data = pd.read_csv(file_path)
        
        jsonl_file = f"/home/ruomeng/gae/dataset/ces_pros/question_codebook.jsonl"  
        codebook = load_jsonl_as_dict_of_dict(jsonl_file, key='id')
        question_set = []
        target = {"1": "Support", "2": "Oppose"}

        for caseid in codebook.keys():
            if codebook[caseid]["options"] == target:
                question_set.append(caseid)

        temp_qs_all, graph_qs = split_questions(question_set, train_ratio=0.667, seed=42)
        train_qs, holdout_qs = split_questions(temp_qs_all, train_ratio=0.75, seed=42)
        print('len(question_set)', len(question_set), len(graph_qs), len(train_qs), len(holdout_qs))
 
        # Assuming you already have train_qs and holdout_qs
        graph_df = pd.DataFrame({"graph_qs": graph_qs})
        train_df = pd.DataFrame({"train_qs": train_qs})
        holdout_df = pd.DataFrame({"holdout_qs": holdout_qs})

        # Save to CSV
        graph_df.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/graph_qs.csv", index=False)
        train_df.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/train_qs.csv", index=False)
        holdout_df.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/holdout_qs.csv", index=False)

        print("Saved graph_qs, train_qs.csv and holdout_qs.csv")

        survey_data_filtered = survey_data[["caseid"] + graph_qs]


        communities, stats = cluster_communities_size_constrained(
            survey_data_filtered, id_col="caseid",
            min_size=10, max_size=200,   # your bounds
            K=None,                      # infer K from n and target_size (optional)
            max_iter=10,
            random_state=42
        )
 
        save_cluster_jsonl_with_ordered_neighbors_wide(
            communities,
            survey_data_filtered,
            f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/neighbors_{year}.jsonl",
            id_col="caseid",
            comm_col="community",
            include_self=False,      # typical
            neighbor_limit=None      # or set an int cap if needed
        )

        intra_summary, centroid_sim = community_similarity_report(survey_data_filtered, communities, id_col="caseid")
        intra_summary.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/intra_summary_graph.csv", index=False)
        draw_centroid_sim(centroid_sim, topk=30, title="Centroid Sim (Top 30)", save_path = f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/centroid_sim_graph.png")


        survey_data_train = survey_data[["caseid"] + train_qs]
        intra_summary, centroid_sim = community_similarity_report(survey_data_train, communities, id_col="caseid")
        intra_summary.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/intra_summary_train.csv", index=False)
        draw_centroid_sim(centroid_sim, topk=30, title="Centroid Sim (Top 30)", save_path = f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/centroid_sim_train.png")

        survey_data_holdout = survey_data[["caseid"] + holdout_qs]
        intra_summary, centroid_sim = community_similarity_report(survey_data_holdout, communities, id_col="caseid")
        intra_summary.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/intra_summary_holdout.csv", index=False)
        draw_centroid_sim(centroid_sim, topk=30, title="Centroid Sim (Top 30)", save_path = f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/centroid_sim_holdout.png")


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
            survey_data_filtered_seen.to_csv(f"/home/ruomeng/gae/dataset/ces_pros/raw/{year}/questions_{split}_{year}.csv", index=False)

    



            

