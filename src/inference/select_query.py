from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Union, Callable
import numpy as np
import torch
from collections import Counter
from inference.sampling import sampling
import copy
from tqdm import tqdm

NodeId = Hashable
QueryId = Hashable
Label = Union[int, float]

def entropy_from_cov(Sigma, ridge_eps=1e-8):
    Sigma = Sigma + ridge_eps * np.eye(Sigma.shape[0])
    sign, logdet = np.linalg.slogdet(Sigma)
    assert sign > 0
    V = Sigma.shape[0]
    return 0.5 * (V * (1 + np.log(2*np.pi)) + logdet)


def majority_answer(samples):
    """
    Args:
        samples: list of dict[nodeid -> answer]
            N samples from Gibbs for one candidate query

    Returns:
        maj: dict[nodeid -> answer]
            majority-voted answer for each respondent
    """
    if not samples:
        return {}

    nodeids = samples[0].keys()
    maj = {}
    for v in nodeids:
        votes = [s[v] for s in samples]
        maj[v] = Counter(votes).most_common(1)[0][0]
    return maj

def _formulate_matrix(
    samples: List[Optional[Dict[NodeId, Label]]],
    nodes: List[NodeId],
    *,
    map_label: Optional[Union[Dict[Label, float], Callable[[Label], float]]] = None,
    fill: float = np.nan,
    dtype=np.float32,
    ignore_unknown_nodes: bool = True,
) -> np.ndarray:
    """
    Build an S × V matrix (no one-hot). Each row s is a full graph assignment for one sample.
    Vectorized fill for efficiency.
    """
    S, V = len(samples), len(nodes)
    Z = np.full((S, V), fill, dtype=dtype)
    node_index = {v: i for i, v in enumerate(nodes)}

    # Prepare mapper
    if map_label is None:
        def _map(x): return float(x)
    elif callable(map_label):
        _map = map_label
    else:
        mapping = {k: float(v) for k, v in map_label.items()}
        def _map(x, _m=mapping):
            return _m[x]

    # Collect all row indices, col indices, values
    row_idx, col_idx, vals = [], [], []
    for s, y in enumerate(samples):
        if not y:
            continue
        for v, lab in y.items():
            i = node_index.get(v)
            if i is None:
                if ignore_unknown_nodes:
                    continue
                raise KeyError(f"Unknown node {v!r} not found in `nodes`.")
            if lab is None:
                continue
            try:
                val = _map(lab)
            except Exception as e:
                raise ValueError(f"Could not map label {lab!r} for node {v!r}") from e
            row_idx.append(s)
            col_idx.append(i)
            vals.append(val)

    if row_idx:
        row_idx = np.fromiter(row_idx, dtype=np.intp)
        col_idx = np.fromiter(col_idx, dtype=np.intp)
        vals = np.fromiter(vals, dtype=dtype)
        Z[row_idx, col_idx] = vals

    return Z


def _covariance(Z: np.ndarray):
    """
    Z: (S, V) with entries in {+1, -1}, no NaNs.
    Returns Σ (V, V) unbiased sample covariance.
    """
    Z = np.asarray(Z, dtype=np.float64)
    S = Z.shape[0]
    mu = Z.mean(axis=0)                            # (V,)
    Sigma = (Z.T @ Z - S * np.outer(mu, mu)) / (S - 1)
    return Sigma, mu



def _neighbors_answers(v, y, neighbors):
    ans: Dict[NodeId, int] = {}
    for u in neighbors:
        ans[u] = y[u]
    return ans

def probs_binary_to_ans_dict(
    probs,   # [[p0_A, p0_B], [p1_A, p1_B], ...]
    nodes,      # same length as probs
    neighbor,
    labels=["A", "B"],      # optional: e.g., ("A","B")
) -> Dict[Hashable, Hashable]:

    ans: Dict[Hashable, Hashable] = {}
    for nid, row in zip(nodes, probs):
        p = np.asarray(row, dtype=np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback uniform if bad row
            p = np.array([0.5, 0.5], dtype=np.float64)
        else:
            p = p / s
        ans[nid] = labels[int(np.argmax(p))]

    neigh_ans_list = dict()
    for idx, v in enumerate(nodes):
            neigh_ans = _neighbors_answers(v, ans, neighbor[v])
            neigh_ans_list[v] = neigh_ans
    return neigh_ans_list


from typing import Sequence, List, Any

def select_probs_for_nodes_list(
    probs_batch: List[Any],          # e.g., [p_i] or [[p_i1, p_i2, ...]]
    nodes: Sequence,                 # full node list aligned with probs_batch
    node_selected: Sequence,         # nodes to pick (order preserved)
) -> List[Any]:
    idx = {n: i for i, n in enumerate(nodes)}
    out: List[Any] = []
    for n in node_selected:
        i = idx.get(n)
        out.append(probs_batch[i])
    return out

def select_queries_group(dataset, model, iid_model, nodes, Xavail, Y_init, observed, probs_batch_dict, mode='', k=1, 
                N: int = 100, ridge_eps: float = 1e-5, rng: Optional[np.random.Generator] = None, verbose: bool = False):
    
    query_cand_list = list(Xavail)

    EIG = dict()
    for query in Xavail:
        EIG[query] = 0

    def entropy_np(P, eps=1e-12):
        P = np.asarray(P, dtype=np.float64)
        P = np.clip(P, eps, 1.0)
        return -(P * np.log(P)).sum(axis=1)  # [N]


    for qid in dataset.X_heldout:
        q_text = dataset.codebook[qid]["question"]
        print(f"\nEvaluation on: {qid}")
        # --- run inference ---
        probs_batch = probs_batch_dict[qid]
        probs_batch_select = select_probs_for_nodes_list(probs_batch, nodes=dataset.graph.nodes, node_selected=nodes)
        entropy_without_designs = entropy_np(probs_batch_select)      # [N]
        H_pre = np.array(entropy_without_designs, dtype=np.float64)

        for query in tqdm(query_cand_list, desc="Evaluating queries"):
            q_text_cand = dataset.codebook[query]["question"]

            if Y_init is None:
                probs_batch_iid = iid_model.predict_batch(nodes=dataset.graph.nodes, query=q_text_cand, asked_queries=dataset.asked_queries, 
                                              neighbors=dataset.graph.neighbor, observed=observed, estimated=None, mode='iid')
            else:
                probs_batch_iid = Y_init[query]
                
            estimated = probs_binary_to_ans_dict(probs_batch_iid, dataset.graph.nodes, neighbor=dataset.graph.neighbor, labels=["A", "B"])  
            probs_batch_w = model.predict_batch(nodes=nodes, query=q_text_cand, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed, estimated=estimated, mode=mode)
            probs_batch_w = np.array(probs_batch_w)  # [num_nodes, 2]
            pA = np.clip(probs_batch_w[:, 0], 1e-12, 1.0)      # align column 0 with 'A'
            pB = np.clip(probs_batch_w[:, 1], 1e-12, 1.0)

            # A
            observed_temp_A = copy.deepcopy(observed)
            for nodeid in nodes:
                observed_temp_A[query][nodeid] = 'A'

            probs_batch_iid = iid_model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed_temp_A, estimated=None, mode='iid')
            estimated = probs_binary_to_ans_dict(probs_batch_iid, dataset.graph.nodes, neighbor=dataset.graph.neighbor, labels=["A", "B"])  
            probs_batch_A = model.predict_batch(nodes=nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed_temp_A, estimated=estimated, mode=mode)
            probs_batch_A = np.array(probs_batch_A, dtype=np.float64)  # [N, C]
            entropy_with_designs_A = entropy_np(probs_batch_A)      # [N]
            
            # B
            observed_temp_B = copy.deepcopy(observed)
            for nodeid in nodes:
                observed_temp_B[query][nodeid] = 'B'
            probs_batch_iid = iid_model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed_temp_B, estimated=None, mode=mode)
            estimated = probs_binary_to_ans_dict(probs_batch_iid, dataset.graph.nodes, neighbor=dataset.graph.neighbor, labels=["A", "B"])  
            probs_batch_B = model.predict_batch(nodes=nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed_temp_B, estimated=estimated, mode=mode)
            probs_batch_B = np.array(probs_batch_B, dtype=np.float64)  # [N, C]
            entropy_with_designs_B = entropy_np(probs_batch_B)      # [N]

            Ey_H_post = pA * entropy_with_designs_A + pB * entropy_with_designs_B

            eig_mean = (H_pre - Ey_H_post).mean()
            EIG[query] += float(eig_mean)

        q_star = max(EIG, key=EIG.get)
        eig_max = EIG[q_star]   # 最大 EIG 值
        print('q_star', q_star, 'eig_max', eig_max)
     
    return [q_star]




def select_queries_iid(dataset, model, nodes, Xavail, observed, probs_batch_dict, mode='', k=1, 
                N: int = 100, ridge_eps: float = 1e-5, rng: Optional[np.random.Generator] = None, verbose: bool = False):
    
    query_cand_list = list(Xavail)

    EIG = dict()
    for query in Xavail:
        EIG[query] = 0

    def entropy_np(P, eps=1e-12):
        P = np.asarray(P, dtype=np.float64)
        P = np.clip(P, eps, 1.0)
        return -(P * np.log(P)).sum(axis=1)  # [N]


    for qid in dataset.X_heldout:
        q_text = dataset.codebook[qid]["question"]
        print(f"\nEvaluation on: {qid}")
        # --- run inference ---
        probs_batch = probs_batch_dict[qid]
        probs_batch_select = select_probs_for_nodes_list(probs_batch, nodes=dataset.graph.nodes, node_selected=nodes)
        entropy_without_designs = entropy_np(probs_batch_select)      # [N]
        H_pre = np.array(entropy_without_designs, dtype=np.float64)

        for query in tqdm(query_cand_list, desc="Evaluating queries"):
            q_text_cand = dataset.codebook[query]["question"]
            probs_batch_w = model.predict_batch(nodes=nodes, query=q_text_cand, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed, estimated=None, mode=mode)
            probs_batch_w = np.array(probs_batch_w)  # [num_nodes, 2]
            pA = np.clip(probs_batch_w[:, 0], 1e-12, 1.0)      # align column 0 with 'A'
            pB = np.clip(probs_batch_w[:, 1], 1e-12, 1.0)

            observed_temp_A = copy.deepcopy(observed)
            for nodeid in nodes:
                observed_temp_A[query][nodeid] = 'A'
            probs_batch_A = model.predict_batch(nodes=nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed_temp_A, estimated=None, mode=mode)
            probs_batch_A = np.array(probs_batch_A, dtype=np.float64)  # [N, C]
            entropy_with_designs_A = entropy_np(probs_batch_A)      # [N]
            
            observed_temp_B = copy.deepcopy(observed)
            for nodeid in nodes:
                observed_temp_B[query][nodeid] = 'B'
            probs_batch_B = model.predict_batch(nodes=nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed_temp_B, estimated=None, mode=mode)
            probs_batch_B = np.array(probs_batch_B, dtype=np.float64)  # [N, C]
            entropy_with_designs_B = entropy_np(probs_batch_B)      # [N]

            Ey_H_post = pA * entropy_with_designs_A + pB * entropy_with_designs_B

            eig_mean = (H_pre - Ey_H_post).mean()
            EIG[query] += float(eig_mean)

        q_star = max(EIG, key=EIG.get)
        eig_max = EIG[q_star]   # 最大 EIG 值
        print('q_star', q_star, 'eig_max', eig_max)
     
    return [q_star]

