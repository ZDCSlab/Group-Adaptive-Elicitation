from typing import Dict, List, Tuple, Any
import json
import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """
    Convert ["a.b=1", "data.demo_cols=[x,y]"] into a nested dictionary.
    Values are eval'd with yaml for convenience (so booleans/nums/lists work).
    """
    out: Dict[str, Any] = {}
    for kv in pairs or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            v_parsed = yaml.safe_load(v)
        except Exception:
            v_parsed = v
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v_parsed
    return out

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_graph_from_raw(
    demo_csv,
    question_csv,
    codebook_path: str,
    demo_cols: List[str],
    question_cols: List[str],
) -> Tuple[HeteroData, Dict[int,int], Dict[str,int], Dict[str, List[int]], List[Tuple[int,str,int]]]:
    """
    Returns:
      data: HeteroData with reverse edges
      uid2idx: {caseid -> user_idx}
      qid2idx: {"<qid>_<optcode>" -> global question-node idx}
      qid2choices: {qid -> [global idx of options in fixed order]}
      edges_u_to_qopt: [(user_idx, qid, qnode_idx)] from U->Q edges
    """
    # --- load ---
    df_demo = demo_csv
    df_question = question_csv
    df_demo["caseid"] = df_demo["caseid"].astype(str)
    df_question["caseid"] = df_question["caseid"].astype(str)

    codebook = load_jsonl_as_dict_of_dict(codebook_path, key="id")

    # --- users ---
    user_nodes = df_demo["caseid"].tolist()
    uid2idx = {u:i for i,u in enumerate(user_nodes)}

    # --- subgroup nodes (demo options) ---
    demo_nodes = []
    for qid in demo_cols:
        valid_val = codebook[qid]["options"].keys()
        for v in valid_val:
            demo_nodes.append(f"{qid}_{int(v)}" if str(v).isdigit() else f"{qid}_{v}")
    sid2idx = {name:i for i,name in enumerate(demo_nodes)}

    # --- question nodes (choice options) ---
    question_nodes = []
    for qid in question_cols:
        valid_val = codebook[qid]["options"].keys()
        for v in valid_val:
            question_nodes.append(f"{qid}_{int(v)}" if str(v).isdigit() else f"{qid}_{v}")
    qid2idx = {name:i for i,name in enumerate(question_nodes)}

    # --- valid sets from codebook ---
    valid_demo = {qid: set(map(str, codebook[qid]["options"].keys())) for qid in demo_cols if qid in codebook}
    valid_q    = {qid: set(map(str, codebook[qid]["options"].keys())) for qid in question_cols if qid in codebook}

    # align users present in both
    users_in_both = set(df_demo["caseid"]).intersection(df_question["caseid"]).intersection(uid2idx.keys())
    df_demo_aln = df_demo[df_demo["caseid"].isin(users_in_both)].copy()
    df_q_aln    = df_question[df_question["caseid"].isin(users_in_both)].copy()

    # helpers
    def _key(qid, val):
        return f"{qid}_{int(val)}" if str(val).isdigit() else f"{qid}_{str(val)}"

    # --- build edges ---
    su_edges = set()  # (sid_idx, uid_idx)
    uq_edges = set()  # (uid_idx, qnode_idx)

    for _, row in df_demo_aln.iterrows():
        u = row["caseid"]
        if u not in uid2idx: continue
        uidx = uid2idx[u]
        for qid in demo_cols:
            if qid not in row or pd.isna(row[qid]): continue
            v = str(row[qid])
            if qid in valid_demo and v not in valid_demo[qid]: continue
            s_name = _key(qid, v)
            if s_name not in sid2idx: continue
            su_edges.add((sid2idx[s_name], uidx))

    for _, row in df_q_aln.iterrows():
        u = row["caseid"]
        if u not in uid2idx: continue
        uidx = uid2idx[u]
        for qid in question_cols:
            if qid not in row or pd.isna(row[qid]): continue
            v = str(row[qid])
            if qid in valid_q and v not in valid_q[qid]: continue
            q_name = _key(qid, v)
            if q_name not in qid2idx: continue
            uq_edges.add((uidx, qid2idx[q_name]))

    su_edge_index = np.array(list(su_edges), dtype=np.int64).T if su_edges else np.empty((2,0), dtype=np.int64)
    uq_edge_index = np.array(list(uq_edges), dtype=np.int64).T if uq_edges else np.empty((2,0), dtype=np.int64)

    # --- build HeteroData with reverse edges ---
    data = HeteroData()
    data["user"].num_nodes = len(uid2idx)
    data["subgroup"].num_nodes = len(sid2idx)
    data["question"].num_nodes = len(qid2idx)
    data[("subgroup","to","user")].edge_index = torch.as_tensor(su_edge_index, dtype=torch.long)
    data[("user","to","question")].edge_index = torch.as_tensor(uq_edge_index, dtype=torch.long)
    data[("user","rev_to","subgroup")].edge_index = data[("subgroup","to","user")].edge_index.flip(0)
    data[("question","rev_to","user")].edge_index = data[("user","to","question")].edge_index.flip(0)

    # --- qid -> ordered list of option node ids (softmax denominator order) ---
    qid2choices: Dict[str, List[int]] = {}
    for qid in question_cols:
        codes = list(codebook[qid]["options"].keys())
        try: codes = sorted(codes, key=lambda s: int(s))
        except Exception: codes = sorted(codes)
        ids = []
        for oc in codes:
            key = f"{qid}_{int(oc)}" if str(oc).isdigit() else f"{qid}_{oc}"
            ids.append(qid2idx[key])
        qid2choices[qid] = ids

    # --- edges_u_to_qopt from U->Q edges ---
    idx2qidname = {idx: name for name, idx in qid2idx.items()}
    edges_u_to_qopt: List[Tuple[int,str,int]] = []
    ei = data[("user","to","question")].edge_index
    for e in range(ei.size(1)):
        uidx = int(ei[0, e]); qnode_idx = int(ei[1, e])
        name = idx2qidname[qnode_idx]        # "327a_1"
        qid = name.rsplit("_", 1)[0]
        edges_u_to_qopt.append((uidx, qid, qnode_idx))

    return data, uid2idx, qid2idx, qid2choices, edges_u_to_qopt


def load_jsonl_as_dict_of_dict(path: str, key: str) -> Dict[str, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj[key]] = obj
    return data

