# dataset.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal as TDict, Set, Optional
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
import json
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os

# --- GNN Imports ---
from utils import build_graph_from_raw


UserIdx = int
QID = str
QNodeIdx = int
EdgeTriple = Tuple[UserIdx, QID, QNodeIdx]


@dataclass
class Split:
    train: List[EdgeTriple]
    val:   List[EdgeTriple]
    test:  List[EdgeTriple]

@dataclass
class QAGraph:
    data: HeteroData
    qid2choices: Dict[QID, List[QNodeIdx]]
    edges_u_to_qopt: List[EdgeTriple]
    uid2idx: Dict[int, int]
    qid2idx: Dict[str, int]

    def clone_mp_view(
        self,
        keep_edges: List[EdgeTriple],
        allowed_users: Optional[Set[int]] = None,
    ) -> HeteroData:
        data_mp = self.data.clone()
        allowed_users = set(allowed_users) if allowed_users is not None else None
        
        # --- Enforce fixed node counts ---
        # Prevents PyG from inferring a smaller graph size when edges are removed
        # This ensures batching offsets remain correct (e.g., +Num_Users instead of +Num_Active_Users).
        for node_type in self.data.node_types:
            data_mp[node_type].num_nodes = self.data[node_type].num_nodes
        # --- Enforce fixed node counts end ---

        valid_pairs = set((u, qopt) for (u, _qid, qopt) in keep_edges)

        # ---- Filter user -> question ----
        if ('user', 'to', 'question') in data_mp.edge_types:
            u2q = data_mp[('user', 'to', 'question')].edge_index
            
            if u2q.numel() > 0:
                u_nodes = u2q[0].tolist()
                q_nodes = u2q[1].tolist()
                
                mask = []
                for u, q in zip(u_nodes, q_nodes):
                    is_valid = (u, q) in valid_pairs
                    is_allowed = (allowed_users is None) or (u in allowed_users)
                    mask.append(is_valid and is_allowed)
                
                keep_mask = torch.tensor(mask, dtype=torch.bool, device=u2q.device)
                u2q = u2q[:, keep_mask]
            
            data_mp[('user', 'to', 'question')].edge_index = u2q
        else:
            data_mp[('user', 'to', 'question')].edge_index = torch.empty((2, 0), dtype=torch.long)

        # ---- Mirror question -> user ----
        if ('question', 'rev_to', 'user') in data_mp.edge_types:
            data_mp[('question', 'rev_to', 'user')].edge_index = \
                data_mp[('user', 'to', 'question')].edge_index.flip(0)

        # ---- Filter subgroup edges ----
        if allowed_users is not None and ('subgroup', 'to', 'user') in data_mp.edge_types:
            su = data_mp[('subgroup', 'to', 'user')].edge_index
            if su.numel() > 0:
                user_idx = su[1]
                allowed_tensor = torch.tensor(list(allowed_users), device=su.device)
                mask_su = torch.isin(user_idx, allowed_tensor)
                su = su[:, mask_su]
            data_mp[('subgroup', 'to', 'user')].edge_index = su

            if ('user', 'rev_to', 'subgroup') in data_mp.edge_types:
                data_mp[('user', 'rev_to', 'subgroup')].edge_index = su.flip(0)

        return data_mp

    def create_weighted_active_view(
        self,
        observed_edges: List[EdgeTriple],
        candidate_edges: List[EdgeTriple],
        candidate_u_local_indices: List[int]
    ) -> Tuple[HeteroData, torch.Tensor, torch.Tensor]:
        # 1. First call the existing filtering logic to get the "observed" part of the graph
        # At this point, data_mp only contains edges from observed_edges
        data_mp = self.clone_mp_view(keep_edges=observed_edges)
        device = data_mp[('user', 'to', 'question')].edge_index.device

        # 2. Prepare the candidate edge tensor [2, num_candidates]
        # candidate_edges: (u_global, q_id, q_opt_node)
        u_cand = [t[0] for t in candidate_edges]
        q_cand = [t[2] for t in candidate_edges] # Note: here we use q_opt_node
        candidate_ei = torch.tensor([u_cand, q_cand], dtype=torch.long, device=device)

        # 3. Manually inject the candidate edges into the graph (Injection)
        obs_ei = data_mp[('user', 'to', 'question')].edge_index
        num_obs = obs_ei.size(1)
        num_cand = candidate_ei.size(1)

        # Merge edge indices
        full_ei = torch.cat([obs_ei, candidate_ei], dim=1)
        data_mp[('user', 'to', 'question')].edge_index = full_ei
        
        # Synchronize the update of the reverse edges
        if ('question', 'rev_to', 'user') in data_mp.edge_types:
            data_mp[('question', 'rev_to', 'user')].edge_index = full_ei.flip(0)

        # 4. Build the mask and mapping
        # Since we manually injected the edges, we can now be 100% sure about the position of the candidate edges
        candidate_mask = torch.zeros(num_obs + num_cand, dtype=torch.bool, device=device)
        candidate_mask[num_obs:] = True # The second half is all candidate edges
        
        # Since it is a 1:1 injection, the mapping is simply the input indices
        edge_to_u_mapping = torch.tensor(candidate_u_local_indices, device=device)

        return data_mp, candidate_mask, edge_to_u_mapping



class SupervisionDataset(Dataset):
    """
    Produces supervision triples for CE over a question's option set:
      returns dict(users, option_ids, gold_idx).
    """
    def __init__(self, sup_edges: List[EdgeTriple], qid2choices: Dict[QID, List[QNodeIdx]]):
        self.sup = sup_edges
        self.qid2choices = qid2choices
        for _, qid, qopt in self.sup[:100]:
            assert qid in qid2choices and qopt in set(qid2choices[qid]), f"Unknown qid/option: {qid}/{qopt}"

    def __len__(self): return len(self.sup)

    def __getitem__(self, idx):
        u, qid, qopt = self.sup[idx]
        option_ids = self.qid2choices[qid]
        gold_idx = option_ids.index(qopt)  
        return int(u), torch.tensor(option_ids, dtype=torch.long), int(gold_idx)

    @staticmethod
    def collate(batch):
        users = torch.tensor([b[0] for b in batch], dtype=torch.long)
        
        # 1. Collect the list of option tensors (variable lengths) 
        seq_list = [b[1] for b in batch]
        
        # 2. Pad sequences. 
        # We use -1 initially to clearly distinguish padding from a valid index 0.
        option_ids = pad_sequence(seq_list, batch_first=True, padding_value=-1)
        
        # 3. Create a boolean mask (True = Real Data, False = Padding)
        # This will be used in the model to ignore the padded options.
        mask = (option_ids != -1)
        
        # 4. Safety fix for Embeddings: 
        # nn.Embedding cannot handle -1. We replace padded -1 with 0.
        # (Since we have the mask, we know to ignore these 0s later)
        option_ids[~mask] = 0 
        
        gold_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
        
        return {
            "users": users, 
            "option_ids": option_ids, 
            "gold_idx": gold_idx,
            "mask": mask # Pass this to the model to ignore the padded options
        }


def load_split_from_json(path: str, edges_all: List[EdgeTriple], uid2idx: Dict[int, int]) -> Split:
    """
    Load a user-based split from JSON, where JSON uses raw user ids (e.g., caseid),
    but edges_all uses internal user indices.

    JSON schema (user lists):
      {
        "train_users": [uid1, uid2, ...],
        "val_users":   [uid3, ...],
        "test_users":  [uid4, ...]
      }

    We map uid -> user_idx via uid2idx, then derive edge splits from edges_all.
    """
    with open(path, "r") as f:
        obj = json.load(f)

    # Only support user-list mode (uid)
    if not all(k in obj for k in ("train_users", "val_users", "test_users")):
        raise ValueError(
            f"JSON at {path} must contain 'train_users', 'val_users', 'test_users' "
            f"with *raw user ids* (e.g., caseid)."
        )

    # Map uid -> idx, if uid is not in uid2idx, skip and print warning
    def uids_to_indices(uids):
        idxs: Set[int] = set()
        missing = 0
        for uid in uids:
            if uid in uid2idx:
                idxs.add(uid2idx[uid])
            else:
                missing += 1
        if missing > 0:
            print(f"[load_split_from_json] Warning: {missing} uids not found in uid2idx and were skipped.")
        return idxs

    tr_users_idx: Set[int] = uids_to_indices(obj["train_users"])
    va_users_idx: Set[int] = uids_to_indices(obj["val_users"])
    te_users_idx: Set[int] = uids_to_indices(obj["test_users"])
    # moe_users_idx: Set[int] = uids_to_indices(obj["train_moe_users"])

    # Use user_idx to split edges_all
    tr, va, te, moe = [], [], [], []
    for e in edges_all:
        u_idx = e[0]
        if u_idx in tr_users_idx:
            tr.append(e)
        elif u_idx in va_users_idx:
            va.append(e)
        elif u_idx in te_users_idx:
            te.append(e)
        # Otherwise, this user is not in any split's user list, discard it (since we only support user-list mode)

    print(
        f"[load_split_from_json] "
        f"train_edges={len(tr)}, val_edges={len(va)}, test_edges={len(te)}"
    )
    return Split(train=tr, val=va, test=te)



class EpochMasker:
    """
    - Train: each epoch, split train_users into 3 groups:
            1/3: keep all edges (all -> mp, no sup)
            1/3: randomly mask half_ratio proportion of edges (some sup, some mp)
            1/3: mask all edges (all -> sup, no mp)
    - Val/Test: completely cold-start (all edges -> sup, mp is empty)
    """
    def __init__(self, split: Split, half_ratio: float = 0.5):
        assert 0.0 < half_ratio < 1.0
        self.split = split
        self.half_ratio = half_ratio

    def _partition_train(self, part_edges: List[EdgeTriple], seed: int):
        """
        Only used for train split's 1/3–1/3–1/3 logic.
        """
        rng = random.Random(seed)

        # 1) Group by user
        by_user: Dict[int, List[EdgeTriple]] = {}
        for e in part_edges:
            u = e[0]
            by_user.setdefault(u, []).append(e)

        users = list(by_user.keys())
        rng.shuffle(users)

        n_users = len(users)
        if n_users == 0:
            return [], []

        base = n_users // 3
        keep_all_users = set(users[:base])              # all mp
        half_mask_users = set(users[base:2 * base])     # half mp half sup
        mask_all_users  = set(users[2 * base:])         # all sup

        sup: List[EdgeTriple] = []
        mp:  List[EdgeTriple] = []

        for u, edges_u in by_user.items():
            e_u = edges_u[:]

            if u in keep_all_users:
                # all edges -> mp
                mp.extend(e_u)

            elif u in mask_all_users:
                # all edges -> sup
                sup.extend(e_u)

            else:
                # half_mask_users: randomly mask some edges
                rng.shuffle(e_u)
                n = len(e_u)
                n_sup = int(self.half_ratio * n)
                sup.extend(e_u[:n_sup])
                mp.extend(e_u[n_sup:])

        return sup, mp

    def _partition_cold(self, part_edges: List[EdgeTriple]):
        """
            val/test split: completely cold-start
            all edges -> sup, mp is empty
        """
        sup = part_edges[:]   # all edges -> sup
        mp: List[EdgeTriple] = []
        return sup, mp

    def epoch_spec(self, epoch: int):
        # train: dynamic 1/3–1/3–1/3
        sup_tr, mp_tr = self._partition_train(self.split.train, seed=epoch * 101 + 1)
        # val/test: each epoch is strict cold-start
        sup_va, mp_va = self._partition_cold(self.split.val)
        sup_te, mp_te = self._partition_cold(self.split.test)

        return {
            "train": (sup_tr, mp_tr),
            "val":   (sup_va, mp_va),
            "test":  (sup_te, mp_te),
        }


def build_loaders_for_epoch(
    graph: QAGraph,
    epoch_spec: Dict[str, Tuple[List[EdgeTriple], List[EdgeTriple]]],
    batch_size: int = 2048, num_workers: int = 0
) -> Dict[str, Dict[str, object]]:
    """Return {split: {'data_mp', 'loader', 'num_sup', 'num_mp'}}."""
    out: TDict[str, TDict[str, object]] = {}

    for split_name, (sup_edges, mp_edges) in epoch_spec.items():
        # All users in this split (train only contains train users, val/test contain all users)
        allowed_users: Set[int] = set(e[0] for e in sup_edges) | set(e[0] for e in mp_edges)

        data_mp = graph.clone_mp_view(mp_edges, allowed_users=allowed_users)

        ds = SupervisionDataset(sup_edges, graph.qid2choices)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=SupervisionDataset.collate, drop_last=False
        )

        out[split_name] = {
            "data_mp": data_mp,
            "loader": loader,
            "num_sup": len(sup_edges),
            "num_mp": len(mp_edges),
        }

    return out



def load_qa_graph(cfg):
    """Load demo/question CSVs and build QAGraph from config."""
    base_dir = cfg["data"]["base_dir"]
    id_dist = cfg["data"]["id_dist"]
    ood_dist = cfg["data"]["ood_dist"]

    demo_train = pd.read_csv(os.path.join(base_dir, f"demo_{id_dist}_train.csv"))
    demo_val = pd.read_csv(os.path.join(base_dir, f"demo_{id_dist}_val.csv"))
    demo_test = pd.read_csv(os.path.join(base_dir, f"demo_{ood_dist}_test.csv"))
    df_demo_all = pd.concat([demo_train, demo_val, demo_test], ignore_index=True)

    q_train = pd.read_csv(os.path.join(base_dir, f"question_{id_dist}_train.csv"))
    q_val = pd.read_csv(os.path.join(base_dir, f"question_{id_dist}_val.csv"))
    q_test = pd.read_csv(os.path.join(base_dir, f"question_{ood_dist}_test.csv"))
    df_question_all = pd.concat([q_train, q_val, q_test], ignore_index=True)

    data, uid2idx, qid2idx, qid2choices, edges_u_to_qopt = build_graph_from_raw(
        demo_csv=df_demo_all,
        question_csv=df_question_all,
        codebook_path=cfg["data"]["codebook_path"],
        demo_cols=list(cfg["data"]["demo_cols"]),
        question_cols=list(cfg["data"]["question_cols"]),
    )

    return QAGraph(
        data=data,
        qid2choices=qid2choices,
        edges_u_to_qopt=edges_u_to_qopt,
        uid2idx=uid2idx,
        qid2idx=qid2idx,
    )