import os
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, Iterable, Optional
from torch.nn import functional as F
import numpy as np
import torch
import pandas as pd
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from src.gnn.model import HGNNModel
from src.gnn.dataset import QAGraph, build_loaders_for_epoch, EdgeTriple
from src.gnn.utils import load_config, parse_overrides, deep_update, build_graph_from_raw


def _pick_device(name: str):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class GNNElicitationPredictor:
    """
    GNN predictor for adaptive elicitation.
    - only build test user graph (demo_ood_dist_test + question_ood_dist_test)
    - only do message passing on test user graph
    - return predictions for all test users for a given qid
    - can add new (uid, qid) observations to mp edges in each round
    """

    def __init__(
        self,
        graph: QAGraph,
        edges_u_to_qopt: List[EdgeTriple],
        model: HGNNModel,
        device: torch.device,
    ):
        self.graph = graph
        self.edges_u_to_qopt = edges_u_to_qopt
        self.model = model
        self.device = device

        # uid mapping
        self.uid2idx: Dict[str, int] = graph.uid2idx
        self.idx2uid: Dict[int, str] = {v: k for k, v in self.uid2idx.items()}

       
        # assume graph object has this attribute (usually graph.uid2idx exists when graph.qid2idx exists)
        if hasattr(graph, 'qid2idx'):
            self.qid2idx: Dict[str, int] = graph.qid2idx
        else:
            # if graph doesn't have it, try to build from node_ids
            # assume graph.question_nodes or similar structure
            # this is a defensive write
            pass 

        self.qid2choices = graph.qid2choices

        # current "observed" message-passing edges (only on test graph)
        self.mp_edges: set[EdgeTriple] = set()

        # convenient to find unique edge triple from (user_idx, qid)
        self._edge_by_uq: Dict[Tuple[int, str], EdgeTriple] = {}
        for u_idx, qid, qnode_idx in self.edges_u_to_qopt:
            self._edge_by_uq[(u_idx, qid)] = (u_idx, qid, qnode_idx)

        # record real observed answers
        # user_history[uid] = [(qid, ans_idx), ...]
        # ans_idx is the index of the option (0-based), corresponding to A/B/C/..., used to overwrite predictions during eval.
        self.user_history: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.original_mp_edges = list(self.mp_edges).copy()
       
    def reset_graph(self):
        """
        reset graph state, undo all observations added through add_observations
        """
        # 1. restore mp_edges to initial backup state
        # because your original_mp_edges is empty in __init__, this will clear all dynamically added edges
        self.mp_edges = set(self.original_mp_edges)

        # 2. clear user history (if your prediction logic uses user_history to overwrite predictions)
        if hasattr(self, 'user_history'):
            self.user_history.clear()

        # 3. [critical] synchronize reset Graph object's edges and cache
        # ensure GNN recognizes edge changes in the next forward pass
        if hasattr(self.graph, 'mp_edges'):
            self.graph.mp_edges = self.mp_edges
            
        # if Graph class has a mechanism to cache Data objects, must clear it
        if hasattr(self.graph, 'cache'):
            self.graph.cache = {}
            
        print(f"[GNNElicitationPredictor] Graph reset. Current mp_edges={len(self.mp_edges)}")

    @classmethod
    def from_config(
        cls,
        config_path: str,
        set_overrides: Optional[Iterable[str]] = None,
        device_name: str = "auto"):

        cfg = load_config(config_path)
        overrides = parse_overrides(set_overrides or [])
        cfg = deep_update(cfg, overrides)

        device = _pick_device(cfg["train"].get("device", device_name))

        base_dir = cfg["data"]["base_dir"]
        id_dist = cfg["data"]["id_dist"]
        ood_dist = cfg["data"]["ood_dist"]

        # ---- 1. only load demo / question from OOD test panel ----
        demo_test_path = os.path.join(base_dir, f"demo_{ood_dist}_test.csv")
        q_test_path = os.path.join(base_dir, f"question_{ood_dist}_test.csv")

        df_demo_test = pd.read_csv(demo_test_path)
        df_question_test = pd.read_csv(q_test_path)
        df_demo_test["caseid"] = df_demo_test["caseid"].astype(int).astype(str)
        df_question_test["caseid"] = df_question_test["caseid"].astype(int).astype(str)

        # ---- 2. build graph using test-only DataFrame (S, C are still full)----
        data, uid2idx, qid2idx, qid2choices, edges_u_to_qopt = build_graph_from_raw(
            demo_csv=df_demo_test,
            question_csv=df_question_test,
            codebook_path=cfg["data"]["codebook_path"],
            demo_cols=list(cfg["data"]["demo_cols"]),
            question_cols=list(cfg["data"]["question_cols"]),
        )

        graph = QAGraph(
            data=data,
            qid2choices=qid2choices,
            edges_u_to_qopt=edges_u_to_qopt,
            uid2idx=uid2idx,
            qid2idx=qid2idx,
        )

        # build GNN model & load checkpoint trained on full-graph
        model = HGNNModel(
            graph.data,
            d_in=int(cfg["model"]["hidden"]),
            d_h=int(cfg["model"]["hidden"]),
            layers=int(cfg["model"]["layers"]),
            dropout=float(cfg["model"]["dropout"]),
        ).to(device)

        ckpt_dir = cfg["checkpoint"]["ckpt_dir"]
        ckpt_prefix = cfg["checkpoint"]["ckpt_prefix"]
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pt")
        print(f"[GNNElicitationPredictor] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        return cls(
            graph=graph,
            edges_u_to_qopt=edges_u_to_qopt,
            model=model,
            device=device,
        )

  
    # update observations: turn new (uid, qid) into visible mp edges
    def add_observations(
        self,
        obs_list: Iterable[Tuple[str, str]],
        *,
        raw_uid: bool = True,
    ):
        """
        add new observation to test graph's message-passing edges.

        parameters:
            obs_list: iterable of (uid, qid)
              - uid: default is original caseid (string); if raw_uid=False, it is considered as internal user_idx
              - qid: question id (e.g. "v161")

        only add edges that exist in test graph; edges that don't exist in test will be skipped.
        """
        added = 0
        for u_raw, qid in obs_list:
            if raw_uid:
                u_lookup = u_raw
                if u_lookup not in self.uid2idx:
                    try:
                        u_canon = str(int(float(u_lookup)))
                        if u_canon in self.uid2idx:
                            u_lookup = u_canon
                    except (ValueError, TypeError):
                        pass
                if u_lookup not in self.uid2idx:
                    print(f"[GNNElicitationPredictor] WARNING: uid={u_raw} not found in test uid2idx, skip.")
                    continue
                u_idx = self.uid2idx[u_lookup]
            else:
                u_idx = int(u_raw)

            key = (u_idx, qid)
            e = self._edge_by_uq.get(key, None)
            if e is None:
                # this (user, qid) has no answer record in test panel
                print(f"[GNNElicitationPredictor] WARNING: no TEST edge for (u_idx={u_idx}, qid={qid}), skip.")
                continue

            if e not in self.mp_edges:
                self.mp_edges.add(e)
                added += 1

        print(f"[GNNElicitationPredictor] Added {added} new observed TEST mp edges, "
              f"total mp_edges(test)={len(self.mp_edges)}")

    def reset_observations(self):
        """clear all observed mp edges, back to strict cold-start."""
        self.mp_edges.clear()
        print("[GNNElicitationPredictor] Cleared all observed TEST mp edges.")

    def record_answer(
        self,
        uid: str,
        qid: str,
        ans_idx: int,
        *,
        allow_duplicate: bool = False,
    ):
        """
        record a real user answer, used to overwrite GNN predictions during eval.

        parameters:
            uid: original user id (caseid)
            qid: question id (e.g. "v161")
            ans_idx: option index (0-based), 0->A, 1->B, ...
            allow_duplicate:
                - False: if the uid already has a record for the same qid, overwrite the old value
                - True: allow the same (uid, qid) to appear multiple times, append to the list
        """
        hist = self.user_history[uid]

        if not allow_duplicate:
            for i, (old_qid, _) in enumerate(hist):
                if old_qid == qid:
                    hist[i] = (qid, ans_idx)
                    break
            else:
                hist.append((qid, ans_idx))
        else:
            hist.append((qid, ans_idx))

    def get_recorded_answer(
        self,
        uid: str,
        qid: str,
    ) -> Optional[int]:
        """
        return whether the uid has a recorded answer for the qid:
            - if yes, return ans_idx (0-based)
            - if no, return None
        """
        for q, ans_idx in self.user_history.get(uid, []):
            if q == qid:
                return ans_idx
        return None

    def summarize_user_history(self):
        """
        print the number of records for each question in user_history:
            - how many records for each question
        (for debugging)
        """
        from collections import Counter

        if not self.user_history:
            print("[TestPredictor] user_history is empty.")
            return

        q_counter = Counter()
        for uid, qa_list in self.user_history.items():
            for qid, ans_idx in qa_list:
                q_counter[qid] += 1

        print("===== Question count in user_history =====")
        for qid, cnt in sorted(q_counter.items(), key=lambda x: -x[1]):
            print(f"qid = {qid:20s}  count = {cnt}")


    # ------------------------------------------------------------------
    # given qid, do one GNN prediction for all test users
    # ------------------------------------------------------------------
    def predict_for_question(
        self,
        qid: str,
        batch_size: int = 256,
    ) -> Dict[str, Dict[str, object]]:
        """
        do one GNN prediction for all test users for the specified question (qid) in the current mp_edges state.

        returns:
            preds[uid] = {
                "probs": List[float],        # [p_0, ..., p_{K-1}]
                "choice_index": int,         # argmax index (0-based)
                "choice_letter": str,        # 'A' / 'B' / ...
                "qopt_node_idx": int,        # selected option's question node idx
            }
        """
        self.model.eval()

        # 1) all test edges for the specified question (qid)
        sup_edges_q: List[EdgeTriple] = [
            e for e in self.edges_u_to_qopt if e[1] == qid
        ]
        
        if not sup_edges_q:
            print(f"[GNNElicitationPredictor] WARNING: no TEST edges found for qid={qid}")
            return {}

        # 2) current test mp edges (all observed (u, q))
        mp_edges = list(self.mp_edges)

        # 3) construct an epoch_spec with only 'test' to pass to build_loaders_for_epoch
        epoch_spec = {
            "test": (sup_edges_q, mp_edges)
        }

        packs = build_loaders_for_epoch(
            self.graph,
            epoch_spec,
            batch_size=batch_size,
            num_workers=0,
        )
        pack = packs["test"]
        data_mp = pack["data_mp"].to(self.device)
        loader = pack["loader"]

        preds: Dict[str, Dict[str, object]] = {}

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                out = self.model(data_mp, batch)
                # training mode: with gold_idx -> (loss, acc, probs)
                if isinstance(out, tuple):
                    _, _, probs = out
                else:
                    probs = out
                probs = probs.detach().cpu().numpy()

                user_idx = batch["users"].detach().cpu().tolist()
                opt_batch = batch["option_ids"].detach().cpu().tolist()  # [B, K]

                B, K = probs.shape
                idx2letter = {i: chr(ord("A") + i) for i in range(K)}

                for u_i, opt_ids, p in zip(user_idx, opt_batch, probs):
                    uid = self.idx2uid[u_i]
                    k = int(np.argmax(p))
                    letter = idx2letter.get(k, str(k))
                    qopt_node_idx = int(opt_ids[k])

                    preds[uid] = {
                        "probs": p.tolist(),
                        "choice_index": k,
                        "choice_letter": letter,
                        "qopt_node_idx": qopt_node_idx,
                    }

        return preds

    # ------------------------------------------------------------------
    # select nodes for query
    # ------------------------------------------------------------------
    def select_nodes(self, candidate_uids: List[str], k: int) -> List[str]:
        """
        use KMeans to cluster the nodes based on their embeddings, then select the nodes closest to the centroids.
        parameters:
            candidate_uids: list of candidate user ids
            k: number of nodes to select

        returns:
            list of selected user ids
        """
        self.model.eval()
        device = self.device

        # active candidates (present in graph)
        active_u_local_indices = [i for i, uid in enumerate(candidate_uids) if uid in self.uid2idx]
        if not active_u_local_indices:
            return candidate_uids[:k]

        k_to_select = min(k, len(active_u_local_indices))

        u_global_idx = torch.tensor(
            [self.uid2idx[candidate_uids[i]] for i in active_u_local_indices],
            device=device
        )

        with torch.no_grad():
            # build current mp view (no injected candidate edges)
            data_mp, _, _ = self.graph.create_weighted_active_view(list(self.mp_edges), [], [])
            data_mp = data_mp.to(device)
            base_weights = {
                etype: torch.ones(data_mp[etype].edge_index.size(1), device=device)
                for etype in data_mp.edge_types
            }

            z = self.model(data_mp, return_z=True)
            zu = z["user"][u_global_idx]
            zu = F.normalize(zu, p=2, dim=-1)

        # KMeans on active users
        features_np = zu.detach().cpu().numpy().astype(np.float64)
  
        kmeans = KMeans(
            n_clusters=k_to_select,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        kmeans.fit(features_np)

        # pick nearest sample to each centroid
        top_k_active_local, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features_np)

        selected_local_idx = [active_u_local_indices[i] for i in top_k_active_local]
        selected_uids = [candidate_uids[i] for i in selected_local_idx]

        # fill to k (if not enough)
        if len(selected_uids) < k:
            existing = set(selected_uids)
            for uid in candidate_uids:
                if len(selected_uids) >= k:
                    break
                if uid not in existing:
                    selected_uids.append(uid)
                    existing.add(uid)

        return selected_uids


def build_gold_answers_from_edges(predictor) -> Dict[tuple, str]:
    """
    build all (uid, qid) ground truth answers from the test graph of the predictor.
    Returns:
        gold_map[(uid, qid)] = 'A' / 'B' / 'C' / ...
    """
    gold = {}
    qid2choices = predictor.qid2choices          # {qid: [qnode_idx0, qnode_idx1, ...]}
    idx2uid = predictor.idx2uid                  # {user_idx: uid(str)}

    for u_idx, qid, qnode_idx in predictor.edges_u_to_qopt:
        uid = idx2uid[u_idx]
        choices = qid2choices[qid]              # e.g. [10, 11] -> A/B

        try:
            k = choices.index(qnode_idx)        # the user selected choices[k]
        except ValueError:
            # theoretically shouldn't happen, skip for safety
            continue

        letter = chr(ord("A") + k)              # 0->A, 1->B, ... -> A/B/C/...
        gold[(uid, qid)] = letter

    print(f"[GNNElicitationPredictor] Built gold answers for {len(gold)} (uid, qid) pairs.")
    return gold