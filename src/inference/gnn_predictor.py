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

from .model import GEMSModel
from .dataset import OpinionQAGraph, build_loaders_for_epoch, EdgeTriple
from .utils import load_config, parse_overrides, deep_update, build_graph_from_raw


def _pick_device(name: str):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class GNNElicitationPredictor:
    """
    GNN predictor for adaptive elicitation.

    """

    def __init__(
        self,
        graph: OpinionQAGraph,
        edges_u_to_qopt: List[EdgeTriple],
        model: GEMSModel,
        device: torch.device,
    ):
        self.graph = graph
        self.edges_u_to_qopt = edges_u_to_qopt
        self.model = model
        self.device = device

        # uid mapping
        self.uid2idx: Dict[str, int] = graph.uid2idx
        self.idx2uid: Dict[int, str] = {v: k for k, v in self.uid2idx.items()}


        # qid mapping
        if hasattr(graph, 'qid2idx'):
            self.qid2idx: Dict[str, int] = graph.qid2idx
        else:
            raise ValueError("graph.qid2idx not found")

        self.qid2choices = graph.qid2choices

        # current observed message-passing edges (only in test graph)
        self.mp_edges: set[EdgeTriple] = set()

        # convenient to find the unique edge triple from (user_idx, qid)
        self._edge_by_uq: Dict[Tuple[int, str], EdgeTriple] = {}
        for u_idx, qid, qnode_idx in self.edges_u_to_qopt:
            self._edge_by_uq[(u_idx, qid)] = (u_idx, qid, qnode_idx)

        # record the true observed answers
        # user_history[uid] = [(qid, ans_idx), ...]
        # ans_idx is the index of the option (0-based), corresponding to A/B/C/..., used to overwrite the prediction during evaluation.
        self.user_history: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.original_mp_edges = list(self.mp_edges).copy()
       
    # ------------------------------------------------------------------
    # 1) construct: build test-only graph + model + ckpt from config
    # ------------------------------------------------------------------
    def reset_graph(self):
        """
        reset the graph state, undo all observations added through add_observations.
        """
        # 1. restore mp_edges to the initial backup state
        # because your original_mp_edges is empty in __init__, this will clear all dynamically added edges
        self.mp_edges = set(self.original_mp_edges)

        # 2. clear user history (if your prediction logic uses user_history to overwrite the prediction value)
        if hasattr(self, 'user_history'):
            self.user_history.clear()

        # 3. [Important] synchronize the reset of the edges and cache inside the Graph object
        # ensure the GNN recognizes the change of edges in the next forward pass
        if hasattr(self.graph, 'mp_edges'):
            self.graph.mp_edges = self.mp_edges
            
        # if the Graph class has a mechanism to cache Data objects, it must be cleared
        if hasattr(self.graph, 'cache'):
            self.graph.cache = {}
            
        # print(f"[TestPredictor] Graph reset. Current mp_edges={len(self.mp_edges)}")

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

        # ---- 1. only load the demo / question of the OOD test panel ----
        demo_test_path = os.path.join(base_dir, f"demo_{ood_dist}_test.csv")
        q_test_path = os.path.join(base_dir, f"question_{ood_dist}_test.csv")

        df_demo_test = pd.read_csv(demo_test_path)
        df_question_test = pd.read_csv(q_test_path)

        # ---- 2. use the test-only DataFrame to construct the graph (S, C are still full)----
        data, uid2idx, qid2idx, qid2choices, edges_u_to_qopt = build_graph_from_raw(
            demo_csv=df_demo_test,
            question_csv=df_question_test,
            codebook_path=cfg["data"]["codebook_path"],
            demo_cols=list(cfg["data"]["demo_cols"]),
            question_cols=list(cfg["data"]["question_cols"]),
        )

        graph = OpinionQAGraph(
            data=data,
            qid2choices=qid2choices,
            edges_u_to_qopt=edges_u_to_qopt,
            uid2idx=uid2idx,
            qid2idx=qid2idx,
        )

        # ---- 3. construct the GNN model & load the checkpoint trained on the full-graph ----
        model = GEMSModel(
            graph.data,
            d_in=int(cfg["model"]["hidden"]),
            d_h=int(cfg["model"]["hidden"]),
            layers=int(cfg["model"]["layers"]),
            dropout=float(cfg["model"]["dropout"]),
        ).to(device)

        ckpt_dir = cfg["checkpoint"]["ckpt_dir"]
        ckpt_prefix = cfg["checkpoint"]["ckpt_prefix"]
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pt")
        print(f"[TestPredictor] Loading checkpoint from: {ckpt_path}")
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

  
    # ------------------------------------------------------------------
    # 2) update observations: make new (uid, qid) visible mp edges
    # ------------------------------------------------------------------
    def add_observations(
        self,
        obs_list: Iterable[Tuple[str, str]],
        *,
        raw_uid: bool = True,
    ):
        """
        add new observations to the message-passing edges of the test graph.

        parameters:
            obs_list: iterable of (uid, qid)
              - uid: default is the original caseid (string); if raw_uid=False, it is considered as the internal user_idx
              - qid: question id (e.g. "v161")

        only add edges that exist in the test graph; edges that do not exist in the test will be skipped.
        """
        added = 0
        for u_raw, qid in obs_list:
            if raw_uid:
                if u_raw not in self.uid2idx:
                    print(f"[TestPredictor] WARNING: uid={u_raw} not found in test uid2idx, skip.")
                    continue
                u_idx = self.uid2idx[u_raw]
            else:
                u_idx = int(u_raw)

            key = (u_idx, qid)
            e = self._edge_by_uq.get(key, None)
            if e is None:
                # 说明这个 (user, qid) 在 test panel 里没有回答记录
                print(f"[TestPredictor] WARNING: no TEST edge for (u_idx={u_idx}, qid={qid}), skip.")
                continue

            if e not in self.mp_edges:
                self.mp_edges.add(e)
                added += 1

        print(f"[TestPredictor] Added {added} new observed TEST mp edges, "
              f"total mp_edges(test)={len(self.mp_edges)}")

    def reset_observations(self):
        """clear all observed mp edges, back to strict cold-start."""
        self.mp_edges.clear()
        print("[TestPredictor] Cleared all observed TEST mp edges.")

    def record_answer(
        self,
        uid: str,
        qid: str,
        ans_idx: int,
        *,
        allow_duplicate: bool = False,
    ):
        """
        record one true user answer, used to overwrite the GNN prediction during evaluation.

        parameters:
            uid: original user id (caseid)
            qid: question id (e.g. "v161")
            ans_idx: option index (0-based), 0->A, 1->B, ...
            allow_duplicate:
                - False: if the uid has already recorded the same qid, overwrite the old value
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
        return the recorded answer for the uid on qid:
            - if there is a recorded answer, return ans_idx (0-based)
            - if there is no recorded answer, return None
        """
        for q, ans_idx in self.user_history.get(uid, []):
            if q == qid:
                return ans_idx
        return None

    def summarize_user_history(self):
        """
        print the number of times each question is recorded in the user_history.
        assume:
            - user_history: Dict[str, List[Tuple[str, int]]]
            - each element is (qid, ans_idx)
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


    def add_imputations(
        self,
        qid: str,
        min_conf: float = 0.0,
        uids: Optional[Iterable[str]] = None,
        overwrite: bool = False,
    ) -> int:
        """
        use the current GNN prediction, add the high-confidence predictions for the specified question as "pseudo-observations" to the mp_edges.

        logic:
            1. call predict_for_question(qid) to get the predictions for all test users;
            2. for each user:
                - if uids is provided, only do imputations on the subset;
                - take samples with max prob >= min_conf;
                - add the (user_idx, qid, predicted option corresponding to qopt_node_idx) as a new mp edge;
                - if the (user, qid) is already in mp_edges:
                    - overwrite=False: skip
                    - overwrite=True: delete the existing edge, then insert the new imputed edge

        返回：
            the number of actual/overwritten mp edges added
        """
        # first predict on the current mp_edges state
        preds = self.predict_for_question(qid)
        if not preds:
            print(f"[TestPredictor] add_imputations: no preds for qid={qid}, nothing added.")
            return 0

        # if only do imputation on a subset of users, build a set to filter
        uid_allow: Optional[set] = None
        if uids is not None:
            uid_allow = set(map(str, uids))

        added = 0

        # to avoid duplicate edges, first collect the combinations of (u_idx, qid) in the existing mp_edges
        existing_uq = set((u_idx, q) for (u_idx, q, _) in self.mp_edges)

        for uid, info in preds.items():
            if uid_allow is not None and uid not in uid_allow:
                continue

            probs = np.asarray(info["probs"], dtype=float)
            max_p = float(probs.max())
            if max_p < min_conf:
                # not confident enough, not add
                continue

            # predicted option corresponding to the question node idx (from predict_for_question)
            qopt_node_idx = int(info["qopt_node_idx"])

            # user internal idx
            if uid not in self.uid2idx:
                # theoretically not possible, because the uid in preds comes from idx2uid
                continue
            u_idx = self.uid2idx[uid]

            key_uq = (u_idx, qid)

            if key_uq in existing_uq:
                if not overwrite:
                    # existing edge and not overwrite, skip
                    continue
                else:
                    # overwrite: first delete the original (u_idx, qid, *)
                    to_remove = [e for e in self.mp_edges if e[0] == u_idx and e[1] == qid]
                    for e in to_remove:
                        self.mp_edges.discard(e)
                    existing_uq.discard(key_uq)

            # new imputed edge
            e_new: EdgeTriple = (u_idx, qid, qopt_node_idx)
            self.mp_edges.add(e_new)
            existing_uq.add(key_uq)
            added += 1

        print(
            f"[TestPredictor] add_imputations(qid={qid}, min_conf={min_conf}) "
            f"added/overwrote {added} edges, total mp_edges(test)={len(self.mp_edges)}"
        )
        return added


    # ------------------------------------------------------------------
    # 3) core: given qid, do one GNN prediction for all test users
    # ------------------------------------------------------------------
    def predict_for_question(
        self,
        qid: str,
        batch_size: int = 256,
    ) -> Dict[str, Dict[str, object]]:
        """
        do one GNN prediction for all test users on the specified question (qid) in the current mp_edges state.

        return:
            preds[uid] = {
                "probs": List[float],        # [p_0, ..., p_{K-1}]
                "choice_index": int,         # argmax index
                "choice_letter": str,        # 'A' / 'B' / ...
                "qopt_node_idx": int,        # the question node idx corresponding to the selected option
            }
        """
        self.model.eval()

        # 1) all test edges for the specified question (qid)
        sup_edges_q: List[EdgeTriple] = [
            e for e in self.edges_u_to_qopt if e[1] == qid
        ]
        if not sup_edges_q:
            print(f"[TestPredictor] WARNING: no TEST edges found for qid={qid}")
            return {}

        # 2) current test mp edges (all observed (u, q))
        mp_edges = list(self.mp_edges)

        # 3) construct an epoch_spec with only 'test' for the build_loaders_for_epoch
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

        # print(f"[TestPredictor] [TEST-only MP] qid={qid}, predicted {len(preds)} users "
        #       f"(mp_edges(test)={len(self.mp_edges)})")
        return preds
    
    def clone_with_shared_model(self):
        """
        lightweight clone:
        - share graph / model / mapping (uid2idx, qid2choices, edges_u_to_qopt)
        - copy a separate mp_edges / user_history (each worker has its own)
        """
        new = self.__class__.__new__(self.__class__)

        # shared read-only structure
        new.graph = self.graph
        new.edges_u_to_qopt = self.edges_u_to_qopt
        new.model = self.model       # share the same GNN model instance (only forward)
        new.device = self.device

        new.uid2idx = self.uid2idx
        new.idx2uid = self.idx2uid
        new.qid2choices = self.qid2choices
        new._edge_by_uq = self._edge_by_uq

        # state: each clone has its own
        new.mp_edges = self.mp_edges.copy()

        # === NEW: user_history deep copy, keep current observations consistent, but subsequent updates are independent ===
        new.user_history = deepcopy(self.user_history)

        return new
    

    # ------------------------------------------------------------------
    # 4) Support for Information Gain: Virtual Prediction
    # ------------------------------------------------------------------
    def predict_with_virtual_label(
        self,
        qid: str,
        node_id: str,
        label_idx: int,
        batch_size: int = 256
    ) -> Dict[str, Dict[str, object]]:
        """
        Simulates: If 'node_id' answers 'qid' with option 'label_idx', 
        what is the new prediction for 'qid' across the graph?
        """
        if node_id not in self.uid2idx:
            return {}
        
        # 1. Get graph indices
        choice_node_ids = self.qid2choices[qid] 
        target_qopt_node_idx = int(choice_node_ids[label_idx])
        u_idx = self.uid2idx[node_id]

        # 2. Construct the virtual edge (u, q, opt)
        virtual_edge: EdgeTriple = (u_idx, qid, target_qopt_node_idx)

        # 3. Inject into graph state
        # We check if it exists to avoid double-adding or removing existing truth
        is_new_edge = virtual_edge not in self.mp_edges
        if is_new_edge:
            self.mp_edges.add(virtual_edge)

        try:
            # 4. Run Inference (Forward Pass)
            # We predict specifically for the SAME qid
            preds = self.predict_for_question(qid, batch_size=batch_size)
        finally:
            # 5. Revert state (Critical!)
            if is_new_edge:
                self.mp_edges.remove(virtual_edge)

        return preds
    
    # ------------------------------------------------------------------
    # 5) Helper: Extract Embeddings for Active Learning (K-Center)
    # ------------------------------------------------------------------
    def get_user_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Runs the Encoder (RGCN) on the current graph state (mp_edges)
        and returns the latent embedding vector for every user.
        
        Returns:
            Dict[uid, np.array]: { 'caseid_123': array([0.1, ...]), ... }
        """
        self.model.eval()
        
        # 1. Construct the Graph Data (data_mp)
        # We reuse build_loaders_for_epoch logic to ensure edge_index is built 
        # exactly the same way as it is during prediction.
        # We pass an empty list for supervision edges because we don't need a dataloader,
        # we just need the 'data_mp' object.
        mp_edges = list(self.mp_edges)
        epoch_spec = {
            "embedding_extraction": ([], mp_edges) 
        }

        # This helper builds the HeteroData object with the correct edge indices
        packs = build_loaders_for_epoch(
            self.graph,
            epoch_spec,
            batch_size=1, # Irrelevant, we won't iterate
            num_workers=0,
        )
        
        # Get the constructed graph data and move to device
        data_mp = packs["embedding_extraction"]["data_mp"].to(self.device)

        with torch.no_grad():
            # 2. Get Input Features (LearnableTables)
            # x_in = {'user': [U, d], 'question': [Q, d], ...}
            x_in = self.model.tables()
            
            # 3. Run the Encoder
            # z = {'user': [U, Hidden], 'question': [Q, Hidden], ...}
            z = self.model.enc(x_in, data_mp.edge_index_dict)
            
            # 4. Extract User Tensor
            user_emb_tensor = z['user']
            user_emb_np = user_emb_tensor.detach().cpu().numpy()

        if len(user_emb_np) > 1:
            diff = np.linalg.norm(user_emb_np[0] - user_emb_np[1])
            if diff < 1e-6:
                print("[WARNING] User Embeddings are identical! Message Passing might not be working (or graph is empty).")
            else:
                print(f"[INFO] User Embeddings are distinct. Mean Diff: {diff:.4f}")
        # 5. Map back to UIDs
        # uid2idx maps 'caseid' -> integer index
        embeddings_dict = {}
        for uid, idx in self.uid2idx.items():
            # Safety check to ensure index is within bounds
            if idx < len(user_emb_np):
                embeddings_dict[uid] = user_emb_np[idx]

        return embeddings_dict


    def predict_batch_virtual_scenarios(self, qid: str, scenarios: List[Dict]) -> List[Dict]:
        """
        Executes multiple 'what-if' scenarios in a single GPU batch.
        
        Args:
            qid: The question ID being evaluated.
            scenarios: List of dicts, each containing {'node_id': str, 'label_idx': int}
        """
        from torch_geometric.data import Batch
        import torch

        # 1. Prepare Base Data
        current_edges_list = list(self.mp_edges)
        
        if qid not in self.qid2choices:
            raise ValueError(f"QID {qid} not found in qid2choices map.")
        choices_node_indices = self.qid2choices[qid] 

        data_list = []
        
        # 2. Construct Virtual Graphs (The "Map" Step)
        for task in scenarios:
            uid = task['node_id']
            label_idx = task['label_idx']
            
            # Convert string UID to integer index
            u_idx = self.uid2idx[uid]
            
            # Skip invalid labels
            if label_idx >= len(choices_node_indices):
                 continue 
            
            # Identify the specific option node index
            q_choice_node_idx = choices_node_indices[label_idx]
            
            # Construct the virtual edge: (User_Idx, QID, Option_Node_Idx)
            virtual_edge = (u_idx, qid, q_choice_node_idx)
            virtual_edges_list = current_edges_list + [virtual_edge]
            
            # Create the view (using your fixed clone_mp_view)
            data_view = self.graph.clone_mp_view(keep_edges=virtual_edges_list)
            data_list.append(data_view)

        if not data_list:
            return []

        # 3. Create GPU Batch
        batch_data = Batch.from_data_list(data_list)
        batch_data = batch_data.to(self.device)

        # 4. Inference (The "Batch" Step)
        self.model.eval()
        with torch.no_grad():
            # Call with return_z=True to get raw embeddings (Shape: [B*Num_Nodes, Dim])
            # Note: Ensure your GEMSModel.forward accepts return_z and tiles inputs!
            z = self.model(batch_data, batch=None, return_z=True)
            
        # 5. Manual Decoding (The "Reduce" Step)
        results = []
        
        num_users_per_graph = self.graph.data['user'].num_nodes 
        num_questions = self.graph.data['question'].num_nodes
        
        # Extract embeddings
        z_users = z['user']       # [Batch_Size * Num_Users, Dim]
        z_questions = z['question'] # [Batch_Size * Num_Questions, Dim]
        
        for i in range(len(data_list)):
            # Define slices for the i-th graph in the batch
            u_start = i * num_users_per_graph
            u_end = (i + 1) * num_users_per_graph
            
            # A. Get User Embeddings for this scenario
            # Shape: [Num_Users, Dim]
            u_emb = z_users[u_start:u_end]
            
            # B. Get Question Option Embeddings for this scenario
            # The question nodes are also duplicated in the batch, so we must offset indices!
            q_offset = i * num_questions
            current_choice_indices = [idx + q_offset for idx in choices_node_indices]
            
            # Shape: [Num_Options, Dim]
            q_emb = z_questions[current_choice_indices]
            
            # C. Compute Logits (Dot Product)
            # [Num_Users, Dim] @ [Dim, Num_Options] -> [Num_Users, Num_Options]
            logits = torch.matmul(u_emb, q_emb.t())
            probs = torch.softmax(logits, dim=-1)
            
            # D. Save results
            results.append(self._tensor_to_preds_dict(probs.cpu()))

        return results

    def _tensor_to_preds_dict(self, probs_tensor):
        """
        Helper: maps a tensor of shape (Num_Users, Num_Classes) to a dict:
        { 'U1': {'probs': [...]}, 'U2': {'probs': [...]}, ... }
        """
        res = {}
        # Convert tensor rows back to UIDs
        # We iterate only up to the number of users in the graph
        for idx in range(len(probs_tensor)):
            uid = self.idx2uid.get(idx)
            if uid:
                res[uid] = {"probs": probs_tensor[idx].numpy()}
        return res

    import torch
    import torch.nn.functional as F
    import numpy as np

    @torch.no_grad()
    def _prepare_single_target(self, target_qid, candidate_uids):
        """
        Refined: Generates pseudo-labels for candidate users on a specific query 
        by leveraging the prediction logic used in the global target preparation.
        """
        self.model.eval()
        device = self.device
        
        # 1. Use the existing prediction logic for consistency
        # This ensures that any logic in predict_for_question (like specific 
        # normalization or temperature scaling) is mirrored here.
        preds = self.predict_for_question(target_qid)
        
        # 2. Extract choice indices based on candidate_uids order
        # This aligns exactly with the logic in _prepare_global_targets
        pseudo_labels_list = [preds[uid]['choice_index'] for uid in candidate_uids]
        pseudo_labels = torch.tensor(pseudo_labels_list, device=device)
        
        # 3. Get the option node indices for the graph
        option_node_indices = torch.tensor(self.qid2choices[target_qid], device=device)
        
        return pseudo_labels, option_node_indices

    def _prepare_global_targets(self, all_qids, candidate_uids):
        all_labels = []
        all_opt_nodes = []
        for qid in all_qids:
            # get the current model's prediction for the question
            preds = self.predict_for_question(qid)
            # arrange pseudo-labels in the order of candidate_uids
            labels = [preds[uid]['choice_index'] for uid in candidate_uids]
            all_labels.append(labels)
            # get the option node indices for the question
            all_opt_nodes.append(self.qid2choices[qid])
            
        return (
            torch.tensor(all_labels, device=self.device), # [Q, U]
            torch.tensor(all_opt_nodes, device=self.device) # [Q, K]
        )

    def select_nodes_single_query_entropy(
        self, 
        target_qid, 
        candidate_uids, 
        k, 
        steps=100, 
        lr=0.2, 
        target_ratio=0.1, # suggested to set to the real ratio, e.g. 0.1
        save_path=None,
        diversity=False
    ):
        """
        Label-free subset selection based on Uncertainty (Entropy) reduction.
        find the nodes that can maximally reduce the uncertainty of the target question prediction.
        """
        self.model.eval()
        num_u = len(candidate_uids)
        target_device = self.device
        
        # --- 1. Preparation ---
        active_u_local_indices = [] 
        # get the option node indices for the target question
        option_node_indices = torch.tensor(self.qid2choices[target_qid], device=target_device)

        # only keep valid users in the graph
        for i, uid in enumerate(candidate_uids):
            if uid in self.uid2idx:
                active_u_local_indices.append(i)

        if not active_u_local_indices:
            return candidate_uids[:k]

        active_u_local_t = torch.tensor(active_u_local_indices, device=target_device)
        u_active_global_idx = torch.tensor([self.uid2idx[candidate_uids[i]] for i in active_u_local_indices], device=target_device)


        # --- 2. Build Candidate Graph View (Corrected for Entropy) ---
        all_candidate_edges = []
        u_indices_for_mapping = []
        
        # even in Entropy mode, use pseudo-labels to build the "virtual path"
        all_pseudo_labels, _ = self._prepare_single_target(target_qid, candidate_uids)
        
        for i, uid in enumerate(candidate_uids):
            if uid in self.uid2idx:
                u_global = self.uid2idx[uid]
                label_idx = all_pseudo_labels[i].item() # use the model's own prediction
                q_opt_node = option_node_indices[label_idx].item()
                
                # inject this "assumed" edge
                all_candidate_edges.append((u_global, target_qid, q_opt_node))
                u_indices_for_mapping.append(i)

        # the function call must match the real parameters signature of OpinionQAGraph
        # assume the function signature is (bg_edges, candidate_edges)
        data_mp, cand_mask, edge_to_u_t = self.graph.create_weighted_active_view(
            list(self.mp_edges), 
            all_candidate_edges,
            u_indices_for_mapping
        )
        
        data_mp = data_mp.to(target_device)
        cand_mask = cand_mask.to(target_device)
        edge_to_u_t = edge_to_u_t.to(target_device)

        # --- 3. Optimization Setup ---
        w_logits = torch.randn((num_u,), device=target_device) * 0.01
        w_logits.requires_grad = True
        optimizer = torch.optim.Adam([w_logits], lr=lr)
        
        base_weights = {etype: torch.ones(data_mp[etype].edge_index.size(1), device=target_device) 
                        for etype in data_mp.edge_types}

        # Logging setup
        log_file = None
        csv_writer = None
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            log_file = open(save_path, 'w', newline='')
            csv_writer = csv.writer(log_file)
            if diversity:
                csv_writer.writerow(['step', 'total_loss', 'proxy_loss', 'sparsity_loss', 'diversity_loss', 'w_mean', 'w_max'])
            else:
                csv_writer.writerow(['step', 'total_loss', 'proxy_loss', 'sparsity_loss', 'w_mean', 'w_max'])

        # --- 4. Optimization Loop ---
        for step in range(steps):
            optimizer.zero_grad()
            w = torch.sigmoid(w_logits) 
            
            weight_dict = {etype: wts.clone() for etype, wts in base_weights.items()}
            # apply the learned weights to the graph
            if edge_to_u_t.numel() > 0:
                weight_dict[('user', 'to', 'question')][cand_mask] = w[edge_to_u_t]

            # Forward pass
            z = self.model.forward_with_weight(data_mp, edge_weight_dict=weight_dict, return_z=True)
            
            # --- Entropy Calculation (The Core Change) ---
            zu_norm = F.normalize(z['user'][u_active_global_idx], p=2, dim=-1)
            zq_norm = F.normalize(z['question'][option_node_indices], p=2, dim=-1)
            
            # calculate the prediction distribution (Logits -> Softmax)
            logits = torch.matmul(zu_norm, zq_norm.t()) / 1.0 
            probs = F.softmax(logits, dim=-1)
            
            # calculate the entropy for each candidate: H(p) = -sum(p * log(p))
            # the higher the entropy, the more uncertain the model's view of the user
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            
            # Proxy Loss: the optimizer will raise w to reduce the remaining weights of these high-entropy nodes (1-w)
            p_loss = ((1.0 - w[active_u_local_t]) * entropy).mean()

            # Sparsity Loss (budget locking)
            current_ratio = torch.mean(w)
            s_loss = 10.0 * (current_ratio - target_ratio)**2

            if diversity:
                active_u_embeds_norm = F.normalize(z['user'][u_active_global_idx], p=2, dim=-1)
                sim_matrix = torch.matmul(active_u_embeds_norm, active_u_embeds_norm.t())
                w_active = w[active_u_local_t].view(-1, 1)
                d_loss = (w_active @ w_active.t() * sim_matrix).mean()
                total_loss = p_loss + s_loss + d_loss
            else:
                total_loss = p_loss + s_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([w_logits], 1.0)
            optimizer.step()

            if csv_writer:
                w_np = w.detach().cpu().numpy()
                if diversity:
                    csv_writer.writerow([step, total_loss.item(), p_loss.item(), s_loss.item(), d_loss.item(), np.mean(w_np), np.max(w_np)])
                else:   
                    csv_writer.writerow([step, total_loss.item(), p_loss.item(), s_loss.item(), np.mean(w_np), np.max(w_np)])


        # --- 5. Final Selection ---
        final_w = torch.sigmoid(w_logits).detach().cpu().numpy()
        selection_mask = np.zeros(num_u)
        selection_mask[active_u_local_indices] = 1.0
        final_w = final_w * selection_mask - (1.0 - selection_mask) 

        top_k_idx = np.argsort(final_w)[-k:]
        return [candidate_uids[i] for i in top_k_idx]

   
    def select_nodes_single_query(self, target_qid, candidate_uids, k, **kwargs):
        self.model.eval()
        num_u = len(candidate_uids)
        device = self.device
        
        # 1. extract active indices
        active_u_local_indices = []
        # add a check: whether there are any observations (to determine if it's a cold start)
  
        # all_pseudo_labels, _ = self._prepare_single_target(target_qid, candidate_uids)
        for i, uid in enumerate(candidate_uids):
            if uid in self.uid2idx:
                active_u_local_indices.append(i)
                
        if not active_u_local_indices:
            return candidate_uids[:k]

        u_global_idx = torch.tensor([self.uid2idx[candidate_uids[i]] for i in active_u_local_indices], device=device)
        option_nodes = torch.tensor(self.qid2choices[target_qid], device=device)

        # 2. extract features and uncertainty
        with torch.no_grad():
            data_mp, _, _ = self.graph.create_weighted_active_view(list(self.mp_edges), [], [])
            data_mp = data_mp.to(device)
            base_weights = {etype: torch.ones(data_mp[etype].edge_index.size(1), device=device) for etype in data_mp.edge_types}
            
            z = self.model.forward_with_weight(data_mp, edge_weight_dict=base_weights, return_z=True)
            zu = z['user'][u_global_idx] 
            
            # --- improvement 1: feature normalization ---
            # ensure K-Means measures semantic similarity rather than magnitude
            zu_normalized = F.normalize(zu, p=2, dim=-1)
            
          
            zq = z['question'][option_nodes]
            # use init_temp to scale logits
            logits = torch.matmul(zu_normalized, F.normalize(zq, p=2, dim=-1).t()) / kwargs.get('init_temp', 0.5)
            probs = F.softmax(logits, dim=-1)
            uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            # --- improvement 2: robust normalization ---
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max > u_min:
                uncertainty = (uncertainty - u_min) / (u_max - u_min)
            else:
                uncertainty = torch.zeros_like(uncertainty)

        # 1. cluster features: only preserve semantic geometry, no scaling
        features_np = zu_normalized.cpu().numpy()
        # 2. uncertainty as sample weight
        alpha = kwargs.get('init_scale', 0.1)   # control the bias strength
        sample_weight = (1.0 + alpha * uncertainty).detach().cpu().numpy()


        # 4. execute K-Means++
        k_to_select = min(k, len(active_u_local_indices))
        # add n_jobs or optimize, if N is very large
        # kmeans = KMeans(n_clusters=k_to_select, init='k-means++', n_init=10, random_state=42)
        # kmeans.fit(features_np)

        kmeans = KMeans(
            n_clusters=k_to_select,
            init='k-means++',
            n_init=10,
            random_state=42)
        kmeans.fit(features_np, sample_weight=sample_weight)


        # 5. sample mapping
        top_k_active_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features_np)
        
        final_selected_local_idx = [active_u_local_indices[i] for i in top_k_active_idx]
        selected_uids = [candidate_uids[i] for i in final_selected_local_idx]

        # fill in the logic
        if len(selected_uids) < k:
            # avoid duplicate selection
            existing = set(selected_uids)
            for i in range(num_u):
                if len(selected_uids) >= k: break
                uid = candidate_uids[i]
                if uid not in existing:
                    selected_uids.append(uid)
                    existing.add(uid)

        return (selected_uids, {'method': 'Discrete-Coreset'}) if kwargs.get('return_metrics') else selected_uids


