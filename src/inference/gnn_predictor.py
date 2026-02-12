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

    - 只构建 test user 图（demo_ood_dist_test + question_ood_dist_test）
    - 只在 test user graph 上做 message passing
    - 对给定 qid，返回所有 test user 的预测
    - 每轮可以把新的 (uid, qid) observation 加入 mp edges
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

        # uid 映射
        self.uid2idx: Dict[str, int] = graph.uid2idx
        self.idx2uid: Dict[int, str] = {v: k for k, v in self.uid2idx.items()}

        # === 【修复】添加 qid 映射 ===
        # 假设 graph 对象里有这个属性 (通常 graph.uid2idx 存在时 graph.qid2idx 也存在)
        if hasattr(graph, 'qid2idx'):
            self.qid2idx: Dict[str, int] = graph.qid2idx
        else:
            # 如果 graph 没有现成的，尝试从 node_ids 构建
            # 假设 graph.question_nodes 或者是类似结构
            # 这是一个防御性写法
            pass 

        self.qid2choices = graph.qid2choices

        # 当前 “已观察” 的 message-passing edges (只在 test graph 里)
        self.mp_edges: set[EdgeTriple] = set()

        # 方便从 (user_idx, qid) 找到唯一的 edge triple
        self._edge_by_uq: Dict[Tuple[int, str], EdgeTriple] = {}
        for u_idx, qid, qnode_idx in self.edges_u_to_qopt:
            self._edge_by_uq[(u_idx, qid)] = (u_idx, qid, qnode_idx)

        # === NEW: 记录真实观测到的答案 ===
        # user_history[uid] = [(qid, ans_idx), ...]
        # ans_idx 是选项的 index（0-based），对应 A/B/C/...，用于 eval 时覆盖预测。
        self.user_history: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.original_mp_edges = list(self.mp_edges).copy()
       
    # ------------------------------------------------------------------
    # 1) 构造：从 config 直接构建 test-only graph + model + ckpt
    # ------------------------------------------------------------------
    def reset_graph(self):
        """
        重置图状态，撤销所有通过 add_observations 添加的观测。
        """
        # 1. 恢复 mp_edges 到初始备份状态
        # 因为你的 original_mp_edges 在 __init__ 时是空的，这会清空所有动态添加的边
        self.mp_edges = set(self.original_mp_edges)

        # 2. 清空用户历史（如果你的预测逻辑会用到 user_history 覆盖预测值）
        if hasattr(self, 'user_history'):
            self.user_history.clear()

        # 3. 【关键】同步重置 Graph 对象内部的边和缓存
        # 确保 GNN 在下一次 forward 时识别到边的变化
        if hasattr(self.graph, 'mp_edges'):
            self.graph.mp_edges = self.mp_edges
            
        # 如果 Graph 类中有缓存 Data 对象的机制，必须清除
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

        # ---- 1. 只加载 OOD test panel 的 demo / question ----
        demo_test_path = os.path.join(base_dir, f"demo_{ood_dist}_test.csv")
        q_test_path = os.path.join(base_dir, f"question_{ood_dist}_test.csv")

        df_demo_test = pd.read_csv(demo_test_path)
        df_question_test = pd.read_csv(q_test_path)

        # ---- 2. 用 test-only DataFrame 构图（S, C 仍然是全量）----
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

        # ---- 3. 构建 GNN 模型 & 加载 full-graph 训练好的 checkpoint ----
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
    # 2) 更新观测：把新的 (uid, qid) 变成可见 mp edges
    # ------------------------------------------------------------------
    def add_observations(
        self,
        obs_list: Iterable[Tuple[str, str]],
        *,
        raw_uid: bool = True,
    ):
        """
        将新的 observation 加入 test graph 的 message-passing edges。

        参数：
            obs_list: 可迭代的 (uid, qid)
              - uid: 默认是原始 caseid（字符串）；如果 raw_uid=False，则视为内部 user_idx
              - qid: question id（如 "v161"）

        只会添加 test graph 里存在的 edges；不在 test 的 (uid, qid) 会被跳过。
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
        """清空所有已观察的 mp edges，回到 strict cold-start。"""
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
        记录一次真实的用户答案，用于之后 eval 时覆盖 GNN 预测。

        参数：
            uid: 原始 user id（caseid）
            qid: question id，比如 "v161"
            ans_idx: 选项 index（0-based），0->A, 1->B, ...
            allow_duplicate:
                - False: 如果该 uid 对同一个 qid 已经有记录，则直接覆盖旧值
                - True: 允许同一个 (uid, qid) 多次出现，按追加处理
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
        返回该 uid 对 qid 是否有已记录的答案：
            - 如果有，返回 ans_idx (0-based)
            - 如果没有，返回 None
        """
        for q, ans_idx in self.user_history.get(uid, []):
            if q == qid:
                return ans_idx
        return None

    def summarize_user_history(self):
        """
        简单打印当前 user_history 中：
            - 每个 question 出现多少条记录
        （方便 debug）
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
        使用当前 GNN 预测，对指定 question 的高置信度预测作为“伪观测”加入 mp_edges。

        逻辑：
            1. 调用 predict_for_question(qid) 得到所有 test user 的预测结果；
            2. 对每个 user:
                - 如果提供了 uids，则只在该子集上做 imputations；
                - 取 max prob >= min_conf 的样本；
                - 将 (user_idx, qid, 预测选项对应的 qopt_node_idx) 作为新的 mp edge 加入；
                - 如果该 (user, qid) 已经在 mp_edges 里:
                    - overwrite=False: 跳过
                    - overwrite=True: 删除原有 edge，再插入新的 imputed edge

        参数：
            qid:  要做 imputation 的 question id
            min_conf:  置信度阈值（max prob >= min_conf 才加入）
            uids:  限制只对这些 uid 做 imputation；若为 None 则对所有 test user
            overwrite: 是否覆盖已有 (user, qid) 的 mp edge

        返回：
            实际新增 / 覆盖加入的 mp edges 数量
        """
        # 先在当前 mp_edges 状态下做一次预测
        preds = self.predict_for_question(qid)
        if not preds:
            print(f"[TestPredictor] add_imputations: no preds for qid={qid}, nothing added.")
            return 0

        # 如果只对部分 user 做 imputation，先构建一个 set 方便过滤
        uid_allow: Optional[set] = None
        if uids is not None:
            uid_allow = set(map(str, uids))

        added = 0

        # 为了避免重复 edge，先把现有 mp_edges 中 (u_idx, qid) 的组合收集出来
        existing_uq = set((u_idx, q) for (u_idx, q, _) in self.mp_edges)

        for uid, info in preds.items():
            if uid_allow is not None and uid not in uid_allow:
                continue

            probs = np.asarray(info["probs"], dtype=float)
            max_p = float(probs.max())
            if max_p < min_conf:
                # 不够自信，不加
                continue

            # 预测选项对应的 question 节点 idx（来自 predict_for_question）
            qopt_node_idx = int(info["qopt_node_idx"])

            # user 内部 idx
            if uid not in self.uid2idx:
                # 理论上不会发生，因为 preds 里的 uid 来自 idx2uid
                continue
            u_idx = self.uid2idx[uid]

            key_uq = (u_idx, qid)

            if key_uq in existing_uq:
                if not overwrite:
                    # 已有 edge 且不覆盖，跳过
                    continue
                else:
                    # 覆盖：先把原来的 (u_idx, qid, *) 删掉
                    to_remove = [e for e in self.mp_edges if e[0] == u_idx and e[1] == qid]
                    for e in to_remove:
                        self.mp_edges.discard(e)
                    existing_uq.discard(key_uq)

            # 新的 imputed edge
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
    # 3) 核心：给定 qid，对所有 test user 做一次 GNN 预测
    # ------------------------------------------------------------------
    def predict_for_question(
        self,
        qid: str,
        batch_size: int = 256,
    ) -> Dict[str, Dict[str, object]]:
        """
        在当前 mp_edges 状态下，对 test graph 中所有 user 的
        指定 question (qid) 做一次 GNN 预测。

        返回:
            preds[uid] = {
                "probs": List[float],        # [p_0, ..., p_{K-1}]
                "choice_index": int,         # argmax index
                "choice_letter": str,        # 'A' / 'B' / ...
                "qopt_node_idx": int,        # 选中选项对应的 question 节点 idx
            }
        """
        self.model.eval()

        # 1) 所有 test edges 里，该 qid 的 supervision edges
        sup_edges_q: List[EdgeTriple] = [
            e for e in self.edges_u_to_qopt if e[1] == qid
        ]
        if not sup_edges_q:
            print(f"[TestPredictor] WARNING: no TEST edges found for qid={qid}")
            return {}

        # 2) 当前 test mp edges（全部已观察的 (u, q)）
        mp_edges = list(self.mp_edges)

        # 3) 构造一个只有 'test' 的 epoch_spec，交给原来的 build_loaders_for_epoch
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
                # 训练模式：有 gold_idx -> (loss, acc, probs)
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
        轻量级克隆：
        - 共享 graph / model / 映射（uid2idx, qid2choices, edges_u_to_qopt）
        - 拷贝一份独立的 mp_edges / user_history（每个 worker 自己改自己的）
        """
        new = self.__class__.__new__(self.__class__)

        # 共享的只读结构
        new.graph = self.graph
        new.edges_u_to_qopt = self.edges_u_to_qopt
        new.model = self.model       # 共享同一个 GNN 模型实例（只 forward）
        new.device = self.device

        new.uid2idx = self.uid2idx
        new.idx2uid = self.idx2uid
        new.qid2choices = self.qid2choices
        new._edge_by_uq = self._edge_by_uq

        # 状态：每个 clone 都有独立的一份
        new.mp_edges = self.mp_edges.copy()

        # === NEW: user_history 深拷贝，保持当前观测一致，但后续各自独立更新 ===
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
            # 获取当前模型对该问题的预测
            preds = self.predict_for_question(qid)
            # 按 candidate_uids 顺序排列伪标签
            labels = [preds[uid]['choice_index'] for uid in candidate_uids]
            all_labels.append(labels)
            # 获取该问题的选项节点 ID
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
        target_ratio=0.1, # 建议设为真实比例，如 0.1
        save_path=None,
        diversity=False
    ):
        """
        Label-free subset selection based on Uncertainty (Entropy) reduction.
        寻找那些能最大程度降低目标问题预测不确定性的节点。
        """
        self.model.eval()
        num_u = len(candidate_uids)
        target_device = self.device
        
        # --- 1. Preparation ---
        active_u_local_indices = [] 
        # 获取目标问题的选项节点索引
        option_node_indices = torch.tensor(self.qid2choices[target_qid], device=target_device)

        # 仅保留在图中存在的合法用户
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
        
        # 即使是 Entropy 模式，也要用伪标签来建立“虚拟路径”
        all_pseudo_labels, _ = self._prepare_single_target(target_qid, candidate_uids)
        
        for i, uid in enumerate(candidate_uids):
            if uid in self.uid2idx:
                u_global = self.uid2idx[uid]
                label_idx = all_pseudo_labels[i].item() # 使用模型自己的预测
                q_opt_node = option_node_indices[label_idx].item()
                
                # 注入这条“假设存在”的边
                all_candidate_edges.append((u_global, target_qid, q_opt_node))
                u_indices_for_mapping.append(i)

        # 这里的函数调用必须匹配你 OpinionQAGraph 的真实参数签名
        # 假设你的函数签名是 (bg_edges, candidate_edges)
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
            # 将学习到的权重应用到图中
            if edge_to_u_t.numel() > 0:
                weight_dict[('user', 'to', 'question')][cand_mask] = w[edge_to_u_t]

            # Forward pass
            z = self.model.forward_with_weight(data_mp, edge_weight_dict=weight_dict, return_z=True)
            
            # --- Entropy Calculation (The Core Change) ---
            zu_norm = F.normalize(z['user'][u_active_global_idx], p=2, dim=-1)
            zq_norm = F.normalize(z['question'][option_node_indices], p=2, dim=-1)
            
            # 计算预测分布 (Logits -> Softmax)
            logits = torch.matmul(zu_norm, zq_norm.t()) / 1.0 
            probs = F.softmax(logits, dim=-1)
            
            # 计算每个候选人的预测熵: H(p) = -sum(p * log(p))
            # 熵越高，表示模型对该用户的观点越不确定
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            
            # Proxy Loss: 优化器会调高 w 来降低这些高熵节点的剩余权重 (1-w)
            # 换句话说：w 越大，表示我们越倾向于“观测”这个高不确定性的点
            p_loss = ((1.0 - w[active_u_local_t]) * entropy).mean()

            # Sparsity Loss (Budget Locking)
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

    # def select_nodes_single_query(
    #     self, 
    #     target_qid, 
    #     candidate_uids, 
    #     k, 
    #     steps=50, # opinionQA: 500, ces: 100
    #     lr=0.2, 
    #     target_ratio=1,
    #     use_gt=False,
    #     gt_labels=None,
    #     save_path=None,
    #     diversity=False
    # ):
    #     """
    #     Differentiable subset selection for a single query using GNN weight optimization.
    #     Refined for ORCA Lab research on Opinion Modeling and Adaptive Elicitation.
    #     """
    #     self.model.eval()
    #     num_u = len(candidate_uids)
    #     target_device = self.device
        
    #     # --- 1. Label & Index Preparation ---
    #     active_u_local_indices = [] 
    #     label_indices = []          
        
    #     # Get the option node IDs for this specific target question
    #     option_node_indices = torch.tensor(self.qid2choices[target_qid], device=target_device)
    #     num_options = len(option_node_indices)

    #     if use_gt:
    #         # Scenario A: Ground Truth (for upper-bound analysis)
    #         for i, uid in enumerate(candidate_uids):
    #             if uid in self.uid2idx and uid in gt_labels and target_qid in gt_labels[uid]:
    #                 gt_val = gt_labels[uid][target_qid]
    #                 try:
    #                     idx = ord(gt_val.upper()) - ord('A')
    #                     if 0 <= idx < num_options:
    #                         active_u_local_indices.append(i)
    #                         label_indices.append(idx)
    #                 except: continue
    #     else:
    #         # Scenario B: Batch Pseudo-labels (The core research method)
    #         all_pseudo_labels, _ = self._prepare_single_target(target_qid, candidate_uids)
    #         for i, uid in enumerate(candidate_uids):
    #             if uid in self.uid2idx:
    #                 active_u_local_indices.append(i)
    #                 label_indices.append(all_pseudo_labels[i].item())

    #     if not active_u_local_indices:
    #         print(f"[WARNING] No valid users found for target {target_qid}. Returning random top-k.")
    #         return candidate_uids[:k]

    #     # Pre-move tensors to GPU to avoid overhead in the loop
    #     active_u_local_t = torch.tensor(active_u_local_indices, device=target_device)
    #     target_labels = torch.tensor(label_indices, device=target_device)
    #     u_active_global_idx = torch.tensor([self.uid2idx[candidate_uids[i]] for i in active_u_local_indices], device=target_device)

    #     # --- 2. Build Candidate Graph View (Injection Logic) ---
    #     all_candidate_edges = []
    #     u_indices_for_mapping = []
        
    #     for i, label_idx in enumerate(label_indices):
    #         u_local_idx = active_u_local_indices[i]
    #         u_global = u_active_global_idx[i].item()
    #         q_opt_node = option_node_indices[label_idx].item()
            
    #         all_candidate_edges.append((u_global, target_qid, q_opt_node))
    #         u_indices_for_mapping.append(u_local_idx)

    #     # create_weighted_active_view injects the candidate edges into the graph
    #     data_mp, cand_mask, edge_to_u_t = self.graph.create_weighted_active_view(
    #         list(self.mp_edges), 
    #         all_candidate_edges,
    #         u_indices_for_mapping
    #     )
        
    #     # Ensure view data is on target device
    #     data_mp = data_mp.to(target_device)
    #     cand_mask = cand_mask.to(target_device)
    #     edge_to_u_t = edge_to_u_t.to(target_device)

    #     # --- 3. Optimization Setup ---
    #     # Random small initialization prevents sigmoid saturation issues
    #     w_logits = torch.randn((num_u,), device=target_device) * 0.01
    #     w_logits.requires_grad = True
    #     optimizer = torch.optim.Adam([w_logits], lr=lr)
        
    #     # Pre-populate base weights to prevent device mismatch in model.py's torch.cat()
    #     base_weights = {}
    #     for etype in data_mp.edge_types:
    #         num_edges = data_mp[etype].edge_index.size(1)
    #         base_weights[etype] = torch.ones(num_edges, device=target_device)

    #     # Logging setup
    #     log_file = None
    #     csv_writer = None
    #     if save_path:
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         log_file = open(save_path, 'w', newline='')
    #         csv_writer = csv.writer(log_file)
    #         if diversity:
    #             csv_writer.writerow(['step', 'total_loss', 'proxy_loss', 'sparsity_loss', 'diversity_loss', 'w_mean', 'w_max'])
    #         else:
    #             csv_writer.writerow(['step', 'total_loss', 'proxy_loss', 'sparsity_loss', 'w_mean', 'w_max'])

    #     # --- 4. Optimization Loop ---
    #     for step in range(steps):
    #         optimizer.zero_grad()
    #         w = torch.sigmoid(w_logits) 
            
    #         # Construct weight_dict for current step
    #         weight_dict = {etype: wts.clone() for etype, wts in base_weights.items()}
    #         current_w = w[edge_to_u_t]
    #         weight_dict[('user', 'to', 'question')][cand_mask] = current_w
            
    #         if ('question', 'rev_to', 'user') in weight_dict:
    #             weight_dict[('question', 'rev_to', 'user')][cand_mask] = current_w

    #         # Forward pass: GNN message passing incorporates learnable edge weights
    #         z = self.model.forward_with_weight(data_mp, edge_weight_dict=weight_dict, return_z=True)
            
    #         # Calculate Alignment Loss
    #         zu_norm = F.normalize(z['user'][u_active_global_idx], p=2, dim=-1)
    #         zq_norm = F.normalize(z['question'][option_node_indices], p=2, dim=-1)
            
    #         # Using 0.1 temperature for sharp alignment
    #         logits = torch.matmul(zu_norm, zq_norm.t()) / 1.0 
    #         ind_loss = F.cross_entropy(logits, target_labels, reduction='none') 

    #         # Proxy Loss: Optimizer picks w->1 for high-loss (high-information) nodes
    #         if use_gt:
    #             p_loss = ((1.0 - w[active_u_local_t]) * ind_loss).mean()
    #         else:
    #             # norm_ind_loss = ind_loss / (ind_loss.mean() + 1e-8)
    #             norm_ind_loss = ind_loss
    #             p_loss = ((1.0 - w[active_u_local_t]) * norm_ind_loss).mean()
    #         # Sparsity Loss: Forces the optimizer to be selective
    #         # s_loss = lmbda * torch.mean(w) 
    #         current_ratio = torch.mean(w)
    #         s_loss = 10.0 * (current_ratio - target_ratio)**2 

    #         if diversity:
    #             # print("Diversity Loss: Forces the optimizer to be diverse")
    #             # 1. 归一化，确保相似度在 [-1, 1]
    #             active_u_embeds = z['user'][u_active_global_idx]  # [num_candidates, dim]
    #             active_u_embeds_norm = F.normalize(active_u_embeds, p=2, dim=-1)
    #             # 2. 计算相似度矩阵 S
    #             sim_matrix = torch.matmul(active_u_embeds_norm, active_u_embeds_norm.t())
    #             # 3. 施加惩罚：Sum(w_i * w_j * Sim_ij)
    #             w_active = w[active_u_local_t].view(-1, 1)
    #             d_loss =  1.0 * (w_active @ w_active.t() * sim_matrix).mean() 
    #             total_loss = p_loss + s_loss + d_loss
    #         else:
    #             total_loss = p_loss + s_loss
            
    #         total_loss.backward()
            
    #         # Gradient clipping for stability
    #         torch.nn.utils.clip_grad_norm_([w_logits], 1.0)
    #         optimizer.step()
            
    #         if csv_writer:
    #             w_np = w.detach().cpu().numpy()
    #             if diversity:
    #                 csv_writer.writerow([step, total_loss.item(), p_loss.item(), s_loss.item(), d_loss.item(), np.mean(w_np), np.max(w_np)])
    #             else:   
    #                 csv_writer.writerow([step, total_loss.item(), p_loss.item(), s_loss.item(), np.mean(w_np), np.max(w_np)])

    #     if log_file:
    #         log_file.close()

    #     # --- 5. Final Selection ---
    #     final_w = torch.sigmoid(w_logits).detach().cpu().numpy()
        
    #     # Mask out invalid users to prevent them from entering top-k
    #     selection_mask = np.zeros(num_u)
    #     selection_mask[active_u_local_indices] = 1.0
    #     final_w = final_w * selection_mask - (1.0 - selection_mask) 

    #     top_k_idx = np.argsort(final_w)[-k:]
    #     return [candidate_uids[i] for i in top_k_idx]

    # def select_nodes_single_query(
    #     self, 
    #     target_qid, 
    #     candidate_uids, 
    #     k, 
    #     # --- Optimization Hyperparameters ---
    #     steps=5000,             
    #     lr=0.1,               
    #     target_ratio=0.1,     
    #     init_temp=0.5,         
    #     min_temp=0.05,        
    #     sparsity_weight=20,   # Optimized based on S0.1_D1 results, 2
    #     diversity_weight=0.5, # used to be 1
    #     grad_clip=1.0,        
    #     init_scale=0.001,      
    #     # --- Early Stopping Parameters ---
    #     entropy_threshold=0.15, # Threshold for decision certainty
    #     budget_tolerance=0.01,  # Threshold for budget compliance
    #     min_steps=100,          # Minimum steps to allow for annealing phase
    #     # --- Running Config ---
    #     use_gt=False,         
    #     gt_labels=None,       
    #     diversity=True,       
    #     return_metrics=False
    # ):
    #     """
    #     GNN weight optimization with Logit Bias initialization, Prior-based warm start,
    #     and Early Stopping based on entropy and budget error.
    #     """
    #     self.model.eval()
    #     num_u = len(candidate_uids)
    #     device = self.device
        
    #     # 1. DATA PREPARATION
    #     active_u_local_indices, label_indices = [], []          
    #     option_nodes = torch.tensor(self.qid2choices[target_qid], device=device)
    #     num_options = len(option_nodes)

    #     if use_gt:
    #         for i, uid in enumerate(candidate_uids):
    #             if uid in self.uid2idx and uid in gt_labels and target_qid in gt_labels[uid]:
    #                 try:
    #                     idx = ord(gt_labels[uid][target_qid].upper()) - ord('A')
    #                     if 0 <= idx < num_options:
    #                         active_u_local_indices.append(i); label_indices.append(idx)
    #                 except: continue
    #     else:
    #         all_pseudo_labels, _ = self._prepare_single_target(target_qid, candidate_uids)
    #         for i, uid in enumerate(candidate_uids):
    #             if uid in self.uid2idx:
    #                 active_u_local_indices.append(i); label_indices.append(all_pseudo_labels[i].item())

    #     if not active_u_local_indices:
    #         return (candidate_uids[:k], {}) if return_metrics else candidate_uids[:k]

    #     active_u_local_t = torch.tensor(active_u_local_indices, device=device)
    #     target_labels = torch.tensor(label_indices, device=device)
    #     u_global_idx = torch.tensor([self.uid2idx[candidate_uids[i]] for i in active_u_local_indices], device=device)

    #     # 2. GRAPH INJECTION
    #     candidate_edges, u_mapping = [], []
    #     for i, label_idx in enumerate(label_indices):
    #         candidate_edges.append((u_global_idx[i].item(), target_qid, option_nodes[label_idx].item()))
    #         u_mapping.append(active_u_local_indices[i])

    #     data_mp, cand_mask, edge_to_u_t = self.graph.create_weighted_active_view(
    #         list(self.mp_edges), candidate_edges, u_mapping
    #     )
    #     data_mp, cand_mask, edge_to_u_t = data_mp.to(device), cand_mask.to(device).bool(), edge_to_u_t.to(device)
    #     base_weights = {etype: torch.ones(data_mp[etype].edge_index.size(1), device=device) for etype in data_mp.edge_types}

    #     # 3. OPTIMIZATION SETUP (Warm Start & Logit Bias)
    #     with torch.no_grad():
    #         init_z = self.model.forward_with_weight(data_mp, edge_weight_dict=base_weights, return_z=True)
    #         zu_init = F.normalize(init_z['user'][u_global_idx], p=2, dim=-1)
    #         zq_init = F.normalize(init_z['question'][option_nodes], p=2, dim=-1)
    #         init_logits = torch.matmul(zu_init, zq_init.t()) / init_temp
    #         initial_ind_loss = F.cross_entropy(init_logits, target_labels, reduction='none')
    #         # Standardize individual loss to use as a prior perturbation
    #         prior = (initial_ind_loss - initial_ind_loss.mean()) / (initial_ind_loss.std() + 1e-6)

    #     # Initialize weights at the budget line: sigmoid(bias) = target_ratio
    #     initial_bias = torch.log(torch.tensor(target_ratio / (1 - target_ratio + 1e-6)))
    #     w_logits = torch.full((num_u,), initial_bias.item(), device=device)
    #     w_logits[active_u_local_t] += prior * init_scale # Inject prior knowledge
    #     w_logits += torch.randn((num_u,), device=device) * 1e-4 # Tiny noise for symmetry breaking
        
    #     w_logits.requires_grad = True
    #     optimizer = torch.optim.Adam([w_logits], lr=lr)
    #     history = {'total_loss': [], 'p_loss': [], 's_loss': [], 'd_loss': [], 'entropy': []}

    #     # 4. OPTIMIZATION LOOP
    #     final_step = steps
    #     for step in range(steps):
    #         optimizer.zero_grad()
    #         fraction = step / float(steps - 1) if steps > 1 else 0
    #         current_temp = init_temp * ((min_temp / init_temp) ** fraction) # Exponential annealing
            
    #         w = torch.sigmoid(w_logits) 
    #         weight_dict = {etype: wts.clone() for etype, wts in base_weights.items()}
    #         curr_w = w[edge_to_u_t]
    #         weight_dict[('user', 'to', 'question')][cand_mask] = curr_w
    #         if ('question', 'rev_to', 'user') in weight_dict:
    #             weight_dict[('question', 'rev_to', 'user')][cand_mask] = curr_w

    #         z = self.model.forward_with_weight(data_mp, edge_weight_dict=weight_dict, return_z=True)
            
    #         # --- LOSS A: Alignment (Normalized for stability) ---
    #         zu_norm = F.normalize(z['user'][u_global_idx], p=2, dim=-1)
    #         zq_norm = F.normalize(z['question'][option_nodes], p=2, dim=-1)
    #         logits = torch.matmul(zu_norm, zq_norm.t()) / current_temp
    #         ind_loss = F.cross_entropy(logits, target_labels, reduction='none') 
    #         norm_factor = ind_loss.detach().mean() + 1e-6
    #         p_loss = ((1.0 - w[active_u_local_t]) * (ind_loss / norm_factor)).mean()
           
    #         # --- LOSS B: Sparsity (Relative L1) ---
    #         # s_loss = sparsity_weight * torch.abs(torch.mean(w) - target_ratio) / (target_ratio + 1e-6)
    #         s_loss = sparsity_weight * (torch.mean(w) - target_ratio) ** 2


    #         # --- LOSS C: Centroid-based diversity (outlier coverage) ---
    #         d_loss = torch.tensor(0.0, device=device)
    #         if diversity:
    #             w_act = w[active_u_local_t]  # (n_active,)
    #             # population centroid in embedding space (detach to avoid representation hacking)
    #             c = F.normalize(zu_norm.mean(dim=0, keepdim=True).detach(), p=2, dim=-1)  # (1, d)
    #             # distance to centroid: in [0, 2] roughly; larger => more "non-mainstream"
    #             dist = 1.0 - (zu_norm * c).sum(dim=-1)  # (n_active,)
    #             # normalize to keep scale stable across batches
    #             dist = dist / (dist.mean().detach() + 1e-6)
    #             # encourage selecting far-from-centroid users
    #             # NOTE: negative sign => minimizing loss increases selected dist
    #             d_loss = -diversity_weight * (w_act * dist.detach()).mean()

              
                            
    #         total_loss = p_loss + s_loss + d_loss

    #         print(f"p_loss: {p_loss.item()}, s_loss: {s_loss.item()}, d_loss: {d_loss.item()}")
    #         # --- EARLY STOPPING CHECK ---
    #         if step >= min_steps and step % 10 == 0:
    #             with torch.no_grad():
    #                 w_clamped = torch.clamp(w, 1e-6, 1-1e-6)
    #                 curr_entropy = -torch.mean(w_clamped * torch.log(w_clamped) + (1-w_clamped) * torch.log(1-w_clamped)).item()
    #                 curr_error = torch.abs(torch.mean(w) - target_ratio).item()
    #                 if curr_entropy < entropy_threshold and curr_error < budget_tolerance:
    #                     final_step = step
    #                     break

    #         if return_metrics:
    #             history['total_loss'].append(total_loss.item())
    #             history['p_loss'].append(p_loss.item())
    #             history['s_loss'].append(s_loss.item())
    #             history['d_loss'].append(d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss)
    #             history['entropy'].append(curr_entropy if 'curr_entropy' in locals() else 0.69)
                
    #         total_loss.backward()
    #         torch.nn.utils.clip_grad_norm_([w_logits], grad_clip)
    #         optimizer.step()

    #     with torch.no_grad():
    #         final_w = torch.sigmoid(w_logits)

    #         w_act = final_w[active_u_local_t]
    #         print("active: mean/min/max =", float(w_act.mean()), float(w_act.min()), float(w_act.max()))
    #         print("active: top10 =", torch.topk(w_act, k=min(10, w_act.numel())).values.tolist())
    #         print("active: bot10 =", torch.topk(-w_act, k=min(10, w_act.numel())).values.neg().tolist())


    #     # 5. FINAL SELECTION
    #     with torch.no_grad():
    #         final_w = torch.sigmoid(w_logits)
    #         w_clamped = torch.clamp(final_w, 1e-6, 1-1e-6)
    #         entropy = -torch.mean(w_clamped * torch.log(w_clamped) + (1-w_clamped) * torch.log(1-w_clamped)).item()
    #         # metrics = {'final_w_mean': final_w.mean().item(), 'final_entropy': entropy, 'steps': final_step, 'history': history}
    #         metrics = {'final_w_mean': final_w.mean().item(), 'final_entropy': entropy, 'steps': final_step}
            
    #         final_w_np = final_w.cpu().numpy()
    #         mask = np.zeros(num_u); mask[active_u_local_indices] = 1.0
    #         final_w_np = final_w_np * mask - (1.0 - mask) * 1e6
    #         top_k_idx = np.argsort(final_w_np)[-k:]
            
    #         pairs = [
    #             (candidate_uids[i], float(final_w[i]))
    #             for i in top_k_idx
    #         ]

    #         pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    #         print("=== Top 10 ===")
    #         print(pairs_sorted[:10])

    #         print("=== Bottom 10 ===")
    #         print(pairs_sorted[-10:])
    #         print(metrics)
    #         print("mean(w)=", float(w.mean().item()), "target=", target_ratio)
    #         print("mean(w_active)=", float(w[active_u_local_t].mean().item()))

    #         input()

                        
    #     return ([candidate_uids[i] for i in top_k_idx], metrics) if return_metrics else [candidate_uids[i] for i in top_k_idx]

   
   

    

    def select_nodes_single_query(self, target_qid, candidate_uids, k, **kwargs):
        self.model.eval()
        num_u = len(candidate_uids)
        device = self.device
        
        # 1. 提取活跃索引
        active_u_local_indices = []
        # 增加一个判断：是否已经有观测数据（判定是否为冷启动）
  
        # all_pseudo_labels, _ = self._prepare_single_target(target_qid, candidate_uids)
        for i, uid in enumerate(candidate_uids):
            if uid in self.uid2idx:
                active_u_local_indices.append(i)
                
        if not active_u_local_indices:
            return candidate_uids[:k]

        u_global_idx = torch.tensor([self.uid2idx[candidate_uids[i]] for i in active_u_local_indices], device=device)
        option_nodes = torch.tensor(self.qid2choices[target_qid], device=device)

        # 2. 提取特征与不确定性
        with torch.no_grad():
            data_mp, _, _ = self.graph.create_weighted_active_view(list(self.mp_edges), [], [])
            data_mp = data_mp.to(device)
            base_weights = {etype: torch.ones(data_mp[etype].edge_index.size(1), device=device) for etype in data_mp.edge_types}
            
            z = self.model.forward_with_weight(data_mp, edge_weight_dict=base_weights, return_z=True)
            zu = z['user'][u_global_idx] 
            
            # --- 改进 1: 表征归一化 ---
            # 确保 K-Means 衡量的是方向（语义）相似性而非模长
            zu_normalized = F.normalize(zu, p=2, dim=-1)
            
          
            zq = z['question'][option_nodes]
            # 使用 init_temp 缩放 logits
            logits = torch.matmul(zu_normalized, F.normalize(zq, p=2, dim=-1).t()) / kwargs.get('init_temp', 0.5)
            probs = F.softmax(logits, dim=-1)
            uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            # --- 改进 2: 稳健归一化 ---
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max > u_min:
                uncertainty = (uncertainty - u_min) / (u_max - u_min)
            else:
                uncertainty = torch.zeros_like(uncertainty)

        # 3. 构造聚类特征
        # 结合 Embedding (空间覆盖) 和 Uncertainty (难点优先)
        # 使用 1.0 + alpha * uncertainty 来控制难点的权重
        # alpha = kwargs.get('init_scale', 0.1) 
        # features = zu_normalized * (1.0 + alpha * uncertainty.view(-1, 1))
        # features_np = features.cpu().numpy()

        # 1. 聚类特征：只保留语义几何，不再缩放
        features_np = zu_normalized.cpu().numpy()
        # 2. 不确定性作为 sample weight
        alpha = kwargs.get('init_scale', 0.1)   # 控制偏置强度
        sample_weight = (1.0 + alpha * uncertainty).detach().cpu().numpy()


        # 4. 执行 K-Means++
        k_to_select = min(k, len(active_u_local_indices))
        # 增加 n_jobs 或优化，如果 N 非常大
        # kmeans = KMeans(n_clusters=k_to_select, init='k-means++', n_init=10, random_state=42)
        # kmeans.fit(features_np)

        kmeans = KMeans(
            n_clusters=k_to_select,
            init='k-means++',
            n_init=10,
            random_state=42)
        kmeans.fit(features_np, sample_weight=sample_weight)


        # 5. 样本映射
        top_k_active_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features_np)
        
        final_selected_local_idx = [active_u_local_indices[i] for i in top_k_active_idx]
        selected_uids = [candidate_uids[i] for i in final_selected_local_idx]

        # 补齐逻辑
        if len(selected_uids) < k:
            # 避免重复选择
            existing = set(selected_uids)
            for i in range(num_u):
                if len(selected_uids) >= k: break
                uid = candidate_uids[i]
                if uid not in existing:
                    selected_uids.append(uid)
                    existing.add(uid)

        return (selected_uids, {'method': 'Discrete-Coreset'}) if kwargs.get('return_metrics') else selected_uids



    def select_nodes_clue(self, target_qid, candidate_uids, k, **kwargs):
        self.model.eval()
        device = self.device
        num_u = len(candidate_uids)

        # NEW: switch
        use_uncertainty = bool(kwargs.get("use_uncertainty", True))  # default on

        # 1) active candidates (present in graph)
        active_u_local_indices = [i for i, uid in enumerate(candidate_uids) if uid in self.uid2idx]
        if not active_u_local_indices:
            return candidate_uids[:k]

        k_to_select = min(k, len(active_u_local_indices))

        u_global_idx = torch.tensor(
            [self.uid2idx[candidate_uids[i]] for i in active_u_local_indices],
            device=device
        )
        option_nodes = torch.tensor(self.qid2choices[target_qid], device=device)

        with torch.no_grad():
            # 2) build current mp view (no injected candidate edges)
            data_mp, _, _ = self.graph.create_weighted_active_view(list(self.mp_edges), [], [])
            data_mp = data_mp.to(device)
            base_weights = {
                etype: torch.ones(data_mp[etype].edge_index.size(1), device=device)
                for etype in data_mp.edge_types
            }

            z = self.model.forward_with_weight(data_mp, edge_weight_dict=base_weights, return_z=True)

            zu = z["user"][u_global_idx]
            zu = F.normalize(zu, p=2, dim=-1)

            # sample weights
            if use_uncertainty:
                zq = z["question"][option_nodes]
                zq = F.normalize(zq, p=2, dim=-1)

                T = float(kwargs.get("temp", 1.0))
                logits = (zu @ zq.t()) / max(T, 1e-6)
                probs = torch.softmax(logits, dim=-1)

                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [n_active]
                w_raw = entropy / (entropy.mean() + 1e-12)

                # robust clipping
                q_lo = float(kwargs.get("w_q_lo", 0.10))
                q_hi = float(kwargs.get("w_q_hi", 0.90))
                lo = torch.quantile(w_raw, q_lo).item()
                hi = torch.quantile(w_raw, q_hi).item()
                w = torch.clamp(w_raw, lo, hi)
            else:
                # NEW: no uncertainty => uniform weights
                w = torch.ones((zu.size(0),), device=device)

        # 5) KMeans on active users ONLY
        features_np = zu.detach().cpu().numpy().astype(np.float64)
        sample_weight = w.detach().cpu().numpy().astype(np.float64)

        kmeans = KMeans(
            n_clusters=k_to_select,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        # NEW: optionally don't pass sample_weight at all (exactly equals standard kmeans)
        if use_uncertainty:
            kmeans.fit(features_np, sample_weight=sample_weight)
        else:
            kmeans.fit(features_np)

        # 6) pick nearest sample to each centroid
        top_k_active_local, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features_np)

        selected_local_idx = [active_u_local_indices[i] for i in top_k_active_local]
        selected_uids = [candidate_uids[i] for i in selected_local_idx]

        # 7) fill to k
        if len(selected_uids) < k:
            existing = set(selected_uids)
            for uid in candidate_uids:
                if len(selected_uids) >= k:
                    break
                if uid not in existing:
                    selected_uids.append(uid)
                    existing.add(uid)

        return (selected_uids, {"method": "CLUE-kmeans", "use_uncertainty": use_uncertainty}) \
            if kwargs.get("return_metrics") else selected_uids
