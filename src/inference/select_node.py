from typing import Dict, List, Set, Optional, Iterable, Tuple, Union
import numpy as np
import random
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from typing import Dict, List, Set, Optional, Iterable, Tuple, Union
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from tqdm import tqdm
from collections import defaultdict


class NodeSelector:
    """
    General Node Selector with support for:
    1. Single-Question Selection (Entropy, Margin, Disagreement)
    2. Multi-Question Global Selection (Monte-Carlo Entropy)
    3. Diversity Selection (K-Center)
    """
    def __init__(
        self,
        all_nodes: Iterable[str],
        initial_selected: Optional[Iterable[str]] = None,
    ):
        self.all_nodes: List[str] = list(all_nodes)
        self.selected_set: Set[str] = set(initial_selected) if initial_selected else set()

    @property
    def candidate_nodes(self) -> List[str]:
        """Returns nodes that have NOT been selected yet."""
        return [n for n in self.all_nodes if n not in self.selected_set]

    def mark_selected(self, nodes: Iterable[str]):
        """Updates internal state to mark nodes as queried."""
        self.selected_set.update(nodes)

    # ----------------- Scoring Logic (Statics) -----------------

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    @staticmethod
    def _margin(p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        s = np.sort(p)[::-1]
        margin = s[0] - s[1] if len(s) >= 2 else 0.0
        return float(-margin) # Negative so higher = more uncertainty

    def score_from_preds(
        self,
        preds: Dict[str, Dict[str, object]],
        mode: str = "entropy",
        only_candidates: bool = True,
        second_preds: Optional[Dict[str, Dict[str, object]]] = None,
        ground_truths: Optional[Dict[str, int]] = None, # <--- NEW ARGUMENT
    ) -> Dict[str, float]:
        """
        Calculates utility score for a single batch of predictions.
        """
        if not preds:
            return {}

        allowed = set(self.candidate_nodes) if only_candidates else None
        scores: Dict[str, float] = {}

        for node_id, info in preds.items():
            if allowed is not None and node_id not in allowed:
                continue

            p = np.array(info["probs"], dtype=float)

            if mode in ["entropy", "entropy_tgt"]:
                scores[node_id] = self._entropy(p)
            elif mode == "margin":
                scores[node_id] = self._margin(p)
            elif mode == "disagreement":
                if second_preds and node_id in second_preds:
                    p2 = np.array(second_preds[node_id]["probs"], dtype=float)
                    # MSE calculation
                    scores[node_id] = float(np.sum((p - p2) ** 2))
                else:
                    scores[node_id] = 0.0
            # ----------------- NEW LOGIC START -----------------
            elif mode == "oracle":
                if ground_truths is None:
                    raise ValueError("mode='oracle_loss' requires 'ground_truths' dictionary.")
                
                # If we don't have GT for this specific node, skip or score 0
                if node_id not in ground_truths:
                    scores[node_id] = 0.0
                    continue

                true_label_idx = ground_truths[node_id]
                
                # Ensure label is within bounds of probability vector
                if true_label_idx >= len(p):
                    scores[node_id] = 0.0
                    continue

                prob_true = np.clip(p[true_label_idx], 1e-12, 1.0)
                
                # Cross Entropy: -log(P(True_Label))
                # If prob_true is Low (Model is wrong), Score is High.
                scores[node_id] = float(-np.log(prob_true))
            # ----------------- NEW LOGIC END -----------------
            else:
                raise ValueError(f"Unknown scoring mode: {mode}")

        return scores

    # ----------------- K-Center (Diversity) -----------------

    def _select_k_center(self, embeddings: Dict[str, np.ndarray], k: int, verbose: bool) -> List[str]:
        # Filter candidates
        cand_ids = [n for n in self.candidate_nodes if n in embeddings]
        labeled_ids = list(self.selected_set)
        
        if not cand_ids: return []
        
        # --- FIX START: Handle Return List ---
        result_nodes = [] 
        
        # Cold Start: If nothing selected yet, pick one random node as anchor
        if not labeled_ids: 
            first_node = random.choice(cand_ids)
            
            # Add to result
            result_nodes.append(first_node) 
            
            # Treat as labeled for calculation purposes
            labeled_ids = [first_node]
            
            # Remove from candidates so we don't pick it again in the loop
            cand_ids.remove(first_node)
            
            # We found 1, so we need k-1 more
            k -= 1 
            
            if k == 0: return result_nodes
        # --- FIX END ---

        # Create Matrices
        cand_matrix = np.array([embeddings[n] for n in cand_ids])
        labeled_matrix = np.array([embeddings[n] for n in labeled_ids])

        # 1. Initialize Min Distances
        # Calculate distance from every candidate to the NEAREST existing labeled node
        dists = pairwise_distances(cand_matrix, labeled_matrix, metric='euclidean')
        min_dists = dists.min(axis=1)

        selected_indices_local = []

        # 2. Greedy Loop
        for _ in range(k):
            # Pick node with largest min_distance (Farthest from current set)
            idx_best = np.argmax(min_dists)
            selected_indices_local.append(idx_best)
            
            # Update min_dists strictly based on the NEWLY added node
            # (We don't need to re-calculate against the whole labeled_matrix)
            new_emb = cand_matrix[idx_best].reshape(1, -1)
            dist_to_new = pairwise_distances(cand_matrix, new_emb, metric='euclidean').flatten()
            
            # The new min_dist is min(old_min_dist, dist_to_new_guy)
            min_dists = np.minimum(min_dists, dist_to_new)
            
            # Mask the selected node so it isn't picked again
            min_dists[idx_best] = -1.0

        # 3. Combine results
        # Map local indices back to UIDs
        greedy_nodes = [cand_ids[i] for i in selected_indices_local]
        result_nodes.extend(greedy_nodes)

        if verbose: 
            print(f"[NodeSelector] Selected {len(result_nodes)} nodes via K-Center.")
            
        return result_nodes
    
    def select_next_nodes_multi_question(
        self,
        predictor,
        candidate_qids: List[str],
        k: int,
        batch_size: int,
        mode: str = "entropy",
        verbose: bool = True
    ) -> List[str]:
        """
        Selects nodes based on Average Score across multiple questions (Monte-Carlo).
        Correctly handles dimension mismatch by averaging Scalar Scores, not vectors.
        """
        if k <= 0: return []
        
        if verbose:
            print(f"[NodeSelector] Aggregating {mode} from {len(candidate_qids)} questions...")

        user_total_scores = defaultdict(float)
        user_counts = defaultdict(int)

        # 1. Iterate over questions (Monte Carlo Sampling)
        for qid in candidate_qids:
            # Run inference
            preds = predictor.predict_for_question(qid, batch_size=batch_size)
            
            # Calculate SCALAR scores immediately for this question
            # This avoids the dimension mismatch problem
            scores = self.score_from_preds(preds, mode=mode, only_candidates=True)

            for uid, s in scores.items():
                user_total_scores[uid] += s
                user_counts[uid] += 1

        # 2. Average the scores
        avg_scores = []
        for uid, total in user_total_scores.items():
            cnt = user_counts[uid]
            if cnt > 0:
                avg_scores.append((uid, total / cnt))

        if not avg_scores:
            if verbose: print("[NodeSelector] No scores computed.")
            return []

        # 3. Sort Descending (Higher Score = Higher Uncertainty)
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [u for u, s in avg_scores[:k]]

        if verbose:
            print(f"[NodeSelector] Selected {len(top_nodes)} nodes using Global {mode}.")
            
        return top_nodes

    # ----------------- Standard Selection Entry Point -----------------

    def select_next_nodes(
        self,
        preds: Dict[str, Dict[str, object]],
        k: int = 50,
        mode: str = "entropy",
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        second_preds: Optional[Dict[str, Dict[str, object]]] = None,
        ground_truths: Optional[Dict[str, int]] = None, # <--- NEW ARGUMENT
        verbose: bool = True,
    ) -> List[str]:
        """
        Standard selection for a SINGLE question or structural strategy.
        """
        if k <= 0 and mode != "full": return []

        if mode == "full":
            return [n for n in self.candidate_nodes if n in preds]
        
        if mode == "random":
            cand = [n for n in self.candidate_nodes if n in preds]
            k_eff = min(k, len(cand))
            return random.sample(cand, k_eff)

        if mode == "k_center":
            if embeddings is None: raise ValueError("k_center requires embeddings")
            return self._select_k_center(embeddings, k, verbose)

        # Score-based (Single Question) - Pass ground_truths here
        scores = self.score_from_preds(
            preds, 
            mode=mode, 
            only_candidates=True, 
            second_preds=second_preds,
            ground_truths=ground_truths # <--- PASS IT DOWN
        )
        
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        top_nodes = [uid for uid, _ in items[:k]]
        if verbose: print(f"[NodeSelector] Selected {len(top_nodes)} via {mode}.")
        return top_nodes
    
    def _select_adaptive_cluster_entropy(
        self, 
        embeddings: Dict[str, np.ndarray], 
        scores: Dict[str, float], 
        k: int, 
        verbose: bool
    ) -> List[str]:
        """
        Fully Automatic Clustering Strategy.
        
        1. Determine 'n_clusters' automatically using Silhouette Score analysis.
        2. Cluster nodes.
        3. Allocate budget proportional to cluster density.
        4. Select Top-Entropy nodes.
        """
        cand_ids = [n for n in self.candidate_nodes if n in embeddings and n in scores]
        
        # 0. 基础检查
        if len(cand_ids) < k:
            return cand_ids
        
        cand_matrix = np.array([embeddings[n] for n in cand_ids])
        
        # 1. 检查 Embedding 坍缩 (Collapse Check)
        # 如果唯一向量太少，直接回退到 Pure Entropy
        unique_rows = np.unique(cand_matrix.round(decimals=4), axis=0)
        num_unique = len(unique_rows)
        
        if num_unique <= 2:
            if verbose: print("[Auto-Cluster] Embeddings collapsed (<=2 unique). Fallback to Top-K.")
            return sorted(cand_ids, key=lambda u: scores[u], reverse=True)[:k]

        # 2. 自动搜索最佳 N_Clusters (Auto-K Search)
        
        # --- FIX: Budget-Aware Search Range ---
        
        # 下限 (Min): 
        # 不要从 2 开始搜。
        # 既然我们有 k 个预算，即使每个簇分 10 个人(很宽松了)，我们也至少需要 k/10 个簇。
        # 对于 k=89，min_search 约为 8。这样强迫算法去寻找更细微的子结构。
        min_search = max(2, k // 10)
        
        # 上限 (Max): 
        # 保持之前的逻辑，不超过 Unique Embedding，不超过总预算 k，硬上限 60
        max_search = min(num_unique - 1, k//2)
        
        if min_search >= max_search:
            # 搜索空间被压扁了（比如 unique 很少），直接取最大可能值
            search_candidates = [max_search]
        else:
            # 动态步长
            step = 2
            # if (max_search - min_search) > 20: step = 3
            # if (max_search - min_search) > 40: step = 5
            
            search_candidates = list(range(min_search, max_search + 1, step))
            
            # 确保边界值被包含
            if max_search not in search_candidates:
                search_candidates.append(max_search)
                
        # --------------------------------------

        best_score = -1.0
        # 默认值设为下限，而不是 2
        best_n_clusters = min_search 
        best_labels = None
        
        if verbose: 
            print(f"[Auto-Cluster] Searching best K in {search_candidates} (Budget-Aware Min={min_search})...")

        for n_c in search_candidates:
            # ... (中间 K-Means 和 Silhouette 计算逻辑不变) ...
            try:
                kmeans = KMeans(n_clusters=n_c, random_state=42, n_init=5) # n_init小一点为了速度
                labels = kmeans.fit_predict(cand_matrix)
                
                # 计算轮廓系数 (Silhouette Score)
                # 范围 [-1, 1]，越高代表聚类越合理
                if len(set(labels)) < 2: continue # 聚失败了
                
                score = silhouette_score(cand_matrix, labels, sample_size=1000) # sample_size加速
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_c
                    best_labels = labels
            except:
                continue
        
        if best_labels is None:
            # 万一全崩了，默认分 5 类
            print('ERROR!!!!!!!!!!!!!!')
            best_n_clusters = min(5, num_unique)
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(cand_matrix)

        if verbose:
            print(f"[Auto-Cluster] Selected K={best_n_clusters} (Silhouette={best_score:.3f})")

        # 3. 组织聚类结果
        cluster_indices = defaultdict(list)
        for idx, lab in enumerate(best_labels):
            cluster_indices[lab].append(idx)

        # 4. 名额分配 (Proportional Budget Allocation - Largest Remainder)
        cluster_sizes = {c: len(idxs) for c, idxs in cluster_indices.items()}
        total_size = len(cand_ids)
        
        quotas = {}
        remainders = {}
        allocated_sum = 0
        
        for c, size in cluster_sizes.items():
            exact_quota = k * (size / total_size)
            floor_quota = int(exact_quota)
            quotas[c] = floor_quota
            remainders[c] = exact_quota - floor_quota
            allocated_sum += floor_quota
            
        remainder_k = k - allocated_sum
        sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(remainder_k):
            c_idx = sorted_remainders[i][0]
            quotas[c_idx] += 1

        # 5. 簇内选点 (Top-Entropy Selection)
        selected_indices = []
        for c, idxs in cluster_indices.items():
            quota = quotas.get(c, 0)
            if quota == 0: continue
            
            node_ids = [cand_ids[i] for i in idxs]
            node_scores = [(uid, scores[uid]) for uid in node_ids]
            
            # 簇内按 Entropy 排序
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected_indices.extend([u for u, _ in node_scores[:quota]])

        return selected_indices
    # # ----------------- Public Method -----------------

    def select_next_nodes_cluster_entropy(
        self,
        predictor,
        candidate_qids: List[str], # For calculating Entropy
        k: int,
        batch_size: int,
        embeddings: Dict[str, np.ndarray], # For Clustering
        mode: str = "entropy",
        verbose: bool = True
    ) -> List[str]:
        """
        Hybrid Strategy:
        1. Calculate Global Average Entropy (Monte-Carlo over candidate_qids).
        2. Perform Adaptive Clustering + Density-Weighted Allocation on Candidates.
        3. Select Top-Entropy nodes per cluster.
        """
        
        # Step 1: Compute Scores (Uncertainty)
        if verbose:
            print(f"[Cluster-Entropy] Step 1: Computing {mode} scores on {len(candidate_qids)} questions...")
            
        user_total_scores = defaultdict(float)
        user_counts = defaultdict(int)

        # Monte Carlo accumulation
        for qid in candidate_qids:
            preds = predictor.predict_for_question(qid, batch_size=batch_size)
            # Use 'full' mode here to get raw scores for aggregation
            curr_scores = self.score_from_preds(preds, mode=mode, only_candidates=True)
            
            for uid, s in curr_scores.items():
                user_total_scores[uid] += s
                user_counts[uid] += 1
        
        # Normalize to Average Score
        final_scores = {}
        for uid, total in user_total_scores.items():
            if user_counts[uid] > 0:
                final_scores[uid] = total / user_counts[uid]
                
        if not final_scores:
            if verbose: print("[Cluster-Entropy] No scores computed. Returning empty.")
            return []

        # Step 2: Adaptive Clustering Selection
        if verbose:
            print(f"[Cluster-Entropy] Step 2: Running Adaptive Clustering selection...")
            
        return self._select_adaptive_cluster_entropy(
            embeddings=embeddings,
            scores=final_scores,
            k=k,
            verbose=verbose
        )

    def select_next_nodes_entropy(
        self,
        predictor,
        candidate_qids: List[str],  # 用于计算熵
        k: int,
        batch_size: int,
        verbose: bool = True
    ) -> List[str]:
        """
        Pure Uncertainty Strategy:
        1. Calculate Global Average Entropy (Monte-Carlo over candidate_qids).
        2. Sort users by their average entropy scores.
        3. Select Top-k highest entropy users.
        """
        
        # Step 1: Compute Scores (Uncertainty)
        if verbose:
            print(f"[Entropy-Only] Step 1: Computing entropy scores on {len(candidate_qids)} questions...")
            
        user_total_scores = defaultdict(float)
        user_counts = defaultdict(int)

        # Monte Carlo accumulation: 遍历问题，累积每个用户的熵
        for qid in candidate_qids:
            preds = predictor.predict_for_question(qid, batch_size=batch_size)
            # 获取当前所有候选用户的熵值
            curr_scores = self.score_from_preds(preds, mode="entropy", only_candidates=True)
            
            for uid, s in curr_scores.items():
                user_total_scores[uid] += s
                user_counts[uid] += 1
        
        # 计算平均熵 (Normalize to Average Score)
        user_avg_entropy = []
        for uid, total in user_total_scores.items():
            if user_counts[uid] > 0:
                avg_s = total / user_counts[uid]
                user_avg_entropy.append((uid, avg_s))
                
        if not user_avg_entropy:
            if verbose: print("[Entropy-Only] No scores computed. Returning empty.")
            return []

        # Step 2: Top-K Selection
        # 按照熵值从大到小排序
        user_avg_entropy.sort(key=lambda x: x[1], reverse=True)
        
        # 选出前 k 个用户
        selected_uids = [uid for uid, score in user_avg_entropy[:k]]
        
        if verbose:
            print(f"[Entropy-Only] Selected top {len(selected_uids)} users based on maximum average entropy.")
            
        return selected_uids


    def _select_adaptive_cluster_center(
        self, 
        embeddings: Dict[str, np.ndarray], 
        scores: Dict[str, float], 
        k: int, 
        verbose: bool
    ) -> List[str]:
        """
        Adaptive Clustering + Proportional Budget Allocation.
        
        簇内选点：完全基于距离中心点最近的原则（忽略scores/Entropy）。
        """
        cand_ids = [n for n in self.candidate_nodes if n in embeddings]
        
        # 0. 基础检查 (保持不变，但scores现在只用于过滤候选集，而不是排序)
        if len(cand_ids) < k:
            return cand_ids
        
        cand_matrix = np.array([embeddings[n] for n in cand_ids])
        
        # --- 1. 聚类数量确定与执行 (与原函数逻辑相同) ---
        
        # (省略了 N_Clusters 确定和 K-Means 执行的逻辑，假设它们的结果是 best_labels 和 kmeans.cluster_centers_)
        
        # 为了让代码可以运行，我们暂时使用一个简化的聚类过程来获取 centers 和 labels：
        # 假设 num_unique > 2
        
        num_unique = len(np.unique(cand_matrix.round(decimals=4), axis=0))
        # 默认聚类数 (N_Clusters logic needs to be simplified here)
        # n_clusters = min(num_unique, max(2, k // 4), 20) 

        # 2. 自动搜索最佳 N_Clusters (Auto-K Search)
        
        # --- FIX: Budget-Aware Search Range ---
        
        # 下限 (Min): 
        # 不要从 2 开始搜。
        # 既然我们有 k 个预算，即使每个簇分 10 个人(很宽松了)，我们也至少需要 k/10 个簇。
        # 对于 k=89，min_search 约为 8。这样强迫算法去寻找更细微的子结构。
        min_search = max(2, k // 10)
        
        # 上限 (Max): 
        # 保持之前的逻辑，不超过 Unique Embedding，不超过总预算 k，硬上限 60
        max_search = min(num_unique - 1, k//2)
        
        if min_search >= max_search:
            # 搜索空间被压扁了（比如 unique 很少），直接取最大可能值
            search_candidates = [max_search]
        else:
            # 动态步长
            step = 2
            # if (max_search - min_search) > 20: step = 3
            # if (max_search - min_search) > 40: step = 5
            
            search_candidates = list(range(min_search, max_search + 1, step))
            
            # 确保边界值被包含
            if max_search not in search_candidates:
                search_candidates.append(max_search)
                
        # --------------------------------------

        best_score = -1.0
        # 默认值设为下限，而不是 2
        best_n_clusters = min_search 
        best_labels = None
        
        if verbose: 
            print(f"[Auto-Cluster] Searching best K in {search_candidates} (Budget-Aware Min={min_search})...")

        for n_c in search_candidates:
            # ... (中间 K-Means 和 Silhouette 计算逻辑不变) ...
            try:
                kmeans = KMeans(n_clusters=n_c, random_state=42, n_init=5) # n_init小一点为了速度
                labels = kmeans.fit_predict(cand_matrix)
                
                # 计算轮廓系数 (Silhouette Score)
                # 范围 [-1, 1]，越高代表聚类越合理
                if len(set(labels)) < 2: continue # 聚失败了
                
                score = silhouette_score(cand_matrix, labels, sample_size=1000) # sample_size加速
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_c
                    best_labels = labels
            except:
                continue
        
        if best_labels is None:
            # 万一全崩了，默认分 5 类
            print('ERROR!!!!!!!!!!!!!!')
            best_n_clusters = min(5, num_unique)
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(cand_matrix)

        if verbose:
            print(f"[Auto-Cluster] Selected K={best_n_clusters} (Silhouette={best_score:.3f})")

        
        if best_n_clusters <= 1:
            # 无法聚类，直接返回随机/Top-K（这里我们返回随机以避免引入 Entropy 排序）
            if verbose: print("[Center-Select] Fallback to Random due to single cluster.")
            return random.sample(cand_ids, k)

        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cand_matrix)
        cluster_centers = kmeans.cluster_centers_ # <--- 关键：获取聚类中心
        
        
        # --- 2. 统计信息与名额分配 (与原函数逻辑相同) ---
        
        cluster_indices = defaultdict(list)
        for idx, lab in enumerate(cluster_labels):
            cluster_indices[lab].append(idx)
            
        cluster_sizes = {c: len(idxs) for c, idxs in cluster_indices.items()}
        total_size = len(cand_ids)
        
        # 使用与原函数相同的分配逻辑 (最大余额法)
        quotas = {}
        remainders = {}
        allocated_sum = 0
        
        for c, size in cluster_sizes.items():
            exact_quota = k * (size / total_size)
            floor_quota = int(exact_quota)
            quotas[c] = floor_quota
            remainders[c] = exact_quota - floor_quota
            allocated_sum += floor_quota
            
        remainder_k = k - allocated_sum
        sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(remainder_k):
            c_idx = sorted_remainders[i][0]
            quotas[c_idx] += 1


        # --- 3. 簇内选点：基于最小距离 (Minimal Distance Selection) ---
        selected_nodes_by_center = []
        
        for c_label, idxs in cluster_indices.items():
            quota = quotas.get(c_label, 0)
            if quota == 0: continue
            
            # 1. 获取该簇的候选节点和嵌入
            cluster_cand_ids = [cand_ids[i] for i in idxs]
            cluster_matrix = cand_matrix[idxs]
            
            # 2. 获取聚类中心点
            center = cluster_centers[c_label].reshape(1, -1)
            
            # 3. 计算所有簇内节点到中心点的距离
            # dists: shape (N_cluster_members, 1)
            dists = pairwise_distances(cluster_matrix, center, metric='euclidean').flatten()
            
            # 4. 结合 ID 和距离
            node_dist_pairs = list(zip(cluster_cand_ids, dists))
            
            # 5. 按距离升序排列 (最小距离优先)
            node_dist_pairs.sort(key=lambda x: x[1], reverse=False)
            
            # 6. 选择前 quota 个节点 (距离中心点最近的节点)
            top_nodes = [uid for uid, _ in node_dist_pairs[:quota]]
            
            selected_nodes_by_center.extend(top_nodes)

        if verbose:
            print(f"[Center-Select] Selected {len(selected_nodes_by_center)} nodes via Cluster Center Distance.")
            
        return selected_nodes_by_center

    import numpy as np
    import random
    from collections import defaultdict
    from typing import Dict, List
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, pairwise_distances

    def _select_adaptive_cluster_coverage(
        self, 
        embeddings: Dict[str, np.ndarray], 
        scores: Dict[str, float], 
        k: int, 
        asked_users_set: Set[str],
        verbose: bool = False
    ) -> List[str]:
        """
        Adaptive Clustering + Proportional Budget Allocation + Coverage Maximization (FPS).
        
        逻辑：
        1. 自动搜索最佳聚类数 K。
        2. 按簇大小分配配额（Quota）。
        3. 簇内使用 FPS (Furthest Point Sampling) 以最大化特征空间的覆盖度。
        """
        # 0. 基础过滤与准备
        cand_ids = [n for n in self.candidate_nodes if n in embeddings]
        if len(cand_ids) <= k:
            if verbose: print(f"[Cluster] Budget {k} >= Candidates {len(cand_ids)}, returning all.")
            return cand_ids
        
        cand_matrix = np.array([embeddings[n] for n in cand_ids])
        num_samples = len(cand_ids)
        # 简单的去重统计用于限制聚类上限
        num_unique = len(np.unique(cand_matrix.round(decimals=4), axis=0))

        # --- 1. 自动确定最佳聚类数量 (Auto-K Search) ---
        
        # 搜索范围：至少每个簇分10个点（k/10），至多每个簇2个点（k/2）
        min_search = max(2, k // 10)
        max_search = min(num_unique - 1, k // 2, 60) # 60 为性能硬上限
        
        if min_search >= max_search:
            search_candidates = [max(2, max_search)]
        else:
            step = 2 if (max_search - min_search) < 20 else 4
            search_candidates = list(range(min_search, max_search + 1, step))
            if max_search not in search_candidates:
                search_candidates.append(max_search)

        best_score = -1.0
        best_n_clusters = min_search 
        best_labels = None
        
        if verbose: 
            print(f"[Auto-Cluster] Searching K in {search_candidates} for {num_samples} samples...")

        for n_c in search_candidates:
            try:
                # 使用较小的 n_init 以加速搜索
                km = KMeans(n_clusters=n_c, random_state=42, n_init=5)
                labels = km.fit_predict(cand_matrix)
                
                if len(np.unique(labels)) < 2: continue
                
                # 抽样计算轮廓系数加速计算
                score = silhouette_score(cand_matrix, labels, sample_size=1000, random_state=42)
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_c
                    best_labels = labels
            except Exception as e:
                continue
        
        # Final KMeans with best K
        if best_labels is None:
            best_n_clusters = min(5, num_unique)
            km = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            best_labels = km.fit_predict(cand_matrix)
        else:
            # 重新跑一遍完整的 KMeans 以获取精确的 centers
            km = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            best_labels = km.fit_predict(cand_matrix)

        cluster_centers = km.cluster_centers_

        # --- 2. 名额分配 (Proportional Allocation via Largest Remainder Method) ---
        
        cluster_indices = defaultdict(list)
        for idx, lab in enumerate(best_labels):
            cluster_indices[lab].append(idx)
            
        quotas = {}
        remainders = {}
        allocated_sum = 0
        
        for c, idxs in cluster_indices.items():
            exact_quota = k * (len(idxs) / num_samples)
            floor_quota = int(exact_quota)
            quotas[c] = floor_quota
            remainders[c] = exact_quota - floor_quota
            allocated_sum += floor_quota
            
        # 分配剩余名额给余数最大的簇
        remainder_k = k - allocated_sum
        sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        for i in range(remainder_k):
            quotas[sorted_remainders[i][0]] += 1

        # --- 3. 簇内选点：FPS 最大化覆盖 ---
        # 相比于直接选中心，FPS 能选出更有代表性的边缘特征点
        
        selected_node_ids = []
        
        for c_label, idxs in cluster_indices.items():
            quota = quotas.get(c_label, 0)
            if quota <= 0: continue
            
            cluster_cand_ids = [cand_ids[i] for i in idxs]
            cluster_matrix = cand_matrix[idxs]
            
            if quota >= len(cluster_cand_ids):
                selected_node_ids.extend(cluster_cand_ids)
                continue

            # A. 初始点：选择距离簇质心最近的点（作为簇的锚点）
            center = cluster_centers[c_label].reshape(1, -1)
            dists_to_center = pairwise_distances(cluster_matrix, center, metric='euclidean').flatten()
            first_idx = np.argmin(dists_to_center)
            
            current_selected_sub_idxs = [first_idx]
            
            # B. FPS 迭代：每次选距离已选集合最远的点
            # 初始化最小距离向量：所有点到第一个选中点的距离
            min_distances = pairwise_distances(
                cluster_matrix, 
                cluster_matrix[first_idx].reshape(1, -1), 
                metric='euclidean'
            ).flatten()
            
            for _ in range(1, quota):
                # 选择当前到已选集合距离最大的点
                next_idx = np.argmax(min_distances)
                current_selected_sub_idxs.append(next_idx)
                
                # 更新最小距离向量：比较旧的最小距离和到新选点之间的距离
                new_dists = pairwise_distances(
                    cluster_matrix, 
                    cluster_matrix[next_idx].reshape(1, -1), 
                    metric='euclidean'
                ).flatten()
                min_distances = np.minimum(min_distances, new_dists)
            
            # 将局部索引映射回原始 ID
            for sub_idx in current_selected_sub_idxs:
                selected_node_ids.append(cluster_cand_ids[sub_idx])

        if verbose:
            print(f"[Adaptive-Coverage] Final Selected: {len(selected_node_ids)} nodes (K={best_n_clusters})")
            
        return selected_node_ids

    # def _select_adaptive_cluster_coverage(
    #     self, 
    #     embeddings: Dict[str, np.ndarray], 
    #     scores: Dict[str, float], 
    #     k: int, 
    #     asked_users_set: Set[str], # 历史已选用户
    #     verbose: bool = False) -> List[str]:
    #     # --- 1. 基础准备：保留所有候选人 ---
    #     # 不再剔除 asked_users_set，只确保在 embedding 字典中
    #     cand_ids = [n for n in self.candidate_nodes if n in embeddings]
        
    #     if len(cand_ids) <= k:
    #         return cand_ids
        
    #     cand_matrix = np.array([embeddings[n] for n in cand_ids])
    #     num_samples = len(cand_ids)
    #     num_unique = len(np.unique(cand_matrix.round(decimals=4), axis=0))
        
    #     # 提取历史已选用户的 Embedding
    #     asked_embeds = np.array([embeddings[uid] for uid in asked_users_set if uid in embeddings])

    #     # --- 2. 自动聚类 (Auto-K Search) ---
    #     # (保持你原有的 KMeans 和 Quota 分配逻辑不变...)
    #     # 假设得到 cluster_indices, quotas, cluster_centers...
    #     #     # --- 1. 自动确定最佳聚类数量 (Auto-K Search) ---
        
    #     # 搜索范围：至少每个簇分10个点（k/10），至多每个簇2个点（k/2）
    #     min_search = max(2, k // 10)
    #     max_search = min(num_unique - 1, k // 2, 60) # 60 为性能硬上限
        
    #     if min_search >= max_search:
    #         search_candidates = [max(2, max_search)]
    #     else:
    #         step = 2 if (max_search - min_search) < 20 else 4
    #         search_candidates = list(range(min_search, max_search + 1, step))
    #         if max_search not in search_candidates:
    #             search_candidates.append(max_search)

    #     best_score = -1.0
    #     best_n_clusters = min_search 
    #     best_labels = None
        
    #     if verbose: 
    #         print(f"[Auto-Cluster] Searching K in {search_candidates} for {num_samples} samples...")

    #     for n_c in search_candidates:
    #         try:
    #             # 使用较小的 n_init 以加速搜索
    #             km = KMeans(n_clusters=n_c, random_state=42, n_init=5)
    #             labels = km.fit_predict(cand_matrix)
                
    #             if len(np.unique(labels)) < 2: continue
                
    #             # 抽样计算轮廓系数加速计算
    #             score = silhouette_score(cand_matrix, labels, sample_size=1000, random_state=42)
                
    #             if score > best_score:
    #                 best_score = score
    #                 best_n_clusters = n_c
    #                 best_labels = labels
    #         except Exception as e:
    #             continue
        
    #     # Final KMeans with best K
    #     if best_labels is None:
    #         best_n_clusters = min(5, num_unique)
    #         km = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    #         best_labels = km.fit_predict(cand_matrix)
    #     else:
    #         # 重新跑一遍完整的 KMeans 以获取精确的 centers
    #         km = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    #         best_labels = km.fit_predict(cand_matrix)

    #     cluster_centers = km.cluster_centers_

    #     # --- 2. 名额分配 (Proportional Allocation via Largest Remainder Method) ---
        
    #     cluster_indices = defaultdict(list)
    #     for idx, lab in enumerate(best_labels):
    #         cluster_indices[lab].append(idx)
            
    #     quotas = {}
    #     remainders = {}
    #     allocated_sum = 0
        
    #     for c, idxs in cluster_indices.items():
    #         exact_quota = k * (len(idxs) / num_samples)
    #         floor_quota = int(exact_quota)
    #         quotas[c] = floor_quota
    #         remainders[c] = exact_quota - floor_quota
    #         allocated_sum += floor_quota
            
    #     # 分配剩余名额给余数最大的簇
    #     remainder_k = k - allocated_sum
    #     sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    #     for i in range(remainder_k):
    #         quotas[sorted_remainders[i][0]] += 1

    #     # --- 3. 簇内选点：FPS 最大化覆盖 ---
    #     # 相比于直接选中心，FPS 能选出更有代表性的边缘特征点

    #     # --- 3. 簇内选点：带历史惩罚的 FPS ---
    #     selected_node_ids = []
        
    #     for c_label, idxs in cluster_indices.items():
    #         quota = quotas.get(c_label, 0)
    #         if quota <= 0: continue
            
    #         cluster_cand_ids = [cand_ids[i] for i in idxs]
    #         cluster_matrix = cand_matrix[idxs]
            
    #         # --- 软惩罚核心：初始化距离向量 ---
    #         if len(asked_embeds) > 0:
    #             # 计算当前簇内所有候选点到“历史集合”的最小距离
    #             # 这会给已经选过的点（或离得近的点）一个极小的起始距离，从而实现惩罚
    #             history_dists = pairwise_distances(cluster_matrix, asked_embeds, metric='euclidean')
    #             min_distances = history_dists.min(axis=1) 
    #         else:
    #             # 如果没有历史点，则以到簇质心的距离作为初始反向参考（可选）
    #             # 或者初始化为无穷大，从簇中心开始选
    #             center = cluster_centers[c_label].reshape(1, -1)
    #             min_distances = pairwise_distances(cluster_matrix, center, metric='euclidean').flatten()

    #         current_selected_sub_idxs = []
            
    #         for _ in range(quota):
    #             # FPS 核心：选择当前到“已覆盖集合”距离最大的点
    #             # 被选过的人 min_distances 会是 0，因此除非 quota 极大，否则不会被重复选中
    #             next_idx = np.argmax(min_distances)
    #             current_selected_sub_idxs.append(next_idx)
                
    #             # 更新最小距离：考虑本轮新选中的点
    #             new_dists = pairwise_distances(
    #                 cluster_matrix, 
    #                 cluster_matrix[next_idx].reshape(1, -1), 
    #                 metric='euclidean'
    #             ).flatten()
    #             min_distances = np.minimum(min_distances, new_dists)
            
    #         for sub_idx in current_selected_sub_idxs:
    #             selected_node_ids.append(cluster_cand_ids[sub_idx])

    #     return selected_node_ids

    def _select_kmeans_unique_center(
        self, 
        embeddings: Dict[str, np.ndarray], 
        k: int, 
        verbose: bool = True
    ) -> List[str]:
        """
        基于 K-Means 的中心点选择策略。
        逻辑：
        1. 将候选节点聚类为 k 个簇。
        2. 在每个簇内，找到距离该簇质心 (Centroid) 最近的节点。
        """
        # 1. 准备数据
        cand_ids = [n for n in self.candidate_nodes if n in embeddings]
        if len(cand_ids) <= k:
            return cand_ids
        
        cand_matrix = np.array([embeddings[n] for n in cand_ids])
        
        # 2. 执行 K-Means 聚类
        # n_clusters=k 确保我们从 k 个不同的语义区域选点
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cand_matrix)
        centroids = kmeans.cluster_centers_
        
        # 3. 按簇组织节点索引
        cluster_to_node_indices = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            cluster_to_node_indices[label].append(idx)
            
        selected_nodes = []
        
        # 4. 从每个簇中提取最靠近质心的唯一中心点
        for label, node_idxs in cluster_to_node_indices.items():
            # 获取该簇的特征矩阵和对应的质心
            cluster_features = cand_matrix[node_idxs]
            centroid = centroids[label].reshape(1, -1)
            
            # 计算该簇内所有点到质心的欧氏距离
            dists = pairwise_distances(cluster_features, centroid, metric='euclidean').flatten()
            
            # 找到最小距离对应的局部索引，并映射回全局 ID
            best_local_idx = np.argmin(dists)
            best_node_id = cand_ids[node_idxs[best_local_idx]]
            selected_nodes.append(best_node_id)
            
        if verbose:
            print(f"[K-Means Center] Partitioned into {len(selected_nodes)} clusters and selected centroids.")
            
        return selected_nodes


    def select_next_nodes_cluster_center(
        self,
        embeddings: Dict[str, np.ndarray],
        k: int,
        verbose: bool = True
    ) -> List[str]:
        """
        Public method to select nodes based purely on distance to cluster centers.
        (This strategy ignores the Entropy score and maximizes Representativeness/Diversity).
        """
        # 传入一个空的 scores 字典，因为 _select_adaptive_cluster_center 只需要 scores 来过滤候选集
        # 而不需要用于排序。
        dummy_scores = {uid: 1.0 for uid in embeddings.keys()} 
        
        return self._select_adaptive_cluster_center(
            embeddings=embeddings,
            scores=dummy_scores, # 传入一个假的 scores，只用作候选集过滤
            k=k,
            verbose=verbose
        )


    def select_next_nodes_cluster_coverage(
        self,
        embeddings: Dict[str, np.ndarray],
        asked_users_set: Set[str],
        k: int,
        verbose: bool = True
    ) -> List[str]:
        """
        Public method to select nodes based purely on distance to cluster centers.
        (This strategy ignores the Entropy score and maximizes Representativeness/Diversity).
        """
        # 传入一个空的 scores 字典，因为 _select_adaptive_cluster_center 只需要 scores 来过滤候选集
        # 而不需要用于排序。
        dummy_scores = {uid: 1.0 for uid in embeddings.keys()} 
        
        return self._select_adaptive_cluster_coverage(
            embeddings=embeddings,
            scores=dummy_scores, # 传入一个假的 scores，只用作候选集过滤
            k=k,
            asked_users_set=asked_users_set,
            verbose=verbose
        )

    def select_next_nodes_unique_center(
        self,
        embeddings: Dict[str, np.ndarray],
        k: int,
        verbose: bool = True
    ) -> List[str]:
        """
        Public method to select nodes based purely on distance to cluster centers.
        (This strategy ignores the Entropy score and maximizes Representativeness/Diversity).
        """
        # 传入一个空的 scores 字典，因为 _select_adaptive_cluster_center 只需要 scores 来过滤候选集
        # 而不需要用于排序。
        dummy_scores = {uid: 1.0 for uid in embeddings.keys()} 
        
        return self._select_kmeans_unique_center(
            embeddings=embeddings,
            k=k,
            verbose=verbose
        )



    @staticmethod
    def _build_subgroup_adj_from_graph(graph) -> tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """
        Build u2s and s2u from full static subgroup edges in graph.data.
        """
        su = graph.data[('subgroup', 'to', 'user')].edge_index  # [2, E]
        u2s = defaultdict(set)
        s2u = defaultdict(set)
        if su is not None and su.numel() > 0:
            s_list = su[0].tolist()
            u_list = su[1].tolist()
            for s, u in zip(s_list, u_list):
                s2u[s].add(u)
                u2s[u].add(s)
        return u2s, s2u

    @staticmethod
    def _build_option_adj_from_edges(edges_u_to_qopt) -> tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """
        Build u2a and a2u from observed (real) edges list.
        Note: question node is actually option node in your setup (qid2choices).
        """
        u2a = defaultdict(set)
        a2u = defaultdict(set)
        for u, _qid, a in edges_u_to_qopt:
            u2a[u].add(a)
            a2u[a].add(u)
        return u2a, a2u

    # [FIX] 增加 total_candidates 参数用于动态计算阈值
    def _reach_users(self, u: int, u2s, s2u, u2a, a2u, total_candidates: int, beta: float = 1.0) -> Set[int]:
        reached = set()
        
        # 动态阈值：如果一个 Subgroup 包含超过 60% 的候选人，视为无效的“背景噪声”
        # 例如：如果所有候选人都是 "USA"，那么 "USA" 这个节点就没有任何区分度
        ignore_threshold = total_candidates * 0.5 

        # 1. Subgroup 覆盖 (User -> Subgroup -> Other Users)
        if u in u2s:
            subgroups = u2s[u]
            for s in subgroups:
                if s in s2u:
                    members = s2u[s]
                    
                    # [Optimization] 过滤掉超级节点 (Super Nodes)
                    if len(members) > ignore_threshold:
                        # 可选：打印一下看看过滤了谁 (Debug用)
                        # print(f"   [Filter] Ignoring subgroup {s} (Size: {len(members)} > {ignore_threshold})")
                        continue 
                        
                    reached.update(members)
        
        # 2. Option 覆盖 (User -> Answer -> Other Users)
        # 答案通常具有很高的区分度，一般不过滤，或者设置更高的阈值 (e.g. 90%)
        if u in u2a:
            options = u2a[u]
            for op in options:
                if op in a2u:
                    reached.update(a2u[op])
                    
        return reached

    def select_next_nodes_influence_greedy(
        self,
        graph,
        observed_edges,
        k: int,
        beta: float = 1.0,
        verbose: bool = True,
    ) -> List[str]:
        # ... (前置的 Mapping 逻辑保持不变) ...
        
        if not hasattr(graph, 'uid2idx'):
            raise ValueError("Graph object must have 'uid2idx'.")
        uid2idx = graph.uid2idx
        idx2uid = {v: k for k, v in uid2idx.items()}

        # 转换 Candidates
        cand_indices = []
        for n in self.candidate_nodes:
            if n in uid2idx: cand_indices.append(uid2idx[n])
            elif isinstance(n, (int, np.integer)): cand_indices.append(int(n))
        cand = cand_indices
        if k <= 0 or not cand: return []

        # 转换 Edges
        mapped_edges = []
        for val in observed_edges:
            u_str = val[0]
            if u_str in uid2idx:
                new_edge = (uid2idx[u_str],) + tuple(val[1:])
                mapped_edges.append(new_edge)

        # 构建邻接表
        u2s, s2u = self._build_subgroup_adj_from_graph(graph)
        u2a, a2u = self._build_option_adj_from_edges(mapped_edges)

        # [NEW] 获取候选人总数
        N_total = len(cand)

        # Precompute Reach Sets
        reach_cache: Dict[int, Set[int]] = {}
        for u in cand:
            # [FIX] 传入 N_total
            neighbors = self._reach_users(u, u2s, s2u, u2a, a2u, total_candidates=N_total, beta=beta)
            neighbors.add(u) # 强制自身覆盖
            reach_cache[u] = neighbors

        selected_indices: List[int] = []
        covered: Set[int] = set()
        k_eff = min(k, len(cand))
        
        # Greedy Loop
        for _ in range(k_eff):
            best_u = None
            best_gain = -1

            for u in cand:
                if u in self.selected_set or u in selected_indices:
                    continue
                
                gain = len(reach_cache[u] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_u = u

            if best_u is None or best_gain <= 0:
                remaining = [x for x in cand if x not in self.selected_set and x not in selected_indices]
                if remaining:
                    needed = k_eff - len(selected_indices)
                    selected_indices.extend(remaining[:needed])
                    if verbose: print(f"[NodeSelector] Fallback: Randomly filled {needed} users.")
                break

            selected_indices.append(best_u)
            covered |= reach_cache[best_u]

        # 映射回 String
        selected_caseids = []
        for idx in selected_indices:
            if idx in idx2uid:
                selected_caseids.append(idx2uid[idx])

        if verbose:
            print(f"[NodeSelector] Selected {len(selected_caseids)} users via Influence-Greedy (SuperNode Filter Active).")
    
        return selected_caseids


import numpy as np
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

class IGNodeSelector(NodeSelector):
    """
    Selects nodes that maximize Expected Information Gain (EIG) using batched inference.
    """
    def _calculate_subset_entropy(
        self, 
        preds: Dict[str, Dict[str, np.ndarray]], 
        target_uids: set, 
        ignore_uid: str = None
    ) -> float:
        """
        Calculates the average Shannon entropy over the target subset of users.
        """
        total_entropy = 0.0
        count = 0
        
        for uid in target_uids:
            # Skip the user we are currently simulating (their answer is fixed/virtual)
            if uid == ignore_uid:
                continue
            
            # If for some reason this user is missing from predictions, skip
            if uid not in preds:
                continue
                
            # Get probabilities: shape [Num_Options]
            probs = preds[uid]["probs"]
            
            # robust log to avoid log(0)
            # H(p) = - sum( p * log(p) )
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            
            total_entropy += entropy
            count += 1
            
        if count == 0:
            return 0.0
            
        # Return average entropy across the population
        return total_entropy / count

    def select_next_nodes(
        self,
        predictor,
        qid: str,
        candidate_list: List[str],
        k: int = 10,
        candidates_pool_size: int = 50,  
        target_subset_size: int = 200,   
        batch_size: int = 256,
        inference_batch_size: int = 16 
    ) -> List[str]:
        
        # --- 1. Filter & Downsample ---
        candidates = [
            uid for uid in candidate_list 
            if predictor.get_recorded_answer(uid, qid) is None
        ]
        if len(candidates) > candidates_pool_size:
            candidates = random.sample(candidates, candidates_pool_size)

        if not candidates:
            print("[IG-Selector] No valid candidates found.")
            return []

        print(f"[IG-Selector] Calculating Subgroup-Aware EIG for {len(candidates)} candidates...")

        # --- 2. Baseline Predictions ---
        base_preds = predictor.predict_for_question(qid, batch_size=batch_size)
        
        # --- 3. [NEW] Pre-compute Peer Groups (Subgroup Sharing) ---
        candidate_peers_map = {}
        
        try:
            # 1. Get raw edge index
            edge_index_s2u = predictor.graph.data['subgroup', 'to', 'user'].edge_index.cpu()
            
            from torch_geometric.utils import to_scipy_sparse_matrix
            
            num_subgroups = predictor.graph.data['subgroup'].num_nodes
            num_users = predictor.graph.data['user'].num_nodes
            
            # 2. Build Adjacency: Rows=Subgroups, Cols=Users
            # Note: We must specify the shape explicitly to ensure dimensions match
            adj_s2u_coo = to_scipy_sparse_matrix(
                edge_index_s2u, 
                num_nodes=max(num_subgroups, num_users) 
            )
            
            # --- FIX: Convert to CSR/CSC for arithmetic and slicing ---
            adj_s2u = adj_s2u_coo.tocsr() # Optimization for matrix multiplication
            adj_u2s = adj_s2u.T           # Transpose: Rows=Users, Cols=Subgroups
            
            # 3. Compute Peers
            for uid in candidates:
                u_idx = predictor.graph.uid2idx[uid]
                
                # Slicing works now (CSR format)
                user_subgroups = adj_u2s[u_idx] 
                
                # Matrix Mult: (1 x S) @ (S x U) -> (1 x U)
                peers_vector = user_subgroups @ adj_s2u 
                
                peer_indices = peers_vector.indices 
                
                # --- FIX: Access idx2uid from the PREDICTOR, not the GRAPH ---
                peers = [predictor.idx2uid[idx] for idx in peer_indices]
                
                # Filter out the user themselves
                peers = [p for p in peers if p != uid]

                if len(peers) > target_subset_size:
                    peers = random.sample(peers, target_subset_size)
                
                if not peers:
                     peers = [uid] 
                
                candidate_peers_map[uid] = set(peers)
                
        except Exception as e:
            # Error handling remains the same...
            print(f"[IG-Selector] Warning: Could not calculate subgroups ({e}). Fallback to global random.")
            # ...
            all_uids = list(base_preds.keys())
            global_fallback = set(random.sample(all_uids, min(len(all_uids), target_subset_size)))
            for uid in candidates:
                candidate_peers_map[uid] = global_fallback

        # --- 4. PREPARE SCENARIOS ---
        inference_tasks = [] 
        for u_node in candidates:
            if u_node not in base_preds: continue
            
            # Skip candidates with no peers (rare, but possible)
            if not candidate_peers_map.get(u_node):
                continue

            probs_u = np.array(base_preds[u_node]["probs"], dtype=float)
            for label_idx, p_label in enumerate(probs_u):
                if p_label < 1e-4: continue
                inference_tasks.append({
                    "node_id": u_node,
                    "label_idx": label_idx,
                    "p_label": p_label
                })

        # --- 5. BATCHED INFERENCE ---
        scenario_results = {} 
        total_tasks = len(inference_tasks)
        
        for i in tqdm(range(0, total_tasks, inference_batch_size), desc="Batched EIG Inference"):
            batch_tasks = inference_tasks[i : i + inference_batch_size]
            
            batch_preds_list = predictor.predict_batch_virtual_scenarios(
                qid=qid,
                scenarios=batch_tasks
            )
            
            for task, preds in zip(batch_tasks, batch_preds_list):
                # [CHANGED] Use the candidate-specific peer set
                target_peers = candidate_peers_map[task["node_id"]]
                
                sys_entropy = self._calculate_subset_entropy(
                    preds, 
                    target_peers, # <--- Only evaluate entropy on peers!
                    ignore_uid=task["node_id"]
                )
                scenario_results[(task["node_id"], task["label_idx"])] = sys_entropy

        # --- 6. AGGREGATE SCORES ---
        final_scores: List[Tuple[str, float]] = []
        for u_node in candidates:
            if u_node not in base_preds: continue
            
            expected_conditional_entropy = 0.0
            probs_u = np.array(base_preds[u_node]["probs"], dtype=float)
            valid_calc = False
            
            for label_idx, p_label in enumerate(probs_u):
                if p_label < 1e-4: continue
                h_result = scenario_results.get((u_node, label_idx))
                if h_result is not None:
                    expected_conditional_entropy += p_label * h_result
                    valid_calc = True
            
            if valid_calc:
                final_scores.append((u_node, -expected_conditional_entropy))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, score in final_scores[:k]]