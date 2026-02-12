import numpy as np
from copy import deepcopy
from tqdm import tqdm


def select_queries(dataset, pool, nodes, Xavail, observed, cur_asked_queries, verbose=True):
    query_cand_list = list(Xavail)
    if not query_cand_list: return []

    # -----------------------------------------------------------
    # 1. Setup Metadata & Masking
    # -----------------------------------------------------------
    heldout_qids = dataset.X_heldout
    heldout_text = [dataset.codebook[qid]["question"] for qid in heldout_qids]
    heldout_opts = [list(dataset.codebook[qid]["options"].keys()) for qid in heldout_qids]
    
    max_h_opts = max(len(o) for o in heldout_opts)
    mask_heldout = np.zeros((len(heldout_qids), max_h_opts), dtype=bool)
    for i, opts in enumerate(heldout_opts):
        mask_heldout[i, :len(opts)] = True

    def masked_entropy(probs, mask, eps=1e-12):
        # Aligns with: -torch.sum(probs * torch.log(probs), dim=-1)
        p = probs * mask[None, :, :]
        p /= (p.sum(axis=-1, keepdims=True) + eps)
        log_p = np.zeros_like(p)
        pos = p > eps
        log_p[pos] = np.log(p[pos])
        return -(p * log_p).sum(axis=-1) # [N_Nodes, N_Heldout]

    # -----------------------------------------------------------
    # 2. Baseline Calculation (entropy_without_designs)
    # -----------------------------------------------------------
    probs_no_design = pool.predict(
        items=nodes, shard_arg="nodes", query=heldout_text, 
        candidate_options=heldout_opts, asked_queries=cur_asked_queries, observed=observed
    )
    # entropy_without_designs in ref code is mean over targets
    h_pre_per_target_node = masked_entropy(probs_no_design, mask_heldout) # [N_Nodes, N_Heldout]
    baseline_entropy = np.mean(h_pre_per_target_node) 

    # -----------------------------------------------------------
    # 3. Candidate Outcome Probabilities (conditional_design_probs)
    # -----------------------------------------------------------
    cand_text = [dataset.codebook[qid]["question"] for qid in query_cand_list]
    cand_opts = [list(dataset.codebook[qid]["options"].keys()) for qid in query_cand_list]
    
    # Get P(y | context) for all candidates
    probs_cands = pool.predict(
        items=nodes, shard_arg="nodes", query=cand_text, 
        candidate_options=cand_opts, asked_queries=cur_asked_queries, observed=observed
    )

    assert probs_no_design.shape[:2] == (len(nodes), len(heldout_qids))
    assert probs_cands.shape[:2] == (len(nodes), len(query_cand_list))

    # -----------------------------------------------------------
    # 4. EIG Loop (Equivalent to the nested loops in ref code)
    # -----------------------------------------------------------
    EIG = {}
    pbar = tqdm(enumerate(query_cand_list), total=len(query_cand_list), 
                desc="📊 EIG Eval", disable=not verbose, leave=False)
    
    for i, query_id in pbar:
        prev = dict(observed.get(query_id, {})) 
        assert len(prev) == 0
        assert query_id not in {qid for (qid, _) in cur_asked_queries}

        current_opts = cand_opts[i]
        # p_y_given_context: [N_Nodes, N_Actual_Opts]
        p_y_given_context = probs_cands[:, i, :len(current_opts)]
        p_y_given_context /= (p_y_given_context.sum(axis=-1, keepdims=True) + 1e-12)
        
        # Accumulator for E_y [ H(targets | context, y) ]
        # Shape matches [N_Nodes, N_Heldout]
        expected_h_post = np.zeros((len(nodes), len(heldout_qids)))

        for opt_idx, label in enumerate(current_opts):
            # p_y: Weight for this outcome branch
            p_y = p_y_given_context[:, opt_idx] # [N_Nodes]
            
            # Update context for the simulation (temp = designs.clone() ... temp[mask] = token)
            simulated_obs = {**observed, query_id: {nodeid: label for nodeid in nodes}}
            simulated_asked = cur_asked_queries + [(query_id, dataset.codebook[query_id]["question"])]
            
            # Get posterior probabilities: P(Targets | context, Design=y)
            probs_post = pool.predict(
                items=nodes, shard_arg="nodes", query=heldout_text, 
                candidate_options=heldout_opts, asked_queries=simulated_asked, 
                observed=simulated_obs
            )
            
            # H(post) per target per node
            h_post_k = masked_entropy(probs_post, mask_heldout) # [N_Nodes, N_Heldout]
            
            # Weight the entropy by the probability of this answer: p(y|context) * H(post)
            expected_h_post += p_y[:, None] * h_post_k

        # Final IG: Mean(H_pre) - Mean(E[H_post])
        # Aligns with: return torch.mean(entropy_without_designs) - torch.mean(torch.stack(design_entropies), dim=0)
        # Note: ref code uses mean(dim=0) which results in a score per design.
        current_design_ig = baseline_entropy - np.mean(expected_h_post)
        EIG[query_id] = current_design_ig

    return EIG

# def select_queries(dataset, pool, nodes, Xavail, observed, cur_asked_queries, verbose=False):
#     query_cand_list = list(Xavail)
#     if not query_cand_list: return {}

#     heldout_queries_text = [dataset.codebook[qid]["question"] for qid in dataset.X_heldout]
#     heldout_queries_options = [list(dataset.codebook[qid]["options"].keys()) for qid in dataset.X_heldout]
#     cand_queries_text = [dataset.codebook[qid]["question"] for qid in query_cand_list]
#     cand_queries_options = [list(dataset.codebook[qid]["options"].keys()) for qid in query_cand_list]
    
#     node_to_idx = {n: i for i, n in enumerate(nodes)}
#     target_indices = [node_to_idx[n] for n in nodes]
    
#     H_pre_list = []
#     for qid in dataset.X_heldout:
#         q_text = dataset.codebook[qid]["question"]
#         options = list(dataset.codebook[qid]["options"].keys())
#         probs_batch = pool.predict(items=nodes, shard_arg="nodes", query=q_text, candidate_options=options, asked_queries=dataset.asked_queries, 
#                                     observed=observed)
#         H_pre_list.append(entropy_vectorized(probs_batch))
#     H_pre_matrix = np.array(H_pre_list) # [N_Heldout, N_Target]

#     probs_cands_all = pool.predict(
#         items=nodes, shard_arg="nodes", query=cand_queries_text, 
#         candidate_options=cand_queries_options, asked_queries=cur_asked_queries, observed=observed
#     )
#     probs_cands_target = np.array([probs_cands_all[i] for i in target_indices])

#     EIG = {}
#     pbar = tqdm(enumerate(query_cand_list), total=len(query_cand_list), 
#                 desc="📊 EIG Eval", disable=not verbose, leave=False)
    
#     for i, query in pbar:
#         current_opts = cand_queries_options[i]
#         p_outcome_matrix = probs_cands_target[:, i, :] # [N_Target, N_Opts]
        
#         Ey_H_post = np.zeros_like(H_pre_matrix) 
        
#         for opt_idx, label in enumerate(current_opts):
#             p_k = p_outcome_matrix[:, opt_idx]
            
#             temp_obs = deepcopy(observed)
#             if query not in temp_obs: temp_obs[query] = {}
#             for nodeid in nodes: temp_obs[query][nodeid] = label
   
#             probs_post = pool.predict(items=nodes, shard_arg="nodes", query=heldout_queries_text, 
#                 candidate_options=heldout_queries_options, asked_queries=cur_asked_queries + [(query, dataset.codebook[query]["question"])], 
#                 observed=temp_obs)
            
#             H_post_k = entropy_vectorized(np.array(probs_post)).T
#             Ey_H_post += (p_k[None, :] * H_post_k)

#         # IG = H_pre - E[H_post]
#         IG_matrix = np.maximum(0, H_pre_matrix - Ey_H_post)
#         EIG[query] = np.mean(IG_matrix, axis=1).sum()

#     return EIG

def entropy_vectorized(P, eps=1e-12):
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, eps, 1.0)
    return -(P * np.log(P)).sum(axis=-1)



import numpy as np

import numpy as np

def inject_pseudo_labels(
    qid,
    gnn_preds,
    observed_dict,
    confidence_margin: float = 0.0,
    mode: str = "argmax",              # "argmax" | "sampling"
    rng=None,
    temp: float = 1.0,                 # temperature for sampling (and optionally argmax if you want)
    return_per_uid_stats: bool = False # if True, include per-uid stats for debugging
):
    """
    Inject high-confidence GNN predictions into observed_dict[qid].

    - Filters uids by max_prob >= 1/K + confidence_margin.
    - mode:
        * "argmax": choose argmax under probs_T (default probs_T=probs unless temp!=1)
        * "sampling": sample under probs_T
    - Temperature scaling on probs (prob-based): p_T ∝ p^(1/T)

    Returns:
        injected_labels: {uid: choice_letter}
        stats: aggregated stats useful for reward shaping:
            {
              "impute_count": int,
              "conf_mean": float,
              "conf_sum": float,
              "entropy_mean": float,
              "entropy_sum": float,
              "maxprob_mean": float,
              "maxprob_sum": float,
              (optional) "per_uid": {uid: {...}}
            }
    """
    if rng is None:
        rng = np.random.default_rng()

    if qid not in observed_dict:
        observed_dict[qid] = {}
    current_obs = observed_dict[qid]

    injected_labels = {}

    # aggregated stats
    conf_list = []
    ent_list = []
    maxp_list = []
    per_uid = {} if return_per_uid_stats else None

    # small helper: normalize + temperature scaling
    def _normalize_probs(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return None
        p = p / (s + 1e-12)
        return p

    def _apply_temp(p: np.ndarray, T: float) -> np.ndarray:
        # prob-based temperature scaling: p_T ∝ p^(1/T)
        if T is None or T == 1.0:
            return p
        pT = np.power(p, 1.0 / float(T))
        pT = pT / (pT.sum() + 1e-12)
        return pT

    for uid, info in (gnn_preds or {}).items():
        if not isinstance(info, dict):
            continue

        probs_raw = info.get("probs", None)
        if probs_raw is None:
            continue

        p = _normalize_probs(probs_raw)
        if p is None:
            continue

        K = int(p.shape[0])
        if K <= 0:
            continue

        # confidence filter uses normalized p (not temp-scaled)
        max_prob = float(np.max(p))
        dynamic_threshold = (1.0 / K) + float(confidence_margin)
        if max_prob < dynamic_threshold:
            continue

        labels = [chr(65 + i) for i in range(K)]

        # choose under temperature-scaled distribution (esp. for sampling)
        pT = _apply_temp(p, temp)

        if mode == "argmax":
            chosen_idx = int(np.argmax(pT))
        elif mode == "sampling":
            chosen_idx = int(rng.choice(np.arange(K), p=pT))
        else:
            raise ValueError("mode must be 'argmax' or 'sampling'")

        pseudo_ans = labels[chosen_idx]

        # stats computed under pT (consistent with the actual choice policy)
        chosen_conf = float(pT[chosen_idx])
        entropy = float(-np.sum(pT * np.log(pT + 1e-12)))
        maxp = float(np.max(pT))

        # write result
        current_obs[uid] = pseudo_ans
        injected_labels[uid] = pseudo_ans

        conf_list.append(chosen_conf)
        ent_list.append(entropy)
        maxp_list.append(maxp)

        if return_per_uid_stats:
            per_uid[uid] = {
                "chosen": pseudo_ans,
                "chosen_conf": chosen_conf,
                "entropy": entropy,
                "max_prob": maxp,
                "K": K,
            }

    # aggregate stats (stable even if none injected)
    impute_count = len(injected_labels)
    stats = {
        "impute_count": impute_count,
        "conf_mean": float(np.mean(conf_list)) if conf_list else 0.0,
        "conf_sum": float(np.sum(conf_list)) if conf_list else 0.0,
        "entropy_mean": float(np.mean(ent_list)) if ent_list else 0.0,
        "entropy_sum": float(np.sum(ent_list)) if ent_list else 0.0,
        "maxprob_mean": float(np.mean(maxp_list)) if maxp_list else 0.0,
        "maxprob_sum": float(np.sum(maxp_list)) if maxp_list else 0.0,
    }
    if return_per_uid_stats:
        stats["per_uid"] = per_uid

    return injected_labels, stats


import numpy as np

def compute_step_reward(eig: float, stats: dict, K: int,
                        lam: float = 0.5,
                        beta: float = 0.0,
                        use_multiplicative: bool = False) -> float:
    """
    Combine EIG with imputation reliability.

    stats should contain:
      - entropy_mean
      - impute_count (optional)
      - maxprob_mean (optional)
    """
    eig = float(eig)

    # entropy_norm in [0,1] (roughly), if probs are normalized
    logK = float(np.log(max(K, 2)))
    entropy = float(stats.get("entropy_mean", 0.0))
    entropy_norm = entropy / (logK + 1e-12)

    # optional coverage bonus (saturating)
    impute_count = float(stats.get("impute_count", 0.0))
    coverage = np.log1p(max(impute_count, 0.0))

    if use_multiplicative:
        reward = eig * (1.0 - entropy_norm) 
    else:
        reward = eig - lam * entropy_norm 

    return float(reward)


def update_history(history, qid, outcomes):
    if qid not in history: history[qid] = {}
    history[qid].update(outcomes)

import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm

# Detailed logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MCTS")

def select_queries_mcts(dataset, pool, nodes, Xavail, observed_real, predictor_gnn,
                       depth=3, n_iter=3, top_k=5, confidence_margin=0.0, rng=None):
    
    logger.info(f"\n" + "╔" + "═"*58 + "╗")
    logger.info(f"║ {'🚀 MCTS LOOK-AHEAD INITIALIZED':^56} ║")
    logger.info(f"╠" + "═"*58 + "╣")
    logger.info(f"║  Root Candidates: {top_k:<7} | MC Trials: {n_iter:<7} | Depth: {depth:<6} ║")
    logger.info(f"╚" + "═"*58 + "╝")
    
    # --- PHASE 1: PRUNING ---
    nodes_eval_initial = rng.choice(nodes, size=int(len(nodes) * 0.10), replace=False).tolist()
    initial_eig_dict = select_queries(dataset, pool, nodes_eval_initial, Xavail, observed_real, dataset.asked_queries)
    top_k_queries = sorted(initial_eig_dict, key=initial_eig_dict.get, reverse=True)[:top_k]
    
    logger.info(f"\n📍 [Phase 1] Top-{top_k} Seeds Selected via 1-Step EIG")
    for idx, qid in enumerate(top_k_queries):
        logger.info(f"   {idx+1}. {qid:<10} | Score: {initial_eig_dict[qid]:.6f}")

    q_values = {qid: initial_eig_dict[qid] for qid in top_k_queries}
    # q_values = {qid: 0.0 for qid in top_k_queries}
    future_gains_summary = {qid: 0.0 for qid in top_k_queries}

    # --- PHASE 2: PARALLEL TRIALS ---
    # [Monte Carlo Tree Search expansion showing a root node branching into multiple trial paths]
    for q_root in tqdm(top_k_queries, desc="MCTS Expansion"):
        total_future_gain = 0
        logger.info(f"\n🔭 Simulating Future Paths for: {q_root}")

        for i in range(n_iter):
            # ISOLATION: sim_observed is our 'Imaginary State'
            sim_observed = deepcopy(observed_real)
            sim_asked_queries = dataset.asked_queries + [(q_root, dataset.codebook[q_root]["question"])]
            cumulative_trial_gain = 0
            rollout_path_log = []
            
            # --- ROOT STEP IMPUTATION ---
            # GNN fills gaps based on what we 'learned' at root
            root_preds = predictor_gnn.predict_for_question(q_root, batch_size=8192)
            pseudo_y_root, root_stats = inject_pseudo_labels(q_root, root_preds, sim_observed, confidence_margin, mode='sampling', rng=rng)
            
            # CRITICAL: We update the state locally within this trial only
            if q_root not in sim_observed: sim_observed[q_root] = {}
            sim_observed[q_root].update(pseudo_y_root)
            
            current_Xavail = [q for q in Xavail if q != q_root]
            
            # --- PHASE 3: ROLLOUT ---
            # [Image of a Reinforcement Learning agent exploring a state-action tree]
            for d in range(depth):
                if not current_Xavail: break
                
                # 1. Update LLM Beliefs: How does the LLM see the world now with GNN's help?
                nodes_eval_rollout = rng.choice(nodes, size=int(len(nodes) * 0.05), replace=False).tolist()
                
                # 2. Selection: Pick best action in this imaginary state
                rollout_eig = select_queries(dataset, pool, nodes_eval_rollout, current_Xavail, 
                                             sim_observed, sim_asked_queries)
                # q_next = max(rollout_eig, key=rollout_eig.get)
                top_items = sorted(rollout_eig.items(), key=lambda x: x[1], reverse=True)[:3]
                qs, scores = zip(*top_items)
                scores = np.array(scores, dtype=np.float64)
                scores = scores - scores.max()        # 数值稳定
                probs = np.exp(scores)
                probs = probs / probs.sum()

                q_next = rng.choice(qs, p=probs)


                reward = rollout_eig[q_next]

                # 3. Transition: GNN Imputation based on q_next
                gnn_probs = predictor_gnn.predict_for_question(q_next, batch_size=8192)
                pseudo_y_next, next_stats = inject_pseudo_labels(q_next, gnn_probs, sim_observed, confidence_margin, mode='argmax', rng=rng)
                
                if q_next not in sim_observed: sim_observed[q_next] = {}
                sim_observed[q_next].update(pseudo_y_next)

                K = len(dataset.codebook[q_next]["options"])  # 或从 probs 长度取
                step_reward = compute_step_reward(reward, next_stats, K, lam=0.0, beta=0.00, use_multiplicative=False)
                cumulative_trial_gain += reward
                
                # cumulative_trial_gain += reward #+ 1.0 * next_stats["entropy_mean"]
                impute_count = len(pseudo_y_next)
                rollout_path_log.append(f"{q_next}(+{reward:.3f} | 🧩{impute_count})")
                
                # Advance rollout state
                current_Xavail = [q for q in current_Xavail if q != q_next]
                sim_asked_queries.append((q_next, dataset.codebook[q_next]["question"]))
            
            total_future_gain += cumulative_trial_gain
            logger.info(f"   ↳ Trial {i+1}: {' → '.join(rollout_path_log)} | Path Total: {cumulative_trial_gain:.4f}")
            
        avg_future_gain = total_future_gain / n_iter
        future_gains_summary[q_root] = avg_future_gain
        q_values[q_root] += avg_future_gain

    # --- FINAL SUMMARY ---
    # [Image of a data table comparing model performance metrics]
    logger.info(f"\n📊 MCTS FINAL RANKING")
    logger.info("━" * 85)
    logger.info(f"{'Query ID':<15} ┃ {'Base EIG':<15} ┃ {'Future Gain (Avg)':<20} ┃ {'Q-Value':<15}")
    logger.info("━" * 85)
    
    for qid in top_k_queries:
        logger.info(f"{qid:<15} ┃ {initial_eig_dict[qid]:<15.6f} ┃ {future_gains_summary[qid]:<20.6f} ┃ {q_values[qid]:<15.6f}")
    
    logger.info("━" * 85)
    best_q = max(q_values, key=q_values.get)
    logger.info(f"\n🏆 CHAMPION ACTION: {best_q}")
    logger.info(f"   Value: {q_values[best_q]:.6f} (Future boost: {future_gains_summary[best_q]:.4f})")
    
    return [best_q]

