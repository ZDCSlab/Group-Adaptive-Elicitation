from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Union, Callable, Sequence, List, Any
import numpy as np
import torch
from collections import Counter
import copy
from tqdm import tqdm

NodeId = Hashable
QueryId = Hashable
Label = Union[int, float]



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

    q_star = max(EIG, key=EIG.get)
    if verbose:
        print(f"\n✅ Selection Complete")
        print(f"Selected: {q_star} | Max EIG: {EIG[q_star]:.6f}")
        
        # Sort and print the top 5 candidates for comparison
        sorted_eig = sorted(EIG.items(), key=lambda x: x[1], reverse=True)
        print("📊 Top 5 EIG Candidates:")
        for qid, score in sorted_eig[:5]:
            print(f"  - {qid}: {score:.6f}")

    return [q_star]


