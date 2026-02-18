from collections import Counter
import torch
import json
import numpy as np
import random
import os


def set_all_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Add these for absolute consistency:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If using PyTorch Geometric / Scatter operations:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
        torch.use_deterministic_algorithms(True, warn_only=True)

def load_jsonl_as_dict(path, key='id'):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj[key]] = obj  
    return data

def diagnostic_imputation(dataset_llm, q_selected, selected_users, gold_map_gnn):
    """
    diagnose the quality of the pseudo-labels injected in the current round
    """
    # 1. extract all records in the LLM observation dictionary for the current question
    current_obs = dataset_llm.observed_dict.get(q_selected, {})
    selected_set = set(selected_users) # exclude the real nodes queried in this round
    
    injected_uids = [uid for uid in current_obs if uid not in selected_set]
    
    correct = 0
    total = len(injected_uids)
    
    # 2. compare with the ground truth in gold_map_gnn
    for uid in injected_uids:
        pseudo_ans = current_obs[uid]
        # gold_map_gnn is stored as {(uid, qid): 'A'}
        gt_ans = gold_map_gnn.get((uid, q_selected)) 
        
        if gt_ans is not None and pseudo_ans == gt_ans:
            correct += 1
            
    precision = correct / total if total > 0 else 0
    if len(dataset_llm.graph.nodes) - len(selected_set) > 0:
        coverage = total / (len(dataset_llm.graph.nodes) - len(selected_set))
    else:
        coverage = 1.0
    
    return {
        "q_id": q_selected,
        "num_injected": total,
        "precision": precision,
        "coverage": f"{coverage:.2%}"
    }

def inject_pseudo_labels(
    dataset_llm, 
    q_selected: str, 
    gnn_preds: dict, 
    selected_users: list, 
    confidence_margin: float = 0.0  
) -> int:
    """
    Injects pseudo-labels using a Dynamic Threshold based on Option Space.
    Threshold = (1 / Num_Options) + confidence_margin
    
    Examples with margin=0.3:
      - 2 Options (Yes/No): Threshold = 0.5 + 0.3 = 0.80
      - 3 Options:          Threshold = 0.33 + 0.3 = 0.63
      - 5 Options:          Threshold = 0.20 + 0.3 = 0.50
    """
    
    selected_set = set(selected_users)
    injected_count = 0

    if q_selected not in dataset_llm.observed_dict:
        dataset_llm.observed_dict[q_selected] = {}
        
    current_qid_observations = dataset_llm.observed_dict[q_selected]

    for uid, info in gnn_preds.items():
        
        # A. Skip Ground Truth users
        if uid in selected_set: continue
        
        # B. Get Probs
        probs = info.get("probs", [])
        if not probs: continue
            
        # --- C. DYNAMIC THRESHOLD CALCULATION ---
        K = len(probs) # Number of options
        if K == 0: continue
        random_chance = 1.0 / K
        dynamic_threshold = random_chance + confidence_margin
        max_prob = max(probs)
        if max_prob < dynamic_threshold:
            continue
        # D. Inject
        pseudo_ans = info.get("choice_letter")
        
        if pseudo_ans:
            current_qid_observations[uid] = pseudo_ans
            injected_count += 1
            
    return injected_count
