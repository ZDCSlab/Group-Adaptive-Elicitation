from collections import Counter
import torch
import json
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple
import numpy as np

from collections import defaultdict
import math
import random
import os


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
    confidence_margin: float = 0.3  # <--- NEW: Additive margin above random chance
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



def probs_to_an(
    probs,   
    nodes,   
    labels: List[Any] = ["A", "B"],
) -> Dict[Hashable, Any]:

    ans: Dict[Hashable, Any] = {}
    probs_array = np.asarray(probs, dtype=np.float64)
    num_classes = len(labels)

    for i, nid in enumerate(nodes):
        row = probs_array[i]
        s = np.sum(row)
        
        # dynamic fallback, adapt to different lengths of labels
        if not np.isfinite(s) or s <= 1e-9: # add a small threshold to prevent precision issues
            idx = 0 # or random, or set to default class
        else:
            idx = int(np.argmax(row))
        
        ans[nid] = labels[idx]

    return ans


def get_gt_labels_for_qid(gold_map: Dict[tuple, str], target_qid: str) -> Dict[str, int]:
    """
    Extracts ground truth labels for a specific QID from the global gold map.
    Converts letter answers ('A', 'B') to integer indices (0, 1).
    """
    gt_labels = {}
    
    for (uid, qid), letter_ans in gold_map.items():
        if qid == target_qid:
            # Convert 'A' -> 0, 'B' -> 1, etc.
            label_idx = ord(letter_ans) - ord('A')
            gt_labels[uid] = label_idx
            
    return gt_labels

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



def build_group_candidates(nodes, node_label_dict):
    """
    node_label_dict: dict-like so that node_label_dict[node_id] -> group_id
    returns: dict[group_id] -> list[node_id]
    """
    groups = defaultdict(list)
    for n in nodes:
        g = node_label_dict.get(n, None)   # or raise if missing
        if g is not None:
            groups[g].append(n)
    return dict(groups)

def allocate_k_per_group(groups, k_total=None, k_each=None):
    """
    Exactly one of (k_total, k_each) must be provided.
    - k_each: fixed picks per group
    - k_total: distribute proportionally to group sizes (rounding + residual fix)
    returns: dict[group_id] -> k for that group
    """
    if (k_total is None) == (k_each is None):
        raise ValueError("Specify exactly one of k_total or k_each.")

    sizes = {g: len(v) for g, v in groups.items()}
    if k_each is not None:
        return {g: min(k_each, sizes[g]) for g in groups}

    # proportional allocation
    total = sum(sizes.values())
    if total == 0 or k_total == 0:
        return {g: 0 for g in groups}

    # initial floor allocation
    k_by_group = {g: min(len(groups[g]), math.floor(k_total * sizes[g] / total)) for g in groups}
    assigned = sum(k_by_group.values())
    # distribute remaining (largest fractional parts, but capped by group size)
    # compute fractional parts for tie-breaking
    fracs = sorted(
        ((g, (k_total * sizes[g] / total) - math.floor(k_total * sizes[g] / total)) for g in groups),
        key=lambda x: x[1],
        reverse=True,
    )
    i = 0
    while assigned < k_total and i < len(fracs):
        g = fracs[i][0]
        if k_by_group[g] < sizes[g]:
            k_by_group[g] += 1
            assigned += 1
        i += 1
    return k_by_group


def options_to_string(options: dict, prefix="Options: ", sep=", "):
    """
    Render {"1":"Yes","2":"No"} -> "Options: [1] Yes, [2] No"
    Sorts numerically if possible, else lexicographically.
    """
    def try_int(s):
        try: return int(s)
        except: return None

    # Normalize keys to strings for display, but sort by numeric value if possible
    items = []
    for k, v in options.items():
        ks = str(k).strip()
        kn = try_int(ks)
        items.append((ks, v, (0, kn) if kn is not None else (1, ks)))
    items.sort(key=lambda x: x[2])

    body = sep.join(f"[{ks}] {v}" for ks, v, _ in items)
    return f"{prefix}{body}"

def load_jsonl_as_dict_of_dict(path, key=None):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj[key]] = obj  
    return data


@torch.no_grad()
def batched_infer_with_candidates(
    model, dataset, split, device, batch_size, block_size,
    cand_ids_tensor=None, decode=True
):
    """
    Full batched inference: process the entire split in batches.

    Args:
        cand_ids_tensor: torch.LongTensor [K], candidate token IDs on same device.
        decode: if True, also decode tokens into strings (requires dataset.tokenizer).

    Returns:
        List of dicts, one per batch:
          {
            "input_ids": [B,T],
            "target_ids": [B,T],
            "pred_ids":   [B,T],
            "mask":       [B,T],
            "target_tokens": [[str]],  # optional
            "pred_tokens":   [[str]]   # optional
          }
    """
    model.eval()
    results = []

    num_samples = len(dataset.data_dict[split])
    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in range(num_batches):
        X, Y, gradient_mask = dataset.get_batch(split, batch_size, block_size) # TODO
        X = X.to(device); Y = Y.to(device)
        m = gradient_mask.to(device).to(torch.float32)

        logits = model(input_ids=X).logits  # [B,T,V]

        if cand_ids_tensor is not None:
            cand_ids_tensor = cand_ids_tensor.to(device)
            cand_logits = logits.index_select(dim=-1, index=cand_ids_tensor)  # [B,T,K]
            pred_idx = cand_logits.argmax(dim=-1)                             # [B,T]
            preds = cand_ids_tensor[pred_idx]                                 # [B,T]
        else:
            preds = logits.argmax(dim=-1)                                     # [B,T]

        batch_res = {
            "input_ids": X.detach().cpu(),
            "target_ids": Y.detach().cpu(),
            "pred_ids": preds.detach().cpu(),
            "mask": m.detach().cpu(),
        }

        if decode and hasattr(dataset, "tokenizer"):
            B, T = Y.shape
            tgt_strs, pred_strs = [], []
            for b in range(B):
                tgt_strs.append(dataset.tokenizer.decode(Y[b].tolist()))
                pred_strs.append(dataset.tokenizer.decode(preds[b].tolist()))
            batch_res["target_tokens"] = tgt_strs
            batch_res["pred_tokens"] = pred_strs

        results.append(batch_res)

    return results


# dist_predict.py
# dist_predict_group.py
from typing import Any, List, Tuple, Optional
import torch.distributed as dist
from accelerate import Accelerator

def dist_predict_batch_grouped(
    accelerator: Accelerator,
    subgroup: dist.ProcessGroup,          # group_model or group_iid
    participate: bool,                    # True if this rank belongs to subgroup
    predict_fn,                           # callable like model.predict_batch
    items: List[Any],                     # the global list to process
    *,
    shard_arg_name: str,                  # e.g., "nodes"
    world_broadcast: bool = True,         # broadcast outputs to all ranks after subgroup gather
    **kwargs
) -> List[Any]:
    """
    Shard `items` within the subgroup; only participating ranks run predict_fn.
    Gather results within subgroup, then broadcast to all ranks so everyone
    receives the same outputs (useful for main-process logic).
    Returns outputs ordered like `items` on ALL ranks.
    """
    # 1) Make sure everyone sees the same `items` object
    [items] = accelerator.broadcast_object_list([items])  # world group

    # 2) Build (index, item) pairs so we can restore order
    indexed = list(enumerate(items))

    # 3) Split across subgroup ranks only
    if participate:
        # get subgroup local rank order
        ranks = list(subgroup.ranks) if hasattr(subgroup, "ranks") else None
        # Fallback: compute subgroup ranks from world if needed
        # Simpler: use world split then mask by participate
        # We'll manually slice:
        world_size = accelerator.num_processes
        world_rank = accelerator.process_index
        # Filter indices owned by subgroup by modulo trick on subgroup size
        # Better: scatter via all_gather_object of indices; here use a simple chunking:
        # compute subgroup_size:
        subgroup_size = dist.get_world_size(group=subgroup)
        # compute this rank's subgroup_rank
        subgroup_rank = dist.get_rank(group=subgroup)
        # Round-robin assign
        my_indexed = [p for i, p in enumerate(indexed) if (i % subgroup_size) == subgroup_rank]
        my_items = [it for (_, it) in my_indexed]
    else:
        my_indexed = []
        my_items = []

    # 4) Local predict on subgroup participants
    local_out: List[Any] = []
    if participate and len(my_items) > 0:
        shard_kwargs = dict(kwargs)
        shard_kwargs[shard_arg_name] = my_items
        local_out = predict_fn(**shard_kwargs)  # must align one-to-one with my_items

    # 5) Pair with global indices and gather within subgroup
    local_pairs = list(zip([i for (i, _) in my_indexed], local_out))
    gathered_pairs: List[Tuple[int, Any]] = []
    dist.all_gather_object(gathered_pairs, local_pairs, group=subgroup)  # list concat across subgroup

    # Flatten + restore order
    flat = [p for lst in gathered_pairs for p in lst]
    flat.sort(key=lambda z: z[0])
    outputs = [y for _, y in flat]

    # 6) Optionally broadcast outputs from the subgroup leader to the entire world
    if world_broadcast:
        # Pick subgroup leader (rank-0 within subgroup). We need its *world* rank to do world-broadcast.
        leader_world_rank = None
        # A simple way: let everyone set outputs=None except participants; then use world broadcast from rank-0 (world)
        # Instead, we can just world-broadcast via Accelerate from current process, but only one should be the source.
        # Easiest: pack outputs only on world rank 0; others receive it.
        if accelerator.is_main_process:
            src_payload = outputs
        else:
            src_payload = None
        [outputs] = accelerator.broadcast_object_list([src_payload])
    return outputs

NodeId = Hashable
QueryId = Hashable

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


def ans_to_nei_dict(
    ans,  
    nodes,      # same length as probs
    neighbor,
    labels=["A", "B"],      # optional: e.g., ("A","B")
) -> Dict[Hashable, Hashable]:

    neigh_ans_list = dict()
    for idx, v in enumerate(nodes):
            neigh_ans = _neighbors_answers(v, ans, neighbor[v])
            neigh_ans_list[v] = neigh_ans
    return neigh_ans_list