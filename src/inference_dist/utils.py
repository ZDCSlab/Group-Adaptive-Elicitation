from collections import Counter
import torch
import json

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
