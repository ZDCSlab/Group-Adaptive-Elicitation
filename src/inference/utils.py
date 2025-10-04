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
