import math
import torch
import torch.nn.functional as F
import json 
import os
from torch.optim.lr_scheduler import LambdaLR

# --- Load the mapping: Question ID -> Option Tokens ---
def load_qid_to_tokenized_options(jsonl_path, tokenizer, strict_single_token=True, id_key="id"):
    """
    Loads options for individual questions.
    """
    qid_to_options = {}
    print(f"Loading variable options from {jsonl_path} using id_key='{id_key}'...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            data = json.loads(line)
            raw_id = data.get(id_key)
            if raw_id is None: continue
            qid = str(raw_id) 

            option_labels = sorted(list(data.get('options', {}).keys()))
            
            token_ids = []
            for label in option_labels:
                ids = tokenizer.encode(label, add_special_tokens=False)
                if len(ids) != 1:
                    if strict_single_token: continue
                    else: ids = [ids[-1]]
                token_ids.append(ids[0])
            
            if token_ids:
                qid_to_options[qid] = token_ids

    print(f"Loaded options for {len(qid_to_options)} questions.")
    return qid_to_options

# --- Load training data ---
def _load_jsonl_qtexts(path: str, text_key: str = "q_text_all", qids_key: str = "question_ids"):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            data.append({
                "text": rec.get(text_key, ""), 
                "qids": rec.get(qids_key, []) 
            })
    return data

def load_dataset(data_dir: str):
    paths = {
        "train": os.path.join(data_dir, "train.jsonl"),
        "val":   os.path.join(data_dir, "val.jsonl"),
        "test":  os.path.join(data_dir, "test.jsonl"),
    }
    texts = {}
    for split, p in paths.items():
        if os.path.exists(p):
            texts[split] = _load_jsonl_qtexts(p)
        else:
            texts[split] = []
    return texts


# --- Loss function ---
def get_loss_variable_candidates(model, X, Y, gradient_mask, attention_mask, batch_cand_ids, device=None):
    """
    Args:
        batch_cand_ids: [B, T, K]. IGNORED in General Training logic.
    """
    outputs = model(input_ids=X, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]
    
    # 3. Prepare Labels for Standard Generation (Output Side)
    labels = Y.clone()
    
    # Apply the Gradient Mask (Masks History/Prompt)
    labels[gradient_mask == 0] = -100
    
    # SAFETY: Also ensure actual padding tokens in Y are masked to -100
    # (In case gradient_mask was 1 at a padding position by mistake)
    if hasattr(model, "config") and hasattr(model.config, "pad_token_id"):
         pad_id = model.config.pad_token_id
         if pad_id is not None:
             labels[Y == pad_id] = -100

    # 4. Compute Standard Cross Entropy Loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        labels.view(-1), 
        ignore_index=-100, 
        reduction='mean'
    )

    return loss


# --- Estimate loss and accuracy ---
@torch.no_grad()
def estimate_loss_and_acc_variable(
    model, dataset, split, device, batch_size, tokenizer, eval_iters=100, debug=False
):
    model.eval()
    losses = []
    gen_accs = [] # Generative accuracy (Strict)
    mc_accs = []  # Multiple choice accuracy (Loose)

    for k in range(eval_iters):
        try:
            batch_data = dataset.get_batch(split, batch_size)
            # Ensure we unpack attention_mask correctly
            if len(batch_data) == 5:
                X, Y, gradient_mask, attention_mask, batch_cand_ids = batch_data
            else:
                break
        except Exception as e:
            print(f"Eval Error: {e}")
            break

        X, Y = X.to(device), Y.to(device)
        gradient_mask = gradient_mask.to(device)
        attention_mask = attention_mask.to(device)
        batch_cand_ids = batch_cand_ids.to(device)

        # --- Debug section (Runs only for the first batch) ---
        if k == 0: 
            print("\n" + "="*40)
            print(" DEBUG DIAGNOSTICS (Sample 0)")
            print("="*40)
            
            # 1. Check Input Format
            print(f"[Input Prompt]:\n{tokenizer.decode(X[0], skip_special_tokens=False)}\n")
            
            # 2. Check Answer Alignment & Candidates
            valid_indices = (gradient_mask[0] > 0).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                t = valid_indices[0].item() # Check the first valid answer position
                
                true_label_id = Y[0, t].item()
                true_label_str = tokenizer.decode([true_label_id])
                
                # Check what the model strictly predicts
                # We need to run a quick inference just for this debug print
                # (We do the full batch later, this is just for the log)
                with torch.no_grad():
                    temp_logits = model(input_ids=X[0:1], attention_mask=attention_mask[0:1]).logits
                    pred_id = temp_logits[0, t].argmax().item()
                    pred_str = tokenizer.decode([pred_id])

                cands = batch_cand_ids[0, t, :]
                valid_cands = [idx.item() for idx in cands if idx.item() != -100]
                decoded_cands = [tokenizer.decode([x]) for x in valid_cands]
                
                print(f"[Timestep t={t}]")
                print(f"  True Label: ID={true_label_id} | Str='{true_label_str}'")
                print(f"  Model Pred: ID={pred_id} | Str='{pred_str}'")
                print(f"  Candidates: {decoded_cands}")
                print(f"  Cand IDs:   {valid_cands}")
                
                if true_label_id not in valid_cands:
                    print("  !!! CRITICAL WARNING: True Label is NOT in Candidates! MC Acc will be 0.")
                
                if pred_id in valid_cands:
                    print("  [OK] Model prediction is inside the candidate list.")
                else:
                    print("  [INFO] Model prediction is OUTSIDE candidates (Gen Acc=0, but MC Acc might be OK).")

            else:
                print("[WARNING] No valid gradient_mask found in Sample 0.")
            print("="*40 + "\n")
        # -----------------------------------------------------

        # --- MAIN EVALUATION LOGIC ---

        # Forward Pass (WITH attention_mask)
        logits = model(input_ids=X, attention_mask=attention_mask).logits # [B, T, V]

        # 1. Generative metrics
        labels = Y.clone()
        labels[gradient_mask == 0] = -100
        
        # Loss value
        loss_val = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        losses.append(loss_val.item())

        # Generative accuracy (Strict)
        active_mask = (labels != -100)
        if active_mask.any():
            preds = logits[active_mask].argmax(dim=-1)
            gen_accs.append((preds == labels[active_mask]).float().mean().item())

        # 2. Multiple choice metrics
        gather_ids = batch_cand_ids.clone()
        gather_ids[gather_ids == -100] = 0
        
        # Gather logits only for the candidates
        cand_logits = torch.gather(logits, 2, gather_ids) 
        
        # Mask out padding candidates (to avoid -inf)
        cand_logits = cand_logits.masked_fill(batch_cand_ids == -100, -1e4)

        # Find the index of the true label inside the candidates list
        matches = (batch_cand_ids == Y.unsqueeze(-1))
        target_local_idx = matches.long().argmax(dim=-1) # [B, T]
        
        elig = (gradient_mask > 0)
        if elig.any():
            mc_preds = cand_logits[elig].argmax(dim=-1)
            mc_accs.append((mc_preds == target_local_idx[elig]).float().mean().item())

    out_loss    = float(sum(losses) / len(losses)) if losses else 0.0
    out_gen_acc = float(sum(gen_accs) / len(gen_accs)) if gen_accs else 0.0
    out_mc_acc  = float(sum(mc_accs) / len(mc_accs)) if mc_accs else 0.0
    
    model.train()
    return out_loss, out_gen_acc, out_mc_acc


# --- LR schedulers ---
def get_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
):
    num_warmup_steps = int(max(0, num_warmup_steps))
    num_training_steps = int(max(1, num_training_steps))
    denom = max(1, num_training_steps - num_warmup_steps)

    def lr_lambda(current_step: int):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # cosine decay
        progress = float(current_step - num_warmup_steps) / float(denom)
        # clamp to [0, 1]
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(it, warmup_iters = 2000, lr_decay_iters = 600000, min_lr = 6e-5, learning_rate = 6e-4):
    if it < warmup_iters: 
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)