import math
import torch
import torch.nn.functional as F
import json 
from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import trange


def print_tokens_shape(tokens_list):
    """
    Print the shape of nested tokens: [paragraphs][sentences][tokens]
    """
    num_paragraphs = len(tokens_list)
    print(f"Paragraphs: {num_paragraphs}")
    
    for i, paragraph in enumerate(tokens_list):
        num_sentences = len(paragraph)
        sentence_lengths = [len(sentence) for sentence in paragraph]
        print(f"  Paragraph {i}: {num_sentences} sentences")
        print(f"    Token lengths per sentence: {sentence_lengths}")


def load_dataset(data_dir):
    with open(data_dir) as f:
        data_dict = json.load(f)
    return data_dict


def get_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,   # 最小学习率 = base_lr * ratio
    last_epoch: int = -1,
):
    """
    Cosine decay with warmup + nonzero min_lr.

    Args:
        optimizer: torch optimizer
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率比例，比如 0.1 表示 min_lr = 0.1 * base_lr
        last_epoch: for PyTorch scheduler API
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦退火
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 从 1.0 衰减到 min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(it, warmup_iters = 2000, lr_decay_iters = 600000, min_lr = 6e-5, learning_rate = 6e-4):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters: 
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_loss_with_candidates(
    model,
    X, Y,
    gradient_mask,
    nmask,
    *,
    mask: bool = True,
    return_scalar: bool = True,
    cand_ids_tensor: torch.Tensor = None, 
    device = None
):
    """
    Compute cross-entropy loss for a language model, optionally restricted to a candidate set.

    Args:
        model: a HuggingFace-style model returning logits [B, T, V].
        X: [B, T] input token IDs.
        Y: [B, T] target token IDs (shifted by one).
        gradient_mask: [B, T] binary mask where 1 marks supervised positions (<Answer> tokens).
        mask: if True, only compute loss on masked positions.
        return_scalar: if True, return scalar mean loss; else return per-token loss.
        cand_ids_tensor: optional [C] LongTensor of candidate token IDs.
                         If provided, cross-entropy is restricted to this candidate set.

    Returns:
        Scalar loss (float tensor) if return_scalar=True,
        otherwise [B, T] tensor of per-token losses (masked if requested).
    """

    outputs = model(input_ids=X)
    # Compute embeddings
    B, T = X.shape
    # emb = model.module.get_input_embeddings()(X)
    # causal_mask = torch.tril(torch.ones((T, T))).unsqueeze(0).expand(B, -1, -1).to(device)  # [B,T,T]
    # full_mask = causal_mask * nmask  # [B,T,T]
    # attention_mask = full_mask.unsqueeze(1)     # [B,1,T,T]s
    # outputs = model(inputs_embeds=emb, attention_mask=attention_mask)

    logits = outputs.logits  # [B,T,V]

    # Case 1: Full-vocabulary loss
    if cand_ids_tensor is None:
        per_tok_ce = F.cross_entropy(
            logits.view(-1, V).float(),     # [B*T, V]
            Y.view(-1),                     # [B*T]
            reduction="none",
        ).view(B, T)                        # [B, T]

        if not mask:
            return per_tok_ce.mean() if return_scalar else per_tok_ce

        # Apply supervision mask (<Answer> tokens only)
        m = gradient_mask.to(dtype=per_tok_ce.dtype)  # [B, T]
        masked = per_tok_ce * m

        if not return_scalar:
            return masked

        denom = m.sum()
        if denom.item() == 0:
            return torch.zeros((), device=device, dtype=per_tok_ce.dtype)
        return masked.sum() / denom

    # Case 2: Candidate-restricted loss
    else:
        # Extract logits only at candidate IDs → [B, T, C]
        cand_logits = logits.index_select(dim=-1, index=cand_ids_tensor)

        # Build target indices relative to candidate set
        #   For Y[b,t] that is in cand_ids_tensor, get its index in [0..C-1]
        #   If not in candidates, mark as -100 (ignored by cross_entropy)
        mapping = torch.full(
            (logits.size(-1),),  # V = vocab size
            fill_value=-100,
            dtype=torch.long,
            device=Y.device,
        )
        mapping[cand_ids_tensor] = torch.arange(len(cand_ids_tensor), device=Y.device)
        # Step 2. Apply mapping to Y in one shot
        Y_cand = mapping[Y]   # [B, T], values in [0..C-1] or -100

        per_tok_ce = F.cross_entropy(
            cand_logits.view(-1, cand_logits.size(-1)).float(),  # [B*T, C]
            Y_cand.view(-1),                                     # [B*T]
            reduction="none",
            ignore_index=-100,
        ).view(B, T)

        if not mask:
            return per_tok_ce.mean() if return_scalar else per_tok_ce

        # Apply supervision mask (<Answer> tokens only)
        m = gradient_mask.to(dtype=per_tok_ce.dtype)  # [B, T]
        masked = per_tok_ce * m

        if not return_scalar:
            return masked

        denom = m.sum()
        if denom.item() == 0:
            return torch.zeros((), device=device, dtype=per_tok_ce.dtype)
        return masked.sum() / denom


@torch.no_grad()
def estimate_loss_and_acc_with_candidates(
    model, dataset, split, device, batch_size, block_size, eval_iters=100,
    cand_ids_tensor=None
):
    """
    Returns: (avg_ce_loss, avg_token_acc) over eval_iters.

    Behavior:
      - Candidate-only mode (cand_ids_tensor provided):
          Restrict logits to candidate IDs.
          Keep only positions where gold ∈ candidates.
          Compute CE and ACC only on <Answer> positions.
      - Full-vocab mode (cand_ids_tensor=None):
          Compute CE/ACC at <Answer> positions over full vocabulary.
    """
    model.eval()
    losses = torch.zeros(eval_iters, dtype=torch.float32)
    accs   = torch.zeros(eval_iters, dtype=torch.float32)

    if cand_ids_tensor is not None:
        cand_ids_tensor = cand_ids_tensor.to(device)
        K = cand_ids_tensor.numel()
        # Precompute vocab → candidate index map
        V = model.config.vocab_size
        mapping = torch.full((V,), -100, dtype=torch.long, device=device)
        mapping[cand_ids_tensor] = torch.arange(K, device=device)

    for k in trange(eval_iters, desc=f"Eval on {split}", unit="iter"):
        X, Y, gradient_mask, nmask = dataset.get_batch(split, batch_size, block_size)
        X, Y, m, nmask = X.to(device), Y.to(device), gradient_mask.to(device).float(), nmask.to(device)

        logits = model(input_ids=X).logits  # [B,T,V]
        # B, T = X.shape
        # emb = model.module.get_input_embeddings()(X)
        # causal_mask = torch.tril(torch.ones((T, T))).unsqueeze(0).expand(B, -1, -1)  # [B,T,T]
        # full_mask = causal_mask * nmask  # [B,T,T]
        # attention_mask = full_mask.unsqueeze(1)     # [B,1,T,T]s
        # logits = model(inputs_embeds=emb, attention_mask=attention_mask).logits 


        elig = (m > 0)                      # supervised positions
        N = int(elig.sum().item())

        if N == 0:
            losses[k] = accs[k] = 0.0
            continue

        logits_e  = logits[elig]   # [N,V]
        targets_e = Y[elig]        # [N]

        # ---------- Candidate-only mode ----------
        if cand_ids_tensor is not None:
            logits_cand = logits_e.index_select(dim=-1, index=cand_ids_tensor)  # [N,K]
            target_map = mapping[targets_e]  # [N], in [0..K-1] or -100
            valid = (target_map != -100)

            if valid.any():
                per_tok_ce = F.cross_entropy(
                    logits_cand.float(), target_map, reduction="none", ignore_index=-100
                )
                ce = per_tok_ce[valid].mean()
                pred = logits_cand.argmax(dim=-1)
                acc = (pred[valid] == target_map[valid]).float().mean()
            else:
                ce = acc = torch.tensor(0.0, device=device)

        # ---------- Full-vocab mode ----------
        else:
            per_tok_ce = F.cross_entropy(
                logits_e.float(), targets_e, reduction="none"
            )
            ce = per_tok_ce.mean()
            pred = logits_e.argmax(dim=-1)
            acc = (pred == targets_e).float().mean()

        losses[k] = ce.item()
        accs[k]   = acc.item()

    model.train()
    return losses.mean().item(), accs.mean().item()
