import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Dict, Any

class NeighborAggPrefix(nn.Module):
    def __init__(self, d_z: int, d_pair: int, d_lm: int, m_prefix: int = 16, attn_dim: int = None, proj_w: bool = True):
        super().__init__()
        h = attn_dim or d_z
        self.Wv = nn.Linear(d_z, h, bias=False) if proj_w else nn.Identity()
        self.Wu = nn.Linear(d_z, h, bias=False) if proj_w else nn.Identity()
        self.prefix = nn.Sequential(
            nn.Linear(d_pair, 4*d_lm), nn.GELU(),
            nn.Linear(4*d_lm, m_prefix*d_lm)
        )
        self.m, self.d_lm, self.h = m_prefix, d_lm, h

    def forward(self, Z_self, Z_neigh_flat, E_pair_flat, ptr):
        B = Z_self.size(0); q_all = self.Wv(Z_self)
        out = []
        for b in range(B):
            s, t = ptr[b].item(), ptr[b+1].item()
            if s == t:
                EvX = torch.zeros(E_pair_flat.size(-1), device=Z_self.device)
            else:
                kb = self.Wu(Z_neigh_flat[s:t])                 # [K,h]
                qb = q_all[b:b+1]                               # [1,h]
                e  = (kb @ qb.t()).squeeze(-1) / (self.h**0.5) # [K]
                a  = e.softmax(dim=0)                           # [K]
                Eb = E_pair_flat[s:t]                           # [K,d_pair]
                EvX = (a.unsqueeze(1) * Eb).sum(dim=0)          # [d_pair]
            P = self.prefix(EvX).view(self.m, self.d_lm)         # [m,d_lm]
            out.append(P)
        return torch.stack(out, dim=0)                           # [B,m,d_lm]

class SimplePairEncoder(nn.Module):
    """Replace with your real (Q, neighbor-answer) text encoder if available."""
    def __init__(self, num_nodes: int, num_questions: int, d_pair: int, d_hidden: int = 256):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, d_hidden)
        self.q_emb    = nn.Embedding(num_questions, d_hidden)
        self.proj = nn.Sequential(
            nn.Linear(2*d_hidden, 2*d_hidden), nn.GELU(),
            nn.Linear(2*d_hidden, d_pair)
        )

    def forward(self, neigh_ids_flat: torch.Tensor, q_ids_flat: torch.Tensor) -> torch.Tensor:
        if neigh_ids_flat.numel() == 0:
            return self.node_emb.weight.new_zeros((0, self.proj[-1].out_features))
        n = self.node_emb(neigh_ids_flat)
        q = self.q_emb(q_ids_flat)
        return self.proj(torch.cat([n, q], dim=-1))

def build_flat_neighbors_and_ptr(blk_neigh_lists: List[List[int]], blk_q_ids: torch.Tensor):
    device = blk_q_ids.device
    B = len(blk_neigh_lists)
    lengths = [len(x) for x in blk_neigh_lists]
    ptr = torch.empty(B+1, dtype=torch.long, device=device); ptr[0] = 0
    if B: ptr[1:] = torch.cumsum(torch.tensor(lengths, device=device), dim=0)
    if ptr[-1].item() == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                ptr)
    flat_neigh, flat_q = [], []
    for b, neighs in enumerate(blk_neigh_lists):
        if len(neighs) == 0: continue
        flat_neigh.extend(neighs)
        flat_q.extend([blk_q_ids[b].item()] * len(neighs))
    return (torch.tensor(flat_neigh, dtype=torch.long, device=device),
            torch.tensor(flat_q,    dtype=torch.long, device=device),
            ptr)

def mask_first_after_token(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    is_tok = (input_ids == token_id)
    m = torch.zeros_like(is_tok); m[:, 1:] = is_tok[:, :-1]
    return m

def splice_prefixes_per_block(tok_emb, input_ids, attention_mask,
                              neighbor_pos_list: List[List[int]],
                              prefix_list: List[torch.Tensor],
                              pad_token_id: int):
    B, T, d = tok_emb.size()
    new_embs, new_ids, new_attns = [], [], []
    cursor, maxL = 0, 0
    for b in range(B):
        pos_b = neighbor_pos_list[b]
        cur = 0
        parts_e, parts_i, parts_a = [], [], []
        for p in pos_b:
            parts_e.append(tok_emb[b, cur:p+1, :]); parts_i.append(input_ids[b, cur:p+1]); parts_a.append(attention_mask[b, cur:p+1])
            P = prefix_list[cursor]; cursor += 1
            parts_e.append(P)
            parts_i.append(torch.full((P.size(0),), pad_token_id, device=input_ids.device, dtype=input_ids.dtype))
            parts_a.append(torch.ones(P.size(0), device=attention_mask.device, dtype=attention_mask.dtype))
            cur = p + 1
        parts_e.append(tok_emb[b, cur:, :]); parts_i.append(input_ids[b, cur:]); parts_a.append(attention_mask[b, cur:])
        Eb = torch.cat(parts_e, dim=0); Ib = torch.cat(parts_i, dim=0); Ab = torch.cat(parts_a, dim=0)
        new_embs.append(Eb); new_ids.append(Ib); new_attns.append(Ab); maxL = max(maxL, Eb.size(0))
    def pad_vec(v, L): out = v.new_zeros((L, v.size(1))); out[:v.size(0)] = v; return out
    def pad_ids(v, L, pid): out = v.new_full((L,), pid, dtype=v.dtype); out[:v.size(0)] = v; return out
    def pad_attn(v, L): out = v.new_zeros((L,), dtype=v.dtype); out[:v.size(0)] = v; return out
    L = maxL
    new_emb  = torch.stack([pad_vec(e, L) for e in new_embs], dim=0)
    new_ids  = torch.stack([pad_ids(i, L, pad_token_id) for i in new_ids], dim=0)
    new_attn = torch.stack([pad_attn(a, L) for a in new_attns], dim=0)
    return new_emb, new_ids, new_attn


def neighboragg_loss_step(model, batch, Z_table, pair_encoder, aggregator, ans_token_id, pad_token_id):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    blk_node_ids = batch["blk_node_ids"].to(device)         # [SumK]
    blk_q_ids    = batch["blk_q_ids"].to(device)            # [SumK]
    blk_neigh_lists = batch["blk_neigh_lists"]              # List[List[int]]
    neighbor_pos_list = batch["neighbor_pos"]               # List[List[int]]

    neigh_ids_flat, q_ids_flat, ptr = build_flat_neighbors_and_ptr(blk_neigh_lists, blk_q_ids)
    Z_self  = Z_table[blk_node_ids]                                              # [SumK,d_z]
    Z_neigh = Z_table[neigh_ids_flat] if neigh_ids_flat.numel() else Z_self.new_zeros((0, Z_table.size(1)))
    E_pair  = pair_encoder(neigh_ids_flat, q_ids_flat) if neigh_ids_flat.numel() else Z_self.new_zeros((0, aggregator.prefix[0].in_features))

    P_blocks = aggregator(Z_self, Z_neigh, E_pair, ptr)                           # [SumK,m,d_lm]
    prefix_list = [P_blocks[i] for i in range(P_blocks.size(0))]

    tok_emb = model.model.embed_tokens(input_ids)                                  # [B,T,d_lm]
    new_emb, new_ids, new_attn = splice_prefixes_per_block(tok_emb, input_ids, attn_mask, neighbor_pos_list, prefix_list, pad_token_id)

    out = model(inputs_embeds=new_emb, attention_mask=new_attn)
    logits = out.logits                                                            # [B,L,V]

    m = mask_first_after_token(new_ids, ans_token_id)
    targets = new_ids[:, 1:].contiguous(); logits = logits[:, :-1, :].contiguous(); m = m[:, 1:].contiguous()
    sel = m.bool().view(-1)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1))[sel], targets.view(-1)[sel])
    return loss


def eval_split_neighboragg(model, dataset, Z_table, pair_encoder, aggregator, split, args, ANS_ID, PAD_ID):
    model.eval(); aggregator.eval(); pair_encoder.eval()
    total_loss, total_tokens = 0.0, 0
    steps = max(1, getattr(args, "eval_steps", 50))
    for _ in range(steps):
        batch = dataset.get_batch_neighboragg(split, args.batch_size, args.block_size)
        with torch.no_grad():
            loss = neighboragg_loss_step(model, batch, Z_table, pair_encoder, aggregator,
                                         ans_token_id=ANS_ID, pad_token_id=PAD_ID)
        # approximate token count by number of supervised positions
        B = batch["input_ids"].size(0)
        # recompute mask on the post-splice stream is expensive here; treat each batch as weight 1
        total_loss += float(loss.item()); total_tokens += 1
    return total_loss / max(1, total_tokens)
