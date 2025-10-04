import torch 
import random
import torch.nn.functional as F
import os 
import json 
import re
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional
from typing import List, Dict, Tuple


# def shuffle_row_sync(row_tok, row_qid, block_size):

#     pairs = list(zip(row_tok, row_qid))  # [(tok_sent, qid_sent), ...]
#     random.shuffle(pairs)

#     flat_tok = [t for (t_sent, _) in pairs for t in t_sent]
#     flat_qid = [q for (_, q_sent) in pairs for q in q_sent]

#     tok = torch.tensor(flat_tok, dtype=torch.long)
#     qid = torch.tensor(flat_qid, dtype=torch.long)
#     L = min(tok.numel(), qid.numel(), block_size)
#     return tok[:L], qid[:L]


def shuffle_row(row, block_size):
    random.shuffle(row)
    row = torch.tensor([item for sublist in row for item in sublist])
    return row[:block_size]


def get_batch(
    data,
    batch_size: int,
    block_size: int,
    answer_token: int,
    neighbor_token: int = None,
    prefix_ids=None,
    device=None,
    mode: str = "all",   # "all" or "no_self_neighbor"
):
    """
    Sample a mini-batch of token sequences.

    Returns:
        X: [B, block_size] input tokens
        Y: [B, block_size] target tokens (shifted by one)
        gradient_mask: [B, block_size] 1 where token == <Answer>
        neighbor_mask: [B, block_size, block_size], attention mask:
                       neighbor_mask[b, t_ans, j] = 0 means:
                       when predicting at t_ans, token j is invisible.
    """
    rows = []
    tries = 0
    max_tries = batch_size * 3

    # --- Sample rows ---
    while len(rows) < batch_size and tries < max_tries:
        idx = int(torch.randint(0, len(data), (1,)).item())
        row = shuffle_row(data[idx], block_size + 1)
        if not torch.is_tensor(row):
            row = torch.tensor(row, dtype=torch.long)
        if row.numel() == block_size + 1 and (row[:-1] == answer_token).any():
            rows.append(row)
        tries += 1

    while len(rows) < batch_size:
        idx = int(torch.randint(0, len(data), (1,)).item())
        row = shuffle_row(data[idx], block_size + 1)
        if torch.is_tensor(row) and row.numel() == block_size + 1:
            rows.append(row)

    batch = torch.stack(rows)  # [B, L]

    # --- Build tensors ---
    X = batch[:, :-1].contiguous()          # [B, block_size]
    Y = batch[:,  1:].contiguous()          # [B, block_size]
    gradient_mask = (X == answer_token).long()

    B, T = X.shape
    neighbor_mask = torch.ones((B, T, T), dtype=torch.long, device=X.device)

    if mode == "no_self_neighbor" and neighbor_token is not None:
        for b in range(B):
            ans_pos = (X[b] == answer_token).nonzero(as_tuple=True)[0]  # positions of <Answer>
            for pos in ans_pos:
                j = pos - 1
                # scan left until another <Answer> or start of sequence
                while j >= 0 and X[b, j] != answer_token:
                    if X[b, j] == neighbor_token:
                        # ✅ mask <Neighbor> itself
                        neighbor_mask[b, pos, j] = 0
                        # ✅ mask everything after <Neighbor> until reaching <Answer> (pos)
                        k = j + 1
                        while k < pos and X[b, k] != answer_token:
                            neighbor_mask[b, pos, k] = 0
                            k += 1
                    j -= 1

    return X, Y, gradient_mask, neighbor_mask



# def get_batch(
#     data, qid_data,                      # 新增：qid_data 与 data 同 shape
#     batch_size, block_size, answer_token,
#     prefix_ids=None, pad_id=0, device=None
# ):
#     """
#     逻辑与无 qid 的旧版完全一致，仅在每处取 row 时同步取 row_qid。
#     要求：你已有 `shuffle_row_sync(sample_tok, sample_qid, target_len)`，
#     返回长度恰为 target_len (= block_size+1) 的 (row_tok, row_qid)，二者已对齐。
#     """
#     rows_tok, rows_qid = [], []
#     tries = 0
#     max_tries = batch_size * 3  # 避免极端数据死循环
#     target_len = block_size + 1

#     # ---- 采样阶段（含监督位过滤）----
#     while len(rows_tok) < batch_size and tries < max_tries:
#         idx = int(torch.randint(0, len(data), (1,)).item())   # -> 标量索引

#         # 与无 qid 版本唯一不同：同步 shuffle
#         row_tok, row_qid = shuffle_row_sync(data[idx], qid_data[idx], target_len)

#         if not torch.is_tensor(row_tok):
#             row_tok = torch.tensor(row_tok, dtype=torch.long)
#         if not torch.is_tensor(row_qid):
#             row_qid = torch.tensor(row_qid, dtype=torch.long)

#         # 长度必须匹配；且 X=row[:-1] 中至少有一个 <Answer> 监督位
#         if row_tok.numel() == target_len and row_qid.numel() == target_len \
#            and (row_tok[:-1] == answer_token).sum().item() >= 1:
#             rows_tok.append(row_tok)
#             rows_qid.append(row_qid)

#         tries += 1

#     # ---- 若极端情况下没抽满，降级补齐（不做监督位过滤）----
#     while len(rows_tok) < batch_size:
#         idx = int(torch.randint(0, len(data), (1,)).item())
#         row_tok, row_qid = shuffle_row_sync(data[idx], qid_data[idx], target_len)
#         if torch.is_tensor(row_tok) and torch.is_tensor(row_qid) \
#            and row_tok.numel() == target_len and row_qid.numel() == target_len:
#             rows_tok.append(row_tok)
#             rows_qid.append(row_qid)

#     batch_tok = torch.stack(rows_tok)  # [B, L]
#     batch_qid = torch.stack(rows_qid)  # [B, L]，与 batch_tok 对齐

#     # ---- prefix（保持与旧版一致；qid 前缀用 0）----
#     if prefix_ids is not None and len(prefix_ids) > 0:
#         if not torch.is_tensor(prefix_ids):
#             prefix_ids = torch.tensor(prefix_ids, dtype=torch.long)
#         if device is None:
#             device = batch_tok.device

#         prefix_tok = prefix_ids.to(device).unsqueeze(0).expand(batch_tok.size(0), -1)  # [B, P]
#         prefix_qid = torch.zeros_like(prefix_tok)  # 前缀部分的 qid 填 0

#         batch_tok = torch.cat([prefix_tok, batch_tok.to(device)], dim=1)  # [B, P+L]
#         batch_qid = torch.cat([prefix_qid, batch_qid.to(device)], dim=1)  # [B, P+L]

#         # 左截断，保留右侧（确保 <Answer>→gold 邻接仍在末尾区域）
#         target_len = block_size + 1
#         if batch_tok.size(1) > target_len:
#             batch_tok = batch_tok[:, -target_len:]
#             batch_qid = batch_qid[:, -target_len:]

#     # ---- 组 X, Y, mask（与旧版一致，新增 qid_mask）----
#     X = batch_tok[:, :-1].contiguous()          # [B, block_size]
#     Y = batch_tok[:,  1:].contiguous()          # [B, block_size]
#     gradient_mask = (X == answer_token).long()  # [B, block_size]
#     qid_mask = batch_qid[:, :-1].contiguous()   # [B, block_size] —— 与 X 对齐

#     return X, Y, gradient_mask, qid_mask


class TextDataClass():
    def __init__(self, text_data_dict, tokenizer, mode):
        self.tokenizer = tokenizer
        # tokens
        self.answer_token  = tokenizer("<Answer>")['input_ids'][-1]
        try:
            self.neighbor_token = tokenizer("<Neighbor>", add_special_tokens=False)["input_ids"][-1]
        except Exception:
            self.neighbor_token = tokenizer.convert_tokens_to_ids("<Neighbor>")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
                      else tokenizer.convert_tokens_to_ids("[PAD]")

        # original dict & tokenized splits
        self.raw_dict  = text_data_dict
        self.data_dict = self.get_token_dict(text_data_dict)

        self.mode = mode

        # SYSTEM_PROMPT = (
        #     "Given the current question and options, a history of "
        #     "<question–neighbor–answer> pairs, and the current neighbors' answers, infer the user's current answer. "
        #     "Output exactly ONE candidate option ID. No extra text."
        # )

        # self.prefix_ids = tokenizer(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]


    def build_cand_ids_tensor(self, tokenizer, candidate_strs, device="cpu", strict_single_token=True):
        cand_ids = []
        for s in candidate_strs:
            ids = tokenizer(s, add_special_tokens=False).input_ids
            if len(ids) != 1:
                if strict_single_token:
                    raise ValueError(f"Candidate {s!r} tokenized into multiple tokens: {ids}")
                else:
                    # skip multi-token candidate
                    continue
            cand_ids.append(ids[0])
        
        if len(cand_ids) == 0:
            raise ValueError("No valid single-token candidates found.")
        
        return torch.tensor(cand_ids, dtype=torch.long, device=device)


    def get_batch(self, split, batch_size, block_size):
        
        return get_batch(self.data_dict[split], batch_size, block_size, self.answer_token, prefix_ids=None, mode=self.mode)
    

    def turn_text_into_tokens(self, text):
        text_list = [sentence for sentence in text.strip("<EOP>").split("<EOP>") if sentence]
        text_list = [text_list[i].strip("<EOS>").split("<EOS>") for i in range(len(text_list))]
        tokens_list = [self.tokenizer(text_list[i])['input_ids'] for i in range(len(text_list))]
        return tokens_list
    
    def get_token_dict(self, text_data_dict):
        tokens_dict = {}
        for key, value in text_data_dict.items():
            if key in ['train', 'val', 'test']:
                tokens_dict[key] = self.turn_text_into_tokens(value)
                
        return tokens_dict
    

    # def _strip_exact(self, s: str, token: str) -> str:
    #     if not s: return s
    #     L = len(token)
    #     while s.startswith(token): s = s[L:]
    #     while s.endswith(token):   s = s[:-L]
    #     return s


    # def turn_text_into_tokens(self, text: str) -> Tuple[List[List[int]], List[List[int]], Dict[int, List[int]]]:
    #     entity = [seg.strip() for seg in self._strip_exact(text, "<EOP>").split("<EOP>") if seg and seg.strip()]

    #     token_list, qid_list = [], []
   
    #     for entity_id, entity_text in enumerate(entity):
    #         sentences = [s.strip() for s in self._strip_exact(entity_text, "<EOS>").split("<EOS>") if s and s.strip()]
          
    #         tokens_per_q, qids_per_q = [], []
    #         for q_id, q_text in enumerate(sentences):
    #             enc = self.tokenizer(q_text, add_special_tokens=False)
    #             tok_ids: List[int] = enc["input_ids"]
    #             tokens_per_q.append(tok_ids) 
    #             # print('q_text', q_text)
    #             # print('nums', nums)
    #             # print('qids_per_q', qids_per_q)
    #             # input()
    #         token_list.append(tokens_per_q)

    #     return token_list


    
