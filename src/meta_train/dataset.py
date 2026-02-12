import torch
from typing import List, Optional, Dict


class QTextDataset:
    def __init__(self, text_data, tokenizer, block_size: int, option_dict: Dict = None, device=None):
        self.texts = text_data
        self.splits = list(text_data.keys()) 
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.option_dict = option_dict 
        self.device = device
        
        self.answer_token_id  = tokenizer("<Answer>")['input_ids'][-1]
        self.pad_token_id = tokenizer.pad_token_id 

    def _encode_one(self, text: str):
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)

    def get_batch(self, split, batch_size: int):
        rows = []
        batch_cand_tensors_list = [] 
        
        target_len = self.block_size + 1 
        split_data = self.texts[split]

        for _ in range(batch_size):
            # 1. Sample
            idx = int(torch.randint(0, len(split_data), (1,)).item())
            item = split_data[idx]
            text = item['text']
            # Expecting a list of strings for multi-question
            qids_list = item['qids'] 

            # 2. Encode
            seq = self._encode_one(text)
            L = seq.numel()
            if L >= target_len:
                window = seq[:target_len]
            else:
                pad_len = target_len - L
                pad = seq.new_full((pad_len,), self.pad_token_id)
                window = torch.cat([seq, pad], dim=0)
            rows.append(window)

            # 3. Build Candidates
            input_window = window[:-1]
            answer_locs = (input_window == self.answer_token_id).nonzero(as_tuple=True)[0]
            
            row_options = {} 
            max_k_row = 1
            
            for i, loc_idx in enumerate(answer_locs):
                loc = loc_idx.item()
                # Use the i-th ID for the i-th Answer token
                if i < len(qids_list):
                    qid = str(qids_list[i])
                    opts = self.option_dict.get(qid, [0]) if self.option_dict else [0]
                else:
                    opts = [0]
   
                row_options[loc] = opts
                max_k_row = max(max_k_row, len(opts))
            
            row_tensor = torch.full((self.block_size, max_k_row), -100, dtype=torch.long)
            
            for loc, opts in row_options.items():
                if loc < self.block_size:
                    row_tensor[loc, :len(opts)] = torch.tensor(opts, dtype=torch.long)
            
            batch_cand_tensors_list.append(row_tensor)

        # 4. Stack
        batch = torch.stack(rows).to(self.device)
        X = batch[:, :-1].contiguous()
        Y = batch[:,  1:].contiguous()
        gradient_mask = (X == self.answer_token_id).long()
        B, T = X.shape
        attention_mask = (X != self.pad_token_id).long()

        # 5. Stack Candidates
        max_k_batch = max((t.size(1) for t in batch_cand_tensors_list), default=1)
        final_cand_tensor = torch.full((B, T, max_k_batch), -100, dtype=torch.long, device=self.device)
        
        for b, t_tensor in enumerate(batch_cand_tensors_list):
            k = t_tensor.size(1)
            final_cand_tensor[b, :, :k] = t_tensor.to(self.device)

        return X, Y, gradient_mask, attention_mask, final_cand_tensor