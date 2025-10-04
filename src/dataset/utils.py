# dataset_pipeline.py
import math, random, torch
from torch.utils.data import Dataset, DataLoader, Sampler

IGNORE_INDEX = -100

# ---------- Dataset: one sample = one full token sequence ----------
class TokenSeqDataset(Dataset):
    def __init__(self, sequences):                # List[List[int]]
        self.sequences = sequences
        self.lengths = [len(s) for s in sequences]
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]

# ---------- Bucketed batch sampler (cuts padding waste) ----------
class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, num_buckets=50, shuffle=True, seed=0):
        self.batch_size = batch_size
        rnd = random.Random(seed)
        idxs = list(range(len(lengths)))
        idxs.sort(key=lambda i: lengths[i])                 # sort by length
        # split into buckets
        buckets = [idxs[i::num_buckets] for i in range(num_buckets)]
        self.batches = []
        for b in buckets:
            if shuffle: rnd.shuffle(b)
            for i in range(0, len(b), batch_size):
                batch = b[i:i+batch_size]
                if batch: self.batches.append(batch)
        if shuffle: rnd.shuffle(self.batches)
    def __len__(self): return len(self.batches)
    def __iter__(self):
        for b in self.batches: yield b

# ---------- Collate: crop/pad to T+1 then build X/Y/labels ----------
class AnswerCollator:
    """
    mode:
      - 'at_answer': supervise exactly on <Answer> positions
      - 'after_answer': supervise from token after <Answer> to window end
      - 'answer_to_eos': supervise from after <Answer> up to <EOS> (needs eos_id)
    """
    def __init__(self, block_size, answer_id, eos_id=None, pad_id=0, mode="answer_to_eos"):
        self.T = block_size
        self.answer_id = answer_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.mode = mode

    def _random_crop(self, seq, need_len):
        if len(seq) >= need_len:
            start = random.randint(0, len(seq) - need_len)
            return seq[start:start+need_len]
        # right-pad if too short
        return seq + [self.pad_id] * (need_len - len(seq))

    def _build_mask(self, X):  # X: [B, T]
        B, T = X.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
        if self.mode == "at_answer":
            mask = (X == self.answer_id)
        elif self.mode == "after_answer":
            idx = (X == self.answer_id).nonzero(as_tuple=False)
            for b, t in idx:
                if t + 1 < T: mask[b, t+1:] = True
        elif self.mode == "answer_to_eos":
            assert self.eos_id is not None, "eos_id required for answer_to_eos"
            for b in range(B):
                t = 0
                while t < T:
                    pos = (X[b, t:] == self.answer_id).nonzero(as_tuple=False)
                    if len(pos) == 0: break
                    a = t + pos[0].item()
                    if a + 1 >= T: break
                    eos_rel = (X[b, a+1:] == self.eos_id).nonzero(as_tuple=False)
                    if len(eos_rel) > 0:
                        e = a + 1 + eos_rel[0].item()
                        if e > a + 1: mask[b, a+1:e] = True
                        t = e + 1
                    else:
                        mask[b, a+1:] = True
                        break
        else:
            raise ValueError(self.mode)
        return mask

    def __call__(self, batch):           # batch: List[List[int]]
        need = self.T + 1
        crops = [self._random_crop(seq, need) for seq in batch]
        batch_data = torch.tensor(crops, dtype=torch.long)     # [B, T+1]

        X = batch_data[:, :-1].contiguous()                    # [B, T]
        Y = batch_data[:,  1:].contiguous()                    # [B, T]

        attn = (X != self.pad_id).long()
        mask = self._build_mask(X)                             # bool [B, T]
        labels = Y.masked_fill(~mask, IGNORE_INDEX)

        return {"input_ids": X, "labels": labels, "attention_mask": attn}

# ---------- Loader builder ----------
def build_loaders(train_sequences, val_sequences, block_size, batch_size,
                  answer_id, eos_id, pad_id=0, mode="answer_to_eos",
                  seed=0, num_workers=2, num_buckets=50):
    train_ds = TokenSeqDataset(train_sequences)
    val_ds   = TokenSeqDataset(val_sequences)

    train_sampler = BucketBatchSampler(train_ds.lengths, batch_size, num_buckets=num_buckets, shuffle=True,  seed=seed)
    val_sampler   = BucketBatchSampler(val_ds.lengths,   batch_size, num_buckets=num_buckets, shuffle=False, seed=seed)

    collate = AnswerCollator(block_size, answer_id, eos_id, pad_id=pad_id, mode=mode)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                              pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0))
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   collate_fn=collate,
                              pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0))
    return train_loader, val_loader
