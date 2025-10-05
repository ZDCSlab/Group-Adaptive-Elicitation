from collections import Counter
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import ray
from typing import List, Any, Dict

@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, which: str, base_model: str, checkpoint: str):
        # Load only the requested model on this GPU
        if which == "group":
            self.m = Meta_Model(base_model=base_model, checkpoint=checkpoint, device=[0])  # Ray pins 1 GPU => local 0
        elif which == "iid":
            self.m = Meta_Model(base_model=base_model, checkpoint=checkpoint, device=[0])
        else:
            raise ValueError(which)

    def predict_batch(self, **kwargs):
        # kwargs includes nodes, query, asked_queries, neighbors, observed, estimated, mode
        return self.m.predict_batch(**kwargs)

class PredictorPool:
    def __init__(self, group_cfg, iid_cfg, group_gpus: int, iid_gpus: int):
        ray.init(ignore_reinit_error=True)
        self.model_actors = [ModelWorker.remote("group", **group_cfg) for _ in range(group_gpus)]
        self.iid_actors   = [ModelWorker.remote("iid",   **iid_cfg)   for _ in range(iid_gpus)]

    @staticmethod
    def _split_round_robin(seq, n):
        return [seq[i::n] for i in range(n)]

    def _map(self, actors, items: List[Any], shard_arg: str, **kwargs) -> List[Any]:
        if not items:
            return []

        # 1) 贴下标，避免重复元素导致歧义
        indexed = list(enumerate(items))          # [(idx, item), ...]
        shards  = self._split_round_robin(indexed, len(actors))

        # 2) 并发提交；每个 worker 收到的是纯 item 列表，顺序不变
        futs = []
        for act, shard in zip(actors, shards):
            shard_idx = [i for i, _ in shard]
            shard_itm = [x for _, x in shard]
            kw = dict(kwargs)
            kw[shard_arg] = shard_itm
            # 让 worker 返回与输入等长的结果列表
            futs.append((shard_idx, act.predict_batch.remote(**kw)))

        # 3) 收集结果并与原始下标重新绑定
        outs = []
        results = ray.get([f for _, f in futs])   # 等待全部完成；如需“谁先完成先返回”，可用 ray.wait 循环
        for (shard_idx, _), out_list in zip(futs, results):
            assert len(shard_idx) == len(out_list), "predict_batch 输出长度应与输入一致"
            outs.extend(zip(shard_idx, out_list))  # [(idx, pred), ...]

        # 4) 按原始下标排序，恢复全局顺序（支持重复元素）
        outs.sort(key=lambda p: p[0])
        return [pred for _, pred in outs]

    def predict(self, which: str, items: List[Any], shard_arg="nodes", **kwargs) -> List[Any]:
        actors = self.model_actors if which == "group" else self.iid_actors
        if not items:
            return []
        return self._map(actors, items, shard_arg, **kwargs)


class Meta_Model:
    def __init__(self, base_model, checkpoint, device, seed=42):
        self.rng = np.random.default_rng(seed)   # create new Generator
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<EOS>", "<EOP>", "[PAD]", "<Answer>"]})

        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            # self.tokenizer.pad_token = self.tokenizer.eos_token  # common for decoder-only
            self.tokenizer.pad_token = "[PAD]"  # else consider tok_base.pad_token = tok_base.eos_token


        # model
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

        # 2) Tokenizer + padding setup
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.eval()

        self.model = model.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))


    def _format_prompt(self, query, options, estimated=None, history=None):
  
        parts = []
        if history is not None:
            hist_txt = history
            parts.append(hist_txt)
        
        # 1) Insert the question
        parts.append(f"<Question>{query}")

        # 2) Insert neighbor answers (majority, or raw dict)
        if estimated:
            # compute majority key
            counter = Counter(estimated.values())
            maj_key, _ = counter.most_common(1)[0]

            # meaning lookup if provided
            maj_meaning = f"({options[str(maj_key)]})" 
            neighbor_txt = f"<Neighbor>Majority Answer: {maj_key}{maj_meaning}"
            parts.append(neighbor_txt)

        # 4) Ask for the target answer
        parts.append("<Answer>")
        
        return "".join(parts)

    def predict_batch(self, nodes, query, asked_queries, neighbors=None, observed=None, estimated=None, mode=None, options={'A': 'Support', 'B': 'Oppose'}):
        """
        Generate probabilities for each node in batch.
        """
        prompts = []
        for caseid in nodes:
            hist_transition = []
            for (qid, q_text) in asked_queries:
                ans = observed[qid][caseid]
                nei_ans = [observed[qid][nb] for nb in neighbors.get(caseid, [])]
                # print('nei_ans', nei_ans)
                maj_nei_ans, _ = Counter(nei_ans).most_common(1)[0]
                # print('maj_nei_ans', maj_nei_ans)
                if 'iid' in mode:
                    content = f"<Question>{q_text}<<Answer>{ans}"
                elif 'group' in mode:
                    content = f"<Question>{q_text}<Neighbor>Majority Answer: {maj_nei_ans}<Answer>{ans}"
                # print(content)
                hist_transition.append(content)
            h_text = "\n".join(hist_transition)
            if estimated is None:
                prompt = self._format_prompt(query, options, None, h_text)
            else:
                prompt = self._format_prompt(query, options, estimated[caseid], h_text)
            prompts.append(prompt)

        # print(mode, 'Prompt Example:', prompts[0])
        # print('prompt type', len(set(prompts)))
        # ---- send prompts to LLM ----
        # Here you’d call your underlying LM to get logits / probs
        # Example: returns list of probability distributions
        probs_batch = self.score_candidates(prompts, options=["A", "B"], per_gpu_batch_size=512)

        return probs_batch
    
    def score_candidates_dummy(self, prompts, options=["A", "B"], per_gpu_batch_size=8):
        """
        Simulate scoring candidate answers for each prompt.
        Returns: list of probability vectors (softmaxed).
        """
        
        probs_batch = []
        for p in prompts:
            # random logits
            logits = self.rng.normal(size=2)
            # softmax to convert to probs
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            probs_batch.append(probs)
        return probs_batch


    def score_candidates(self, prompts, options=["A", "B"], per_gpu_batch_size=8):
        """
        Score candidate options with a LLaMA base model (multi-GPU data parallel).

        Args:
            prompts: list[str]                 # input prompts
            options: list[str]                 # candidate answers as strings
            per_gpu_batch_size: int            # batch size per GPU

        Returns:
            probs_batch: np.ndarray, shape [num_prompts, num_options]
                Per-prompt probability distribution over options.
        """
        # --- 1) Tokenize candidate options once ---
        option_token_ids = [self.tokenizer.encode(opt, add_special_tokens=False, 
                                                  truncation=True, max_length=2048) for opt in options]
        for tok in option_token_ids:
            if len(tok) != 1:
                raise ValueError(f"Option '{tok}' is not a single token (multi-token not yet supported).")
        option_token_ids = [tok[0] for tok in option_token_ids]
 
        # --- 2) Compute total batch size = per-GPU × num_GPUs ---
        n_gpus = len(self.device)
        batch_size = per_gpu_batch_size * max(1, n_gpus)
        # print(f"Using {n_gpus} GPUs → total batch size = {batch_size}")

        probs_batch = []

        # --- 3) Loop over batches with tqdm ---
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                                    add_special_tokens=False).to(next(self.model.parameters()).device)

            with torch.no_grad():
                out = self.model(**inputs)             # 并行 forward
            logits = out.logits                      # [B, T, V]
            attn = inputs["attention_mask"].to(dtype=torch.long)
            lengths = attn.sum(dim=1)   
            last_idx = (lengths - 1).clamp(min=0, max=logits.size(1)-1)
            rows = torch.arange(logits.size(0), device=logits.device)

            # Cast to float32 BEFORE log_softmax for stability (avoid bf16 NaNs)
            step0_logits = logits[rows, last_idx, :].float()      # [B, V]
            # Optional: early sanity check
            if torch.isnan(step0_logits).any() or torch.isinf(step0_logits).any():
                print("[WARN] step0_logits has NaN/Inf; likely overlong prompt or precision issue.", flush=True)

            # Ensure option_token_ids is a proper LongTensor on the same device
            if not isinstance(option_token_ids, torch.Tensor):
                option_token_ids = torch.tensor(option_token_ids, dtype=torch.long)
            option_token_ids = option_token_ids.to(step0_logits.device)

            log_probs = torch.log_softmax(step0_logits, dim=-1)   # [B, V] (stable)
            sel_logp = torch.index_select(log_probs, dim=1, index=option_token_ids)  # [B, C]
            probs = torch.softmax(sel_logp, dim=1)                # [B, C], float32
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sums = probs.sum(dim=1, keepdim=True)
            needs_fallback = row_sums <= 1e-12
            if needs_fallback.any():
                # uniform fallback for degenerate rows (e.g., all -inf)
                probs[needs_fallback] = 1.0 / probs.size(1)

            probs = probs / probs.sum(dim=1, keepdim=True)
            probs_batch.append(probs.cpu().numpy())


        # --- 6) Concatenate results ---
        probs_batch = np.concatenate(probs_batch, axis=0)
        # print("probs_batch shape:", probs_batch.shape)


        return probs_batch
