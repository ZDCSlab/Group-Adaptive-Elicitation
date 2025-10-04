from collections import Counter
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

SYSTEM_PROMPT = (
            "Given the current question and options, a history of "
            "<question-neighbor-answer> pairs, and the current neighbors' answers, infer the user's current answer."
            "Output exactly ONE candidate option ID. No extra text."
        )

class Meta_Model:
    def __init__(self, base_model, checkpoint, device_id, seed=42):
        self.rng = np.random.default_rng(seed)   # create new Generator
        self.device_id = device_id

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<EOS>", "<EOP>", "[PAD]", "<Answer>"]})

        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            # self.tokenizer.pad_token = self.tokenizer.eos_token  # common for decoder-only
            self.tokenizer.pad_token = "[PAD]"  # else consider tok_base.pad_token = tok_base.eos_token


        # model
        if len(device_id): 
            primary = device_id[0]
            print(f"Using {len(device_id)} GPUs with DataParallel")
            # 1) Load on CPU first (no device_map if you’re using DataParallel)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",  # ensure your model supports FA2
            )

            # 2) Tokenizer + padding setup
            model.resize_token_embeddings(len(self.tokenizer))
            model.config.pad_token_id = self.tokenizer.pad_token_id

            # 3) Put the **base** model on the primary GPU BEFORE wrapping
            torch.cuda.set_device(primary)
            model.to(f"cuda:{primary}")

            # 4) Wrap with DataParallel (do NOT .to() after this)
            model = torch.nn.DataParallel(model, device_ids=device_id, output_device=primary)
            model.eval()

            self.model = model

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
        probs_batch = self.score_candidates(prompts, options=["A", "B"], per_gpu_batch_size=32)

        return probs_batch
    
    def score_candidates_dummay(self, prompts, options=["A", "B"], per_gpu_batch_size=8):
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
                                                  truncation=True, max_length=1024) for opt in options]
        for tok in option_token_ids:
            if len(tok) != 1:
                raise ValueError(f"Option '{tok}' is not a single token (multi-token not yet supported).")
        option_token_ids = [tok[0] for tok in option_token_ids]
 
        # --- 2) Compute total batch size = per-GPU × num_GPUs ---
        n_gpus = len(self.device_id)
        batch_size = per_gpu_batch_size * max(1, n_gpus)
        # print(f"Using {n_gpus} GPUs → total batch size = {batch_size}")

        probs_batch = []

        # --- 3) Loop over batches with tqdm ---
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False, max_length=1024,
                                    add_special_tokens=False).to(next(self.model.module.parameters()).device)

            # --- 4) Forward pass (DataParallel splits across GPUs automatically) ---
            # with torch.no_grad():
            #     gen = self.model.module.generate(
            #         **inputs,
            #         max_new_tokens=1,         # 只生成一个 token（第1步）
            #         do_sample=False,          # 贪心；如需采样可改 True 并设 temperature/top_p
            #         return_dict_in_generate=True,
            #         output_scores=True        # 关键：输出每一步的 logits
            #     )

            # # step0_logits: [B, V]
            # step0_logits: torch.Tensor = gen.scores[0] # 形状 [batch_size, vocab_size]
            # print("step0_logits shape:", tuple(step0_logits.shape))

            with torch.no_grad():
                out = self.model(**inputs)             # 并行 forward
            logits = out.logits                      # [B, T, V]
            attn = inputs["attention_mask"]
            last_idx = attn.sum(dim=1) - 1
            rows = torch.arange(logits.size(0))
            step0_logits = logits[rows, last_idx, :]  # 每行“下一个 token”的分布


            # Ensure option_token_ids is a proper LongTensor on the same device
            if not isinstance(option_token_ids, torch.Tensor):
                option_token_ids = torch.tensor(option_token_ids, dtype=torch.long)
            option_token_ids = option_token_ids.to(step0_logits.device)

            # Stable log-softmax
            log_probs = torch.log_softmax(step0_logits, dim=-1)  # [B, V]

            # Safer column select (device-safe)
            sel_logp = torch.index_select(log_probs, dim=1, index=option_token_ids)  # [B, C]

            # Normalize to get probs over your options
            probs = sel_logp.exp()
            probs = probs / probs.sum(dim=-1, keepdim=True)
            probs_batch.append(probs.float().cpu().numpy())


        # --- 6) Concatenate results ---
        probs_batch = np.concatenate(probs_batch, axis=0)
        # print("probs_batch shape:", probs_batch.shape)


        return probs_batch
