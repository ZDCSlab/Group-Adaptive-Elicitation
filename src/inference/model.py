import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import ray
from typing import List, Any, Union
from peft import PeftModel, PeftConfig


class PredictorPool:
    def __init__(self, checkpoint, gpus: int, seed: int, batch_size: int):
        self.gpus = gpus
        ray.init(ignore_reinit_error=True, include_dashboard=False) 
        self.actors = [ModelWorker.remote(checkpoint=checkpoint, seed=seed, batch_size=batch_size)   
                        for _ in range(gpus)]
                       
    @staticmethod
    def _split_round_robin(seq, n):
        return [seq[i::n] for i in range(n)]

    def _map(self, actors, items: List[Any], shard_arg: str, **kwargs) -> List[Any]:
        if not items:
            return []

        indexed = list(enumerate(items))
        shards  = self._split_round_robin(indexed, len(actors))

        futs = []
        for act, shard in zip(actors, shards):
            shard_idx = [i for i, _ in shard]
            shard_itm = [x for _, x in shard]
            
            kw = dict(kwargs)
        
            kw['items'] = shard_itm
            kw['shard_arg'] = shard_arg 
            
            futs.append((shard_idx, act.predict_batch.remote(**kw)))

        outs = []
        results = ray.get([f for _, f in futs])
        
        for (shard_idx, _), out_list in zip(futs, results):
            assert len(shard_idx) == len(out_list)
            outs.extend(zip(shard_idx, out_list))

        outs.sort(key=lambda p: p[0])
        return [pred for _, pred in outs]

    def predict(self, items: List[Any], shard_arg="nodes", **kwargs) -> Union[np.ndarray, List[Any]]:

        actors = self.actors
        if not items:
            return []
        
        results = self._map(actors, items, shard_arg, **kwargs)
        
        try:
            return np.array(results)
        except:
            return results

    def close(self):
        """Gracefully shut down Ray actors and the Ray runtime."""
        if hasattr(self, "actors"):
            for actor in self.actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            self.actors = []

        if ray.is_initialized():
            ray.shutdown()

        
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, checkpoint: str, seed: int, batch_size: int):
        self.m = Meta_Model(checkpoint=checkpoint, seed=seed, device=[0], batch_size=batch_size)

    def predict_batch(self, items: List[Any], query: Union[str, List[str]], candidate_options: Union[List[str], List[List[str]]] = None, shard_arg: str = "nodes", **kwargs):
        call_kwargs = kwargs.copy()
        call_kwargs[shard_arg] = items

        # Default options logic
        if candidate_options is None:
            if isinstance(query, str): candidate_options = ["A", "B"]
            else: candidate_options = [["A", "B"]] * len(query)

        # Case 1: Single Query
        if isinstance(query, str):
            if not isinstance(candidate_options[0], str):
                 raise ValueError("Single query requires simple list of options")
            return self.m.predict_batch(query=query, options=candidate_options, **call_kwargs)

        # Case 2: Multiple Queries
        elif isinstance(query, list):
            n_queries = len(query)
            n_items = len(items)
            all_prompts = []
            options_per_prompt = []
            
            # 1. Build all prompts for all (Node, Query) pairs
            for q_idx, q_text in enumerate(query):
                opts = candidate_options[q_idx]
                # We need a method in Meta_Model that just returns strings
                prompts = self.m.get_prompts_only(nodes=items, query=q_text, **kwargs)
                # --- DEBUG PRINT: See the first prompt of the first query ---
                if q_idx == 0:
                    answer_delimiter = "<Answer>"
                    sample_prompt = prompts[0]
                    if not sample_prompt.strip().endswith(answer_delimiter):
                        print(f"CRITICAL WARNING: Prompt does not end with {answer_delimiter}!")
                        print(f"Actual ending: ...{sample_prompt[-20:]}")

                all_prompts.extend(prompts)
                assert len(prompts) == n_items, f"Node count mismatch for query {q_idx}"
                # Keep track of which options belong to which flattened prompt
                options_per_prompt.extend([opts] * len(prompts))

            # 2. Call the new high-speed method
            # flat_probs shape: [N_items * N_queries, Max_Opts_in_Batch]
            flat_probs = self.m.score_candidates_heterogeneous(all_prompts, options_per_prompt)
            assert flat_probs.shape[0] == n_queries * n_items, "Flat output size mismatch"

            # 3. Reshape back to [Nodes, Queries, Max_Options]
            n_queries = len(query)
            n_items = len(items)
            max_opts = flat_probs.shape[1]
            
            # Reshape to [Queries, Nodes, Max_Options]
            results_matrix = flat_probs.reshape(n_queries, n_items, max_opts)
            
            # Transpose to [Nodes, Queries, Max_Options] for the Pool
            return list(results_matrix.transpose(1, 0, 2))



class Meta_Model:
    def __init__(self, checkpoint, device, seed=42, batch_size=128):
        self.rng = np.random.default_rng(seed)
        self.device = device

        print(f'Loading checkpoint: {checkpoint}')
        
        # 1. Load Tokenizer (Always load from checkpoint to get new special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        self.tokenizer.truncation_side = "left"      
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = batch_size
        self._option_id_cache = {}

        # 2. Check if this is an Adapter (LoRA) or a Full Model
        # Adapters always have an 'adapter_config.json' file
        is_adapter = os.path.exists(os.path.join(checkpoint, "adapter_config.json"))

        if is_adapter:
            print(f"Detected Adapter (SFT) at {checkpoint}. Loading Base Model first...")
            # A) Find the base model path from the adapter config
            peft_config = PeftConfig.from_pretrained(checkpoint)
            base_model_path = peft_config.base_model_name_or_path
            # B) Load Base Model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2")
            # C) Resize Base Model
            model.resize_token_embeddings(len(self.tokenizer))
            # D) Load Adapter
            model = PeftModel.from_pretrained(model, checkpoint)
            # E) Merge adapter
            model = model.merge_and_unload()
        
        else:
            print(f"Loading Full Model from {checkpoint}...")
            # Standard loading for merged/full models
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                ignore_mismatched_sizes=True # Safety net for full models
            )
            model.resize_token_embeddings(len(self.tokenizer))

        # Final Config Setup
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.eval()
        self.model = model.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
    def _format_prompt(self, query, history=None):
        parts = []
        if history is not None:
            parts.append(history)
        parts.append(f"<Question>{query}")
        parts.append("<Answer>") # Ensure no trailing space if your options don't start with space
        return "".join(parts)


    def predict_batch(self, nodes, query, asked_queries, observed=None, options=None):
        # Default options if None passed
        if options is None:
            options = ["A", "B"]

        prompts = []
        for caseid in nodes:
            hist_transition = []
            if observed: 
                for (qid, q_text) in asked_queries:
                    try:
                        ans = observed[qid][caseid]
                        content = f"<Question>{q_text}<Answer>{ans}"
                        hist_transition.append(content)
                    except:
                        continue
            h_text = "\n".join(hist_transition)
            prompt = self._format_prompt(query, h_text)
            prompts.append(prompt)

        # Pass specific options to score_candidates
        probs_batch = self.score_candidates(prompts, options=options, per_gpu_batch_size=self.batch_size)
        return probs_batch
    
    def score_candidates(self, prompts, options, per_gpu_batch_size=8):
        if len(prompts) == 0:
            return np.zeros((0, len(options)), dtype=np.float32)

        # --- 1) Dynamic Caching Logic ---
        # Create a tuple key for the cache (lists are not hashable)
        options_key = tuple(options)
        
        if options_key not in self._option_id_cache:
            option_token_ids = []
            for opt in options:
                # Use add_special_tokens=False. 
                ids = self.tokenizer.encode(opt, add_special_tokens=False)
                if len(ids) != 1:
                    ids = [ids[-1]] 
                option_token_ids.append(ids[0])
            
            self._option_id_cache[options_key] = torch.tensor(option_token_ids, dtype=torch.long, device=self.model.device)
        
        # Retrieve from cache
        option_token_ids = self._option_id_cache[options_key]

        probs_batch = []

        # --- Loop over batches ---
        for start in range(0, len(prompts), per_gpu_batch_size):
            batch_prompts = prompts[start:start + per_gpu_batch_size]

            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024,
                add_special_tokens=True 
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model(**inputs)
            
            logits = out.logits 
            
            attn = inputs["attention_mask"]
            last_idx = (attn.sum(dim=1) - 1) 
            
            step0_logits = logits[torch.arange(logits.size(0)), last_idx, :].float()

            sel_logits = torch.index_select(step0_logits, dim=1, index=option_token_ids)
            probs = torch.softmax(sel_logits, dim=1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs_batch.append(probs.cpu().numpy())

        return np.concatenate(probs_batch, axis=0)

    def get_prompts_only(self, nodes, query, asked_queries, observed=None, **kwargs):
        """Generates the text prompts for a batch of nodes for a single query."""
        prompts = []
        for caseid in nodes:
            hist_transition = []
            if observed:
                for (qid, q_text) in asked_queries:
                    # Look up if this specific node has an answer for this past question
                    ans = observed.get(qid, {}).get(caseid)
                    if ans:
                        hist_transition.append(f"<Question>{q_text}<Answer>{ans}")
            
            h_text = "\n".join(hist_transition)
            prompts.append(self._format_prompt(query, h_text))
        return prompts

   
    def score_candidates_heterogeneous(self, prompts, options_per_prompt):
        """
        High-performance vectorized scoring for prompts with varying options.
        """
        if not prompts:
            return np.array([])

        # Map all unique option strings to their token IDs
        unique_opts = sorted({opt for sublist in options_per_prompt for opt in sublist})

        opt_token_ids = []
        for o in unique_opts:
            ids = self.tokenizer.encode(o, add_special_tokens=False)

            # Option must be single-token
            assert len(ids) == 1, (
                f"Option '{o}' is not single-token under this tokenizer. "
                f"Got token ids: {ids}. "
                f"Check <Answer> formatting and option strings."
            )

            opt_token_ids.append(ids[0])

        # No token collision
        assert len(set(opt_token_ids)) == len(opt_token_ids), (
            "Token collision detected: multiple options map to the same token id. "
            f"Options: {dict(zip(unique_opts, opt_token_ids))}"
        )

        token_tensor = torch.tensor(opt_token_ids, device=self.model.device)
        opt_to_union_idx = {opt: i for i, opt in enumerate(unique_opts)}

        all_results = []
        max_opts_global = max(len(o) for o in options_per_prompt)

        # Process Mega-Batch
        for start in range(0, len(prompts), self.batch_size):
            b_prompts = prompts[start : start + self.batch_size]
            b_opts_list = options_per_prompt[start : start + self.batch_size]
            curr_bs = len(b_prompts)

            # LLM Tokenization
            inputs = self.tokenizer(
                b_prompts, return_tensors="pt", padding=True, 
                truncation=True, max_length=1024, add_special_tokens=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attn = inputs["attention_mask"]
                last_idx = attn.sum(dim=1) - 1
                last_token_logits = outputs.logits[torch.arange(outputs.logits.size(0)), last_idx, :].float()

            # Narrow the search space to only our unique options
            # Shape: [Batch, Len(unique_opts)]
            relevant_logits = torch.index_select(last_token_logits, dim=1, index=token_tensor)
            
            # Vectorized Indexing: Prepare a map for each prompt's specific options
            max_opts_in_batch = max(len(o) for o in b_opts_list)
            indices_tensor = torch.zeros((curr_bs, max_opts_in_batch), dtype=torch.long, device=self.model.device)
            mask = torch.zeros((curr_bs, max_opts_in_batch), dtype=torch.bool, device=self.model.device)
            
            for i, row_options in enumerate(b_opts_list):
                row_idx = [opt_to_union_idx[o] for o in row_options]
                indices_tensor[i, :len(row_idx)] = torch.tensor(row_idx, device=self.model.device)
                mask[i, :len(row_idx)] = True

            # Gather & Mask: Pull the specific logits for each prompt into a dense matrix
            # Pull the specific logits for each prompt into a dense matrix
            gathered_logits = torch.gather(relevant_logits, 1, indices_tensor)
            
            # Mask out padding positions with -infinity
            gathered_logits.masked_fill_(~mask, float('-inf'))
            
            # Softmax across the valid options only
            probs_tensor = torch.softmax(gathered_logits, dim=1)
            row_sums = probs_tensor.sum(dim=1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-4
            ), "Softmax row sums not ~1; mask or option indexing is wrong."
            
            # Reconstruct the output matrix with global padding
            batch_probs = np.zeros((curr_bs, max_opts_global))
            batch_probs[:, :max_opts_in_batch] = probs_tensor.cpu().numpy()
            
            all_results.append(batch_probs)

        return np.concatenate(all_results, axis=0)