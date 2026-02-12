from __future__ import annotations
import os
import json
import random
import argparse
import ast
import numpy as np
import pandas as pd
import wandb as wb
import copy


from src.inference.dataset import Dataset
from src.inference.model import PredictorPool
from src.inference.select_query import select_queries
from src.inference.utils_llm import load_jsonl_as_dict_of_dict, set_all_seeds, probs_to_an
from src.inference.evaluation import evaluate_model, evaluate_model_on_hard_groups



def parse_args():
    p = argparse.ArgumentParser()
    
    # General / Shared Args
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=10, help="Number of adaptive rounds.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log_path", type=str, default="./src/inference_hybrid/logs")
    
    # GNN Specific Args
    p.add_argument("--gnn_config_path", type=str, default="/home/ruomeng/gae_graph/src/gnn/config.yaml")
    p.add_argument("--gnn_batch_size", type=int, default=8192)
    p.add_argument("--node_selection", type=str, default="entropy", 
                   choices=["random", "entropy", "margin", "full", "info_gain", "explainer", "explainer_oracle", "explainer_diversity", "explainer_oracle_diversity", "entropy_diversity", "entropy"],
                   help="Strategy for GNN to select users.")
    p.add_argument("--node_selection_prec", type=float, default=0.1, help="Fraction of users to query.")

    # LLM Specific Args
    p.add_argument("--llm_checkpoint", type=str, required=True, help="Path to LLM checkpoint")
    p.add_argument("--llm_batch_size", type=int, default=512) # LLMs usually need smaller BS
    p.add_argument("--query_selection", type=str, default="info_gain", 
                   choices=["random", "mean_entropy", "info_gain"],
                   help="Strategy for LLM to select questions.")
    p.add_argument("--impute_thres", type=float, default=0.8) # LLMs usually need smaller BS
    p.add_argument("--imputation", action="store_true")
    
    # Data Paths (Hardcoded based on your snippet, but ideally arguments)
    p.add_argument("--runs", type=str, default="./src/inference_llm/logs", help="Where to write per-round log (JSONL).")
    p.add_argument("--dataset", type=str, default="./src/inference_llm/logs", help="Where to write per-round log (JSONL).")
    p.add_argument("--infer_data", type=str, default="./src/inference_llm/logs", help="Where to write per-round log (JSONL).")

    return p.parse_args()




if __name__ == "__main__":
    args = parse_args()
    pool_llm = PredictorPool(checkpoint=args.llm_checkpoint, gpus=4, seed=args.seed, batch_size=args.llm_batch_size)

    for seed in range(10):
        args.seed = seed
        set_all_seeds(args.seed)
        
        # --- Logging Setup ---
        os.makedirs(args.log_path, exist_ok=True)
        fname = f"hybrid_seed{args.seed}_Q-{args.query_selection}_N-{args.node_selection}-{args.node_selection_prec}_impute-{args.impute_thres}.jsonl"
        log_full_path = os.path.join(args.log_path, fname)
        print(f"[LOG] Writing to {log_full_path}")

        if args.wandb:
            wb.init(
                project="hybrid_adaptive_elicitation",
                name=f"HYBRID_{args.query_selection}_{args.node_selection}_{args.node_selection_prec}",
                config=vars(args)
            )

        # ==========================================
        # 1. Initialize Data & Candidates
        # ==========================================
        
        # Load IDs and Question Splits
        df_seed = pd.read_csv(args.runs)
        rng = np.random.default_rng(args.seed)
        seed_row = df_seed[df_seed['seed'] == args.seed]
        cand_qids = seed_row['cand_qids'].apply(ast.literal_eval).values.tolist()[0]
        eval_qids = seed_row['eval_qids'].apply(ast.literal_eval).values.tolist()[0]
        
        # Load Codebook
        codebook = load_jsonl_as_dict_of_dict( f"/home/ruomeng/gae_graph/dataset/{args.dataset}/codebook.jsonl"  , key='id')
        
        # Load Raw Data
        df_all = pd.read_csv(args.infer_data)
        df_survey = df_all[["caseid"] + cand_qids].copy()
        df_heldout = df_all[["caseid"] + eval_qids].copy()

        # ==========================================
        # 2. Initialize Models (The Hybrid Setup)
        # ==========================================

        print(">>> Initializing LLM (Query Selector)...")
        # Dataset handles the "State" for the LLM
        dataset_llm = Dataset.load_dataset(df_survey=df_survey, df_heldout=df_heldout, codebook=codebook, verbose=True)
        
        # ==========================================
        # 3. Synchronization Check
        # ==========================================
        # Ensure GNN and LLM are talking about the same questions
        # Filter Xavail (available questions) to overlap with cand_qids
        model_name = args.llm_checkpoint.split('/')[-2] if args.llm_checkpoint.split('/')[4] == 'logs' else args.llm_checkpoint.split('/')[-1]
        print(f"Model name: {model_name}")
     
        Xavail = [q for q in dataset_llm.Xpool if q in cand_qids]
        # hard_groups_path = f'/home/ruomeng/gae_graph/dataset/{args.dataset}/user_difficulty/seed{args.seed}_{model_name}_user_difficulty.json'

        # Number of nodes to select per round
        K_per_round = int(len(dataset_llm.graph.nodes) * args.node_selection_prec)

        # ==========================================
        # 4. Adaptive Loop
        # ==========================================
        
        with open(log_full_path, "w", encoding="utf-8") as log_f:
            
            # 2. CAPTURE PPL (Perplexity) here
            _, _, mean_acc_llm, mean_ppl_llm, mean_f1_llm, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
            # hard_group_results = evaluate_model_on_hard_groups(pool_llm, dataset_llm, dataset_llm.Y_heldout, hard_groups_path)
            print(f"[Round 0] LLM Acc: {mean_acc_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
            
            # Log Round 0
            log_f.write(json.dumps({
                "round": 0, 
                "q_selected": None,
                "gnn_acc": None, 
                "llm_acc": mean_acc_llm,
                "llm_ppl": mean_ppl_llm,
                "llm_f1": mean_f1_llm,
                "llm_bs": mean_bs_llm,
                "real_query_counts": {},
                "imputed_query_counts": {},
                "hard_groups": None,
            }) + "\n")
            log_f.flush()
            
            users_to_query = None
            for t in range(args.T):
                print(f"\n===== Hybrid Round {t+1} / {args.T} =====")
                
                if not Xavail:
                    print("No more questions available.")
                    break

                # -------------------------------------------------
                # A. LLM Selects Query (Global Info Gain)
                # -------------------------------------------------
                print(f"[Step A] LLM selecting query from {len(Xavail)} candidates...")
                
                if args.query_selection == 'random':
                    rng = np.random.default_rng(args.seed + t)
                    x_star = [rng.choice(Xavail)]
                
                elif args.query_selection == 'info_gain':
                    # Subsample nodes for speed if graph is huge
                    nodes_eval = dataset_llm.graph.nodes
                    nodes_eval = rng.choice(nodes_eval, size=int(len(dataset_llm.graph.nodes) * 0.1), replace=False).tolist()
                    print(f"Nodes eval: {nodes_eval[:10]}")
    
                    # LLM Inference to find best Question
                    x_star = select_queries(
                        dataset_llm,
                        pool_llm,
                        nodes=nodes_eval,
                        Xavail=Xavail,
                        observed=copy.deepcopy(dataset_llm.observed_dict),
                        cur_asked_queries=dataset_llm.asked_queries,
                    )

                q_selected = str(x_star[0]) # The chosen Question ID
                print(f"Selected Query: {q_selected}")

                # 2. Select nodes based on GNN uncertainty (Entropy/Margin)
                # The GNN selector needs to map its internal IDs to the ones used in selection
                if K_per_round == 0:
                    users_to_query = []
                else:
                    if args.node_selection == 'random':
                        cand = list(dataset_llm.graph.nodes)
                        k_eff = min(K_per_round, len(cand))
                        users_to_query = random.sample(cand, k_eff)
              
                    # -------------------------------------------------
                    # C. Synchronization (Update Both Models)
                    # -------------------------------------------------
                    dataset_llm.update_observed(q_selected, users_to_query)
                    
                dataset_llm.asked_queries.append((q_selected, dataset_llm.codebook[q_selected]["question"]))
                # Remove selected question from candidates
                Xavail = [q for q in Xavail if q != q_selected]
                # -------------------------------------------------
                # D. Evaluation & Logging
                # -------------------------------------------------
                if args.imputation:
                    V_rest = list(set(dataset_llm.graph.nodes) - set(users_to_query))
                    q_text = dataset_llm.codebook[q_selected]["question"]
                    options = list(dataset_llm.codebook[q_selected]["options"].keys())
                    probs_batch_node_rest = pool_llm.predict(items=V_rest, shard_arg="nodes", query=q_text, candidate_options=options, asked_queries=dataset_llm.asked_queries, 
                                                    observed=dataset_llm.observed_dict)
                    V_rest_estimated = probs_to_an(probs_batch_node_rest, V_rest, labels=options) 
                    dataset_llm.update_observed_estimated(q_selected, V_rest=V_rest_estimated)
       
                
                # Evaluate GNN Accuracy
                _, _, mean_acc_llm, mean_ppl_llm, mean_f1_llm, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
                # hard_group_results = evaluate_model_on_hard_groups(pool_llm, dataset_llm, dataset_llm.Y_heldout, hard_groups_path)
                print(f"[Round {t+1}] LLM Acc: {mean_acc_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
                
                # 4. LOG EVERYTHING
                log_entry = {
                    "round": t + 1,
                    "q_selected": q_selected,
                    "gnn_acc": None,
                    "llm_acc": mean_acc_llm,
                    "llm_ppl": mean_ppl_llm,           # New
                    "llm_f1": mean_f1_llm,
                    "llm_bs": mean_bs_llm,
                    "hard_groups": None,
                }
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()
                if args.wandb: wb.log(log_entry)
        if args.wandb:
            wb.finish()
        print(f"Done. Logs saved to {log_full_path}")
    