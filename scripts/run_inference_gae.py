from __future__ import annotations
import os
import json
import random
import argparse
import ast
import numpy as np
import pandas as pd
import torch
import wandb as wb
from collections import Counter
import copy


from src.inference.dataset import Dataset
from src.inference.model import PredictorPool
from src.inference.select_query import select_queries
from src.inference.select_query_lookahaed import select_queries_mcts
from src.inference.utils_llm import set_all_seeds, diagnostic_imputation, inject_pseudo_labels, load_jsonl_as_dict_of_dict, get_gt_labels_for_qid
from src.inference.evaluation import evaluate_model, evaluate_model_on_hard_groups
from src.inference.gnn_predictor import GNNElicitationPredictor
from src.inference.select_node import NodeSelector
from src.inference.utils_gnn import build_gold_answers_from_edges




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
                   choices=["random", "entropy", "margin", "full", "info_gain", "explainer", "cluster", "clue", "clue_test", "entropy_diversity", "entropy"],
                   help="Strategy for GNN to select users.")
    p.add_argument("--node_selection_prec", type=float, default=0.1, help="Fraction of users to query.")
    p.add_argument("--imputation", action="store_true")
    

    # LLM Specific Args
    p.add_argument("--llm_checkpoint", type=str, required=True, help="Path to LLM checkpoint")
    p.add_argument("--llm_batch_size", type=int, default=512) # LLMs usually need smaller BS
    p.add_argument("--query_selection", type=str, default="info_gain", 
                   choices=["random", "mean_entropy", "info_gain", "mcts_lookahead", "mcts_lookahead_newnew"],
                   help="Strategy for LLM to select questions.")
    p.add_argument("--impute_thres", type=float, default=0.8) # LLMs usually need smaller BS
    p.add_argument("--temp", type=float, default=1.0) # LLMs usually need smaller BS
    
    # Data Paths (Hardcoded based on your snippet, but ideally arguments)
    p.add_argument("--runs", type=str, default="./src/inference_llm/logs", help="Where to write per-round log (JSONL).")
    p.add_argument("--dataset", type=str, default="./src/inference_llm/logs", help="Where to write per-round log (JSONL).")
    p.add_argument("--infer_data", type=str, default="./src/inference_llm/logs", help="Where to write per-round log (JSONL).")

    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    pool_llm = PredictorPool(checkpoint=args.llm_checkpoint, gpus=4, seed=args.seed, batch_size=args.llm_batch_size)

    for seed in range(5,10):
        args.seed = seed
        set_all_seeds(args.seed)
        
        # --- Logging Setup ---
        os.makedirs(args.log_path, exist_ok=True)
        if args.imputation:
            fname = f"hybrid_seed{args.seed}_Q-{args.query_selection}_N-{args.node_selection}-{args.node_selection_prec}_impute-{args.impute_thres}.jsonl"
        else:
            fname = f"hybrid_seed{args.seed}_Q-{args.query_selection}_N-{args.node_selection}-{args.node_selection_prec}_no_imputation.jsonl"
        log_full_path = os.path.join(args.log_path, fname)
        print(f"[LOG] Writing to {log_full_path}")

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
        
        print(">>> Initializing GNN (Node Selector)...")
        predictor_gnn = GNNElicitationPredictor.from_config(config_path=args.gnn_config_path)
        selector_node_gnn = NodeSelector(all_nodes=list(predictor_gnn.uid2idx.keys()))
        gold_map_gnn = build_gold_answers_from_edges(predictor_gnn)

        # ==========================================
        # 3. Synchronization Check
        # ==========================================
        # Ensure GNN and LLM are talking about the same questions
        # Filter Xavail (available questions) to overlap with cand_qids
        model_name = args.llm_checkpoint.split('/')[-2]
        Xavail = [q for q in dataset_llm.Xpool if q in cand_qids]
        hard_groups_path = f'/home/ruomeng/gae_graph/dataset/{args.dataset}/user_difficulty/seed{args.seed}_{model_name}_user_difficulty.json'
        # hard_groups_path = f'/home/ruomeng/gae_graph/dataset/{args.dataset}/hard_user_groups_West.json'
        # Number of nodes to select per round
        K_per_round = int(len(dataset_llm.graph.nodes) * args.node_selection_prec)

        # ==========================================
        # 4. Adaptive Loop
        # ==========================================
        
        with open(log_full_path, "w", encoding="utf-8") as log_f:
            
            # 2. CAPTURE PPL (Perplexity) here
            # acc_gnn, total_gnn = eval_full_accuracy(predictor_gnn, gold_map_gnn, eval_qids, args.gnn_batch_size)
            _, _, mean_acc_llm, mean_ppl_llm, mean_f1_llm, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
            hard_group_results = evaluate_model_on_hard_groups(pool_llm, dataset_llm, dataset_llm.Y_heldout, hard_groups_path)
            print(f"[Round 0] LLM Acc: {mean_acc_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
            
            # Log Round 0
            log_f.write(json.dumps({
                "round": 0, 
                "q_selected": None,
                # "gnn_acc": acc_gnn, 
                "imputed_precision": None,
                "llm_acc": mean_acc_llm,
                "llm_ppl": mean_ppl_llm,
                "llm_f1": mean_f1_llm,
                "llm_bs": mean_bs_llm,
                "real_query_counts": {},
                "imputed_query_counts": {},
                "hard_groups": hard_group_results,
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
                    
                    # df = pd.read_csv(f'/home/ruomeng/gae_graph/scripts/runs/{args.dataset}_{model_name}_queries.csv')
                    # queries = df[(df['seed'] == seed) & (df['node_selection_prec'] == args.node_selection_prec)]['queries'].values[0]
                    # queries = ast.literal_eval(queries)
                    # x_star = [queries[t]]

                elif args.query_selection == 'mcts_lookahead_newnew':
                    nodes_eval = dataset_llm.graph.nodes
                    # if t == 0:
                    #     rng = np.random.default_rng(args.seed + t)
                    #     x_star = [rng.choice(Xavail)]
                
                    # else:
                    dataset_llm_copy = copy.deepcopy(dataset_llm)
                    predictor_gnn_copy = copy.deepcopy(predictor_gnn)
                    x_star = select_queries_mcts(
                        dataset_llm_copy,
                        pool_llm,
                        nodes=nodes_eval,
                        Xavail=Xavail,
                        observed_real=dataset_llm_copy.observed_dict,
                        predictor_gnn=predictor_gnn_copy,
                        depth=args.T-t-1,
                        n_iter=3,
                        top_k=10,
                        confidence_margin=0.0,
                        rng=rng
                    )
        

                q_selected = str(x_star[0]) # The chosen Question ID
                print(f"Selected Query: {q_selected}")

                # -------------------------------------------------
                # B. GNN Selects Nodes (Local Uncertainty)
                # -------------------------------------------------
                print(f"[Step B] GNN selecting {K_per_round} nodes for query {q_selected}...")
                
                # 1. Get GNN predictions for this specific question
                # This returns probability distributions for all users on q_selected
                gnn_preds = predictor_gnn.predict_for_question(q_selected, batch_size=args.gnn_batch_size)
                
                # 2. Select nodes based on GNN uncertainty (Entropy/Margin)
                # The GNN selector needs to map its internal IDs to the ones used in selection
                if K_per_round == 0:
                    users_to_query = []
                    gnn_preds_updated = predictor_gnn.predict_for_question(q_selected, batch_size=args.gnn_batch_size)
                    num_injected = inject_pseudo_labels(
                        dataset_llm=dataset_llm,
                        q_selected=q_selected,
                        gnn_preds=gnn_preds_updated,
                        selected_users=users_to_query,
                        confidence_margin=args.impute_thres
                    )
                    print(f"Injected {num_injected} pseudo-labels for next round InfoGain calculation.")
    
                else:               
                    if args.node_selection == 'cluster':
                        users_to_query = predictor_gnn.select_nodes_clue(q_selected, dataset_llm.graph.nodes, K_per_round, use_gt=False, target_ratio=args.node_selection_prec,  gt_labels=dataset_llm.graph.y, diversity=False, use_uncertainty=False)
                    else:
                        current_q_gt = get_gt_labels_for_qid(gold_map_gnn, q_selected)
                        users_to_query = selector_node_gnn.select_next_nodes(
                            preds=gnn_preds,
                            k=K_per_round,
                            mode=args.node_selection,
                            ground_truths=current_q_gt)

                
                    # -------------------------------------------------
                    # C. Synchronization (Update Both Models)
                    # -------------------------------------------------
                    
                    # 2. Update GNN Predictor
                    # We need to form tuples (uid, qid) for the GNN to add to its edge list
                    new_obs_gnn = [(uid, q_selected) for uid in users_to_query]
                    predictor_gnn.add_observations(new_obs_gnn, raw_uid=True)

                    print(f"Re-predicting {q_selected} with updated graph evidence...")
                    gnn_preds_updated = predictor_gnn.predict_for_question(q_selected, batch_size=args.gnn_batch_size)

                    # 1. Update LLM Dataset
                    # dataset_llm.update_observed automatically looks up the ground truth in df_survey
                    # assuming 'users_to_query' matches the IDs in df_survey['caseid']
                    dataset_llm.update_observed(q_selected, users_to_query)

                    if args.imputation:
                        num_injected = inject_pseudo_labels(
                            dataset_llm=dataset_llm,
                            q_selected=q_selected,
                            gnn_preds=gnn_preds_updated,
                            selected_users=users_to_query,
                            confidence_margin=args.impute_thres
                        )
                        print(f"Injected {num_injected} pseudo-labels for next round InfoGain calculation.")

                diag_res = diagnostic_imputation(dataset_llm, q_selected, users_to_query, gold_map_gnn)
                print(f">>> [Imputation Check] Q: {diag_res['q_id']} | "
                    f"Count: {diag_res['num_injected']} | "
                    f"Precision: {diag_res['precision']:.4f} | "
                    f"Coverage: {diag_res['coverage']}")
            

                dataset_llm.asked_queries.append((q_selected, dataset_llm.codebook[q_selected]["question"]))
                # Remove selected question from candidates
                Xavail = [q for q in Xavail if q != q_selected]
                # -------------------------------------------------
                # D. Evaluation & Logging
                # -------------------------------------------------
              

                # Evaluate GNN Accuracy
                # acc_gnn, total_gnn = eval_full_accuracy(predictor_gnn, gold_map_gnn, eval_qids, args.gnn_batch_size)
                _, _, mean_acc_llm, mean_ppl_llm, mean_f1_llm, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
                hard_group_results = evaluate_model_on_hard_groups(pool_llm, dataset_llm, dataset_llm.Y_heldout, hard_groups_path)
                print(f"[Round {t+1}] LLM Acc: {mean_acc_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
                
                # 4. LOG EVERYTHING
                log_entry = {
                    "round": t + 1,
                    "q_selected": q_selected,
                    "imputed_precision": diag_res['precision'],
                    # "gnn_acc": acc_gnn,
                    "llm_acc": mean_acc_llm,
                    "llm_ppl": mean_ppl_llm,
                    "llm_f1": mean_f1_llm,
                    "llm_bs": mean_bs_llm,
                    "hard_groups": hard_group_results,

                }
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()
      
        print(f"Done. Logs saved to {log_full_path}")
    