from __future__ import annotations
import os
import json
import random
import argparse
import ast
import numpy as np
import pandas as pd
import copy

from src.inference.dataset import Dataset
from src.inference.model import PredictorPool
from src.inference.select_query import select_queries
from src.inference.utils_llm import load_jsonl_as_dict, set_all_seeds, probs_to_an
from src.inference.evaluation import evaluate_model


def parse_args():
    p = argparse.ArgumentParser()
    
    # General / Shared Args
    p.add_argument("--T", type=int, default=10, help="Number of adaptive rounds.")
    p.add_argument("--log_path", type=str, default="results")
    p.add_argument("--cuda", type=str, default="0,1,2,3", help="CUDA devices to use.")
    
    # LLM Specific Args
    p.add_argument("--llm_checkpoint", type=str, required=True, help="Path to LLM checkpoint")
    p.add_argument("--llm_batch_size", type=int, default=512)
    p.add_argument("--query_selection", type=str, default="info_gain", choices=["random", "info_gain"], help="Query selection strategy.")
    p.add_argument("--imputation", action="store_true")
    p.add_argument("--smaple_size", type=float, default=0.1, help="Fraction of nodes to sample for calculation of info gain.")
    
    # Data Paths (Hardcoded based on your snippet, but ideally arguments)
    p.add_argument("--runs_id", type=int, default=0, choices=range(10), help="Runs ID.")
    p.add_argument("--runs", type=str, default="scripts/runs/${dataset}.csv", help="Path to the runs file.")
    p.add_argument("--dataset", type=str, default="${dataset}", help="Dataset name.")
    p.add_argument("--infer_data", type=str, default="dataset/${dataset}/data/question_${region}_test.csv", help="Path to the test data.")

    return p.parse_args()




if __name__ == "__main__":
    args = parse_args()
    n_gpus = len(args.cuda.split(','))
    set_all_seeds(seed=args.runs_id) # set seed for reproducibility
    
    pool_llm = PredictorPool(checkpoint=args.llm_checkpoint, gpus=n_gpus, seed=args.runs_id, batch_size=args.llm_batch_size)

    # Logging Setup
    os.makedirs(args.log_path, exist_ok=True)
    if args.imputation:
        imputation_suffix = "_imputed"
    else:
        imputation_suffix = ""
    fname = f"runs-{args.runs_id}_Q-{args.query_selection}_N-{args.node_selection}-{args.node_selection_prec}{imputation_suffix}.jsonl"
    log_full_path = os.path.join(args.log_path, fname)
    print(f"[LOG] Writing to {log_full_path}")

    # Load IDs and Question Splits
    df_seed = pd.read_csv(args.runs)
    seed_row = df_seed[df_seed['seed'] == args.runs_id]
    cand_qids = seed_row['cand_qids'].apply(ast.literal_eval).values.tolist()[0]
    eval_qids = seed_row['eval_qids'].apply(ast.literal_eval).values.tolist()[0]
    
    # Load Codebook
    codebook = load_jsonl_as_dict( f"dataset/{args.dataset}/codebook.jsonl", key='id')
    
    # Load Raw Data
    df_all = pd.read_csv(args.infer_data)
    df_survey = df_all[["caseid"] + cand_qids].copy()
    df_heldout = df_all[["caseid"] + eval_qids].copy()

    # ==========================================
    # Initialize Models
    # ==========================================
    print(">>> Initializing LLM (Query Selector)...")
    dataset_llm = Dataset.load_dataset(df_survey=df_survey, df_heldout=df_heldout, codebook=codebook, verbose=True)
    # Get available questions
    Xavail = [q for q in dataset_llm.Xpool if q in cand_qids]
    # Number of nodes to select per round
    K_per_round = int(len(dataset_llm.graph.nodes) * args.node_selection_prec)
    
     # ==========================================
    # Adaptive Loop
    # ==========================================
    with open(log_full_path, "w", encoding="utf-8") as log_f:
        
        # Evaluate LLM Accuracy
        _, _, mean_acc_llm, mean_ppl_llm, _, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
        print(f"[Round 0] LLM Acc: {mean_acc_llm:.4f} -- BS: {mean_bs_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
        
        # Log Round 0
        log_f.write(json.dumps({"round": 0, "q_selected": None, "llm_acc": round(mean_acc_llm, 4), "llm_ppl": round(mean_ppl_llm, 4), "llm_bs": round(mean_bs_llm, 4)}) + "\n")
        log_f.flush()
        
        users_to_query = None
        for t in range(args.T):
            print(f"\n===== Round {t+1} / {args.T} =====")
            
            if not Xavail:
                print("No more questions available.")
                break

            # LLM Selects Query (Global Info Gain)
            print(f"LLM selecting query from {len(Xavail)} candidates...")
            
            if args.query_selection == 'random':
                rng = np.random.default_rng(args.runs_id + t)
                x_star = [rng.choice(Xavail)]
            
            elif args.query_selection == 'info_gain':
                # Subsample nodes for speed
                nodes_eval_all = list(dataset_llm.graph.nodes)
                nodes_eval = rng.choice(nodes_eval_all, size=int(len(nodes_eval_all) * args.smaple_size), replace=False).tolist()

                # LLM Inference to find best query Question
                x_star = select_queries(
                    dataset_llm,
                    pool_llm,
                    nodes=nodes_eval,
                    Xavail=Xavail,
                    observed=copy.deepcopy(dataset_llm.observed_dict),
                    cur_asked_queries=dataset_llm.asked_queries,
                )

            q_selected = str(x_star[0]) # The chosen query ID
            print(f"Selected Query: {q_selected}")

            # Select nodes (Randomly)
            if K_per_round == 0:
                users_to_query = []
            else:
                if args.node_selection == 'random':
                    cand = list(dataset_llm.graph.nodes)
                    k_eff = min(K_per_round, len(cand))
                    users_to_query = random.sample(cand, k_eff)
            
                dataset_llm.update_observed(q_selected, users_to_query)
                
            dataset_llm.asked_queries.append((q_selected, dataset_llm.codebook[q_selected]["question"]))
            Xavail = [q for q in Xavail if q != q_selected]
            print(f"Remaining candidates: {len(Xavail)}")
            if args.imputation:
                V_rest = list(set(dataset_llm.graph.nodes) - set(users_to_query))
                q_text = dataset_llm.codebook[q_selected]["question"]
                options = list(dataset_llm.codebook[q_selected]["options"].keys())
                probs_batch_node_rest = pool_llm.predict(items=V_rest, shard_arg="nodes", query=q_text, candidate_options=options, asked_queries=dataset_llm.asked_queries, 
                                                observed=dataset_llm.observed_dict)
                V_rest_estimated = probs_to_an(probs_batch_node_rest, V_rest, labels=options) 
                dataset_llm.update_observed_estimated(q_selected, V_rest=V_rest_estimated)
    
            # Evaluate LLM Accuracy
            _, _, mean_acc_llm, mean_ppl_llm, _, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
            print(f"[Round {t+1}] LLM Acc: {mean_acc_llm:.4f} -- BS: {mean_bs_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
            
            # Log Round {t+1}
            log_entry = {"round": t + 1, "q_selected": q_selected, "llm_acc": round(mean_acc_llm, 4), "llm_ppl": round(mean_ppl_llm, 4), "llm_bs": round(mean_bs_llm, 4)}
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

    print(f"Done. Logs saved to {log_full_path}")
    pool_llm.close()
