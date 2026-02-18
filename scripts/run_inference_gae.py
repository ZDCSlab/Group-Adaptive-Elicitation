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
from src.inference.select_query import select_queries, select_queries_mcts
from src.inference.utils import set_all_seeds, diagnostic_imputation, inject_pseudo_labels, load_jsonl_as_dict
from src.inference.evaluation import evaluate_model
from src.inference.gnn_predictor import GNNElicitationPredictor, build_gold_answers_from_edges


def parse_args():
    p = argparse.ArgumentParser()
    
    # General / Shared Args
    p.add_argument("--T", type=int, default=10, help="Number of adaptive rounds.")
    p.add_argument("--log_path", type=str, default="results")
    p.add_argument("--cuda", type=str, default="0,1,2,3", help="CUDA devices to use.")
    
    # GNN Specific Args
    p.add_argument("--gnn_config_path", type=str, default="scripts/args_gnn/config_${dataset}.yaml")
    p.add_argument("--gnn_batch_size", type=int, default=8192)
    p.add_argument("--node_selection", type=str, default="random", choices=["random", "relational"], help="Node selection strategy.")
    p.add_argument("--node_selection_prec", type=float, default=0.1, help="Fraction of users to query.")

    # LLM Specific Args
    p.add_argument("--llm_checkpoint", type=str, required=True, help="Path to LLM checkpoint")
    p.add_argument("--llm_batch_size", type=int, default=512)
    p.add_argument("--query_selection", type=str, default="info_gain", choices=["random", "info_gain", "mcts_lookahead"], help="Query selection strategy.")
    p.add_argument("--imputation", action="store_true")
    p.add_argument("--smaple_size", type=float, default=0.1, help="Fraction of nodes to sample for calculation of info gain.")

    # MCTS Specific Args
    p.add_argument("--mcts_n_iter", type=int, default=3)
    p.add_argument("--mcts_top_k", type=int, default=10)
    
    # Data Paths (Hardcoded based on your snippet, but ideally arguments)
    p.add_argument("--runs_id", type=int, default=0, choices=range(10), help="Runs ID.")
    p.add_argument("--runs", type=str, default="scripts/runs/${dataset}.csv", help="Path to the runs file.")
    p.add_argument("--dataset", type=str, default="${dataset}", help="Dataset name.")
    p.add_argument("--infer_data", type=str, default="dataset/${dataset}/data/question_${region}_test.csv", help="Path to the test data.")

    p.add_argument("--debug", action="store_true", help="Debug mode.")

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

    # Initialize Data & Candidates
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

    # Initialize Models
    dataset_llm = Dataset.load_dataset(df_survey=df_survey, df_heldout=df_heldout, codebook=codebook, verbose=True)
    predictor_gnn = GNNElicitationPredictor.from_config(config_path=args.gnn_config_path)
    gold_map_gnn = build_gold_answers_from_edges(predictor_gnn)

    # Filter Xavail (available questions) to overlap with cand_qids
    Xavail = [q for q in dataset_llm.Xpool if q in cand_qids]
    K_per_round = int(len(dataset_llm.graph.nodes) * args.node_selection_prec)

    # Adaptive Loop
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
            rng = np.random.default_rng(args.runs_id + t)
            if args.query_selection == 'random':
                x_star = [rng.choice(Xavail)]
            
            elif args.query_selection == 'info_gain':
                # Subsample nodes for speed if graph is huge
                nodes_eval_all = list(dataset_llm.graph.nodes)
                nodes_eval = rng.choice(nodes_eval_all, size=int(len(nodes_eval_all) * args.smaple_size), replace=False).tolist()

                # LLM Inference to find best Question
                x_star, _ = select_queries(
                    dataset_llm,
                    pool_llm,
                    nodes=nodes_eval,
                    Xavail=Xavail,
                    observed=copy.deepcopy(dataset_llm.observed_dict),
                    cur_asked_queries=dataset_llm.asked_queries,
                )
           
            elif args.query_selection == 'mcts_lookahead':
                nodes_eval = list(dataset_llm.graph.nodes)
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
                    n_iter=args.mcts_n_iter,
                    top_k=args.mcts_top_k,
                    rng=rng
                )
    
            q_selected = str(x_star[0])
            print(f"Selected Query: {q_selected}")

            # GNN Selects Nodes (Local Uncertainty)
            print(f"GNN selecting {K_per_round} nodes for query {q_selected}...")
            gnn_preds = predictor_gnn.predict_for_question(q_selected, batch_size=args.gnn_batch_size)
            
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
                if args.node_selection == 'random':
                    cand = list(dataset_llm.graph.nodes)
                    k_eff = min(K_per_round, len(cand))
                    users_to_query = random.sample(cand, k_eff)
            
                elif args.node_selection == 'relational':
                    users_to_query = predictor_gnn.select_nodes(list(dataset_llm.graph.nodes), K_per_round)
                else:
                    raise ValueError(f"Invalid node selection strategy: {args.node_selection}")

                # Update GNN Predictor
                new_obs_gnn = [(uid, q_selected) for uid in users_to_query]
                predictor_gnn.add_observations(new_obs_gnn, raw_uid=True)

                # Re-predict GNN
                print(f"Re-predicting {q_selected} with updated graph evidence...")
                gnn_preds_updated = predictor_gnn.predict_for_question(q_selected, batch_size=args.gnn_batch_size)

                # Update LLM Dataset
                dataset_llm.update_observed(q_selected, users_to_query)

                if args.imputation:
                    # Inject pseudo-labels
                    num_injected = inject_pseudo_labels(
                        dataset_llm=dataset_llm,
                        q_selected=q_selected,
                        gnn_preds=gnn_preds_updated,
                        selected_users=users_to_query
                    )
                    print(f"Injected {num_injected} pseudo-labels for next round InfoGain calculation.")

            # Diagnostic Imputation
            if args.debug:
                diag_res = diagnostic_imputation(dataset_llm, q_selected, users_to_query, gold_map_gnn)
                print(f">>> [Imputation Check] Q: {diag_res['q_id']} | "
                    f"Count: {diag_res['num_injected']} | "
                    f"Precision: {diag_res['precision']:.4f} | "
                    f"Coverage: {diag_res['coverage']}")
        
            # Update LLM Asked Queries
            dataset_llm.asked_queries.append((q_selected, dataset_llm.codebook[q_selected]["question"]))
            Xavail = [q for q in Xavail if q != q_selected]

            # Evaluation & Logging
            _, _, mean_acc_llm, mean_ppl_llm, _, mean_bs_llm, held_out_predict, _ = evaluate_model(pool_llm, dataset_llm, dataset_llm.Y_heldout)
            print(f"[Round {t+1}] LLM Acc: {mean_acc_llm:.4f} -- BS: {mean_bs_llm:.4f} -- PPL: {mean_ppl_llm:.4f}")
            
            log_f.write(json.dumps({"round": t+1, "q_selected": q_selected, "llm_acc": round(mean_acc_llm, 4), "llm_ppl": round(mean_ppl_llm, 4), "llm_bs": round(mean_bs_llm, 4)}) + "\n")
            log_f.flush()
    
    print(f"Done. Logs saved to {log_full_path}")
    pool_llm.close()
