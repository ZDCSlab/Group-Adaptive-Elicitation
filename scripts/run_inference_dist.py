from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple
import numpy as np
import pandas as pd
import os, json
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
from collections import Counter

from inference_dist.dataset import Dataset
from inference_dist.select_node import select_nodes, select_nodes_per_community, select_nodes_occurence
from inference_dist.select_query import select_queries_iid, select_queries_group
from inference_dist.sampling import sampling
from inference_dist.impute import impute_mode
from inference_dist.model import Meta_Model, PredictorPool
from inference_dist.utils import probs_binary_to_ans_dict, load_jsonl_as_dict_of_dict, ans_to_nei_dict, build_group_candidates, allocate_k_per_group
from inference_dist.evaluation import evaluate_model


def probs_to_an(
    probs,   # [[p0_A, p0_B], [p1_A, p1_B], ...]
    nodes,      # same length as probs
    labels=["A", "B"],      # optional: e.g., ("A","B")
) -> Dict[Hashable, Hashable]:

    ans: Dict[Hashable, Hashable] = {}
    for nid, row in zip(nodes, probs):
        p = np.asarray(row, dtype=np.float64)
        print('porbs', p)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback uniform if bad row
            p = np.array([0.5, 0.5], dtype=np.float64)
        else:
            p = p / s
        ans[nid] = labels[int(np.argmax(p))]

    return ans


def save_results(results, save_path, reset=False):
    if reset and os.path.exists(save_path):
        os.remove(save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "a") as f:
        f.write(json.dumps(results, default=str) + "\n")

    print(f"Appended results to {save_path}")


def run_group_adaptive_elicitation(
    dataset: Any,
    pool,
    mode: str,
    node_select: str,
    T: int,
    k_nodes: int,
    N_samples: int = 100,
    ridge_eps: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
    save_path: str = './results/temp', 
    wandb: bool = False,
    MIG: bool = False,
    imputation: str = 'majority',
    top_k_nei: int = 20
) -> Any:
    """Run T adaptive rounds of node/query selection, observation, and imputation.
    """

    if wandb:
        wb.init(
            project="gae-inference-pro",
            name=f"{year}_{mode}_node_select_{node_select}_percent{selected_respodent}_MIG_{MIG}_T{T}_{x_heldout}_{imputation}",
            config={
                "year": args.year,
                "mode": args.mode,
                "selected_respondent": args.selected_respondent,
                "checkpoint": args.checkpoint,
            },
        )
    
    try:
        Xavail = [x for x in dataset.Xpool if x not in {q[0] for q in dataset.asked_queries}]

    except Exception:
        raise AttributeError("dataset must provide Xpool iterable of query IDs")
    
    # dataset.update_neighbors_info(K=top_k_nei)
    
    # Evaluation on Hold-out Set
    all_acc, all_ppl, mean_acc, mean_ppl, held_out_predict = evaluate_model(pool, dataset, dataset.Y_heldout, mode)
    results = {
        "iteration": 0,
        "selected_queries": "None",
        "all_acc": all_acc,
        "all_ppl": all_ppl,
        "mean_acc": mean_acc,
        "mean_ppl": mean_ppl,
        "asked_respodent": 0,
        "migs_count": 0,
    }

    save_results(results, save_path)
    print(f"Saved results to {save_path}")
    if wandb:
        wb.log(results, step=0)
 
    for t in range(T):
        if not Xavail:
            if progress:
                print(f"[alg6] round={t} stop: Xavail empty")
            break

        # Select Query
        # query_lst = ["333d", "327b", "332f", "334h", "327d"]
        # # query_lst = ["332g", "331a", "334g", "332d",  "333d"]
        # query_lst = ["333d","327b", "332f", "334h", "327d"]
        query_lst = ["332g", "331a", "330c",  "355e", "334g"]
        x_star = [query_lst[t]]
        # if 'random' in mode:
        #     x_star = [rng.choice(Xavail)]

        # elif 'iid_entropy' in mode:
        #     x_star = select_queries_iid(
        #         dataset,
        #         pool,
        #         nodes=dataset.graph.nodes,
        #         Xavail=Xavail,
        #         observed=dataset.observed_dict,
        #         probs_batch_dict=held_out_predict,
        #         mode=mode,
        #         k=1,  # select how many query
        #         N=N_samples,
        #         ridge_eps=ridge_eps,
        #         rng=rng,
        #         verbose=False)
 
        # elif 'group_entropy' in mode:
        #     x_star = select_queries_group(
        #         dataset,
        #         pool,
        #         nodes=dataset.graph.nodes,
        #         Xavail=Xavail,
        #         Y_init=None,
        #         observed=dataset.observed_dict,
        #         probs_batch_dict=held_out_predict,
        #         mode=mode,
        #         k=1,  # select how many query
        #         N=N_samples,
        #         ridge_eps=ridge_eps,
        #         rng=rng,
        #         verbose=False
        #     )
           

        print("Selected queries:", x_star)
 
        # 2) Node selection (Alg 4)
        if not MIG and k_nodes == len(dataset.graph.nodes):
            V_sel = dataset.graph.nodes
            migs_count = 0
        elif k_nodes == 0:
            V_sel = []
            migs_count = 0
        elif node_select == 'random':
            n = len(dataset.graph.nodes)
            nodes_list = list(dataset.graph.nodes)
            idx = rng.choice(len(nodes_list), size=k_nodes, replace=False)
            V_sel = [nodes_list[int(i)] for i in idx]             # gather by index
            migs_count = 0
        elif node_select == 'random_perc':
            nodes = list(dataset.graph.nodes)
            node_label_dict = dataset.graph.label   # or dataset.node.label, if that's where your mapping lives
            groups = build_group_candidates(nodes, node_label_dict)          # dict[group_id] -> [node_ids]
            k_by_group = allocate_k_per_group(groups, k_total=k_nodes)       # proportional to group sizes

            V_sel = []
            for g, cand_nodes in groups.items():
                k_g = min(k_by_group.get(g, 0), len(cand_nodes))
                if k_g <= 0:
                    continue
                # sample k_g nodes uniformly at random within this group
                idx = rng.choice(len(cand_nodes), size=k_g, replace=False)
                V_sel.extend(cand_nodes[int(i)] for i in idx)
            migs_count = 0
        elif node_select == 'entropy':
            V_sel, migs_count = select_nodes(
                dataset,
                pool,
                Xavail=x_star,
                k_nodes=k_nodes,
                observed=dataset.observed_dict,                 # no per-query observations yet in this round
                N=N_samples,
                ridge_eps=ridge_eps,
                rng=rng,
                MIG=MIG, 
                verbose=False
            )
        elif node_select == 'entropy_perc':
            V_sel, migs_count = select_nodes_per_community(
                dataset,
                pool,
                Xavail=x_star,
                k_nodes=k_nodes,
                observed=dataset.observed_dict,                 # no per-query observations yet in this round
                N=N_samples,
                ridge_eps=ridge_eps,
                rng=rng,
                MIG=MIG, 
                verbose=False
            )
        elif node_select == 'top_caseids_top5':
            path = '/home/ruomeng/gae/dataset/ces_golden_demo/raw/24/top_caseids_top5.csv'
            V_sel = pd.read_csv(path)["caseid"].astype(str).tolist()
            migs_count = 0
        elif node_select == 'occurence':
            V_sel, migs_count = select_nodes_occurence(
                dataset,
                pool,
                Xavail=x_star,
                k_nodes=k_nodes,
                observed=dataset.observed_dict,                 # no per-query observations yet in this round
                N=N_samples,
                ridge_eps=ridge_eps,
                rng=rng,
                MIG=MIG, 
                verbose=False
            )
        
   
        # print('len(V_sel)', len(V_sel))
        # records = [{"id": v, "label": dataset.graph.label[v]} for v in V_sel]
        # df = pd.DataFrame(records)
        # df.to_csv(f"./results_tuning/nodes_labels_greedy_{t}.csv", index=False)                      # CSV


        # n = len(dataset.graph.nodes)
        # nodes_list = list(dataset.graph.nodes)
        # idx = rng.choice(n, size=len(V_sel), replace=False)
        # V_sel_random = [nodes_list[int(i)] for i in idx]             # gather by index
        # records = [{"id": v, "label": dataset.graph.label[v]} for v in V_sel_random]
        # df = pd.DataFrame(records)
        # df.to_csv(f"./results_tuning/nodes_labels_random_{t}.csv", index=False)                      # CSV


        # 3) Observe labels for selected nodes on chosen query
        query_star = x_star[0]
        
        # 4) Impute
        cur_observed_dict = dataset.update_observed(query_star, V_sel)
        # dataset.update_neighbors_info(K=top_k_nei)

        node_rest = list(set(dataset.graph.nodes) - set(V_sel))
        LABELS = ("A", "B")
        if len(node_rest) > 0:
            # Majority
            if imputation == 'majority':
                V_rest, V_estimated_lst = {}, []
                for v in node_rest:
                    v_nei = dataset.graph.neighbor[v]
                    # answers from neighbors that have an entry in cur_observed_dict
                    nei_ans = [cur_observed_dict[u] for u in v_nei if u in cur_observed_dict]

                    if not nei_ans:
                        # no neighbor has an answer -> random A/B
                        V_rest[v] = rng.choice(LABELS)
                        V_estimated_lst.append(v)
                        continue

                    # majority vote among neighbors' answers
                    counts = Counter(nei_ans)
                    max_cnt = max(counts.values())
                    winners = [ans for ans, cnt in counts.items() if cnt == max_cnt]
                    print('winners', winners)

                    # tie-break randomly among winners
                    V_rest[v] = rng.choice(winners)

            # Predict
            elif imputation == 'prediction':
                probs_batch_rest = pool.predict("iid", items=node_rest, shard_arg="nodes", query=query_star, asked_queries=dataset.asked_queries,
                                            neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode="iid")
                V_rest = probs_to_an(probs_batch_rest, node_rest, labels=["A", "B"]) 
                ans_dict = cur_observed_dict | V_rest
                estimated = ans_to_nei_dict(ans_dict, dataset.graph.nodes, dataset.graph.neighbor)
                probs_batch_node_rest = pool.predict("group", items=node_rest, shard_arg="nodes", query=query_star, asked_queries=dataset.asked_queries,
                                            neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=estimated, mode="group")
                V_rest = probs_to_an(probs_batch_node_rest, node_rest, labels=["A", "B"]) 
                
            # print("=======================")
            # print('V_rest', V_rest)
            # print("=======================")
            dataset.update_observed_estimated(query_star, V_rest=V_rest)

            # if len(V_estimated_lst) > 0:
            #     probs_batch_node_rest = iid_model.predict_batch(nodes=V_estimated_lst, query=query_star, asked_queries=dataset.asked_queries, 
            #                                     neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode='iid')
            #     V_estimated = probs_to_an(probs_batch_node_rest, V_estimated_lst, labels=["A", "B"]) 
            #     dataset.update_observed_estimated(query_star, V_rest=V_estimated)
    
            #     assert len(V_rest) + len(V_estimated) + len(V_sel) == len(dataset.graph.nodes)
            # else:
            assert len(V_rest) + len(V_sel) == len(dataset.graph.nodes)
            # print(f"random_count: {random_count} / {len(V_rest)}")

        dataset.asked_queries.append((query_star, dataset.codebook[query_star]["question"]))
        Xavail = [q for q in Xavail if q not in x_star]

        if progress:
            print(f"Round={t} done: asked={query_star}, remaining={len(Xavail)}")

        # Evaluation on Hold-out Set
        all_acc, all_ppl, mean_acc, mean_ppl, held_out_predict = evaluate_model(pool, dataset, dataset.Y_heldout, mode)

        results = {
        "iteration": t+1,
        "selected_queries": x_star[0],
        "all_acc": all_acc,
        "all_ppl": all_ppl,
        "mean_acc": mean_acc,
        "mean_ppl": mean_ppl,
        "asked_respodent": len(V_sel),
        "migs_count": migs_count,
        }
        save_results(results, save_path)
        print(f"Saved results to {save_path}")
        if wandb:
            wb.log(results, step=t+1)
  
    if wandb:
        wb.finish()


# -----------------------------
# Minimal self-test / example
# -----------------------------
import argparse
import wandb as wb
# seeds_basic.py
import os, random

def set_all_seeds(seed: int = 42) -> None:
    """
    Minimal seeding for reproducible *randomness* (not forcing deterministic kernels).
    Safe to call at program start and inside Ray actors.
    """
    # Python hash & stdlib RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="24", type=str)
    p.add_argument("--selected-respondent", default=1.0, type=float)   # note spelling
    p.add_argument("--mode", default="iid_random", choices=["group_random","group_entropy","iid_random","iid_entropy"])
    p.add_argument("--node_select", default="random", choices=["random", "entropy", "random_perc", "entropy_perc", "top_caseids_top5", "occurence"])
    p.add_argument("--x-heldout", default="355a")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--MIG", action="store_true", help="Enable MIG > 0.")
    p.add_argument("--imputation", default="majority", choices=["majority","prediction"])
    p.add_argument("--top_k_nei", default=100, type=int)
    p.add_argument("--N", default=3, type=int)
    p.add_argument("--T", default=5, type=int)
    p.add_argument("--seed", default=42, type=int)   # note spelling
    return p.parse_args()



if __name__ == "__main__":


    args = parse_args()
    year = args.year
    selected_respodent = args.selected_respondent  # keep your variable name if other code expects it
    mode = args.mode
    x_heldout = args.x_heldout.split('-')
    checkpoint = args.checkpoint
    node_select = args.node_select
    seed= args.seed

    checkpoint = "/home/ruomeng/gae/logs/ces_golden_demo/24/neighbor_1/Llama-3.2-1B/20250923_070237"
    checkpoint_iid = "/home/ruomeng/gae/logs/ces_golden_demo/24/neighbor_0/Llama-3.2-1B/20250923_084121"

    jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/question_codebook.jsonl"  
    codebook = load_jsonl_as_dict_of_dict(jsonl_file, key='id')
  
    jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/neighbors_noisy50_{year}.jsonl"  
    neighbors_info = load_jsonl_as_dict_of_dict(jsonl_file, key='caseid')

    df_all = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/questions_test_{year}.csv")
    question_candidates = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/train_qs.csv")['train_qs'].tolist()
    question_heldout = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/holdout_qs.csv")['holdout_qs'].tolist()
    df_survey = df_all[["caseid"] + question_candidates].copy()
    df_heldout = df_all[["caseid"] + question_heldout].copy()

    dataset = Dataset.load_dataset(df_survey=df_survey, df_heldout=df_heldout, neighbors_info=neighbors_info, codebook=codebook, top_k_nei=args.top_k_nei, verbose=True)

    if 'group' in mode:
        pool = PredictorPool(
            group_cfg={"base_model": '/home/ruomeng/model/meta-llama/Llama-3.2-1B', "checkpoint": checkpoint},
            iid_cfg={"base_model": '/home/ruomeng/model/meta-llama/Llama-3.2-1B', "checkpoint": checkpoint_iid},
            group_gpus=3, iid_gpus=3)
    else:
        pool = PredictorPool(
            group_cfg={"base_model": '/home/ruomeng/model/meta-llama/Llama-3.2-1B', "checkpoint": checkpoint_iid},
            iid_cfg={"base_model": '/home/ruomeng/model/meta-llama/Llama-3.2-1B', "checkpoint": checkpoint_iid},
            group_gpus=3, iid_gpus=3)

    set_all_seeds(seed=seed)
    rng = np.random.default_rng(seed=seed)   # create new Generator
    T = args.T
    N = args.N
    ridge_eps = 1e-8

    if args.MIG:
        save_path = f'./results_tuning_occur/results_{mode}/{year}_{mode}_node_select_{node_select}_MIG_T{T}_{x_heldout}_{args.imputation}.jsonl'

    else:
        save_path = f'./results_tuning_occur/results_{mode}/{year}_{mode}_node_select_{node_select}_N{args.N}_percent{selected_respodent}_T{T}_{x_heldout}_{args.imputation}_top_k_nei{args.top_k_nei}.jsonl'

    dataset.X_heldout = x_heldout
    print('dataset.X_heldout',  dataset.X_heldout)

    run_group_adaptive_elicitation(dataset, pool=pool, mode=mode, node_select=node_select, T=T, 
                                   k_nodes=int(len(dataset.graph.nodes) * selected_respodent), 
                                   N_samples=N, ridge_eps=ridge_eps, rng=rng, progress=True, save_path=save_path, wandb=args.wandb, 
                                   MIG=args.MIG, imputation=args.imputation, top_k_nei=args.top_k_nei)


