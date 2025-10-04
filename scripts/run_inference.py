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

from inference.dataset import Dataset
from inference.select_node import select_nodes
from inference.select_query import select_queries_iid, select_queries_group
from inference.sampling import sampling
from inference.impute import impute_mode
from inference.model import Meta_Model
from inference.utils import load_jsonl_as_dict_of_dict, options_to_string

NodeId = Hashable
QueryId = Hashable

def _neighbors_answers(v, y, neighbors):
    ans: Dict[NodeId, int] = {}
    for u in neighbors:
        ans[u] = y[u]
    return ans

def probs_binary_to_ans_dict(
    probs,   # [[p0_A, p0_B], [p1_A, p1_B], ...]
    nodes,      # same length as probs
    neighbor,
    labels=["A", "B"],      # optional: e.g., ("A","B")
) -> Dict[Hashable, Hashable]:

    ans: Dict[Hashable, Hashable] = {}
    for nid, row in zip(nodes, probs):
        p = np.asarray(row, dtype=np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback uniform if bad row
            p = np.array([0.5, 0.5], dtype=np.float64)
        else:
            p = p / s
        ans[nid] = labels[int(np.argmax(p))]

    neigh_ans_list = dict()
    for idx, v in enumerate(nodes):
            neigh_ans = _neighbors_answers(v, ans, neighbor[v])
            neigh_ans_list[v] = neigh_ans
    return neigh_ans_list

def evaluate_model(model, iid_model, dataset, Y_heldout, mode):
    all_acc, all_ppl = {}, {}
    held_out_predict = dict()

    opt2idx = {"A": 0, "B": 1}

    for qid in dataset.X_heldout:
        q_text = dataset.codebook[qid]["question"]
        print(f"\nEvaluation on: {qid}")

        # --- run inference ---
        if 'group' in mode:
            probs_batch_iid = iid_model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                            neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode='iid')
            
            estimated = probs_binary_to_ans_dict(probs_batch_iid, dataset.graph.nodes, neighbor=dataset.graph.neighbor, labels=["A", "B"])  

            probs_batch = model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                            neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=estimated, mode=mode)
        else:
            probs_batch = model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                            neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode=mode)
    
        probs_batch = np.array(probs_batch)  # [num_nodes, 2]
        held_out_predict[qid] = probs_batch
        print(f"[DEBUG] probs_batch.shape = {probs_batch.shape}")
        print(f"[DEBUG] first 5 probs_batch rows =\n{probs_batch[:5]}")


        gold_idx = []
        for nodeid in dataset.graph.nodes:
            gold_idx.append(opt2idx[str(Y_heldout[nodeid][qid])])
        gold_idx = np.array(gold_idx)
      
        # --- accuracy ---
        preds = probs_batch.argmax(axis=1)
        accuracy = (preds == gold_idx).mean()

        # --- perplexity ---
        chosen_probs = probs_batch[np.arange(len(gold_idx)), gold_idx]
        nll = -np.log(chosen_probs + 1e-12)
        perplexity = np.exp(nll.mean())

        all_acc[qid] = accuracy
        all_ppl[qid] = perplexity

        print(f"Accuracy={accuracy:.4f}, Perplexity={perplexity:.4f}")

    # --- overall averages ---
    mean_acc = np.mean(list(all_acc.values()))
    mean_ppl = np.mean(list(all_ppl.values()))
    print(f"\nOverall: Accuracy={mean_acc:.4f}, Perplexity={mean_ppl:.4f}")

    return all_acc, all_ppl, mean_acc, mean_ppl, held_out_predict


def setup_ddp():
    # 1. 初始化默认进程组
    dist.init_process_group(backend="nccl")

    # 2. 读取 local_rank 并绑定到对应 GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def convert(o):
        if isinstance(o, np.generic):
            return o.item()   # numpy.float32/int64 -> python float/int
        raise TypeError

import os, json

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
    model: Any,
    iid_model: Any,
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
) -> Any:
    """Run T adaptive rounds of node/query selection, observation, and imputation.
    """
    # Available queries at start
    # once at the start of the script
    if wandb:
        wb.init(
            project="gae-inference-select-node-ablation",
            name=f"{year}_{mode}_node_select_{node_select}_percent{selected_respodent}_T{T}_{x_heldout}",
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
    
    # Evaluation on Hold-out Set
    all_acc, all_ppl, mean_acc, mean_ppl, held_out_predict = evaluate_model(model, iid_model, dataset, dataset.Y_heldout, mode)
    results = {
        "iteration": 0,
        "selected_queries": "None",
        # "all_acc": all_acc,
        # "all_ppl": all_ppl,
        "mean_acc": mean_acc,
        "mean_ppl": mean_ppl,
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

        # 1) Node selection (Alg 4)
        if k_nodes == len(dataset.graph.nodes):
            V_sel = dataset.graph.nodes
        elif node_select == 'random':
            n = len(dataset.graph.nodes)
            nodes_list = list(dataset.graph.nodes)
            idx = rng.choice(n, size=k_nodes, replace=False)
            V_sel = [nodes_list[int(i)] for i in idx]             # gather by index
            Y_init = None
        else:
            x_star = [rng.choice(Xavail)]
            V_sel, Y_init = select_nodes(
                dataset,
                model,
                iid_model,
                Xavail=Xavail,
                k_nodes=k_nodes,
                observed=dataset.observed_dict,                 # no per-query observations yet in this round
                N=N_samples,
                ridge_eps=ridge_eps,
                rng=rng,
                verbose=False
            )

        print('len(V_sel)', len(V_sel))

        if 'random' in mode:
            x_star = [rng.choice(Xavail)]

        elif 'iid_entropy' in mode:
            # x_star = select_queries_iid(
            #     dataset,
            #     model,
            #     nodes=dataset.graph.nodes,
            #     Xavail=Xavail,
            #     observed=dataset.observed_dict,
            #     probs_batch_dict=held_out_predict,
            #     mode=mode,
            #     k=1,  # select how many query
            #     N=N_samples,
            #     ridge_eps=ridge_eps,
            #     rng=rng,
            #     verbose=False)
            
            gold_query = {"333b": ["332c", "334c", "333d", "355e", "332g"],
                            "355a": ["332c", "334c", "333d", "355e", "331a"],
                            "330b": ["332c", "334c", "333d",  "355e", "333c"]}

            x_star = [gold_query[dataset.X_heldout[0]][t]]

        elif 'group_entropy' in mode:
            # x_star = select_queries_group(
            #     dataset,
            #     model,
            #     iid_model = iid_model,
            #     nodes=dataset.graph.nodes,
            #     Xavail=Xavail,
            #     Y_init=Y_init,
            #     observed=dataset.observed_dict,
            #     probs_batch_dict=held_out_predict,
            #     mode=mode,
            #     k=1,  # select how many query
            #     N=N_samples,
            #     ridge_eps=ridge_eps,
            #     rng=rng,
            #     verbose=False
            # )

            gold_query = {"333b": ["332d", "332c", "331a", "332f", "334g"],
                            "355a": ["332d", "332c", "331a", "332f", "332a"],
                            "330b": ["332d", "332c", "331a",  "332f", "332a"]}
            x_star = [gold_query[dataset.X_heldout[0]][t]]


        print("Selected queries:", x_star)


        # 3) Observe labels for selected nodes on chosen query
        query_star = x_star[0]
        
        # 4) Impute
        dataset.update_observed(query_star, V_sel)

        node_rest = list(set(dataset.graph.nodes) - set(V_sel))
        if len(node_rest) > 0:
            probs_batch_node_rest = iid_model.predict_batch(nodes=node_rest, query=query_star, asked_queries=dataset.asked_queries, 
                                            neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode='iid')
            V_rest = probs_to_an(probs_batch_node_rest, node_rest, labels=["A", "B"]) 
            dataset.update_observed_estimated(query_star, V_rest=V_rest)
            assert len(V_rest) + len(V_sel) == len(dataset.graph.nodes)


        dataset.asked_queries.append((query_star, dataset.codebook[query_star]["question"]))
        

        Xavail = [q for q in Xavail if q not in x_star]

        if progress:
            print(f"Round={t} done: asked={query_star}, remaining={len(Xavail)}")

        # Evaluation on Hold-out Set
        all_acc, all_ppl, mean_acc, mean_ppl, held_out_predict = evaluate_model(model, iid_model, dataset, dataset.Y_heldout, mode)

        results = {
        "iteration": t+1,
        "selected_queries": x_star[0],
        # "all_acc": all_acc,
        # "all_ppl": all_ppl,
        "mean_acc": mean_acc,
        "mean_ppl": mean_ppl,
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="24", type=str)
    p.add_argument("--selected-respondent", default=1.0, type=float)   # note spelling
    p.add_argument("--mode", default="iid_random", choices=["group_random","group_entropy","iid_random","iid_entropy"])
    p.add_argument("--node_select", default="random", choices=["random", "entropy"])
    p.add_argument("--x-heldout", default="355a", choices=['333b', '355a', '330b', '334e', '331b', '331d', '334d', '327a', '330a', '334a'])
    p.add_argument("--checkpoint", default="")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()

    year = args.year
    selected_respodent = args.selected_respondent  # keep your variable name if other code expects it
    mode = args.mode
    x_heldout = args.x_heldout
    checkpoint = args.checkpoint
    node_select = args.node_select

    checkpoint = "/home/ruomeng/gae/logs/ces_golden_demo/24/neighbor_1/Llama-3.2-1B/20250923_070237"
    checkpoint_iid = "/home/ruomeng/gae/logs/ces_golden_demo/24/neighbor_0/Llama-3.2-1B/20250923_084121"

    jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/question_codebook.jsonl"  
    codebook = load_jsonl_as_dict_of_dict(jsonl_file, key='id')
  
    jsonl_file = f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/neighbors_{year}.jsonl"  
    neighbors_info = load_jsonl_as_dict_of_dict(jsonl_file, key='caseid')

    df_all = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/questions_test_{year}.csv")
    question_candidates = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/train_qs.csv")['train_qs'].tolist()
    question_heldout = pd.read_csv(f"/home/ruomeng/gae/dataset/ces_golden_demo/raw/{year}/holdout_qs.csv")['holdout_qs'].tolist()
    df_survey = df_all[["caseid"] + question_candidates].copy()
    df_heldout = df_all[["caseid"] + question_heldout].copy()


    dataset = Dataset.load_dataset(df_survey=df_survey, df_heldout=df_heldout, neighbors_info=neighbors_info, codebook=codebook, verbose=True)

    if 'group' in mode:
        model = Meta_Model(base_model='/home/ruomeng/model/meta-llama/Llama-3.2-1B', checkpoint=checkpoint, device_id=[0,1,2,3])
        iid_model = Meta_Model(base_model='/home/ruomeng/model/meta-llama/Llama-3.2-1B', checkpoint=checkpoint_iid, device_id=[4,5,6,7])

    else:
        model = Meta_Model(base_model='/home/ruomeng/model/meta-llama/Llama-3.2-1B', checkpoint=checkpoint_iid, device_id=[0,1,2,3,4,5,6,7])
        iid_model = model
        

    rng = np.random.default_rng(seed=2)   # create new Generator]
    T = 5
    N = 3
    ridge_eps = 1e-8

    save_path = f'./results_ablation_new/results_{mode}/{year}_{mode}_node_select_{node_select}_percent{selected_respodent}_T{T}_{x_heldout}.jsonl'

    dataset.X_heldout = [x_heldout]
    print('dataset.X_heldout',  dataset.X_heldout)

    run_group_adaptive_elicitation(dataset, model, iid_model=iid_model, mode=mode, node_select=node_select, T=T, k_nodes=int(len(dataset.graph.nodes) * selected_respodent), 
                                N_samples=N, ridge_eps=ridge_eps, rng=rng, progress=True, save_path=save_path, wandb=args.wandb)


