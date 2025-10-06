from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple
from inference_dist.utils import *



def evaluate_model(pool, dataset, Y_heldout, mode):
    all_acc, all_ppl = {}, {}
    held_out_predict = dict()

    opt2idx = {"A": 0, "B": 1}

    for qid in dataset.X_heldout:
        q_text = dataset.codebook[qid]["question"]
        print(f"\nEvaluation on: {qid}")

        # --- run inference ---
        if 'group' in mode:
            probs_batch_iid = pool.predict("iid", items=dataset.graph.nodes, shard_arg="nodes", query=q_text, asked_queries=dataset.asked_queries,
                                        neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode="iid")
            estimated = probs_binary_to_ans_dict(probs_batch_iid, dataset.graph.nodes, neighbor=dataset.graph.neighbor, labels=["A", "B"])  
            probs_batch= pool.predict("group", items=dataset.graph.nodes, shard_arg="nodes", query=q_text, asked_queries=dataset.asked_queries,
                                        neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=estimated, mode="group")
            
            
        else:
            # probs_batch = model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
            #                                 neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode=mode)
            probs_batch = pool.predict("iid", items=dataset.graph.nodes, shard_arg="nodes", query=q_text, asked_queries=dataset.asked_queries,
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

