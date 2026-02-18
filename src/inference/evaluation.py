from __future__ import annotations
import numpy as np
import json
from sklearn.metrics import f1_score


def evaluate_model(pool, dataset, Y_heldout):
    all_acc, all_ppl, all_f1, all_bs = {}, {}, {}, {}
    held_out_predict = dict()
    
    # Store the correctness of each user
    # key: nodeid, value: list of 0 or 1
    user_results = {nodeid: [] for nodeid in dataset.graph.nodes}

    for qid in dataset.X_heldout:
        q_text = dataset.codebook[qid]["question"]
        options = list(dataset.codebook[qid]["options"].keys())
        
        # Run inference
        probs_batch = pool.predict(items=dataset.graph.nodes, shard_arg="nodes", query=q_text, 
                                    candidate_options=options, asked_queries=dataset.asked_queries, 
                                    observed=dataset.observed_dict)

        probs_batch = np.array(probs_batch) 
        held_out_predict[qid] = probs_batch

        gold_idx = []
        opt2idx = dict(zip(options, range(len(options))))
        for nodeid in dataset.graph.nodes:
            gold_idx.append(opt2idx[str(Y_heldout[nodeid][qid])])
        gold_idx = np.array(gold_idx)

        # Calculate Brier Score (BS)
        # Convert gold_idx to one-hot encoding
        num_classes = probs_batch.shape[1]
        gold_one_hot = np.eye(num_classes)[gold_idx]

        # Calculate the sum of squared errors for each sample, then take the average
        brier_scores = np.sum((probs_batch - gold_one_hot) ** 2, axis=1)
        bs = np.mean(brier_scores)
      
        # Calculate accuracy
        preds = probs_batch.argmax(axis=1)
        
        # Record the correctness of each user
        is_correct = (preds == gold_idx)  # True/False for each user
        for i, nodeid in enumerate(dataset.graph.nodes):
            user_results[nodeid].append(is_correct[i])

        # Calculate accuracy at the question level
        accuracy = is_correct.mean()
        f1 = f1_score(gold_idx, preds, average='macro')
        chosen_probs = probs_batch[np.arange(len(gold_idx)), gold_idx]
        nll = -np.log(chosen_probs + 1e-12)
        perplexity = np.exp(nll.mean())

        all_acc[qid] = accuracy
        all_ppl[qid] = perplexity
        all_f1[qid] = f1
        all_bs[qid] = bs

    # Calculate the average accuracy for each user
    user_acc_dict = {nodeid: float(np.mean(hits)) for nodeid, hits in user_results.items()}

    # --- overall averages ---
    mean_acc = float(np.mean(list(all_acc.values())))
    mean_ppl = float(np.mean(list(all_ppl.values())))
    mean_f1 = float(np.mean(list(all_f1.values())))
    mean_bs = float(np.mean(list(all_bs.values())))
    
    # Add user_acc_dict to the return values
    return all_acc, all_ppl, mean_acc, mean_ppl, mean_f1, mean_bs, held_out_predict, user_acc_dict


def evaluate_model_on_hard_groups(pool, dataset, Y_heldout, hard_groups_path='dataset/opinionqa/hard_user_groups_west.json'):
    """
    Evaluate model on hard user groups (top_50%, top_30%, top_10%, top_5%).
    More efficient: runs inference once, then filters results by group.
    
    Args:
        pool: Predictor pool for inference
        dataset: Dataset object with graph and codebook
        Y_heldout: Ground truth answers for heldout questions
        hard_groups_path: Path to JSON file containing hard user groups
    
    Returns:
        dict: Results for each group with keys 'top_50%', 'top_30%', 'top_10%', 'top_5%'
              Each entry contains: {'mean_acc', 'mean_ppl', 'mean_f1'}
    """
    # Load hard user groups
    with open(hard_groups_path, 'r') as f:
        hard_groups = json.load(f)
    
    # Convert caseids in hard_groups to strings (matching dataset format)
    hard_groups_str = {}
    for group_name, caseids in hard_groups.items():
        hard_groups_str[group_name] = [str(cid) for cid in caseids]
    
    group_names = ['top_50%', 'top_30%', 'top_10%', 'top_5%']
    
    # Collect all nodes that are in any hard group (union of all groups)
    all_group_nodes = set()
    for group_name in group_names:
        if group_name in hard_groups_str:
            all_group_nodes.update(hard_groups_str[group_name])
    
    # Filter to only nodes that exist in dataset
    all_group_nodes = [node for node in all_group_nodes if node in dataset.graph.nodes]
    print(f"\n{'='*60}")
    print(f"Running inference on {len(all_group_nodes)} nodes (union of all hard groups)")
    print(f"{'='*60}")
    
    if len(all_group_nodes) == 0:
        print("Warning: No nodes from hard groups found in dataset")
        return {g: {'mean_acc': 0.0, 'mean_ppl': 0.0, 'mean_f1': 0.0} for g in group_names}
    
    # Create mapping from node to index for efficient lookup
    node_to_idx = {node: i for i, node in enumerate(all_group_nodes)}
    
    # Store predictions and gold labels for all questions
    predictions = {}
    gold_labels = {}
    
    # Run inference once for all questions
    for qid in dataset.X_heldout:
        q_text = dataset.codebook[qid]["question"]
        options = list(dataset.codebook[qid]["options"].keys())
        print(f"\nRunning inference on: {qid}")
        
        # Run inference on all group nodes
        probs_batch = pool.predict(
            items=all_group_nodes, 
            shard_arg="nodes", 
            query=q_text, 
            candidate_options=options, 
            asked_queries=dataset.asked_queries, 
            observed=dataset.observed_dict
        )
        
        probs_batch = np.array(probs_batch)
        
        # Store gold labels for nodes that have them
        opt2idx = dict(zip(options, range(len(options))))
        qid_gold_labels = {}
        
        for i, nodeid in enumerate(all_group_nodes):
            if nodeid in Y_heldout and qid in Y_heldout[nodeid]:
                qid_gold_labels[i] = opt2idx[str(Y_heldout[nodeid][qid])]
        
        predictions[qid] = probs_batch
        gold_labels[qid] = qid_gold_labels
        print(f"  Stored predictions for {len(qid_gold_labels)} nodes with gold labels")
    
    # Now calculate metrics for each hard group by filtering the stored predictions
    results = {}
    
    for group_name in group_names:
        if group_name not in hard_groups_str:
            print(f"\nWarning: Group {group_name} not found in hard_groups")
            results[group_name] = {
                'mean_acc': 0.0, 'mean_ppl': 0.0, 'mean_f1': 0.0
            }
            continue
        
        # Get nodes for this group (intersection with dataset nodes)
        group_nodes = [node for node in hard_groups_str[group_name] if node in dataset.graph.nodes]
        print(f"\n{'='*60}")
        print(f"Calculating metrics for {group_name}: {len(group_nodes)} nodes")
        print(f"{'='*60}")
        
        if len(group_nodes) == 0:
            print(f"Warning: No nodes from {group_name} found in dataset")
            results[group_name] = {
                'mean_acc': 0.0, 'mean_ppl': 0.0, 'mean_f1': 0.0
            }
            continue
        
        # Get indices of group nodes in our all_group_nodes list
        group_indices = [node_to_idx[node] for node in group_nodes if node in node_to_idx]
        
        all_acc, all_ppl, all_f1 = {}, {}, {}
        
        for qid in dataset.X_heldout:
            probs_batch = predictions[qid]
            qid_gold_labels = gold_labels[qid]
            
            # Filter to only group nodes that have gold labels
            valid_indices = [idx for idx in group_indices if idx in qid_gold_labels]
            
            if len(valid_indices) == 0:
                continue
            
            # Get predictions and gold labels for this group
            group_probs = probs_batch[valid_indices]
            group_gold = np.array([qid_gold_labels[idx] for idx in valid_indices])
            
            # --- accuracy ---
            preds = group_probs.argmax(axis=1)
            accuracy = (preds == group_gold).mean()
            
            # --- f1 score ---
            f1 = f1_score(group_gold, preds, average='macro')
            
            # --- perplexity ---
            chosen_probs = group_probs[np.arange(len(group_gold)), group_gold]
            nll = -np.log(chosen_probs + 1e-12)
            perplexity = np.exp(nll.mean())
            
            all_acc[qid] = accuracy
            all_ppl[qid] = perplexity
            all_f1[qid] = f1
        
        # --- overall averages for this group ---
        if len(all_acc) > 0:
            mean_acc = float(np.mean(list(all_acc.values())))
            mean_ppl = float(np.mean(list(all_ppl.values())))
            mean_f1 = float(np.mean(list(all_f1.values())))
        else:
            mean_acc = mean_ppl = mean_f1 = 0.0
        
        results[group_name] = {
            'mean_acc':round(mean_acc, 4),
            'mean_ppl':round(mean_ppl, 4),
            'mean_f1':round(mean_f1, 4)
        }
        
        print(f"{group_name} Overall: Acc={round(mean_acc, 4)}, F1={round(mean_f1, 4)}, PPL={round(mean_ppl, 4)}")
    
    return results


