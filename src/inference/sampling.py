"""
Alg 2 — Gibbs sampler for group adaptive elicitation

This module implements a categorical Gibbs sampler over a single query X on a
node set V (graph nodes). Observed node labels are clamped; the remaining
(unobserved) node labels are sampled one-by-one conditioned on the current
state using a provided conditional-probability model.

Public entry point
------------------
    gibbs_sample_full(dataset, X, observed, model, H, N=100, burn_in=20,
                      thin=2, init=None, rng=None, progress=False)

Expected interfaces (duck-typed, no hard dependency):
    - dataset: an object with attributes
        • graph.G  (networkx.Graph) — nodes are hashable NodeId
        • graph.Y  (optional ground-truth; unused here)
        • option_sizes: Dict[QueryId, int]  — #options per query
    - model: an object with method
        • cond_probs(v, X, neighbors_answers: Dict[NodeId, int], H) -> np.ndarray
          Returns a 1D numpy array of shape (K,) of probabilities for answers
          {0, 1, ..., K-1}. The array must sum to 1 (within tolerance).
    - H: a History object (carried through, not modified).

Notes
-----
• The sampler iterates random-scan by default (node order shuffled per sweep)
  for better mixing. Deterministic behavior is guaranteed under a fixed RNG.
• Observed values are *never* changed. They are enforced each sweep.
• The state space is categorical {0..K-1} per node for the chosen query X.
• K is read from dataset.option_sizes[X].

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import copy
import numpy as np
from tqdm import tqdm
import torch

NodeId = Hashable
QueryId = Hashable


@dataclass(frozen=True)
class SamplingConfig:
    N: int = 100            # number of kept samples after burn-in/thinning
    burn_in: int = 20       # burn-in sweeps
    thin: int = 2           # keep one sample every `thin` sweeps
    random_scan: bool = True
    enforce_clamped_each_sweep: bool = True  # re-assert observed labels
    tol: float = 1e-6       # probability normalization tolerance



def _init_state(
    V: List[NodeId],
    K: int,
    rng: np.random.Generator,
) -> Dict[NodeId, int]:
    """Initialize full state y for all nodes.

    Priority: observed → init → random.
    """
    cand = ['A', 'B']
    y: Dict[NodeId, int] = {}
    for v in V:
        y[v] = cand[int(rng.integers(low=0, high=K))]
    return y


def _neighbors_answers(v, y, neighbors):
    ans: Dict[NodeId, int] = {}
    for u in neighbors:
        ans[u] = y[u]
    return ans

def options_to_string(options: dict, prefix="Options: ", sep=", "):
    """
    Render {"1":"Yes","2":"No"} -> "Options: [1] Yes, [2] No"
    Sorts numerically if possible, else lexicographically.
    """
    def try_int(s):
        try: return int(s)
        except: return None

    # Normalize keys to strings for display, but sort by numeric value if possible
    items = []
    for k, v in options.items():
        ks = str(k).strip()
        kn = try_int(ks)
        items.append((ks, v, (0, kn) if kn is not None else (1, ks)))
    items.sort(key=lambda x: x[2])

    body = sep.join(f"[{ks}] {v}" for ks, v, _ in items)
    return f"{prefix}{body}"


def batched_updates(
    y,
    observed,
    dataset,
    model,
    X,
    mode: str = "icm",
    max_iter: int = 10,
    tol: int = 0.05,
    verbose: bool = False
):
    unobs = dataset.graph.nodes
   
    trajectory = []
    for it in range(max_iter):
        if verbose:
            print(f"\nIteration {it+1}/{max_iter}")
            print(f"Update unobs: {len(unobs)}")
          
        # 1) Collect neighbor answers
        neighbors_batch = dict()
        for idx, v in enumerate(unobs):
            neigh_ans = _neighbors_answers(v, y, dataset.graph.neighbor[v])
            neighbors_batch[v] = neigh_ans
            if verbose and idx == 0:
                print(f"Node {v}: neighbors = {dataset.graph.neighbor[v]}, "
                      f"answers = {neigh_ans}")
        
        # TODO: continue with model.predict_batch / argmax / Gibbs step...
        q_text = dataset.codebook[X]["question"]
        probs_batch = model.predict_batch(nodes=dataset.graph.nodes, query=q_text, asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=observed, estimated=neighbors_batch, mode='group')
        
        # probs_batch: shape [len(nodes), K]
        if verbose:
            print('probs_batch: shape [len(nodes), K]', len(probs_batch), len(probs_batch[0]))
            print('probs_batch: shape [len(nodes), K]', probs_batch[0])

        # 3) Updates
        changes = 0
        for v, probs in zip(dataset.graph.nodes, probs_batch):
            if mode == "icm":
                new_val_ids = int(np.argmax(probs)) 
            elif mode == "gibbs":
                # new_val = int(np.random.choice(len(probs), p=probs)) + 1
                probs = torch.tensor(probs, dtype=torch.float64, device="cpu")
                new_val_ids = int(torch.multinomial(probs, 1).item()) 
            else:
                raise ValueError(f"Unknown mode: {mode}")

            new_val = "A" if new_val_ids == 0 else "B"
            if new_val != y[v]:
                changes += 1
            y[v] = new_val

        if isinstance(y, dict):
            snapshot = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                        for k, v in y.items()}
        elif torch.is_tensor(y):
            snapshot = y.detach().cpu().clone()
        else:
            snapshot = copy.deepcopy(y)  # 
        trajectory.append(snapshot)


        # 4) Convergence check (ICM only)
        perc_changes = changes / len(dataset.graph.nodes) 

        if mode == "icm" and perc_changes <= tol:
            # print(f"Converged after {it+1} sweeps (changes={perc_changes})")
            break
        # else:
        #     print(f"Current: {it+1} sweeps (changes={perc_changes})")
        # input()

    return trajectory

def probs_to_an(
    probs,   # [[p0_A, p0_B], [p1_A, p1_B], ...]
    nodes,      # same length as probs
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

    return ans

def sampling(
    dataset,
    X: QueryId,
    observed: Mapping[NodeId, int],
    model,
    iid_model,
    N: int = 100,
    rng: Optional[np.random.Generator] = None,
    mode: str = "icm",
    verbose: bool = False
) -> List[Dict[NodeId, int]]:
    """Run Gibbs sampling for a single query X over all nodes.

    Returns
    -------
    List[Dict[NodeId, int]]
        A list of `N` full assignments y^G (dicts mapping node→label).
    """
    
    cfg = SamplingConfig(N=N)
    V = dataset.graph.nodes
    K = dataset.option_sizes[X]

    # Initialize state w/ iid_model
    if iid_model is None:
        y_init = _init_state(V, K, rng)
    else:
        probs_batch_node_rest = iid_model.predict_batch(nodes=V, query=dataset.codebook[X]["question"], asked_queries=dataset.asked_queries, 
                                          neighbors=dataset.graph.neighbor, observed=dataset.observed_dict, estimated=None, mode='iid')
        y_init = probs_to_an(probs_batch_node_rest, V, labels=["A", "B"]) 

    if verbose:
        print('observed', len(observed))
        print('y_init', len(y_init))

    # sampling
    traj = batched_updates(y_init, observed, dataset, model, X, mode=mode, max_iter=N, verbose=verbose)

    return traj, y_init
 


    
