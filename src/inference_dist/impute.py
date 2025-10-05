from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Tuple
import numpy as np
from typing import Dict, Hashable, Mapping, Optional, List
import numpy as np
from inference_dist.sampling import sampling

NodeId = Hashable
QueryId= Hashable

def impute_mode(samples: List[Dict[NodeId, int]]) -> Dict[NodeId, int]:
    """
    Alg 3: return y_hat^G by taking the majority label at each node
    from the empirical distribution D produced by Gibbs (Alg 2).
    """
    if not samples:
        raise ValueError("Empty samples for imputation.")
    nodes = list(samples[0].keys())
    y_hat: Dict[NodeId, int] = {}
    for v in nodes:
        counts = Counter(s[v] for s in samples)
        # deterministic tie-break: smallest label wins
        y_hat[v] = min([lab for lab, cnt in counts.items() if cnt == max(counts.values())])
    return y_hat

def empirical_probs(samples: List[Dict[NodeId, int]]) -> Dict[NodeId, np.ndarray]:
    """
    Optional: return per-node categorical probabilities estimated from D.
    Useful if you want uncertainty (entropy, variance) downstream.
    """
    if not samples:
        raise ValueError("Empty samples for probability estimation.")
    nodes = list(samples[0].keys())
    # infer K by scanning labels
    max_label = 0
    for s in samples:
        max_label = max(max_label, max(int(v) for v in s.values()))
    K = max_label + 1

    prob: Dict[NodeId, np.ndarray] = {}
    for v in nodes:
        counts = np.zeros(K, dtype=float)
        for s in samples:
            counts[int(s[v])] += 1.0
        prob[v] = counts / counts.sum()
    return prob

def impute_argmax(samples: List[Dict[NodeId, int]]) -> Dict[NodeId, int]:
    """
    Same as mode but computed via empirical_probs + argmax.
    """
    probs = empirical_probs(samples)
    return {v: int(np.argmax(p)) for v, p in probs.items()}

def impute_from_gibbs(
    samples: List[Dict[NodeId, int]],
    strategy: str = "mode",
) -> Dict[NodeId, int]:
    """
    Convenience wrapper: choose 'mode' (default) or 'argmax'.
    """
    if strategy == "mode":
        return impute_mode(samples)
    if strategy == "argmax":
        return impute_argmax(samples)
    raise ValueError(f"Unknown strategy: {strategy}")



def impute_query_full(
    dataset,
    X: QueryId,
    observed: Mapping[NodeId, int],
    model,
    H,
    *,
    N: int = 100,
    burn_in: int = 20,
    thin: int = 2,
    init: Optional[Mapping[NodeId, int]] = None,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "mode",
):
    """
    Runs Alg 2 to get D, then returns (y_hat^G, probs) for Alg 3.
    """
    samples = gibbs_sample_full(
        dataset, X, observed, model, H,
        N=N, burn_in=burn_in, thin=thin, init=init, rng=rng, progress=False
    )
    y_hat = impute_from_gibbs(samples, strategy=strategy)
    probs = empirical_probs(samples)  # optional uncertainty
    return y_hat, probs
