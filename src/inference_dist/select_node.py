from __future__ import annotations
from typing import Dict, Hashable, Union, List, Mapping, Optional, Callable, Sequence
import numpy as np
from numpy.linalg import LinAlgError
from inference_dist.sampling import sampling
from multiprocessing import Pool
from tqdm import tqdm
import random
import torch

NodeId = Hashable
QueryId = Hashable
Label = Union[int, float]


def _formulate_matrix(
    samples: List[Optional[Dict[NodeId, Label]]],
    nodes: List[NodeId],
    *,
    map_label: Optional[Union[Dict[Label, float], Callable[[Label], float]]] = None,
    fill: float = np.nan,
    dtype=np.float32,
    ignore_unknown_nodes: bool = True,
) -> np.ndarray:
    """
    Build an S × V matrix (no one-hot). Each row s is a full graph assignment for one sample.
    Vectorized fill for efficiency.
    """
    S, V = len(samples), len(nodes)
    Z = np.full((S, V), fill, dtype=dtype)
    node_index = {v: i for i, v in enumerate(nodes)}

    # Prepare mapper
    if map_label is None:
        def _map(x): return float(x)
    elif callable(map_label):
        _map = map_label
    else:
        mapping = {k: float(v) for k, v in map_label.items()}
        def _map(x, _m=mapping):
            return _m[x]

    # Collect all row indices, col indices, values
    row_idx, col_idx, vals = [], [], []
    for s, y in enumerate(samples):
        if not y:
            continue
        for v, lab in y.items():
            i = node_index.get(v)
            if i is None:
                if ignore_unknown_nodes:
                    continue
                raise KeyError(f"Unknown node {v!r} not found in `nodes`.")
            if lab is None:
                continue
            try:
                val = _map(lab)
            except Exception as e:
                raise ValueError(f"Could not map label {lab!r} for node {v!r}") from e
            row_idx.append(s)
            col_idx.append(i)
            vals.append(val)

    if row_idx:
        row_idx = np.fromiter(row_idx, dtype=np.intp)
        col_idx = np.fromiter(col_idx, dtype=np.intp)
        vals = np.fromiter(vals, dtype=dtype)
        Z[row_idx, col_idx] = vals

    return Z


def _covariance(Z: np.ndarray):
    """
    Z: (S, V) with entries in {+1, -1}, no NaNs.
    Returns Σ (V, V) unbiased sample covariance.
    """
    Z = np.asarray(Z, dtype=np.float64)
    S = Z.shape[0]
    mu = Z.mean(axis=0)                            # (V,)
    Sigma = (Z.T @ Z - S * np.outer(mu, mu)) / (S - 1)
    return Sigma, mu

import numpy as np

# def _covariance_spd(Z: np.ndarray,
#                     unbiased: bool = True,
#                     ridge: float = 0.0,
#                     max_tries: int = 8):
#     """
#     Z: (S, V) with entries in {+1, -1} (or real numbers), no NaNs.
#     Returns:
#       Sigma_spd: (V,V) SPD covariance suitable for MIG,
#       mu: (V,),
#       report: dict with diagnostics.
#     Strategy:
#       1) unbiased sample covariance
#       2) symmetrize
#       3) shrinkage towards tau*I if rank-deficient or ill-conditioned
#       4) jitter escalation until Cholesky succeeds
#     """
#     Z = np.asarray(Z, dtype=np.float64)
#     assert Z.ndim == 2, "Z must be 2D (S,V)"
#     S, V = Z.shape
#     if S < 2:
#         raise ValueError("Need at least 2 samples to compute unbiased covariance.")

#     mu = Z.mean(axis=0)                      # (V,)
#     # unbiased sample covariance
#     if unbiased:
#         Sigma = (Z.T @ Z - S * np.outer(mu, mu)) / (S - 1)
#     else:
#         # MLE: 1/S
#         Sigma = (Z.T @ Z - S * np.outer(mu, mu)) / S

#     # Step 1: symmetrize & optional base ridge
#     Sigma = 0.5 * (Sigma + Sigma.T)
#     if ridge > 0.0:
#         Sigma = Sigma + ridge * np.eye(V)

#     # Step 2: decide if shrinkage is needed
#     # 事实：当 S-1 < V 时，unbiased 协方差至少 rank-deficient（半正定），需收缩/加岭。
#     need_shrink = (S - 1) < V

#     # 以 tau*I 为目标做简单的对角收缩；alpha 自适应（样本少时更大）
#     # 经验设定：alpha0 = min(0.99, max(0.0, V / max(S - 1, 1) * 0.1))
#     tr = float(np.trace(Sigma)) / V if V > 0 else 1.0
#     T = tr * np.eye(V)
#     alpha = 0.0
#     if need_shrink:
#         alpha = min(0.99, max(0.0, V / max(S - 1, 1) * 0.1))
#         Sigma = (1.0 - alpha) * Sigma + alpha * T
#         Sigma = 0.5 * (Sigma + Sigma.T)

#     # Step 3: robustify with jitter escalation until Cholesky succeeds
#     jitter = 0.0
#     ok = False
#     for _ in range(max_tries):
#         try:
#             np.linalg.cholesky(Sigma + jitter * np.eye(V))
#             ok = True
#             break
#         except np.linalg.LinAlgError:
#             # 先尝试加小 jitter；若还失败，再提高收缩比例 alpha
#             if jitter == 0.0:
#                 jitter = 1e-12
#             else:
#                 jitter *= 10.0

#             if not need_shrink:
#                 # 若原本认为不需要收缩，但仍失败，则启用轻微收缩
#                 alpha = max(alpha, 0.05)
#                 need_shrink = True

#             # 逐步增大收缩到 tau*I（最多到 0.99）
#             alpha = min(0.99, alpha * 1.5 if alpha > 0 else 0.05)
#             Sigma = (1.0 - alpha) * (0.5 * (Sigma + Sigma.T)) + alpha * T
#             Sigma = 0.5 * (Sigma + Sigma.T)

#     if not ok:
#         # 最后再加一个保底 jitter
#         jitter += 1e-9
#         # 若仍不行就抛错（基本不会）
#         np.linalg.cholesky(Sigma + jitter * np.eye(V))

#     Sigma_spd = Sigma + jitter * np.eye(V)

#     # 诊断信息
#     try:
#         cond = np.linalg.cond(Sigma_spd)
#     except np.linalg.LinAlgError:
#         cond = np.inf

#     report = {
#         "S": S,
#         "V": V,
#         "used_shrinkage": bool(alpha > 0),
#         "shrinkage_alpha": float(alpha),
#         "jitter": float(jitter),
#         "condition_number": float(cond),
#         "ok_for_use": True
#     }
#     return Sigma_spd, mu, report


from typing import List, Sequence, Hashable, Union, Dict, Iterable, Optional
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

import numpy as np
from numpy.linalg import cholesky, solve

# ---------- helpers (与之前一致) ----------
def symmetrize(S):
    return 0.5 * (S + S.T)

def omega_diag_from_cov(S):
    Omega = np.linalg.inv(S)
    return np.diag(Omega)

def chol_extend(L, S, A, v, jitter=0.0):
    if len(A) == 0:
        val = S[v, v] + jitter
        if val <= 0:
            raise np.linalg.LinAlgError("Non-positive variance on diagonal.")
        return np.array([[np.sqrt(val)]], dtype=float)
    S_vA = S[np.ix_([v], A)]
    y = solve(L, S_vA.T)
    diag_term = S[v, v] + jitter - float(y.T @ y)
    if diag_term <= 0:
        diag_term = 1e-12  # 数值兜底
    alpha = np.sqrt(diag_term)
    m = L.shape[0]
    L_new = np.zeros((m+1, m+1), dtype=float)
    L_new[:m, :m] = L
    L_new[m, :m] = y.T
    L_new[m, m] = alpha
    return L_new

def batch_var_given_A(S, A, C, L):
    """一次性算出所有候选 C 的 Var(v|A)。"""
    if len(A) == 0 or L.size == 0:
        return np.diag(S)[C].astype(float)
    S_AC = S[np.ix_(A, C)]             # (|A|, |C|)
    Y = solve(L, S_AC)                 # L Y = S_AC
    red = np.sum(Y * Y, axis=0)        # 列向量的平方和
    return np.diag(S)[C].astype(float) - red

# ---------- multi-Sigma greedy selection----------
# def greedy_select_node_mig_multi(
#     Sigma_list,               # list of (n,n) 协方差矩阵
#     k,                        # 选择个数
#     A_init=None,
#     ridge=1e-6,
#     weights=None,             # 可选，对每个 Sigma 的权重，默认均等
#     clip_var=1e-15,           # 防止 log(<=0)
#     verbose=False
# ):
#     """
#     在多个协方差上聚合节点自身 MIG，逐步贪心选择。
#     返回: A (选择顺序), mig_hist (每步的总增益), eval_hist (每轮候选数)
#     """
#     S_list = []
#     L_list  = []
#     var_all_list = []
#     n = Sigma_list[0].shape[0]
#     A = [] if A_init is None else list(A_init)
#     chosen = set(A)
#     C = np.array([i for i in range(n) if i not in chosen], dtype=int)

#     S_count = len(Sigma_list)
#     if weights is None:
#         w = np.ones(S_count, dtype=float)
#     else:
#         w = np.asarray(weights, dtype=float)
#         assert w.shape == (S_count,)
#     # 预处理每个 Sigma：对称+ridge、Omega_diag、初始 L
#     for Sigma in Sigma_list:
#         S = symmetrize(Sigma) + ridge * np.eye(n)
#         S_list.append(S)
#         # Var(v | V\{v}) = 1 / Omega_vv
#         Omega_diag = omega_diag_from_cov(S)
#         var_all = 1.0 / Omega_diag
#         var_all_list.append(var_all)
#         # 初始 Cholesky
#         L = cholesky(S[np.ix_(A, A)]) if len(A) > 0 else np.zeros((0,0))
#         L_list.append(L)

#     mig_hist = []
#     eval_hist = []

#     for t in range(k):
#         if C.size == 0:
#             break

#         # 聚合所有 Sigma 的 MIG：sum_s w_s * [log Var_s(v|A) - log Var_all_s(v)]
#         agg_mig = np.zeros(C.size, dtype=float)
#         for s, (S, L, var_all_s, ws) in enumerate(zip(S_list, L_list, var_all_list, w)):
#             var_C_s = batch_var_given_A(S, A, C.tolist(), L)
#             # 数值裁剪避免 log 非法
#             var_C_s = np.maximum(var_C_s, clip_var)
#             vals = np.log(var_C_s) - np.log(var_all_s[C])
#             agg_mig += ws * vals

#         idx = int(np.argmax(agg_mig))
#         v_star = int(C[idx])
#         mig_best = float(agg_mig[idx])

#         if verbose:
#             print(f"[Multi-BatchGreedy] step {t+1}: pick v={v_star}, sum-MIG={mig_best:.6f}")

#         # 同步扩展每个 Sigma 的 Cholesky
#         for i in range(S_count):
#             try:
#                 L_list[i] = chol_extend(L_list[i], S_list[i], A, v_star, jitter=0.0)
#             except np.linalg.LinAlgError:
#                 L_list[i] = chol_extend(L_list[i], S_list[i], A, v_star, jitter=1e-9)

#         # 更新集合
#         A.append(v_star)
#         chosen.add(v_star)
#         C = C[C != v_star]

#         mig_hist.append(mig_best)
#         eval_hist.append(len(C) + 1)  # 这一轮评估的候选数（每个 Sigma 都做了批量求解）

#     return A, mig_hist, eval_hist


import numpy as np
from scipy.linalg import cholesky as chol, solve_triangular

def _batch_var_given_complement_precision(Omega, A, C, R, clip_var=1e-12):
    C = np.asarray(C, dtype=int)
    if len(A) == 0:
        denom = np.maximum(np.diag(Omega)[C], clip_var)
        return 1.0 / denom

    A = np.asarray(A, dtype=int)
    Ovv = np.diag(Omega)[C]                     # [|C|]
    OAv = Omega[np.ix_(A, C)]                   # [|A|, |C|]
    # R is chol(Ω_AA) lower ⇒ R R^T = Ω_AA
    y = solve_triangular(R, OAv, lower=True, check_finite=False)
    x = solve_triangular(R.T, y, lower=False, check_finite=False)  # Ω_AA^{-1} Ω_Av
    denom = Ovv - np.einsum('ac,ac->c', OAv, x)
    denom = np.maximum(denom, clip_var)
    return 1.0 / denom


# ---------- multi-Sigma greedy selection (paper-accurate) ----------
def greedy_select_node_mig_multi_exact(
    Sigma_list,               # list of (n,n) covariance matrices
    k,                        # number to select
    A_init=None,
    ridge=1e-6,
    weights=None,             # optional weights per Sigma, default uniform
    clip_var=1e-15,           # guard for logs / inversion
    MIG=False, 
    verbose=False
):
    """
    Multi-Σ greedy node selection using the paper's exact MIG:
        MIG(v; A) = log Var(v | A) - log Var(v | V \\ {v} \\ A)

    - First term uses Σ and the Cholesky of Σ_AA (as in the original).
    - Second term uses Ω=Σ^{-1} and the Cholesky of Ω_AA (new).

    Returns: A (selection order), mig_hist (gain at each step), eval_hist (candidates evaluated per step)
    """
    n = Sigma_list[0].shape[0]
    A = [] if A_init is None else list(A_init)
    chosen = set(A)
    C = np.array([i for i in range(n) if i not in chosen], dtype=int)

    S_count = len(Sigma_list)
    w = np.ones(S_count, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    assert w.shape == (S_count,)

    # Per-Σ caches
    S_list, L_list = [], []             # for first term Var(v | A)
    Omega_list, R_list = [], []         # for second term Var(v | complement of A)

    for Sigma in Sigma_list:
        # Stabilize S and Ω
        S = symmetrize(Sigma) + ridge * np.eye(n)
        S_list.append(S)

        L = chol(S[np.ix_(A, A)], lower=True, check_finite=False) if len(A) > 0 else np.zeros((0, 0))
        L_list.append(L)

        # Build Ω with symmetry fix; allow rescue ridge
        try:
            Omega = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_safe = S + 1e-9 * np.eye(n)
            Omega = np.linalg.inv(S_safe)
        Omega = symmetrize(Omega)   # important after inv!
        Omega_list.append(Omega)

        R = chol(Omega[np.ix_(A, A)], lower=True, check_finite=False) if len(A) > 0 else np.zeros((0, 0))
        R_list.append(R)

    mig_hist, eval_hist = [], []

    for t in range(k):
        if C.size == 0:
            break

        agg_mig = np.zeros(C.size, dtype=float)
        for s, (S, L, Omega, R, ws) in enumerate(zip(S_list, L_list, Omega_list, R_list, w)):
            var_C   = batch_var_given_A(S, A, C.tolist(), L)
            var_C   = np.maximum(var_C, clip_var)
            var_comp= _batch_var_given_complement_precision(Omega, A, C, R, clip_var=clip_var)

            # per-Σ sanity (optional but good)
            if not (np.all(np.isfinite(var_C)) and np.all(var_C > 0) and
                    np.all(np.isfinite(var_comp)) and np.all(var_comp > 0)):
                raise RuntimeError("Non-finite or non-positive conditional variance detected.")
            viol = (var_comp - var_C) > 1e-8
            if np.any(viol) and verbose:
                print("[WARN][Σ idx {}] Var(v|Z) > Var(v|A) for candidates: {}".format(s, C[viol][:10]))

            agg_mig += ws * (np.log(var_C) - np.log(var_comp))

        idx = int(np.argmax(agg_mig))
        v_star = int(C[idx])
        mig_best = float(agg_mig[idx])

        if verbose:
            print(f"[Multi-Σ Greedy (exact)] step {t+1}: pick v={v_star}, sum-MIG={mig_best:.6f}")

        # Extend both Choleskies for every Σ_s
        for i in range(S_count):
            try:
                L_list[i] = chol_extend(L_list[i], S_list[i], A, v_star, jitter=0.0)
            except np.linalg.LinAlgError:
                L_list[i] = chol_extend(L_list[i], S_list[i], A, v_star, jitter=1e-9)
            try:
                R_list[i] = chol_extend(R_list[i], Omega_list[i], A, v_star, jitter=0.0)
            except np.linalg.LinAlgError:
                R_list[i] = chol_extend(R_list[i], Omega_list[i], A, v_star, jitter=1e-12)

        # Update sets
        if MIG and mig_best < 1e-12:
            break
        A.append(v_star)
        chosen.add(v_star)
        C = C[C != v_star]

        mig_hist.append(mig_best)
        eval_hist.append(len(C) + 1)

    return A, mig_hist, eval_hist


def validate_sigma(Sigma, sym_tol=1e-8, pd_tol=1e-12, max_tries=6):
    report = {}
    Sigma = np.asarray(Sigma, dtype=float)

    # Shape checks
    report["is_square"] = (Sigma.ndim == 2 and Sigma.shape[0] == Sigma.shape[1])
    n = Sigma.shape[0] if report["is_square"] else None

    # Finite
    report["all_finite"] = np.isfinite(Sigma).all()

    if not report["is_square"] or not report["all_finite"]:
        report["ok_for_use"] = False
        return report

    # Symmetry check
    Ssym = 0.5 * (Sigma + Sigma.T)
    sym_err = np.linalg.norm(Sigma - Sigma.T, ord="fro") / max(1.0, np.linalg.norm(Ssym, ord="fro"))
    report["is_symmetric"] = (sym_err <= sym_tol)
    report["sym_err_fro"] = sym_err

    # Positive-definiteness via robust Cholesky (with jitter escalation)
    jitter = 0.0
    ok_pd = False
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(Ssym + jitter * np.eye(n))
            ok_pd = True
            break
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0

    report["spd_cholesky"] = ok_pd
    report["needed_jitter"] = jitter if ok_pd else None

    # Invertibility / precision-diagonal
    # (Only try if SPD check passed; otherwise inversion is invalid.)
    if ok_pd:
        try:
            Omega = np.linalg.inv(Ssym + jitter * np.eye(n))
            diag_ok = np.all(np.diag(Omega) > pd_tol)
            report["invertible"] = True
            report["precision_diag_positive"] = bool(diag_ok)
            report["cond_number"] = np.linalg.cond(Ssym + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            report["invertible"] = False
            report["precision_diag_positive"] = False
            report["cond_number"] = np.inf
    else:
        report["invertible"] = False
        report["precision_diag_positive"] = False
        report["cond_number"] = np.inf

    # Final verdict
    report["ok_for_use"] = bool(report["is_square"] and report["all_finite"] and report["spd_cholesky"] and report["invertible"] and report["precision_diag_positive"])
    return report


def select_nodes(
    dataset,
    pool,
    Xavail: List[QueryId],
    k_nodes: int,
    observed: Dict,
    N: int = 100,
    ridge_eps: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
    MIG: bool = False, 
    verbose: bool = False
) -> List:
    """Greedy node selection (Alg 4) using Gaussian log-det gains over one-hot blocks.
    """
  
    nodes = dataset.graph.nodes
    K_per_query = dataset.option_sizes

    if verbose:
        print('nodes', len(nodes))
        print('K_per_query', K_per_query)

    # 1) For each query x, obtain (or build) covariance Σ_x
    Sigmas: Dict[QueryId, np.ndarray] = {}
    Y_init = dict()
 
    for idx, x in enumerate(tqdm(Xavail, desc=f"[Select Node] Sampling")):
        # print(f'{idx}/{len(Xavail)} - ', 'Sampling on ', x)
        samples, y_init = sampling(dataset, x, observed, pool, N=N, rng=rng, mode="gibbs", verbose=verbose)
       
        Z = _formulate_matrix(samples, nodes, map_label={'A': 1, 'B': -1})
        Sigma, mu = _covariance(Z)
        # Sigma, mu, rep = _covariance_spd(Z, unbiased=True, ridge=ridge_eps)
        # print(rep)          
        print("validate_sigma:", validate_sigma(Sigma))
        Sigmas[x] = Sigma
        Y_init[x] = y_init
 
    print('Sigma.shape', Sigmas[Xavail[0]].shape) # (V, V)

    # Greedy selection
    sigmas_list = [Sigmas[x] for x in Xavail]   # convert dict -> ordered list
    A, migs, evals = greedy_select_node_mig_multi_exact(sigmas_list, k_nodes, ridge=ridge_eps, MIG=MIG, verbose=True)
    V_sel = [nodes[int(i)] for i in A]

    return V_sel, Y_init
