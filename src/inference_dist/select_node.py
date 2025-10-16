from __future__ import annotations
from typing import Dict, Hashable, Union, List, Mapping, Optional, Callable, Sequence
import numpy as np
from numpy.linalg import LinAlgError
from inference_dist.sampling import sampling
from multiprocessing import Pool
from tqdm import tqdm
import random
import torch
from inference_dist.utils import build_group_candidates, allocate_k_per_group


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
    clip_var=1e-12,           # guard for logs / inversion
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
    migs_count = sum(1 for x in migs if x > 1e-12)

    return V_sel, migs_count



from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import math

def _one_run_consistency(
    samples: Dict[str, Any],
    neighbor_dict: Dict[str, List[str]],
    default_if_no_neighbors: Optional[float] = None
) -> Dict[str, dict]:
    out = {}
    for cid, ans in samples.items():
        nbrs = neighbor_dict.get(cid, [])
        agree = 0
        total = 0
        # collect neighbor answers for majority calc
        nbr_ans = []
        for n in nbrs:
            if n in samples:
                total += 1
                nbr_ans.append(samples[n])
                if samples[n] == ans:
                    agree += 1
        consistency = (agree / total) if total > 0 else default_if_no_neighbors

        # neighbor majority (ties -> None)
        maj = None
        if total > 0:
            c = Counter(nbr_ans)
            most_common = c.most_common()
            if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
                maj = most_common[0][0]

        out[cid] = {
            'answer': ans,
            'agree': agree,
            'total': total,
            'consistency': consistency,
            'neighbor_majority': maj,
            'match_neighbor_majority': (maj is not None and ans == maj)
        }
    return out

import math
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

def neighbor_consistency_out_multi(
    samples_runs: List[Dict[str, Any]],
    in_neighbors: Dict[str, List[str]],          # 仍然传入“入邻居字典” v->u 是通过反推得到的
    default_if_no_out_neighbors: Optional[float] = None,
    include_all_nodes: bool = False,
    exclude_self_from_majority: bool = False,    # 计算 u 的多数派时，是否把 v 本人从 u 的入邻居票中排除
) -> Dict[str, dict]:
    """
    针对“出邻居”视角：对每个节点 v，考察所有把 v 当入邻居的节点 u（即 v 的出邻居），
    计算在多次 runs 中的 (1) 简单一致率：ans(v) 与 ans(u) 是否相同；(2) 多数派匹配：ans(v) 是否等于 u 的入邻居多数派。

    Args:
        samples_runs: [run0, run1, ...]，每个 run: {caseid: answer}
        in_neighbors: 入邻居 {u: [v1, v2, ...]}，表示 v->u
        default_if_no_out_neighbors: 当某 run 中 v 的出邻居总数为 0 时的一致性取值（None 则不计入均值）
        include_all_nodes: 是否把仅出现在图结构、但从未在任何 run 出现过的节点纳入输出
        exclude_self_from_majority: 计算 u 的多数派时是否把 v 从 u 的入邻居票中去掉（避免“自带一票”影响）

    Returns:
        {
          v: {
            'per_run': [
              {
                'answer_v': ...,
                'agree_simple': int,            # 与出邻居 u 的答案相同的个数
                'total_simple': int,            # 有答案的出邻居数量
                'consistency_out': float|None,  # agree_simple / total_simple 或 default_if_no_out_neighbors
                'majority_match_count': int,    # v 的答案等于 u 的邻居多数派 的个数
                'majority_defined_total': int,  # 有定义多数派的 u 的个数（平票/无票的不计）
                'majority_match_rate': float|None,  # 上面两者之比（若分母为0则 None）
              }, ...
            ],
            'agg': {
              'runs_present': int,
              'runs_with_out_neighbors': int,
              'mean_consistency_out': float|None,
              'std_consistency_out': float|None,
              'min_consistency_out': float|None,
              'max_consistency_out': float|None,
              'majority_match_rate_mean': float|None,  # 跨 runs 的平均多数派匹配率
              'answer_values': list,
              'answer_unique_count': int
            }
          }
        }
    """

    # —— 0) 从入邻居反推出邻居映射 O(v) = {u | v in in_neighbors[u]}
    out_neighbors = defaultdict(list)
    nodes_all = set(in_neighbors.keys())
    for u, ins in in_neighbors.items():
        nodes_all.update(ins)
        for v in ins:
            out_neighbors[v].append(u)
    for n in nodes_all:
        out_neighbors.setdefault(n, [])  # 保证全覆盖

    # —— 1) 全部出现过的节点集合
    all_case_ids = set()
    for run in samples_runs:
        all_case_ids.update(run.keys())
    if include_all_nodes:
        all_case_ids.update(nodes_all)

    # —— 2) 工具：u 的入邻居多数派（按某个 run 的可见答案）
    def neighbor_majority_for_u(u: str, run_samples: Dict[str, Any], exclude_v: Optional[str] = None):
        ins = in_neighbors.get(u, [])
        if exclude_v is not None:
            ins = [w for w in ins if w != exclude_v]
        votes = [run_samples[w] for w in ins if w in run_samples]
        if not votes:
            return None
        cnt = Counter(votes).most_common()
        if len(cnt) >= 2 and cnt[0][1] == cnt[1][1]:
            return None  # 平票：无定义
        return cnt[0][0]

    # —— 3) 统计聚合
    out: Dict[str, dict] = {}
    for v in all_case_ids:
        per_run_list = []
        answers_seen = []

        runs_present = 0
        runs_with_out_neighbors = 0

        cons_vals_defined = []    # 跨 runs 的 consistency_out
        maj_rates_defined = []    # 跨 runs 的 majority_match_rate

        for run_samples in samples_runs:
            if v not in run_samples:
                continue  # 该 run v 没有答案

            ans_v = run_samples[v]
            runs_present += 1

            # 出邻居集合（谁把 v 当入邻居）
            outs = out_neighbors.get(v, [])

            # —— 简单一致率：ans(u) 与 ans(v) 是否相同
            u_with_ans = [u for u in outs if u in run_samples]
            total_simple = len(u_with_ans)
            if total_simple == 0:
                consistency_out = default_if_no_out_neighbors
                agree_simple = 0
            else:
                agree_simple = sum(1 for u in u_with_ans if run_samples[u] == ans_v)
                consistency_out = agree_simple / total_simple

            # —— 多数派匹配：ans(v) 是否等于每个 u 的“入邻居多数派”
            majority_defined_total = 0
            majority_match_count = 0
            for u in outs:
                maj_u = neighbor_majority_for_u(
                    u, run_samples,
                    exclude_v=v if exclude_self_from_majority else None
                )
                if maj_u is None:
                    continue
                majority_defined_total += 1
                if ans_v == maj_u:
                    majority_match_count += 1
            majority_match_rate = (
                (majority_match_count / majority_defined_total)
                if majority_defined_total > 0 else None
            )

            # 记录 per_run
            per_run_list.append({
                'answer_v': ans_v,
                'agree_simple': agree_simple,
                'total_simple': total_simple,
                'consistency_out': consistency_out,
                'majority_match_count': majority_match_count,
                'majority_defined_total': majority_defined_total,
                'majority_match_rate': majority_match_rate,
            })
            answers_seen.append(ans_v)

            # 汇总口径
            if total_simple > 0 and consistency_out is not None:
                runs_with_out_neighbors += 1
                cons_vals_defined.append(consistency_out)
            if majority_match_rate is not None:
                maj_rates_defined.append(majority_match_rate)

        # 统计函数
        def _mean(xs):
            return sum(xs) / len(xs) if xs else None

        def _std(xs, mean=None):
            if not xs:
                return None
            m = mean if mean is not None else _mean(xs)
            if len(xs) < 2:
                return 0.0
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

        mean_c = _mean(cons_vals_defined)
        std_c  = _std(cons_vals_defined, mean_c)
        min_c  = min(cons_vals_defined) if cons_vals_defined else None
        max_c  = max(cons_vals_defined) if cons_vals_defined else None
        maj_rate_mean = _mean(maj_rates_defined)

        out[v] = {
            'per_run': per_run_list,
            'agg': {
                'runs_present': runs_present,
                'runs_with_out_neighbors': runs_with_out_neighbors,
                'mean_consistency_out': mean_c,
                'std_consistency_out': std_c,
                'min_consistency_out': min_c,
                'max_consistency_out': max_c,
                'majority_match_rate_mean': maj_rate_mean,
                'answer_values': answers_seen,
                'answer_unique_count': len(set(answers_seen)),
            }
        }

    return out


def neighbor_consistency_multi(
    samples_runs: List[Dict[str, Any]],
    neighbor_dict: Dict[str, List[str]],
    default_if_no_neighbors: Optional[float] = None
) -> Dict[str, dict]:
    """
    Args:
        samples_runs: [run0_samples, run1_samples, ...], each like {'caseid': answer}
        neighbor_dict: {'caseid': [neighbor_id, ...]}
        default_if_no_neighbors: used for per-run consistency when total=0

    Returns (per caseid):
        {
          'per_run': [{'consistency':..., 'agree':..., 'total':..., 'match_neighbor_majority':..., 'answer':...}, ...],
          'agg': {
              'runs_present': int,
              'runs_with_neighbors': int,
              'mean_consistency': float|None,
              'std_consistency': float|None,
              'min_consistency': float|None,
              'max_consistency': float|None,
              'neighbor_majority_agree_rate': float|None,  # over runs with a defined majority
              'answer_values': [list of answers seen],
              'answer_unique_count': int
          }
        }
    """
    per_run_all: List[Dict[str, dict]] = []
    all_case_ids = set()
    for run_samples in samples_runs:
        res = _one_run_consistency(run_samples, neighbor_dict, default_if_no_neighbors)
        per_run_all.append(res)
        all_case_ids.update(run_samples.keys())

    out: Dict[str, dict] = {}
    for cid in all_case_ids:
        per_run_list = []
        cons_vals = []
        cons_vals_defined = []
        maj_agree_flags = []
        answers_seen = []

        runs_present = 0
        runs_with_neighbors = 0

        for res in per_run_all:
            if cid not in res:
                continue
                # skip runs where cid has no label (not present)

            runs_present += 1
            r = res[cid]
            per_run_list.append({
                'answer': r['answer'],
                'agree': r['agree'],
                'total': r['total'],
                'consistency': r['consistency'],
                'match_neighbor_majority': r['match_neighbor_majority'],
                'neighbor_majority': r['neighbor_majority'],
            })
            answers_seen.append(r['answer'])

            # consistency stats
            cons_vals.append(r['consistency'])
            if r['total'] > 0:
                runs_with_neighbors += 1
                if r['consistency'] is not None:
                    cons_vals_defined.append(r['consistency'])

            # majority-agree stats (only when majority is defined)
            if r['neighbor_majority'] is not None:
                maj_agree_flags.append(bool(r['match_neighbor_majority']))

        # aggregate
        def _mean(xs):
            return sum(xs) / len(xs) if xs else None

        def _std(xs, mean=None):
            if not xs:
                return None
            m = mean if mean is not None else _mean(xs)
            if len(xs) < 2:
                return 0.0
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

        mean_c = _mean(cons_vals_defined)
        std_c  = _std(cons_vals_defined, mean_c)
        min_c  = min(cons_vals_defined) if cons_vals_defined else None
        max_c  = max(cons_vals_defined) if cons_vals_defined else None
        maj_rate = _mean([1.0 if f else 0.0 for f in maj_agree_flags]) if maj_agree_flags else None

        out[cid] = {
            'per_run': per_run_list,
            'agg': {
                'runs_present': runs_present,
                'runs_with_neighbors': runs_with_neighbors,
                'mean_consistency': mean_c,
                'std_consistency': std_c,
                'min_consistency': min_c,
                'max_consistency': max_c,
                'neighbor_majority_agree_rate': maj_rate,
                'answer_values': answers_seen,
                'answer_unique_count': len(set(answers_seen))
            }
        }
    return out

from typing import Dict, List, Tuple, Any

def select_top_k_influential(
    res_multi: Dict[str, Dict[str, Any]],
    k: int,
    method: str = "agree_sum",   # "agree_sum" | "hybrid" | "consistency_mean"
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),  # used for "hybrid": (agree_sum, mean_consistency, maj_agree_rate)
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Rank caseids by an 'influence' score built from res_multi (output of neighbor_consistency_multi).

    Returns: list of (caseid, metrics) sorted descending by score, top-K only.
             metrics includes: score, agree_sum, total_sum, mean_consistency, maj_agree_rate, runs_present, runs_with_neighbors
    """
    scored: List[Tuple[str, Dict[str, float]]] = []
    w_agree, w_mean_c, w_maj = weights

    for cid, data in res_multi.items():
        per_run = data.get('per_run', [])
        agg    = data.get('agg', {})

        agree_sum   = float(sum((r.get('agree') or 0) for r in per_run))
        total_sum   = float(sum((r.get('total') or 0) for r in per_run))
        mean_c      = float(agg.get('mean_consistency')) if agg.get('mean_consistency') is not None else 0.0
        maj_rate    = float(agg.get('neighbor_majority_agree_rate')) if agg.get('neighbor_majority_agree_rate') is not None else 0.0
        runs_present = int(agg.get('runs_present', 0))
        runs_with_neighbors = int(agg.get('runs_with_neighbors', 0))

        # Scoring options
        if method == "agree_sum":
            score = agree_sum
        elif method == "consistency_mean":
            score = mean_c
        elif method == "hybrid":
            # Reward broad agreement coverage and reliability
            # Multiply consistency and majority rate by runs_with_neighbors to prefer stability across more contexts.
            score = (w_agree * agree_sum
                     + w_mean_c * (mean_c * runs_with_neighbors)
                     + w_maj * (maj_rate * runs_with_neighbors))
        else:
            raise ValueError(f"Unknown method: {method}")

        scored.append((
            cid,
            {
                "score": score,
                "agree_sum": agree_sum,
                "total_sum": total_sum,
                "mean_consistency": mean_c,
                "maj_agree_rate": maj_rate,
                "runs_present": runs_present,
                "runs_with_neighbors": runs_with_neighbors,
            }
        ))

    # Sort by score desc; tie-break by mean_consistency, then runs_with_neighbors, then agree_sum
    scored.sort(key=lambda x: (
        x[1]["score"],
        x[1]["mean_consistency"],
        x[1]["runs_with_neighbors"],
        x[1]["agree_sum"]
    ), reverse=True)

    return scored[:k]


def top_k_per_community(
    res_multi: Dict[str, Dict[str, Any]],
    labels: Dict[str, Any],                 # caseid -> community_id
    k_each: int,
    method: str = "consistency_mean",
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> Dict[Any, List[Tuple[str, Dict[str, float]]]]:
    """
    Returns: {community_id: [(caseid, metrics_dict), ... up to k_each]}
    """
    # group caseids by community (only those present in res_multi)
    comm2nodes = defaultdict(list)
    for cid, comm in labels.items():
        if cid in res_multi:
            comm2nodes[comm].append(cid)

    out = {}
    for comm, nodes in comm2nodes.items():
        # restrict res_multi to this community
        sub = {cid: res_multi[cid] for cid in nodes}
        # rank within community
        ranked = select_top_k_influential(sub, k=min(k_each, len(sub)),
                                          method=method, weights=weights)
        out[comm] = ranked
    return out

def top_k_ids_per_community(
    res_multi: Dict[str, Dict[str, Any]],
    labels: Dict[str, Any],
    k_each: int,
    method: str = "consistency_mean",
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> Dict[Any, List[str]]:
    """Same as above but returns only node IDs per community."""
    ranked_map = top_k_per_community(res_multi, labels, k_each, method, weights)
    return {comm: [cid for cid, _ in ranked] for comm, ranked in ranked_map.items()}

from collections import defaultdict
from typing import Dict, Any, List, Tuple
import math

def allocate_k_per_community_by_size(
    labels: Dict[str, Any],            # caseid -> community_id
    res_multi: Dict[str, Dict[str, Any]],
    k_total: int,
    min_per_comm: int = 0,
) -> Dict[Any, int]:
    """
    Proportional allocation by community size with largest-remainder.
    Caps by available candidates per community (present in res_multi) and redistributes leftovers.
    Returns: {community_id: k_allocated}
    """
    # candidates available per community (only caseids that exist in res_multi)
    comm2nodes = defaultdict(list)
    for cid, comm in labels.items():
        if cid in res_multi:
            comm2nodes[comm].append(cid)

    # if nothing to allocate
    if k_total <= 0 or not comm2nodes:
        return {}

    # total available candidates
    total_avail = sum(len(v) for v in comm2nodes.values())
    k_total = min(k_total, total_avail)  # cannot select more than available

    # ideal proportional quotas
    sizes = {c: len(v) for c, v in comm2nodes.items()}
    N = sum(sizes.values())
    ideals = {c: (sizes[c] / N) * k_total for c in sizes}

    # base = floor(ideal) but also respect min_per_comm
    base = {c: max(min_per_comm, int(math.floor(ideals[c]))) for c in sizes}

    # cap by availability immediately
    for c in base:
        base[c] = min(base[c], sizes[c])

    # compute remainder to distribute
    allocated = sum(base.values())
    remainder = k_total - allocated

    if remainder > 0:
        # prepare fractional parts, but skip communities already full
        fracs = [(ideals[c] - math.floor(ideals[c]), sizes[c], c)
                 for c in sizes if base[c] < sizes[c]]
        # sort by fractional part desc, then by larger community size
        fracs.sort(key=lambda x: (x[0], x[1]), reverse=True)

        i = 0
        # largest-remainder distribution with availability cap
        while remainder > 0 and i < len(fracs):
            _, _, c = fracs[i]
            if base[c] < sizes[c]:
                base[c] += 1
                remainder -= 1
            i += 1
        # If still remainder (e.g., many small/full communities), round-robin any with spare
        if remainder > 0:
            spare = [c for c in sizes if base[c] < sizes[c]]
            j = 0
            while remainder > 0 and spare:
                c = spare[j % len(spare)]
                if base[c] < sizes[c]:
                    base[c] += 1
                    remainder -= 1
                j += 1

    return base  # k_each per community


def select_top_ids_proportional(
    res_multi: Dict[str, Dict[str, Any]],
    labels: Dict[str, Any],                # caseid -> community_id
    k_total: int,
    method: str = "consistency_mean",
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> List[str]:
    """
    Proportionally allocate k_total across communities by size,
    rank within each community using select_top_k_influential, and return one flattened list.
    """
    # Build community -> candidate list (present in res_multi)
    comm2nodes = defaultdict(list)
    for cid, comm in labels.items():
        if cid in res_multi:
            comm2nodes[comm].append(cid)

    # quotas per community
    k_per_comm = allocate_k_per_community_by_size(labels, res_multi, k_total)

    selected = []
    for comm, nodes in comm2nodes.items():
        if k_per_comm.get(comm, 0) <= 0:
            continue
        sub = {cid: res_multi[cid] for cid in nodes}
        ranked = select_top_k_influential(sub, k=k_per_comm[comm], method=method, weights=weights)
        selected.extend([cid for cid, _ in ranked])

    # just in case quotas + availability didn’t reach k_total, backfill globally
    if len(selected) < k_total:
        ranked_global = select_top_k_influential(res_multi, k=len(res_multi), method=method, weights=weights)
        have = set(selected)
        for cid, _ in ranked_global:
            if len(selected) >= k_total:
                break
            if cid not in have:
                selected.append(cid)
                have.add(cid)

    return selected[:k_total]


def select_nodes_occurence(
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


    for idx, x in enumerate(tqdm(Xavail, desc=f"[Select Node] Sampling")):
        # print(f'{idx}/{len(Xavail)} - ', 'Sampling on ', x)
        samples, y_init = sampling(dataset, x, observed, pool, N=N, rng=rng, mode="icm", verbose=verbose)
        res_multi = neighbor_consistency_out_multi(samples_runs=[samples[-1]], in_neighbors=dataset.graph.neighbor)

        labels = dataset.graph.label  # caseid -> community
        all_selected = select_top_ids_proportional(
            res_multi=res_multi,
            labels=labels,
            k_total=k_nodes,
            method="agree_sum"   # or "agree_sum" / "hybrid"
            )
    migs_count = 0
    print('all_selected:', all_selected)
    return all_selected, migs_count

def select_nodes_per_community(
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

    node_label_dict = dataset.graph.label  # you said: dataset.node.label[case_id] is the group
    groups = build_group_candidates(nodes, node_label_dict)
    k_by_group = allocate_k_per_group(groups, k_total=k_nodes)

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

    # helper: run greedy on a subset using precomputed Sigmas
    def greedy_on_subset(subset_nodes, k):
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        subset_idx = [node_to_idx[n] for n in subset_nodes if n in node_to_idx]
        if not subset_idx or k <= 0:
            return [], 0, {"A_global": [], "A_subset": [], "subset_idx": subset_idx}

        k_eff = min(k, len(subset_idx))
        sigmas_subset_list = [Sigmas[x][np.ix_(subset_idx, subset_idx)] for x in Xavail]
        A_subset, migs, evals = greedy_select_node_mig_multi_exact(
            sigmas_subset_list, k_eff, ridge=ridge_eps, MIG=MIG, verbose=verbose
        )
        A_global = [subset_idx[int(i)] for i in A_subset]
        V_sel = [nodes[i] for i in A_global]
        migs_count = sum(1 for x in migs if x > 1e-12)
        return V_sel, migs_count, {"A_global": A_global, "A_subset": A_subset, "subset_idx": subset_idx}

    per_group = {}
    all_selected = []
    for g, cand_nodes in groups.items():
        k_g = k_by_group[g]
        V_sel, migs_count, meta = greedy_on_subset(cand_nodes, k_g)
        per_group[g] = {"V_sel": V_sel, "migs_count": migs_count, "meta": meta, "group_size": len(cand_nodes), "k": k_g}
        all_selected.extend(V_sel)
    
    migs_count = 0
    for g in per_group:
        print(per_group[g])
        migs_count += per_group[g]["migs_count"]

    return all_selected, migs_count


