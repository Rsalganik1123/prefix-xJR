import cvxpy as cp
import numpy as np
import pandas as pd 
import ipdb 
import pickle 
import math
from typing import Dict, List, Optional, Tuple

from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.io import load_consensus_ranking, load_sampled_preferences
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee



def ilp_prefix_jr(borda_ranking, approvals_by_k, n_voters, *, debug_sizes=False):
    n = len(borda_ranking)

    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p] = 1 if cand c at position p
    y = cp.Variable((n, n), boolean=True)   # y[c,d] = 1 if c ranked above d
    pos = cp.Variable(n, integer=True)      # pos[c] = position index of candidate c

    constraints = []

    # Permutation constraints on x
    constraints += [cp.sum(x, axis=1) == 1]   # each candidate assigned once
    constraints += [cp.sum(x, axis=0) == 1]   # each position filled once

    # Define pos[c] = sum_p p * x[c,p]
    p_idx = np.arange(n)
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(p_idx, x[c, :])))

    # IMPORTANT: explicit bounds (helps model build a lot)
    constraints += [pos >= 0, pos <= n - 1]

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M
    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # Prefix-JR variables
    z = cp.Variable((n_voters, n), boolean=True)

    for k in range(1, n + 1):
        quota = math.ceil(n_voters / k)
        approvals_k = approvals_by_k[k]

        # z link: represented if any approved candidate appears in top-k
        for v in range(n_voters):
            A_vk = approvals_k.get(v, [])
            if not A_vk:
                constraints.append(z[v, k - 1] == 0)
                continue

            # FAST version: no Python list of scalar atoms
            sum_in_topk = cp.sum(x[A_vk, :k])

            # This direction matches your previous logic:
            constraints.append(z[v, k - 1] <= sum_in_topk)

            # OPTIONAL (recommended): enforce "if sum_in_topk >= 1 then z=1"
            # comment out if you want the weakest constraints possible
            constraints.append(sum_in_topk <= k * z[v, k - 1]) #len(A_vk)

        # JR constraints per candidate
        for c in range(n):
            # build Vc without scanning all voters expensively if you want,
            # but for n=30 this is fine.
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                continue
            constraints.append(cp.sum(1 - z[Vc, k - 1]) <= quota - 1)

    # Objective: minimize pairwise disagreements with Borda
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])
    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)

    # Optional: print canonicalized sizes (true “problem size”)
    if debug_sizes:
        data = problem.get_problem_data(cp.GUROBI)[0]
        print("\n=== Canonicalized GUROBI data sizes ===")
        for name, obj in data.items():
            if hasattr(obj, "shape"):
                nnz = getattr(obj, "nnz", "NA")
                print(f"{name:>12}: shape={obj.shape}, nnz={nnz}, type={type(obj)}")
        print("======================================\n")

    # Solve with robust failure handling
    try:
        problem.solve(
            solver=cp.GUROBI,
            verbose=False,
            Threads=1,
            Presolve=0,
            # If you ever see INF_OR_UNBD later, these help:
            # reoptimize=True,
            # DualReductions=0,
        )
    except cp.error.SolverError as e:
        # try a fallback solver so your pipeline keeps running
        try:
            problem.solve(solver=cp.SCIP, verbose=False)
        except Exception:
            return None, None, f"GUROBI failed during build/solve; SCIP fallback failed too: {e}"

    if problem.status not in ("optimal", "optimal_inaccurate"):
        # Another fallback
        try:
            problem.solve(solver=cp.CBC, verbose=False)
        except Exception as e:
            return None, None, f"Non-optimal status {problem.status}; CBC failed: {e}"

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None, None, f"Optimization failed with status: {problem.status}"

    # Decode ranking from x
    x_sol = x.value
    if x_sol is None:
        return None, None, "Solved but x.value is None (unexpected)."

    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

    return ranking, float(problem.value)

def ilp_prefix_jr_plus_fair(borda_ranking, approvals_by_k, n_voters, 
    alphas, betas, k_fair, attribute_dict, num_attributes,
    threads=1, time_limit=300, mip_gap=None, debug_sizes=False):
    n = len(borda_ranking)

    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p] = 1 if cand c at position p
    y = cp.Variable((n, n), boolean=True)   # y[c,d] = 1 if c ranked above d
    pos = cp.Variable(n, integer=True)      # pos[c] = position index of candidate c

    constraints = []

    # Permutation constraints on x
    constraints += [cp.sum(x, axis=1) == 1]   # each candidate assigned once
    constraints += [cp.sum(x, axis=0) == 1]   # each position filled once

    # Define pos[c] = sum_p p * x[c,p]
    p_idx = np.arange(n)
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(p_idx, x[c, :])))

    # IMPORTANT: explicit bounds (helps model build a lot)
    constraints += [pos >= 0, pos <= n - 1]

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M
    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # Prefix-JR variables
    z = cp.Variable((n_voters, n), boolean=True)

    for k in range(1, n + 1):
        quota = math.ceil(n_voters / k)
        approvals_k = approvals_by_k[k]

        # z link: represented if any approved candidate appears in top-k
        for v in range(n_voters):
            A_vk = approvals_k.get(v, [])
            if not A_vk:
                constraints.append(z[v, k - 1] == 0)
                continue

            # FAST version: no Python list of scalar atoms
            sum_in_topk = cp.sum(x[A_vk, :k])

            # This direction matches your previous logic:
            constraints.append(z[v, k - 1] <= sum_in_topk)

            # OPTIONAL (recommended): enforce "if sum_in_topk >= 1 then z=1"
            # comment out if you want the weakest constraints possible
            constraints.append(sum_in_topk <= k * z[v, k - 1]) #len(A_vk)

        # JR constraints per candidate
        for c in range(n):
            # build Vc without scanning all voters expensively if you want,
            # but for n=30 this is fine.
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                continue
            constraints.append(cp.sum(1 - z[Vc, k - 1]) <= quota - 1)

    # ---------- FAIRNESS QUOTAS ON TOP-k ----------
    # candidate_in_topk[c] = 1 if c is placed in positions 0..k_fair-1
    # candidate_in_topk = cp.sum(x[:, :k_fair], axis=1)  # shape (n,)
    
    # Y = cp.Variable(n, boolean = True)
    # for attribute in range(num_attributes):
    #     coeff = np.zeros(n)
    #     for i in range(n):
    #         if attribute_dict[i] == attribute:
    #             coeff[i] = 1
    #     lb = math.floor(alphas[attribute] * k_fair)
    #     ub = math.ceil(betas[attribute] * k_fair)
    #     constraints.extend([coeff @ Y >= lb])
    #     constraints.extend([coeff @ Y <= ub]) 
    
    
    candidate_in_topk = cp.sum(x[:, :k_fair], axis=1)  # 0/1 because of permutation constraints

    for attribute in range(num_attributes):
        coeff = np.zeros(n)
        for i in range(n):
            if attribute_dict[i] == attribute:
                coeff[i] = 1

        lb = math.floor(alphas[attribute] * k_fair)
        ub = math.ceil(betas[attribute] * k_fair)

        constraints.append(coeff @ candidate_in_topk >= lb)
        constraints.append(coeff @ candidate_in_topk <= ub)



    # Objective: minimize pairwise disagreements with Borda
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])
    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)

    # Optional: print canonicalized sizes (true “problem size”)
    if debug_sizes:
        data = problem.get_problem_data(cp.GUROBI)[0]
        print("\n=== Canonicalized GUROBI data sizes ===")
        for name, obj in data.items():
            if hasattr(obj, "shape"):
                nnz = getattr(obj, "nnz", "NA")
                print(f"{name:>12}: shape={obj.shape}, nnz={nnz}, type={type(obj)}")
        print("======================================\n")

    # Solve with robust failure handling
    try:
        problem.solve(
            solver=cp.GUROBI,
            verbose=False,
            Threads=1,
            Presolve=0,
            # If you ever see INF_OR_UNBD later, these help:
            # reoptimize=True,
            # DualReductions=0,
        )
    except cp.error.SolverError as e:
        # try a fallback solver so your pipeline keeps running
        try:
            problem.solve(solver=cp.SCIP, verbose=False)
        except Exception:
            return None, None, f"GUROBI failed during build/solve; SCIP fallback failed too: {e}"

    if problem.status not in ("optimal", "optimal_inaccurate"):
        # Another fallback
        try:
            problem.solve(solver=cp.CBC, verbose=False)
        except Exception as e:
            return None, None, f"Non-optimal status {problem.status}; CBC failed: {e}"

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None, None, f"Optimization failed with status: {problem.status}"

    # Decode ranking from x
    x_sol = x.value
    if x_sol is None:
        return None, None, "Solved but x.value is None (unexpected)."

    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

    return ranking, float(problem.value)

def _get_cohesive_grps_(rankings, all_items, user_key): 
    n = len(rankings[user_key].unique())
    k = len(rankings['Ranked_Items'].unique())
    cohesive_grps = []  
    for c in all_items:
        counter = 0 
        approving_voters = rankings[rankings['Ranked_Items']
                                         == c][user_key].unique().tolist()
        if len(approving_voters) >= n/k: 
            cohesive_grps.append((approving_voters, c))     
    return cohesive_grps

def prefix_JR_greedy(rankings, all_items, user_key):
    ranking = []  
    satisfied_voters = set()
    for prefix_idx in range(len(all_items)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            rankings
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )
        cohesive_grps = _get_cohesive_grps_(preferences_at_prefix, all_items, user_key) 
        while cohesive_grps: 
            for (voters, c) in cohesive_grps: 
                if len(np.intersect1d(list(satisfied_voters), voters)) > 0: 
                    cohesive_grps.remove((voters, c))
                    continue 
                ranking.append(c)
                cohesive_grps.remove((voters, c))
                satisfied_voters.update(voters)
    if len(ranking) < len(all_items):
        for c in all_items:
            if c not in ranking:
                ranking.append(c)
    return ranking 

    
