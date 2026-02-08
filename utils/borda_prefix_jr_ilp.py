import cvxpy as cp
import numpy as np
import pandas as pd 
import ipdb 
import pickle 
import cvxpy as cp
import numpy as np

from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.io import load_consensus_ranking, load_sampled_preferences
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee


import cvxpy as cp
import numpy as np
import math
import math
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

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

def prefix_JR_joe(rankings, all_items, user_key):
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

# usage:
# get all cohesive groups 
# run ilp, include it in the evaluation pipeline 
# -> sanity check: must satisfy JR always. otherwise something is very wrong

def test3(): 
    
    preferences = pickle.load(open("/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl", 'rb'))
    all_candidates = preferences['Ranked_Items'].explode().unique().tolist()
    
    ranking = prefix_JR_joe(preferences, all_candidates, 'User_ID')
    print("Prefix-JR ranking (best to worst):")
    print(f"  {ranking}")
    
    for prefix_idx in range(len(ranking)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            preferences
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )
        flag = JR_check_satisfaction_given_committee(ranking[:k], preferences_at_prefix, all_candidates, n=len(preferences), k=k, user_key='User_ID')
        print(k, flag)
    
# test3() 

def test(): 
    num_candidates = 6
    num_voters = 6
    preferences = pd.DataFrame({
        'User_ID': [0, 1, 2, 3, 4, 5],
        'Ranked_Items': [[2,0,1,5,3,4], [2,0,1,5,3,4], [3,0,1,5,2,4], [3,0,1,5,2,4], [4,0,1,5,3,2], [4,0,1,5,3,2]]
    })
    all_candidates = preferences['Ranked_Items'].explode().unique().tolist()
    
    # borda_ranking = [5,1, 3, 2, 4,0] # SHOULD FAIL 
    borda_ranking = [0,1,2,3,4,5] # SHOULD PASS
    # borda_ranking = [2,3,4,0,1,5] # SHOULD PASS
    # borda_ranking = [1,5,4,3,2,0] # SHOULD FAIL
    print(preferences)
    alphas=[0.5,0.5]
    betas=[1.0, 1.0]
    item_attribute = {0:0, 1:1, 2:1, 3:1, 4:1, 5:0}
    # cohesive_groups = {}
    satisfaction = {}
    approvals_by_k = {}

    for prefix_idx in range(len(borda_ranking)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            preferences
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )

        approvals_k = {}
        for v in range(num_voters):
            approvals_k[v] = (
                preferences_at_prefix[preferences_at_prefix["User_ID"] == v]["Ranked_Items"]
                .unique().tolist()
            )
        approvals_by_k[k] = approvals_k
        
    ILP_ranking, ILP_obj_value = ilp_prefix_jr(borda_ranking, approvals_by_k, n_voters=num_voters)
    
    print("Borda ranking order (best to worst):")
    print(f"  {borda_ranking}")
    
    print(f"Alphas:{alphas}, Betas:{betas}, k:{3}, Attribute Map:{item_attribute}")

    print(f"\nPrefix-JR constrained ranking (best to worst):")
    print(f"  {ILP_ranking}")
    
    print(f"\nNumber of disagreements with Borda: {ILP_obj_value}")

    
    # FAIR_ranking, FAIR_obj_value = ilp_prefix_jr_plus_fair(borda_ranking, approvals_by_k, n_voters=num_voters, k=3, alphas=alphas, betas=betas, attribute_dict=item_attribute, num_attributes=2)
    FAIR_ranking, FAIR_obj_value = ilp_prefix_jr_plus_fair(borda_ranking, approvals_by_k, n_voters=num_voters, alphas=alphas, betas=betas, k_fair = 10, attribute_dict=item_attribute, num_attributes=len(set(item_attribute.values())))
    
    
    print(f"\nPrefix-JR_FAIR constrained ranking (best to worst):")
    print(f"  {FAIR_ranking}")
    
    print(f"\nNumber of disagreements with Borda: {FAIR_obj_value}")
    
    
def test2(): 
    preferences = pickle.load(open("/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl", 'rb'))
    all_candidates = preferences['Ranked_Items'].explode().unique().tolist()
    fair_info = pickle.load(open("/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/fair_ranking_process.pkl", 'rb')) 
    alphas = fair_info['alphas']
    betas = fair_info['betas']
    item_attribute = fair_info['attributes_map']
    
    # ipdb.set_trace() 
    
    borda_ranking = load_consensus_ranking("/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/BordaCount.txt")
    item_to_idx = dict(zip(all_candidates, range(len(all_candidates))))
    user_to_idx = dict(zip(preferences['User_ID'].unique(), range(len(preferences['User_ID'].unique()))))
    borda_ranking = [item_to_idx[item] for item in borda_ranking]
    
    # ipdb.set_trace() 
    
    satisfaction = {}
    approvals_by_k = {}

    for prefix_idx in range(len(borda_ranking)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            preferences
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )
        preferences_at_prefix['Ranked_Items'] = preferences_at_prefix['Ranked_Items'].map(item_to_idx)
        preferences_at_prefix['User_ID'] = preferences_at_prefix['User_ID'].map(user_to_idx)
        num_voters = len(preferences_at_prefix['User_ID'].unique())
        approvals_k = {}
        for v in range(num_voters):
            approvals_k[v] = (
                preferences_at_prefix[preferences_at_prefix["User_ID"] == v]["Ranked_Items"]
                .unique().tolist()
            )
        approvals_by_k[k] = approvals_k
    
    
    ILP_ranking, ILP_obj_value = ilp_prefix_jr(borda_ranking, approvals_by_k, n_voters=num_voters)
    print(f"\nPrefix-JR constrained ranking (best to worst):")
    print(f"  {ILP_ranking}")

    FAIR_ranking, FAIR_obj_value = ilp_prefix_jr_plus_fair(borda_ranking, approvals_by_k, n_voters=num_voters, alphas=alphas, betas=betas, k_fair = 10, attribute_dict=item_attribute, num_attributes=len(set(item_attribute.values())))
    print(f"\nPrefix-JR_FAIR constrained ranking (best to worst):")
    print(f"  {FAIR_ranking}")
    
    print(f"\nNumber of disagreements with Borda: {FAIR_obj_value}")
    # ipdb.set_trace()
    
    #Check prefix-JR satisfaction
    for prefix_idx in range(len(FAIR_ranking)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            preferences
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )
        preferences_at_prefix['Ranked_Items'] = preferences_at_prefix['Ranked_Items'].map(item_to_idx)
        preferences_at_prefix['User_ID'] = preferences_at_prefix['User_ID'].map(user_to_idx)
        
        # flag1 = JR_check_satisfaction_given_committee(ILP_ranking[:k], preferences_at_prefix, all_candidates, n=num_voters, k=k, user_key='User_ID')
        # print(f"ILP ---WE ARE SATISFYING:{flag1}@{k}")
        
        flag2 = JR_check_satisfaction_given_committee(FAIR_ranking[:k], preferences_at_prefix, all_candidates, n=num_voters, k=k, user_key='User_ID')
        print(f"FAIR -- WE ARE SATISFYING:{flag2}@{k}")
    
    grp_cnt = dict.fromkeys(list(item_attribute.values()), 0)
    for idx in range(10):
        item = FAIR_ranking[idx]
        item_grp = item_attribute[item]
        grp_cnt[item_grp] += 1
    
    print(grp_cnt)
          
# test2() 
    
 
 
 
################################    ################################     ################################     ################################   
    
#OLD SHIT 


def ilp_old(borda_ranking, l_cohesive):
    # here i'm imagining l_cohesive is built iteratively based on level (k)
    # ipdb.set_trace() 
    n = len(borda_ranking) # assuming we have rankings of equal size (prefix-JR and Borda)
    # borda_dict = {}
    # for i in range(len(borda_ranking)): # makes things easier later
    #     borda_dict[i] = borda_ranking[i]
    borda_dict = {cand: pos for pos, cand in enumerate(borda_ranking)}

    
    x = cp.Variable((n,n), boolean=True) # x_{c,p} = 1 if candidate c is in position p, 0 otw
    y = cp.Variable((n,n), boolean=True) # y_{c,d} = 1 if candidate c is ranked above d in ranking, 0 otw

    constraints = []

    # variable constraints
    print("LINE:22", len(constraints))
    for c in range(n): #0
        constraints.append(cp.sum(x[c, :]) == 1) # 1 position per candidate
    print("LINE:25", len(constraints))
    for p in range(n): #6 
        constraints.append(cp.sum(x[:, p]) == 1) # 1 candidate per position

    print("LINE:29", len(constraints))
    # for c in range(n):  
    #     for d in range(n):
    #         if c != d:
    #             for p in range(n):
    #                 for q in range(p + 1, n):
    #                     constraints.append(y[c,d] >= x[c,p] + x[d, q] - 1) # enforces y_{c,d} to be correct
    
    print("LINE:36", len(constraints))
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c,d] + y[d,c] == 1)
    print("LINE:40", len(constraints))
    for c in range(n):
        constraints.append(y[c,c] == 0)
    
    # prefix-JR constraints:
    # ipdb.set_trace()
    # for k in range(1, n + 1):
    #     # cohesive_groups = l_cohesive[k]['voter_sets']
    #     # alts = l_cohesive[k]['candidate_sets']
    #     alts = l_cohesive[k] 
    #     # ipdb.set_trace()
    #     # cohesive_k = cohesive_groups
    #     alts_k = alts
    #     try: 
    #         constraints.append(cp.sum([x[c, p] for c in alts_k for p in range(k)]) >= 1)
    #     except Exception as e:
    #         ipdb.set_trace()
    #         print(e)
            

    objective_terms = []

    for c in range(n):
        for d in range(n):
            if c != d:
                if borda_dict[d] < borda_dict[c]:
                    objective_terms.append(y[c,d])
    
    objective = cp.Minimize(cp.sum(objective_terms))

    # solving ilp
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
   # output ranking
    x_sol = x.value
    ranking = [0] * n
    for c in range(n):
        for p in range(n):
            if x_sol[c, p] > 0.5:  
                ranking[p] = c
                break
    
    return ranking, problem.value


def ilp(borda_ranking, cohesive_groups):
    n = len(borda_ranking)

    # candidate -> Borda position (0 = best)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p]
    y = cp.Variable((n, n), boolean=True)   # y[c,d]
    pos = cp.Variable(n, integer=True)      # pos[c]

    constraints = []

    # Permutation constraints on x
    for c in range(n):
        constraints.append(cp.sum(x[c, :]) == 1)
    for p in range(n):
        constraints.append(cp.sum(x[:, p]) == 1)

    # Define pos[c] = sum_p p * x[c,p]
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(np.arange(n), x[c, :])))
        

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M
    # If y[c,d]=1 => pos[c] <= pos[d] - 1
    # If y[c,d]=0 => constraint relaxed
    M = n  # safe big-M
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            # Optional: also enforce the opposite direction when y[c,d]=0
            # i.e., if y[c,d]=0 then pos[c] >= pos[d] + 1
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # Prefix-JR constraints (per cohesive group)
    # cohesive_groups[k]['candidate_sets'] is a list of sets/lists of candidates
    for k in range(1, n + 1):
        cand_sets = cohesive_groups.get(k, {}).get('candidate_sets', [])
        # ipdb.set_trace()
        for S in cand_sets:
            constraints.append(
                cp.sum([x[c, p] for c in S for p in range(k)]) >= 1
            )

    # Objective: minimize pairwise disagreements with Borda
    # If Borda wants c above d (borda_pos[c] < borda_pos[d]),
    # then disagreement occurs when y[d,c] = 1.
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])  # Borda says c>d, but model says d>c
    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Decode ranking from x
    x_sol = x.value
    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

    return ranking, problem.value

import numpy as np



def check_any(name, obj):
    if obj is None:
        return
    arr = obj.toarray() if hasattr(obj, "toarray") else np.asarray(obj)
    if arr.size and not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise ValueError(f"{name} has non-finite values; example index {bad[0][0]}")




def ilp_prefix_jr_old(borda_ranking, approvals_by_k, n_voters):
    """
    borda_ranking: list[int] length n (candidate IDs 0..n-1)
    approvals_by_k: dict[int, dict[int, list[int]]]
        approvals_by_k[k][v] = list of candidates voter v approves at prefix k
        (in your setup, that's the voter's top-k ranked items)
    n_voters: int
    """
    n = len(borda_ranking)

    # candidate -> Borda position (0 = best)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p] = 1 if cand c at position p
    y = cp.Variable((n, n), boolean=True)   # y[c,d] = 1 if c ranked above d
    pos = cp.Variable(n, integer=True)      # pos[c] = position index of candidate c

    constraints = []

    # Permutation constraints on x
    for c in range(n):
        constraints.append(cp.sum(x[c, :]) == 1)
    for p in range(n):
        constraints.append(cp.sum(x[:, p]) == 1)

    # Define pos[c] = sum_p p * x[c,p]
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(np.arange(n), x[c, :])))

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M (tight, two-sided)
    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            # if y[c,d]=1 then pos[c] <= pos[d]-1
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            # if y[c,d]=0 then pos[c] >= pos[d]+1
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # ------------------------------------------------------------
    # Prefix-JR constraints: JR must hold for every prefix k
    # ------------------------------------------------------------
    # z[v,k] = 1 if voter v is represented in top-(k+1) (k is 0-indexed here)
    z = cp.Variable((n_voters, n), boolean=True)

    for k in range(1, n + 1):
        quota = math.ceil(n_voters / k)  # match JR definition cleanly
        topk_positions = range(k)
        approvals_k = approvals_by_k[k]  # dict voter -> list of approved cands at prefix k

        # Link z[v,k-1] to whether top-k contains ANY approved candidate for voter v
        for v in range(n_voters):
            A_vk = approvals_k.get(v, [])
            if len(A_vk) == 0:
                # If voter approves nothing at this prefix (shouldn't happen in your setup),
                # force them unrepresented.
                constraints.append(z[v, k - 1] == 0)
                continue

            # z[v,k-1] <= sum_{a in A_vk} sum_{p<k} x[a,p]
            constraints.append(
                z[v, k - 1] <= cp.sum([x[a, p] for a in A_vk for p in topk_positions])
            )

        # JR condition per your checker:
        # For each candidate c, among voters who approve c, you cannot have quota (or more)
        # who are unrepresented => sum_{v in Vc} (1 - z[v,k-1]) <= quota - 1
        for c in range(n):
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                # Even if all are unrepresented, can't reach quota -> no JR constraint needed
                continue

            constraints.append(
                cp.sum([1 - z[v, k - 1] for v in Vc]) <= quota - 1
            )

    # Objective: minimize pairwise disagreements with Borda
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])  # disagreement if model says d > c

    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)
    data = problem.get_problem_data(cp.GUROBI)[0]
    for k, v in data.items():
        # many entries are scalars/metadata; only check array-like
        try:
            check_any(k, v)
        except Exception:
            # ignore non-array entries
            print(f"Skipping check for {k}")
    
    # problem.solve(solver=cp.GUROBI, verbose=True)
    try:
        # problem.solve(
        #     solver=cp.GUROBI,
        #     verbose=True,
        #     DualReductions=0,
        #     QCPDual=0,
        # )
        
        problem.solve(
            solver=cp.GUROBI,
            verbose=False,
            Threads=1,
            Presolve=0,
            outputflag=0,
        )
    except Exception as e:
        print("CVXPY problem.status:", problem.status)
        print("CVXPY solver_stats:", problem.solver_stats)
        raise
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Decode ranking from x
    x_sol = x.value
    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

    return ranking, problem.value


def check_fair_bounds(n, k_fair, attribute_dict, alphas, betas, num_attributes):
    sizes = [0]*num_attributes
    for c in range(n):
        sizes[attribute_dict[c]] += 1

    lbs, ubs = [], []
    for g in range(num_attributes):
        lb = math.floor(alphas[g] * k_fair)
        ub = math.ceil(betas[g] * k_fair)
        lbs.append(lb); ubs.append(ub)

        if lb > sizes[g]:
            raise ValueError(f"Group {g}: lb {lb} > available items {sizes[g]}")
        if ub > sizes[g]:
            # not fatal, but clamp is reasonable
            pass

    if sum(lbs) > k_fair:
        raise ValueError(f"Sum of lbs {sum(lbs)} > k_fair {k_fair}")
    if sum(ubs) < k_fair:
        raise ValueError(f"Sum of ubs {sum(ubs)} < k_fair {k_fair}")


def ilp_prefix_jr_plus_fair_old(borda_ranking, approvals_by_k, n_voters, alphas, betas, k, attribute_dict, num_attributes):
    """
    borda_ranking: list[int] length n (candidate IDs 0..n-1)
    approvals_by_k: dict[int, dict[int, list[int]]]
        approvals_by_k[k][v] = list of candidates voter v approves at prefix k
        (in your setup, that's the voter's top-k ranked items)
    n_voters: int
    """
    n = len(borda_ranking)

    # candidate -> Borda position (0 = best)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p] = 1 if cand c at position p
    y = cp.Variable((n, n), boolean=True)   # y[c,d] = 1 if c ranked above d
    pos = cp.Variable(n, integer=True)      # pos[c] = position index of candidate c

    constraints = []

    # Permutation constraints on x
    # for c in range(n):
    #     constraints.append(cp.sum(x[c, :]) == 1)
    # for p in range(n):
    #     constraints.append(cp.sum(x[:, p]) == 1)
    constraints += [cp.sum(x, axis=1) == 1]   # each candidate assigned once
    constraints += [cp.sum(x, axis=0) == 1]   # each position filled once

    # Define pos[c] = sum_p p * x[c,p]
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(np.arange(n), x[c, :])))

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M (tight, two-sided)
    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            # if y[c,d]=1 then pos[c] <= pos[d]-1
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            # if y[c,d]=0 then pos[c] >= pos[d]+1
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # ------------------------------------------------------------
    # Prefix-JR constraints: JR must hold for every prefix k
    # ------------------------------------------------------------
    # z[v,k] = 1 if voter v is represented in top-(k+1) (k is 0-indexed here)
    z = cp.Variable((n_voters, n), boolean=True)

    for k in range(1, n + 1):
        quota = math.ceil(n_voters / k)  # match JR definition cleanly
        topk_positions = range(k)
        approvals_k = approvals_by_k[k]  # dict voter -> list of approved cands at prefix k

        # Link z[v,k-1] to whether top-k contains ANY approved candidate for voter v
        for v in range(n_voters):
            A_vk = approvals_k.get(v, [])
            if len(A_vk) == 0:
                # If voter approves nothing at this prefix (shouldn't happen in your setup),
                # force them unrepresented.
                constraints.append(z[v, k - 1] == 0)
                continue

            # z[v,k-1] <= sum_{a in A_vk} sum_{p<k} x[a,p]
            constraints.append(
                z[v, k - 1] <= cp.sum([x[a, p] for a in A_vk for p in topk_positions])
            )

        # JR condition per your checker:
        # For each candidate c, among voters who approve c, you cannot have quota (or more)
        # who are unrepresented => sum_{v in Vc} (1 - z[v,k-1]) <= quota - 1
        for c in range(n):
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                # Even if all are unrepresented, can't reach quota -> no JR constraint needed
                continue

            constraints.append(
                cp.sum([1 - z[v, k - 1] for v in Vc]) <= quota - 1
            )
    # ------------------------------------------------------------
    # Fairness constraints: for each attribute, enforce lower/upper bounds
    # ------------------------------------------------------------
    #Lower and upper bound constraints per attribute
    Y = cp.Variable(n, boolean = True)
    for attribute in range(num_attributes):
        coeff = np.zeros(n)
        for i in range(n):
            if attribute_dict[i] == attribute:
                coeff[i] = 1
        lb = math.floor(alphas[attribute] * k)
        ub = math.ceil(betas[attribute] * k)
        constraints.extend([coeff @ Y >= lb])
        constraints.extend([coeff @ Y <= ub]) 

    # Objective: minimize pairwise disagreements with Borda
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])  # disagreement if model says d > c

    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=True)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Decode ranking from x
    x_sol = x.value
    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

    return ranking, problem.value

def ilp_prefix_jr_plus_fair_old2(
    borda_ranking: List[int],
    approvals_by_k: Dict[int, Dict[int, List[int]]],
    n_voters: int,
    alphas: List[float],
    betas: List[float],
    k_fair: int,
    attribute_dict: Dict[int, int],
    num_attributes: int,
    *,
    solver: str = "GUROBI",
    threads: int = 1,
    presolve: int = 0,
    fallback_solvers: Tuple[str, ...] = ("SCIP", "CBC"),
) -> Tuple[Optional[List[int]], Optional[float], Optional[str]]:
    """
    Same as ilp_prefix_jr, plus group fairness constraints for the *top k_fair* positions.
    NOTE: This implementation enforces fairness using x directly (recommended),
          rather than introducing an unrelated Y variable.

    Returns:
        (ranking, objective_value, error_message)
    """
    n = len(borda_ranking)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)
    y = cp.Variable((n, n), boolean=True)
    pos = cp.Variable(n, integer=True)

    constraints = []
    constraints += [cp.sum(x, axis=1) == 1]
    constraints += [cp.sum(x, axis=0) == 1]

    p_idx = np.arange(n)
    constraints += [pos == x @ p_idx]
    constraints += [pos >= 0, pos <= n - 1]

    constraints += [cp.diag(y) == 0]
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # Prefix-JR constraints
    z = cp.Variable((n_voters, n), boolean=True)
    for k in range(1, n + 1):
        quota = math.ceil(n_voters / k)
        approvals_k = approvals_by_k[k]

        for v in range(n_voters):
            A_vk = approvals_k.get(v, [])
            if not A_vk:
                constraints.append(z[v, k - 1] == 0)
                continue
            sum_in_topk = cp.sum(x[A_vk, :k])
            constraints.append(z[v, k - 1] <= sum_in_topk)
            constraints.append(sum_in_topk <= (len(A_vk) * k) * z[v, k - 1])

        for c in range(n):
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                continue
            constraints.append(cp.sum(1 - z[Vc, k - 1]) <= quota - 1)

    # Fairness constraints on the top-k_fair positions, using x
    # count_g = sum_{c in group g} sum_{p<k_fair} x[c,p]
    for g in range(num_attributes):
        group_members = [i for i in range(n) if attribute_dict.get(i) == g]
        if not group_members:
            continue

        count_g = cp.sum(x[group_members, :k_fair])
        lb = math.floor(alphas[g] * k_fair)
        ub = math.ceil(betas[g] * k_fair)
        constraints += [count_g >= lb, count_g <= ub]

    # Objective
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])
    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)
    
    try:
        # problem.solve(
        #     solver=cp.GUROBI,
        #     verbose=False,
        #     Threads=1,
        #     Presolve=0,
        #     # If you ever see INF_OR_UNBD later, these help:
        #     # reoptimize=True,
        #     # DualReductions=0,
        # )
        problem.solve(
                    solver=cp.GUROBI,
                    verbose=False,
                    OutputFlag=0,
                    Threads=threads,
                    Presolve=2,
                    TimeLimit=60,    # seconds
                    MIPGap=0.01,     # 1% gap
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

    # def _try_solve(slv: str) -> bool:
    #     try:
    #         if slv.upper() == "GUROBI":
    #             # problem.solve(
    #             #     solver=cp.GUROBI,
    #             #     verbose=False,
    #             #     Threads=threads,
    #             #     Presolve=presolve,
    #             #     OutputFlag=0,
    #             # )
    #             # problem.solve(
    #             #     solver=cp.GUROBI,
    #             #     verbose=False,
    #             #     OutputFlag=0,
    #             #     Threads=threads,
    #             #     Presolve=2,      # default/strong presolve
    #             # )
    #             problem.solve(
    #                 solver=cp.GUROBI,
    #                 verbose=True,
    #                 OutputFlag=0,
    #                 Threads=threads,
    #                 Presolve=2,
    #                 TimeLimit=60,    # seconds
    #                 MIPGap=0.01,     # 1% gap
    #             )

    #         elif slv.upper() == "SCIP":
    #             problem.solve(solver=cp.SCIP, verbose=False)
    #         elif slv.upper() == "CBC":
    #             problem.solve(solver=cp.CBC, verbose=False)
    #         else:
    #             problem.solve(solver=getattr(cp, slv), verbose=False)
    #         return problem.status in ("optimal", "optimal_inaccurate")
    #     except Exception:
    #         return False

    # ok = _try_solve(solver)
    # if not ok:
    #     for fb in fallback_solvers:
    #         ok = _try_solve(fb)
    #         if ok:
    #             break

    # if not ok:
    #     return None, None, f"Optimization failed; status={problem.status}"

    # x_sol = x.value
    # if x_sol is None:
    #     return None, None, "Solved but x.value is None (unexpected)."

    # ranking = [None] * n
    # for c in range(n):
    #     ranking[int(np.argmax(x_sol[c, :]))] = c

    # return ranking, float(problem.value)








def ilp_prefix_jr_plus_fair_old4(
    borda_ranking, approvals_by_k, n_voters,
    alphas, betas, k, k_fair, attribute_dict, num_attributes,
    *, threads=1, time_limit=300, mip_gap=None
):
    import cvxpy as cp
    import numpy as np
    import math

    n = len(borda_ranking)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    #Feasibility checks 
    # assert check_fair_bounds(n, k_fair, attribute_dict, alphas, betas, num_attributes)
    
    x = cp.Variable((n, n), boolean=True)
    y = cp.Variable((n, n), boolean=True)
    pos = cp.Variable(n, integer=True)

    constraints = []

    # permutation
    constraints += [cp.sum(x, axis=1) == 1]
    constraints += [cp.sum(x, axis=0) == 1]

    # pos
    p_idx = np.arange(n)
    constraints += [pos >= 0, pos <= n - 1]
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(p_idx, x[c, :])))

    # tournament
    constraints += [cp.diag(y) == 0]
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # link y <-> pos
    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # ---------- Prefix-JR (use your faster x[A_vk, :k] form) ----------
    z = cp.Variable((n_voters, n), boolean=True)

    for kk in range(1, n + 1):
        quota = math.ceil(n_voters / kk)
        approvals_kk = approvals_by_k[kk]

        for v in range(n_voters):
            A_vk = approvals_kk.get(v, [])
            if not A_vk:
                constraints.append(z[v, kk - 1] == 0)
                continue

            sum_in_top = cp.sum(x[A_vk, :kk])
            constraints.append(z[v, kk - 1] <= sum_in_top)
            constraints.append(sum_in_top <= len(A_vk) * kk * z[v, kk - 1])

        for c in range(n):
            Vc = [v for v in range(n_voters) if c in approvals_kk.get(v, [])]
            if len(Vc) >= quota:
                constraints.append(cp.sum(1 - z[Vc, kk - 1]) <= quota - 1)

    # ---------- FAIRNESS ON TOP-k_fair ----------
    # candidate_in_topk[c] = 1 if c is placed in positions 0..k_fair-1
    candidate_in_topk = cp.sum(x[:, :k_fair], axis=1)  # shape (n,)

    # Build group indicator matrix G: shape (num_attributes, n)
    G = np.zeros((num_attributes, n), dtype=float)
    for c in range(n):
        g = attribute_dict[c]
        G[g, c] = 1.0

    # count per group
    group_counts = G @ candidate_in_topk  # shape (num_attributes,)

    for g in range(num_attributes):
        lb = math.floor(alphas[g] * k_fair)
        ub = math.ceil(betas[g] * k_fair)
        constraints += [group_counts[g] >= lb, group_counts[g] <= ub]

    # problem = cp.Problem(cp.Minimize(0), constraints)
    # problem.solve(solver=cp.GUROBI, verbose=False, LogFile="gurobi_feas.log")

    # objective
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c != d and borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])
    objective = cp.Minimize(cp.sum(obj_terms))
    problem = cp.Problem(objective, constraints)

    
    
    # solve_kwargs = dict(
    #     solver=cp.SCIP,
    #     verbose=True,
    #     # OutputFlag=1,
    #     # Threads=8,
    #     # Presolve=2,
    #     # MIPFocus=1,
    #     # Heuristics=0.5,
        
    # )
    # if time_limit is not None:
    #     solve_kwargs["TimeLimit"] = float(time_limit)
    # if mip_gap is not None:
    #     solve_kwargs["MIPGap"] = float(mip_gap)

    # problem.solve(**solve_kwargs)
    
    
    problem.solve(
    solver=cp.SCIP,
    verbose=False,
    scip_params={
        "limits/time": float(100),
        "limits/gap": 0.05,
        "parallel/maxnthreads": 8
    }
)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        # fallback
        problem.solve(solver=cp.SCIP, verbose=True,)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Optimization failed with status: {problem.status}")

    x_sol = x.value
    ranking = [None] * n
    for c in range(n):
        ranking[int(np.argmax(x_sol[c, :]))] = c

    return ranking, float(problem.value)



import cvxpy as cp
import numpy as np
import math


def ilp_prefix_jr_plus_fair_new(
    borda_ranking, approvals_by_k, n_voters,
    alphas, betas, k, k_fair, attribute_dict, num_attributes,
    *, threads=1, time_limit=300, mip_gap=None
):
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
            constraints.append(sum_in_topk <= len(A_vk) * k * z[v, k - 1])

        # JR constraints per candidate
        for c in range(n):
            # build Vc without scanning all voters expensively if you want,
            # but for n=30 this is fine.
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                continue
            constraints.append(cp.sum(1 - z[Vc, k - 1]) <= quota - 1)

    # ---------- FAIRNESS QUOTAS ON TOP-k ----------
    # candidate_in_topk[c] = number in {0,1} indicating whether c is placed in positions 0..k-1
    candidate_in_topk = cp.sum(x[:, :k], axis=1)  # shape (n,)

    # Build group indicator matrix G: shape (num_attributes, n)
    G = np.zeros((num_attributes, n), dtype=float)
    for c in range(n):
        g = attribute_dict[c]          # g in {0, ..., num_attributes-1}
        G[g, c] = 1.0

    # Count per group in top-k
    group_counts = G @ candidate_in_topk  # shape (num_attributes,)

    for g in range(num_attributes):
        lb = math.floor(alphas[g] * k)
        ub = math.ceil(betas[g] * k)
        constraints += [group_counts[g] >= lb, group_counts[g] <= ub]

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

