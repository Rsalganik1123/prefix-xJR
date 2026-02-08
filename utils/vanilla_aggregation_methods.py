import numpy as np
import csv
import os
from collections import defaultdict
import argparse
import sys
import ipdb 

# =============================================================================
# Helper Functions
# =============================================================================

def get_item_scores_from_rankings(rankings, all_items):
    item_scores = defaultdict(list)
    for ranking in rankings:
        n = len(ranking)
        items_in_ranking = set(ranking)
        for pos, item in enumerate(ranking):
            score = 1.0 - (pos / n) if n > 0 else 0
            item_scores[item].append(score)
        for item in all_items:
            if item not in items_in_ranking:
                item_scores[item].append(0.0)
    return item_scores


# =============================================================================
# Aggregation Methods
# =============================================================================

# --- Comb* Family ---
def comb_min(rankings, all_items):
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {item: min(s_list) for item, s_list in item_scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])

def comb_max(rankings, all_items):
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {item: max(s_list) for item, s_list in item_scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])

def comb_sum(rankings, all_items):
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {item: sum(s_list) for item, s_list in item_scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])

def comb_anz(rankings, all_items):
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {}
    for item, s_list in item_scores.items():
        non_zero = [s for s in s_list if s > 0]
        scores[item] = sum(non_zero) / len(non_zero) if non_zero else 0
    return sorted(scores.items(), key=lambda x: -x[1])

def comb_mnz(rankings, all_items):
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {}
    for item, s_list in item_scores.items():
        non_zero = [s for s in s_list if s > 0]
        scores[item] = sum(non_zero) * len(non_zero)
    return sorted(scores.items(), key=lambda x: -x[1])

# --- Markov Chains ---
def build_pairwise_matrix(rankings, all_items, top_k=500): 
    items = list(all_items)
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    pref = np.zeros((n, n))
    for ranking in rankings:
        top_items = ranking[:top_k]
        for i, item_i in enumerate(top_items):
            idx_i = item_to_idx[item_i]
            for j in range(i + 1, len(top_items)):
                item_j = top_items[j]
                idx_j = item_to_idx[item_j]
                pref[idx_i][idx_j] += 1
    return pref, items, item_to_idx

def mc1(rankings, all_items, top_k=500, max_iter=50, tol=1e-6):
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    transition = np.zeros((n, n))
    for i in range(n):
        winners = [j for j in range(n) if pref[j][i] > 0]
        if winners:
            for j in winners:
                transition[i][j] = 1.0 / len(winners)
        else:
            transition[i] = 1.0 / n
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol: break
        scores = new_scores
    return sorted([(items[i], scores[i]) for i in range(n)], key=lambda x: -x[1])

def mc2(rankings, all_items, top_k=500, max_iter=50, tol=1e-6):
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    transition = np.zeros((n, n))
    for i in range(n):
        total = sum(pref[j][i] for j in range(n))
        if total > 0:
            for j in range(n):
                transition[i][j] = pref[j][i] / total
        else:
            transition[i] = 1.0 / n
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol: break
        scores = new_scores
    return sorted([(items[i], scores[i]) for i in range(n)], key=lambda x: -x[1])

def mc3(rankings, all_items, top_k=500, damping=0.85, max_iter=50, tol=1e-6):
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    transition = np.zeros((n, n))
    for i in range(n):
        winners = [j for j in range(n) if pref[j][i] > 0]
        if winners:
            for j in winners:
                transition[i][j] = 1.0 / len(winners)
        else:
            transition[i] = 1.0 / n
    transition = damping * transition + (1 - damping) / n
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol: break
        scores = new_scores
    return sorted([(items[i], scores[i]) for i in range(n)], key=lambda x: -x[1])

def mc4(rankings, all_items, top_k=500, damping=0.85, max_iter=50, tol=1e-6):
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    transition = np.zeros((n, n))
    for i in range(n):
        total = sum(pref[j][i] for j in range(n))
        if total > 0:
            for j in range(n):
                transition[i][j] = pref[j][i] / total
        else:
            transition[i] = 1.0 / n
    transition = damping * transition + (1 - damping) / n
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol: break
        scores = new_scores
    return sorted([(items[i], scores[i]) for i in range(n)], key=lambda x: -x[1])

# --- Position & Reciprocal ---
def borda_count(rankings, all_items):
    scores = defaultdict(float)
    for ranking in rankings:
        k = len(ranking)
        for pos, item in enumerate(ranking):
            scores[item] += (k - 1 - pos)
    return sorted(scores.items(), key=lambda x: -x[1])

def dowdall(rankings, all_items):
    scores = defaultdict(float)
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            scores[item] += 1.0 / (pos + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

def median_rank(rankings, all_items):
    item_ranks = defaultdict(list)
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        for item in all_items:
            item_ranks[item].append(rank_dict.get(item, max_rank))
    scores = {item: -np.median(ranks) for item, ranks in item_ranks.items()}
    return sorted(scores.items(), key=lambda x: -x[1])

def mean_rank(rankings, all_items):
    item_ranks = defaultdict(list)
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        for item in all_items:
            item_ranks[item].append(rank_dict.get(item, max_rank))
    scores = {item: -np.mean(ranks) for item, ranks in item_ranks.items()}
    return sorted(scores.items(), key=lambda x: -x[1])

def hpa(rankings, all_items):
    item_best_rank = {item: float('inf') for item in all_items}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            if rank < item_best_rank[item]:
                item_best_rank[item] = rank
    max_r = 1000 
    scores = {item: max_r - r for item, r in item_best_rank.items()}
    return sorted(scores.items(), key=lambda x: -x[1])

def rrf(rankings, all_items, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

def irank(rankings, all_items):
    scores = defaultdict(float)
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += 1.0 / rank
    return sorted(scores.items(), key=lambda x: -x[1])

def er(rankings, all_items):
    scores = defaultdict(float)
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += (1.0 / rank) / rank
    return sorted(scores.items(), key=lambda x: -x[1])

def postndcg(rankings, all_items):
    scores = defaultdict(float)
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += 1.0 / np.log2(rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

def cg(rankings, all_items):
    scores = defaultdict(float)
    for ranking in rankings:
        max_pos = len(ranking)
        for pos, item in enumerate(ranking):
            relevance = max_pos - pos
            scores[item] += relevance
    return sorted(scores.items(), key=lambda x: -x[1])

def dibra(rankings, all_items):
    items = list(all_items)
    n_items = len(items)
    item_ranks = defaultdict(list)
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        for item in all_items:
            item_ranks[item].append(rank_dict.get(item, max_rank))
    scores = {item: -np.mean(ranks) for item, ranks in item_ranks.items()}
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    consensus_rank = {item: pos + 1 for pos, (item, _) in enumerate(sorted_items)}
    new_scores = defaultdict(float)
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        dist = 0
        for item in ranking:
            r1 = rank_dict[item]
            r2 = consensus_rank.get(item, n_items)
            dist += (r1 - r2) ** 2
        weight = 1.0 / (1 + np.sqrt(dist / len(ranking)) + 1e-9)
        for item in ranking:
            new_scores[item] += weight * (len(ranking) - rank_dict[item])
    return sorted(new_scores.items(), key=lambda x: -x[1])

