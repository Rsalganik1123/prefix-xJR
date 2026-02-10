import numpy as np
import ranky as rk
import ipdb
import pandas as pd 
from itertools import permutations

import numpy as np
import pickle 

def pairwise_wins_from_rankings(rankings):
    """
    rankings: list[list[int]] where each inner list is [item_at_rank0, item_at_rank1, ...]
    Assumes every ranking contains the same items (same set + same length, e.g. 30).
    Returns:
      items: np.array of item IDs in the reference order used for W
      W: pairwise win matrix, W[i,j] = #voters who prefer items[i] over items[j]
    """
    items = np.array(rankings[0], dtype=int)
    n = len(items)
    idx = {item: k for k, item in enumerate(items)}

    W = np.zeros((n, n), dtype=np.int32)

    for order in rankings:
        # positions of each item in this voter's ranking
        pos = np.empty(n, dtype=np.int32)
        for r, item in enumerate(order):
            pos[idx[item]] = r

        # update pairwise counts
        for i in range(n):
            for j in range(i + 1, n):
                if pos[i] < pos[j]:
                    W[i, j] += 1
                else:
                    W[j, i] += 1

    return items, W

def kemeny_score(order_idx, W):
    """Maximize sum_{a before b} W[a,b]. order_idx is a permutation of indices [0..n-1]."""
    s = 0
    n = len(order_idx)
    for p in range(n):
        a = order_idx[p]
        for q in range(p + 1, n):
            b = order_idx[q]
            s += W[a, b]
    return int(s)

def delta_adjacent_swap(order_idx, W, k):
    """Score change if swapping positions k and k+1 in order_idx."""
    a = order_idx[k]
    b = order_idx[k + 1]
    return int(W[b, a] - W[a, b])

def kemeny_approx(W, restarts=30, max_passes=2000, seed=42):
    """
    Heuristic Kemeny:
      - Start from net-wins (Borda-like) ordering
      - Improve via repeated adjacent swaps that increase score
      - Do random restarts to escape local optima
    Returns:
      best_order_idx, best_score
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]

    # net wins: wins - losses
    net = W.sum(axis=1) - W.sum(axis=0)
    base = np.argsort(-net)  # descending

    best_order = base.copy()
    best_score = kemeny_score(best_order, W)

    for r in range(restarts):
        order = base.copy()
        if r > 0:
            rng.shuffle(order)

        score = kemeny_score(order, W)

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1

            for k in range(n - 1):
                d = delta_adjacent_swap(order, W, k)
                if d > 0:
                    order[k], order[k + 1] = order[k + 1], order[k]
                    score += d
                    improved = True

        if score > best_score:
            best_score = score
            best_order = order.copy()

    return best_order, best_score

def kemeny_from_dataframe(df, ranked_items_col="Ranked_Items", restarts=30, seed=42):
    """
    df[ranked_items_col] must contain lists like [item_at_rank0, ..., item_at_rank29].
    Returns:
      kemeny_item_order: list of item IDs best->worst
      best_score: Kemeny objective value (pairwise agreements)
      items, W: useful for debugging/analysis
    """
    rankings = df[ranked_items_col].tolist()

    # (optional sanity checks)
    n0 = len(rankings[0])
    if any(len(r) != n0 for r in rankings):
        raise ValueError("Not all rankings have the same length. (Partial rankings need a different builder.)")

    items, W = pairwise_wins_from_rankings(rankings)
    best_order_idx, best_score = kemeny_approx(W, restarts=restarts, seed=seed)

    kemeny_item_order = items[best_order_idx].tolist()
    return kemeny_item_order, best_score, items, W

