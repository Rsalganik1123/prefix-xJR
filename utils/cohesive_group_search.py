import sys
from collections import defaultdict, deque
import argparse
import time
import ipdb
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from tqdm import tqdm 
import math 

def read_edgelist_from_csv(path, user_key):
    df = pd.read_csv(path)
    U = set()
    V = set()
    edges = set()
    voters = df[user_key].apply(lambda x: f"V_{x}")
    candidates = df['Ranked_Items'].apply(lambda x: f"C_{x}")
    for i in range(len(voters)):
        # Prefix to disambiguate sides
        u = voters.iloc[i]
        v = candidates.iloc[i]
        U.add(u)
        V.add(v)
        edges.add((u, v))
    return U, V, edges


def read_edgelist_from_df(df, user_key):
    U = set()
    V = set()
    edges = set()
    voters = df[user_key].apply(lambda x: f"V_{x}")
    candidates = df['Ranked_Items'].apply(lambda x: f"C_{x}")
    # with tqdm(total=len(voters), desc="Reading edgelist") as pbar:
    for i in range(len(voters)):
        # Prefix to disambiguate sides
        u = voters.iloc[i]
        v = candidates.iloc[i]
        U.add(u)
        V.add(v)
        edges.add((u, v))
            # pbar.update(1)
    return U, V, edges


def read_aggregate_list(path):
    df = pd.read_csv(path)
    candidates = df.iloc[:, 2].apply(lambda x: f"C_{x}")
    return list(candidates)


def build_clique_extended_graph(U, V, edges):
    """
    Build adjacency dictionary for GC: nodes = U union V.
    Edges = original edges + all-pairs edges inside U + all-pairs edges inside V.
    Returns: dict node -> set(neighbors)
    """
    GC = dict()
    # with tqdm(total=len(U), desc="Building GC adjacency") as pbar:
    for u in U:
        # neighbors = all other U plus neighbors in V from edges
        neighbors = set(U)
        neighbors.remove(u)
        # add actual neighbors in V
        for (_, v) in filter(lambda e: e[0] == u, edges):
            neighbors.add(v)
        GC[u] = neighbors
            # pbar.update(1)
    # For speed, build mapping from u->list of v neighbors (to avoid many filters)
    neighU = defaultdict(set)
    neighV = defaultdict(set)
    # with tqdm(total=len(edges), desc="Adding GC neighbor edges") as pbar:
    for u, v in edges:
        neighU[u].add(v)
        neighV[v].add(u)
        # pbar.update(1)
    # recompute with these
    for u in U:
        GC[u] = (set(U) - {u}) | neighU[u]
    for v in V:
        GC[v] = (set(V) - {v}) | neighV[v]
    return GC

# Degeneracy ordering (Eppstein style) on an arbitrary undirected graph


def degeneracy_ordering(adj):
    """
    Return degeneracy ordering of nodes (smallest-first removal).
    adj: dict node->set(neighbors)
    Returns: list order (nodes), and core number mapping.
    """
    n = len(adj)
    deg = {u: len(adj[u]) for u in adj}
    maxdeg = max(deg.values()) if deg else 0
    bins = [deque() for _ in range(maxdeg+1)]
    for u, d in deg.items():
        bins[d].append(u)
    order = []
    core = dict()
    curr_deg = 0
    removed = set()
    for k in range(n):
        # find non-empty bin starting from curr_deg
        i = 0
        while i <= maxdeg and not bins[i]:
            i += 1
        if i > maxdeg:
            break
        u = bins[i].popleft()
        order.append(u)
        core[u] = i
        removed.add(u)
        # decrease degree of neighbors
        for w in list(adj[u]):
            if w in removed:
                continue
            d_old = deg[w]
            deg[w] -= 1
            bins[d_old].remove(w)
            bins[d_old-1].append(w)
    # Reverse order gives degeneracy order suitable for BK variant (process later nodes first)
    return order[::-1], core

# Bron-Kerbosch with pivot (classic implementation)


def bron_kerbosch_pivot(adj, R, P, X, output_callback):
    """
    adj: dict node->set(neighbors)
    R, P, X: sets
    output_callback(R) called when R is maximal clique
    """
    if not P and not X:
        output_callback(R)
        return
    # choose pivot u from P U X that maximizes |P ∩ N(u)|
    Px = P | X
    # choose pivot heuristically
    max_int = -1
    pivot = None
    for u in Px:
        inter = len(P & adj[u])
        if inter > max_int:
            max_int = inter
            pivot = u
    # for nodes in P \ N(pivot)
    for v in list(P - adj.get(pivot, set())):
        Nv = adj[v]
        bron_kerbosch_pivot(adj, R | {v}, P & Nv, X & Nv, output_callback)
        P.remove(v)
        X.add(v)


def maximal_bicliques_from_gc(GC_adj, Uset, Vset):
    """
    Enumerate maximal cliques of GC (which map to maximal bicliques of original G).
    For each maximal clique C, output C_U and C_V (split by membership).
    """
    final_voters, final_candidates = [], []
    # degeneracy order
    order, core = degeneracy_ordering(GC_adj)
    # process nodes according to degeneracy order: for each v, compute P = N(v) ∩ later_nodes
    nodes_pos = {node: i for i, node in enumerate(order)}
    # We'll process nodes in 'order' (which is reversed degeneracy)
    # with tqdm(total=len(order), desc="Finding maximal bicliques") as pbar:
    for v in order:
        N_v = GC_adj[v]
        # P = neighbors of v that come after v in the order
        P = {w for w in N_v if nodes_pos[w] > nodes_pos[v]}
        X = {w for w in N_v if nodes_pos[w] < nodes_pos[v]}
        R = {v}

        def cb(Rclique):
            # convert clique to biclique: split into U and V parts
            CU = set([int(x.strip('V_')) for x in Rclique if x in Uset])
            CV = set([int(x.strip('C_')) for x in Rclique if x in Vset])
            # According to theorem, both sets are non-empty for relevant bicliques
            if CU and CV:
                # out_writer(CU, CV)
                final_voters.append(CU)
                final_candidates.append(CV)

        bron_kerbosch_pivot(GC_adj, R, P, X, cb)
            # pbar.update(1)
    return final_voters, final_candidates


def add_subsets(voter_sets, candidate_sets, k, n):
    cohesive_voter_blocks = []
    cohesive_candidate_blocks = []
    l_cohesive = {} 
    for l in range(1, k+1):
        final_candidate_sets = []
        final_voter_sets = []
        for i in range(len(candidate_sets)):
            if len(candidate_sets[i]) >= l and len(voter_sets[i]) >= (l*n)/k:
                # final_candidate_sets.append(candidate_sets[i])
                # final_voter_sets.append(voter_sets[i])
                
                #Just for testing
                cohesive_voter_blocks.append(voter_sets[i])
                cohesive_candidate_blocks.append(candidate_sets[i])
                
                voter_subsets = list(
                    combinations(voter_sets[i], r=int(math.ceil((l*n)/k)) ))
                
                final_voter_sets.extend(voter_subsets)
                final_candidate_sets.extend([candidate_sets[i]] * len(voter_subsets))
        final_voter_sets = [list(s) for s in final_voter_sets]
        final_candidate_sets = [list(s) for s in final_candidate_sets]
        l_cohesive[l] = {'voter_sets': final_voter_sets,
                         'candidate_sets': final_candidate_sets}
    # print(cohesive_voter_blocks, cohesive_candidate_blocks)
    return l_cohesive


def find_maximal_cohesive_groups(partial_lists, committee_size, data_cfg):
    U, V, edges = read_edgelist_from_df(partial_lists, data_cfg['dataset']['keys']['user_key'])
    # print('Building clique-extended graph...')
    GC = build_clique_extended_graph(U, V, edges)
    # print('Finding maximal bicliques via GC...')
    voter_sets, candidate_sets = maximal_bicliques_from_gc(GC, U, V)
    return voter_sets, candidate_sets


def find_all_cohesive_groups(partial_lists, committee_size, number_voters, data_cfg):
    U, V, edges = read_edgelist_from_df(partial_lists, data_cfg['dataset']['keys']['user_key'])
    GC = build_clique_extended_graph(U, V, edges)
    voter_sets, candidate_sets = maximal_bicliques_from_gc(GC, U, V) 
    l_cohesive = add_subsets(voter_sets, candidate_sets,
                             committee_size, number_voters)
    return voter_sets, candidate_sets, l_cohesive


