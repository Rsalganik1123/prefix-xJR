import os
import sys
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

import json
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict


from utils.vanilla_aggregation_methods import *
from utils.fair_aggregation_methods import *
from utils.borda_prefix_jr_ilp import * 
from utils.io import load_rankings_to_df
from utils.sampling_logic import generate_sample_sets, generate_sample_sets_stratified_by_bin

# =============================================================================
# Helper Functions
# =============================================================================

def _safe_worker_call(kind: str, method_name: str, fn, *args, **kwargs) -> dict:
    """
    Run a worker function and ALWAYS return a dict:
      {
        "ok": bool,
        "kind": "VANILLA"|"FAIR"|"OUR",
        "method": str,
        "result": Any (if ok),
        "error": {type, message, traceback} (if not ok)
      }
    """
    try:
        out = fn(*args, **kwargs)
        # fn should return (method_name, result) in your current pattern
        name, result = out
        return {
            "ok": True,
            "kind": kind,
            "method": name,
            "result": result,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "kind": kind,
            "method": method_name,
            "result": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }

def validate_full_rankings(rankings, candidate_items, context=""):
    """
    Enforce that each ranking is a FULL permutation of candidate_items.

    Checks:
      - correct length
      - no duplicates
      - contains exactly the same items as candidate_items
    """
    cand_list = list(candidate_items)
    cand_set = set(cand_list)
    d = len(cand_list)

    if len(cand_set) != d:
        raise ValueError(f"{context} candidate_items contains duplicates (size={d}, unique={len(cand_set)}).")

    for idx, r in enumerate(rankings):
        if not isinstance(r, (list, tuple)):
            raise TypeError(f"{context} ranking[{idx}] is not a list/tuple: {type(r)}")

        if len(r) != d:
            missing = sorted(cand_set - set(r))
            extra = sorted(set(r) - cand_set)
            raise ValueError(
                f"{context} ranking[{idx}] is partial or wrong length: len={len(r)} expected={d}\n"
                f"  missing={missing[:30]}\n"
                f"  extra={extra[:30]}"
            )

        if len(set(r)) != len(r):
            # show first few duplicates
            seen = set()
            dupes = []
            for x in r:
                if x in seen:
                    dupes.append(x)
                seen.add(x)
            raise ValueError(f"{context} ranking[{idx}] contains duplicates: {dupes[:30]}")

        if set(r) != cand_set:
            missing = sorted(cand_set - set(r))
            extra = sorted(set(r) - cand_set)
            raise ValueError(
                f"{context} ranking[{idx}] does not match candidate set.\n"
                f"  missing={missing[:30]}\n"
                f"  extra={extra[:30]}"
            )

def format_sampled_rankings(sampled_preferences: pd.DataFrame):
    return [row["Ranked_Items"] for _, row in sampled_preferences.iterrows()]

def process_for_fair_ranking_safe(candidates, group_df, ranks_for_fairness, item_key, complete=False):
    """
    Safe version:
    - drops candidate items missing from group_df (no label)
    - uses d = number of labeled items (NOT len(candidates))
    - compresses group labels to 0..g-1
    - filters+maps rankings and optionally completes them to full permutations
    """
    # Restrict to candidates that have group labels
    attributes_df = (
        group_df[group_df[item_key].isin(candidates)][[item_key, "binned"]]
        .drop_duplicates(subset=[item_key])
        .sort_values(item_key)
        .reset_index(drop=True)
    )

    candidates_set = set(candidates)
    labeled_set = set(attributes_df[item_key])
    missing_labels = sorted(candidates_set - labeled_set)
    if missing_labels:
        print(f"[WARN] {len(missing_labels)} sampled items missing group labels; dropping. "
              f"Example: {missing_labels[:10]}")

    # The effective candidate universe is ONLY those with labels
    items = attributes_df[item_key].tolist()
    d = len(items)
    if d == 0:
        raise ValueError("No labeled candidates left after filtering by group_df. "
                         "Check group_df.item coverage or your sampled_items.")

    # item -> 0..d-1 and back
    item_to_idx = {item: i for i, item in enumerate(items)}
    idx_to_item = {i: item for item, i in item_to_idx.items()}

    # compress binned values to 0..g-1 for this sample
    group_labels = attributes_df["binned"].tolist()
    uniq_groups = sorted(set(group_labels))
    group_to_gid = {g: i for i, g in enumerate(uniq_groups)}
    num_attributes = len(uniq_groups)

    # idx -> attribute (group id)
    idx_to_attribute = {
        item_to_idx[item]: group_to_gid[g]
        for item, g in zip(items, group_labels)
    }

    # map rankings to idx space, dropping unlabeled candidates
    mapped_rankings = []
    for r in ranks_for_fairness:
        filtered = [x for x in r if x in item_to_idx]
        mapped = [item_to_idx[x] for x in filtered]

        # de-duplicate while preserving order
        seen = set()
        mapped = [x for x in mapped if not (x in seen or seen.add(x))]
        mapped_rankings.append(mapped)

    if complete:
        universe = list(range(d))
        completed = []
        for r in mapped_rankings:
            seen = set(r)
            missing = [x for x in universe if x not in seen]
            completed.append(r + missing)
        mapped_rankings = completed

    alphas = [1.0 / num_attributes] * num_attributes
    betas = [1.0] * num_attributes

    return alphas, betas, mapped_rankings, idx_to_attribute, idx_to_item, num_attributes

def validate_all_candidates_labeled(sampled_items, group_df, context, item_key):
    labeled = set(group_df[item_key].unique())
    missing = sorted(set(sampled_items) - labeled)
    if missing:
        raise ValueError(
            f"Some sampled items are missing group labels in group_df.\n"
            f"Missing count={len(missing)} example={missing[:30]}"
        )


VANILLA_METHODS = {
    "CombMIN": comb_min, "CombMAX": comb_max, "CombSUM": comb_sum,
    "CombANZ": comb_anz, "CombMNZ": comb_mnz,
    "MC1": mc1, "MC2": mc2, "MC3": mc3, "MC4": mc4,
    "BordaCount": borda_count, "Dowdall": dowdall,
    "Median": median_rank, "Mean": mean_rank,
    "RRF": rrf, "iRANK": irank, "ER": er,
    "PostNDCG": postndcg, "CG": cg, "DIBRA": dibra, 
}
FAIR_METHODS = {
    "KuhlmanConsensus": Consensus,
    "FairMedian": FairILP,
}
OUR_METHODS = {
    'Our_Prefix_ILP': ilp_prefix_jr, 
    'Our_Prefix_Fair_ILP': ilp_prefix_jr_plus_fair,
    'Joe_Prefix_JR': prefix_JR_joe
}

# =============================================================================
# Worker helpers
# =============================================================================

def _run_vanilla_method(method_name: str, rankings, sampled_items):
    try:
        method = VANILLA_METHODS[method_name]
        result = method(rankings, sampled_items)
        return method_name, result
    except Exception as e:
        raise RuntimeError(f"[VANILLA method {method_name}] failed") from e

def _run_fair_method(method_name: str, alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes, fairness_k):
    try:
        method = FAIR_METHODS[method_name]
        result = method(alphas, betas, ranks_for_fairness, attributes_map, num_attributes, fairness_k)

        # map back to original item ids (raise a helpful error if something is out of range)
        mapped_back = []
        for i in result:
            if i not in idx_to_item:
                raise KeyError(f"Fair method returned index {i} not in idx_to_item keys "
                               f"(expected 0..{max(idx_to_item.keys())}).")
            mapped_back.append(idx_to_item[i])

        return method_name, mapped_back
    except Exception as e:
        raise RuntimeError(f"[FAIR method {method_name}] failed from {e}")

def _run_our_method(method_name: str, borda_ranking, approvals_by_k, n_voters,
                    alphas, betas, k, fairness_k, attributes_map, num_attributes,
                    idx_to_item, rankings, all_items, user_key ):
    try:
        method = OUR_METHODS[method_name]

        if method_name == "Our_Prefix_ILP":
            result, obj = method(borda_ranking, approvals_by_k, n_voters)
            mapped_back = [idx_to_item[i] for i in result]
        if method_name == 'Our_Prefix_Fair_ILP':
            result, obj = method(borda_ranking, approvals_by_k, n_voters,
                                      alphas, betas, fairness_k, attributes_map, num_attributes)
            mapped_back = [idx_to_item[i] for i in result]
        if method_name == 'Joe_Prefix_JR':
            result = method(rankings, all_items, user_key)
            mapped_back = result
        return method_name, mapped_back

    except Exception as e:
        raise RuntimeError(f"[Our method {method_name}] failed") from e


# =============================================================================
# Main Execution
# =============================================================================

def main():
    #Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['ml-1m', 'goodreads'])
    parser.add_argument("--user-sample-size", "-us", type=int, default=10)
    parser.add_argument("--item-sample-size", "-is", type=int, default=30)
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for aggregated rankings')
    parser.add_argument("--n-samples", "-n", type=int, default=1)
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1, help="Number of parallel worker processes")
    parser.add_argument("--sampling_method", type=str, choices=['basic', 'stratified_by_bin'], default='stratified_by_bin',)
    parser.add_argument("--fairness_k", type=int, default=10, help="k parameter for our fair methods.") 
    args = parser.parse_args()
    # Load dataset config
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)
    fairness_k = args.fairness_k
    user_key = dataset_cfg['dataset']["keys"]["user_key"]

    
    print("=" * 60)
    print("Rank Aggregation (Parallel, Safe Fair Mapping)")
    print("=" * 60)
    print(args)

    # File housekeeping :) outdir = f"/data2/rsalgani/Prefix/{args.dataset}/agg_files"
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load full rankings
    rankings_df, all_items = load_rankings_to_df(dataset_cfg)

    
    # Load group labels for items (used in stratified sampling and fair methods)
    group_df = pickle.load(open(dataset_cfg['dataset']['item_group_file_path'], "rb"))

    # Generate sample sets
    if args.sampling_method == 'basic':
        sampled_rankings_dfs, sampled_items, sampled_users = generate_sample_sets(dataset_cfg, n_samples=args.n_samples, n_users=args.user_sample_size,n_items=args.item_sample_size,all_items=list(all_items),preferences=rankings_df,)
    else: 
        sampled_rankings_dfs, sampled_items, sampled_users = generate_sample_sets_stratified_by_bin(dataset_cfg, args.n_samples, args.user_sample_size, args.item_sample_size,
        all_items=list(all_items), preferences=rankings_df, group_df=group_df, bin_col="binned", seed0=42) 
        
    # Format sampled rankings -- df to list for aggregation method compatibility 
    formatted_sampled_rankings = [format_sampled_rankings(df) for df in sampled_rankings_dfs]

    # Process each sample in parallel
    for seed in range(args.n_samples):
        
        # Create output directory for this sample
        write_dir = os.path.join(args.outdir, f"sample_{seed}")
        os.makedirs(write_dir, exist_ok=True)

        # Save sampled artifacts for reproducibility 
        pickle.dump(sampled_rankings_dfs[seed], open(os.path.join(write_dir, "sampled_rankings.pkl"), "wb"))
        pickle.dump(sampled_items[seed], open(os.path.join(write_dir, "sampled_items.pkl"), "wb"))
        pickle.dump(sampled_users[seed], open(os.path.join(write_dir, "sampled_users.pkl"), "wb"))
        
        # Launch 
        total = len(VANILLA_METHODS) + len(FAIR_METHODS) + len(OUR_METHODS)
        print(f"\n[seed={seed}] Processing {total} methods with jobs={args.jobs}...")

        # Sanity check: validate complete rankings and all candidates labeled -- THROW ERROR if ids missing from samples  
        rankings_seed = formatted_sampled_rankings[seed]
        sampled_items_seed = list(sampled_items[seed])     
        validate_full_rankings(rankings_seed,sampled_items_seed,context=f"[seed={seed}]")
        validate_all_candidates_labeled(sampled_items_seed, group_df, context=f"[seed={seed}]", item_key=dataset_cfg['dataset']["keys"]["item_key"])
        lens = [len(r) for r in rankings_seed]
        assert min(lens) ==  sum(lens)/len(lens) == max(lens) == len(sampled_items_seed)
        
        # Precompute fairness inputs once per seed using mapping
        alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes = process_for_fair_ranking_safe(
            sampled_items_seed, group_df, rankings_seed, complete=False, item_key=dataset_cfg['dataset']["keys"]["item_key"])
        pickle.dump(
            {'alphas': alphas,
            'betas': betas,
            'attributes_map': attributes_map,
            'idx_to_item': idx_to_item,
            'num_attributes': num_attributes}, open(os.path.join(write_dir, "fair_ranking_process.pkl"), 'wb')) 
        
        user_to_idx = dict(zip(sampled_rankings_dfs[seed][user_key].unique(), range(len(sampled_rankings_dfs[seed][user_key].unique()))))
        
        # Precompute inputs for OUR ILP Methods ---
        borda_ranking = borda_count(ranks_for_fairness, list(range(len(idx_to_item))))
        borda_ranking = [x for x, _ in borda_ranking]
        n_voters = len(ranks_for_fairness)
        # invert idx_to_item (idx->item) to item->idx
        item_to_idx = {item: idx for idx, item in idx_to_item.items()}
        sampled_pref_df = sampled_rankings_dfs[seed].copy()
        # map user ids -> 0..n_voters-1
        sampled_pref_df[user_key] = sampled_pref_df[user_key].map(user_to_idx)
        approvals_by_k = {}
        for k in range(1, len(borda_ranking) + 1):
            tmp = (
                sampled_pref_df
                .assign(Ranked_Items=lambda df: df["Ranked_Items"].apply(lambda x: x[:k]))
                .explode("Ranked_Items")
                .dropna(subset=["Ranked_Items"])
            )

            tmp["Ranked_Items"] = tmp["Ranked_Items"].map(item_to_idx)

            approvals_k = {}
            for v in range(n_voters):
                approvals_k[v] = (
                    tmp[tmp[user_key] == v]["Ranked_Items"]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
            approvals_by_k[k] = approvals_k


        failures = [] # collect failure records per seed
        futures = [] # track futures for parallelism 
        results_vanilla, results_fair, results_ours = {}, {} , {} 

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            # vanilla
            for name in VANILLA_METHODS.keys():
                futures.append(ex.submit(
                    _safe_worker_call,
                    "VANILLA", name, _run_vanilla_method,
                    name, rankings_seed, sampled_items_seed
                ))

            # fair
            for name in FAIR_METHODS.keys():
                futures.append(ex.submit(
                    _safe_worker_call,
                    "FAIR", name, _run_fair_method,
                    name, alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes, fairness_k
                ))
             
            # ours   
            for name in OUR_METHODS.keys():
                futures.append(ex.submit(
                    _safe_worker_call,
                    "OURS", name, _run_our_method,
                    name,
                    borda_ranking, approvals_by_k, n_voters,
                    alphas, betas, k, fairness_k, attributes_map, num_attributes,
                    idx_to_item, sampled_rankings_dfs[seed], sampled_items_seed, user_key
                ))

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"seed {seed}"):
                payload = fut.result()  # never raises now
                if payload["ok"]:
                    name = payload["method"]
                    if payload["kind"] == "VANILLA":
                        results_vanilla[name] = payload["result"]
                    if payload["kind"] == "FAIR":
                        results_fair[name] = payload["result"]
                    if payload["kind"] == "OURS": 
                        results_ours[name] = payload["result"]
                else:
                    failures.append(payload)

            # Write vanilla outputs
            for name, result in results_vanilla.items():
                file_path = os.path.join(write_dir, f"{name}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Method: {name}\n")
                    f.write(f"# Rank ItemID Score\n")
                    for rank, (item, score) in enumerate(result, 1):
                        f.write(f"{rank} {item} {score:.6f}\n")

            # Write fair outputs
            for name, ranking in results_fair.items():
                file_path = os.path.join(write_dir, f"{name}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Method: {name}\n")
                    for rank, item in enumerate(ranking, 1):
                        f.write(f"{rank} {item}\n")

            # Write our outputs
            for name, ranking in results_ours.items():
                file_path = os.path.join(write_dir, f"{name}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Method: {name}\n")
                    for rank, item in enumerate(ranking, 1):
                        f.write(f"{rank} {item}\n")

        # Save failures
        fail_path = os.path.join(write_dir, "failures.jsonl")
        with open(fail_path, "w", encoding="utf-8") as f:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")

        print(f"[seed={seed}] successes: vanilla={len(results_vanilla)} fair={len(results_fair)} ours={len(results_ours)}; "
            f"failures={len(failures)}")
        if failures:
            # print a quick summary
            by_method = {}
            for rec in failures:
                by_method.setdefault(rec["method"], 0)
                by_method[rec["method"]] += 1
            print("[seed=%d] failed methods: %s" % (seed, ", ".join(sorted(by_method.keys()))))

    print("\n" + "=" * 60)
    print("Aggregation Complete!")
    print(f"All files are located in: {args.outdir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
