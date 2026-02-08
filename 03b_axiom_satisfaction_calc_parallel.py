#!/usr/bin/env python3


import os
import glob
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import ipdb 
import yaml


from utils.cohesive_group_search import find_all_cohesive_groups
from utils.axiom_checks import (
    JR_check_satisfaction_given_committee,
    PJR_check_satisfaction_given_committee,
    EJR_check_satisfaction_given_committee,
)
from utils.io import load_consensus_ranking, load_sampled_preferences

# =============================================================================
# Worker Globals (set once per process)
# =============================================================================
_PREF = None
_NUM_VOTERS = None
_ALL_CANDIDATES = None


def _worker_init(preferences: pd.DataFrame, number_voters: int, all_candidates):
    """
    Store large, read-only objects once per process to avoid repeated pickling.
    """
    global _PREF, _NUM_VOTERS, _ALL_CANDIDATES
    _PREF = preferences
    _NUM_VOTERS = number_voters
    _ALL_CANDIDATES = all_candidates

def evaluate_agg_dir(agg_dir: str, pref_path: str, workers: int, dataset_cfg: dict):
    print("(2) Loading Consensus Files")
    print("-" * 70)

    consensus_files = glob.glob(os.path.join(agg_dir, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {agg_dir}")
        return None

    preferences = load_sampled_preferences(pref_path)
    number_voters = len(preferences[dataset_cfg['dataset']['keys']['user_key']].unique())
    all_candidates = preferences.explode("Ranked_Items")["Ranked_Items"].unique()
    number_candidates = len(all_candidates)

    print("-" * 70)
    print(f"RUN STATS for: {agg_dir}")
    print("-" * 70)
    print("Number of Voters:", number_voters)
    print("Number of Candidates:", number_candidates)
    print("Number of methods to evaluate:", len(consensus_files))
    print("-" * 70)

    print("(4) Calculating Axiom Satisfaction (parallel over methods)")
    results = run_parallel(
        preferences=preferences,
        consensus_files=consensus_files,
        all_candidates=all_candidates,
        number_voters=number_voters,
        max_workers=workers,
        data_cfg=dataset_cfg,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Method':<20} | {'JR':^5} | {'PJR':^5} | {'EJR':^5}")
    print("-" * 60)

    def mark(x):
        return "✓" if all(x) else "✗"

    for method, satisfaction in results:
        jr, pjr, ejr = satisfaction["JR"], satisfaction["PJR"], satisfaction["EJR"]
        print(f"{method:<20} | {mark(jr):^5} | {mark(pjr):^5} | {mark(ejr):^5}")

    print("=" * 60)

    # Save summary boolean table
    results_df = pd.DataFrame(
        {method: {k: all(v) for k, v in metrics.items()} for method, metrics in results}
    ).T

    # print("POTATO")
    # exit()
    out_path = os.path.join(agg_dir, "axiom_satisfaction_results.pkl")
    # print(out_path)
    pickle.dump(results_df, open(out_path, "wb"))
    print(f"Saved: {out_path}")

    return results_df

# =============================================================================
# 2. Method Evaluation (runs inside worker)
# =============================================================================
def _eval_one_method(file_path: str, data_cfg):
    global _PREF, _NUM_VOTERS, _ALL_CANDIDATES

    method_name = os.path.splitext(os.path.basename(file_path))[0]
    committee = load_consensus_ranking(file_path)
    if not committee:
        return method_name, None

    satisfaction = {"JR": [], "PJR": [], "EJR": []}

    n_prefixes = len(committee)
    t0 = time.time()

    # print progress about 10 times per method
    report_every = max(1, n_prefixes // 10)

    for prefix_idx in range(n_prefixes):
        k = prefix_idx + 1

        preferences_at_prefix = (
            _PREF.assign(Ranked_Items=lambda df: df["Ranked_Items"].apply(lambda x: x[:k]))
            .explode("Ranked_Items")
            .reset_index(drop=True)
        )

        _, _, l_cohesive = find_all_cohesive_groups(
            preferences_at_prefix, committee_size=k, number_voters=_NUM_VOTERS, data_cfg=data_cfg
        )

        comm_k = committee[:k]

        satisfaction["JR"].append(
            JR_check_satisfaction_given_committee(
                comm_k,
                partial_lists=preferences_at_prefix,
                all_candidates=_ALL_CANDIDATES,
                n=_NUM_VOTERS,
                k=k,
                user_key=data_cfg['dataset']['keys']['user_key'],
            )
        )
        satisfaction["PJR"].append(
            PJR_check_satisfaction_given_committee(
                comm_k,
                partial_lists=preferences_at_prefix,
                l_cohesive=l_cohesive,
                user_key=data_cfg['dataset']['keys']['user_key'],
            )
        )
        satisfaction["EJR"].append(
            EJR_check_satisfaction_given_committee(comm_k, preferences_at_prefix, user_key=data_cfg['dataset']['keys']['user_key'], data_cfg=data_cfg)
        )

        # ---- STATUS PRINTS ----
        if (prefix_idx + 1) % report_every == 0 or (prefix_idx + 1) == n_prefixes:
            elapsed = time.time() - t0
            done = prefix_idx + 1
            rate = elapsed / done
            remaining = (n_prefixes - done) * rate
            print(
                f"[{method_name}] prefix {done}/{n_prefixes} "
                f"elapsed={elapsed:.1f}s  eta={remaining:.1f}s",
                flush=True
            )

    total = time.time() - t0
    print(f"[{method_name}] DONE in {total:.2f}s", flush=True)

    return method_name, satisfaction

# =============================================================================
# 3. Parallel Runner
# =============================================================================
def run_parallel(preferences, consensus_files, all_candidates, number_voters, max_workers=None, data_cfg=None):
    import time

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(preferences, number_voters, all_candidates),
    ) as ex:
        futures = {ex.submit(_eval_one_method, fp, data_cfg): fp for fp in consensus_files}

        completed = 0
        total = len(futures)

        for fut in as_completed(futures):
            method_name, satisfaction = fut.result()
            completed += 1

            elapsed = time.time() - t0
            avg_per_method = elapsed / completed
            eta = (total - completed) * avg_per_method

            print(
                f"[MAIN] Completed {completed}/{total} methods | "
                f"elapsed={elapsed/60:.2f} min | eta={eta/60:.2f} min",
                flush=True
            )

            if satisfaction is not None:
                results.append((method_name, satisfaction))
                print(results)

    results.sort(key=lambda x: x[0])
    return results

# =============================================================================
# 4. Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agg",
        "-a",
        default="consensus_results",
        help="Directory containing consensus TXT files",
    )
    parser.add_argument(
        "--pref",
        "-p",
        default="recommendations.csv",
        help="Path to sampled preferences pickle",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: os.cpu_count())",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        choices=["ml-1m", "goodreads"],
        help="dataset name (for loading dataset config)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="single",
        choices=["single", "multi_k"],
        help="single: read --agg/*.txt. multi_k: read --agg/k_*/. (k_fair experiment layout)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("(1) Loading data")
    print("-" * 70)

    with open(f"data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)

    preferences = load_sampled_preferences(args.pref)
    number_voters = len(preferences[dataset_cfg['dataset']['keys']['user_key']].unique())
    all_candidates = preferences.explode("Ranked_Items")["Ranked_Items"].unique()
    number_candidates = len(all_candidates)

    print("(2) Loading Consensus Files")
    print("-" * 70)
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    
    with open(f"data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
        print(dataset_cfg)

    if args.mode == "single":
        evaluate_agg_dir(args.agg, args.pref, args.workers, dataset_cfg)
        return

    # multi_k behavior: args.agg points to sample dir (contains k_*/ subdirs)
    k_dirs = sorted([d for d in glob.glob(os.path.join(args.agg, "k_*")) if os.path.isdir(d)])
    if not k_dirs:
        print(f"[multi_k] No k_* directories found under {args.agg}")
        return

    print(f"[multi_k] Found {len(k_dirs)} k directories to evaluate.")
    for kd in k_dirs:
        print(f"\n[multi_k] Evaluating {kd}")
        evaluate_agg_dir(kd, args.pref, args.workers, dataset_cfg)



if __name__ == "__main__":
    main()
