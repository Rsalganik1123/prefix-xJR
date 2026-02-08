import os
import glob
import numpy as np
from scipy.stats import kendalltau
import argparse
import pickle
import yaml
import multiprocessing as mp
import ipdb 

from utils.io import load_sampled_preferences, format_sampled_rankings_kt, load_consensus_ranking_kt

# =============================================================================
# 2. Kendall Tau Calculation
# =============================================================================

def calculate_average_tau(user_rankings, consensus_map):
    taus = []
    for uid, user_list in user_rankings.items():
        if len(user_list) < 2:
            continue

        user_ranks = list(range(len(user_list)))
        default_rank = 100000
        consensus_ranks = [consensus_map.get(str(item), default_rank) for item in user_list]

        tau, _ = kendalltau(user_ranks, consensus_ranks)
        if not np.isnan(tau):
            taus.append(tau)

    return float(np.mean(taus)) if taus else 0.0

def calculate_average_tau_by_group(user_rankings, user_groups, consensus_map):
    """
    user_rankings: dict {uid: [item1, item2, ...]}
    user_groups:   dict {uid: group_label}  (exactly 3 groups, but works for any)
    consensus_map: dict {item: rank_int}

    Returns:
      overall_mean: float
      group_means:  dict {group_label: float}
      group_counts: dict {group_label: int}  (#users contributing taus)
    """
    taus_all = []
    group_taus = {}

    for uid, user_list in user_rankings.items():
        if len(user_list) < 2:
            continue

        user_ranks = list(range(len(user_list)))
        default_rank = 100000
        consensus_ranks = [consensus_map.get(str(item), default_rank) for item in user_list]

        tau, _ = kendalltau(user_ranks, consensus_ranks)
        if np.isnan(tau):
            continue

        taus_all.append(tau)

        g = user_groups.get(uid, "UNKNOWN")
        group_taus.setdefault(g, []).append(tau)

    overall_mean = float(np.mean(taus_all)) if taus_all else 0.0
    group_means = {g: (float(np.mean(v)) if len(v) else 0.0) for g, v in group_taus.items()}
    group_counts = {g: len(v) for g, v in group_taus.items()}

    return overall_mean, group_means, group_counts

# =============================================================================
# Parallel worker plumbing
# =============================================================================

_GLOBAL_USER_RANKINGS = None
_GLOBAL_USER_GROUPS = None

def _init_worker(user_rankings, user_groups):
    global _GLOBAL_USER_RANKINGS, _GLOBAL_USER_GROUPS
    _GLOBAL_USER_RANKINGS = user_rankings
    _GLOBAL_USER_GROUPS = user_groups
    
def _score_one_consensus_file(file_path):
    global _GLOBAL_USER_RANKINGS, _GLOBAL_USER_GROUPS

    method_name = os.path.splitext(os.path.basename(file_path))[0]
    consensus_map = load_consensus_ranking_kt(file_path)
    if not consensus_map:
        return None

    overall, by_group, counts = calculate_average_tau_by_group(
        _GLOBAL_USER_RANKINGS, _GLOBAL_USER_GROUPS, consensus_map
    )

    # return a richer record
    return {
        "method": method_name,
        "overall": overall,
        "by_group": by_group,
        "counts": counts,
    }

def evaluate_agg_dir(agg_dir: str, pref_path: str, dataset_cfg: dict, workers: int, no_parallel: bool):
    user_key = dataset_cfg['dataset']['keys']['user_key']

    # Load preferences once per agg_dir
    rankings = load_sampled_preferences(pref_path)
    user_groups_df = pickle.load(open(dataset_cfg['dataset']['user_group_file_path'], 'rb'))

    user_rankings_df = rankings.merge(user_groups_df, on=user_key)
    user_groups = user_rankings_df.set_index(user_key)['entropy_bin'].to_dict()
    user_rankings = user_rankings_df.set_index(user_key)["Ranked_Items"].to_dict()

    # Find consensus files in this agg dir
    consensus_files = glob.glob(os.path.join(agg_dir, "*.txt"))
    if not consensus_files:
        print(f"[KT] No .txt files found in {agg_dir}")
        return None

    print(f"\n[KT] agg_dir={agg_dir}")
    print(f"[KT] Found {len(consensus_files)} consensus files; users={len(user_rankings)}")

    results = []
    if no_parallel or workers <= 1:
        for fp in consensus_files:
            out = _score_one_consensus_file(fp)  # uses globals only in parallel mode; here it reads globals? -> so do direct calc
            # safer: do direct load/calc when no_parallel
            consensus_map = load_consensus_ranking_kt(fp)
            if not consensus_map:
                continue
            overall, by_group, counts = calculate_average_tau_by_group(user_rankings, user_groups, consensus_map)
            results.append({"method": os.path.splitext(os.path.basename(fp))[0],
                            "overall": overall, "by_group": by_group, "counts": counts})
    else:
        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(user_rankings, user_groups)) as pool:
            for out in pool.imap_unordered(_score_one_consensus_file, consensus_files, chunksize=1):
                if out is not None:
                    results.append(out)

    results.sort(key=lambda d: d["overall"], reverse=True)

    out_path = os.path.join(agg_dir, "kendall_results_by_group.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[KT] Saved results -> {out_path} (methods={len(results)})")

    return results

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', '-p', default='', help='Path to sampled preferences (csv/pkl depending on your loader)')
    parser.add_argument('--agg', '-c', default='agg_results', help='Directory containing consensus TXT files')
    parser.add_argument('--dataset', '-d', default='ml-1m', choices=['ml-1m', 'goodreads'], help='Dataset name for config loading')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Number of parallel workers for consensus methods')
    parser.add_argument('--no-parallel', action='store_true', help='Disable multiprocessing (debug)')
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="single",
        choices=["single", "multi_k"],
        help="single: read --agg/*.txt. multi_k: read --agg/k_*/. (k_fair experiment layout)",
    )
    args = parser.parse_args()
    print(args)
    
    

    with open(f"data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)

    if args.mode == "single":
        evaluate_agg_dir(
            agg_dir=args.agg,
            pref_path=args.pref,
            dataset_cfg=dataset_cfg,
            workers=args.workers,
            no_parallel=args.no_parallel,
        )
        return

    # multi_k: args.agg points to sample dir containing k_* subdirs
    k_dirs = sorted([d for d in glob.glob(os.path.join(args.agg, "k_*")) if os.path.isdir(d)])
    if not k_dirs:
        print(f"[multi_k] No k_* directories found under {args.agg}")
        return

    print(f"[multi_k] Found {len(k_dirs)} k directories to evaluate.")
    for kd in k_dirs:
        print(f"\n[multi_k] Evaluating {kd}")
        evaluate_agg_dir(
            agg_dir=kd,
            pref_path=args.pref,
            dataset_cfg=dataset_cfg,
            workers=args.workers,
            no_parallel=args.no_parallel,
        )


if __name__ == "__main__":
    main()
