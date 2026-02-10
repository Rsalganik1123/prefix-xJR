import numpy as np
import pandas as pd 
import ipdb 

def generate_sample_sets_stratified_by_bin(dataset_cfg, n_samples, n_users, n_items, all_items,preferences, group_df,bin_col="binned",seed0=0,replace_items=False, min_per_bin=0):
    """
    Stratified sampling over item categories given by group_df[bin_col] (e.g., 'binned').

    Returns:
      dfs: list of sampled user preference dfs (same schema as your preferences df)
      items: list of np arrays of sampled item ids
      users: list of np arrays of sampled user indices (same as your original code)
    """
    user_key = dataset_cfg['dataset']["keys"]["user_key"]
    item_key = dataset_cfg['dataset']["keys"]["item_key"]

    # Users universe
    all_users = list(preferences[user_key].unique())

    # Items universe intersection with group_df labels
    all_items_set = set(all_items)
    labeled = group_df[[item_key, bin_col]].drop_duplicates(subset=[item_key])
    labeled = labeled[labeled[item_key].isin(all_items_set)]
    if labeled.empty:
        raise ValueError(f"No overlap between all_items and group_df[{item_key}] after filtering.")

    # Build bin -> list(items)
    bin_to_items = {}
    for b, sub in labeled.groupby(bin_col):
        bin_to_items[b] = sub[item_key].to_numpy()

    # Decide which bins are eligible
    bins = sorted(bin_to_items.keys())
    if len(bins) == 0:
        raise ValueError(f"No bins found in group_df[{bin_col}] after filtering.")

    # Bin weights: default equal over bins
    
    weights = np.array([1.0] * len(bins), dtype=float)
    
    weights = weights / weights.sum()

    dfs, items, users = [], [], []

    for seed in range(n_samples):
        rng = np.random.default_rng(seed0 + seed)

        # sample users
        if n_users > len(all_users):
            raise ValueError(f"n_users={n_users} > total users={len(all_users)}")
        user_idx = rng.choice(len(all_users), size=n_users, replace=False)
        sampled_user_ids = [all_users[i] for i in user_idx]

        sampled_pref = preferences[preferences[user_key].isin(sampled_user_ids)].copy()

        # --- stratified item sampling ---
        # target counts per bin
        target = np.floor(weights * n_items).astype(int)

        # ensure total sums to n_items by distributing remainder
        remainder = n_items - target.sum()
        if remainder > 0:
            # give extra items to bins with largest fractional parts
            frac = (weights * n_items) - np.floor(weights * n_items)
            order = np.argsort(-frac)
            for t in order[:remainder]:
                target[t] += 1

        # enforce min_per_bin if requested (best-effort)
        if min_per_bin > 0:
            for i, b in enumerate(bins):
                if target[i] < min_per_bin:
                    target[i] = min_per_bin
            # re-balance if we exceeded n_items
            excess = target.sum() - n_items
            if excess > 0:
                # subtract from bins with largest targets first, but don't go below min_per_bin
                order = np.argsort(-target)
                for i in order:
                    if excess == 0:
                        break
                    can_take = target[i] - min_per_bin
                    if can_take > 0:
                        take = min(can_take, excess)
                        target[i] -= take
                        excess -= take

        sampled_items_list = []
        deficits = 0

        # First pass: sample per bin up to availability
        for i, b in enumerate(bins):
            pool = bin_to_items[b]
            need = int(target[i])
            if need <= 0:
                continue

            if (not replace_items) and need > len(pool):
                # take all, record deficit
                sampled_items_list.extend(pool.tolist())
                deficits += (need - len(pool))
            else:
                draw = rng.choice(pool, size=need, replace=replace_items)
                sampled_items_list.extend(draw.tolist())

        # Second pass: fill deficits from remaining pools
        if deficits > 0:
            # build a pool of remaining items not already sampled (if no replacement)
            if replace_items:
                # just sample from all labeled items with weights
                all_labeled_items = labeled[item_key].to_numpy()
                fill = rng.choice(all_labeled_items, size=deficits, replace=True)
                sampled_items_list.extend(fill.tolist())
            else:
                already = set(sampled_items_list)
                remaining = [x for x in labeled[item_key].to_numpy() if x not in already]
                if len(remaining) < deficits:
                    raise ValueError(
                        f"Not enough distinct labeled items to fill deficits: "
                        f"need={deficits}, remaining={len(remaining)}. "
                        f"Consider replace_items=True or lower n_items."
                    )
                fill = rng.choice(np.array(remaining), size=deficits, replace=False)
                sampled_items_list.extend(fill.tolist())

        # Ensure exactly n_items (can happen with min_per_bin tweaks)
        sampled_items_arr = np.array(sampled_items_list, dtype=labeled[item_key].dtype)
        if len(sampled_items_arr) > n_items:
            # trim deterministically for this seed
            sampled_items_arr = sampled_items_arr[:n_items]
        elif len(sampled_items_arr) < n_items:
            # shouldn't happen, but protect
            raise RuntimeError(f"Internal error: sampled {len(sampled_items_arr)} < n_items={n_items}")

        # Filter rankings to sampled items
        item_set = set(sampled_items_arr.tolist())
        sampled_pref["Ranked_Items"] = sampled_pref["Ranked_Items"].apply(
            lambda x: [item for item in x if item in item_set]
        )

        dfs.append(sampled_pref)
        items.append(sampled_items_arr)
        users.append(user_idx)

    return dfs, items, users

def generate_sample_sets(dataset_cfg, n_samples, n_users, n_items, all_items, preferences):
    user_key = dataset_cfg['dataset']["keys"]["user_key"]
    item_key = dataset_cfg['dataset']["keys"]["item_key"]
    est_rating_key = dataset_cfg['dataset']["keys"]["est_rating_key"]
    
    all_users = list(preferences[user_key].unique())
    dfs, items, users = [], [], []
    all_items = np.array(list(all_items))

    for seed in range(n_samples):
        user_idx = np.random.choice(len(all_users), size=n_users, replace=False)
        item_idx = np.random.choice(all_items, size=n_items, replace=False)

        sampled_pref = preferences[preferences[user_key].isin([all_users[i] for i in user_idx])].copy()

        # Filter to sampled items
        item_set = set(item_idx.tolist())
        sampled_pref["Ranked_Items"] = sampled_pref["Ranked_Items"].apply(
            lambda x: [item for item in x if item in item_set]
        )

        dfs.append(sampled_pref)
        items.append(item_idx)
        users.append(user_idx)

    return dfs, items, users

