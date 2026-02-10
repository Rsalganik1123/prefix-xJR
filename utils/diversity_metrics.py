import ipdb 

def calculate_pop_group_item_diversity_old(full_committee, item_groups, k, dataset_cfg):
    ipdb.set_trace()
    user_key = dataset_cfg['dataset']['keys']['user_key']
    item_key = dataset_cfg['dataset']['keys']['item_key']
    committee_at_k = full_committee[:k]
    coverage_results, percentage_results = {}, {}
    df = item_groups[item_groups[item_key].isin(full_committee)]
    for group in item_groups['binned'].unique():
        covered_items = df[(df['binned'] == group) & (df[item_key].isin(committee_at_k))][item_key].tolist()
        group_items = df[df['binned'] == group][item_key].to_list()
        coverage = len(covered_items) / len(group_items) if group_items else 0
        percentage = len(covered_items) / k
        percentage_results[group] = percentage
        coverage_results[group] = coverage
    return coverage_results, percentage_results

def calculate_pop_group_item_diversity(full_committee, item_groups, k, dataset_cfg, bin_col="binned"):
    """
    Computes, per popularity bin/group:
      - coverage[group]   = (# of items from group that appear in top-k) / (total # distinct committee items in that group)
      - percentage[group] = (# of items from group that appear in top-k) / k
    """
    
    item_key = dataset_cfg["dataset"]["keys"]["item_key"]

    committee_at_k = list(full_committee[:k])
    committee_set = set(full_committee)
    topk_set = set(committee_at_k)

    # Restrict to items that appear in the committee (all positions)
    df = item_groups.loc[item_groups[item_key].isin(committee_set), [item_key, bin_col]].drop_duplicates()

    coverage_results = {}
    percentage_results = {}

    # Precompute totals per group (within committee universe)
    group_to_items = (
    df.groupby(bin_col, observed=True)[item_key]
      .apply(list)
      .to_dict()
    )   

    for group, group_items in group_to_items.items():
        group_item_set = set(group_items)

        # How many from this group appear in top-k
        covered_count = len(group_item_set & topk_set)

        coverage_results[group] = (covered_count / len(group_items)) if group_items else 0.0
        percentage_results[group] = covered_count / float(k) if k > 0 else 0.0

    return coverage_results, percentage_results
