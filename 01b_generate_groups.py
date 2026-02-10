import pandas as pd 
import ipdb 
import math 
import pickle 
import numpy as np
import yaml 
import argparse 

def generate_item_groups(args, dataset_cfg): 
    dataset = dataset_cfg['dataset']['name']
    item_key = dataset_cfg['dataset']['keys']['item_key']
    item_column = dataset_cfg['dataset']['format']['columns']['item_column']
    user_column = dataset_cfg['dataset']['format']['columns']['user_column']
    output_path = dataset_cfg['dataset']['item_group_file_path']
    file_path = dataset_cfg['dataset']['ratings_file_path']
    
    if dataset == 'ml-1m': 
        df = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    if dataset == 'goodreads': 
        df = pd.read_csv(file_path, sep=',', engine='python')
        
    item_grouped = df.groupby(item_column).count().reset_index().rename(columns={user_column: 'count_of_users'})
    item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(lambda x: math.log(x, 5))
    item_grouped['binned'] = pd.cut(item_grouped['log_smoothed'], 5, retbins=False, labels=list(range(5)))
    item_grouped = item_grouped.rename(columns ={item_column: item_key})
    print(item_grouped.groupby('binned').count())
    pickle.dump(item_grouped, open(output_path, 'wb'))


def user_bin_distribution(user_grouped, n_bins, user_column): 
    """
    Turns item_grps list-of-bins into:
      - counts_0..counts_4
      - p_0..p_4 (normalized)
    """
    def counts_from_list(lst):
        c = np.bincount(np.asarray(lst, dtype=int), minlength=n_bins)
        return c

    counts = np.vstack(user_grouped["item_grps"].apply(counts_from_list).to_numpy())
    counts_df = pd.DataFrame(counts, columns=[f"count_{i}" for i in range(n_bins)])

    # proportions
    row_sums = counts_df.sum(axis=1).replace(0, np.nan)
    p_df = counts_df.div(row_sums, axis=0).fillna(0.0)
    p_df.columns = [f"p_{i}" for i in range(n_bins)]

    return pd.concat([user_grouped[[user_column]].reset_index(drop=True), counts_df, p_df], axis=1)


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy. Higher => more heterogeneous."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def generate_user_groups(args, dataset_cfg): 
    dataset = dataset_cfg['dataset']['name']
    item_key = dataset_cfg['dataset']['keys']['item_key']
    user_key = dataset_cfg['dataset']['keys']['user_key']
    item_column = dataset_cfg['dataset']['format']['columns']['item_column']
    user_column = dataset_cfg['dataset']['format']['columns']['user_column']
    output_path = dataset_cfg['dataset']['user_group_file_path']
    file_path = dataset_cfg['dataset']['ratings_file_path']
    
    if dataset == 'ml-1m': 
        # define the format: UserID :: MovieID :: Rating :: Timestamp
        df = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    if dataset == 'goodreads':
        df = pd.read_csv(file_path, sep=',', engine='python')
        
    df = df[df.rating >=4] #Isolate items they actually liked 
    
    # popularity bins for items
    item_grouped = df.groupby(item_column).count().reset_index().rename(columns={user_column: 'count_of_users'})
    item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(lambda x: math.log(x, 5))
    item_grouped['binned'] = pd.cut(item_grouped['log_smoothed'], 5, retbins=False, labels=list(range(5)))
    df_w_bins = df.merge(item_grouped, on=item_column)
    
    # collect user taste profiles 
    user_grouped = df_w_bins.groupby(user_column)['binned'].apply(lambda x: list(x)).reset_index().rename(columns={'binned':'item_grps'})
    n_item_bins = len(item_grouped['binned'].unique().tolist()) 
    user_feats = user_bin_distribution(user_grouped, n_bins=n_item_bins, user_column=user_column)
    p_cols = [f"p_{i}" for i in range(n_item_bins)]
    P = user_feats[p_cols].to_numpy()

    # diversity metrics
    user_feats["entropy"] = [entropy_from_probs(p) for p in P]   
    user_feats["entropy_bin"] = pd.cut(
    user_feats["entropy"], bins = 3,
    labels=[0, 1, 2]
    )
    # ipdb.set_trace() 
    user_feats = user_feats.rename(columns={user_column: user_key})
    print(user_feats.groupby('entropy_bin').count())
    
    pickle.dump(user_feats, open(output_path, 'wb'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        default="ml-1m",
        choices=["ml-1m", "goodreads"],
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=5,
    )
        
    args = parser.parse_args()
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    
    generate_item_groups(args, dataset_cfg) 
    generate_user_groups(args, dataset_cfg)

