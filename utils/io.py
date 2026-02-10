import pickle 
import pandas as pd 
from collections import defaultdict
import sys

# =============================================================================
# Aggregation Utils
# =============================================================================

def load_rankings_to_df_old(dataset_cfg): 
    filepath = dataset_cfg['dataset']['rec_set_path']
    print(f"Loading data from '{filepath}'...")

    user_key = dataset_cfg['dataset']["keys"]["user_key"]
    item_key = dataset_cfg['dataset']["keys"]["item_key"]
    est_rating_key = dataset_cfg['dataset']["keys"]["est_rating_key"]
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    df[est_rating_key] = pd.to_numeric(df[est_rating_key], errors="coerce")
    df = df.dropna(subset=[est_rating_key])

    df = df.sort_values([user_key, est_rating_key], ascending=[True, False])

    rankings_df = (
        df.groupby(user_key)[item_key]
          .apply(list)
          .reset_index(name="Ranked_Items")
    )

    all_items = set(df[item_key].unique())

    print(f"Loaded rankings for {len(rankings_df)} users.")
    print(f"Total unique items found: {len(all_items)}")

    return rankings_df, all_items

import sys
import pandas as pd

def load_rankings_to_df(dataset_cfg):
    filepath = dataset_cfg['dataset']['rec_set_path']
    print(f"Loading data from '{filepath}'...")

    user_key = dataset_cfg['dataset']["keys"]["user_key"]
    item_key = dataset_cfg['dataset']["keys"]["item_key"]
    est_rating_key = dataset_cfg['dataset']["keys"]["est_rating_key"]

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    # numeric scores only
    df[est_rating_key] = pd.to_numeric(df[est_rating_key], errors="coerce")
    df = df.dropna(subset=[est_rating_key])

    # # (optional but recommended) normalize item ids to str everywhere
    # df[item_key] = df[item_key].astype(str)
    # df[user_key] = df[user_key].astype(str)

    # stable sort so order is deterministic even with ties
    df = df.sort_values([user_key, est_rating_key], ascending=[True, False], kind="mergesort")

    rankings_df = (
        df.groupby(user_key, sort=False)
          .apply(lambda g: pd.Series({
              "Ranked_Items": g[item_key].tolist(),
              "Ranked_Scores": g[est_rating_key].astype(float).tolist(),
          }))
          .reset_index()
    )

    all_items = set(df[item_key].unique())

    print(f"Loaded rankings for {len(rankings_df)} users.")
    print(f"Total unique items found: {len(all_items)}")

    return rankings_df, all_items


# =============================================================================
# Axiom Utils
# =============================================================================

def load_consensus_ranking(file_path):
    """
    Loads a consensus ranking file (rank item score).
    Returns a Dictionary: { 'ItemID': Rank_Integer }
    """
    rank_map = {}
    item_id_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Format: Rank ItemID Score
                    # We only care about the ItemID and its Rank (order)
                    rank = int(parts[0])
                    item_id = parts[1]
                    item_id_list.append(int(item_id)) 
                    rank_map[item_id] = rank
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
        
    return item_id_list

def load_sampled_preferences(file_path):
    """
    Loads sampled user preferences from a CSV file.
    Expects columns: User_ID, Movie_ID, Estimated_Rating
    """
    # ipdb.set_trace() 
    preferences = pickle.load(open(file_path, 'rb')) #.explode('Ranked_Items').reset_index(drop=True)
    return preferences


# =============================================================================
# Kendall Tau Utils
# =============================================================================

def format_sampled_rankings_kt(df, data_cfg):
    """
    Loads user lists from recommendations.csv.
    Returns: { 'User_ID': [item1, item2, item3...] }
    Items are sorted by the estimated rating (descending).
    """
    # print(f"Loading user lists from {csv_path}...")
    user_items = {}
    print(df)
    user_key = data_cfg['dataset']['keys']['user_key']
    item_key = data_cfg['dataset']['keys']['item_key']
    est_rating_key = data_cfg['dataset']['keys']['est_rating_key']
    for uid in df[user_key].unique():
        items = df[df[user_key] == uid]
        user_items[str(uid)] = []
        for _, row in items.iterrows():
            user_items[str(uid)].extend(row['Ranked_Items'])
    # print(user_items)
    
    print(f"Loaded {len(user_items)} user lists.")
    
    return user_items


def load_consensus_ranking_kt(file_path):
    """
    Loads a consensus ranking file (rank item score).
    Returns a Dictionary: { 'ItemID': Rank_Integer }
    """
    rank_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Format: Rank ItemID Score
                    # We only care about the ItemID and its Rank (order)
                    rank = int(parts[0])
                    item_id = parts[1]
                    rank_map[item_id] = rank
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
        
    return rank_map