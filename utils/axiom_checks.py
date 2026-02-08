import os
import csv
import glob
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse
import ipdb 
from tqdm import tqdm 
import pandas as pd 
import math 
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups

def JR_check_satisfaction_given_committee(proposed_committee, partial_lists, all_candidates, n, k, user_key): 
    for candidate in all_candidates:
        counter = 0 
        approving_voters = partial_lists[partial_lists['Ranked_Items']
                                         == candidate][user_key].unique().tolist()
        for voter in approving_voters: 
            approval_set = partial_lists[partial_lists[user_key] ==  voter]['Ranked_Items'].unique().tolist()
            if len(np.intersect1d(approval_set, proposed_committee)) == 0: counter += 1 
        if counter >= n/k: 
            return False             
    return True

def PJR_check_satisfaction_given_committee(proposed_committee, partial_lists, l_cohesive, user_key):
    n, k = len(partial_lists[user_key].unique()), len(proposed_committee)
    for l in range(1, k+1): #iterate through l, increasing from 1
            voter_sets = l_cohesive[l]['voter_sets']
            candidate_sets = l_cohesive[l]['candidate_sets']
            for i in range(len(candidate_sets)): #go through all bicliques found in the graph
                voter_group, candidate_group = voter_sets[i], candidate_sets[i] #cohesive groups of voters agreeing on specific group of cancidates
                if len(voter_group) >= l*(n/k) and len(candidate_group) >= l: #we need voter group of size ln/k agreeing on l-sized group of candidates
                    # ipdb.set_trace() 
                    approval_set = partial_lists[partial_lists[user_key].isin(voter_group)]['Ranked_Items'].unique(
                    ).tolist() #find the union of candidates that they all agree on 
                    if len(np.intersect1d(approval_set, proposed_committee)) < l: #if less than l of them in the committee W
                        return False       
    return True

def prune_satisfied_for_EJR(partial_lists, proposed_committee, l, user_key):
    for voter in partial_lists[user_key].unique().tolist(): #look through each voter
        approval_set = partial_lists[partial_lists[user_key] == voter]['Ranked_Items'].unique(
        ).tolist() #find the candidates he approves of
        if len(np.intersect1d(approval_set, proposed_committee)) >= l: #if the voter is l-satisfied
            partial_lists = partial_lists[partial_lists[user_key] != voter] #prune that voter 
    return partial_lists.reset_index(drop=True)

def EJR_check_satisfaction_given_committee(proposed_committee, partial_lists, user_key, data_cfg): 
    n, k = len(partial_lists[user_key].unique()), len(proposed_committee)
    # for l in tqdm(range(1, k+1)): #iterate through l, increasing from 1
    for l in range(1, k+1):
        unsatisfied_voter_set = prune_satisfied_for_EJR(partial_lists, proposed_committee, l, user_key)
        # if len(proposed_committee) == 5: 
            # ipdb.set_trace() 
        voter_sets, candidate_sets = find_maximal_cohesive_groups(unsatisfied_voter_set, committee_size=k, data_cfg=data_cfg)
        # voter_sets, cand_sets = find_maximal_cohesive_groups_groupby(partial_lists)
        for idx, v in enumerate(voter_sets): 
            if len(v) >= (l*n)/k and len(candidate_sets[idx]) >= l: 
                return False
    return True
