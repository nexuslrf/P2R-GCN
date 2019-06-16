import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.datasets.ccf_ai import ccf_ai
from rec.utils import cuda
from dgl import DGLGraph
import matplotlib.pyplot as plt
import argparse
import pickle
import os

cache_file = 'ccf-collect.pkl'
with open(cache_file, 'rb') as f:
    db = pickle.load(f)

g = db.g
h = torch.load('hidden_embeddings.pt')
offset = 0
period = 1
user_offset = len(db.papers)

def id_remap(pids, offset, period):
    hids = pids[np.where((pids >= offset) & ((pids - offset) % period == 0))]
    return (hids - offset) // period


def get_recommend(user)
    uid = user_offset + user
    u_nid = uid
    uhid = (u_nid - offset)//period

    pids_exclude = db.links[
        (db.links['idx_A'] == uid) &
        (db.links['train'] | db.links['test' if validation else 'valid'])
        ]['idx_P'].values
    pids_candidate = db.links[
        (db.links['idx_A'] == uid) &
        db.links['valid' if validation else 'test']
        ]['idx_P'].values

    pids = np.setdiff1d(range(len(db.paper_ids_map)), pids_exclude)

    hids = id_remap(pids, offset, period)
    hids_candidate = id_remap(pids_candidate, offset, period)

    dst = torch.from_numpy(hids)
    src = torch.zeros_like(dst).fill_(uhid)
    h_dst = h[dst]
    h_src = h[src]

    score = (h_src * h_dst).sum(1)
    score_sort_idx = score.sort(descending=True)[1].cpu().numpy()
    return score_sort_idx

def main():
    while True:
        auid = input("Please input the author id:")
        auid = eval(auid)
        print("Your publications are:")
        print(db.papers.iloc[db.author_write[user]][['title','venue','year']])
        score_sort_idx = get_recommend(auid)
        print("Here is your recommendation:")
        print(db.papers.iloc[hids[score_sort_idx[:20]]][['title','venue','year']])

if __name__ == '__main__':
    main()

