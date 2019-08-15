# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.append('../')

TOTAL_T_STEPS = 144

# # Get adjacency matrix for our partitions of Jurbey map

# +
from src.graph_utils import partition_graph_by_lonlat
import networkx as nx
from jurbey.jurbey import JURBEY

with open("../data/1556798416403.jurbey", 'rb') as tempf:
    g = JURBEY.load(tempf.read())
    
g_partition = partition_graph_by_lonlat(g)

A = nx.adjacency_matrix(g_partition)
# -

# # Convert timeseries raw data into sparse tensor
# Tensor size: 144 timestamps by adjacency matrix of speeds

import pandas

df = pandas.read_csv("../data/timeseries_speed_april_first_week.csv")

df.head()

# +
#Our speed data uses segment ids, but the model uses sequential indexes, based on `.nodes()`
import math
id_to_idx = {}
# defaultdict won't do what you expect in Pandas

for id_ in df["from_node"].unique():
    id_to_idx[id_] = math.nan
for id_ in df["to_node"].unique():
    id_to_idx[id_] = math.nan
    
for idx, id_ in enumerate(g_partition.nodes()):
    id_to_idx[id_] = idx
# -

# Let's transform ids to indeces
df["from_node_idx"] = df.replace({"from_node": id_to_idx})["from_node"]
df["to_node_idx"] = df.replace({"to_node": id_to_idx})["to_node"]

df.head()

# ## First let's build sparse 3D data tensor

# +
import torch

def snapshot(t, df=df, g_partition=g_partition):
    df_t = df[[t, "from_node_idx", "to_node_idx"]]
    df_t = df_t.dropna()
    row = df_t["from_node_idx"].tolist()
    col = df_t["to_node_idx"].tolist()
    data = df_t[t].tolist()
    size = len(g_partition.nodes())  

    return {"indices": (row, col), "values": data, "shape": (size, size)}


# +
from scipy.sparse import hstack

def build_sparse_dataset(from_=0, to=TOTAL_T_STEPS):
    dataset = {"indices": ([], [], []), "values": []}
    for t in range(from_, to):

        snap = snapshot(str(t))
        dataset["indices"][0].extend([t] * len(snap["indices"][0]))
        dataset["indices"][1].extend(snap["indices"][0])
        dataset["indices"][2].extend(snap["indices"][1])
        dataset["values"].extend(snap["values"])

    i = torch.LongTensor(dataset["indices"])
    v = torch.FloatTensor(dataset["values"])
    return torch.sparse.FloatTensor(i, v, torch.Size((to, *snap["shape"])))

dataset = build_sparse_dataset()
# -

# ## Now let's split sparse TxKxK Tensor into 3 TxKxK tensors for training, validation and testing

nonzero_values_cnt = len(dataset._values())
# what percent goes into training/validation/testing
tng_pct = 0.7
val_pct = 0.1
tst_pct = 1 - tng_pct - val_pct
# now we want to split list of all non-zeros promortionally:
# [0, split1_idx], [split1_idx, split2_idx] and [split2_idx:]
split1_idx = int(nonzero_values_cnt * tng_pct)
split2_idx = -int(nonzero_values_cnt * tst_pct)
# + {}
from random import shuffle

# but we select indexes randomly
idxs = list(range(nonzero_values_cnt))
shuffle(idxs)
# these are non-zero indexes
tng_idxs = idxs[:split1_idx]
val_idxs = idxs[split1_idx:split2_idx]
tst_inxs = idxs[split2_idx:]
# -
dataset_split = {}
for name, idxs in [('tng', tng_idxs), ('val', val_idxs), ('tst', tst_inxs)]:
    i = torch.LongTensor([
        dataset._indices()[0][idxs].tolist(),
        dataset._indices()[1][idxs].tolist(),
        dataset._indices()[2][idxs].tolist()
    ])
    v = torch.FloatTensor(dataset._values()[idxs])
    
    dataset_split[name] = torch.sparse.FloatTensor(i, v, dataset.shape)

dataset_split

# # Let's train model

from src.nmf.lsm_rn import LSM_RN

# +
from pytorch_lightning import Trainer

model = LSM_RN(TOTAL_T_STEPS, n=3475, k=50, λ=0.5, adj_mat=A, datasets=dataset_split)
# most basic trainer, uses good defaults
trainer = Trainer()    
trainer.fit(model)   
# -


