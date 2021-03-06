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

# ## Get Jurbey Sub-Graph

# +
from src.utils.graph_utils import partition_graph_by_lonlat
from jurbey.jurbey import JURBEY

with open("../data/1556798416403.jurbey", 'rb') as tempf:
    g = JURBEY.load(tempf.read())
g_partition = partition_graph_by_lonlat(g)
# -

# ## Build a dataframe with all time and static features

import pandas

df = pandas.read_csv("../data/timeseries_speed_april_first_week.csv")
df = df.drop(columns=["Unnamed: 0"])

df.head()

# ### Let's add more columns for static features

import math
def get_static_features(row):
    arc = g_partition[row['from_node']][row['to_node']]
    return (
        arc['data'].metadata['highway'],
        arc['data'].metadata.get('surface', None),
        arc['data'].roadClass.name,
        arc['data'].metadata.get('maxspeed', math.nan),
        arc['data'].metadata.get('lanes', '1')
    )


df["highway"], df["surface"], df["roadClass"], df["maxspeed"], df["lines"] = zip(*df.apply(get_static_features, axis=1))

df.head()

df_dummies = pandas.get_dummies(df, columns=["surface", "maxspeed"], dummy_na=True)
df_dummies = pandas.get_dummies(df_dummies, columns=["highway", "roadClass", "lines"], dummy_na=False)

df_dummies.head()

df_unique = df_dummies.drop_duplicates()

df_unique.head()

df_preprocessed = df_unique

# ## Let's now make an adjecancy matrix, that matches the order in our dataframe

# +
import networkx as nx
import scipy.sparse as sp
import numpy as np


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
                                    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# +
L = nx.line_graph(nx.DiGraph(g_partition))

nodelist = [tuple(x) for x in df_preprocessed[['from_node','to_node']].values]

# +
adj = nx.to_scipy_sparse_matrix(L, format="coo", nodelist=nodelist)
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

adj = normalize(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj)
# -

# ### Now let's build time-series dataset

static_features = ['highway_access_ramp',
 'highway_corridor',
 'highway_living_street',
 'highway_platform',
 'highway_primary',
 'highway_residential',
 'highway_secondary',
 'highway_secondary_link',
 'highway_service',
 'highway_tertiary',
 'highway_tertiary_link',
 'highway_unclassified',
 'surface_asphalt',
 'surface_cobblestone',
 'surface_cobblestone:flattened',
 'surface_concrete',
 'surface_concrete:plates',
 'surface_grass_paver',
 'surface_paved',
 'surface_paving_stones',
 'surface_sett',
 'surface_nan',
 'roadClass_DirtRoad',
 'roadClass_LocalRoad',
 'roadClass_MajorRoad',
 'maxspeed_10',
 'maxspeed_20',
 'maxspeed_30',
 'maxspeed_5',
 'maxspeed_50',
 'maxspeed_nan',
 'lines_1',
 'lines_2',
 'lines_3',
 'lines_4',
 'lines_5'
]

len(static_features)

SPEED_COLUMNS = list(map(str, range(TOTAL_T_STEPS)))

mask_df = df_preprocessed.notna()
static_df = df_preprocessed[static_features]

# +
# Speed preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import pandas as pd

speed_df = df_preprocessed[SPEED_COLUMNS]
speed_df = speed_df.fillna(speed_df.mean())
speed_pipeline = Pipeline([
    #('log', FunctionTransformer(func=np.expm1, inverse_func=np.expm1, validate=True))
    ('scaler', RobustScaler())
])
speed_pipeline.fit(speed_df.values.flatten().reshape(-1, 1))
speed_df = pd.DataFrame(speed_pipeline.transform(speed_df.values), index=speed_df.index, columns=speed_df.columns)
# -

mask_df.head()


# +
import torch
WINDOW = 10

def build_sliding_speed_dataset(speed_df, mask_df, window=WINDOW):
    speed = []
    mask = []
    for i in range(window, TOTAL_T_STEPS + 1):
        columns = list(map(str, range(i - window, i)))
        speed.append(torch.Tensor(speed_df[columns].values))
        mask.append(torch.Tensor(mask_df[columns].values))
        
    return torch.stack(speed), torch.stack(mask)


# -

speed, mask = build_sliding_speed_dataset(speed_df, mask_df)
speed_seq = speed.unsqueeze(3)
mask_seq = mask.unsqueeze(3)
print(mask_seq.shape)
print(speed_seq.shape)

static = torch.Tensor(static_df.values) - 0.5
static_seq = static.unsqueeze(0)
static_seq = static_seq.unsqueeze(2)
static_seq = static_seq.expand([speed_seq.shape[0], -1, speed_seq.shape[2], -1])
print(static_seq.shape)

ts_dataset = torch.utils.data.TensorDataset(speed_seq, mask_seq, static_seq)

adj_dense = adj.to_dense()

# +
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
from src.models.gcn_lstm.gcn_lstm_model import GCNLSTMModel

model = GCNLSTMModel(38, 6, 3, adj_dense, ts_dataset, speed_transform=speed_pipeline, timesteps=WINDOW - 1, batch_size=32)
exp = Experiment(save_dir='gcnlstm_logs')
checkpoint_callback = ModelCheckpoint(
    filepath='gcnlstm.ckpt',
    save_best_only=True,
    verbose=True,
    monitor='avg_val_mae',
    mode='min'
)

# most basic trainer, uses good defaults
trainer = Trainer(experiment=exp, checkpoint_callback=checkpoint_callback)  
#trainer = Trainer(experiment=exp)    
trainer.fit(model)
#TODO lr decay
# -


