# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
import sys
import pandas
sys.path.append('../')

from src.graph_utils import partition_graph_by_lonlat
import networkx as nx
from jurbey.jurbey import JURBEY

with open("../data/berlin.jurbey", 'rb') as tempf:
    g = JURBEY.load(tempf.read())
print(g.number_of_nodes())
g_partition = partition_graph_by_lonlat(g)
# -

# **Convert to edge-based graph**

import networkx as nx
L = nx.line_graph(nx.DiGraph(g_partition))

nodes = list(L.nodes())
g_partition[nodes[10][0]][nodes[10][1]]['data']

# **Extract dynamic (speed) + static features from nodes**

# +
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
enc = OneHotEncoder(handle_unknown='ignore')
ienc = OrdinalEncoder()
scaler = StandardScaler()
def arc_features(arc):
    arc = g_partition[arc[0]][arc[1]]
    return [
        arc['data'].metadata['highway'],
        arc['data'].metadata.get('surface', 'no_sur'),
        arc['data'].roadClass.name
    ],  [float(arc['data'].metadata.get('maxspeed', '50')), 
        int(arc['data'].metadata.get('lanes', '1'))]

def construct_features():
    data = list()
    data_ord = list()
    for node in L.nodes:
        data.append(arc_features(node)[0])
        data_ord.append(arc_features(node)[1])
    return enc.fit_transform(data), ienc.fit_transform(data_ord)
    
x, y = construct_features()
  
# -

enc.categories_

ienc.categories_

x.shape

x

# **Preprocess adjacency matrix**

# +
adj = nx.to_scipy_sparse_matrix(L, format="coo")
import scipy.sparse as sp
import numpy as np
import torch

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
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

adj = normalize(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj)
# -

adj.shape

# +
#Our speed data uses segment ids, but the model uses sequential indexes, based on `.nodes()`
import math
id_to_idx = {}
# defaultdict won't do what you expect in Pandas
df = pandas.read_csv("../data/timeseries_speed_april_first_week.csv")
df = df.T
l = (df.isnull().mean() < 0.5).tolist()

indices = [i for i, x in enumerate(l) if x == True]
print(indices)

# +
id_to_idx = {}

for idx, id_ in enumerate(L.nodes()):
    id_to_idx[id_] = idx
df = df.T
df = df.loc[:, df.columns != 'Unnamed: 0']

df2 = df['from_node']
df3 = df['to_node']

df_filled = df.loc[:, df.columns != 'from_node']
df_filled = df.loc[:, df.columns != 'to_node']


df_filled = df_filled.T
for column in df_filled:
    df_filled[column] = pandas.to_numeric(df_filled[column])

df_filled = df_filled.interpolate(method='nearest', axis=1)

df_filled = df_filled.fillna(method='backfill')
df_filled = df_filled.T
df_filled['from_node'] = df2
df_filled['to_node'] = df3

print(df_filled[0:10])

# -

df[0:10]

# **Create rolling window tensor dataset**

# +
import torch
import scipy.sparse
TOTAL_T_STEPS = 144

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
ienc = OrdinalEncoder()
    
def build_dataset_to_numpy_tensor(from_=0, to=TOTAL_T_STEPS, df=None):
    """
    We extract features from speed (actual speed, whether speed is missing)
    and combine with static features.
    :return:
         np.ndarray: dataset tensor of shape [num_time_steps, num_nodes, num_features]
    """
    dataset = list()
    for t in range(from_, to):
        cat_features_at_t = [['primary', 'asphalt', 'MajorRoad']] * len(L.nodes)
        ord_features_at_t = [[50.0, 4]] * len(L.nodes)
        speed_features_at_t = [50] * len(L.nodes) 
        speed_is_nan_feature = [1] * len(L.nodes)
        for _, row in df.iterrows():

            arc = (row['from_node'], row['to_node'])
            cat_features_at_t[id_to_idx[arc]], ord_features_at_t[id_to_idx[arc]]  = arc_features(arc)
            speed_features_at_t[id_to_idx[arc]] = row[str(t)]
            if np.isnan(row[str(t)]): 
                speed_is_nan_feature[id_to_idx[arc]] = 0
        dataset.append(np.concatenate([scaler.fit_transform(np.array(speed_features_at_t).reshape(-1, 1)), 
                                       np.array(speed_is_nan_feature).reshape(-1, 1), 
                                       ienc.fit_transform(ord_features_at_t),
                                       enc.fit_transform(cat_features_at_t).toarray()], axis=1))
    return np.stack(dataset, axis=0)

data = build_dataset_to_numpy_tensor(df=df)
# -

# Build mask tensor
data_speed_only = data[:,:,0]
data_masked = torch.where(torch.isnan(torch.from_numpy(data_speed_only)), torch.tensor([0]), torch.tensor([1]))
data_masked = data_masked.bool()

data.shape

data_masked.shape

split_line1 = int(data.shape[0] * 0.7)
split_line2 = int(data.shape[0] * 0.9)

# +
trg_data = data[:split_line1,:, :]
val_data = data[split_line1:split_line2,:, :]
tst_data = data[split_line2:,:,:]

trg_mask = data[:split_line1,:, :]
val_mask = data[split_line1:split_line2,:, :]
tst_mask = data[split_line2:,:,:]
# -

from src.utils.dataset import SlidingWindowDataset
from torch.utils.data import DataLoader

# +
import numpy as np
import torch
import torch.utils.data


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors, window=1, horizon=1, dtype=torch.float):
        super().__init__()
        print(locals())
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)

        self._tensors = tensors
        self._window = window
        self._horizon = horizon
        self._dtype = dtype

    def __getitem__(self, index):
        item = []
        for t in self._tensors:
            x = t[index : index + self._window]
            y = t[index + self._window : index + self._window + self._horizon]
            item.append(
                {
                    "x": torch.from_numpy(x).type(self._dtype),
                    "y": torch.from_numpy(y).type(self._dtype),
                }
            )
        return item

    def __len__(self):
        return self._tensors[0].shape[0] - self._window - self._horizon + 1



# -

dataset = SlidingWindowDataset(trg_data, trg_mask, window=10, horizon=1)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

next(iter(dataloader))


