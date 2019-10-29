from torch.utils.data import Dataset
from functools import lru_cache
import torch
import h5py
import yaml
import os
import numpy as np

with open("configs/configs.yaml") as ymlfile:
    cfg = yaml.safe_load(ymlfile)['DataConfig']


class GraphTensorDataset(Dataset):
    CHUNK_SIZE = 100

    def __init__(self, datasets, adj_list, mode, cluster_idx_ids, time_steps):
        self.datasets = datasets
        self.adj_list = adj_list
        self.cluster_idx_ids = cluster_idx_ids
        self.mode = mode
        self.time_steps = time_steps
        self.prev_gidx = 0
        self.h = h5py.File(os.path.join(cfg['save_dir_data'], f"cluster_id={self.cluster_idx_ids[0]}.hdf5"), "r")
        self.cache = dict()

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        graph_idx = int(index / self.time_steps)
        batch_idx = index % self.time_steps
        cluster_id = self.cluster_idx_ids[graph_idx]
        adj = self.adj_list[graph_idx]
        if graph_idx != self.prev_gidx:
            self.h.close()
            self.h = self._load_tensor_from_path(cluster_id=cluster_id)
            # empirical settings
            if 10000 > adj.shape[0] > 6000:
                # reduce the chunk size for too big graph
                self.CHUNK_SIZE = 40
            if adj.shape[0] >= 10000:
                self.CHUNK_SIZE = 15
            self.prev_gidx = graph_idx

        data = self._load_item(idx=batch_idx, _type="data")
        target = self._load_item(idx=batch_idx, _type="target")
        mask = self._load_item(idx=batch_idx, _type="mask")
        return data, target, adj, mask

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    def _load_item(self, idx, _type):
        if idx % self.CHUNK_SIZE == 0:
            cidx = (idx // self.CHUNK_SIZE) * self.CHUNK_SIZE
            self.cache[_type] = self.h[_type][self.mode][cidx:cidx + self.CHUNK_SIZE]
        return np.array(self.cache[_type][idx % self.CHUNK_SIZE])

    @staticmethod
    def _load_tensor_from_path(cluster_id):
        h = h5py.File(os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5"), "r")
        return h
