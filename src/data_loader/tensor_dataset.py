from torch.utils.data import Dataset
from functools import lru_cache
import torch
import h5py
import numpy as np


class CustomTensorDataset(Dataset):
    def __init__(self, datasets, adj_list, mode, cluster_idx_ids, time_steps):
        print(f"init custom dataset: {cluster_idx_ids}")
        self.datasets = datasets
        self.adj_list = adj_list
        self.cluster_idx_ids = cluster_idx_ids
        self.mode = mode
        self.time_steps = time_steps

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        print(f"get item: {index}")
        graph_idx = int(index / self.time_steps)
        batch_idx = index % self.time_steps
        cluster_id = self.cluster_idx_ids[graph_idx]
        h = self._load_tensor_from_path(cluster_id=cluster_id)
        data = np.array(h["data"][self.mode][batch_idx])
        print(f"data shape: {data.shape}")
        print(f"data shape: {torch.from_numpy(np.array(data)).shape}")
        target = np.array(h["target"][self.mode][batch_idx])
        mask = np.array(h["mask"][self.mode][batch_idx])
        return torch.from_numpy(data), torch.from_numpy(target), \
               self.adj_list[graph_idx], torch.from_numpy(mask)

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    @lru_cache(maxsize=512)
    def _load_tensor_from_path(self, cluster_id):
        h = h5py.File(f"data/test_cache/cluster_id={cluster_id}.hdf5", "r")
        return h


