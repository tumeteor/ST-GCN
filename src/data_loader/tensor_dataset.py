from torch.utils.data import TensorDataset
from functools import lru_cache
import torch
import h5py
import yaml
import numpy as np

with open("src/configs/configs.yaml") as ymlfile:
    cfg = yaml.load(ymlfile)['DataConfig']


class GraphTensorDataset(TensorDataset):
    def __init__(self, datasets, adj_list, mode, cluster_idx_ids, time_steps):
        self.datasets = datasets
        self.adj_list = adj_list
        self.cluster_idx_ids = cluster_idx_ids
        self.mode = mode
        self.time_steps = time_steps
        self.prev_gidx = 0
        self.h = h5py.File(f"data/cache/cluster_id={self.cluster_idx_ids[0]}.hdf5", "r")

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        graph_idx = int(index / self.time_steps)
        batch_idx = index % self.time_steps
        cluster_id = self.cluster_idx_ids[graph_idx]
        if graph_idx != self.prev_gidx:
            self.h.close()
            self.h = self._load_tensor_from_path(cluster_id=cluster_id)
            self.prev_gidx = graph_idx
        data = np.array(self.h["data"][self.mode][batch_idx])
        target = np.array(self.h["target"][self.mode][batch_idx])
        mask = np.array(self.h["mask"][self.mode][batch_idx])
        return torch.from_numpy(data), torch.from_numpy(target), \
               self.adj_list[graph_idx], torch.from_numpy(mask)

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    @staticmethod
    def _load_tensor_from_path(cluster_id):
        h = h5py.File(np.os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5", "r"))
        return h



