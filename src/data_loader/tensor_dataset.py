from torch.utils.data import Dataset
from functools import lru_cache
import torch
import h5py
import yaml
import numpy as np

with open("src/configs/configs.yaml") as ymlfile:
    cfg = yaml.load(ymlfile)['DataConfig']

class CustomTensorDataset(Dataset):
    def __init__(self, datasets, adj_list, mode, cluster_idx_ids, time_steps):
        self.datasets = datasets
        self.adj_list = adj_list
        self.cluster_idx_ids = cluster_idx_ids
        self.mode = mode
        self.time_steps = time_steps

    @lru_cache(maxsize=128)
    def __getitem__(self, index):
        graph_idx = int(index / self.time_steps)
        batch_idx = index % self.time_steps
        cluster_id = self.cluster_idx_ids[graph_idx]
        h = self._load_tensor_from_path(cluster_id=cluster_id)
        data = np.array(h["data"][self.mode][batch_idx])
        target = np.array(h["target"][self.mode][batch_idx])
        mask = np.array(h["mask"][self.mode][batch_idx])
        return torch.from_numpy(data), torch.from_numpy(target), \
               self.adj_list[graph_idx], torch.from_numpy(mask)

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    @lru_cache(maxsize=2048)
    def _load_tensor_from_path(self, cluster_id):
        h = h5py.File(np.os.path.join(cfg['save_dir_data'], f"cluster_id={cluster_id}.hdf5", "r"))
        return h

