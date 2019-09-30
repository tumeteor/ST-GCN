from torch.utils.data import Dataset
from functools import lru_cache
import torch
import h5py


class CustomTensorDataset(Dataset):
    def __init__(self, datasets, adj_list, mode, cluster_idx_ids, time_steps):

        self.datasets = datasets
        self.adj_list = adj_list
        self.cluster_idx_ids = cluster_idx_ids
        self.mode = mode
        self.time_steps = time_steps

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        graph_idx = int(index / self.time_steps)
        batch_idx = index % self.time_steps
        cluster_id = self.cluster_idx_ids[graph_idx]
        data = self._load_tensor_from_path(cluster_id=cluster_id,
                                           idx=batch_idx,
                                           mode=self.mode)

        return tuple(torch.from_numpy(data["data"]), torch.from_numpy(data["target"]),
                      self.adj_list[graph_idx], torch.from_numpy(data["mask"]))

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    @lru_cache(maxsize=512)
    def _load_tensor_from_path(self, cluster_id, idx, mode):
        h = h5py.File(f"data/cache/cluster_id={cluster_id}.hdf5", "r")
        return h[mode][idx]


