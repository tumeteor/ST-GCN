from torch.utils.data import Dataset
from functools import lru_cache
import torch


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, adj_list, mask_list, time_steps):
        self.tensors = tensors
        self.adj_list = adj_list
        self.mask_list = mask_list
        self.time_steps = time_steps

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        graph_idx = int(index / self.time_steps)

        return tuple((torch.from_numpy(self.tensors[index][0]), torch.from_numpy(self.tensors[index][1]),
                      self.adj_list[graph_idx], self.mask_list[index]))

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    def _load_tensor_from_path(self):
        # TODO
        pass
