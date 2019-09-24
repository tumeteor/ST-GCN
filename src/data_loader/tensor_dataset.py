from torch.utils.data import TensorDataset
from functools import lru_cache


class CustomTensorDataset(TensorDataset):
    def __init__(self, *tensors, adj_tensor):
        super().__init__(*tensors)

        self.adj_tensor = adj_tensor

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        return tuple((tensor[index], self.adj_tensor) for tensor in self.tensors)

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.tensors[0].size(0)
